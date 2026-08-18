from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from array import array
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from rdkit import Chem, rdBase

from stage1.config import PretrainConfig
from stage1.features import (
    ROLE_TO_ID,
    build_entity_sample,
    inspect_entity_qc,
    load_stage1_feature_inputs,
)
from stage1.descriptors import (
    DescriptorSchema,
    DescriptorStandardizer,
    calculate_descriptors,
    rdkit_descriptor_names,
)
from stage1.masking import MultimodalPacker
from stage1.model import LoadedStage1Model, load_stage1_model
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.progress import ProgressReporter
from common.training import canonical_json_sha256, resolve_device
from .atom_targets import (
    load_structure_manifest,
    map_partial_charges,
    parse_mol2,
    verify_structure,
)
from .config import Stage2Config
from .data import (
    STAGE2_ARTIFACT_KIND,
    STAGE2_ARTIFACT_VERSION,
    Stage2EntityDataset,
    Stage2TaskDataset,
)
from .model import RECONSTRUCTION_MODULES, build_model_contract
from .registry import Stage2Registry, TaskSpec, load_stage2_registry
from .runtime import configure_stage2_math
from stage1.tokenizer import SmilesTokenizer


MISSING_MARKERS = frozenset({"", "nan", "na", "n/a", "null"})


def _canonicalize(smiles: str, context: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES in {context}: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def _finite_float(raw: str, context: str) -> float:
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"Non-numeric value in {context}: {raw}") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value in {context}: {raw}")
    return value


def _target_value(raw: str, context: str, *, allow_missing: bool) -> float | None:
    stripped = (raw or "").strip()
    if allow_missing and stripped.lower() in MISSING_MARKERS:
        return None
    return _finite_float(stripped, context)


def _key_hash(parts: Sequence[str]) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.digest()


@dataclass
class StagedTaskRows:
    entity_ids: list[list[int]]
    conditions: list[list[float]]
    targets: list[list[float]]
    target_mask: list[list[bool]]
    source_rows: list[int]
    mol_ids: list[str]
    atom_targets: list[list[float]]

    def __len__(self) -> int:
        return len(self.source_rows)


@dataclass(frozen=True)
class CollectedStage2Data:
    entity_keys: tuple[tuple[str, str], ...]
    rows: dict[str, dict[str, StagedTaskRows]]
    source_counts: dict[str, dict[str, dict[str, int]]]
    duplicate_rows: tuple[dict[str, Any], ...]
    missing_target_rows: tuple[dict[str, Any], ...]
    mapping_audit: tuple[dict[str, Any], ...]


def _role_for(canonical: str, policy: str, row: dict[str, str], context: str) -> str:
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES in {context}")
    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    inferred = "cation" if charge > 0 else ("anion" if charge < 0 else "neutral")
    if policy == "formal_charge":
        return inferred
    if policy == "manifest":
        role = row.get("role", "").strip()
        try:
            declared_charge = int(row.get("formal_charge", ""))
        except ValueError as error:
            raise ValueError(f"Invalid partial-charge formal_charge in {context}") from error
        if role != inferred or declared_charge != charge:
            raise ValueError(f"Partial-charge role/formal_charge mismatch in {context}")
        return role
    if policy not in ROLE_TO_ID or policy != inferred:
        raise ValueError(f"Entity role mismatch in {context}: expected {policy}, got {inferred}")
    return policy


def _iter_rows(path: Path, expected: tuple[str, ...]) -> Iterator[tuple[int, dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stage 2 source: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"Unexpected Stage 2 columns in {path}: {reader.fieldnames}; expected {list(expected)}")
        for row_number, row in enumerate(reader, start=2):
            yield row_number, row


def _collect_sources(config: Stage2Config, registry: Stage2Registry, reporter: ProgressReporter) -> CollectedStage2Data:
    canonical_cache: dict[str, str] = {}
    entity_key_ids: dict[tuple[str, str], int] = {}
    staged: dict[str, dict[str, StagedTaskRows]] = {}
    source_counts: dict[str, dict[str, dict[str, int]]] = {}
    duplicate_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    mapping_audit: list[dict[str, Any]] = []
    with reporter.bar(total=len(registry.tasks) * 2, desc="Stage 2 catalog scan", unit="file") as progress:
        for spec in registry.tasks:
            staged[spec.task_id] = {}
            source_counts[spec.task_id] = {}
            manifest = None
            if spec.target_level == "atom":
                manifest_path = spec.dataset.resource_manifest_path(config.data.data_root)
                if manifest_path is None:
                    raise ValueError("Atom task is missing a structure manifest")
                manifest = load_structure_manifest(manifest_path)
            train_keys: set[bytes] = set()
            for split in ("train", "valid"):
                values = StagedTaskRows([], [], [], [], [], [], [])
                seen: set[bytes] = set()
                detailed: dict[tuple[str, ...], tuple[int, tuple[str, ...]]] = {}
                counts: dict[str, int] = {}
                expected = (
                    ("mol_id", *spec.entity_columns, "role", "formal_charge", "source_list")
                    if spec.target_level == "atom"
                    else (*spec.entity_columns, *spec.condition_columns, *spec.target_columns, "source_list")
                )
                for row_number, row in _iter_rows(spec.dataset.split_path(config.data.data_root, split), expected):
                    canonicals: list[str] = []
                    roles: list[str] = []
                    for column, policy in zip(spec.entity_columns, spec.role_policy, strict=True):
                        raw = (row[column] or "").strip()
                        if not raw:
                            raise ValueError(f"Empty {column} in {spec.task_id}/{split}:{row_number}")
                        canonical = canonical_cache.get(raw)
                        if canonical is None:
                            canonical = _canonicalize(raw, f"{spec.task_id}/{split}:{row_number}/{column}")
                            canonical_cache[raw] = canonical
                        canonicals.append(canonical)
                        roles.append(_role_for(canonical, policy, row, f"{spec.task_id}/{split}:{row_number}"))
                    conditions = [_finite_float(row[name], f"{spec.task_id}/{split}:{row_number}/{name}") for name in spec.condition_columns]
                    allow_missing = config.loss.task_loss_modes.get(spec.task_id, "element_mean") == "masked_target_macro"
                    targets = [] if spec.target_level == "atom" else [
                        _target_value(row[name], f"{spec.task_id}/{split}:{row_number}/{name}", allow_missing=allow_missing)
                        for name in spec.target_columns
                    ]
                    missing = [] if spec.target_level == "atom" else [
                        name for name, value in zip(spec.target_columns, targets, strict=True)
                        if value is None
                    ]
                    if missing:
                        all_missing = len(missing) == len(spec.target_columns)
                        missing_rows.append({"task": spec.task_id, "split": split, "source_row": row_number, "missing_columns": ";".join(missing), "valid_target_count": len(targets) - len(missing), "action": "excluded" if all_missing else "retained"})
                        if all_missing:
                            continue
                    atom_targets: list[float] = []
                    mol_id = ""
                    if spec.target_level == "atom":
                        mol_id = row["mol_id"].strip()
                        entry = None if manifest is None else manifest.get(mol_id)
                        audit = {"mol_id": mol_id, "canonical_smiles": canonicals[0], "structure_path": "" if entry is None else str(entry.path.relative_to(config.data.data_root.resolve())), "model_atom_count": Chem.MolFromSmiles(canonicals[0]).GetNumAtoms(), "structure_atom_count": 0, "mapped_atom_count": 0, "mapping_count": 0, "selected_mapping_rank": 0, "bond_match_mode": "", "unparsed_bond_types": "", "bond_fallback_reason": "", "status": "excluded", "reason": ""}
                        try:
                            if entry is None:
                                raise ValueError("missing_structure_manifest_entry")
                            verify_structure(entry)
                            graph = parse_mol2(entry.path)
                            result = map_partial_charges(canonicals[0], graph)
                            atom_targets = list(result.charges)
                            audit.update({"structure_atom_count": result.structure_atom_count, "mapped_atom_count": len(result.charges), "mapping_count": result.mapping_count, "selected_mapping_rank": result.selected_mapping_rank, "bond_match_mode": result.bond_match_mode, "unparsed_bond_types": ";".join(result.unparsed_bond_types), "bond_fallback_reason": result.bond_fallback_reason, "status": "mapped", "reason": ""})
                        except (OSError, ValueError) as error:
                            audit["reason"] = str(error)
                            mapping_audit.append(audit)
                            continue
                        mapping_audit.append(audit)
                    input_parts = (
                        *((mol_id,) if spec.target_level == "atom" else ()),
                        *canonicals,
                        *(format(value, ".17g") for value in conditions),
                    )
                    hashed = _key_hash(input_parts)
                    if split == "valid" and hashed in train_keys:
                        raise ValueError(f"Stage 2 train/valid input overlap in {spec.task_id}: " + " | ".join(input_parts))
                    target_text = tuple("missing" if value is None else format(value, ".17g") for value in targets)
                    if input_parts in detailed:
                        previous = detailed[input_parts]
                        duplicate_rows.append({"task": spec.task_id, "split": split, "input_key": " | ".join(input_parts), "first_row": previous[0], "duplicate_row": row_number, "first_targets": " | ".join(previous[1]), "duplicate_targets": " | ".join(target_text)})
                    else:
                        detailed[input_parts] = (row_number, target_text)
                    ids: list[int] = []
                    for role, canonical in zip(roles, canonicals, strict=True):
                        key = (role, canonical)
                        if key not in entity_key_ids:
                            entity_key_ids[key] = len(entity_key_ids)
                        ids.append(entity_key_ids[key])
                    values.entity_ids.append(ids)
                    values.conditions.append(conditions)
                    values.targets.append([0.0 if value is None else value for value in targets])
                    values.target_mask.append([value is not None for value in targets])
                    values.source_rows.append(row_number)
                    values.mol_ids.append(mol_id)
                    values.atom_targets.append(atom_targets)
                    source = row["source_list"].strip()
                    if not source:
                        raise ValueError(f"Empty source_list in {spec.task_id}/{split}:{row_number}")
                    counts[source] = counts.get(source, 0) + 1
                    seen.add(hashed)
                staged[spec.task_id][split] = values
                source_counts[spec.task_id][split] = dict(sorted(counts.items()))
                if split == "train":
                    train_keys = seen
                progress.update(1)
    sorted_keys = tuple(sorted(entity_key_ids, key=lambda value: (ROLE_TO_ID[value[0]], value[1])))
    sorted_ids = {key: index for index, key in enumerate(sorted_keys)}
    remap = {old: sorted_ids[key] for key, old in entity_key_ids.items()}
    for task_rows in staged.values():
        for rows in task_rows.values():
            rows.entity_ids[:] = [[remap[value] for value in row] for row in rows.entity_ids]
    return CollectedStage2Data(sorted_keys, staged, source_counts, tuple(duplicate_rows), tuple(missing_rows), tuple(mapping_audit))


def _write_duplicate_audit(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = (
        "task",
        "split",
        "input_key",
        "first_row",
        "duplicate_row",
        "first_targets",
        "duplicate_targets",
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_missing_target_audit(
    path: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = (
        "task",
        "split",
        "source_row",
        "missing_columns",
        "valid_target_count",
        "action",
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@dataclass(frozen=True)
class EntityFeatureResult:
    candidate_id: int
    role: str
    canonical_smiles: str
    sample: dict[str, Any] | None
    exclusion: dict[str, Any] | None


def _sample_to_ipc_payload(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return np.array(
            value.detach().cpu().contiguous().numpy(),
            copy=True,
            order="C",
        )
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True, order="C")
    if isinstance(value, dict):
        return {
            key: _sample_to_ipc_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_sample_to_ipc_payload(item) for item in value)
    if isinstance(value, list):
        return [_sample_to_ipc_payload(item) for item in value]
    return value


def _sample_from_ipc_payload(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, dict):
        return {
            key: _sample_from_ipc_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_sample_from_ipc_payload(item) for item in value)
    if isinstance(value, list):
        return [_sample_from_ipc_payload(item) for item in value]
    return value


def _materialize_entity_feature(
    result: EntityFeatureResult,
) -> EntityFeatureResult:
    return EntityFeatureResult(
        candidate_id=result.candidate_id,
        role=result.role,
        canonical_smiles=result.canonical_smiles,
        sample=(
            None
            if result.sample is None
            else _sample_from_ipc_payload(result.sample)
        ),
        exclusion=result.exclusion,
    )


_ENTITY_WORKER_STATE: tuple[
    PretrainConfig,
    SmilesTokenizer,
    DescriptorSchema,
    DescriptorStandardizer,
    tuple[str, ...],
] | None = None


def _initialize_entity_worker(
    pretrain_config: PretrainConfig,
    vocabulary: SmilesTokenizer,
    schema: DescriptorSchema,
    standardizer: DescriptorStandardizer,
) -> None:
    global _ENTITY_WORKER_STATE
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    _ENTITY_WORKER_STATE = (
        pretrain_config,
        vocabulary,
        schema,
        standardizer,
        tuple(rdkit_descriptor_names()),
    )


def _build_entity_feature_result(
    item: tuple[int, str, str],
) -> EntityFeatureResult:
    if _ENTITY_WORKER_STATE is None:
        raise RuntimeError("Stage 2 entity worker was not initialized")
    candidate_id, role, canonical_smiles = item
    pretrain_config, vocabulary, schema, standardizer, raw_names = (
        _ENTITY_WORKER_STATE
    )
    record = {
        "sample_id": "pending",
        "role": role,
        "role_id": ROLE_TO_ID[role],
        "canonical_smiles": canonical_smiles,
        "sources": ("stage2",),
        "split": "stage2",
        "is_augmented": False,
        "seed_smiles": (),
    }
    qc = inspect_entity_qc(record)
    token_count = vocabulary.token_count(canonical_smiles)
    if token_count > pretrain_config.data.max_smiles_tokens:
        qc.reasons.append("smiles_overlength")
    sample: dict[str, Any] | None = None
    error_message = ""
    if not qc.reasons:
        try:
            mol = Chem.MolFromSmiles(canonical_smiles)
            if mol is None:
                raise ValueError("RDKit parsing failed after canonicalization")
            raw = calculate_descriptors(mol, raw_names)
            sample = build_entity_sample(
                record,
                raw,
                schema,
                standardizer,
                vocabulary,
                pretrain_config,
            )
        except (RuntimeError, ValueError, OverflowError) as error:
            qc.reasons.append("feature_error")
            error_message = str(error)
    exclusion = None
    if sample is None:
        exclusion = {
            "role": role,
            "canonical_smiles": canonical_smiles,
            "exclusion_reasons": ";".join(qc.reasons),
            "unsupported_bond_types": ";".join(qc.unsupported_bond_types),
            "ipc": format(qc.ipc, ".17g"),
            "token_count": token_count,
            "max_smiles_tokens": pretrain_config.data.max_smiles_tokens,
            "detail": error_message,
        }
    return EntityFeatureResult(
        candidate_id=candidate_id,
        role=role,
        canonical_smiles=canonical_smiles,
        sample=sample,
        exclusion=exclusion,
    )


def _compute_entity_feature(
    item: tuple[int, str, str],
) -> EntityFeatureResult:
    result = _build_entity_feature_result(item)
    return EntityFeatureResult(
        candidate_id=result.candidate_id,
        role=result.role,
        canonical_smiles=result.canonical_smiles,
        sample=(
            None
            if result.sample is None
            else _sample_to_ipc_payload(result.sample)
        ),
        exclusion=result.exclusion,
    )


def _entity_feature_results(
    config: Stage2Config,
    collected: CollectedStage2Data,
    pretrain_config: PretrainConfig,
    vocabulary: SmilesTokenizer,
    schema: DescriptorSchema,
    standardizer: DescriptorStandardizer,
) -> Iterator[EntityFeatureResult]:
    inputs = (
        (candidate_id, role, canonical_smiles)
        for candidate_id, (role, canonical_smiles) in enumerate(
            collected.entity_keys
        )
    )
    workers = config.preparation.workers
    if workers == 1:
        _initialize_entity_worker(
            pretrain_config, vocabulary, schema, standardizer
        )
        for item in inputs:
            try:
                yield _materialize_entity_feature(
                    _compute_entity_feature(item)
                )
            except BaseException as error:
                candidate_id, role, canonical_smiles = item
                raise RuntimeError(
                    "Stage 2 entity worker failed for "
                    f"candidate_id={candidate_id}, role={role}, "
                    f"canonical_smiles={canonical_smiles}"
                ) from error
        return
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_entity_worker,
        initargs=(pretrain_config, vocabulary, schema, standardizer),
    ) as executor:
        pending: list[tuple[tuple[int, str, str], Future[EntityFeatureResult]]] = []
        iterator = iter(inputs)
        for _ in range(workers * 2):
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.append((item, executor.submit(_compute_entity_feature, item)))
        while pending:
            item, future = pending.pop(0)
            try:
                result = future.result()
            except BrokenProcessPool as error:
                candidate_id, role, canonical_smiles = item
                raise RuntimeError(
                    "Stage 2 entity worker pool failed while awaiting "
                    f"candidate_id={candidate_id}, role={role}, "
                    f"canonical_smiles={canonical_smiles}; the awaiting "
                    "candidate is not necessarily the process-pool failure cause"
                ) from error
            except BaseException as error:
                candidate_id, role, canonical_smiles = item
                raise RuntimeError(
                    "Stage 2 entity worker failed for "
                    f"candidate_id={candidate_id}, role={role}, "
                    f"canonical_smiles={canonical_smiles}"
                ) from error
            yield _materialize_entity_feature(result)
            try:
                next_item = next(iterator)
            except StopIteration:
                continue
            pending.append(
                (next_item, executor.submit(_compute_entity_feature, next_item))
            )


def _build_entity_shards(
    config: Stage2Config,
    collected: CollectedStage2Data,
    pretrain_config: PretrainConfig,
    vocabulary: SmilesTokenizer,
    schema: DescriptorSchema,
    standardizer: DescriptorStandardizer,
    reporter: ProgressReporter,
) -> tuple[list[dict[str, Any]], list[int], int]:
    output_dir = config.data.artifacts_dir
    shard_dir = output_dir / "entities"
    shard_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    retained_id_by_candidate = [-1] * len(collected.entity_keys)
    excluded_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    shard_number = 0
    active_paths: set[Path] = set()

    def flush() -> None:
        nonlocal samples, shard_number
        if not samples:
            return
        path = shard_dir / f"entities_{shard_number:05d}.pt"
        atomic_torch_save(
            path,
            {
                "format_version": STAGE2_ARTIFACT_VERSION,
                "kind": STAGE2_ARTIFACT_KIND,
                "samples": samples,
            },
        )
        active_paths.add(path)
        start = len(entries)
        for offset, sample in enumerate(samples):
            entries.append(
                {
                    "entity_id": start + offset,
                    "role": sample["role"],
                    "role_id": sample["role_id"],
                    "canonical_smiles": sample["canonical_smiles"],
                    "shard": str(path.relative_to(output_dir)),
                    "offset": offset,
                }
            )
        samples = []
        shard_number += 1

    with reporter.bar(
        total=len(collected.entity_keys),
        desc="Stage 2 entity features",
        unit="entity",
    ) as progress:
        for result in _entity_feature_results(
            config,
            collected,
            pretrain_config,
            vocabulary,
            schema,
            standardizer,
        ):
            if result.sample is None:
                if result.exclusion is None:
                    raise AssertionError("Excluded Stage 2 entity has no audit row")
                excluded_rows.append(result.exclusion)
                progress.update(1)
                continue
            retained_id = len(entries) + len(samples)
            result.sample["sample_id"] = f"stage2_entity_{retained_id:08d}"
            retained_id_by_candidate[result.candidate_id] = retained_id
            samples.append(result.sample)
            if len(samples) >= config.data.entity_shard_size:
                flush()
            progress.update(1)
    flush()

    for stale in shard_dir.glob("entities_*.pt"):
        if stale not in active_paths:
            stale.unlink()

    path = output_dir / "excluded_entities.csv"
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = (
        "role",
        "canonical_smiles",
        "exclusion_reasons",
        "unsupported_bond_types",
        "ipc",
        "token_count",
        "max_smiles_tokens",
        "detail",
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(excluded_rows)
    temporary.replace(path)
    return entries, retained_id_by_candidate, len(excluded_rows)


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Stage 2 train scaler has no values")
    array_value = np.asarray(values, dtype=np.float64)
    scale = float(array_value.std())
    return {"count": len(values), "mean": float(array_value.mean()), "scale": scale if math.isfinite(scale) and scale > 0 else 1.0}


def _write_mapping_audit(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = (
        "mol_id", "canonical_smiles", "structure_path", "model_atom_count",
        "structure_atom_count", "mapped_atom_count", "mapping_count",
        "selected_mapping_rank", "bond_match_mode", "unparsed_bond_types",
        "bond_fallback_reason", "status", "reason",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _retained_rows(staged: StagedTaskRows, retained_map: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    candidates = torch.tensor(staged.entity_ids, dtype=torch.long)
    entities = retained_map[candidates]
    indices = (entities >= 0).all(dim=1).nonzero(as_tuple=False).flatten().tolist()
    return entities[indices], indices


def _write_task_tensors(
    config: Stage2Config, registry: Stage2Registry, collected: CollectedStage2Data,
    retained_id_by_candidate: Sequence[int], reporter: ProgressReporter,
) -> tuple[dict[str, dict[str, int]], int, dict[str, Any]]:
    output_dir = config.data.artifacts_dir
    retained_map = torch.tensor(retained_id_by_candidate, dtype=torch.long)
    row_counts: dict[str, dict[str, int]] = {}
    scalers: dict[str, Any] = {}
    excluded_count = 0
    excluded_path = output_dir / "excluded_rows.csv"
    temporary = excluded_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("task", "split", "source_row", "reason"))
        writer.writeheader()
        with reporter.bar(total=len(registry.tasks) * 2, desc="Stage 2 task tensors", unit="file") as progress:
            for spec in registry.tasks:
                train = collected.rows[spec.task_id]["train"]
                _, train_keep = _retained_rows(train, retained_map)
                if not train_keep:
                    raise ValueError(f"Stage 2 QC removed every train row from {spec.task_id}")
                condition_scalers = {
                    name: _stats([train.conditions[index][column] for index in train_keep])
                    for column, name in enumerate(spec.condition_columns)
                }
                if spec.target_level == "atom":
                    molecules = [train.atom_targets[index] for index in train_keep]
                    molecule_means = [sum(values) / len(values) for values in molecules]
                    mean = sum(molecule_means) / len(molecule_means)
                    second = sum(sum(value * value for value in values) / len(values) for values in molecules) / len(molecules)
                    scale = math.sqrt(max(0.0, second - mean * mean))
                    target_scalers = {spec.target_columns[0]: {"count": len(molecules), "atom_count": sum(map(len, molecules)), "weighting": "molecule_equal", "mean": mean, "scale": scale if scale > 0 and math.isfinite(scale) else 1.0}}
                else:
                    target_scalers = {}
                    for column, name in enumerate(spec.target_columns):
                        values = [train.targets[index][column] for index in train_keep if train.target_mask[index][column]]
                        target_scalers[name] = _stats(values)
                scalers[spec.task_id] = {"conditions": condition_scalers, "targets": target_scalers}
                row_counts[spec.task_id] = {}
                for split in ("train", "valid"):
                    staged = collected.rows[spec.task_id][split]
                    entities, keep = _retained_rows(staged, retained_map)
                    kept = set(keep)
                    for index, row_number in enumerate(staged.source_rows):
                        if index not in kept:
                            excluded_count += 1
                            writer.writerow({"task": spec.task_id, "split": split, "source_row": row_number, "reason": "excluded_entity"})
                    if not keep:
                        raise ValueError(f"Stage 2 QC removed every row from {spec.task_id}/{split}")
                    raw_conditions = torch.tensor([staged.conditions[index] for index in keep], dtype=torch.float32).reshape(len(keep), len(spec.condition_columns))
                    conditions = raw_conditions.clone()
                    for column, name in enumerate(spec.condition_columns):
                        stats = condition_scalers[name]
                        conditions[:, column] = (conditions[:, column] - float(stats["mean"])) / float(stats["scale"])
                    payload: dict[str, Any] = {
                        "format_version": STAGE2_ARTIFACT_VERSION, "kind": STAGE2_ARTIFACT_KIND,
                        "task": spec.task_id, "split": split, "entity_indices": entities,
                        "conditions": conditions,
                        "source_rows": torch.tensor([staged.source_rows[index] for index in keep], dtype=torch.long),
                        "condition_columns": list(spec.condition_columns), "target_columns": list(spec.target_columns),
                    }
                    if spec.target_level == "object":
                        raw_targets = torch.tensor([staged.targets[index] for index in keep], dtype=torch.float32)
                        mask = torch.tensor([staged.target_mask[index] for index in keep], dtype=torch.bool)
                        targets = raw_targets.clone()
                        for column, name in enumerate(spec.target_columns):
                            stats = target_scalers[name]
                            normalized = (targets[:, column] - float(stats["mean"])) / float(stats["scale"])
                            targets[:, column] = torch.where(mask[:, column], normalized, torch.zeros_like(normalized))
                        payload.update({"targets": targets, "target_mask": mask})
                        if split == "valid":
                            payload["raw_targets"] = raw_targets
                    else:
                        raw_chunks = [staged.atom_targets[index] for index in keep]
                        raw = torch.tensor([value for chunk in raw_chunks for value in chunk], dtype=torch.float32)
                        offsets = [0]
                        for chunk in raw_chunks:
                            offsets.append(offsets[-1] + len(chunk))
                        stats = target_scalers[spec.target_columns[0]]
                        payload.update({
                            "atom_target_values": (raw - float(stats["mean"])) / float(stats["scale"]),
                            "atom_target_offsets": torch.tensor(offsets, dtype=torch.long),
                            "atom_target_mask": torch.ones_like(raw, dtype=torch.bool),
                            "mol_ids": [staged.mol_ids[index] for index in keep],
                        })
                        if split == "valid":
                            payload["raw_atom_target_values"] = raw
                    path = output_dir / "tasks" / spec.task_id / f"{split}.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_torch_save(path, payload)
                    row_counts[spec.task_id][split] = len(keep)
                    progress.update(1)
    temporary.replace(excluded_path)
    return row_counts, excluded_count, scalers


def prepare_stage2_data(config: Stage2Config, *, reporter: ProgressReporter | None = None) -> dict[str, Any]:
    config.validate()
    registry = load_stage2_registry(config.data.task_catalog_path)
    config.validate_registry(registry)
    reporter = reporter or ProgressReporter()
    output_dir = config.data.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = [config.data.task_catalog_path]
    for spec in registry.tasks:
        source_paths.extend(spec.dataset.split_path(config.data.data_root, split) for split in ("train", "valid"))
        manifest = spec.dataset.resource_manifest_path(config.data.data_root)
        if manifest is not None:
            source_paths.append(manifest)
    source_hashes = {str(path): sha256_file(path) for path in source_paths}
    pretrain_config, vocabulary, schema, standardizer, artifact_hash = load_stage1_feature_inputs(config.initialization.checkpoint, config.data.pretrain_artifacts_dir)
    feature_contract = {"artifact_hash": artifact_hash, "max_smiles_tokens": pretrain_config.data.max_smiles_tokens, "fingerprint": pretrain_config.to_dict()["fingerprint"]}
    model_contract = build_model_contract(
        pretrain_config.model.d_model, pretrain_config.model.n_heads, registry,
        object_layers=config.model.object_layers, object_ffn_dim=config.model.object_ffn_dim, dropout=config.model.dropout,
    )
    data_signature = canonical_json_sha256({"source_hashes": source_hashes, "feature_contract": feature_contract, "registry_hash": registry.registry_hash, "model_contract": model_contract, "entity_shard_size": config.data.entity_shard_size})
    metadata_path = output_dir / "metadata.json"
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("format_version") == STAGE2_ARTIFACT_VERSION and existing.get("kind") == STAGE2_ARTIFACT_KIND and existing.get("data_signature") == data_signature:
            Stage2EntityDataset(output_dir)
            for task in registry.task_ids:
                for split in ("train", "valid"):
                    Stage2TaskDataset(output_dir, task, split)
            reporter.emit_json({"event": "stage2_data_reused", **existing["summary"]})
            return existing
    collected = _collect_sources(config, registry, reporter)
    _write_duplicate_audit(output_dir / "duplicate_conditions.csv", collected.duplicate_rows)
    _write_missing_target_audit(output_dir / "missing_targets.csv", collected.missing_target_rows)
    _write_mapping_audit(output_dir / "partial_charge_mapping_audit.csv", collected.mapping_audit)
    entries, retained_id_by_candidate, excluded_entities = _build_entity_shards(config, collected, pretrain_config, vocabulary, schema, standardizer, reporter)
    row_counts, excluded_rows, scalers = _write_task_tensors(config, registry, collected, retained_id_by_candidate, reporter)
    atomic_json(output_dir / "scalers.json", scalers)
    atomic_json(output_dir / "entity_index.json", {"format_version": STAGE2_ARTIFACT_VERSION, "kind": STAGE2_ARTIFACT_KIND, "entries": entries})
    artifact_files = [
        "entity_index.json", "scalers.json", "excluded_entities.csv", "excluded_rows.csv",
        "duplicate_conditions.csv", "missing_targets.csv", "partial_charge_mapping_audit.csv",
        *[f"tasks/{task}/{split}.pt" for task in registry.task_ids for split in ("train", "valid")],
    ]
    shard_paths = sorted({entry["shard"] for entry in entries})
    artifact_files.extend(shard_paths)
    artifact_hashes = {relative: sha256_file(output_dir / relative) for relative in artifact_files}
    entity_artifact_hash = _semantic_payload_sha256({
        "entity_index": {"format_version": STAGE2_ARTIFACT_VERSION, "kind": STAGE2_ARTIFACT_KIND, "entries": entries},
        "entity_shards": [torch.load(output_dir / relative, map_location="cpu", weights_only=False) for relative in shard_paths],
    })
    metadata = {
        "format_version": STAGE2_ARTIFACT_VERSION, "kind": STAGE2_ARTIFACT_KIND,
        "data_signature": data_signature, "entity_artifact_hash": entity_artifact_hash,
        "pretrain_artifact_hash": artifact_hash, "feature_contract": feature_contract,
        "registry": registry.snapshot(), "registry_hash": registry.registry_hash,
        "catalog_sha256": registry.catalog_sha256, "model_contract": model_contract,
        "source_hashes": source_hashes, "source_counts": collected.source_counts,
        "rdkit_version": rdBase.rdkitVersion, "numpy_version": np.__version__, "torch_version": torch.__version__,
        "preparation": config.to_dict()["preparation"],
        "tensor_contract": {"conditions": "task_train_normalized_float32", "object_targets": "task_train_normalized_float32_masked_zero", "atom_targets": "ragged_molecule_equal_train_normalized_float32", "validation_raw_targets": "float32"},
        "scalers": scalers,
        "summary": {"entities_selected": len(collected.entity_keys), "entities_retained": len(entries), "entities_excluded": excluded_entities, "rows": row_counts, "rows_excluded": excluded_rows, "duplicate_conditions": len(collected.duplicate_rows), "partial_target_rows": sum(row["action"] == "retained" for row in collected.missing_target_rows), "all_target_missing_rows": sum(row["action"] == "excluded" for row in collected.missing_target_rows), "atom_mappings": sum(row["status"] == "mapped" for row in collected.mapping_audit), "atom_mapping_exclusions": sum(row["status"] == "excluded" for row in collected.mapping_audit)},
        "artifact_hashes": artifact_hashes,
    }
    atomic_json(metadata_path, metadata)
    reporter.emit_json({"event": "stage2_data_complete", **metadata["summary"]})
    return metadata


TEACHER_CACHE_VERSION = 3
TEACHER_CACHE_KIND = "ilume_stage2_object_teacher"
TEACHER_EXTRACTION_CONTRACT_VERSION = 1


def _semantic_payload_sha256(payload: Any) -> str:
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode())
            digest.update(b"\0")
            digest.update(
                tensor.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()
            )
        elif isinstance(value, np.ndarray):
            array_value = np.ascontiguousarray(value)
            digest.update(b"ndarray\0")
            digest.update(str(array_value.dtype).encode())
            digest.update(b"\0")
            digest.update(json.dumps(list(array_value.shape)).encode())
            digest.update(b"\0")
            digest.update(array_value.tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict\0")
            for key in sorted(value):
                update(key)
                update(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(b"list\0")
            for item in value:
                update(item)
        elif isinstance(value, array):
            digest.update(b"array\0")
            digest.update(value.typecode.encode())
            digest.update(b"\0")
            digest.update(value.tobytes())
        elif value is None:
            digest.update(b"none\0")
        elif isinstance(value, (str, int, float, bool)):
            digest.update(type(value).__name__.encode())
            digest.update(b"\0")
            digest.update(repr(value).encode())
            digest.update(b"\0")
        else:
            raise TypeError(
                "Unsupported Stage 2 semantic hash value: "
                f"{type(value).__name__}"
            )

    update(payload)
    return digest.hexdigest()


def _is_reconstruction_parameter(name: str) -> bool:
    return any(
        name == prefix or name.startswith(prefix + ".")
        for prefix in RECONSTRUCTION_MODULES
    )


def stage1_encoding_state_hash(loaded: LoadedStage1Model) -> str:
    digest = hashlib.sha256()
    raw_config = loaded.config.to_dict()
    semantic_config = {
        "data": {
            "max_smiles_tokens": raw_config["data"]["max_smiles_tokens"],
        },
        "descriptor": raw_config["descriptor"],
        "fingerprint": raw_config["fingerprint"],
        "model": raw_config["model"],
    }
    digest.update(
        json.dumps(
            semantic_config,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for name, tensor in sorted(loaded.model.state_dict().items()):
        if _is_reconstruction_parameter(name):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(
            value.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def teacher_cache_identity(
    data_metadata: dict[str, Any],
    loaded: LoadedStage1Model,
    math_contract: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    payload = {
        "extraction_contract_version": TEACHER_EXTRACTION_CONTRACT_VERSION,
        "entity_artifact_hash": data_metadata["entity_artifact_hash"],
        "encoding_state_hash": stage1_encoding_state_hash(loaded),
        "model_contract": data_metadata["model_contract"],
        "dtype": "float32",
        "math_contract": math_contract,
    }
    return canonical_json_sha256(payload), payload


def teacher_cache_dir(config: Stage2Config, identity: str) -> Path:
    return config.data.artifacts_dir / "teachers" / identity


def _checkpoint_relative_path(path: Path) -> str:
    return os.path.relpath(path.resolve(), Path.cwd())


def prepare_teacher_cache(
    config: Stage2Config,
    *,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    config.validate()
    reporter = reporter or ProgressReporter()
    data_metadata = prepare_stage2_data(config, reporter=reporter)
    device = resolve_device(config.training.device)
    math_contract = configure_stage2_math(device)
    loaded = load_stage1_model(
        config.initialization.checkpoint,
        config.data.pretrain_artifacts_dir,
        device=device,
        backbone_dropout=0.0,
    )
    if loaded.artifact_hash != data_metadata["pretrain_artifact_hash"]:
        raise ValueError("Teacher checkpoint does not match Stage 2 entity features")
    identity, identity_payload = teacher_cache_identity(
        data_metadata, loaded, math_contract
    )
    output_dir = teacher_cache_dir(config, identity)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    embeddings_path = output_dir / "embeddings.pt"
    if metadata_path.is_file() and embeddings_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("format_version") == TEACHER_CACHE_VERSION
            and metadata.get("kind") == TEACHER_CACHE_KIND
            and metadata.get("identity") == identity
            and metadata.get("identity_payload") == identity_payload
            and metadata.get("embeddings_hash") == sha256_file(embeddings_path)
        ):
            embeddings = torch.load(
                embeddings_path,
                map_location="cpu",
                weights_only=True,
            )
            if tuple(embeddings.shape) == (
                int(metadata["entity_count"]),
                int(metadata["embedding_dim"]),
            ):
                checkpoint = _checkpoint_relative_path(
                    config.initialization.checkpoint
                )
                if metadata.get("checkpoint") != checkpoint:
                    metadata["checkpoint"] = checkpoint
                    atomic_json(metadata_path, metadata)
                reporter.emit_json(
                    {
                        "event": "stage2_teacher_cache_reused",
                        "identity": identity,
                    }
                )
                return {**metadata, "cache_reused": True}

    entity_dataset = Stage2EntityDataset(config.data.artifacts_dir)
    packer = MultimodalPacker(loaded.vocabulary)
    embeddings = torch.empty(
        (len(entity_dataset), loaded.config.model.d_model),
        dtype=torch.float32,
    )
    loaded.model.eval()
    with torch.inference_mode(), reporter.bar(
        total=len(entity_dataset),
        desc="Stage 2 teacher embeddings",
        unit="entity",
    ) as progress:
        for start in range(
            0, len(entity_dataset), config.preparation.teacher_batch_size
        ):
            end = min(
                len(entity_dataset),
                start + config.preparation.teacher_batch_size,
            )
            batch = packer(
                [entity_dataset[index] for index in range(start, end)]
            ).to(device)
            encoded = loaded.model.encode(batch).float().cpu()
            if not torch.isfinite(encoded).all():
                raise RuntimeError(
                    f"Non-finite teacher embedding in entity rows {start}:{end}"
                )
            embeddings[start:end] = encoded
            progress.update(end - start)
    atomic_torch_save(embeddings_path, embeddings)
    metadata = {
        "format_version": TEACHER_CACHE_VERSION,
        "kind": TEACHER_CACHE_KIND,
        "identity": identity,
        "identity_payload": identity_payload,
        "checkpoint": _checkpoint_relative_path(
            config.initialization.checkpoint
        ),
        "pretrain_artifact_hash": loaded.artifact_hash,
        "entity_artifact_hash": data_metadata["entity_artifact_hash"],
        "entity_count": len(entity_dataset),
        "embedding_dim": loaded.config.model.d_model,
        "model_contract": data_metadata["model_contract"],
        "dtype": "float32",
        "embeddings_hash": sha256_file(embeddings_path),
    }
    atomic_json(metadata_path, metadata)
    reporter.emit_json(
        {
            "event": "stage2_teacher_cache_complete",
            "entity_count": len(entity_dataset),
            "embedding_dim": loaded.config.model.d_model,
            "checkpoint": str(config.initialization.checkpoint),
            "identity": identity,
        }
    )
    return {**metadata, "cache_reused": False}


def load_teacher_embeddings(
    config: Stage2Config,
    loaded: LoadedStage1Model,
    data_metadata: dict[str, Any],
    math_contract: dict[str, Any],
    *,
    expected_count: int,
    expected_dim: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    identity, identity_payload = teacher_cache_identity(
        data_metadata, loaded, math_contract
    )
    output_dir = teacher_cache_dir(config, identity)
    metadata_path = output_dir / "metadata.json"
    embeddings_path = output_dir / "embeddings.pt"
    if not metadata_path.is_file() or not embeddings_path.is_file():
        raise FileNotFoundError(
            "Missing Stage 2 teacher cache; run scripts/stage2/prepare.py first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("format_version") != TEACHER_CACHE_VERSION
        or metadata.get("kind") != TEACHER_CACHE_KIND
    ):
        raise ValueError("Unsupported Stage 2 teacher cache format")
    if metadata.get("identity") != identity:
        raise ValueError("Stage 2 teacher cache identity mismatch")
    if metadata.get("identity_payload") != identity_payload:
        raise ValueError("Stage 2 teacher cache contract mismatch")
    if metadata.get("embeddings_hash") != sha256_file(embeddings_path):
        raise ValueError("Stage 2 teacher embedding hash mismatch")
    embeddings = torch.load(
        embeddings_path,
        map_location="cpu",
        weights_only=True,
    )
    if tuple(embeddings.shape) != (expected_count, expected_dim):
        raise ValueError("Stage 2 teacher embedding shape mismatch")
    if embeddings.dtype != torch.float32 or not torch.isfinite(embeddings).all():
        raise ValueError("Stage 2 teacher embeddings must be finite FP32")
    return embeddings, metadata
