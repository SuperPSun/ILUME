from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from array import array
from collections import Counter
from concurrent.futures import Future, ProcessPoolExecutor
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
from .config import STAGE2_TASKS, Stage2Config
from .data import (
    STAGE2_ARTIFACT_KIND,
    STAGE2_ARTIFACT_VERSION,
    Stage2EntityDataset,
    Stage2TaskDataset,
)
from .model import RECONSTRUCTION_MODULES
from .runtime import configure_stage2_math
from stage1.tokenizer import SmilesTokenizer


IL_TASKS = ("density", "heat_capacity", "thermal_expansion")
QM_MISSING_MARKERS = frozenset({"", "nan", "na", "n/a", "null"})
QM_TARGETS = (
    "ESP_max",
    "ESP_min",
    "ESP_std",
    "ESP_pos_frac",
    "Dipole",
    "Quadrupole",
    "q_max",
    "q_min",
    "q_std",
    "q_pos_frac",
    "gap_eV",
)


@dataclass(frozen=True)
class Stage2TaskSpec:
    name: str
    entity_columns: tuple[str, ...]
    entity_roles: tuple[str, ...]
    condition_columns: tuple[str, ...]
    target_columns: tuple[str, ...]

    @property
    def fieldnames(self) -> tuple[str, ...]:
        return (
            *self.entity_columns,
            *self.condition_columns,
            *self.target_columns,
            "source_list",
        )


TASK_SPECS: dict[str, Stage2TaskSpec] = {
    "simulated_qm_elec_hf": Stage2TaskSpec(
        "simulated_qm_elec_hf",
        ("SMILES",),
        ("neutral",),
        (),
        QM_TARGETS,
    ),
    "density": Stage2TaskSpec(
        "density",
        ("cation", "anion"),
        ("cation", "anion"),
        ("temperature_K",),
        ("density_g/cm^3",),
    ),
    "heat_capacity": Stage2TaskSpec(
        "heat_capacity",
        ("cation", "anion"),
        ("cation", "anion"),
        ("temperature_K",),
        ("heat_capacity_J/mol/K",),
    ),
    "thermal_expansion": Stage2TaskSpec(
        "thermal_expansion",
        ("cation", "anion"),
        ("cation", "anion"),
        ("temperature_K",),
        ("thermal_expansion_K^-1",),
    ),
    "transfer_organic": Stage2TaskSpec(
        "transfer_organic",
        ("solute", "solvent"),
        ("neutral", "neutral"),
        (),
        ("transfer_organic_kcal/mol",),
    ),
}


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
    if allow_missing and stripped.lower() in QM_MISSING_MARKERS:
        return None
    return _finite_float(stripped, context)


def _key_hash(parts: Sequence[str]) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.digest()


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    squared_deviations: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.squared_deviations += delta * (value - self.mean)

    def to_dict(self) -> dict[str, float | int]:
        variance = (
            self.squared_deviations / self.count if self.count else 0.0
        )
        scale = math.sqrt(max(variance, 0.0))
        if not math.isfinite(scale) or scale == 0.0:
            scale = 1.0
        return {"count": self.count, "mean": self.mean, "scale": scale}


@dataclass(frozen=True)
class CollectedStage2Data:
    entity_keys: tuple[tuple[str, str], ...]
    rows: dict[str, dict[str, "StagedTaskRows"]]
    scalers: dict[str, Any]
    source_counts: dict[str, dict[str, dict[str, int]]]
    duplicate_rows: tuple[dict[str, Any], ...]
    missing_target_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class StagedTaskRows:
    entity_ids: array
    conditions: array
    targets: array
    target_mask: array
    source_rows: array

    def __len__(self) -> int:
        return len(self.source_rows)


def _iter_rows(
    stage2_dir: Path,
    task: str,
    split: str,
) -> Iterator[tuple[int, dict[str, str]]]:
    spec = TASK_SPECS[task]
    path = stage2_dir / task / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stage 2 source: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != spec.fieldnames:
            raise ValueError(
                f"Unexpected Stage 2 columns in {path}: {reader.fieldnames}; "
                f"expected {list(spec.fieldnames)}"
            )
        for row_number, row in enumerate(reader, start=2):
            yield row_number, row


def _collect_sources_and_scalers(
    config: Stage2Config,
    reporter: ProgressReporter,
) -> CollectedStage2Data:
    canonical_cache: dict[str, str] = {}
    entity_key_ids: dict[tuple[str, str], int] = {}
    temperature_stats = RunningStats()
    target_stats = {
        column: RunningStats()
        for spec in TASK_SPECS.values()
        for column in spec.target_columns
    }
    source_counts: dict[str, dict[str, dict[str, int]]] = {}
    duplicate_rows: list[dict[str, Any]] = []
    missing_target_rows: list[dict[str, Any]] = []
    staged_rows: dict[str, dict[str, StagedTaskRows]] = {}

    with reporter.bar(
        total=len(STAGE2_TASKS) * 2,
        desc="Stage 2 scan/scalers",
        unit="file",
    ) as progress:
        for task in STAGE2_TASKS:
            spec = TASK_SPECS[task]
            train_keys: set[bytes] = set()
            source_counts[task] = {}
            staged_rows[task] = {}
            for split in ("train", "valid"):
                entity_values = array("q")
                condition_values = array("f")
                target_values = array("f")
                target_mask_values = array("b")
                source_rows = array("q")
                seen_keys: set[bytes] = set()
                detailed_seen: dict[tuple[str, ...], tuple[int, tuple[str, ...]]] = {}
                counts: Counter[str] = Counter()
                for row_number, row in _iter_rows(config.data.stage2_dir, task, split):
                    canonical_entities: list[str] = []
                    for column, role in zip(
                        spec.entity_columns, spec.entity_roles, strict=True
                    ):
                        raw = (row[column] or "").strip()
                        if not raw:
                            raise ValueError(
                                f"Empty {column} in {task}/{split}:{row_number}"
                            )
                        canonical = canonical_cache.get(raw)
                        if canonical is None:
                            canonical = _canonicalize(
                                raw, f"{task}/{split}:{row_number}/{column}"
                            )
                            canonical_cache[raw] = canonical
                        canonical_entities.append(canonical)
                    conditions = tuple(
                        _finite_float(
                            row[column], f"{task}/{split}:{row_number}/{column}"
                        )
                        for column in spec.condition_columns
                    )
                    targets = tuple(
                        _target_value(
                            row[column],
                            f"{task}/{split}:{row_number}/{column}",
                            allow_missing=task == "simulated_qm_elec_hf",
                        )
                        for column in spec.target_columns
                    )
                    source = (row["source_list"] or "").strip()
                    if not source:
                        raise ValueError(
                            f"Empty source_list in {task}/{split}:{row_number}"
                        )
                    missing_columns = tuple(
                        column
                        for column, value in zip(
                            spec.target_columns, targets, strict=True
                        )
                        if value is None
                    )
                    if missing_columns:
                        all_missing = len(missing_columns) == len(spec.target_columns)
                        missing_target_rows.append(
                            {
                                "task": task,
                                "split": split,
                                "source_row": row_number,
                                "missing_columns": ";".join(missing_columns),
                                "valid_target_count": len(spec.target_columns)
                                - len(missing_columns),
                                "action": "excluded" if all_missing else "retained",
                            }
                        )
                        if all_missing:
                            continue
                    for role, canonical in zip(
                        spec.entity_roles, canonical_entities, strict=True
                    ):
                        key = (role, canonical)
                        entity_id = entity_key_ids.get(key)
                        if entity_id is None:
                            entity_id = len(entity_key_ids)
                            entity_key_ids[key] = entity_id
                        entity_values.append(entity_id)
                    condition_values.extend(conditions)
                    target_values.extend(
                        0.0 if value is None else value for value in targets
                    )
                    target_mask_values.extend(value is not None for value in targets)
                    source_rows.append(row_number)
                    counts[source] += 1
                    input_parts = (
                        *canonical_entities,
                        *(format(value, ".17g") for value in conditions),
                    )
                    hashed = _key_hash(input_parts)
                    if split == "valid" and hashed in train_keys:
                        raise ValueError(
                            f"Stage 2 train/valid input overlap in {task}: "
                            + " | ".join(input_parts)
                        )
                    if task != "transfer_organic":
                        previous = detailed_seen.get(input_parts)
                        target_text = tuple(
                            "missing" if value is None else format(value, ".17g")
                            for value in targets
                        )
                        if previous is not None:
                            duplicate_rows.append(
                                {
                                    "task": task,
                                    "split": split,
                                    "input_key": " | ".join(input_parts),
                                    "first_row": previous[0],
                                    "duplicate_row": row_number,
                                    "first_targets": " | ".join(previous[1]),
                                    "duplicate_targets": " | ".join(target_text),
                                }
                            )
                        else:
                            detailed_seen[input_parts] = (row_number, target_text)
                    seen_keys.add(hashed)
                    if split == "train":
                        for value in conditions:
                            temperature_stats.update(value)
                        for column, value in zip(
                            spec.target_columns, targets, strict=True
                        ):
                            if value is not None:
                                target_stats[column].update(value)
                source_counts[task][split] = dict(sorted(counts.items()))
                staged_rows[task][split] = StagedTaskRows(
                    entity_ids=entity_values,
                    conditions=condition_values,
                    targets=target_values,
                    target_mask=target_mask_values,
                    source_rows=source_rows,
                )
                if split == "train":
                    train_keys = seen_keys
                progress.update(1)

        missing_train_targets = [
            name for name, stats in target_stats.items() if stats.count == 0
        ]
        if missing_train_targets:
            raise ValueError(
                "Stage 2 train split has no finite values for target columns: "
                + ", ".join(missing_train_targets)
            )

        sorted_keys = tuple(
            sorted(
                entity_key_ids,
                key=lambda value: (ROLE_TO_ID[value[0]], value[1]),
            )
        )
        sorted_id_by_key = {key: index for index, key in enumerate(sorted_keys)}
        remap = [None] * len(entity_key_ids)
        for key, old_id in entity_key_ids.items():
            remap[old_id] = sorted_id_by_key[key]
        for task_rows in staged_rows.values():
            for rows in task_rows.values():
                for index, old_id in enumerate(rows.entity_ids):
                    rows.entity_ids[index] = remap[old_id]

        return CollectedStage2Data(
            entity_keys=sorted_keys,
            rows=staged_rows,
            scalers={
                "temperature_K": temperature_stats.to_dict(),
                "targets": {
                    name: stats.to_dict() for name, stats in target_stats.items()
                },
            },
            source_counts=source_counts,
            duplicate_rows=tuple(duplicate_rows),
            missing_target_rows=tuple(missing_target_rows),
        )


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


def _compute_entity_feature(
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
                yield _compute_entity_feature(item)
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
                yield future.result()
            except BaseException as error:
                candidate_id, role, canonical_smiles = item
                raise RuntimeError(
                    "Stage 2 entity worker failed for "
                    f"candidate_id={candidate_id}, role={role}, "
                    f"canonical_smiles={canonical_smiles}"
                ) from error
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


def _write_task_tensors(
    config: Stage2Config,
    collected: CollectedStage2Data,
    retained_id_by_candidate: Sequence[int],
    scalers: dict[str, Any],
    reporter: ProgressReporter,
) -> tuple[dict[str, dict[str, int]], int]:
    output_dir = config.data.artifacts_dir
    task_dir = output_dir / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    row_counts: dict[str, dict[str, int]] = {}
    excluded_count = 0
    retained_id_map = torch.tensor(retained_id_by_candidate, dtype=torch.long)
    excluded_path = output_dir / "excluded_rows.csv"
    excluded_temporary = excluded_path.with_suffix(".csv.tmp")
    with excluded_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task", "split", "source_row", "reason"),
        )
        writer.writeheader()
        with reporter.bar(
            total=len(STAGE2_TASKS) * 2,
            desc="Stage 2 task tensors",
            unit="file",
        ) as progress:
            for task in STAGE2_TASKS:
                spec = TASK_SPECS[task]
                row_counts[task] = {}
                for split in ("train", "valid"):
                    staged = collected.rows[task][split]
                    row_total = len(staged)
                    candidate_entities = torch.tensor(
                        staged.entity_ids, dtype=torch.long
                    ).reshape(row_total, len(spec.entity_columns))
                    entities = retained_id_map[candidate_entities]
                    keep = (entities >= 0).all(dim=1)
                    staged_source_rows = torch.tensor(
                        staged.source_rows, dtype=torch.long
                    )
                    for row_number in staged_source_rows[~keep].tolist():
                        excluded_count += 1
                        writer.writerow(
                            {
                                "task": task,
                                "split": split,
                                "source_row": row_number,
                                "reason": "excluded_entity",
                            }
                        )
                    entities = entities[keep]
                    raw_conditions = torch.tensor(
                        staged.conditions, dtype=torch.float32
                    ).reshape(row_total, len(spec.condition_columns))[keep]
                    raw_targets = torch.tensor(
                        staged.targets, dtype=torch.float32
                    ).reshape(row_total, len(spec.target_columns))[keep]
                    target_mask = torch.tensor(
                        staged.target_mask, dtype=torch.bool
                    ).reshape(row_total, len(spec.target_columns))[keep]
                    rows_tensor = staged_source_rows[keep]
                    retained = int(keep.sum())
                    if retained == 0:
                        raise ValueError(
                            f"Stage 2 QC removed every row from {task}/{split}"
                        )
                    conditions = raw_conditions.clone()
                    for column, name in enumerate(spec.condition_columns):
                        stats = scalers[name]
                        conditions[:, column] = (
                            conditions[:, column] - float(stats["mean"])
                        ) / float(stats["scale"])
                    targets = raw_targets.clone()
                    for column, name in enumerate(spec.target_columns):
                        stats = scalers["targets"][name]
                        standardized = (
                            targets[:, column] - float(stats["mean"])
                        ) / float(stats["scale"])
                        targets[:, column] = torch.where(
                            target_mask[:, column],
                            standardized,
                            torch.zeros_like(standardized),
                        )
                    path = task_dir / f"{task}_{split}.pt"
                    payload: dict[str, Any] = {
                        "format_version": STAGE2_ARTIFACT_VERSION,
                        "kind": STAGE2_ARTIFACT_KIND,
                        "task": task,
                        "split": split,
                        "entity_indices": entities,
                        "conditions": conditions,
                        "targets": targets,
                        "target_mask": target_mask,
                        "source_rows": rows_tensor,
                        "condition_columns": list(spec.condition_columns),
                        "target_columns": list(spec.target_columns),
                    }
                    if split == "valid":
                        payload["raw_targets"] = raw_targets
                    atomic_torch_save(
                        path,
                        payload,
                    )
                    row_counts[task][split] = retained
                    progress.update(1)
    excluded_temporary.replace(excluded_path)
    return row_counts, excluded_count


def prepare_stage2_data(
    config: Stage2Config,
    *,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    config.validate()
    reporter = reporter or ProgressReporter()
    output_dir = config.data.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = [
        config.data.stage2_dir / task / f"{split}.csv"
        for task in STAGE2_TASKS
        for split in ("train", "valid")
    ]
    source_hashes = {str(path): sha256_file(path) for path in source_paths}
    (
        pretrain_config,
        vocabulary,
        schema,
        standardizer,
        artifact_hash,
    ) = load_stage1_feature_inputs(
        config.initialization.checkpoint,
        config.data.pretrain_artifacts_dir,
    )
    feature_contract = {
        "artifact_hash": artifact_hash,
        "max_smiles_tokens": pretrain_config.data.max_smiles_tokens,
        "fingerprint": pretrain_config.to_dict()["fingerprint"],
    }
    model_contract = {
        "d_model": pretrain_config.model.d_model,
        "n_heads": pretrain_config.model.n_heads,
    }
    data_signature = canonical_json_sha256(
        {
            "source_hashes": source_hashes,
            "feature_contract": feature_contract,
            "model_contract": model_contract,
            "entity_shard_size": config.data.entity_shard_size,
        }
    )
    metadata_path = output_dir / "metadata.json"
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            existing.get("format_version") == STAGE2_ARTIFACT_VERSION
            and existing.get("kind") == STAGE2_ARTIFACT_KIND
            and existing.get("data_signature") == data_signature
        ):
            Stage2EntityDataset(output_dir)
            for task in STAGE2_TASKS:
                for split in ("train", "valid"):
                    Stage2TaskDataset(output_dir, task, split)
            reporter.emit_json(
                {
                    "event": "stage2_data_reused",
                    **existing["summary"],
                }
            )
            return existing

    collected = _collect_sources_and_scalers(
        config,
        reporter,
    )
    _write_duplicate_audit(
        output_dir / "duplicate_conditions.csv",
        collected.duplicate_rows,
    )
    _write_missing_target_audit(
        output_dir / "missing_targets.csv",
        collected.missing_target_rows,
    )
    atomic_json(output_dir / "scalers.json", collected.scalers)
    entries, retained_id_by_candidate, excluded_entities = _build_entity_shards(
        config,
        collected,
        pretrain_config,
        vocabulary,
        schema,
        standardizer,
        reporter,
    )
    row_counts, excluded_rows = _write_task_tensors(
        config,
        collected,
        retained_id_by_candidate,
        collected.scalers,
        reporter,
    )
    index_path = output_dir / "entity_index.json"
    atomic_json(
        index_path,
        {
            "format_version": STAGE2_ARTIFACT_VERSION,
            "kind": STAGE2_ARTIFACT_KIND,
            "entries": entries,
        },
    )
    artifact_files = [
        "entity_index.json",
        "scalers.json",
        "excluded_entities.csv",
        "excluded_rows.csv",
        "duplicate_conditions.csv",
        "missing_targets.csv",
        *[
            f"tasks/{task}_{split}.pt"
            for task in STAGE2_TASKS
            for split in ("train", "valid")
        ],
    ]
    shard_paths = sorted({entry["shard"] for entry in entries})
    artifact_files.extend(shard_paths)
    artifact_hashes = {
        relative: sha256_file(output_dir / relative)
        for relative in artifact_files
    }
    entity_artifact_hash = _semantic_payload_sha256(
        {
            "entity_index": {
                "format_version": STAGE2_ARTIFACT_VERSION,
                "kind": STAGE2_ARTIFACT_KIND,
                "entries": entries,
            },
            "entity_shards": [
                torch.load(
                    output_dir / relative,
                    map_location="cpu",
                    weights_only=False,
                )
                for relative in shard_paths
            ],
        }
    )
    metadata = {
        "format_version": STAGE2_ARTIFACT_VERSION,
        "kind": STAGE2_ARTIFACT_KIND,
        "data_signature": data_signature,
        "entity_artifact_hash": entity_artifact_hash,
        "pretrain_artifact_hash": artifact_hash,
        "feature_contract": feature_contract,
        "model_contract": model_contract,
        "source_hashes": source_hashes,
        "source_counts": collected.source_counts,
        "rdkit_version": rdBase.rdkitVersion,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "preparation": config.to_dict()["preparation"],
        "tensor_contract": {
            "conditions": "train_normalized_float32",
            "targets": "train_normalized_float32_masked_zero",
            "validation_raw_targets": "float32",
        },
        "tasks": {
            name: {
                "entity_columns": list(spec.entity_columns),
                "entity_roles": list(spec.entity_roles),
                "condition_columns": list(spec.condition_columns),
                "target_columns": list(spec.target_columns),
            }
            for name, spec in TASK_SPECS.items()
        },
        "scalers": collected.scalers,
        "summary": {
            "entities_selected": len(collected.entity_keys),
            "entities_retained": len(entries),
            "entities_excluded": excluded_entities,
            "rows": row_counts,
            "rows_excluded": excluded_rows,
            "duplicate_conditions": len(collected.duplicate_rows),
            "partial_target_rows": sum(
                row["action"] == "retained"
                for row in collected.missing_target_rows
            ),
            "all_target_missing_rows": sum(
                row["action"] == "excluded"
                for row in collected.missing_target_rows
            ),
        },
        "artifact_hashes": artifact_hashes,
    }
    atomic_json(metadata_path, metadata)
    reporter.emit_json(
        {"event": "stage2_data_complete", **metadata["summary"]}
    )
    return metadata


TEACHER_CACHE_VERSION = 2
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
