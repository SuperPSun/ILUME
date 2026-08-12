from __future__ import annotations

import csv
import hashlib
import json
import math
from array import array
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from rdkit import Chem, rdBase
from torch.utils.data import Dataset

from stage1.config import PretrainConfig
from stage1.data import (
    MultimodalBatch,
)
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
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.progress import ProgressReporter
from common.training import canonical_json_sha256
from .config import STAGE2_TASKS, Stage2Config
from stage1.tokenizer import SmilesTokenizer


STAGE2_ARTIFACT_VERSION = 2
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
    scalers: dict[str, Any]
    source_counts: dict[str, dict[str, dict[str, int]]]
    duplicate_rows: tuple[dict[str, Any], ...]
    missing_target_rows: tuple[dict[str, Any], ...]


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
) -> CollectedStage2Data:
    canonical_cache: dict[str, str] = {}
    entity_keys: set[tuple[str, str]] = set()
    temperature_stats = RunningStats()
    target_stats = {
        column: RunningStats()
        for spec in TASK_SPECS.values()
        for column in spec.target_columns
    }
    source_counts: dict[str, dict[str, dict[str, int]]] = {}
    duplicate_rows: list[dict[str, Any]] = []
    missing_target_rows: list[dict[str, Any]] = []

    for task in STAGE2_TASKS:
        spec = TASK_SPECS[task]
        train_keys: set[bytes] = set()
        source_counts[task] = {}
        for split in ("train", "valid"):
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
                    entity_keys.add((role, canonical))
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
            if split == "train":
                train_keys = seen_keys

    missing_train_targets = [
        name for name, stats in target_stats.items() if stats.count == 0
    ]
    if missing_train_targets:
        raise ValueError(
            "Stage 2 train split has no finite values for target columns: "
            + ", ".join(missing_train_targets)
        )

    return CollectedStage2Data(
        entity_keys=tuple(
            sorted(
                entity_keys,
                key=lambda value: (ROLE_TO_ID[value[0]], value[1]),
            )
        ),
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


def _load_pretrain_inputs(
    config: Stage2Config,
) -> tuple[
    PretrainConfig,
    SmilesTokenizer,
    DescriptorSchema,
    DescriptorStandardizer,
    str,
]:
    return load_stage1_feature_inputs(
        config.initialization.checkpoint,
        config.data.pretrain_artifacts_dir,
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


def _build_entity_shards(
    config: Stage2Config,
    collected: CollectedStage2Data,
    pretrain_config: PretrainConfig,
    vocabulary: SmilesTokenizer,
    schema: DescriptorSchema,
    standardizer: DescriptorStandardizer,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int], int]:
    output_dir = config.data.artifacts_dir
    shard_dir = output_dir / "entities"
    shard_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    valid_ids: dict[tuple[str, str], int] = {}
    excluded_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    shard_number = 0
    active_paths: set[Path] = set()
    raw_names = rdkit_descriptor_names()

    def flush() -> None:
        nonlocal samples, shard_number
        if not samples:
            return
        path = shard_dir / f"entities_{shard_number:05d}.pt"
        atomic_torch_save(
            path,
            {
                "format_version": STAGE2_ARTIFACT_VERSION,
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

    for role, canonical_smiles in collected.entity_keys:
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
                entity_id = len(entries) + len(samples)
                record["sample_id"] = f"stage2_entity_{entity_id:08d}"
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
        if sample is None:
            excluded_rows.append(
                {
                    "role": role,
                    "canonical_smiles": canonical_smiles,
                    "exclusion_reasons": ";".join(qc.reasons),
                    "unsupported_bond_types": ";".join(
                        qc.unsupported_bond_types
                    ),
                    "ipc": format(qc.ipc, ".17g"),
                    "token_count": token_count,
                    "max_smiles_tokens": pretrain_config.data.max_smiles_tokens,
                    "detail": error_message,
                }
            )
            continue
        valid_ids[(role, canonical_smiles)] = len(entries) + len(samples)
        samples.append(sample)
        if len(samples) >= config.data.entity_shard_size:
            flush()
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
    return entries, valid_ids, len(excluded_rows)


def _write_task_tensors(
    config: Stage2Config,
    valid_ids: dict[tuple[str, str], int],
    scalers: dict[str, Any],
) -> tuple[dict[str, dict[str, int]], int]:
    output_dir = config.data.artifacts_dir
    task_dir = output_dir / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    canonical_cache: dict[str, str] = {}
    row_counts: dict[str, dict[str, int]] = {}
    excluded_count = 0
    excluded_path = output_dir / "excluded_rows.csv"
    excluded_temporary = excluded_path.with_suffix(".csv.tmp")
    with excluded_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task", "split", "source_row", "reason"),
        )
        writer.writeheader()
        for task in STAGE2_TASKS:
            spec = TASK_SPECS[task]
            row_counts[task] = {}
            for split in ("train", "valid"):
                entity_values = array("q")
                condition_values = array("f")
                target_values = array("f")
                target_mask_values = array("b")
                source_rows = array("q")
                retained = 0
                for row_number, row in _iter_rows(
                    config.data.stage2_dir, task, split
                ):
                    parsed_targets = tuple(
                        _target_value(
                            row[column],
                            f"{task}/{split}:{row_number}/{column}",
                            allow_missing=task == "simulated_qm_elec_hf",
                        )
                        for column in spec.target_columns
                    )
                    if all(value is None for value in parsed_targets):
                        continue
                    indices: list[int] = []
                    missing = False
                    for column, role in zip(
                        spec.entity_columns, spec.entity_roles, strict=True
                    ):
                        raw = row[column].strip()
                        canonical = canonical_cache.get(raw)
                        if canonical is None:
                            canonical = _canonicalize(
                                raw, f"{task}/{split}:{row_number}/{column}"
                            )
                            canonical_cache[raw] = canonical
                        entity_id = valid_ids.get((role, canonical))
                        if entity_id is None:
                            missing = True
                            break
                        indices.append(entity_id)
                    if missing:
                        excluded_count += 1
                        writer.writerow(
                            {
                                "task": task,
                                "split": split,
                                "source_row": row_number,
                                "reason": "excluded_entity",
                            }
                        )
                        continue
                    entity_values.extend(indices)
                    condition_values.extend(
                        _finite_float(
                            row[column],
                            f"{task}/{split}:{row_number}/{column}",
                        )
                        for column in spec.condition_columns
                    )
                    target_values.extend(
                        0.0 if value is None else value
                        for value in parsed_targets
                    )
                    target_mask_values.extend(
                        value is not None for value in parsed_targets
                    )
                    source_rows.append(row_number)
                    retained += 1
                if retained == 0:
                    raise ValueError(f"Stage 2 QC removed every row from {task}/{split}")
                entities = torch.tensor(entity_values, dtype=torch.long).reshape(
                    retained, len(spec.entity_columns)
                )
                conditions = torch.tensor(
                    condition_values, dtype=torch.float32
                ).reshape(retained, len(spec.condition_columns))
                targets = torch.tensor(
                    target_values, dtype=torch.float32
                ).reshape(retained, len(spec.target_columns))
                target_mask = torch.tensor(
                    target_mask_values, dtype=torch.bool
                ).reshape(retained, len(spec.target_columns))
                rows_tensor = torch.tensor(source_rows, dtype=torch.long)
                systems: dict[tuple[int, ...], list[int]] = {}
                if task in IL_TASKS:
                    for row_index, key in enumerate(entities.tolist()):
                        systems.setdefault(tuple(key), []).append(row_index)
                    system_offsets = [0]
                    system_rows: list[int] = []
                    for grouped_rows in systems.values():
                        system_rows.extend(grouped_rows)
                        system_offsets.append(len(system_rows))
                else:
                    system_offsets = []
                    system_rows = []
                path = task_dir / f"{task}_{split}.pt"
                atomic_torch_save(
                    path,
                    {
                        "format_version": STAGE2_ARTIFACT_VERSION,
                        "task": task,
                        "split": split,
                        "entity_indices": entities,
                        "conditions": conditions,
                        "targets": targets,
                        "target_mask": target_mask,
                        "source_rows": rows_tensor,
                        "system_offsets": torch.tensor(
                            system_offsets, dtype=torch.long
                        ),
                        "system_rows": torch.tensor(
                            system_rows, dtype=torch.long
                        ),
                        "condition_columns": list(spec.condition_columns),
                        "target_columns": list(spec.target_columns),
                        "scalers": scalers,
                    },
                )
                row_counts[task][split] = retained
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
    ) = _load_pretrain_inputs(config)
    feature_contract = {
        "artifact_hash": artifact_hash,
        "max_smiles_tokens": pretrain_config.data.max_smiles_tokens,
        "fingerprint": pretrain_config.to_dict()["fingerprint"],
    }
    data_signature = canonical_json_sha256(
        {
            "source_hashes": source_hashes,
            "feature_contract": feature_contract,
            "entity_shard_size": config.data.entity_shard_size,
        }
    )
    metadata_path = output_dir / "metadata.json"
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            existing.get("format_version") == STAGE2_ARTIFACT_VERSION
            and existing.get("data_signature") == data_signature
        ):
            Stage2EntityDataset(output_dir, config.data.shard_cache_size)
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

    with reporter.status("Stage 2 scan and scaler fit"):
        collected = _collect_sources_and_scalers(config)
    _write_duplicate_audit(
        output_dir / "duplicate_conditions.csv",
        collected.duplicate_rows,
    )
    _write_missing_target_audit(
        output_dir / "missing_targets.csv",
        collected.missing_target_rows,
    )
    atomic_json(output_dir / "scalers.json", collected.scalers)
    with reporter.status("Stage 2 entity features"):
        entries, valid_ids, excluded_entities = _build_entity_shards(
            config,
            collected,
            pretrain_config,
            vocabulary,
            schema,
            standardizer,
        )
    with reporter.status("Stage 2 task tensors"):
        row_counts, excluded_rows = _write_task_tensors(
            config,
            valid_ids,
            collected.scalers,
        )
    index_path = output_dir / "entity_index.json"
    atomic_json(
        index_path,
        {"format_version": STAGE2_ARTIFACT_VERSION, "entries": entries},
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
    metadata = {
        "format_version": STAGE2_ARTIFACT_VERSION,
        "data_signature": data_signature,
        "pretrain_artifact_hash": artifact_hash,
        "feature_contract": feature_contract,
        "source_hashes": source_hashes,
        "source_counts": collected.source_counts,
        "rdkit_version": rdBase.rdkitVersion,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
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
        "artifact_hashes": {
            relative: sha256_file(output_dir / relative)
            for relative in artifact_files
        },
    }
    atomic_json(metadata_path, metadata)
    reporter.emit_json(
        {"event": "stage2_data_complete", **metadata["summary"]}
    )
    return metadata


class Stage2EntityDataset(Dataset):
    def __init__(
        self,
        artifact_dir: str | Path,
        shard_cache_size: int = 4,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.metadata = json.loads(
            (self.artifact_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if self.metadata.get("format_version") != STAGE2_ARTIFACT_VERSION:
            raise ValueError("Unsupported Stage 2 artifact format")
        if self.metadata.get("rdkit_version") != rdBase.rdkitVersion:
            raise ValueError("RDKit version does not match the Stage 2 artifact")
        index_path = self.artifact_dir / "entity_index.json"
        expected_index_hash = self.metadata.get("artifact_hashes", {}).get(
            "entity_index.json"
        )
        if (
            expected_index_hash is None
            or sha256_file(index_path) != expected_index_hash
        ):
            raise ValueError("Stage 2 artifact hash mismatch: entity_index.json")
        payload = json.loads(
            index_path.read_text(encoding="utf-8")
        )
        if payload.get("format_version") != STAGE2_ARTIFACT_VERSION:
            raise ValueError("Unsupported Stage 2 entity index format")
        self.entries = payload["entries"]
        self.shard_cache_size = shard_cache_size
        self._cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._verified: set[str] = set()

    def __len__(self) -> int:
        return len(self.entries)

    def _load_shard(self, relative: str) -> list[dict[str, Any]]:
        if relative in self._cache:
            samples = self._cache.pop(relative)
            self._cache[relative] = samples
            return samples
        path = self.artifact_dir / relative
        if relative not in self._verified:
            expected = self.metadata["artifact_hashes"].get(relative)
            if expected is None or sha256_file(path) != expected:
                raise ValueError(f"Stage 2 artifact hash mismatch: {relative}")
            self._verified.add(relative)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format_version") != STAGE2_ARTIFACT_VERSION:
            raise ValueError(f"Unsupported Stage 2 entity shard: {relative}")
        samples = payload["samples"]
        self._cache[relative] = samples
        while len(self._cache) > self.shard_cache_size:
            self._cache.popitem(last=False)
        return samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        return self._load_shard(entry["shard"])[int(entry["offset"])]


class Stage2TaskDataset:
    def __init__(self, artifact_dir: str | Path, task: str, split: str) -> None:
        if task not in TASK_SPECS:
            raise ValueError(f"Unknown Stage 2 task: {task}")
        if split not in {"train", "valid"}:
            raise ValueError("Stage 2 split must be train or valid")
        self.artifact_dir = Path(artifact_dir)
        path = self.artifact_dir / "tasks" / f"{task}_{split}.pt"
        metadata = json.loads(
            (self.artifact_dir / "metadata.json").read_text(encoding="utf-8")
        )
        expected = metadata["artifact_hashes"].get(str(path.relative_to(self.artifact_dir)))
        if expected is None or sha256_file(path) != expected:
            raise ValueError(f"Stage 2 artifact hash mismatch: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format_version") != STAGE2_ARTIFACT_VERSION:
            raise ValueError("Unsupported Stage 2 task artifact format")
        if payload.get("task") != task or payload.get("split") != split:
            raise ValueError("Stage 2 task artifact identity mismatch")
        self.task = task
        self.split = split
        self.entity_indices = payload["entity_indices"]
        self.conditions = payload["conditions"]
        self.targets = payload["targets"]
        self.target_mask = payload["target_mask"]
        self.source_rows = payload["source_rows"]
        self.system_offsets = payload["system_offsets"]
        self.system_rows = payload["system_rows"]
        self.condition_columns = tuple(payload["condition_columns"])
        self.target_columns = tuple(payload["target_columns"])
        if self.target_mask.shape != self.targets.shape:
            raise ValueError("Stage 2 target mask shape mismatch")
        if task in IL_TASKS:
            if (
                self.system_offsets.ndim != 1
                or self.system_offsets.numel() < 2
                or int(self.system_offsets[0]) != 0
                or int(self.system_offsets[-1]) != len(self)
                or self.system_rows.shape != (len(self),)
            ):
                raise ValueError("Invalid Stage 2 IL system index")
        elif self.system_offsets.numel() or self.system_rows.numel():
            raise ValueError("Non-IL Stage 2 tasks cannot define system indices")

    def __len__(self) -> int:
        return int(self.entity_indices.shape[0])


@dataclass(frozen=True)
class Stage2Batch:
    task: str
    entities: MultimodalBatch
    entity_positions: torch.Tensor
    conditions: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    teacher_embeddings: torch.Tensor

    def to(self, device: torch.device | str) -> "Stage2Batch":
        return Stage2Batch(
            task=self.task,
            entities=self.entities.to(device),
            entity_positions=self.entity_positions.to(device),
            conditions=self.conditions.to(device),
            targets=self.targets.to(device),
            target_mask=self.target_mask.to(device),
            teacher_embeddings=self.teacher_embeddings.to(device),
        )


def build_stage2_batch(
    task_dataset: Stage2TaskDataset,
    indices: Sequence[int] | torch.Tensor,
    entity_dataset: Stage2EntityDataset,
    packer: MultimodalPacker,
    teacher_embeddings: torch.Tensor,
    scalers: dict[str, Any],
) -> Stage2Batch:
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    global_slots = task_dataset.entity_indices[index_tensor]
    unique_ids: list[int] = []
    local_by_global: dict[int, int] = {}
    local_values: list[int] = []
    for value in global_slots.flatten().tolist():
        local = local_by_global.get(value)
        if local is None:
            local = len(unique_ids)
            local_by_global[value] = local
            unique_ids.append(value)
        local_values.append(local)
    positions = torch.tensor(local_values, dtype=torch.long).reshape_as(global_slots)
    entities = packer([entity_dataset[index] for index in unique_ids])
    unique_teacher = teacher_embeddings[torch.tensor(unique_ids, dtype=torch.long)]
    conditions = task_dataset.conditions[index_tensor].clone()
    if task_dataset.condition_columns:
        temperature = scalers["temperature_K"]
        conditions = (conditions - float(temperature["mean"])) / float(
            temperature["scale"]
        )
    targets = task_dataset.targets[index_tensor].clone()
    target_mask = task_dataset.target_mask[index_tensor].clone()
    for column, name in enumerate(task_dataset.target_columns):
        stats = scalers["targets"][name]
        standardized = (
            targets[:, column] - float(stats["mean"])
        ) / float(stats["scale"])
        targets[:, column] = torch.where(
            target_mask[:, column],
            standardized,
            torch.zeros_like(standardized),
        )
    return Stage2Batch(
        task=task_dataset.task,
        entities=entities,
        entity_positions=positions,
        conditions=conditions,
        targets=targets,
        target_mask=target_mask,
        teacher_embeddings=unique_teacher,
    )


class TaskCursor:
    def __init__(self, size: int, seed: int) -> None:
        if size <= 0:
            raise ValueError("TaskCursor requires a non-empty dataset")
        self.size = size
        self.seed = seed
        self.cycle = 0
        self.position = 0
        self._permutation = self._build_permutation()

    def _build_permutation(self) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.seed + self.cycle)
        return torch.randperm(self.size, generator=generator)

    def _next_with_cycles(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        if count <= 0:
            raise ValueError("TaskCursor count must be positive")
        chunks: list[torch.Tensor] = []
        cycle_chunks: list[torch.Tensor] = []
        remaining = count
        while remaining:
            available = self.size - self.position
            take = min(remaining, available)
            chunks.append(self._permutation[self.position : self.position + take])
            cycle_chunks.append(torch.full((take,), self.cycle, dtype=torch.long))
            self.position += take
            remaining -= take
            if self.position == self.size:
                self.cycle += 1
                self.position = 0
                self._permutation = self._build_permutation()
        return torch.cat(chunks), torch.cat(cycle_chunks)

    def next_indices(self, count: int) -> torch.Tensor:
        indices, _ = self._next_with_cycles(count)
        return indices

    def state_dict(self) -> dict[str, int]:
        return {"cycle": self.cycle, "position": self.position}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        cycle = int(state["cycle"])
        position = int(state["position"])
        if cycle < 0 or not 0 <= position < self.size:
            raise ValueError("Invalid Stage 2 task cursor state")
        self.cycle = cycle
        self.position = position
        self._permutation = self._build_permutation()


class ILSystemCursor:
    def __init__(
        self,
        system_offsets: torch.Tensor,
        system_rows: torch.Tensor,
        seed: int,
    ) -> None:
        if system_offsets.ndim != 1 or system_offsets.numel() < 2:
            raise ValueError("ILSystemCursor requires at least one system")
        if int(system_offsets[0]) != 0 or int(system_offsets[-1]) != len(system_rows):
            raise ValueError("Invalid IL system CSR index")
        self.system_offsets = system_offsets.to(dtype=torch.long, device="cpu")
        self.system_rows = system_rows.to(dtype=torch.long, device="cpu")
        self.seed = seed
        self.system_cursor = TaskCursor(len(system_offsets) - 1, seed)

    def _row_for_visit(self, system_id: int, cycle: int) -> int:
        start = int(self.system_offsets[system_id])
        end = int(self.system_offsets[system_id + 1])
        size = end - start
        if size == 1:
            return int(self.system_rows[start])
        row_cycle, offset = divmod(cycle, size)
        mixed_seed = (
            self.seed
            + 1_000_003 * system_id
            + 1_000_000_007 * row_cycle
        ) % (2**63 - 1)
        generator = torch.Generator().manual_seed(mixed_seed)
        selected = int(torch.randperm(size, generator=generator)[offset])
        return int(self.system_rows[start + selected])

    def next_indices(self, count: int) -> torch.Tensor:
        system_ids, cycles = self.system_cursor._next_with_cycles(count)
        return torch.tensor(
            [
                self._row_for_visit(int(system_id), int(cycle))
                for system_id, cycle in zip(system_ids, cycles, strict=True)
            ],
            dtype=torch.long,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "il_system",
            "system_cursor": self.system_cursor.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("kind") != "il_system":
            raise ValueError("Invalid IL system cursor state")
        self.system_cursor.load_state_dict(state["system_cursor"])


class TaskBlockSampler:
    def __init__(
        self,
        probabilities: dict[str, float],
        block_size: int,
        seed: int,
    ) -> None:
        self.probabilities = dict(probabilities)
        self.block_size = block_size
        self.seed = seed
        tasks: list[str] = []
        for task in STAGE2_TASKS:
            quota = probabilities[task] * block_size
            if abs(quota - round(quota)) > 1.0e-8:
                raise ValueError("Task block quotas must be integers")
            tasks.extend([task] * round(quota))
        if len(tasks) != block_size:
            raise ValueError("Task block quotas do not fill the block")
        self._tasks = tuple(tasks)
        self._cache: dict[int, tuple[str, ...]] = {}

    def block(self, block_index: int) -> tuple[str, ...]:
        cached = self._cache.get(block_index)
        if cached is not None:
            return cached
        generator = torch.Generator().manual_seed(self.seed + block_index)
        order = torch.randperm(self.block_size, generator=generator).tolist()
        result = tuple(self._tasks[index] for index in order)
        self._cache = {block_index: result}
        return result

    def task_for_step(self, step: int) -> str:
        if step < 0:
            raise ValueError("Stage 2 step must be non-negative")
        block_index, offset = divmod(step, self.block_size)
        return self.block(block_index)[offset]
