from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import random
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from torch.utils.data import Dataset

from .config import DataConfig, PretrainConfig
from .descriptors import (
    DescriptorSchema,
    DescriptorStandardizer,
    calculate_descriptors,
    rdkit_descriptor_names,
)
from .fingerprints import FingerprintBatch, calculate_fingerprints
from .graph import GraphRecord, PackedGraph, featurize_mol
from common.progress import ProgressReporter
from .tokenizer import SmilesTokenizer, tokenizer_backend_version


ROLE_TO_ID = {"cation": 0, "anion": 1, "neutral": 2}
ROLE_SOURCE_FILES = {
    "cation": "cation.csv",
    "anion": "anion.csv",
    "neutral": "molecule.csv",
}
CORPUS_FORMAT_VERSION = 3
IPC_SQUARE_OVERFLOW_LIMIT = float(np.sqrt(np.finfo(np.float64).max))
BCUT_SUPPORTED_BOND_TYPES = {
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
}


def _preparation_source_paths(config: DataConfig) -> list[Path]:
    paths = [
        config.stage1_dir / filename for filename in ROLE_SOURCE_FILES.values()
    ]
    for role, filename in ROLE_SOURCE_FILES.items():
        if config.augmentation.get(role, 0.0) not in {0, 0.0}:
            paths.append(config.stage1_dir / "augmentation" / filename)
    return paths


def _csv_data_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


@dataclass
class EntityQC:
    record: dict[str, Any]
    reasons: list[str]
    unsupported_bond_types: tuple[str, ...]
    ipc: float
    token_count: int | None = None


@dataclass(frozen=True)
class MaskPlan:
    smiles_labels: torch.Tensor
    atom_mask: torch.Tensor
    bond_mask: torch.Tensor
    descriptor_indicator: torch.Tensor
    descriptor_loss_mask: torch.Tensor
    fingerprint_indicator: dict[str, torch.Tensor]
    fingerprint_loss_mask: dict[str, torch.Tensor]
    modality_dropped: torch.Tensor

    def to(self, device: torch.device | str) -> "MaskPlan":
        return MaskPlan(
            smiles_labels=self.smiles_labels.to(device),
            atom_mask=self.atom_mask.to(device),
            bond_mask=self.bond_mask.to(device),
            descriptor_indicator=self.descriptor_indicator.to(device),
            descriptor_loss_mask=self.descriptor_loss_mask.to(device),
            fingerprint_indicator={
                name: value.to(device) for name, value in self.fingerprint_indicator.items()
            },
            fingerprint_loss_mask={
                name: value.to(device) for name, value in self.fingerprint_loss_mask.items()
            },
            modality_dropped=self.modality_dropped.to(device),
        )


@dataclass(frozen=True)
class MultimodalBatch:
    token_ids: torch.Tensor
    token_padding_mask: torch.Tensor
    graphs: PackedGraph
    descriptors: torch.Tensor
    descriptor_valid: torch.Tensor
    fingerprints: FingerprintBatch
    roles: torch.Tensor
    sample_ids: tuple[str, ...]
    masks: MaskPlan | None = None

    def to(self, device: torch.device | str) -> "MultimodalBatch":
        return MultimodalBatch(
            token_ids=self.token_ids.to(device),
            token_padding_mask=self.token_padding_mask.to(device),
            graphs=self.graphs.to(device),
            descriptors=self.descriptors.to(device),
            descriptor_valid=self.descriptor_valid.to(device),
            fingerprints=self.fingerprints.to(device),
            roles=self.roles.to(device),
            sample_ids=self.sample_ids,
            masks=None if self.masks is None else self.masks.to(device),
        )


def _canonicalize(smiles: str, context: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES in {context}: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def _load_original_smiles(
    stage1_dir: Path, progress: Any
) -> dict[str, dict[str, set[str]]]:
    by_role: dict[str, dict[str, set[str]]] = {}
    for role, filename in ROLE_SOURCE_FILES.items():
        path = stage1_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing Stage 1 source: {path}")
        canonical_to_sources: dict[str, set[str]] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                progress.update(1)
                smiles = (row.get("SMILES") or "").strip()
                canonical = _canonicalize(smiles, f"{path}:{row_number}")
                canonical_to_sources.setdefault(canonical, set()).add(path.stem)
        by_role[role] = canonical_to_sources
    return by_role


def _split_original_records(
    by_role: dict[str, dict[str, set[str]]],
    config: DataConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for role_index, role in enumerate(("cation", "anion", "neutral")):
        items = sorted(by_role[role].items())
        rng = random.Random(config.seed + role_index)
        rng.shuffle(items)
        if config.max_samples_per_role is not None:
            items = items[: config.max_samples_per_role]
        if len(items) < 2:
            raise ValueError(f"Role {role} needs at least two original entities")
        valid_count = max(1, round(len(items) * config.valid_fraction))
        valid_count = min(valid_count, len(items) - 1)
        for index, (smiles, sources) in enumerate(items):
            records.append(
                {
                    "role": role,
                    "role_id": ROLE_TO_ID[role],
                    "canonical_smiles": smiles,
                    "sources": tuple(sorted(sources)),
                    "split": "valid" if index < valid_count else "train",
                    "is_augmented": False,
                    "seed_smiles": (),
                }
            )
    return records


def _seed_values(raw: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in re.split(r"[;|]", raw) if value.strip())


def _augmentation_limit(value: float | str, original_train_count: int, pool_size: int) -> int:
    if value == "all":
        return pool_size
    return min(pool_size, math.floor(float(value) * original_train_count))


def _load_augmentation_records(
    original_records: list[dict[str, Any]],
    config: DataConfig,
    progress: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    augmentation_dir = config.stage1_dir / "augmentation"
    selected: list[dict[str, Any]] = []
    audit: dict[str, dict[str, Any]] = {}
    for role_index, role in enumerate(("cation", "anion", "neutral")):
        requested = config.augmentation.get(role, 0.0)
        original_for_role = [record for record in original_records if record["role"] == role]
        original_all = {record["canonical_smiles"] for record in original_for_role}
        original_train = [record for record in original_for_role if record["split"] == "train"]
        valid_smiles = {
            record["canonical_smiles"] for record in original_for_role if record["split"] == "valid"
        }
        stats: dict[str, Any] = {
            "requested_multiplier": requested,
            "original_train_count": len(original_train),
            "available": 0,
            "excluded_valid_seed": 0,
            "excluded_overlap": 0,
            "excluded_duplicate": 0,
            "eligible": 0,
            "selected": 0,
            "actual_multiplier": 0.0,
            "excluded_qc": 0,
            "selected_after_qc": 0,
            "actual_multiplier_after_qc": 0.0,
        }
        if requested == 0 or requested == 0.0:
            audit[role] = stats
            continue
        filename = ROLE_SOURCE_FILES[role]
        path = augmentation_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing augmentation source: {path}")
        candidates: dict[str, dict[str, Any]] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                progress.update(1)
                smiles = (row.get("SMILES") or "").strip()
                canonical = _canonicalize(smiles, f"{path}:{row_number}")
                stats["available"] += 1
                seeds: list[str] = []
                for seed in _seed_values((row.get("seed_smiles_list") or "").strip()):
                    try:
                        seeds.append(_canonicalize(seed, f"{path}:{row_number} seed"))
                    except ValueError:
                        seeds.append(seed)
                if any(seed in valid_smiles for seed in seeds):
                    stats["excluded_valid_seed"] += 1
                    continue
                if canonical in original_all:
                    stats["excluded_overlap"] += 1
                    continue
                if canonical in candidates:
                    stats["excluded_duplicate"] += 1
                    continue
                candidates[canonical] = {
                    "role": role,
                    "role_id": ROLE_TO_ID[role],
                    "canonical_smiles": canonical,
                    "sources": (f"augmentation/{path.stem}",),
                    "split": "train",
                    "is_augmented": True,
                    "seed_smiles": tuple(sorted(set(seeds))),
                }
        pool = list(candidates.values())
        stats["eligible"] = len(pool)
        rng = random.Random(config.seed + 100 + role_index)
        rng.shuffle(pool)
        limit = _augmentation_limit(requested, len(original_train), len(pool))
        selected.extend(pool[:limit])
        stats["selected"] = limit
        stats["actual_multiplier"] = limit / len(original_train)
        audit[role] = stats
    return selected, audit


def _assign_sample_ids(records: list[dict[str, Any]]) -> None:
    records.sort(
        key=lambda item: (
            item["role_id"],
            item["split"] != "valid",
            item["is_augmented"],
            item["canonical_smiles"],
        )
    )
    counters = {role: 0 for role in ROLE_TO_ID}
    for record in records:
        role = record["role"]
        counters[role] += 1
        record["sample_id"] = f"{role}_{counters[role]:08d}"


def _calculate_ipc(mol: Chem.Mol) -> float:
    return float(Descriptors.Ipc(mol))


def _inspect_entity_qc(record: dict[str, Any]) -> EntityQC:
    mol = Chem.MolFromSmiles(record["canonical_smiles"])
    if mol is None:
        raise RuntimeError("Canonical SMILES unexpectedly failed RDKit parsing")
    unsupported = tuple(
        sorted(
            {
                str(bond.GetBondType())
                for bond in mol.GetBonds()
                if bond.GetBondType() not in BCUT_SUPPORTED_BOND_TYPES
            }
        )
    )
    reasons: list[str] = []
    if unsupported:
        reasons.append("unsupported_bcut_bond_type")
    try:
        ipc = _calculate_ipc(mol)
    except Exception:
        ipc = float("nan")
    if not np.isfinite(ipc):
        reasons.append("ipc_nonfinite")
    elif abs(ipc) > IPC_SQUARE_OVERFLOW_LIMIT:
        reasons.append("ipc_square_overflow")
    return EntityQC(record, reasons, unsupported, ipc)


def _validate_qc_role_splits(entities: list[EntityQC]) -> None:
    missing = [
        f"{role}/{split}"
        for role in ROLE_TO_ID
        for split in ("train", "valid")
        if not any(
            not entity.reasons
            and entity.record["role"] == role
            and entity.record["split"] == split
            for entity in entities
        )
    ]
    if missing:
        raise ValueError(
            "Quality control removed every entity from: " + ", ".join(missing)
        )


def _fit_tokenizer_and_filter_lengths(
    entities: list[EntityQC], config: PretrainConfig, reporter: ProgressReporter
) -> tuple[SmilesTokenizer, list[EntityQC]]:
    pass_index = 0
    while True:
        pass_index += 1
        _validate_qc_role_splits(entities)
        retained = [entity for entity in entities if not entity.reasons]
        with reporter.status(f"Tokenizer fit pass {pass_index}"):
            tokenizer = SmilesTokenizer.fit(
                [
                    entity.record["canonical_smiles"]
                    for entity in retained
                    if entity.record["split"] == "train"
                ],
                backend=config.tokenizer.backend,
                vocab_size=config.tokenizer.vocab_size,
                min_frequency=config.tokenizer.min_frequency,
            )
        newly_excluded = 0
        with reporter.bar(
            total=len(retained),
            desc=f"Token length QC pass {pass_index}",
            unit="entity",
        ) as progress:
            for entity in retained:
                entity.token_count = tokenizer.token_count(
                    entity.record["canonical_smiles"]
                )
                if entity.token_count > config.data.max_smiles_tokens:
                    entity.reasons.append("smiles_overlength")
                    newly_excluded += 1
                progress.update(1)
        if newly_excluded == 0:
            break

    missing_counts = [entity for entity in entities if entity.token_count is None]
    with reporter.bar(
        total=len(missing_counts),
        desc="Excluded token audit",
        unit="entity",
    ) as progress:
        for entity in missing_counts:
            entity.token_count = tokenizer.token_count(
                entity.record["canonical_smiles"]
            )
            if entity.token_count > config.data.max_smiles_tokens:
                entity.reasons.append("smiles_overlength")
            progress.update(1)
    _validate_qc_role_splits(entities)
    return tokenizer, [entity for entity in entities if not entity.reasons]


def _axis_counts(
    values: list[str], denominator: int
) -> dict[str, dict[str, float | int]]:
    return {
        value: {
            "count": count,
            "percent_of_pre_filter": 100.0 * count / denominator,
        }
        for value, count in sorted(Counter(values).items())
    }


def _quality_control_summary(
    entities: list[EntityQC], tokenizer: SmilesTokenizer, max_smiles_tokens: int
) -> dict[str, Any]:
    excluded = [entity for entity in entities if entity.reasons]
    return {
        "pre_filter_total": len(entities),
        "post_filter_total": len(entities) - len(excluded),
        "thresholds": {
            "bcut_supported_bond_types": [
                "SINGLE",
                "DOUBLE",
                "TRIPLE",
                "AROMATIC",
            ],
            "ipc_square_overflow_abs": IPC_SQUARE_OVERFLOW_LIMIT,
            "tokenizer_backend": tokenizer.backend,
            "max_smiles_tokens": max_smiles_tokens,
        },
        "excluded": {
            "total": len(excluded),
            "percent_of_pre_filter": 100.0 * len(excluded) / len(entities),
            "by_role": _axis_counts(
                [entity.record["role"] for entity in excluded], len(entities)
            ),
            "by_split": _axis_counts(
                [entity.record["split"] for entity in excluded], len(entities)
            ),
            "by_source": _axis_counts(
                [
                    source
                    for entity in excluded
                    for source in entity.record["sources"]
                ],
                len(entities),
            ),
            "by_reason": _axis_counts(
                [
                    reason
                    for entity in excluded
                    for reason in entity.reasons
                ],
                len(entities),
            ),
        },
    }


def _write_excluded_entities(
    path: Path,
    entities: list[EntityQC],
    tokenizer: SmilesTokenizer,
    max_smiles_tokens: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "canonical_smiles",
                "role",
                "split",
                "is_augmented",
                "sources",
                "exclusion_reasons",
                "unsupported_bond_types",
                "ipc",
                "tokenizer_backend",
                "token_count",
                "max_smiles_tokens",
            ],
        )
        writer.writeheader()
        for entity in entities:
            if not entity.reasons:
                continue
            record = entity.record
            writer.writerow(
                {
                    "canonical_smiles": record["canonical_smiles"],
                    "role": record["role"],
                    "split": record["split"],
                    "is_augmented": int(record["is_augmented"]),
                    "sources": ";".join(record["sources"]),
                    "exclusion_reasons": ";".join(entity.reasons),
                    "unsupported_bond_types": ";".join(
                        entity.unsupported_bond_types
                    ),
                    "ipc": format(entity.ipc, ".17g"),
                    "tokenizer_backend": tokenizer.backend,
                    "token_count": entity.token_count,
                    "max_smiles_tokens": max_smiles_tokens,
                }
            )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _build_sample(
    record: dict[str, Any],
    raw_descriptors: np.ndarray,
    schema: DescriptorSchema,
    standardizer: DescriptorStandardizer,
    tokenizer: SmilesTokenizer,
    config: PretrainConfig,
) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(record["canonical_smiles"])
    if mol is None:
        raise RuntimeError("Canonical SMILES unexpectedly failed RDKit parsing")
    encoded = tokenizer.encode(
        record["canonical_smiles"], config.data.max_smiles_tokens
    )
    selected = schema.select(raw_descriptors[None, :])
    standardized, descriptor_valid = standardizer.transform(selected)
    invalid = descriptor_valid & ~np.isfinite(standardized)
    if invalid.any():
        names = [
            schema.selected_names[index]
            for index in np.flatnonzero(invalid[0]).tolist()
        ]
        raise ValueError(
            f"Non-finite standardized descriptors for {record['sample_id']}: "
            + ", ".join(names)
        )
    graph = featurize_mol(mol)
    return {
        **record,
        "token_ids": torch.tensor(encoded, dtype=torch.long),
        "atom_categorical": graph.atom_categorical,
        "atom_continuous": graph.atom_continuous,
        "bond_categorical": graph.bond_categorical,
        "bond_index": graph.bond_index,
        "descriptors": torch.from_numpy(standardized[0]),
        "descriptor_valid": torch.from_numpy(descriptor_valid[0]),
        "fingerprints": {
            name: torch.from_numpy(value).float()
            for name, value in calculate_fingerprints(
                mol, config.fingerprint
            ).items()
        },
    }


def _write_shards(
    records: list[dict[str, Any]],
    raw_matrix: np.ndarray,
    schema: DescriptorSchema,
    standardizer: DescriptorStandardizer,
    tokenizer: SmilesTokenizer,
    config: PretrainConfig,
    output_dir: Path,
    shard_size: int,
    preparation_signature: str,
    reporter: ProgressReporter,
) -> tuple[list[dict[str, Any]], list[int], int]:
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    lengths: list[int] = []
    unk_count = 0
    active_paths: set[Path] = set()
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
    for index, record in enumerate(records):
        grouped.setdefault((record["split"], record["role"]), []).append(
            (index, record)
        )
    total_shards = sum(
        math.ceil(len(group) / shard_size) for group in grouped.values()
    )
    built_count = 0
    reused_count = 0
    with reporter.bar(
        total=total_shards,
        desc="Shards",
        unit="shard",
    ) as progress:
        for (split, role), group in sorted(grouped.items()):
            for shard_number, start in enumerate(
                range(0, len(group), shard_size)
            ):
                chunk = group[start : start + shard_size]
                filename = f"{split}_{role}_{shard_number:05d}.pt"
                path = shard_dir / filename
                active_paths.add(path)
                expected_ids = [record["sample_id"] for _, record in chunk]
                samples: list[dict[str, Any]] | None = None
                if path.is_file():
                    existing = torch.load(
                        path, map_location="cpu", weights_only=False
                    )
                    existing_ids = [
                        sample["sample_id"]
                        for sample in existing.get("samples", [])
                    ]
                    if (
                        existing.get("format_version")
                        == CORPUS_FORMAT_VERSION
                        and existing.get("preparation_signature")
                        == preparation_signature
                        and existing_ids == expected_ids
                    ):
                        samples = existing["samples"]
                reused = samples is not None
                if samples is None:
                    samples = [
                        _build_sample(
                            record,
                            raw_matrix[index],
                            schema,
                            standardizer,
                            tokenizer,
                            config,
                        )
                        for index, record in chunk
                    ]
                    temporary = path.with_suffix(".pt.tmp")
                    torch.save(
                        {
                            "format_version": CORPUS_FORMAT_VERSION,
                            "preparation_signature": preparation_signature,
                            "samples": samples,
                        },
                        temporary,
                    )
                    temporary.replace(path)
                    built_count += 1
                else:
                    reused_count += 1
                reporter.emit_json(
                    {
                        "event": "prepare_shard",
                        "shard": filename,
                        "samples": len(samples),
                        "reused": reused,
                    }
                )
                lengths.extend(
                    sample["token_ids"].numel() for sample in samples
                )
                unk_count += sum(
                    int(
                        (sample["token_ids"] == tokenizer.unk_id).sum().item()
                    )
                    for sample in samples
                )
                for offset, sample in enumerate(chunk):
                    record = sample[1]
                    entries.append(
                        {
                            "sample_id": record["sample_id"],
                            "role_id": record["role_id"],
                            "role": record["role"],
                            "split": record["split"],
                            "shard": f"shards/{filename}",
                            "offset": offset,
                        }
                    )
                progress.set_postfix(
                    {"built": built_count, "reused": reused_count},
                    refresh=False,
                )
                progress.update(1)
    for stale in shard_dir.glob("*.pt"):
        if stale not in active_paths:
            stale.unlink()
    return entries, lengths, unk_count


def _preparation_signature(
    config: PretrainConfig,
    source_hashes: dict[str, str],
) -> str:
    config_payload = config.to_dict()
    payload = {
        "signature_version": 3,
        "artifact_format_version": CORPUS_FORMAT_VERSION,
        "rdkit_version": rdBase.rdkitVersion,
        "data": config_payload["data"],
        "tokenizer": config_payload["tokenizer"],
        "descriptor": config_payload["descriptor"],
        "fingerprint": config_payload["fingerprint"],
        "source_hashes": source_hashes,
    }
    payload["data"].pop("artifacts_dir", None)
    payload["data"].pop("shard_cache_size", None)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def prepare_corpus(config: PretrainConfig | DataConfig) -> dict[str, int]:
    if isinstance(config, DataConfig):
        config = PretrainConfig(data=config)
    config.validate()
    data_config = config.data
    output_dir = data_config.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter()
    raw_names = rdkit_descriptor_names()
    if len(raw_names) != data_config.descriptor_dim:
        raise ValueError(
            f"Configured descriptor_dim={data_config.descriptor_dim}, but this RDKit "
            f"provides {len(raw_names)} descriptors"
        )

    source_paths = _preparation_source_paths(data_config)
    with reporter.status("Count input rows"):
        source_row_count = sum(
            _csv_data_row_count(path) for path in source_paths
        )
    with reporter.bar(
        total=source_row_count,
        desc="Load/canonicalize",
        unit="row",
    ) as progress:
        originals = _split_original_records(
            _load_original_smiles(data_config.stage1_dir, progress),
            data_config,
        )
        augmented, augmentation_audit = _load_augmentation_records(
            originals,
            data_config,
            progress,
        )
    selected_records = [*originals, *augmented]
    quality_entities: list[EntityQC] = []
    with reporter.bar(
        total=len(selected_records),
        desc="Entity QC",
        unit="entity",
    ) as progress:
        for record in selected_records:
            quality_entities.append(_inspect_entity_qc(record))
            progress.update(1)
    tokenizer, retained_entities = _fit_tokenizer_and_filter_lengths(
        quality_entities, config, reporter
    )
    records = [entity.record for entity in retained_entities]
    excluded_entities = [entity for entity in quality_entities if entity.reasons]
    _assign_sample_ids(records)
    quality_control = _quality_control_summary(
        quality_entities, tokenizer, data_config.max_smiles_tokens
    )
    for role, stats in augmentation_audit.items():
        retained_count = sum(
            record["role"] == role and record["is_augmented"]
            for record in records
        )
        stats["excluded_qc"] = int(stats["selected"]) - retained_count
        stats["selected_after_qc"] = retained_count
        original_train_count = int(stats["original_train_count"])
        stats["actual_multiplier_after_qc"] = (
            retained_count / original_train_count
        )
    reporter.emit_json(
        {
            "event": "prepare_quality_control",
            "selected": len(selected_records),
            "retained": len(records),
            "excluded": len(excluded_entities),
        }
    )
    with reporter.status("Hash source files"):
        source_hashes = {str(path): _sha256(path) for path in source_paths}
    preparation_signature = _preparation_signature(config, source_hashes)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.is_file():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_metadata.get("preparation_signature") == preparation_signature:
            PreparedCorpusDataset(output_dir, "train", data_config.shard_cache_size)
            PreparedCorpusDataset(output_dir, "valid", data_config.shard_cache_size)
            return {key: int(value) for key, value in existing_metadata["summary"].items()}

    _write_excluded_entities(
        output_dir / "excluded_entities.csv",
        quality_entities,
        tokenizer,
        data_config.max_smiles_tokens,
    )

    state_path = output_dir / "preparation_state.json"
    raw_cache_path = output_dir / ".raw_descriptors.npy"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {}
    )
    reuse_raw = (
        state.get("preparation_signature") == preparation_signature
        and state.get("phase") in {"descriptors", "shards"}
        and raw_cache_path.is_file()
    )
    if reuse_raw:
        raw_matrix = np.load(raw_cache_path, mmap_mode="r+")
        if raw_matrix.shape != (len(records), len(raw_names)):
            reuse_raw = False
    if not reuse_raw:
        raw_matrix = np.lib.format.open_memmap(
            raw_cache_path,
            mode="w+",
            dtype=np.float64,
            shape=(len(records), len(raw_names)),
        )
        with reporter.bar(
            total=len(records),
            desc="Descriptors",
            unit="entity",
        ) as progress:
            for index, record in enumerate(records):
                mol = Chem.MolFromSmiles(record["canonical_smiles"])
                if mol is None:
                    raise RuntimeError(
                        "Canonical SMILES unexpectedly failed RDKit parsing"
                    )
                raw_matrix[index] = calculate_descriptors(mol, raw_names)
                progress.update(1)
                if (index + 1) % 10000 == 0 or index + 1 == len(records):
                    reporter.emit_json(
                        {
                            "event": "prepare_descriptors",
                            "completed": index + 1,
                            "total": len(records),
                        }
                    )
        raw_matrix.flush()
        _atomic_json(
            state_path,
            {
                "format_version": 1,
                "preparation_signature": preparation_signature,
                "phase": "descriptors",
            },
        )
    else:
        with reporter.bar(
            total=len(records),
            desc="Descriptors (reused)",
            unit="entity",
            initial=len(records),
        ):
            pass
    with reporter.status("Descriptor schema/scaler"):
        train_indices = np.asarray(
            [
                index
                for index, record in enumerate(records)
                if record["split"] == "train"
            ],
            dtype=np.int64,
        )
        training_cache_path = output_dir / ".train_descriptors.npy"
        training_matrix = np.lib.format.open_memmap(
            training_cache_path,
            mode="w+",
            dtype=np.float64,
            shape=(len(train_indices), len(raw_names)),
        )
        for start in range(0, len(train_indices), 65536):
            selected_rows = train_indices[start : start + 65536]
            training_matrix[start : start + len(selected_rows)] = raw_matrix[
                selected_rows
            ]
        training_matrix.flush()
        schema = DescriptorSchema.fit(
            training_matrix,
            raw_names,
            mode=config.descriptor.mode,
            token_count=config.descriptor.token_count,
            correlation_threshold=config.descriptor.correlation_threshold,
        )
        standardizer = DescriptorStandardizer.fit_columns(
            training_matrix,
            schema.retained_indices,
            schema.selected_names,
        )
        fitted = standardizer.finite_counts > 0
        invalid_statistics = fitted & (
            ~np.isfinite(standardizer.means)
            | ~np.isfinite(standardizer.scales)
            | (standardizer.scales <= 0.0)
        )
        if invalid_statistics.any():
            names = [
                standardizer.names[index]
                for index in np.flatnonzero(invalid_statistics).tolist()
            ]
            raise ValueError(
                "Non-finite descriptor standardization statistics: "
                + ", ".join(names)
            )
        tokenizer.save(output_dir / "tokenizer.json")
        schema.save(output_dir / "descriptor_schema.json")
        standardizer.save(output_dir / "descriptor_scaler.json")
    _atomic_json(
        state_path,
        {
            "format_version": 1,
            "preparation_signature": preparation_signature,
            "phase": "shards",
        },
    )
    entries, lengths, unk_count = _write_shards(
        records,
        raw_matrix,
        schema,
        standardizer,
        tokenizer,
        config,
        output_dir,
        data_config.shard_size,
        preparation_signature,
        reporter,
    )
    summary = {
        "total": len(records),
        "train": sum(record["split"] == "train" for record in records),
        "valid": sum(record["split"] == "valid" for record in records),
        **{
            role: sum(record["role"] == role for record in records)
            for role in ROLE_TO_ID
        },
        "augmented": sum(record["is_augmented"] for record in records),
        "excluded_entities": len(excluded_entities),
        "descriptor_dim": schema.selected_dim,
    }
    metadata = {
        "format_version": CORPUS_FORMAT_VERSION,
        "preparation_signature": preparation_signature,
        "rdkit_version": rdBase.rdkitVersion,
        "atom_in_smiles_version": importlib.metadata.version("atomInSmiles"),
        "descriptor_raw_names": list(raw_names),
        "descriptor_names": list(schema.selected_names),
        "descriptor_dim": schema.selected_dim,
        "descriptor_mode": schema.mode,
        "descriptor_token_count": schema.token_count,
        "tokenizer_backend": tokenizer.backend,
        "tokenizer_backend_version": tokenizer.backend_version,
        "tokenizer_budget": config.tokenizer.vocab_size,
        "tokenizer_actual_size": len(tokenizer.tokens),
        "max_smiles_tokens": data_config.max_smiles_tokens,
        "tokenizer_statistics": {
            "min_length": min(lengths),
            "max_length": max(lengths),
            "mean_length": float(np.mean(lengths)),
            "p50_length": float(np.percentile(lengths, 50)),
            "p90_length": float(np.percentile(lengths, 90)),
            "p95_length": float(np.percentile(lengths, 95)),
            "p99_length": float(np.percentile(lengths, 99)),
            "unk_count": unk_count,
        },
        "fingerprint_kind": config.fingerprint.kind,
        "role_source_files": ROLE_SOURCE_FILES,
        "ignored_stage1_files": ["simulation_mol.csv", "solute.csv", "solvent.csv", "IL.csv"],
        "augmentation": data_config.augmentation,
        "augmentation_audit": augmentation_audit,
        "quality_control": quality_control,
        "source_hashes": source_hashes,
        "seed": data_config.seed,
        "valid_fraction": data_config.valid_fraction,
        "shard_hashes": {
            shard: _sha256(output_dir / shard)
            for shard in sorted({entry["shard"] for entry in entries})
        },
        "summary": summary,
    }
    _atomic_json(
        output_dir / "corpus_index.json",
        {"format_version": CORPUS_FORMAT_VERSION, "entries": entries},
    )

    manifest_path = output_dir / "manifest.csv"
    manifest_temporary = manifest_path.with_suffix(".csv.tmp")
    with manifest_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "role",
                "canonical_smiles",
                "split",
                "sources",
                "is_augmented",
                "seed_smiles",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "role": record["role"],
                    "canonical_smiles": record["canonical_smiles"],
                    "split": record["split"],
                    "sources": ";".join(record["sources"]),
                    "is_augmented": int(record["is_augmented"]),
                    "seed_smiles": ";".join(record["seed_smiles"]),
                }
            )
    manifest_temporary.replace(manifest_path)
    metadata["artifact_hashes"] = {
        filename: _sha256(output_dir / filename)
        for filename in (
            "tokenizer.json",
            "descriptor_schema.json",
            "descriptor_scaler.json",
            "corpus_index.json",
            "manifest.csv",
            "excluded_entities.csv",
        )
    }
    _atomic_json(output_dir / "metadata.json", metadata)

    _atomic_json(
        state_path,
        {
            "format_version": 1,
            "preparation_signature": preparation_signature,
            "phase": "complete",
        },
    )
    raw_cache_path.unlink(missing_ok=True)
    training_cache_path.unlink(missing_ok=True)
    return summary


class PreparedCorpusDataset(Dataset):
    def __init__(
        self,
        artifact_path: str | Path,
        split: str = "train",
        shard_cache_size: int = 4,
    ) -> None:
        if split not in {"train", "valid", "all"}:
            raise ValueError("split must be train, valid, or all")
        path = Path(artifact_path)
        if path.is_file():
            if path.name == "corpus.pt":
                raise ValueError(
                    "Legacy corpus.pt artifacts are unsupported; rerun scripts/stage1/prepare.py"
                )
            raise ValueError("PreparedCorpusDataset expects an artifact directory")
        self.artifact_dir = path
        self.metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata.get("format_version") != CORPUS_FORMAT_VERSION:
            raise ValueError("Unsupported corpus artifact format; rerun scripts/stage1/prepare.py")
        artifact_hashes = self.metadata.get("artifact_hashes")
        if not isinstance(artifact_hashes, dict) or not artifact_hashes:
            raise ValueError("Corpus artifact hashes are missing; rerun scripts/stage1/prepare.py")
        for filename, expected_hash in artifact_hashes.items():
            if _sha256(path / filename) != expected_hash:
                raise ValueError(f"Artifact hash mismatch: {filename}")
        if self.metadata.get("rdkit_version") != rdBase.rdkitVersion:
            raise ValueError("RDKit version does not match the prepared corpus")
        expected_tokenizer_version = tokenizer_backend_version(
            self.metadata["tokenizer_backend"]
        )
        if self.metadata.get("tokenizer_backend_version") != expected_tokenizer_version:
            raise ValueError("Tokenizer backend version does not match the corpus")
        current_names = rdkit_descriptor_names()
        self.descriptor_schema = DescriptorSchema.load(
            path / "descriptor_schema.json", expected_raw_names=current_names
        )
        DescriptorStandardizer.load(
            path / "descriptor_scaler.json",
            expected_names=self.descriptor_schema.selected_names,
        )
        payload = json.loads((path / "corpus_index.json").read_text(encoding="utf-8"))
        if payload.get("format_version") != CORPUS_FORMAT_VERSION:
            raise ValueError("Unsupported corpus index format")
        self.entries = [
            entry for entry in payload["entries"] if split == "all" or entry["split"] == split
        ]
        self.role_ids = [int(entry["role_id"]) for entry in self.entries]
        self.shard_cache_size = shard_cache_size
        self._cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._verified_shards: set[str] = set()

    def __len__(self) -> int:
        return len(self.entries)

    def _load_shard(self, relative_path: str) -> list[dict[str, Any]]:
        if relative_path in self._cache:
            samples = self._cache.pop(relative_path)
            self._cache[relative_path] = samples
            return samples
        shard_path = self.artifact_dir / relative_path
        if relative_path not in self._verified_shards:
            expected_hash = self.metadata.get("shard_hashes", {}).get(relative_path)
            if expected_hash is None or _sha256(shard_path) != expected_hash:
                raise ValueError(f"Shard hash mismatch: {relative_path}")
            self._verified_shards.add(relative_path)
        payload = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("format_version") != CORPUS_FORMAT_VERSION:
            raise ValueError(f"Unsupported shard format: {relative_path}")
        samples = payload["samples"]
        self._cache[relative_path] = samples
        while len(self._cache) > self.shard_cache_size:
            self._cache.popitem(last=False)
        return samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        return self._load_shard(entry["shard"])[int(entry["offset"])]


def graph_record_from_sample(sample: dict[str, Any]) -> GraphRecord:
    return GraphRecord(
        atom_categorical=sample["atom_categorical"],
        atom_continuous=sample["atom_continuous"],
        bond_categorical=sample["bond_categorical"],
        bond_index=sample["bond_index"],
    )
