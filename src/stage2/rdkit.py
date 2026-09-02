from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from rdkit import Chem, rdBase

from common.descriptor_preprocessing import FeaturePreprocessor
from common.identity import IDENTITY_CONTRACT_VERSION, require_compatible_identity, semantic_identity
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.progress import ProgressReporter
from common.training import canonical_json_sha256
from stage1.descriptors import calculate_descriptors, rdkit_descriptor_names
from stage1.features import ROLE_TO_ID

from .config import Stage2Config
from .data import (
    STAGE2_ARTIFACT_VERSION,
    STAGE2_PREPARATION_CONTRACT_VERSION,
    STAGE2_RDKIT_ARTIFACT_KIND,
    STAGE2_TENSOR_CONTRACT,
)
from .identity import build_rdkit_stage2_data_identity, metadata_identity
from .registry import Stage2Registry, load_stage2_registry


RDKIT_STAGE2_PREPARATION_CONTRACT_VERSION = 1
RDKIT_DESCRIPTOR_NAMES = rdkit_descriptor_names()


def resolved_rdkit_registry(config: Stage2Config) -> Stage2Registry:
    registry = config.resolved_registry(
        load_stage2_registry(config.data.task_catalog_path)
    )
    config.validate_registry(registry)
    if any(task.target_level != "object" for task in registry.tasks):
        raise ValueError("RDKit Stage 2 supports object-level tasks only")
    return registry


def _schema() -> dict[str, Any]:
    return {
        "descriptor_names": list(RDKIT_DESCRIPTOR_NAMES),
        "rdkit_version": rdBase.rdkitVersion,
    }


def _cache_root(config: Stage2Config) -> Path:
    return config.data.artifacts_dir.parent.parent / "cache" / "rdkit_2d"


def _cache_paths(config: Stage2Config, smiles: str) -> tuple[Path, Path]:
    identity = {
        "family": "rdkit_2d",
        "schema": _schema(),
        "canonical_smiles": smiles,
    }
    root = _cache_root(config) / canonical_json_sha256(identity["schema"])
    key = canonical_json_sha256(identity)
    return root / f"{key}.pt", root / f"{key}.json"


def _descriptor(config: Stage2Config, smiles: str) -> tuple[np.ndarray, bool]:
    path, audit_path = _cache_paths(config, smiles)
    identity = {
        "family": "rdkit_2d",
        "schema": _schema(),
        "canonical_smiles": smiles,
    }
    if path.is_file() and audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("sha256") != sha256_file(path):
            raise ValueError(f"Corrupt RDKit Stage 2 descriptor cache: {smiles}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        values = payload.get("values")
        if payload.get("identity") != identity or not isinstance(values, torch.Tensor):
            raise ValueError(f"RDKit Stage 2 descriptor cache identity mismatch: {smiles}")
        array = values.double().numpy()
        if array.shape != (len(RDKIT_DESCRIPTOR_NAMES),):
            raise ValueError(f"RDKit Stage 2 descriptor cache width mismatch: {smiles}")
        return array, True
    if path.exists() or audit_path.exists():
        raise ValueError(f"Incomplete RDKit Stage 2 descriptor cache: {smiles}")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid canonical SMILES in RDKit Stage 2: {smiles}")
    values = calculate_descriptors(molecule, RDKIT_DESCRIPTOR_NAMES)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        path,
        {"identity": identity, "values": torch.from_numpy(values)},
    )
    atomic_json(audit_path, {"sha256": sha256_file(path)})
    return values, False


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def resolve_rdkit_stage2_materialization(
    config: Stage2Config,
    *,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    from .prepare import _collect_sources

    config.validate()
    registry = resolved_rdkit_registry(config)
    progress = reporter or ProgressReporter()
    collected = _collect_sources(config, registry, progress)
    raw_by_smiles: dict[str, np.ndarray] = {}
    hits = 0
    unique_smiles = sorted({smiles for _, smiles in collected.entity_keys})
    with progress.bar(
        total=len(unique_smiles), desc="Stage 2 RDKit descriptors", unit="entity"
    ) as bar:
        for smiles in unique_smiles:
            raw_by_smiles[smiles], hit = _descriptor(config, smiles)
            hits += int(hit)
            bar.update(1)
    raw = np.stack([raw_by_smiles[smiles] for _, smiles in collected.entity_keys])
    train_occurrences = np.stack(
        [
            raw[entity_id]
            for task in registry.task_ids
            for row in collected.rows[task]["train"].entity_ids
            for entity_id in row
        ]
    )
    preprocessor = FeaturePreprocessor.fit(train_occurrences)
    features = preprocessor.transform(raw)
    representation = config.representation
    assert representation is not None
    if len(RDKIT_DESCRIPTOR_NAMES) != representation.raw_width:
        raise ValueError("RDKit Stage 2 raw descriptor width mismatch")
    descriptor_contract = {
        "contract_version": RDKIT_STAGE2_PREPARATION_CONTRACT_VERSION,
        "kind": representation.kind,
        "descriptor_family": representation.descriptor_family,
        "schema": _schema(),
        "fit_scope": "eight-task-train-component-occurrences",
        "fit_occurrences": int(train_occurrences.shape[0]),
        "preprocessing": preprocessor.to_dict(),
        "clip": [-10.0, 10.0],
        "raw_width": len(RDKIT_DESCRIPTOR_NAMES),
        "retained_width": int(features.shape[1]),
        "encoder": {
            "hidden_dim": representation.hidden_dim,
            "output_dim": representation.output_dim,
            "activation": representation.activation,
            "dropout": representation.dropout,
            "normalization": representation.normalization,
        },
        "raw_content_sha256": canonical_json_sha256(
            [
                {
                    "canonical_smiles": smiles,
                    "values_sha256": _array_hash(raw_by_smiles[smiles]),
                }
                for smiles in unique_smiles
            ]
        ),
    }
    data_identity = build_rdkit_stage2_data_identity(
        config, registry, descriptor_contract
    )
    return {
        "registry": registry,
        "collected": collected,
        "features": torch.from_numpy(features),
        "descriptor_contract": descriptor_contract,
        "data_identity": data_identity,
        "cache": {"hits": hits, "misses": len(unique_smiles) - hits},
    }


def _write_empty_excluded_entities(path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            (
                "role",
                "canonical_smiles",
                "exclusion_reasons",
                "unsupported_bond_types",
                "ipc",
                "token_count",
                "max_smiles_tokens",
                "detail",
            )
        )
    temporary.replace(path)


def prepare_rdkit_stage2(
    config: Stage2Config,
    *,
    reporter: ProgressReporter | None = None,
    materialization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from .prepare import (
        _write_duplicate_audit,
        _write_missing_target_audit,
        _write_task_tensors,
    )

    resolved = dict(
        materialization
        or resolve_rdkit_stage2_materialization(config, reporter=reporter)
    )
    root = config.data.artifacts_dir
    metadata_path = root / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("kind") != STAGE2_RDKIT_ARTIFACT_KIND
            or metadata.get("format_version") != STAGE2_ARTIFACT_VERSION
        ):
            raise ValueError("Existing Stage 2 artifact is not the RDKit ablation")
        require_compatible_identity(
            resolved["data_identity"],
            metadata_identity(metadata, "data", context="RDKit Stage 2 artifact"),
            context="RDKit Stage 2 prepared artifact",
        )
        for relative, expected in metadata.get("artifact_hashes", {}).items():
            if sha256_file(root / relative) != expected:
                raise ValueError(f"Stage 2 RDKit artifact hash mismatch: {relative}")
        return {**metadata["summary"], "cache": resolved["cache"], "reused": True}
    root.mkdir(parents=True, exist_ok=True)
    registry: Stage2Registry = resolved["registry"]
    collected = resolved["collected"]
    entries = [
        {
            "entity_id": index,
            "role": role,
            "role_id": int(ROLE_TO_ID[role]),
            "canonical_smiles": smiles,
        }
        for index, (role, smiles) in enumerate(collected.entity_keys)
    ]
    atomic_json(
        root / "entity_index.json",
        {
            "format_version": STAGE2_ARTIFACT_VERSION,
            "kind": STAGE2_RDKIT_ARTIFACT_KIND,
            "entries": entries,
        },
    )
    atomic_torch_save(
        root / "entity_features.pt",
        {
            "format_version": STAGE2_ARTIFACT_VERSION,
            "kind": STAGE2_RDKIT_ARTIFACT_KIND,
            "features": resolved["features"],
        },
    )
    _write_duplicate_audit(root / "duplicate_conditions.csv", collected.duplicate_rows)
    _write_missing_target_audit(root / "missing_targets.csv", collected.missing_target_rows)
    _write_empty_excluded_entities(root / "excluded_entities.csv")
    row_counts, excluded_rows, scalers = _write_task_tensors(
        config,
        registry,
        collected,
        range(len(entries)),
        reporter or ProgressReporter(),
        artifact_kind=STAGE2_RDKIT_ARTIFACT_KIND,
    )
    atomic_json(root / "scalers.json", scalers)
    artifact_files = [
        "entity_index.json",
        "entity_features.pt",
        "scalers.json",
        "excluded_entities.csv",
        "excluded_rows.csv",
        "duplicate_conditions.csv",
        "missing_targets.csv",
        *[
            f"tasks/{task}/{split}.pt"
            for task in registry.task_ids
            for split in ("train", "valid")
        ],
    ]
    artifact_hashes = {
        relative: sha256_file(root / relative) for relative in artifact_files
    }
    entity_identity = semantic_identity(
        "stage2.rdkit-entity-artifact",
        {
            "data_identity": resolved["data_identity"]["hash"],
            "entries": entries,
            "features_sha256": artifact_hashes["entity_features.pt"],
        },
    )
    source_hashes = {
        str(path): sha256_file(path)
        for spec in registry.tasks
        for path in (
            spec.dataset.split_path(config.data.data_root, "train"),
            spec.dataset.split_path(config.data.data_root, "valid"),
        )
    }
    source_hashes[str(config.data.task_catalog_path)] = sha256_file(
        config.data.task_catalog_path
    )
    summary = {
        "entities_selected": len(entries),
        "entities_retained": len(entries),
        "entities_excluded": 0,
        "rows": row_counts,
        "rows_excluded": excluded_rows,
        "duplicate_conditions": len(collected.duplicate_rows),
        "task_count": len(registry.tasks),
    }
    metadata = {
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "format_version": STAGE2_ARTIFACT_VERSION,
        "kind": STAGE2_RDKIT_ARTIFACT_KIND,
        "preparation_contract_version": STAGE2_PREPARATION_CONTRACT_VERSION,
        "rdkit_preparation_contract_version": RDKIT_STAGE2_PREPARATION_CONTRACT_VERSION,
        "data_signature": resolved["data_identity"]["hash"],
        "registry": registry.snapshot(),
        "registry_hash": registry.registry_hash,
        "catalog_sha256": registry.catalog_sha256,
        "source_hashes": source_hashes,
        "descriptor_contract": resolved["descriptor_contract"],
        "tensor_contract": STAGE2_TENSOR_CONTRACT,
        "scalers": scalers,
        "summary": summary,
        "artifact_hashes": artifact_hashes,
        "semantic": {
            "identities": {
                "data": resolved["data_identity"],
                "entity": entity_identity,
            }
        },
        "integrity": {
            "files": {
                relative: {
                    "sha256": digest,
                    "size": (root / relative).stat().st_size,
                }
                for relative, digest in artifact_hashes.items()
            }
        },
        "provenance": {
            "representation": "rdkit_2d_mlp",
            "rdkit_version": rdBase.rdkitVersion,
        },
    }
    atomic_json(metadata_path, metadata)
    return {**summary, "cache": resolved["cache"], "reused": False}


__all__ = [
    "prepare_rdkit_stage2",
    "resolve_rdkit_stage2_materialization",
    "resolved_rdkit_registry",
]
