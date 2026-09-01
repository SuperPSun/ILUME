from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from rdkit import Chem, rdBase

from common.descriptor_preprocessing import FeaturePreprocessor
from common.identity import IDENTITY_CONTRACT_VERSION, require_compatible_identity
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.progress import ProgressReporter
from common.training import canonical_json_sha256
from stage1.descriptors import calculate_descriptors, rdkit_descriptor_names

from .config import Stage3Config
from .data import (
    STAGE3_ARTIFACT_VERSION,
    ObjectKey,
    ResolvedTaskSpec,
    build_task_payload,
    fit_normalization,
    iter_rows,
    object_key_from_row,
    sanitize_task,
    source_hashes,
)
from .identity import build_stage3_rdkit_prepared_identity, metadata_identity


RDKIT_STAGE3_ARTIFACT_KIND = "ilume_stage3_rdkit_sparse_data"
RDKIT_REPRESENTATION_CONTRACT_VERSION = 1
RDKIT_PREPROCESSING_CONTRACT = "joint-training-rows-mlp-v1"
RDKIT_DESCRIPTOR_NAMES = rdkit_descriptor_names()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _descriptor_schema() -> dict[str, Any]:
    return {
        "descriptor_names": list(RDKIT_DESCRIPTOR_NAMES),
        "rdkit_version": rdBase.rdkitVersion,
    }


def _cache_identity(smiles: str) -> dict[str, Any]:
    return {
        "contract_version": RDKIT_REPRESENTATION_CONTRACT_VERSION,
        "family": "rdkit_2d",
        "schema": _descriptor_schema(),
        "canonical_smiles": smiles,
    }


def _cache_paths(cache_dir: Path, smiles: str) -> tuple[Path, Path]:
    identity = _cache_identity(smiles)
    schema_hash = canonical_json_sha256(identity["schema"])
    key = canonical_json_sha256(identity)
    root = cache_dir / f"rdkit-2d-{schema_hash}"
    return root / f"{key}.pt", root / f"{key}.json"


def _load_cached_descriptor(cache_dir: Path, smiles: str) -> np.ndarray | None:
    path, audit_path = _cache_paths(cache_dir, smiles)
    if not path.exists() and not audit_path.exists():
        return None
    if not path.is_file() or not audit_path.is_file():
        raise ValueError(f"Incomplete RDKit descriptor cache entry: {smiles}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("sha256") != sha256_file(path):
        raise ValueError(f"Corrupt RDKit descriptor cache entry: {smiles}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("identity") != _cache_identity(smiles):
        raise ValueError(f"RDKit descriptor cache identity mismatch: {smiles}")
    values = payload.get("values")
    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise ValueError(f"Malformed RDKit descriptor cache entry: {smiles}")
    array = values.double().numpy()
    if len(array) != len(RDKIT_DESCRIPTOR_NAMES):
        raise ValueError(f"RDKit descriptor cache width mismatch: {smiles}")
    return array


def _save_cached_descriptor(
    cache_dir: Path, smiles: str, values: np.ndarray
) -> None:
    path, audit_path = _cache_paths(cache_dir, smiles)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        path,
        {
            "identity": _cache_identity(smiles),
            "values": torch.from_numpy(np.asarray(values, dtype=np.float64)),
        },
    )
    atomic_json(audit_path, {"sha256": sha256_file(path)})


def materialize_raw_descriptors(
    config: Stage3Config,
    objects: Sequence[ObjectKey],
    *,
    reporter: ProgressReporter | None = None,
) -> tuple[list[np.ndarray], dict[str, int]]:
    smiles_values = sorted({smiles for key in objects for _, smiles in key.slots})
    by_smiles: dict[str, np.ndarray] = {}
    hits = 0
    progress = (reporter or ProgressReporter()).bar(
        total=len(smiles_values), desc="Stage3 RDKit descriptor cache", unit="object"
    )
    try:
        for smiles in smiles_values:
            values = _load_cached_descriptor(config.preparation.cache_dir, smiles)
            if values is None:
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    raise ValueError(f"Invalid canonical SMILES in RDKit prepare: {smiles}")
                values = calculate_descriptors(molecule, RDKIT_DESCRIPTOR_NAMES)
                _save_cached_descriptor(config.preparation.cache_dir, smiles, values)
            else:
                hits += 1
            by_smiles[smiles] = values
            progress.update(1)
    finally:
        progress.close()
    raw: list[np.ndarray] = []
    for key in objects:
        if key.topology == "il":
            if tuple(role for role, _ in key.slots) != ("cation", "anion"):
                raise ValueError("RDKit IL objects require ordered cation/anion slots")
            raw.append(np.concatenate([by_smiles[smiles] for _, smiles in key.slots]))
        elif key.topology == "molecule" and len(key.slots) == 1:
            raw.append(by_smiles[key.slots[0][1]])
        else:
            raise ValueError(f"Unsupported RDKit Stage 3 object topology: {key.topology}")
    return raw, {"hits": hits, "misses": len(smiles_values) - hits}


def _fit_fold_preprocessors(
    config: Stage3Config,
    registry: Mapping[str, ResolvedTaskSpec],
    held_out_fold: int,
    object_ids: Mapping[ObjectKey, int],
    raw: Sequence[np.ndarray],
) -> tuple[FeaturePreprocessor, FeaturePreprocessor]:
    train_folds = tuple(fold for fold in range(1, 6) if fold != held_out_fold)
    il_rows: list[np.ndarray] = []
    single_rows: list[np.ndarray] = []
    for task_id, spec in registry.items():
        for _, row_number, row in iter_rows(config, spec, train_folds):
            primary = object_key_from_row(
                task_id, row_number, row, spec.primary_slots
            )
            destination = il_rows if primary.topology == "il" else single_rows
            destination.append(raw[object_ids[primary]])
            if spec.partner_slots:
                partner = object_key_from_row(
                    task_id, row_number, row, spec.partner_slots
                )
                if partner.topology != "molecule":
                    raise ValueError("RDKit Stage 3 partner must be a single object")
                single_rows.append(raw[object_ids[partner]])
    if not il_rows or not single_rows:
        raise ValueError("RDKit Stage 3 preprocessing requires IL and single-object rows")
    return (
        FeaturePreprocessor.fit(np.stack(il_rows)),
        FeaturePreprocessor.fit(np.stack(single_rows)),
    )


def resolve_rdkit_materialization(
    config: Stage3Config,
    registry: Mapping[str, ResolvedTaskSpec],
    objects: Sequence[ObjectKey],
    *,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    raw, cache = materialize_raw_descriptors(config, objects, reporter=reporter)
    object_ids = {key: index for index, key in enumerate(objects)}
    normalization: dict[str, Any] = {}
    preprocessors: dict[str, Any] = {}
    fold_features: dict[int, dict[str, Any]] = {}
    il_ids = [index for index, key in enumerate(objects) if key.topology == "il"]
    single_ids = [
        index for index, key in enumerate(objects) if key.topology == "molecule"
    ]
    for fold in range(1, 6):
        fold_normalization = fit_normalization(config, registry, fold)
        normalization[f"fold{fold}"] = fold_normalization
        il_preprocessor, single_preprocessor = _fit_fold_preprocessors(
            config, registry, fold, object_ids, raw
        )
        preprocessor_payload = {
            "il": il_preprocessor.to_dict(),
            "single": single_preprocessor.to_dict(),
        }
        preprocessors[f"fold{fold}"] = preprocessor_payload
        il_features = il_preprocessor.transform(
            np.stack([raw[index] for index in il_ids])
        )
        single_features = single_preprocessor.transform(
            np.stack([raw[index] for index in single_ids])
        )
        fold_features[fold] = {
            "il_object_ids": torch.tensor(il_ids, dtype=torch.long),
            "il_features": torch.from_numpy(il_features),
            "single_object_ids": torch.tensor(single_ids, dtype=torch.long),
            "single_features": torch.from_numpy(single_features),
            "preprocessing": preprocessor_payload,
        }
    raw_content = canonical_json_sha256(
        [
            {"object": key.identity, "values_sha256": _array_sha256(raw[index])}
            for index, key in enumerate(objects)
        ]
    )
    descriptor_contract = {
        "contract_version": RDKIT_REPRESENTATION_CONTRACT_VERSION,
        "kind": "rdkit_2d_adapter",
        "descriptor_family": "rdkit_2d",
        "adapter": "linear_layernorm",
        "output_dim": 512,
        "fit_scope": "joint_training_rows",
        "preprocessing_contract": RDKIT_PREPROCESSING_CONTRACT,
        "clip": [-10.0, 10.0],
        "schema": _descriptor_schema(),
        "raw_descriptor_content_sha256": raw_content,
        "fold_preprocessing": preprocessors,
        "fold_input_dims": {
            f"fold{fold}": {
                "il": int(fold_features[fold]["il_features"].shape[1]),
                "single": int(fold_features[fold]["single_features"].shape[1]),
            }
            for fold in range(1, 6)
        },
    }
    identity = build_stage3_rdkit_prepared_identity(
        config, registry, objects, normalization, descriptor_contract
    )
    return {
        "raw": raw,
        "cache": cache,
        "normalization": normalization,
        "fold_features": fold_features,
        "descriptor_contract": descriptor_contract,
        "prepared_identity": identity,
    }


def prepare_rdkit_stage3(
    config: Stage3Config,
    registry: Mapping[str, ResolvedTaskSpec],
    objects: Sequence[ObjectKey],
    *,
    reporter: ProgressReporter | None = None,
    materialization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    destination = config.data.artifacts_dir
    if destination.exists():
        raise FileExistsError(f"Stage 3 artifact output already exists: {destination}")
    resolved = dict(
        materialization
        or resolve_rdkit_materialization(config, registry, objects, reporter=reporter)
    )
    object_ids = {key: index for index, key in enumerate(objects)}
    source_digests = source_hashes(config, registry)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="stage3-rdkit-artifacts-", dir=destination.parent))
    try:
        atomic_json(staging / "objects.json", [key.to_dict() for key in objects])
        registry_payload = {
            task_id: spec.to_dict() for task_id, spec in registry.items()
        }
        atomic_json(staging / "registry.json", registry_payload)
        atomic_json(staging / "normalization.json", resolved["normalization"])
        atomic_json(
            staging / "descriptor_contract.json", resolved["descriptor_contract"]
        )
        counts: dict[str, dict[str, int]] = {}
        for fold in range(1, 6):
            fold_dir = staging / "folds" / f"fold{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            representation = {
                "format_version": STAGE3_ARTIFACT_VERSION,
                "kind": RDKIT_STAGE3_ARTIFACT_KIND,
                "fold": fold,
                **resolved["fold_features"][fold],
            }
            atomic_torch_save(fold_dir / "representation.pt", representation)
            for task_id, spec in registry.items():
                counts.setdefault(task_id, {})
                for split in ("train", "valid", "test"):
                    payload = build_task_payload(
                        config,
                        spec,
                        fold,
                        split,
                        object_ids,
                        resolved["normalization"][f"fold{fold}"],
                    )
                    payload["kind"] = RDKIT_STAGE3_ARTIFACT_KIND
                    relative = Path("folds") / f"fold{fold}" / (
                        f"{sanitize_task(task_id)}_{split}.pt"
                    )
                    atomic_torch_save(staging / relative, payload)
                    counts[task_id][f"fold{fold}_{split}"] = int(
                        payload["targets"].shape[0]
                    )
        artifact_hashes = {
            path.relative_to(staging).as_posix(): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        metadata = {
            "identity_contract_version": IDENTITY_CONTRACT_VERSION,
            "format_version": STAGE3_ARTIFACT_VERSION,
            "kind": RDKIT_STAGE3_ARTIFACT_KIND,
            "representation_contract_version": RDKIT_REPRESENTATION_CONTRACT_VERSION,
            "registry_hash": canonical_json_sha256(registry_payload),
            "catalog_sha256": source_digests[str(config.data.task_catalog)],
            "source_hashes": dict(source_digests),
            "object_count": len(objects),
            "output_dim": 512,
            "counts": counts,
            "descriptor_contract": resolved["descriptor_contract"],
            "artifact_hashes": artifact_hashes,
            "locator": {"files": {name: name for name in artifact_hashes}},
            "semantic": {
                "identities": {"prepared": resolved["prepared_identity"]}
            },
            "integrity": {
                "files": {
                    name: {
                        "sha256": digest,
                        "size": (staging / name).stat().st_size,
                    }
                    for name, digest in artifact_hashes.items()
                }
            },
            "provenance": {
                "representation": "rdkit_2d_adapter",
                "rdkit_version": rdBase.rdkitVersion,
            },
        }
        atomic_json(staging / "metadata.json", metadata)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "artifact_kind": RDKIT_STAGE3_ARTIFACT_KIND,
        "format_version": STAGE3_ARTIFACT_VERSION,
        "task_count": len(registry),
        "object_count": len(objects),
        "output_dim": 512,
        "cache": resolved["cache"],
    }


def load_prepared_rdkit_stage3(
    config: Stage3Config,
    metadata: Mapping[str, Any],
    registry: Mapping[str, ResolvedTaskSpec],
) -> dict[str, Any]:
    root = config.data.artifacts_dir
    if metadata.get("identity_contract_version") != IDENTITY_CONTRACT_VERSION:
        raise ValueError("RDKit Stage 3 artifact predates identity contract v1")
    if metadata.get("source_hashes") != source_hashes(config, registry):
        raise ValueError("RDKit Stage 3 artifact source hash mismatch")
    registry_payload = {
        task_id: spec.to_dict() for task_id, spec in registry.items()
    }
    if metadata.get("registry_hash") != canonical_json_sha256(registry_payload):
        raise ValueError("RDKit Stage 3 artifact registry mismatch")
    descriptor_contract = metadata.get("descriptor_contract")
    if not isinstance(descriptor_contract, dict):
        raise ValueError("RDKit Stage 3 artifact lacks descriptor contract")
    if descriptor_contract.get("schema") != _descriptor_schema():
        raise ValueError("RDKit Stage 3 descriptor schema or RDKit version changed")
    for relative, digest in metadata.get("artifact_hashes", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"RDKit Stage 3 artifact hash mismatch: {relative}")
    objects_payload = json.loads((root / "objects.json").read_text(encoding="utf-8"))
    objects = tuple(
        ObjectKey(
            topology=item["topology"],
            slots=tuple(tuple(slot) for slot in item["slots"]),
        )
        for item in objects_payload
    )
    normalization = json.loads(
        (root / "normalization.json").read_text(encoding="utf-8")
    )
    expected_identity = build_stage3_rdkit_prepared_identity(
        config, registry, objects, normalization, descriptor_contract
    )
    require_compatible_identity(
        expected_identity,
        metadata_identity(metadata, "prepared", context="RDKit Stage 3 artifact"),
        context="RDKit Stage 3 prepared artifact",
    )
    return {
        "metadata": dict(metadata),
        "registry": dict(registry),
        "normalization": normalization,
        "objects": {"objects": [key.to_dict() for key in objects]},
    }


__all__ = [
    "RDKIT_STAGE3_ARTIFACT_KIND",
    "load_prepared_rdkit_stage3",
    "materialize_raw_descriptors",
    "prepare_rdkit_stage3",
    "resolve_rdkit_materialization",
]
