from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from rdkit import rdBase

from common.io import atomic_json, atomic_torch_save, sha256_file
from common.progress import ProgressReporter
from common.training import canonical_json_sha256, resolve_device
from stage2 import FrozenObjectSpec, load_frozen_object_encoder
from .config import Stage3Config
from .data import (
    OBJECT_ENCODING_CONTRACT_VERSION,
    STAGE3_ARTIFACT_KIND,
    STAGE3_ARTIFACT_VERSION,
    ObjectKey,
    ResolvedTaskSpec,
    build_task_payload,
    collect_object_keys,
    fit_normalization,
    resolve_task_registry,
    sanitize_task,
    source_hashes,
)


def _cache_identity(stage2_sha: str, key: ObjectKey) -> dict[str, Any]:
    return {
        "encoding_contract_version": OBJECT_ENCODING_CONTRACT_VERSION,
        "stage2_checkpoint_sha256": stage2_sha,
        "rdkit_version": rdBase.rdkitVersion,
        "object": key.to_dict(),
    }


def _cache_paths(cache_dir: Path, stage2_sha: str, key: ObjectKey) -> tuple[Path, Path]:
    root = cache_dir / stage2_sha
    return root / f"{key.identity}.pt", root / f"{key.identity}.json"


def _load_cache_entry(
    cache_dir: Path, stage2_sha: str, key: ObjectKey
) -> torch.Tensor | None:
    path, audit_path = _cache_paths(cache_dir, stage2_sha, key)
    if not path.exists() and not audit_path.exists():
        return None
    if not path.is_file() or not audit_path.is_file():
        raise ValueError(f"Incomplete Stage 3 object cache entry: {key.identity}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("sha256") != sha256_file(path):
        raise ValueError(f"Corrupt Stage 3 object cache entry: {key.identity}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("identity") != _cache_identity(stage2_sha, key):
        raise ValueError(f"Stage 3 object cache identity mismatch: {key.identity}")
    embedding = payload.get("embedding")
    if (
        not isinstance(embedding, torch.Tensor)
        or embedding.ndim != 1
        or embedding.dtype != torch.float32
        or not torch.isfinite(embedding).all()
    ):
        raise ValueError(f"Invalid Stage 3 cached embedding: {key.identity}")
    return embedding


def _save_cache_entry(
    cache_dir: Path,
    stage2_sha: str,
    key: ObjectKey,
    embedding: torch.Tensor,
) -> None:
    path, audit_path = _cache_paths(cache_dir, stage2_sha, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        path,
        {
            "identity": _cache_identity(stage2_sha, key),
            "embedding": embedding.detach().cpu().float().contiguous(),
        },
    )
    atomic_json(audit_path, {"sha256": sha256_file(path)})


def materialize_object_embeddings(
    config: Stage3Config,
    object_keys: Sequence[ObjectKey],
    *,
    reporter: ProgressReporter | None = None,
) -> tuple[torch.Tensor, str, dict[str, int]]:
    checkpoint = config.initialization.stage2_checkpoint
    stage2_sha = sha256_file(checkpoint)
    embeddings: list[torch.Tensor | None] = [None] * len(object_keys)
    misses: dict[str, list[int]] = {}
    cache_hits = 0
    for index, key in enumerate(object_keys):
        cached = _load_cache_entry(config.preparation.cache_dir, stage2_sha, key)
        if cached is None:
            misses.setdefault(key.topology, []).append(index)
        else:
            embeddings[index] = cached
            cache_hits += 1
    if misses:
        device = resolve_device(config.training.device)
        encoder = load_frozen_object_encoder(checkpoint, device=device)
        if encoder.checkpoint_hash != stage2_sha:
            raise ValueError("Stage 2 checkpoint changed during Stage 3 prepare")
        bar = (reporter or ProgressReporter()).bar(
            total=sum(len(indices) for indices in misses.values()),
            desc="Stage3 object cache",
            unit="object",
        )
        try:
            for topology in sorted(misses):
                indices = misses[topology]
                size = config.preparation.encoding_batch_size
                for start in range(0, len(indices), size):
                    batch_indices = indices[start : start + size]
                    specs = [
                        FrozenObjectSpec(
                            topology=object_keys[index].topology,
                            slots=object_keys[index].slots,
                        )
                        for index in batch_indices
                    ]
                    values = encoder.encode(specs)
                    for index, value in zip(batch_indices, values, strict=True):
                        embeddings[index] = value
                        _save_cache_entry(
                            config.preparation.cache_dir,
                            stage2_sha,
                            object_keys[index],
                            value,
                        )
                    bar.update(len(batch_indices))
        finally:
            bar.close()
    if any(value is None for value in embeddings):
        raise AssertionError("Stage 3 object cache materialization is incomplete")
    matrix = torch.stack([value for value in embeddings if value is not None])
    if matrix.ndim != 2:
        raise ValueError("Stage 3 object embedding matrix must be rank two")
    return matrix.float(), stage2_sha, {
        "hits": cache_hits,
        "misses": len(object_keys) - cache_hits,
    }


def _stage_artifacts(
    staging: Path,
    config: Stage3Config,
    registry: Mapping[str, ResolvedTaskSpec],
    objects: Sequence[ObjectKey],
    embeddings: torch.Tensor,
    stage2_sha: str,
    source_digests: Mapping[str, str],
) -> dict[str, Any]:
    object_ids = {key: index for index, key in enumerate(objects)}
    atomic_torch_save(
        staging / "object_embeddings.pt",
        {
            "format_version": STAGE3_ARTIFACT_VERSION,
            "kind": STAGE3_ARTIFACT_KIND,
            "encoding_contract_version": OBJECT_ENCODING_CONTRACT_VERSION,
            "stage2_checkpoint_sha256": stage2_sha,
            "objects": [key.to_dict() for key in objects],
            "embeddings": embeddings,
        },
    )
    registry_payload = {
        task_id: spec.to_dict() for task_id, spec in registry.items()
    }
    atomic_json(staging / "registry.json", registry_payload)
    all_normalization: dict[str, Any] = {}
    counts: dict[str, dict[str, int]] = {}
    for fold in range(1, 6):
        normalization = fit_normalization(config, registry, fold)
        all_normalization[f"fold{fold}"] = normalization
        fold_dir = staging / "folds" / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for task_id, spec in registry.items():
            counts.setdefault(task_id, {})
            for split in ("train", "valid", "test"):
                payload = build_task_payload(
                    config, spec, fold, split, object_ids, normalization
                )
                relative = Path("folds") / f"fold{fold}" / (
                    f"{sanitize_task(task_id)}_{split}.pt"
                )
                atomic_torch_save(staging / relative, payload)
                counts[task_id][f"fold{fold}_{split}"] = int(
                    payload["targets"].shape[0]
                )
    atomic_json(staging / "normalization.json", all_normalization)
    artifact_hashes = {
        path.relative_to(staging).as_posix(): sha256_file(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file()
    }
    metadata = {
        "format_version": STAGE3_ARTIFACT_VERSION,
        "kind": STAGE3_ARTIFACT_KIND,
        "encoding_contract_version": OBJECT_ENCODING_CONTRACT_VERSION,
        "stage2_checkpoint_sha256": stage2_sha,
        "registry_hash": canonical_json_sha256(registry_payload),
        "catalog_sha256": source_digests[str(config.data.task_catalog)],
        "source_hashes": dict(source_digests),
        "rdkit_version": rdBase.rdkitVersion,
        "object_count": len(objects),
        "embedding_dim": int(embeddings.shape[1]),
        "counts": counts,
        "artifact_hashes": artifact_hashes,
    }
    atomic_json(staging / "metadata.json", metadata)
    return metadata


def prepare_stage3(
    config: Stage3Config,
    *,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    config.validate()
    destination = config.data.artifacts_dir
    if destination.exists():
        raise FileExistsError(f"Stage 3 artifact output already exists: {destination}")
    registry = resolve_task_registry(config)
    source_digests = source_hashes(config, registry)
    for fold in range(1, 6):
        fit_normalization(config, registry, fold)
    objects = collect_object_keys(config, registry)
    embeddings, stage2_sha, cache = materialize_object_embeddings(
        config, objects, reporter=reporter
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="stage3-artifacts-", dir=destination.parent))
    try:
        metadata = _stage_artifacts(
            staging, config, registry, objects, embeddings, stage2_sha, source_digests
        )
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "artifact_kind": STAGE3_ARTIFACT_KIND,
        "format_version": STAGE3_ARTIFACT_VERSION,
        "task_count": len(registry),
        "object_count": len(objects),
        "embedding_dim": metadata["embedding_dim"],
        "stage2_checkpoint_sha256": stage2_sha,
        "cache": cache,
    }


def load_prepared_stage3(config: Stage3Config) -> dict[str, Any]:
    root = config.data.artifacts_dir
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("format_version") != STAGE3_ARTIFACT_VERSION
        or metadata.get("kind") != STAGE3_ARTIFACT_KIND
    ):
        raise ValueError("Unsupported Stage 3 sparse artifact")
    if metadata.get("rdkit_version") != rdBase.rdkitVersion:
        raise ValueError("Stage 3 artifact RDKit version mismatch")
    registry = resolve_task_registry(config)
    registry_payload = {
        task_id: spec.to_dict() for task_id, spec in registry.items()
    }
    if metadata.get("registry_hash") != canonical_json_sha256(registry_payload):
        raise ValueError("Stage 3 artifact registry mismatch")
    if metadata.get("source_hashes") != source_hashes(config, registry):
        raise ValueError("Stage 3 artifact source hash mismatch")
    expected_stage2_sha = sha256_file(config.initialization.stage2_checkpoint)
    if metadata.get("stage2_checkpoint_sha256") != expected_stage2_sha:
        raise ValueError("Stage 3 artifact Stage 2 checkpoint mismatch")
    for relative, digest in metadata.get("artifact_hashes", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Stage 3 artifact hash mismatch: {relative}")
    objects = torch.load(
        root / "object_embeddings.pt", map_location="cpu", weights_only=True
    )
    if (
        objects.get("kind") != STAGE3_ARTIFACT_KIND
        or objects.get("stage2_checkpoint_sha256") != expected_stage2_sha
    ):
        raise ValueError("Stage 3 object embedding identity mismatch")
    normalization = json.loads(
        (root / "normalization.json").read_text(encoding="utf-8")
    )
    return {
        "metadata": metadata,
        "registry": registry,
        "normalization": normalization,
        "objects": objects,
    }


__all__ = [
    "load_prepared_stage3",
    "materialize_object_embeddings",
    "prepare_stage3",
]
