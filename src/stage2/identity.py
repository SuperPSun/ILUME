from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Mapping

from common.identity import semantic_identity
from common.io import sha256_file
from stage1.identity import (
    build_stage1_encoder_identity,
    metadata_identity as stage1_metadata_identity,
)
from .config import Stage2Config
from .data import STAGE2_PREPARATION_CONTRACT_VERSION, STAGE2_TENSOR_CONTRACT
from .registry import Stage2Registry
from .atom_targets import load_structure_manifest, verify_structure


STAGE2_TARGET_MATERIALIZATION_CONTRACT_VERSION = 1
STAGE2_PARTIAL_CHARGE_MAPPING_CONTRACT_VERSION = 1
STAGE2_TRAINING_CONTRACT_VERSION = 1
STAGE2_ENCODER_IDENTITY_CONTRACT_VERSION = 1


def metadata_identity(
    metadata: Mapping[str, Any], name: str, *, context: str
) -> Mapping[str, Any]:
    try:
        identity = metadata["semantic"]["identities"][name]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"{context} predates identity contract v1; regenerate it"
        ) from error
    if not isinstance(identity, Mapping):
        raise ValueError(f"Malformed {context} {name} identity")
    return identity


def _source_content(
    config: Stage2Config, registry: Stage2Registry
) -> dict[str, dict[str, Any]]:
    def rows(path: Path) -> int:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return sum(1 for _ in csv.reader(handle)) - 1

    result: dict[str, dict[str, Any]] = {
        "task_catalog": {
            "sha256": sha256_file(config.data.task_catalog_path),
            "size": config.data.task_catalog_path.stat().st_size,
            "rows": rows(config.data.task_catalog_path),
        }
    }
    for spec in registry.tasks:
        for split in ("train", "valid"):
            path = spec.dataset.split_path(config.data.data_root, split)
            result[f"{spec.task_id}:{split}"] = {
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
                "rows": rows(path),
            }
        manifest = spec.dataset.resource_manifest_path(config.data.data_root)
        if manifest is not None:
            result[f"{spec.task_id}:resource_manifest"] = {
                "sha256": sha256_file(manifest),
                "size": manifest.stat().st_size,
                "rows": rows(manifest),
            }
            for mol_id, entry in sorted(load_structure_manifest(manifest).items()):
                verify_structure(entry)
                result[f"{spec.task_id}:structure:{mol_id}"] = {
                    "sha256": entry.sha256,
                    "size": entry.size_bytes,
                }
    return result


def build_stage2_data_identity(
    config: Stage2Config,
    registry: Stage2Registry,
    stage1_feature_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return semantic_identity(
        "stage2.prepared-data",
        {
            "preparation_contract_version": STAGE2_PREPARATION_CONTRACT_VERSION,
            "source_content": _source_content(config, registry),
            "stage1_feature_identity": stage1_feature_identity["hash"],
            "registry_hash": registry.registry_hash,
            "registry": registry.snapshot(),
            "tensor_contract": STAGE2_TENSOR_CONTRACT,
            "normalization_contract": "task-train-population-v1",
            "target_materialization_contract": {
                "version": STAGE2_TARGET_MATERIALIZATION_CONTRACT_VERSION,
                "modes": {
                    task: config.data.target_materialization_modes.get(
                        task, "require_complete"
                    )
                    for task in registry.task_ids
                },
            },
            "partial_charge_mapping_contract_version": (
                STAGE2_PARTIAL_CHARGE_MAPPING_CONTRACT_VERSION
            ),
        },
    )


def build_stage2_training_identity(
    config: Stage2Config,
    *,
    data_identity: Mapping[str, Any],
    teacher_identity: Mapping[str, Any],
    stage1_encoder_identity: Mapping[str, Any],
    registry: Stage2Registry,
    model_contract: Mapping[str, Any],
    normalized_task_weights: Mapping[str, float],
    math_contract: Mapping[str, Any],
    optimizer_implementation: str,
) -> dict[str, Any]:
    raw = config.to_dict()
    training = dict(raw["training"])
    for name in (
        "packing_workers",
        "packing_prefetch_batches",
        "cuda_prefetch_batches",
        "log_every_batches",
        "device",
    ):
        training.pop(name, None)
    return semantic_identity(
        "stage2.training",
        {
            "contract_version": STAGE2_TRAINING_CONTRACT_VERSION,
            "data_identity": data_identity["hash"],
            "teacher_identity": teacher_identity["hash"],
            "stage1_encoder_identity": stage1_encoder_identity["hash"],
            "registry_hash": registry.registry_hash,
            "model_contract": dict(model_contract),
            "loss": {
                **raw["loss"],
                "normalized_task_weights": dict(normalized_task_weights),
            },
            "training": training,
            "optimizer": {
                "name": "AdamW",
                "implementation": optimizer_implementation,
            },
            "scheduler": "joint-cosine-plus-taskwise-refinement-v1",
            "math_contract": dict(math_contract),
            "seed": config.data.seed,
        },
    )


def stage1_identities_from_loaded(
    config: Stage2Config, loaded: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = json.loads(
        (config.data.pretrain_artifacts_dir / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    feature = dict(
        stage1_metadata_identity(
            metadata, "feature", context="Stage 1 feature artifact"
        )
    )
    encoder = build_stage1_encoder_identity(
        model=loaded.model,
        config=loaded.config,
        feature_identity=feature,
    )
    return feature, encoder


def build_stage2_encoder_identity(
    *,
    stage1_feature_identity: Mapping[str, Any],
    stage1_encoding_contract: Mapping[str, Any],
    stage1_state_hash: str,
    object_encoder_contract: Mapping[str, Any],
    object_encoder_state_hash: str,
    role_to_id: Mapping[str, int],
) -> dict[str, Any]:
    return semantic_identity(
        "stage2.encoder",
        {
            "contract_version": STAGE2_ENCODER_IDENTITY_CONTRACT_VERSION,
            "object_encoding_api": "ordered-object-slots-v1",
            "stage1_feature_identity": stage1_feature_identity["hash"],
            "stage1_encoding_contract": dict(stage1_encoding_contract),
            "stage1_state_hash": stage1_state_hash,
            "object_encoder_contract": dict(object_encoder_contract),
            "object_encoder_state_hash": object_encoder_state_hash,
            "role_to_id": dict(role_to_id),
        },
    )


__all__ = [
    "build_stage2_data_identity",
    "build_stage2_encoder_identity",
    "build_stage2_training_identity",
    "metadata_identity",
    "stage1_identities_from_loaded",
]
