from __future__ import annotations

import csv
from typing import Any, Mapping, Sequence

from common.identity import semantic_identity
from common.io import sha256_file
from stage2 import load_stage2_encoder_identity
from .config import Stage3Config
from .data import (
    OBJECT_ENCODING_CONTRACT_VERSION,
    ObjectKey,
    ResolvedTaskSpec,
    fit_normalization,
    source_path,
    test_path,
)


STAGE3_PREPARED_IDENTITY_CONTRACT_VERSION = 2
STAGE3_TRAINING_IDENTITY_CONTRACT_VERSION = 1
STAGE3_EVALUATION_IDENTITY_CONTRACT_VERSION = 1


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
    config: Stage3Config, registry: Mapping[str, ResolvedTaskSpec]
) -> dict[str, dict[str, Any]]:
    def rows(path: Any) -> int:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return sum(1 for _ in csv.reader(handle)) - 1

    result: dict[str, dict[str, Any]] = {
        "task_catalog": {
            "sha256": sha256_file(config.data.task_catalog),
            "size": config.data.task_catalog.stat().st_size,
            "rows": rows(config.data.task_catalog),
        }
    }
    for task_id, spec in sorted(registry.items()):
        for fold in range(1, 6):
            path = source_path(config, spec, fold)
            result[f"{task_id}:fold{fold}"] = {
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
                "rows": rows(path),
            }
        path = test_path(config, spec)
        if path.is_file():
            result[f"{task_id}:test"] = {
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
                "rows": rows(path),
            }
    return result


def build_stage3_prepared_identity(
    config: Stage3Config,
    registry: Mapping[str, ResolvedTaskSpec],
    objects: Sequence[ObjectKey],
    normalization: Mapping[str, Any],
    stage2_encoder_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return semantic_identity(
        "stage3.prepared-data",
        {
            "contract_version": STAGE3_PREPARED_IDENTITY_CONTRACT_VERSION,
            "source_content": _source_content(config, registry),
            "resolved_registry": {
                task: spec.prepared_dict()
                for task, spec in sorted(registry.items())
            },
            "split": {
                "policy": config.data.split_policy,
                "strategies": dict(config.data.split_strategies),
                "cv_repeat": config.data.cv_repeat,
                "cv_repeats": dict(config.data.cv_repeats),
                "seed": config.data.seed,
            },
            "normalization": dict(normalization),
            "objects": [key.to_dict() for key in objects],
            "object_encoding_contract_version": OBJECT_ENCODING_CONTRACT_VERSION,
            "stage2_encoder_identity": stage2_encoder_identity["hash"],
        },
    )


def build_stage3_rdkit_prepared_identity(
    config: Stage3Config,
    registry: Mapping[str, ResolvedTaskSpec],
    objects: Sequence[ObjectKey],
    normalization: Mapping[str, Any],
    descriptor_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return semantic_identity(
        "stage3.rdkit-prepared-data",
        {
            "contract_version": 1,
            "source_content": _source_content(config, registry),
            "resolved_registry": {
                task: spec.to_dict() for task, spec in sorted(registry.items())
            },
            "split": {
                "policy": config.data.split_policy,
                "strategies": dict(config.data.split_strategies),
                "cv_repeat": config.data.cv_repeat,
                "cv_repeats": dict(config.data.cv_repeats),
                "seed": config.data.seed,
            },
            "normalization": dict(normalization),
            "objects": [key.to_dict() for key in objects],
            "representation": dict(descriptor_contract),
        },
    )


def resolve_stage3_prepared_identity(
    config: Stage3Config,
    registry: Mapping[str, ResolvedTaskSpec],
    objects: Sequence[ObjectKey],
) -> dict[str, Any]:
    if config.representation is not None:
        from .rdkit import resolve_rdkit_materialization

        return resolve_rdkit_materialization(config, registry, objects)[
            "prepared_identity"
        ]
    normalization = {
        f"fold{fold}": fit_normalization(config, registry, fold)
        for fold in range(1, 6)
    }
    encoder_path = config.initialization.stage2_encoder
    assert encoder_path is not None
    encoder_identity = load_stage2_encoder_identity(encoder_path)
    return build_stage3_prepared_identity(
        config, registry, objects, normalization, encoder_identity
    )


def build_stage3_training_identity(plan: Mapping[str, Any]) -> dict[str, Any]:
    semantic_plan = {
        name: plan[name]
        for name in (
            "fold",
            "active_tasks",
            "resolved_registry",
            "groups",
            "data",
            "model",
            "optimizer",
            "scheduler",
            "refinement",
            "math",
            "prepared_identity",
            "normalization_hash",
            "ownership_manifest",
            "plugin",
            "trainable_parameters",
            "frozen_parameters",
        )
    }
    if "representation" in plan:
        semantic_plan["representation"] = plan["representation"]
    else:
        semantic_plan["stage2_encoder_identity"] = plan["stage2_encoder_identity"]
    if "training_seed" in plan:
        semantic_plan["training_seed"] = plan["training_seed"]
    return semantic_identity(
        "stage3.training",
        {
            "contract_version": STAGE3_TRAINING_IDENTITY_CONTRACT_VERSION,
            "plan": semantic_plan,
        },
    )


def build_stage3_evaluation_identity(
    *,
    prepared_identity: Mapping[str, Any],
    checkpoint_identities: Sequence[Mapping[str, Any]],
    model_state_hashes: Sequence[str],
    selection_manifest_hashes: Sequence[str] = (),
    split: str,
    fold: int | None,
    checkpoint_epoch: int | None,
    model_selector: str = "epoch_checkpoint",
    tasks: Sequence[str],
    ensemble_folds: bool,
) -> dict[str, Any]:
    return semantic_identity(
        "stage3.evaluation",
        {
            "contract_version": STAGE3_EVALUATION_IDENTITY_CONTRACT_VERSION,
            "prepared_identity": prepared_identity["hash"],
            "checkpoint_training_identities": [
                identity["hash"] for identity in checkpoint_identities
            ],
            "model_state_hashes": list(model_state_hashes),
            "selection_manifest_sha256": list(selection_manifest_hashes),
            "selector": {
                "split": split,
                "fold": fold,
                "checkpoint_epoch": checkpoint_epoch,
                "model_selector": model_selector,
                "tasks": list(tasks),
                "ensemble_folds": ensemble_folds,
            },
        },
    )


__all__ = [
    "build_stage3_evaluation_identity",
    "build_stage3_prepared_identity",
    "build_stage3_rdkit_prepared_identity",
    "build_stage3_training_identity",
    "metadata_identity",
    "resolve_stage3_prepared_identity",
]
