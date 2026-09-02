from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from rdkit import Chem

from common.descriptor_preprocessing import FeaturePreprocessor
from common.identity import (
    IDENTITY_CONTRACT_VERSION,
    require_compatible_identity,
    semantic_identity,
)
from common.io import sha256_file
from common.progress import ProgressReporter
from common.reporting import (
    STAGE2_BENCHMARK_SUITE_CONTRACT,
    role_mae_diagnostics,
    sanitize_task_id,
    write_prediction_csv,
)
from common.training import resolve_device
from stage1.descriptors import calculate_descriptors, rdkit_descriptor_names
from stage1.features import ROLE_TO_ID

from .config import STAGE2_CHECKPOINT_VERSION, Stage2Config
from .atom_evaluation import PARTIAL_CHARGE_UNIT
from .data import STAGE2_RDKIT_ARTIFACT_KIND, load_artifact_registry
from .evaluate import (
    STAGE2_CORE_TASKS,
    _comparison,
    _load_refined_artifact,
    _metrics,
    _read_test_rows,
    _scalers,
    resolve_checkpoint_path,
)
from .identity import metadata_identity
from .model import RDKitDescriptorBackbone, Stage2ObjectModel
from .rdkit_train import STAGE2_RDKIT_CHECKPOINT_KIND, STAGE2_RDKIT_REFINED_KIND
from .registry import ORBITAL_TASK_TARGETS, orbital_audit_columns


def _load_checkpoint(
    config: Stage2Config, path: Path
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("kind") != STAGE2_RDKIT_CHECKPOINT_KIND
        or checkpoint.get("format_version") != STAGE2_CHECKPOINT_VERSION
        or checkpoint.get("identity_contract_version") != IDENTITY_CONTRACT_VERSION
    ):
        raise ValueError("Unsupported RDKit Stage 2 evaluation checkpoint")
    completed_epoch = checkpoint.get("completed_epoch")
    if (
        not isinstance(completed_epoch, int)
        or completed_epoch <= 0
        or path.name != f"checkpoint_epoch_{completed_epoch:05d}.pt"
    ):
        raise ValueError("RDKit Stage 2 checkpoint filename/completed epoch mismatch")
    if not isinstance(checkpoint.get("training_identity"), Mapping):
        raise ValueError("RDKit Stage 2 checkpoint lacks its training identity")
    registry = load_artifact_registry(config.data.artifacts_dir)
    config.validate_registry(registry)
    if (
        checkpoint.get("registry") != registry.snapshot()
        or checkpoint.get("registry_hash") != registry.registry_hash
        or checkpoint.get("catalog_sha256") != registry.catalog_sha256
    ):
        raise ValueError("RDKit Stage 2 evaluation checkpoint registry mismatch")
    metadata = json.loads(
        (config.data.artifacts_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("kind") != STAGE2_RDKIT_ARTIFACT_KIND:
        raise ValueError("RDKit Stage 2 evaluation requires RDKit prepared data")
    require_compatible_identity(
        metadata_identity(metadata, "data", context="RDKit Stage 2 prepared artifact"),
        checkpoint.get("data_identity", {}),
        context="RDKit Stage 2 evaluation prepared-data identity",
    )
    return checkpoint, registry, metadata


def _model(
    config: Stage2Config, registry: Any, metadata: Mapping[str, Any]
) -> Stage2ObjectModel:
    representation = config.representation
    if representation is None:
        raise ValueError("RDKit Stage 2 evaluation requires representation config")
    retained_width = int(metadata["descriptor_contract"]["retained_width"])
    return Stage2ObjectModel(
        RDKitDescriptorBackbone(
            retained_width,
            hidden_dim=representation.hidden_dim,
            output_dim=representation.output_dim,
            dropout=representation.dropout,
        ),  # type: ignore[arg-type]
        registry,
        object_layers=config.model.object_layers,
        object_ffn_dim=config.model.object_ffn_dim,
        dropout=config.model.dropout,
    )


def _evaluation_identity(
    model_path: Path,
    checkpoint: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    model_state_hash: str | None,
    selection_manifest_hash: str | None,
    taskwise_refined: bool,
) -> dict[str, Any]:
    return semantic_identity(
        "stage2.rdkit-evaluation.v1",
        {
            "checkpoint_sha256": sha256_file(model_path),
            "checkpoint_epoch": None if taskwise_refined else checkpoint["completed_epoch"],
            "model_selector": "taskwise_refined" if taskwise_refined else "epoch_checkpoint",
            "model_state_hash": model_state_hash,
            "selection_manifest_sha256": selection_manifest_hash,
            "training_identity": checkpoint["training_identity"]["hash"],
            "prepared_identity": checkpoint["data_identity"]["hash"],
            "tasks": list(STAGE2_CORE_TASKS),
            "unsupported": ["stage2_partial_charge", "stage2_physics_full"],
            "comparison_identity": comparison["hash"],
        },
    )


def resolve_rdkit_stage2_evaluation_contract(
    config: Stage2Config,
    checkpoint_dir: str | Path,
    *,
    checkpoint_epoch: int | None = None,
) -> tuple[dict[str, Any], None]:
    taskwise_refined = checkpoint_epoch is None
    checkpoint_path = resolve_checkpoint_path(checkpoint_dir, checkpoint_epoch)
    checkpoint, registry, _ = _load_checkpoint(config, checkpoint_path)
    model_path = checkpoint_path
    state_hash = None
    manifest_hash = None
    if taskwise_refined:
        model_path, refined, manifest_hash = _load_refined_artifact(
            checkpoint_dir,
            checkpoint,
            artifact_kind=STAGE2_RDKIT_REFINED_KIND,
        )
        state_hash = refined.get("model_state_hash")
    return (
        _evaluation_identity(
            model_path,
            checkpoint,
            _comparison(config, registry, _scalers(config)),
            model_state_hash=state_hash,
            selection_manifest_hash=manifest_hash,
            taskwise_refined=taskwise_refined,
        ),
        None,
    )


def _features(
    rows: list[dict[str, Any]], preprocessor: FeaturePreprocessor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    names = rdkit_descriptor_names()
    raw: list[np.ndarray] = []
    roles: list[int] = []
    for row in rows:
        for role, smiles in zip(row["roles"], row["canonicals"], strict=True):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"Invalid canonical Stage 2 test SMILES: {smiles}")
            raw.append(calculate_descriptors(molecule, names))
            roles.append(ROLE_TO_ID[role])
    return (
        torch.from_numpy(preprocessor.transform(np.stack(raw))).to(device),
        torch.tensor(roles, dtype=torch.long, device=device),
    )


@torch.inference_mode()
def evaluate_rdkit_stage2_checkpoints(
    config: Stage2Config,
    checkpoint_dir: str | Path,
    *,
    checkpoint_epoch: int | None = None,
    predictions_dir: str | Path | None = None,
    expected_evaluation_identity: Mapping[str, Any] | None = None,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    taskwise_refined = checkpoint_epoch is None
    checkpoint_path = resolve_checkpoint_path(checkpoint_dir, checkpoint_epoch)
    checkpoint, registry, metadata = _load_checkpoint(config, checkpoint_path)
    model_payload = checkpoint
    model_path = checkpoint_path
    state_hash = None
    manifest_hash = None
    if taskwise_refined:
        model_path, model_payload, manifest_hash = _load_refined_artifact(
            checkpoint_dir,
            checkpoint,
            artifact_kind=STAGE2_RDKIT_REFINED_KIND,
        )
        state_hash = model_payload.get("model_state_hash")
    scalers = _scalers(config)
    comparison = _comparison(config, registry, scalers)
    identity = _evaluation_identity(
        model_path,
        checkpoint,
        comparison,
        model_state_hash=state_hash,
        selection_manifest_hash=manifest_hash,
        taskwise_refined=taskwise_refined,
    )
    if expected_evaluation_identity is not None:
        require_compatible_identity(
            expected_evaluation_identity,
            identity,
            context="RDKit Stage 2 run-directory evaluation identity",
        )

    device = resolve_device(config.training.device)
    if config.training.amp_dtype == "bf16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("Stage 2 BF16 evaluation requires capable CUDA")
    model = _model(config, registry, metadata).to(device)
    if checkpoint.get("model_contract") != model.model_contract:
        raise ValueError("RDKit Stage 2 evaluation model contract mismatch")
    model.load_state_dict(model_payload["model"], strict=True)
    model.eval()
    preprocessor = FeaturePreprocessor.from_dict(
        metadata["descriptor_contract"]["preprocessing"]
    )
    task_metrics: dict[str, Any] = {}
    prediction_manifests: list[dict[str, Any]] = []
    rows_by_task = {
        task: _read_test_rows(config, registry.by_id(task))
        for task in STAGE2_CORE_TASKS
    }
    progress = (reporter or ProgressReporter()).bar(
        total=sum(math.ceil(len(rows) / config.training.batch_size) for rows in rows_by_task.values()),
        desc="Stage 2 RDKit test evaluation",
        unit="batch",
    )
    try:
        for task in STAGE2_CORE_TASKS:
            spec = registry.by_id(task)
            rows = rows_by_task[task]
            predictions: list[np.ndarray] = []
            for start in range(0, len(rows), config.training.batch_size):
                chunk = rows[start : start + config.training.batch_size]
                features, flat_roles = _features(chunk, preprocessor, device)
                slot_count = len(spec.entity_columns)
                slots = model.encode_entities(features).reshape(len(chunk), slot_count, -1)
                role_tensor = flat_roles.reshape(len(chunk), slot_count)
                conditions = torch.tensor(
                    [
                        [
                            (value - float(scalers[task]["conditions"][name]["mean"]))
                            / float(scalers[task]["conditions"][name]["scale"])
                            for name, value in zip(spec.condition_columns, row["conditions"], strict=True)
                        ]
                        for row in chunk
                    ],
                    dtype=torch.float32,
                    device=device,
                ).reshape(len(chunk), len(spec.condition_columns))
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=config.training.amp_dtype == "bf16",
                ):
                    normalized = model.predict_object(spec, slots, role_tensor, conditions)
                if not torch.isfinite(normalized).all():
                    raise RuntimeError(f"Non-finite RDKit Stage 2 prediction: {task}")
                raw_prediction = normalized.float().cpu().numpy()
                for column, target in enumerate(spec.target_columns):
                    stats = scalers[task]["targets"][target]
                    raw_prediction[:, column] = (
                        raw_prediction[:, column] * float(stats["scale"])
                        + float(stats["mean"])
                    )
                predictions.append(raw_prediction.astype(np.float64))
                progress.update(1)
            predicted = np.concatenate(predictions)
            targets = np.asarray([row["targets"] for row in rows], dtype=np.float64)
            task_metrics[task] = {
                target: _metrics(
                    predicted[:, column],
                    targets[:, column],
                    float(scalers[task]["targets"][target]["scale"]),
                )
                for column, target in enumerate(spec.target_columns)
            }
            if task in ORBITAL_TASK_TARGETS:
                target = ORBITAL_TASK_TARGETS[task]
                task_metrics[task][target]["role_diagnostics"] = role_mae_diagnostics(
                    predicted[:, 0],
                    targets[:, 0],
                    [row["raw"]["ion_role"] for row in rows],
                )
            if predictions_dir is not None:
                fields = [
                    "source_row",
                    *spec.entity_columns,
                    *spec.condition_columns,
                    *orbital_audit_columns(task),
                    "target",
                    "prediction",
                    "absolute_error",
                ]
                output_rows = []
                for index, row in enumerate(rows):
                    actual = float(targets[index, 0])
                    value = float(predicted[index, 0])
                    output_rows.append(
                        {
                            "source_row": row["source_row"],
                            **{
                                name: row["raw"][name]
                                for name in (
                                    *spec.entity_columns,
                                    *spec.condition_columns,
                                    *orbital_audit_columns(task),
                                )
                            },
                            "target": actual,
                            "prediction": value,
                            "absolute_error": abs(value - actual),
                        }
                    )
                manifest = write_prediction_csv(
                    Path(predictions_dir) / f"{sanitize_task_id(task)}.csv",
                    output_rows,
                    fields,
                )
                manifest["path"] = f"predictions/{sanitize_task_id(task)}.csv"
                manifest["task"] = task
                prediction_manifests.append(manifest)
    finally:
        progress.close()

    values = [
        float(task_metrics[task][target]["normalized_mae"])
        for task in STAGE2_CORE_TASKS
        for target in registry.by_id(task).target_columns
    ]
    selected_epoch = None if taskwise_refined else int(checkpoint["completed_epoch"])
    selector = "taskwise_refined" if taskwise_refined else "epoch_checkpoint"
    protocol = {
        "split": "test",
        "ensemble": False,
        "checkpoint_epoch": selected_epoch,
        "model_selector": selector,
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    unsupported = {
        "status": "unsupported",
        "protocol": {"split": "test", "folds": [], "ensemble": False},
        "issues": [],
    }
    return {
        "split": "test",
        "checkpoint_epoch": selected_epoch,
        "model_selector": selector,
        "tasks": task_metrics,
        "core_macro_normalized_mae": {
            "value": sum(values) / len(values),
            "valid_tasks": len(values),
            "total_tasks": len(STAGE2_CORE_TASKS),
        },
        "full_macro_normalized_mae": {
            "value": None,
            "valid_units": len(values),
            "total_units": len(values) + 1,
        },
        "reporting": {
            "schema_version": 1,
            "contract": STAGE2_BENCHMARK_SUITE_CONTRACT,
            "model_id": "rdkit_2d_stage2",
            "model_display_name": "RDKit 2D MLP + Stage2",
            "study_id": f"rdkit-2d-stage2-{checkpoint['training_identity']['hash']}",
            "capabilities": {
                "stage2_core_physics": "supported",
                "stage2_partial_charge": "unsupported",
                "stage2_physics_full": "unsupported",
            },
            "benchmarks": {
                "stage2_core_physics": {
                    "status": "complete",
                    "benchmark": "stage2_physics",
                    "protocol": {**protocol, "expected_tasks": list(STAGE2_CORE_TASKS)},
                    "comparison_identity": comparison,
                    "issues": [],
                },
                "stage2_partial_charge": {
                    **unsupported,
                    "benchmark": "stage2_partial_charge",
                    "protocol": {**unsupported["protocol"], "expected_units": [PARTIAL_CHARGE_UNIT]},
                },
                "stage2_physics_full": {
                    **unsupported,
                    "benchmark": "stage2_physics_full",
                    "protocol": {
                        **unsupported["protocol"],
                        "expected_units": [*STAGE2_CORE_TASKS, PARTIAL_CHARGE_UNIT],
                    },
                },
            },
            "predictions": prediction_manifests,
        },
    }


__all__ = [
    "evaluate_rdkit_stage2_checkpoints",
    "resolve_rdkit_stage2_evaluation_contract",
]
