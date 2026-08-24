from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from rdkit import Chem

from common.identity import IDENTITY_CONTRACT_VERSION, require_compatible_identity, semantic_identity
from common.io import sha256_file
from common.reporting import (
    comparison_identity,
    reporting_block,
    sanitize_task_id,
    write_prediction_csv,
)
from common.training import resolve_device
from stage1.descriptors import calculate_descriptors, rdkit_descriptor_names
from stage1.features import (
    ROLE_TO_ID,
    build_entity_sample,
    inspect_entity_qc,
    load_stage1_feature_inputs,
)
from stage1.masking import MultimodalPacker
from stage1.model import load_stage1_model
from .config import STAGE2_CHECKPOINT_KIND, STAGE2_CHECKPOINT_VERSION, Stage2Config
from .data import load_artifact_registry
from .identity import metadata_identity
from .model import Stage2ObjectModel


STAGE2_PHYSICS_TASKS = (
    "simulation/heat_of_vaporization",
    "simulation/pbe_tzvp_cation_orbitals",
    "simulation/pbe_tzvp_anion_orbitals",
)


def _checkpoint_epoch(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint_epoch_(\d{5})\.pt", path.name)
    return int(match.group(1)) if match else None


def resolve_checkpoint_path(
    checkpoint_dir: str | Path, checkpoint_epoch: int | None = None
) -> Path:
    root = Path(checkpoint_dir)
    if checkpoint_epoch is not None:
        if checkpoint_epoch <= 0:
            raise ValueError("Stage 2 checkpoint epoch must be positive")
        path = root / f"checkpoint_epoch_{checkpoint_epoch:05d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Stage 2 checkpoint: {path}")
        return path
    candidates = [
        (epoch, path)
        for path in root.glob("checkpoint_epoch_*.pt")
        if (epoch := _checkpoint_epoch(path)) is not None
    ]
    if not candidates:
        raise FileNotFoundError(f"No Stage 2 epoch checkpoint under {root}")
    return max(candidates)[1]


def _load_checkpoint_contract(
    config: Stage2Config, checkpoint_path: Path
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("kind") != STAGE2_CHECKPOINT_KIND
        or checkpoint.get("format_version") != STAGE2_CHECKPOINT_VERSION
        or checkpoint.get("identity_contract_version") != IDENTITY_CONTRACT_VERSION
    ):
        raise ValueError("Unsupported Stage 2 evaluation checkpoint")
    epoch = _checkpoint_epoch(checkpoint_path)
    if epoch is None or checkpoint.get("completed_epoch") != epoch:
        raise ValueError("Stage 2 checkpoint filename/completed epoch mismatch")
    registry = load_artifact_registry(config.data.artifacts_dir)
    if (
        checkpoint.get("registry") != registry.snapshot()
        or checkpoint.get("registry_hash") != registry.registry_hash
        or checkpoint.get("catalog_sha256") != registry.catalog_sha256
    ):
        raise ValueError("Stage 2 evaluation checkpoint registry mismatch")
    metadata = json.loads(
        (config.data.artifacts_dir / "metadata.json").read_text(encoding="utf-8")
    )
    stored_data = checkpoint.get("data_identity")
    if not isinstance(stored_data, Mapping):
        raise ValueError("Stage 2 checkpoint lacks its prepared-data identity")
    require_compatible_identity(
        metadata_identity(metadata, "data", context="Stage 2 prepared artifact"),
        stored_data,
        context="Stage 2 evaluation prepared-data identity",
    )
    training_identity = checkpoint.get("training_identity")
    if not isinstance(training_identity, Mapping):
        raise ValueError("Stage 2 checkpoint lacks its training identity")
    return checkpoint, registry, metadata


def _scalers(config: Stage2Config) -> dict[str, Any]:
    path = config.data.artifacts_dir / "scalers.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Malformed Stage 2 scaler artifact")
    return payload


def _scalar_ids(registry: Any) -> tuple[str, ...]:
    return tuple(
        f"{task}::{target}"
        for task in STAGE2_PHYSICS_TASKS
        for target in registry.by_id(task).target_columns
    )


def _comparison(
    config: Stage2Config, registry: Any, scalers: Mapping[str, Any]
) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    normalization: dict[str, Any] = {}
    for task in STAGE2_PHYSICS_TASKS:
        spec = registry.by_id(task)
        for split in ("train", "test"):
            path = spec.dataset.split_path(config.data.data_root, split)
            sources[f"{task}:{split}"] = sha256_file(path)
        for target in spec.target_columns:
            stats = scalers[task]["targets"][target]
            normalization[f"{task}::{target}"] = {
                "scale": float(stats["scale"]),
            }
    return comparison_identity(
        "stage2_physics",
        split="test",
        expected=_scalar_ids(registry),
        sources=sources,
        normalization=normalization,
    )


def resolve_stage2_evaluation_identity(
    config: Stage2Config,
    checkpoint_dir: str | Path,
    *,
    checkpoint_epoch: int | None = None,
) -> dict[str, Any]:
    path = resolve_checkpoint_path(checkpoint_dir, checkpoint_epoch)
    checkpoint, registry, _ = _load_checkpoint_contract(config, path)
    scalers = _scalers(config)
    comparison = _comparison(config, registry, scalers)
    return semantic_identity(
        "stage2.evaluation.v1",
        {
            "checkpoint_sha256": sha256_file(path),
            "checkpoint_epoch": int(checkpoint["completed_epoch"]),
            "training_identity": checkpoint["training_identity"]["hash"],
            "prepared_identity": checkpoint["data_identity"]["hash"],
            "tasks": list(STAGE2_PHYSICS_TASKS),
            "comparison_identity": comparison["hash"],
        },
    )


def _canonical(raw: str, context: str) -> str:
    molecule = Chem.MolFromSmiles((raw or "").strip())
    if molecule is None:
        raise ValueError(f"Invalid Stage 2 test SMILES in {context}: {raw}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _finite(raw: str | None, context: str) -> float:
    try:
        value = float(raw or "")
    except ValueError as error:
        raise ValueError(f"Non-numeric Stage 2 test value in {context}: {raw}") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite Stage 2 test value in {context}: {raw}")
    return value


def _read_test_rows(config: Stage2Config, spec: Any) -> list[dict[str, Any]]:
    path = spec.dataset.split_path(config.data.data_root, "test")
    required = (
        *spec.entity_columns,
        *spec.condition_columns,
        *spec.target_columns,
        "source_list",
    )
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != required:
            raise ValueError(
                f"Unexpected Stage 2 test columns in {path}: {reader.fieldnames}"
            )
        for source_row, raw in enumerate(reader, start=2):
            rows.append(
                {
                    "source_row": source_row,
                    "raw": dict(raw),
                    "canonicals": tuple(
                        _canonical(raw[name], f"{spec.task_id}:{source_row}/{name}")
                        for name in spec.entity_columns
                    ),
                    "conditions": tuple(
                        _finite(raw[name], f"{spec.task_id}:{source_row}/{name}")
                        for name in spec.condition_columns
                    ),
                    "targets": tuple(
                        _finite(raw[name], f"{spec.task_id}:{source_row}/{name}")
                        for name in spec.target_columns
                    ),
                }
            )
    if not rows:
        raise ValueError(f"Stage 2 test split is empty: {spec.task_id}")
    return rows


def _entity_sample(
    role: str,
    canonical_smiles: str,
    *,
    feature_config: Any,
    vocabulary: Any,
    schema: Any,
    standardizer: Any,
) -> dict[str, Any]:
    record = {
        "sample_id": f"stage2-evaluate:{role}:{canonical_smiles}",
        "role": role,
        "role_id": ROLE_TO_ID[role],
        "canonical_smiles": canonical_smiles,
        "sources": ("stage2-evaluate",),
        "split": "test",
        "is_augmented": False,
        "seed_smiles": (),
    }
    qc = inspect_entity_qc(record)
    if vocabulary.token_count(canonical_smiles) > feature_config.data.max_smiles_tokens:
        qc.reasons.append("smiles_overlength")
    if qc.reasons:
        raise ValueError(
            f"Stage 2 test entity fails feature QC: {role}/{canonical_smiles}: "
            + ",".join(qc.reasons)
        )
    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None:
        raise ValueError(f"Invalid canonical Stage 2 test SMILES: {canonical_smiles}")
    raw = calculate_descriptors(molecule, rdkit_descriptor_names())
    return build_entity_sample(
        record, np.asarray(raw), schema, standardizer, vocabulary, feature_config
    )


def _metrics(
    predictions: np.ndarray, targets: np.ndarray, scale: float
) -> dict[str, Any]:
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    actual = np.asarray(targets, dtype=np.float64).reshape(-1)
    if predicted.shape != actual.shape or not len(actual):
        raise ValueError("Stage 2 evaluation metric vectors must be matching and non-empty")
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("Stage 2 evaluation metrics require finite values")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Stage 2 normalized metrics require a positive train scale")
    delta = predicted - actual
    denominator = float(np.square(actual - actual.mean()).sum())
    mae = float(np.abs(delta).mean())
    rmse = float(np.sqrt(np.square(delta).mean()))
    return {
        "count": len(actual),
        "mae": mae,
        "rmse": rmse,
        "r2": (
            float("nan")
            if denominator == 0
            else 1.0 - float(np.square(delta).sum()) / denominator
        ),
        "r2_reason": "constant_target" if denominator == 0 else None,
        "normalized_mae": mae / scale,
        "normalized_rmse": rmse / scale,
    }


@torch.inference_mode()
def evaluate_stage2_checkpoints(
    config: Stage2Config,
    checkpoint_dir: str | Path,
    *,
    checkpoint_epoch: int | None = None,
    predictions_dir: str | Path | None = None,
    expected_evaluation_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_path = resolve_checkpoint_path(checkpoint_dir, checkpoint_epoch)
    checkpoint, registry, _ = _load_checkpoint_contract(config, checkpoint_path)
    scalers = _scalers(config)
    comparison = _comparison(config, registry, scalers)
    evaluation_identity = resolve_stage2_evaluation_identity(
        config, checkpoint_dir, checkpoint_epoch=checkpoint_epoch
    )
    if expected_evaluation_identity is not None:
        require_compatible_identity(
            expected_evaluation_identity,
            evaluation_identity,
            context="Stage 2 run-directory evaluation identity",
        )

    device = resolve_device(config.training.device)
    if config.training.amp_dtype == "bf16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("Stage 2 BF16 evaluation requires capable CUDA")
    loaded = load_stage1_model(
        config.initialization.checkpoint,
        config.data.pretrain_artifacts_dir,
        device=device,
        backbone_dropout=0.0,
    )
    model = Stage2ObjectModel(
        loaded.model,
        registry,
        object_layers=config.model.object_layers,
        object_ffn_dim=config.model.object_ffn_dim,
        dropout=config.model.dropout,
    ).to(device)
    if checkpoint.get("model_contract") != model.model_contract:
        raise ValueError("Stage 2 evaluation checkpoint model contract mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    feature_config, vocabulary, schema, standardizer, feature_hash = (
        load_stage1_feature_inputs(
            config.initialization.checkpoint, config.data.pretrain_artifacts_dir
        )
    )
    if feature_hash != loaded.artifact_hash:
        raise ValueError("Stage 2 evaluation feature identity mismatch")
    packer = MultimodalPacker(vocabulary)
    sample_cache: dict[tuple[str, str], dict[str, Any]] = {}
    task_metrics: dict[str, Any] = {}
    prediction_manifests: list[dict[str, Any]] = []

    for task in STAGE2_PHYSICS_TASKS:
        spec = registry.by_id(task)
        rows = _read_test_rows(config, spec)
        raw_predictions: list[np.ndarray] = []
        for start in range(0, len(rows), config.training.batch_size):
            chunk = rows[start : start + config.training.batch_size]
            samples = []
            role_rows = []
            for row in chunk:
                roles = []
                for role, canonical in zip(
                    spec.role_policy, row["canonicals"], strict=True
                ):
                    key = (role, canonical)
                    if key not in sample_cache:
                        sample_cache[key] = _entity_sample(
                            role,
                            canonical,
                            feature_config=feature_config,
                            vocabulary=vocabulary,
                            schema=schema,
                            standardizer=standardizer,
                        )
                    samples.append(sample_cache[key])
                    roles.append(ROLE_TO_ID[role])
                role_rows.append(roles)
            packed = packer(samples).to(device)
            slots = model.encode_entities(packed).reshape(
                len(chunk), len(spec.entity_columns), -1
            )
            roles = torch.tensor(role_rows, dtype=torch.long, device=device)
            condition_values = []
            for row in chunk:
                condition_values.append(
                    [
                        (value - float(scalers[task]["conditions"][name]["mean"]))
                        / float(scalers[task]["conditions"][name]["scale"])
                        for name, value in zip(
                            spec.condition_columns, row["conditions"], strict=True
                        )
                    ]
                )
            conditions = torch.tensor(
                condition_values, dtype=torch.float32, device=device
            ).reshape(len(chunk), len(spec.condition_columns))
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=config.training.amp_dtype == "bf16",
            ):
                normalized = model.predict_object(spec, slots, roles, conditions)
            if not torch.isfinite(normalized).all():
                raise RuntimeError(f"Non-finite Stage 2 evaluation prediction: {task}")
            prediction = normalized.float().cpu().numpy()
            for column, target in enumerate(spec.target_columns):
                stats = scalers[task]["targets"][target]
                prediction[:, column] = (
                    prediction[:, column] * float(stats["scale"])
                    + float(stats["mean"])
                )
            raw_predictions.append(prediction.astype(np.float64))
        predictions = np.concatenate(raw_predictions, axis=0)
        targets = np.asarray([row["targets"] for row in rows], dtype=np.float64)
        task_metrics[task] = {
            target: _metrics(
                predictions[:, column],
                targets[:, column],
                float(scalers[task]["targets"][target]["scale"]),
            )
            for column, target in enumerate(spec.target_columns)
        }
        if predictions_dir is not None:
            output_rows: list[dict[str, Any]] = []
            fields = ["source_row", *spec.entity_columns, *spec.condition_columns]
            if len(spec.target_columns) == 1:
                fields.extend(("target", "prediction", "absolute_error"))
            else:
                for target in spec.target_columns:
                    fields.extend(
                        (
                            f"{target}_target",
                            f"{target}_prediction",
                            f"{target}_absolute_error",
                        )
                    )
            for row_index, row in enumerate(rows):
                output: dict[str, Any] = {"source_row": row["source_row"]}
                for name in (*spec.entity_columns, *spec.condition_columns):
                    output[name] = row["raw"][name]
                for column, target in enumerate(spec.target_columns):
                    actual = float(targets[row_index, column])
                    predicted = float(predictions[row_index, column])
                    if len(spec.target_columns) == 1:
                        output["target"] = actual
                        output["prediction"] = predicted
                        output["absolute_error"] = abs(predicted - actual)
                    else:
                        output[f"{target}_target"] = actual
                        output[f"{target}_prediction"] = predicted
                        output[f"{target}_absolute_error"] = abs(predicted - actual)
                output_rows.append(output)
            manifest = write_prediction_csv(
                Path(predictions_dir) / f"{sanitize_task_id(task)}.csv",
                output_rows,
                fields,
            )
            manifest["path"] = f"predictions/{sanitize_task_id(task)}.csv"
            manifest["task"] = task
            prediction_manifests.append(manifest)

    scalar_values = [
        float(task_metrics[task][target]["normalized_mae"])
        for task in STAGE2_PHYSICS_TASKS
        for target in registry.by_id(task).target_columns
    ]
    epoch = int(checkpoint["completed_epoch"])
    return {
        "split": "test",
        "checkpoint_epoch": epoch,
        "tasks": task_metrics,
        "macro_normalized_mae": {
            "value": sum(scalar_values) / len(scalar_values),
            "valid_targets": len(scalar_values),
            "total_targets": len(_scalar_ids(registry)),
        },
        "reporting": reporting_block(
            model_id="ilume",
            model_display_name="ILUME",
            benchmark="stage2_physics",
            protocol={
                "split": "test",
                "ensemble": False,
                "expected_tasks": list(STAGE2_PHYSICS_TASKS),
                "expected_targets": list(_scalar_ids(registry)),
                "checkpoint_epoch": epoch,
            },
            comparison=comparison,
            study_id=f"ilume-stage2-{checkpoint['training_identity']['hash']}",
            predictions=prediction_manifests,
        ),
    }


__all__ = [
    "STAGE2_PHYSICS_TASKS",
    "evaluate_stage2_checkpoints",
    "resolve_checkpoint_path",
    "resolve_stage2_evaluation_identity",
]
