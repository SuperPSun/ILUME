from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from common.progress import ProgressReporter
from common.identity import (
    IDENTITY_CONTRACT_VERSION,
    require_compatible_identity,
    tensor_state_hash,
)
from common.training import canonical_json_sha256, resolve_device
from common.io import sha256_file
from common.reporting import (
    comparison_identity,
    reporting_block,
    sanitize_task_id,
    write_prediction_csv,
)
from .config import Stage3Config
from .data import Stage3TaskDataset, iter_rows, source_path, test_path
from .model import Stage3SparseModel
from .prepare import load_prepared_stage3
from .train import (
    STAGE3_CHECKPOINT_KIND,
    STAGE3_CHECKPOINT_VERSION,
    STAGE3_REFINED_KIND,
    regression_metrics,
)
from .identity import (
    build_stage3_evaluation_identity,
    build_stage3_training_identity,
    metadata_identity,
)


def _checkpoint_path(root: Path, fold: int, epoch: int) -> Path:
    filename = f"checkpoint_epoch_{epoch:05d}.pt"
    nested = root / f"fold{fold}" / filename
    return nested if nested.is_file() else root / filename


def _refined_path(root: Path, fold: int) -> Path:
    nested = root / f"fold{fold}" / "taskwise_refined.pt"
    return nested if nested.is_file() else root / "taskwise_refined.pt"


def _validate_refinement_manifest(
    artifact_path: Path, artifact: Mapping[str, Any], expected_epoch: int
) -> str:
    manifest_path = artifact_path.with_name("taskwise_refinement.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Stage 3 refinement manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Stage 3 refinement manifest is unreadable") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("kind") != STAGE3_REFINED_KIND
        or manifest.get("format_version") != 1
        or manifest.get("artifact") != artifact_path.name
        or manifest.get("artifact_sha256") != sha256_file(artifact_path)
        or manifest.get("fold") != artifact.get("fold")
        or manifest.get("ordinary_final_epoch") != expected_epoch
    ):
        raise ValueError("Stage 3 refinement manifest/artifact integrity mismatch")
    for key in (
        "boundary_epoch", "shared_state_hash", "private_state_hashes",
        "selected_tasks", "validation",
    ):
        if json.dumps(manifest.get(key), sort_keys=True) != json.dumps(
            artifact.get(key), sort_keys=True
        ):
            raise ValueError(f"Stage 3 refinement manifest mismatch: {key}")
    return sha256_file(manifest_path)


def _load_model(
    config: Stage3Config,
    prepared: Mapping[str, Any],
    checkpoint_path: Path,
    fold: int,
    epoch: int,
    device: torch.device,
    *,
    taskwise_refined: bool = False,
) -> tuple[Stage3SparseModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = {
        "kind": STAGE3_REFINED_KIND if taskwise_refined else STAGE3_CHECKPOINT_KIND,
        "format_version": 1 if taskwise_refined else STAGE3_CHECKPOINT_VERSION,
        "fold": fold,
        "stage2_encoder_identity": metadata_identity(
            prepared["metadata"], "stage2_encoder", context="Stage 3 prepared artifact"
        )["hash"],
        "resolved_registry": {
            task_id: spec.to_dict() for task_id, spec in prepared["registry"].items()
        },
    }
    if not taskwise_refined:
        expected.update({
            "identity_contract_version": IDENTITY_CONTRACT_VERSION,
            "stage": "stage3",
            "completed_epoch": epoch,
        })
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Stage 3 evaluation checkpoint mismatch: {key}")
    plan = checkpoint.get("resolved_training_plan")
    if not isinstance(plan, dict):
        raise ValueError("Stage 3 checkpoint lacks its resolved training plan")
    training_identity = checkpoint.get("training_identity")
    if not isinstance(training_identity, Mapping):
        raise ValueError("Stage 3 checkpoint predates identity contract v1; retrain it")
    require_compatible_identity(
        training_identity,
        build_stage3_training_identity(plan),
        context="Stage 3 evaluation checkpoint training identity",
    )
    if plan.get("prepared_identity") != metadata_identity(
        prepared["metadata"], "prepared", context="Stage 3 prepared artifact"
    )["hash"]:
        raise ValueError("Stage 3 checkpoint prepared-data identity mismatch")
    if plan.get("normalization_hash") != canonical_json_sha256(
        checkpoint.get("normalization")
    ):
        raise ValueError("Stage 3 checkpoint normalization mismatch")
    d_model = int(prepared["objects"]["embeddings"].shape[1])
    model = Stage3SparseModel(config.model, prepared["registry"], d_model)
    if checkpoint.get("ownership_manifest") != model.ownership_manifest():
        raise ValueError("Stage 3 checkpoint ownership mismatch")
    state_namespace = (
        "stage3.taskwise-refined-state"
        if taskwise_refined
        else "stage3.model-state"
    )
    if checkpoint.get("model_state_hash") != tensor_state_hash(
        state_namespace, checkpoint["model"]
    ):
        raise ValueError("Stage 3 evaluation checkpoint model state hash mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def _predict(
    model: Stage3SparseModel,
    task_id: str,
    dataset: Stage3TaskDataset,
    embeddings: torch.Tensor,
    normalization: Mapping[str, Any],
    config: Stage3Config,
    device: torch.device,
    *,
    progress_bar: Any | None = None,
    fold: int | None = None,
    fold_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized_predictions: list[torch.Tensor] = []
    target_stats = normalization["target"]

    microbatch_size = config.training.microbatch_size
    total_batches = math.ceil(len(dataset) / microbatch_size) if len(dataset) else 0

    for batch_index, start in enumerate(
        range(0, len(dataset), microbatch_size),
        start=1,
    ):
        if progress_bar is not None:
            progress_bar.set_postfix_str(
                (
                    f"fold={fold}/{fold_count} "
                    f"task={task_id} "
                    f"batch={batch_index}/{total_batches}"
                ),
                refresh=True,
            )
        indices = torch.arange(start, min(len(dataset), start + microbatch_size))
        primary = embeddings[dataset.primary_object_ids[indices]].to(device)
        partner_ids = dataset.partner_object_ids[indices]
        partner = (
            embeddings[partner_ids].to(device)
            if len(partner_ids) and bool((partner_ids >= 0).all())
            else None
        )
        conditions = dataset.conditions[indices].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=config.training.amp_dtype == "bf16",
        ):
            prediction = model(
                task_id, primary, conditions, partner_embedding=partner
            ).predictions
        if not torch.isfinite(prediction).all():
            raise RuntimeError(f"Non-finite Stage 3 evaluation prediction: {task_id}")
        normalized_predictions.append(prediction.float().cpu())
    normalized = (
        torch.cat(normalized_predictions) if normalized_predictions else torch.empty(0)
    )
    raw_predictions = normalized * float(target_stats["scale"]) + float(
        target_stats["mean"]
    )
    normalized_targets = (
        dataset.raw_targets.float() - float(target_stats["mean"])
    ) / float(target_stats["scale"])
    return normalized, raw_predictions, normalized_targets


def _raw_ensemble_metrics(
    predictions: torch.Tensor, targets: torch.Tensor, scale: float
) -> dict[str, Any]:
    delta = predictions.double() - targets.double()
    count = int(targets.numel())
    if count == 0:
        return {"count": 0, "reason": "no_samples"}
    denominator = float((targets.double() - targets.double().mean()).square().sum())
    prediction_std = float(predictions.double().std(unbiased=False)) if count else 0.0
    target_std = float(targets.double().std(unbiased=False)) if count else 0.0
    pearson = (
        float(torch.corrcoef(torch.stack((predictions.double(), targets.double())))[0, 1])
        if count >= 2 and prediction_std > 0 and target_std > 0
        else float("nan")
    )
    return {
        "count": count,
        "mae": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "r2": (
            float("nan")
            if denominator == 0
            else 1.0 - float(delta.square().sum()) / denominator
        ),
        "r2_reason": "constant_target" if denominator == 0 else None,
        "pearson_r": pearson,
        "pearson_reason": (
            "insufficient_or_constant_samples" if math.isnan(pearson) else None
        ),
        "normalized_mae": float(delta.abs().mean()) / scale,
        "normalized_rmse": float(delta.square().mean().sqrt()) / scale,
    }


def _macro(
    per_task: Mapping[str, Mapping[str, Any]], registry: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {"macro_task_equal": {}, "macro_group_equal": {}}
    metrics = ("mae", "rmse", "r2", "pearson_r", "normalized_mae", "normalized_rmse")
    for metric in metrics:
        valid = {
            task: float(values[metric])
            for task, values in per_task.items()
            if metric in values and math.isfinite(float(values[metric]))
        }
        result["macro_task_equal"][metric] = {
            "value": sum(valid.values()) / len(valid) if valid else float("nan"),
            "valid_tasks": len(valid),
            "total_tasks": len(per_task),
        }
        grouped = []
        groups = sorted({registry[task].meta_group for task in per_task})
        for group in groups:
            values = [
                value
                for task, value in valid.items()
                if registry[task].meta_group == group
            ]
            if values:
                grouped.append(sum(values) / len(values))
        result["macro_group_equal"][metric] = {
            "value": sum(grouped) / len(grouped) if grouped else float("nan"),
            "valid_groups": len(grouped),
            "total_groups": len(groups),
        }
    return result


def _reporting_comparison(
    config: Stage3Config,
    prepared: Mapping[str, Any],
    *,
    split: str,
    expected_tasks: Sequence[str],
) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    normalization: dict[str, Any] = {}
    for task in expected_tasks:
        spec = prepared["registry"][task]
        for fold in range(1, 6):
            sources[f"{task}:fold{fold}"] = sha256_file(
                source_path(config, spec, fold)
            )
        if split == "test":
            sources[f"{task}:test"] = sha256_file(test_path(config, spec))
        for fold in range(1, 6):
            stats = prepared["normalization"][f"fold{fold}"][task]["target"]
            normalization[f"{task}:fold{fold}"] = {
                "scale": float(stats["scale"]),
            }
    return comparison_identity(
        "stage3_property",
        split=split,
        expected=expected_tasks,
        sources=sources,
        normalization=normalization,
        folds=tuple(range(1, 6)),
        ensemble=split == "test",
    )


def _default_reporting_study_id(
    metadata: Mapping[str, Any], selector: str
) -> str:
    return (
        "ilume-stage3-"
        + metadata_identity(
            metadata, "prepared", context="Stage 3 prepared artifact"
        )["hash"]
        + f"-{selector}"
    )


def resolve_stage3_reporting_study_id(
    config: Stage3Config, *, checkpoint_epoch: int | None = None,
) -> str:
    """Resolve the fold-independent default reporting study identifier."""
    taskwise_refined = checkpoint_epoch is None
    epoch = config.training.epochs if checkpoint_epoch is None else checkpoint_epoch
    if epoch <= 0:
        raise ValueError("Stage 3 checkpoint epoch must be positive")
    metadata = json.loads(
        (config.data.artifacts_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if not isinstance(metadata, dict):
        raise ValueError("Stage 3 prepared metadata must contain a JSON object")
    return _default_reporting_study_id(
        metadata,
        "taskwise-refined" if taskwise_refined else f"epoch{epoch}",
    )


def _prediction_context(
    config: Stage3Config,
    spec: Any,
    *,
    split: str,
    fold: int | None,
) -> list[dict[str, Any]]:
    rows = iter_rows(config, spec, None if split == "test" else (fold,))
    return [
        {
            "source_fold": source_fold,
            "source_row": source_row,
            **{name: row[name] for name in spec.identity_columns},
            **{name: row[name] for name in spec.condition_columns},
            "target": float(row[spec.target_column]),
        }
        for source_fold, source_row, row in rows
    ]


def _write_predictions(
    config: Stage3Config,
    prepared: Mapping[str, Any],
    *,
    split: str,
    fold: int | None,
    tasks: Sequence[str],
    raw_targets: Mapping[str, torch.Tensor],
    fold_predictions: Mapping[int, Mapping[str, torch.Tensor]],
    destination: Path,
) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for task in tasks:
        spec = prepared["registry"][task]
        contexts = _prediction_context(
            config, spec, split=split, fold=fold
        )
        target_values = raw_targets[task].double()
        if len(contexts) != len(target_values) or any(
            int(row["source_row"]) != int(source_row)
            for row, source_row in zip(
                contexts,
                Stage3TaskDataset(
                    config.data.artifacts_dir,
                    1 if split == "test" else int(fold),
                    task,
                    split,
                ).source_rows.tolist(),
                strict=True,
            )
        ):
            raise ValueError(f"Stage 3 prediction/source-row mismatch: {task}")
        if contexts and not torch.allclose(
            torch.tensor([row["target"] for row in contexts], dtype=torch.float64),
            target_values,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(f"Stage 3 prediction/source-target mismatch: {task}")
        fields = [
            "source_row",
            *(("source_fold",) if split == "valid" else ()),
            *spec.identity_columns,
            *spec.condition_columns,
            "target",
        ]
        output_rows: list[dict[str, Any]] = []
        if split == "test":
            fields.extend(f"prediction_fold{current}" for current in range(1, 6))
            fields.extend(("prediction_ensemble", "absolute_error_ensemble"))
            ensemble = torch.stack(
                [fold_predictions[current][task].double() for current in range(1, 6)]
            ).mean(dim=0)
            for index, context in enumerate(contexts):
                row = {name: context[name] for name in fields if name in context}
                for current in range(1, 6):
                    row[f"prediction_fold{current}"] = float(
                        fold_predictions[current][task][index]
                    )
                row["prediction_ensemble"] = float(ensemble[index])
                row["absolute_error_ensemble"] = abs(
                    float(ensemble[index]) - float(context["target"])
                )
                output_rows.append(row)
        else:
            fields.extend(("prediction", "absolute_error"))
            assert fold is not None
            predictions = fold_predictions[fold][task].double()
            for index, context in enumerate(contexts):
                row = {name: context[name] for name in fields if name in context}
                row["prediction"] = float(predictions[index])
                row["absolute_error"] = abs(
                    float(predictions[index]) - float(context["target"])
                )
                output_rows.append(row)
        manifest = write_prediction_csv(
            destination / f"{sanitize_task_id(task)}.csv", output_rows, fields
        )
        manifest["path"] = f"predictions/{sanitize_task_id(task)}.csv"
        manifest["task"] = task
        manifests.append(manifest)
    return manifests


def evaluate_checkpoints(
    config: Stage3Config,
    checkpoint_dir: str | Path,
    *,
    split: str,
    ensemble_folds: bool,
    checkpoint_epoch: int | None = None,
    task_subset: Sequence[str] | None = None,
    fold: int | None = None,
    predictions_dir: str | Path | None = None,
    reporting_study_id: str | None = None,
    expected_evaluation_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if split not in {"valid", "test"}:
        raise ValueError("Stage 3 evaluation split must be valid or test")
    if split == "test" and not ensemble_folds:
        raise ValueError("Stage 3 test evaluation requires five-fold ensemble")
    if split == "valid" and (fold not in range(1, 6) or ensemble_folds):
        raise ValueError("Stage 3 validation requires exactly one --fold")
    device = resolve_device(config.training.device)
    if config.training.amp_dtype == "bf16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("Stage 3 BF16 evaluation requires capable CUDA")
    prepared = load_prepared_stage3(config)
    enabled = tuple(
        task for task, spec in prepared["registry"].items() if spec.enabled
    )
    tasks = tuple(task_subset) if task_subset is not None else enabled
    if not tasks or set(tasks) - set(enabled):
        raise ValueError("Stage 3 evaluation task subset is invalid")
    taskwise_refined = checkpoint_epoch is None
    epoch = config.training.epochs if checkpoint_epoch is None else checkpoint_epoch
    if epoch <= 0:
        raise ValueError("Stage 3 checkpoint epoch must be positive")
    folds = range(1, 6) if split == "test" else (fold,)
    root = Path(checkpoint_dir)
    embeddings = prepared["objects"]["embeddings"].float()
    progress = ProgressReporter()
    evaluation_progress = progress.bar(
        total=len(folds) * len(tasks),
        desc=f"Stage 3 {split} evaluation",
        unit="task",
    )    
    fold_results: dict[str, Any] = {}
    ensemble_predictions: dict[str, list[torch.Tensor]] = {task: [] for task in tasks}
    raw_targets: dict[str, torch.Tensor] = {}
    normalizations: dict[str, list[dict[str, Any]]] = {task: [] for task in tasks}
    checkpoint_identities: list[Mapping[str, Any]] = []
    model_state_hashes: list[str] = []
    selection_manifest_hashes: list[str] = []
    raw_fold_predictions: dict[int, dict[str, torch.Tensor]] = {}
    try:
        for current_fold in folds:
            assert current_fold is not None
            path = (
                _refined_path(root, current_fold)
                if taskwise_refined
                else _checkpoint_path(root, current_fold, epoch)
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing Stage 3 checkpoint: {path}")
            model, checkpoint = _load_model(
                config, prepared, path, current_fold, epoch, device,
                taskwise_refined=taskwise_refined,
            )
            checkpoint_identities.append(checkpoint["training_identity"])
            model_state_hashes.append(checkpoint["model_state_hash"])
            if taskwise_refined:
                selection_manifest_hashes.append(
                    _validate_refinement_manifest(path, checkpoint, epoch)
                )
            per_task: dict[str, Any] = {}
            raw_fold_predictions[current_fold] = {}
            for task in tasks:
                dataset = Stage3TaskDataset(
                    config.data.artifacts_dir, current_fold, task, split
                )
                normalization = checkpoint["normalization"][task]
                normalized, raw, normalized_targets = _predict(
                    model,
                    task,
                    dataset,
                    embeddings,
                    normalization,
                    config,
                    device,
                    progress_bar=evaluation_progress,
                    fold=current_fold,
                    fold_count=len(folds),
                )
                per_task[task] = regression_metrics(
                    normalized, normalized_targets, normalization
                )
                ensemble_predictions[task].append(raw)
                raw_fold_predictions[current_fold][task] = raw
                normalizations[task].append(normalization)
                if task in raw_targets and not torch.equal(
                    raw_targets[task], dataset.raw_targets
                ):
                    raise ValueError(
                        f"Stage 3 test target order differs across folds: {task}"
                    )
                raw_targets[task] = dataset.raw_targets.float()
                evaluation_progress.update(1)
            fold_results[f"fold{current_fold}"] = {
                "tasks": per_task,
                **_macro(per_task, prepared["registry"]),
            }
    finally:
        evaluation_progress.close()
    evaluation_identity = build_stage3_evaluation_identity(
        prepared_identity=metadata_identity(
            prepared["metadata"], "prepared", context="Stage 3 prepared artifact"
        ),
        checkpoint_identities=checkpoint_identities,
        model_state_hashes=model_state_hashes,
        selection_manifest_hashes=selection_manifest_hashes,
        split=split,
        fold=fold,
        checkpoint_epoch=None if taskwise_refined else epoch,
        model_selector="taskwise_refined" if taskwise_refined else "epoch_checkpoint",
        tasks=tasks,
        ensemble_folds=ensemble_folds,
    )
    if expected_evaluation_identity is not None:
        require_compatible_identity(
            expected_evaluation_identity,
            evaluation_identity,
            context="Stage 3 run-directory evaluation identity",
        )
    if split == "valid":
        prediction_manifests = (
            _write_predictions(
                config,
                prepared,
                split=split,
                fold=fold,
                tasks=tasks,
                raw_targets=raw_targets,
                fold_predictions=raw_fold_predictions,
                destination=Path(predictions_dir),
            )
            if predictions_dir is not None
            else []
        )
        comparison = _reporting_comparison(
            config, prepared, split=split, expected_tasks=enabled
        )
        result = {
            "split": split,
            "checkpoint_epoch": None if taskwise_refined else epoch,
            "model_selector": "taskwise_refined" if taskwise_refined else "epoch_checkpoint",
            **next(iter(fold_results.values())),
        }
        default_study_id = _default_reporting_study_id(
            prepared["metadata"],
            "taskwise-refined" if taskwise_refined else f"epoch{epoch}",
        )
        result["reporting"] = reporting_block(
            model_id="ilume",
            model_display_name="ILUME",
            benchmark="stage3_property",
            protocol={
                "split": "valid",
                "fold": fold,
                "folds": list(range(1, 6)),
                "ensemble": False,
                "expected_tasks": list(enabled),
                "checkpoint_epoch": None if taskwise_refined else epoch,
                "model_selector": "taskwise_refined" if taskwise_refined else "epoch_checkpoint",
            },
            comparison=comparison,
            study_id=reporting_study_id or default_study_id,
            predictions=prediction_manifests,
        )
        return result
    ensemble = {
        task: _raw_ensemble_metrics(
            torch.stack(predictions).mean(dim=0),
            raw_targets[task],
            sum(
                float(item["target"]["scale"])
                for item in normalizations[task]
            )
            / len(normalizations[task]),
        )
        for task, predictions in ensemble_predictions.items()
    }
    prediction_manifests = (
        _write_predictions(
            config,
            prepared,
            split=split,
            fold=fold,
            tasks=tasks,
            raw_targets=raw_targets,
            fold_predictions=raw_fold_predictions,
            destination=Path(predictions_dir),
        )
        if predictions_dir is not None
        else []
    )
    expected_test = tuple(
        task
        for task in enabled
        if next(iter_rows(config, prepared["registry"][task], None), None) is not None
    )
    comparison = _reporting_comparison(
        config, prepared, split=split, expected_tasks=expected_test
    )
    result = {
        "split": split,
        "checkpoint_epoch": None if taskwise_refined else epoch,
        "model_selector": "taskwise_refined" if taskwise_refined else "epoch_checkpoint",
        "folds": fold_results,
        "ensemble": {
            "tasks": ensemble,
            **_macro(ensemble, prepared["registry"]),
        },
    }
    default_study_id = _default_reporting_study_id(
        prepared["metadata"],
        "taskwise-refined" if taskwise_refined else f"epoch{epoch}",
    )
    result["reporting"] = reporting_block(
        model_id="ilume",
        model_display_name="ILUME",
        benchmark="stage3_property",
        protocol={
            "split": "test",
            "folds": list(range(1, 6)),
            "ensemble": True,
            "expected_tasks": list(expected_test),
            "enabled_tasks": list(enabled),
            "checkpoint_epoch": None if taskwise_refined else epoch,
            "model_selector": "taskwise_refined" if taskwise_refined else "epoch_checkpoint",
        },
        comparison=comparison,
        study_id=reporting_study_id or default_study_id,
        predictions=prediction_manifests,
    )
    return result


def resolve_stage3_evaluation_identity(
    config: Stage3Config,
    checkpoint_dir: str | Path,
    *,
    split: str,
    ensemble_folds: bool,
    checkpoint_epoch: int | None = None,
    task_subset: Sequence[str] | None = None,
    fold: int | None = None,
) -> dict[str, Any]:
    """Resolve and validate the semantic identity of an evaluation request."""
    if split not in {"valid", "test"}:
        raise ValueError("Stage 3 evaluation split must be valid or test")
    if split == "test" and not ensemble_folds:
        raise ValueError("Stage 3 test evaluation requires five-fold ensemble")
    if split == "valid" and (fold not in range(1, 6) or ensemble_folds):
        raise ValueError("Stage 3 validation requires exactly one --fold")
    prepared = load_prepared_stage3(config)
    enabled = tuple(
        task for task, spec in prepared["registry"].items() if spec.enabled
    )
    tasks = tuple(task_subset) if task_subset is not None else enabled
    if not tasks or set(tasks) - set(enabled):
        raise ValueError("Stage 3 evaluation task subset is invalid")
    taskwise_refined = checkpoint_epoch is None
    epoch = config.training.epochs if checkpoint_epoch is None else checkpoint_epoch
    if epoch <= 0:
        raise ValueError("Stage 3 checkpoint epoch must be positive")
    folds = range(1, 6) if split == "test" else (fold,)
    identities: list[Mapping[str, Any]] = []
    state_hashes: list[str] = []
    selection_manifest_hashes: list[str] = []
    for current_fold in folds:
        assert current_fold is not None
        path = (
            _refined_path(Path(checkpoint_dir), current_fold)
            if taskwise_refined
            else _checkpoint_path(Path(checkpoint_dir), current_fold, epoch)
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing Stage 3 checkpoint: {path}")
        _, checkpoint = _load_model(
            config, prepared, path, current_fold, epoch, torch.device("cpu"),
            taskwise_refined=taskwise_refined,
        )
        identities.append(checkpoint["training_identity"])
        state_hashes.append(checkpoint["model_state_hash"])
        if taskwise_refined:
            selection_manifest_hashes.append(
                _validate_refinement_manifest(path, checkpoint, epoch)
            )
    return build_stage3_evaluation_identity(
        prepared_identity=metadata_identity(
            prepared["metadata"], "prepared", context="Stage 3 prepared artifact"
        ),
        checkpoint_identities=identities,
        model_state_hashes=state_hashes,
        selection_manifest_hashes=selection_manifest_hashes,
        split=split,
        fold=fold,
        checkpoint_epoch=None if taskwise_refined else epoch,
        model_selector="taskwise_refined" if taskwise_refined else "epoch_checkpoint",
        tasks=tasks,
        ensemble_folds=ensemble_folds,
    )


__all__ = [
    "evaluate_checkpoints",
    "resolve_stage3_evaluation_identity",
    "resolve_stage3_reporting_study_id",
]
