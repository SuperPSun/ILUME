from __future__ import annotations

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
from .config import Stage3Config
from .data import Stage3TaskDataset
from .model import Stage3SparseModel
from .prepare import load_prepared_stage3
from .train import STAGE3_CHECKPOINT_KIND, STAGE3_CHECKPOINT_VERSION, regression_metrics
from .identity import (
    build_stage3_evaluation_identity,
    build_stage3_training_identity,
    metadata_identity,
)


def _checkpoint_path(root: Path, fold: int, epoch: int) -> Path:
    filename = f"checkpoint_epoch_{epoch:05d}.pt"
    nested = root / f"fold{fold}" / filename
    return nested if nested.is_file() else root / filename


def _load_model(
    config: Stage3Config,
    prepared: Mapping[str, Any],
    checkpoint_path: Path,
    fold: int,
    epoch: int,
    device: torch.device,
) -> tuple[Stage3SparseModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = {
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "kind": STAGE3_CHECKPOINT_KIND,
        "format_version": STAGE3_CHECKPOINT_VERSION,
        "stage": "stage3",
        "fold": fold,
        "completed_epoch": epoch,
        "stage2_encoder_identity": metadata_identity(
            prepared["metadata"], "stage2_encoder", context="Stage 3 prepared artifact"
        )["hash"],
        "resolved_registry": {
            task_id: spec.to_dict() for task_id, spec in prepared["registry"].items()
        },
    }
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
    if checkpoint.get("model_state_hash") != tensor_state_hash(
        "stage3.model-state", checkpoint["model"]
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


def evaluate_checkpoints(
    config: Stage3Config,
    checkpoint_dir: str | Path,
    *,
    split: str,
    ensemble_folds: bool,
    checkpoint_epoch: int | None = None,
    task_subset: Sequence[str] | None = None,
    fold: int | None = None,
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
    try:
        for current_fold in folds:
            assert current_fold is not None
            path = _checkpoint_path(root, current_fold, epoch)
            if not path.is_file():
                raise FileNotFoundError(f"Missing Stage 3 checkpoint: {path}")
            model, checkpoint = _load_model(
                config, prepared, path, current_fold, epoch, device
            )
            checkpoint_identities.append(checkpoint["training_identity"])
            model_state_hashes.append(checkpoint["model_state_hash"])
            per_task: dict[str, Any] = {}
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
        split=split,
        fold=fold,
        checkpoint_epoch=epoch,
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
        return {
            "split": split,
            "checkpoint_epoch": epoch,
            **next(iter(fold_results.values())),
        }
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
    return {
        "split": split,
        "checkpoint_epoch": epoch,
        "folds": fold_results,
        "ensemble": {
            "tasks": ensemble,
            **_macro(ensemble, prepared["registry"]),
        },
    }


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
    epoch = config.training.epochs if checkpoint_epoch is None else checkpoint_epoch
    if epoch <= 0:
        raise ValueError("Stage 3 checkpoint epoch must be positive")
    folds = range(1, 6) if split == "test" else (fold,)
    identities: list[Mapping[str, Any]] = []
    state_hashes: list[str] = []
    for current_fold in folds:
        assert current_fold is not None
        path = _checkpoint_path(Path(checkpoint_dir), current_fold, epoch)
        if not path.is_file():
            raise FileNotFoundError(f"Missing Stage 3 checkpoint: {path}")
        _, checkpoint = _load_model(
            config, prepared, path, current_fold, epoch, torch.device("cpu")
        )
        identities.append(checkpoint["training_identity"])
        state_hashes.append(checkpoint["model_state_hash"])
    return build_stage3_evaluation_identity(
        prepared_identity=metadata_identity(
            prepared["metadata"], "prepared", context="Stage 3 prepared artifact"
        ),
        checkpoint_identities=identities,
        model_state_hashes=state_hashes,
        split=split,
        fold=fold,
        checkpoint_epoch=epoch,
        tasks=tasks,
        ensemble_folds=ensemble_folds,
    )


__all__ = ["evaluate_checkpoints", "resolve_stage3_evaluation_identity"]
