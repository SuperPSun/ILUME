from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from common.io import sha256_file
from common.training import canonical_json_sha256, resolve_device
from .config import Stage3Config
from .data import Stage3TaskDataset
from .model import Stage3SparseModel
from .prepare import load_prepared_stage3
from .train import STAGE3_CHECKPOINT_KIND, STAGE3_CHECKPOINT_VERSION, regression_metrics


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
        "kind": STAGE3_CHECKPOINT_KIND,
        "format_version": STAGE3_CHECKPOINT_VERSION,
        "stage": "stage3",
        "fold": fold,
        "completed_epoch": epoch,
        "stage2_checkpoint_sha256": prepared["metadata"]["stage2_checkpoint_sha256"],
        "resolved_registry": {
            task_id: spec.to_dict() for task_id, spec in prepared["registry"].items()
        },
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Stage 3 evaluation checkpoint mismatch: {key}")
    plan = checkpoint.get("resolved_training_plan")
    if not isinstance(plan, dict) or checkpoint.get(
        "resolved_training_plan_hash"
    ) != canonical_json_sha256(plan):
        raise ValueError("Stage 3 checkpoint resolved plan hash mismatch")
    if any(
        plan.get("model", {}).get(name) != value
        for name, value in asdict(config.model).items()
    ):
        raise ValueError("Stage 3 checkpoint model config mismatch")
    if plan.get("data", {}).get("source_hashes") != prepared["metadata"]["source_hashes"]:
        raise ValueError("Stage 3 checkpoint data provenance mismatch")
    if plan.get("data", {}).get("artifact_metadata_sha256") != sha256_file(
        config.data.artifacts_dir / "metadata.json"
    ):
        raise ValueError("Stage 3 checkpoint artifact identity mismatch")
    if plan.get("normalization_hash") != canonical_json_sha256(
        checkpoint.get("normalization")
    ):
        raise ValueError("Stage 3 checkpoint normalization mismatch")
    d_model = int(prepared["objects"]["embeddings"].shape[1])
    model = Stage3SparseModel(config.model, prepared["registry"], d_model)
    if checkpoint.get("ownership_manifest") != model.ownership_manifest():
        raise ValueError("Stage 3 checkpoint ownership mismatch")
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized_predictions: list[torch.Tensor] = []
    target_stats = normalization["target"]
    for start in range(0, len(dataset), config.training.microbatch_size):
        indices = torch.arange(start, min(len(dataset), start + config.training.microbatch_size))
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
    fold_results: dict[str, Any] = {}
    ensemble_predictions: dict[str, list[torch.Tensor]] = {task: [] for task in tasks}
    raw_targets: dict[str, torch.Tensor] = {}
    normalizations: dict[str, list[dict[str, Any]]] = {task: [] for task in tasks}
    for current_fold in folds:
        assert current_fold is not None
        path = _checkpoint_path(root, current_fold, epoch)
        if not path.is_file():
            raise FileNotFoundError(f"Missing Stage 3 checkpoint: {path}")
        model, checkpoint = _load_model(
            config, prepared, path, current_fold, epoch, device
        )
        per_task: dict[str, Any] = {}
        for task in tasks:
            dataset = Stage3TaskDataset(
                config.data.artifacts_dir, current_fold, task, split
            )
            normalization = checkpoint["normalization"][task]
            normalized, raw, normalized_targets = _predict(
                model, task, dataset, embeddings, normalization, config, device
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
        fold_results[f"fold{current_fold}"] = {
            "tasks": per_task,
            **_macro(per_task, prepared["registry"]),
        }
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


__all__ = ["evaluate_checkpoints"]
