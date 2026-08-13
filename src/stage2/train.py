from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from common.io import atomic_json, atomic_torch_save, sha256_file
from common.progress import ProgressReporter
from common.training import (
    canonical_json_sha256,
    capture_rng_state,
    cosine_warmup,
    resolve_device,
    restore_rng_state,
    seed_everything,
)
from stage1.masking import MultimodalPacker
from stage1.model import load_stage1_model
from .config import (
    STAGE2_TASKS,
    Stage2Config,
    stage2_config_from_checkpoint_dict,
)
from .data import (
    Stage2EntityDataset,
    Stage2TaskDataset,
    build_stage2_batch,
    epoch_batch_schedule,
    task_batch_counts,
)
from .model import Stage2ObjectModel, stage2_optimizer_groups
from .prepare import load_teacher_embeddings


STAGE2_CHECKPOINT_VERSION = 1
STAGE2_CHECKPOINT_KIND = "ilume_stage2_object"


def _config_hash(config: Stage2Config) -> str:
    return canonical_json_sha256(config.to_dict())


def task_compensation_scale(
    task_weight: float,
    total_epoch_batches: int,
    batch_rows: int,
    task_rows: int,
) -> float:
    if task_weight <= 0.0:
        raise ValueError("Stage 2 task weight must be positive")
    if total_epoch_batches <= 0 or batch_rows <= 0 or task_rows <= 0:
        raise ValueError("Stage 2 compensation counts must be positive")
    return task_weight * total_epoch_batches * batch_rows / task_rows


def _training_geometry(
    config: Stage2Config,
    datasets: dict[str, Stage2TaskDataset],
) -> tuple[dict[str, int], int, int, int]:
    batch_counts = task_batch_counts(datasets, config.training.batch_size)
    epoch_batches = sum(batch_counts.values())
    steps_per_epoch = math.ceil(
        epoch_batches / config.training.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * config.training.epochs
    unfreeze_step = steps_per_epoch * config.training.backbone_frozen_epochs
    return batch_counts, steps_per_epoch, total_steps, unfreeze_step


def _scheduler_lambdas(
    config: Stage2Config,
    total_steps: int,
    unfreeze_step: int,
):
    def backbone(step: int) -> float:
        if step < unfreeze_step:
            return 0.0
        return cosine_warmup(
            step - unfreeze_step,
            total_steps - unfreeze_step,
            config.training.warmup_fraction,
        )

    def new_modules(step: int) -> float:
        return cosine_warmup(
            step, total_steps, config.training.warmup_fraction
        )

    return backbone, new_modules


def _initial_backbone_reference(
    model: Stage2ObjectModel,
) -> dict[str, torch.Tensor]:
    parameter_ids = {id(parameter) for parameter in model.backbone_parameters()}
    return {
        name: parameter.detach().cpu().to(torch.float16).clone()
        for name, parameter in model.backbone.named_parameters()
        if id(parameter) in parameter_ids
    }


@torch.no_grad()
def _relative_parameter_drift(
    model: Stage2ObjectModel,
    reference: dict[str, torch.Tensor],
) -> float:
    delta_squared = 0.0
    reference_squared = 0.0
    for name, parameter in model.backbone.named_parameters():
        if name not in reference:
            continue
        current = parameter.detach().float().cpu()
        initial = reference[name].float()
        delta_squared += float(torch.square(current - initial).sum())
        reference_squared += float(torch.square(initial).sum())
    return math.sqrt(delta_squared / max(reference_squared, 1.0e-30))


def _inverse_targets(
    values: torch.Tensor,
    columns: Sequence[str],
    scalers: dict[str, Any],
) -> torch.Tensor:
    result = values.detach().float().cpu().clone()
    for index, name in enumerate(columns):
        stats = scalers["targets"][name]
        result[:, index] = (
            result[:, index] * float(stats["scale"]) + float(stats["mean"])
        )
    return result


@torch.no_grad()
def evaluate_stage2(
    model: Stage2ObjectModel,
    valid_datasets: dict[str, Stage2TaskDataset],
    entity_dataset: Stage2EntityDataset,
    packer: MultimodalPacker,
    teacher_embeddings: torch.Tensor,
    scalers: dict[str, Any],
    config: Stage2Config,
    device: torch.device,
    *,
    initial_backbone: dict[str, torch.Tensor] | None = None,
) -> dict[str, float | int | str]:
    was_training = model.training
    model.eval()
    result: dict[str, float | int | str] = {"validation_scope": "full"}
    task_normalized_maes: dict[str, float] = {}
    for task in STAGE2_TASKS:
        dataset = valid_datasets[task]
        output_count = len(dataset.target_columns)
        normalized_absolute = torch.zeros(output_count, dtype=torch.float64)
        absolute = torch.zeros(output_count, dtype=torch.float64)
        squared = torch.zeros(output_count, dtype=torch.float64)
        target_sum = torch.zeros(output_count, dtype=torch.float64)
        target_squared = torch.zeros(output_count, dtype=torch.float64)
        target_counts = torch.zeros(output_count, dtype=torch.long)
        teacher_sum = 0.0
        cosine_sum = 0.0
        row_count = 0
        for start in range(0, len(dataset), config.training.batch_size):
            selected = torch.arange(
                start, min(len(dataset), start + config.training.batch_size)
            )
            batch = build_stage2_batch(
                dataset,
                selected,
                entity_dataset,
                packer,
                teacher_embeddings,
                scalers,
            ).to(device)
            output = model(
                task,
                batch.entities,
                batch.entity_positions,
                batch.conditions,
                batch.targets,
                batch.target_mask,
                batch.teacher_embeddings,
                lambda_teacher=config.loss.lambda_teacher,
            )
            size = int(selected.numel())
            mask = batch.target_mask.detach().cpu()
            mask_float = mask.double()
            normalized_difference = (
                output.predictions.detach().cpu()
                - batch.targets.detach().cpu()
            ).double()
            normalized_absolute += (
                torch.abs(normalized_difference) * mask_float
            ).sum(dim=0)
            predicted_raw = _inverse_targets(
                output.predictions, dataset.target_columns, scalers
            ).double()
            target_raw = dataset.targets[selected].double()
            difference = predicted_raw - target_raw
            absolute += (torch.abs(difference) * mask_float).sum(dim=0)
            squared += (torch.square(difference) * mask_float).sum(dim=0)
            target_sum += (target_raw * mask_float).sum(dim=0)
            target_squared += (torch.square(target_raw) * mask_float).sum(dim=0)
            target_counts += mask.sum(dim=0)
            teacher_sum += float(output.teacher_loss.detach().cpu()) * size
            cosine = F.cosine_similarity(
                output.student_slots, output.teacher_slots, dim=-1
            ).mean(dim=1)
            cosine_sum += float(cosine.sum().detach().cpu())
            row_count += size
        valid_targets = target_counts > 0
        if not bool(valid_targets.any()):
            raise ValueError(f"Stage 2 validation has no targets for {task}")
        normalized_by_target = normalized_absolute / target_counts.clamp_min(1)
        task_normalized = float(normalized_by_target[valid_targets].mean())
        task_normalized_maes[task] = task_normalized
        prefix = f"valid_{task}"
        result[f"{prefix}_rows"] = row_count
        result[f"{prefix}_normalized_mae"] = task_normalized
        result[f"{prefix}_teacher_mse"] = teacher_sum / row_count
        result[f"{prefix}_teacher_cosine"] = cosine_sum / row_count
        for index, name in enumerate(dataset.target_columns):
            count = int(target_counts[index])
            result[f"{prefix}_{name}_count"] = count
            if count == 0:
                result[f"{prefix}_{name}_mae"] = float("nan")
                result[f"{prefix}_{name}_rmse"] = float("nan")
                result[f"{prefix}_{name}_r2"] = float("nan")
                continue
            result[f"{prefix}_{name}_mae"] = float(absolute[index] / count)
            result[f"{prefix}_{name}_rmse"] = math.sqrt(
                float(squared[index] / count)
            )
            total_variance = float(
                target_squared[index]
                - torch.square(target_sum[index]) / count
            )
            result[f"{prefix}_{name}_r2"] = (
                1.0 - float(squared[index]) / total_variance
                if total_variance > 0.0
                else float("nan")
            )
    result["valid_macro_normalized_mae"] = float(
        np.mean(list(task_normalized_maes.values()))
    )
    result["valid_weighted_macro_normalized_mae"] = sum(
        config.loss.task_weights[task] * value
        for task, value in task_normalized_maes.items()
    )
    if initial_backbone is not None:
        result["backbone_relative_l2_drift"] = _relative_parameter_drift(
            model, initial_backbone
        )
    if was_training:
        model.train()
    return result


def _append_metric(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=True) + "\n")


def _reconcile_metrics_for_resume(path: Path, completed_epoch: int) -> None:
    if not path.is_file():
        raise FileNotFoundError("Stage 2 resume requires metrics.jsonl")
    retained: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("epoch", 0)) <= completed_epoch:
            retained.append(json.dumps(row, sort_keys=True, allow_nan=True))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "" if not retained else "\n".join(retained) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_epoch_checkpoint(
    path: Path,
    *,
    model: Stage2ObjectModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    completed_epoch: int,
    global_optimizer_step: int,
    config: Stage2Config,
    data_metadata_hash: str,
    teacher_embeddings_hash: str,
    model_contract: dict[str, int],
    task_rows: dict[str, int],
    task_batches: dict[str, int],
    validation: dict[str, Any],
) -> None:
    if path.exists():
        raise FileExistsError(f"Stage 2 checkpoint already exists: {path}")
    atomic_torch_save(
        path,
        {
            "format_version": STAGE2_CHECKPOINT_VERSION,
            "kind": STAGE2_CHECKPOINT_KIND,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "completed_epoch": completed_epoch,
            "global_optimizer_step": global_optimizer_step,
            "rng": capture_rng_state(),
            "config": config.to_dict(),
            "config_hash": _config_hash(config),
            "model_contract": model_contract,
            "data_metadata_hash": data_metadata_hash,
            "teacher_embeddings_hash": teacher_embeddings_hash,
            "task_rows": task_rows,
            "task_batches": task_batches,
            "task_weights": config.loss.task_weights,
            "validation": validation,
        },
    )


def run_stage2_training(
    config: Stage2Config,
    *,
    output_dir: str | Path,
    resume_from: str | Path | None = None,
) -> list[dict[str, Any]]:
    config.validate()
    seed_everything(config.data.seed)
    device = resolve_device(config.training.device)
    loaded = load_stage1_model(
        config.initialization.checkpoint,
        config.data.pretrain_artifacts_dir,
        device=device,
        backbone_dropout=0.0,
    )
    model = Stage2ObjectModel(
        loaded.model,
        object_layers=config.model.object_layers,
        object_ffn_dim=config.model.object_ffn_dim,
        dropout=config.model.dropout,
    ).to(device)
    initial_backbone = _initial_backbone_reference(model)
    entity_dataset = Stage2EntityDataset(
        config.data.artifacts_dir, config.data.shard_cache_size
    )
    data_metadata_path = config.data.artifacts_dir / "metadata.json"
    data_metadata = json.loads(data_metadata_path.read_text(encoding="utf-8"))
    if data_metadata.get("pretrain_artifact_hash") != loaded.artifact_hash:
        raise ValueError("Stage 2 data artifact does not match the checkpoint")
    if data_metadata.get("model_contract") != model.model_contract:
        raise ValueError("Stage 2 data model contract does not match Stage 1")
    teacher_embeddings = load_teacher_embeddings(
        config,
        expected_count=len(entity_dataset),
        expected_dim=loaded.config.model.d_model,
    )
    teacher_path = config.data.artifacts_dir / "teachers" / "embeddings.pt"
    teacher_embeddings_hash = sha256_file(teacher_path)
    train_datasets = {
        task: Stage2TaskDataset(config.data.artifacts_dir, task, "train")
        for task in STAGE2_TASKS
    }
    valid_datasets = {
        task: Stage2TaskDataset(config.data.artifacts_dir, task, "valid")
        for task in STAGE2_TASKS
    }
    task_rows = {task: len(dataset) for task, dataset in train_datasets.items()}
    task_batches, steps_per_epoch, total_steps, unfreeze_step = _training_geometry(
        config, train_datasets
    )
    total_epoch_batches = sum(task_batches.values())
    packer = MultimodalPacker(loaded.vocabulary)
    model.set_backbone_trainable(config.training.backbone_frozen_epochs == 0)
    optimizer = torch.optim.AdamW(
        stage2_optimizer_groups(
            model,
            backbone_learning_rate=config.training.backbone_learning_rate,
            new_module_learning_rate=(
                config.training.stage2_new_module_learning_rate
            ),
            weight_decay=config.training.weight_decay,
        )
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _scheduler_lambdas(config, total_steps, unfreeze_step),
    )
    fp16 = config.training.amp_dtype == "fp16" and device.type == "cuda"
    amp_enabled = config.training.amp_dtype != "none" and device.type == "cuda"
    amp_dtype = torch.float16 if fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    data_metadata_hash = sha256_file(data_metadata_path)
    completed_epoch = 0
    global_optimizer_step = 0
    if resume_from is None:
        if metrics_path.is_file() and metrics_path.stat().st_size:
            raise FileExistsError(
                f"Stage 2 output already contains metrics: {metrics_path}"
            )
    else:
        checkpoint = torch.load(
            Path(resume_from), map_location="cpu", weights_only=False
        )
        if (
            checkpoint.get("format_version") != STAGE2_CHECKPOINT_VERSION
            or checkpoint.get("kind") != STAGE2_CHECKPOINT_KIND
        ):
            raise ValueError("Unsupported Stage 2 object checkpoint")
        checkpoint_config = stage2_config_from_checkpoint_dict(checkpoint["config"])
        if checkpoint_config.to_dict() != config.to_dict():
            raise ValueError("Stage 2 checkpoint config does not match")
        expected = {
            "config_hash": _config_hash(config),
            "model_contract": model.model_contract,
            "data_metadata_hash": data_metadata_hash,
            "teacher_embeddings_hash": teacher_embeddings_hash,
            "task_rows": task_rows,
            "task_batches": task_batches,
            "task_weights": config.loss.task_weights,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"Stage 2 checkpoint {key} does not match")
        completed_epoch = int(checkpoint["completed_epoch"])
        global_optimizer_step = int(checkpoint["global_optimizer_step"])
        if not 1 <= completed_epoch <= config.training.epochs:
            raise ValueError("Stage 2 checkpoint completed epoch is invalid")
        if global_optimizer_step != completed_epoch * steps_per_epoch:
            raise ValueError("Stage 2 checkpoint optimizer step does not match epoch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng"])
        _reconcile_metrics_for_resume(metrics_path, completed_epoch)
    if completed_epoch == config.training.epochs:
        raise ValueError("Stage 2 checkpoint already completed all epochs")

    reporter = ProgressReporter()
    results: list[dict[str, Any]] = []
    for epoch in range(completed_epoch + 1, config.training.epochs + 1):
        backbone_trainable = epoch > config.training.backbone_frozen_epochs
        model.set_backbone_trainable(backbone_trainable)
        model.train()
        schedule = epoch_batch_schedule(
            train_datasets,
            config.training.batch_size,
            seed=config.data.seed,
            epoch=epoch,
        )
        accumulation = config.training.gradient_accumulation_steps
        with reporter.bar(
            total=len(schedule),
            desc=f"Stage 2 object epoch {epoch}",
            unit="batch",
        ) as progress:
            for window_start in range(0, len(schedule), accumulation):
                window = schedule[window_start : window_start + accumulation]
                optimizer.zero_grad(set_to_none=True)
                for descriptor in window:
                    batch = build_stage2_batch(
                        train_datasets[descriptor.task],
                        descriptor.indices,
                        entity_dataset,
                        packer,
                        teacher_embeddings,
                        data_metadata["scalers"],
                    ).to(device)
                    batch_rows = int(descriptor.indices.numel())
                    compensation = task_compensation_scale(
                        config.loss.task_weights[descriptor.task],
                        total_epoch_batches,
                        batch_rows,
                        task_rows[descriptor.task],
                    )
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        output = model(
                            descriptor.task,
                            batch.entities,
                            batch.entity_positions,
                            batch.conditions,
                            batch.targets,
                            batch.target_mask,
                            batch.teacher_embeddings,
                            lambda_teacher=config.loss.lambda_teacher,
                        )
                        weighted_loss = output.total_loss * compensation
                        backward_loss = weighted_loss / len(window)
                    if not torch.isfinite(weighted_loss):
                        raise RuntimeError(
                            f"Non-finite Stage 2 loss in epoch {epoch}"
                        )
                    scaler.scale(backward_loss).backward()
                    row = {
                        "event": "stage2_train_batch",
                        "epoch": epoch,
                        "global_optimizer_step": global_optimizer_step,
                        "task": descriptor.task,
                        "batch_rows": batch_rows,
                        "task_scale": compensation,
                        "loss_property": float(
                            output.property_loss.detach().cpu()
                        ),
                        "loss_teacher": float(output.teacher_loss.detach().cpu()),
                        "loss_total": float(output.total_loss.detach().cpu()),
                        "loss_weighted": float(weighted_loss.detach().cpu()),
                        "backbone_trainable": int(backbone_trainable),
                        "backbone_learning_rate": optimizer.param_groups[0]["lr"],
                        "stage2_new_module_learning_rate": (
                            optimizer.param_groups[1]["lr"]
                        ),
                    }
                    _append_metric(metrics_path, row)
                    results.append(row)
                    progress.update(1)
                if config.training.max_grad_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for group in optimizer.param_groups
                            for parameter in group["params"]
                            if parameter.requires_grad
                        ],
                        config.training.max_grad_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_optimizer_step += 1
        validation = evaluate_stage2(
            model,
            valid_datasets,
            entity_dataset,
            packer,
            teacher_embeddings,
            data_metadata["scalers"],
            config,
            device,
            initial_backbone=initial_backbone,
        )
        validation.update(
            {
                "event": "stage2_full_validation",
                "epoch": epoch,
                "global_optimizer_step": global_optimizer_step,
            }
        )
        _append_metric(metrics_path, validation)
        reporter.emit_json(validation)
        _save_epoch_checkpoint(
            output_dir / f"checkpoint_epoch_{epoch:05d}.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            completed_epoch=epoch,
            global_optimizer_step=global_optimizer_step,
            config=config,
            data_metadata_hash=data_metadata_hash,
            teacher_embeddings_hash=teacher_embeddings_hash,
            model_contract=model.model_contract,
            task_rows=task_rows,
            task_batches=task_batches,
            validation=validation,
        )
        completed_epoch = epoch
    final_metrics = dict(validation)
    final_metrics["event"] = "stage2_training_complete"
    atomic_json(output_dir / "final_metrics.json", final_metrics)
    return results
