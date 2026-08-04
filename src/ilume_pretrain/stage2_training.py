from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .masking import MultimodalPacker
from .progress import ProgressReporter
from .stage2_config import STAGE2_TASKS, Stage2Config
from .stage2_data import (
    Stage2EntityDataset,
    Stage2TaskDataset,
    TaskBlockSampler,
    TaskCursor,
    build_stage2_batch,
)
from .stage2_model import (
    Stage2AlignmentModel,
    load_stage1_model,
    sha256_file,
    stage2_optimizer_groups,
)
from .stage2_prepare import load_teacher_embeddings, resolve_device


STAGE2_CHECKPOINT_VERSION = 1


def _config_hash(config: Stage2Config) -> str:
    payload = config.to_dict()
    payload["training"]["resume_from"] = None
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _lr_lambda(step: int, total_steps: int, warmup_fraction: float) -> float:
    warmup_steps = max(1, round(total_steps * warmup_fraction))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _initial_backbone_reference(
    model: Stage2AlignmentModel,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().to(torch.float16).clone()
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad
    }


@torch.no_grad()
def _relative_parameter_drift(
    model: Stage2AlignmentModel,
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


def _validation_indices(
    dataset: Stage2TaskDataset,
    config: Stage2Config,
    *,
    full: bool,
) -> torch.Tensor:
    if full or dataset.task != "transfer_organic":
        return torch.arange(len(dataset))
    count = min(len(dataset), config.data.transfer_validation_limit)
    generator = torch.Generator().manual_seed(config.data.seed + 500000)
    return torch.randperm(len(dataset), generator=generator)[:count]


@torch.no_grad()
def evaluate_stage2(
    model: Stage2AlignmentModel,
    valid_datasets: dict[str, Stage2TaskDataset],
    entity_dataset: Stage2EntityDataset,
    packer: MultimodalPacker,
    teacher_embeddings: torch.Tensor,
    scalers: dict[str, Any],
    config: Stage2Config,
    device: torch.device,
    *,
    full: bool,
    initial_backbone: dict[str, torch.Tensor] | None = None,
) -> dict[str, float | int | str]:
    was_training = model.training
    model.eval()
    result: dict[str, float | int | str] = {
        "validation_scope": "full" if full else "fast"
    }
    task_normalized_maes: list[float] = []
    for task in STAGE2_TASKS:
        dataset = valid_datasets[task]
        indices = _validation_indices(dataset, config, full=full)
        output_count = len(dataset.target_columns)
        normalized_absolute = torch.zeros(output_count, dtype=torch.float64)
        absolute = torch.zeros(output_count, dtype=torch.float64)
        squared = torch.zeros(output_count, dtype=torch.float64)
        target_sum = torch.zeros(output_count, dtype=torch.float64)
        target_squared = torch.zeros(output_count, dtype=torch.float64)
        alignment_sum = 0.0
        cosine_sum = 0.0
        row_count = 0
        for start in range(0, len(indices), config.training.batch_size):
            selected = indices[start : start + config.training.batch_size]
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
                batch.teacher_embeddings,
                lambda_alignment=config.loss.lambda_alignment,
            )
            size = int(selected.numel())
            normalized_absolute += torch.abs(
                output.predictions.detach().cpu() - batch.targets.detach().cpu()
            ).double().sum(dim=0)
            predicted_raw = _inverse_targets(
                output.predictions, dataset.target_columns, scalers
            ).double()
            target_raw = dataset.targets[selected].double()
            difference = predicted_raw - target_raw
            absolute += torch.abs(difference).sum(dim=0)
            squared += torch.square(difference).sum(dim=0)
            target_sum += target_raw.sum(dim=0)
            target_squared += torch.square(target_raw).sum(dim=0)
            alignment_sum += float(output.alignment_loss.detach().cpu()) * size
            cosine = F.cosine_similarity(
                output.student_slots,
                output.teacher_slots,
                dim=-1,
            ).mean(dim=1)
            cosine_sum += float(cosine.sum().detach().cpu())
            row_count += size
        normalized_by_target = normalized_absolute / row_count
        task_normalized = float(normalized_by_target.mean())
        task_normalized_maes.append(task_normalized)
        prefix = f"valid_{task}"
        result[f"{prefix}_rows"] = row_count
        result[f"{prefix}_normalized_mae"] = task_normalized
        result[f"{prefix}_alignment_mse"] = alignment_sum / row_count
        result[f"{prefix}_alignment_cosine"] = cosine_sum / row_count
        for index, name in enumerate(dataset.target_columns):
            mae = float(absolute[index] / row_count)
            rmse = math.sqrt(float(squared[index] / row_count))
            total_variance = float(
                target_squared[index]
                - torch.square(target_sum[index]) / row_count
            )
            r2 = (
                1.0 - float(squared[index]) / total_variance
                if total_variance > 0.0
                else float("nan")
            )
            result[f"{prefix}_{name}_mae"] = mae
            result[f"{prefix}_{name}_rmse"] = rmse
            result[f"{prefix}_{name}_r2"] = r2
    result["valid_macro_normalized_mae"] = float(
        np.mean(task_normalized_maes)
    )
    if initial_backbone is not None:
        result["backbone_relative_l2_drift"] = _relative_parameter_drift(
            model, initial_backbone
        )
    if was_training:
        model.train()
    return result


def _atomic_torch_save(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_checkpoint(
    path: Path,
    *,
    model: Stage2AlignmentModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    global_step: int,
    micro_step: int,
    cursors: dict[str, TaskCursor],
    task_counts: dict[str, int],
    best_metric: float,
    validations_without_improvement: int,
    config: Stage2Config,
    config_hash: str,
    data_metadata_hash: str,
    teacher_embeddings_hash: str,
    checkpoint_hash: str,
) -> None:
    _atomic_torch_save(
        path,
        {
            "format_version": STAGE2_CHECKPOINT_VERSION,
            "kind": "ilume_stage2_alignment",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": global_step,
            "micro_step": micro_step,
            "cursors": {
                task: cursor.state_dict() for task, cursor in cursors.items()
            },
            "task_block": {
                "block_size": config.sampling.block_size,
                "block_index": global_step // config.sampling.block_size,
                "offset": global_step % config.sampling.block_size,
            },
            "task_counts": task_counts,
            "rng": _rng_state(),
            "best_metric": best_metric,
            "validations_without_improvement": validations_without_improvement,
            "config": config.to_dict(),
            "config_hash": config_hash,
            "data_metadata_hash": data_metadata_hash,
            "teacher_embeddings_hash": teacher_embeddings_hash,
            "pretrain_checkpoint_hash": checkpoint_hash,
        },
    )


def run_stage2_training(
    config: Stage2Config,
) -> list[dict[str, float | int | str]]:
    config.validate()
    random.seed(config.data.seed)
    np.random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.data.seed)
    device = resolve_device(config.training.device)
    loaded = load_stage1_model(
        config.initialization.checkpoint,
        config.data.pretrain_artifacts_dir,
        device=device,
        backbone_dropout=0.0,
    )
    model = Stage2AlignmentModel(
        loaded.model,
        head_dropout=config.model.head_dropout,
    ).to(device)
    initial_backbone = _initial_backbone_reference(model)
    entity_dataset = Stage2EntityDataset(
        config.data.artifacts_dir,
        config.data.shard_cache_size,
    )
    data_metadata_path = config.data.artifacts_dir / "metadata.json"
    data_metadata = json.loads(data_metadata_path.read_text(encoding="utf-8"))
    if data_metadata.get("pretrain_artifact_hash") != loaded.artifact_hash:
        raise ValueError("Stage 2 data artifact does not match the checkpoint")
    teacher_embeddings = load_teacher_embeddings(
        config,
        checkpoint_hash=loaded.checkpoint_hash,
        expected_count=len(entity_dataset),
        expected_dim=loaded.config.model.d_model,
    )
    teacher_path = (
        config.data.artifacts_dir
        / "teachers"
        / loaded.checkpoint_hash
        / "embeddings.pt"
    )
    teacher_embeddings_hash = sha256_file(teacher_path)
    train_datasets = {
        task: Stage2TaskDataset(config.data.artifacts_dir, task, "train")
        for task in STAGE2_TASKS
    }
    valid_datasets = {
        task: Stage2TaskDataset(config.data.artifacts_dir, task, "valid")
        for task in STAGE2_TASKS
    }
    packer = MultimodalPacker(loaded.vocabulary)
    cursors = {
        task: TaskCursor(
            len(dataset), config.data.seed + 10000 * (index + 1)
        )
        for index, (task, dataset) in enumerate(train_datasets.items())
    }
    task_sampler = TaskBlockSampler(
        config.sampling.probabilities,
        config.sampling.block_size,
        config.data.seed,
    )
    optimizer = torch.optim.AdamW(
        stage2_optimizer_groups(
            model,
            backbone_learning_rate=config.training.backbone_learning_rate,
            head_learning_rate=config.training.head_learning_rate,
            weight_decay=config.training.weight_decay,
        )
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_lambda(
            step,
            config.training.max_steps,
            config.training.warmup_fraction,
        ),
    )
    fp16 = config.training.amp_dtype == "fp16" and device.type == "cuda"
    amp_enabled = config.training.amp_dtype != "none" and device.type == "cuda"
    amp_dtype = torch.float16 if fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    output_dir = config.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    current_config_hash = _config_hash(config)
    data_metadata_hash = sha256_file(data_metadata_path)
    global_step = 0
    micro_step = 0
    best_metric = float("inf")
    validations_without_improvement = 0
    task_counts = {task: 0 for task in STAGE2_TASKS}
    if config.training.resume_from is None:
        if metrics_path.is_file() and metrics_path.stat().st_size:
            raise FileExistsError(
                f"Stage 2 output already contains metrics: {metrics_path}"
            )
    else:
        checkpoint = torch.load(
            config.training.resume_from,
            map_location="cpu",
            weights_only=False,
        )
        if (
            checkpoint.get("format_version") != STAGE2_CHECKPOINT_VERSION
            or checkpoint.get("kind") != "ilume_stage2_alignment"
        ):
            raise ValueError("Unsupported Stage 2 checkpoint format")
        expected = {
            "config_hash": current_config_hash,
            "data_metadata_hash": data_metadata_hash,
            "teacher_embeddings_hash": teacher_embeddings_hash,
            "pretrain_checkpoint_hash": loaded.checkpoint_hash,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"Stage 2 checkpoint {key} does not match")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        global_step = int(checkpoint["global_step"])
        micro_step = int(checkpoint["micro_step"])
        if micro_step != (
            global_step * config.training.gradient_accumulation_steps
        ):
            raise ValueError("Stage 2 checkpoint micro_step does not match")
        expected_block = {
            "block_size": config.sampling.block_size,
            "block_index": global_step // config.sampling.block_size,
            "offset": global_step % config.sampling.block_size,
        }
        if checkpoint.get("task_block") != expected_block:
            raise ValueError("Stage 2 checkpoint task block position does not match")
        for task, cursor in cursors.items():
            cursor.load_state_dict(checkpoint["cursors"][task])
        task_counts = {
            task: int(value)
            for task, value in checkpoint["task_counts"].items()
        }
        best_metric = float(checkpoint["best_metric"])
        validations_without_improvement = int(
            checkpoint["validations_without_improvement"]
        )
        _restore_rng_state(checkpoint["rng"])
    if global_step > config.training.max_steps:
        raise ValueError("Stage 2 checkpoint is beyond configured max_steps")

    reporter = ProgressReporter()
    results: list[dict[str, float | int | str]] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    stopped_early = False
    with reporter.bar(
        total=config.training.max_steps,
        initial=global_step,
        desc="Stage 2 alignment",
        unit="step",
    ) as progress:
        while global_step < config.training.max_steps:
            task = task_sampler.task_for_step(global_step)
            supervised_total = 0.0
            alignment_total = 0.0
            total_loss_value = 0.0
            for _ in range(config.training.gradient_accumulation_steps):
                indices = cursors[task].next_indices(config.training.batch_size)
                batch = build_stage2_batch(
                    train_datasets[task],
                    indices,
                    entity_dataset,
                    packer,
                    teacher_embeddings,
                    data_metadata["scalers"],
                ).to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    output = model(
                        task,
                        batch.entities,
                        batch.entity_positions,
                        batch.conditions,
                        batch.targets,
                        batch.teacher_embeddings,
                        lambda_alignment=config.loss.lambda_alignment,
                    )
                    scaled_loss = (
                        output.total_loss
                        / config.training.gradient_accumulation_steps
                    )
                if not torch.isfinite(output.total_loss):
                    raise RuntimeError(
                        f"Non-finite Stage 2 loss at micro step {micro_step}"
                    )
                scaler.scale(scaled_loss).backward()
                supervised_total += float(output.supervised_loss.detach().cpu())
                alignment_total += float(output.alignment_loss.detach().cpu())
                total_loss_value += float(output.total_loss.detach().cpu())
                micro_step += 1
            if config.training.max_grad_norm > 0.0:
                scaler.unscale_(optimizer)
                trainable_parameters = [
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                ]
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    config.training.max_grad_norm,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            task_counts[task] += (
                config.training.batch_size
                * config.training.gradient_accumulation_steps
            )
            divisor = config.training.gradient_accumulation_steps
            row: dict[str, float | int | str] = {
                "global_step": global_step,
                "micro_step": micro_step,
                "task": task,
                "loss": total_loss_value / divisor,
                "loss_supervised": supervised_total / divisor,
                "loss_alignment": alignment_total / divisor,
                "lambda_alignment": config.loss.lambda_alignment,
                "backbone_learning_rate": optimizer.param_groups[0]["lr"],
                "head_learning_rate": optimizer.param_groups[1]["lr"],
                **{
                    f"samples_{name}": count
                    for name, count in task_counts.items()
                },
            }
            should_validate = (
                global_step % config.training.validation_interval_steps == 0
                or global_step == config.training.max_steps
            )
            improved = False
            if should_validate:
                row.update(
                    evaluate_stage2(
                        model,
                        valid_datasets,
                        entity_dataset,
                        packer,
                        teacher_embeddings,
                        data_metadata["scalers"],
                        config,
                        device,
                        full=False,
                        initial_backbone=initial_backbone,
                    )
                )
                current_metric = float(row["valid_macro_normalized_mae"])
                improved = (
                    current_metric
                    < best_metric - config.training.early_stopping_min_delta
                )
                if improved:
                    best_metric = current_metric
                    validations_without_improvement = 0
                else:
                    validations_without_improvement += 1
                row["best_valid_macro_normalized_mae"] = best_metric
                row["validations_without_improvement"] = (
                    validations_without_improvement
                )

            serialized = json.dumps(row, sort_keys=True, allow_nan=True)
            reporter.emit_json(row)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
            results.append(row)
            progress.set_postfix(
                {
                    "task": task,
                    "loss": f"{float(row['loss']):.4f}",
                    "align": f"{float(row['loss_alignment']):.4f}",
                },
                refresh=False,
            )
            progress.update(1)
            if should_validate:
                checkpoint_path = (
                    output_dir / f"checkpoint_step_{global_step:08d}.pt"
                )
                _save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    global_step=global_step,
                    micro_step=micro_step,
                    cursors=cursors,
                    task_counts=task_counts,
                    best_metric=best_metric,
                    validations_without_improvement=(
                        validations_without_improvement
                    ),
                    config=config,
                    config_hash=current_config_hash,
                    data_metadata_hash=data_metadata_hash,
                    teacher_embeddings_hash=teacher_embeddings_hash,
                    checkpoint_hash=loaded.checkpoint_hash,
                )
                if improved:
                    _save_checkpoint(
                        output_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        global_step=global_step,
                        micro_step=micro_step,
                        cursors=cursors,
                        task_counts=task_counts,
                        best_metric=best_metric,
                        validations_without_improvement=(
                            validations_without_improvement
                        ),
                        config=config,
                        config_hash=current_config_hash,
                        data_metadata_hash=data_metadata_hash,
                        teacher_embeddings_hash=teacher_embeddings_hash,
                        checkpoint_hash=loaded.checkpoint_hash,
                    )
                checkpoints = sorted(output_dir.glob("checkpoint_step_*.pt"))
                for stale in checkpoints[: -config.training.keep_last_checkpoints]:
                    stale.unlink()
                if (
                    validations_without_improvement
                    >= config.training.early_stopping_patience
                ):
                    stopped_early = True
                    break

    best_path = output_dir / "best.pt"
    if best_path.is_file():
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best["model"])
    final_metrics = evaluate_stage2(
        model,
        valid_datasets,
        entity_dataset,
        packer,
        teacher_embeddings,
        data_metadata["scalers"],
        config,
        device,
        full=True,
        initial_backbone=initial_backbone,
    )
    final_metrics.update(
        {
            "event": "stage2_full_validation",
            "global_step": global_step,
            "stopped_early": int(stopped_early),
        }
    )
    (output_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics, sort_keys=True, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    reporter.emit_json(final_metrics)
    return results


def with_stage2_overrides(
    config: Stage2Config,
    *,
    lambda_alignment: float | None = None,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> Stage2Config:
    loss = config.loss
    training = config.training
    if lambda_alignment is not None:
        loss = replace(loss, lambda_alignment=lambda_alignment)
    if output_dir is not None:
        training = replace(training, output_dir=Path(output_dir))
    if resume_from is not None:
        training = replace(training, resume_from=Path(resume_from))
    updated = replace(config, loss=loss, training=training)
    updated.validate()
    return updated
