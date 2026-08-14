from __future__ import annotations

import json
import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Sequence

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
    PackedStage2Window,
    Stage2BatchDescriptor,
    Stage2DeviceTaskData,
    Stage2EntityDataset,
    Stage2TaskDataset,
    epoch_batch_schedule,
    pack_stage2_window,
    task_batch_counts,
)
from .model import Stage2ForwardOutput, Stage2ObjectModel, stage2_optimizer_groups
from .prepare import load_teacher_embeddings
from .runtime import configure_stage2_math


STAGE2_CHECKPOINT_VERSION = 2
STAGE2_CHECKPOINT_KIND = "ilume_stage2_object"


def _config_hash(config: Stage2Config) -> str:
    return canonical_json_sha256(config.experiment_dict())


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

    return backbone, new_modules, new_modules


def _schedule_windows(
    schedule: Sequence[Stage2BatchDescriptor], accumulation: int
) -> list[tuple[Stage2BatchDescriptor, ...]]:
    return [
        tuple(schedule[start : start + accumulation])
        for start in range(0, len(schedule), accumulation)
    ]


def _ordered_packed_windows(
    windows: Sequence[tuple[Stage2BatchDescriptor, ...]],
    datasets: dict[str, Stage2TaskDataset],
    entities: Stage2EntityDataset,
    packer: MultimodalPacker,
    *,
    workers: int,
    prefetch_windows: int,
    pin_memory: bool,
) -> Iterator[PackedStage2Window]:
    capacity = max(1, workers * prefetch_windows)
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="stage2-packer",
    ) as executor:
        pending: deque[Future[PackedStage2Window]] = deque()
        iterator = iter(windows)
        for _ in range(capacity):
            try:
                window = next(iterator)
            except StopIteration:
                break
            pending.append(
                executor.submit(
                    pack_stage2_window,
                    window,
                    datasets,
                    entities,
                    packer,
                    pin_memory=pin_memory,
                )
            )
        while pending:
            yield pending.popleft().result()
            try:
                window = next(iterator)
            except StopIteration:
                continue
            pending.append(
                executor.submit(
                    pack_stage2_window,
                    window,
                    datasets,
                    entities,
                    packer,
                    pin_memory=pin_memory,
                )
            )


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


def _optimizer_implementation(device: torch.device) -> str:
    return "fused" if device.type == "cuda" else "single_tensor"


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
    teacher_cache_identity: str,
    model_contract: dict[str, int],
    task_rows: dict[str, int],
    task_batches: dict[str, int],
    validation: dict[str, Any],
    optimizer_implementation: str,
    math_contract: dict[str, Any],
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
            "teacher_cache_identity": teacher_cache_identity,
            "task_rows": task_rows,
            "task_batches": task_batches,
            "task_weights": config.loss.task_weights,
            "optimizer_implementation": optimizer_implementation,
            "math_contract": math_contract,
            "validation": validation,
        },
    )


def _descriptor_tensors(
    descriptor: Stage2BatchDescriptor,
    data: Stage2DeviceTaskData,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = descriptor.indices.to(device)
    return (
        indices,
        data.entity_indices[indices],
        data.conditions[indices],
        data.targets[indices],
        data.target_mask[indices],
    )


def _frozen_output(
    model: Stage2ObjectModel,
    descriptor: Stage2BatchDescriptor,
    task_data: Stage2DeviceTaskData,
    teacher_embeddings: torch.Tensor,
    entity_roles: torch.Tensor,
    config: Stage2Config,
    device: torch.device,
) -> Stage2ForwardOutput:
    _, global_slots, conditions, targets, target_mask = _descriptor_tensors(
        descriptor, task_data, device
    )
    teacher_slots = teacher_embeddings[global_slots]
    return model.forward_from_slots(
        descriptor.task,
        teacher_slots,
        entity_roles[global_slots],
        conditions,
        targets,
        target_mask,
        teacher_slots,
        lambda_teacher=config.loss.lambda_teacher,
        teacher_loss_is_zero=True,
    )


def _unfrozen_outputs(
    model: Stage2ObjectModel,
    packed: PackedStage2Window,
    device_data: dict[str, Stage2DeviceTaskData],
    teacher_embeddings: torch.Tensor,
    config: Stage2Config,
    device: torch.device,
) -> list[tuple[Stage2BatchDescriptor, Stage2ForwardOutput]]:
    entities = packed.entities.to(device, non_blocking=device.type == "cuda")
    unique_ids = packed.unique_entity_ids.to(
        device, non_blocking=device.type == "cuda"
    )
    positions = [
        value.to(device, non_blocking=device.type == "cuda")
        for value in packed.entity_positions
    ]
    student_unique = model.encode_entities(entities)
    teacher_unique = teacher_embeddings[unique_ids]
    outputs: list[tuple[Stage2BatchDescriptor, Stage2ForwardOutput]] = []
    for descriptor, entity_positions in zip(
        packed.descriptors, positions, strict=True
    ):
        task_data = device_data[descriptor.task]
        _, _, conditions, targets, target_mask = _descriptor_tensors(
            descriptor, task_data, device
        )
        outputs.append(
            (
                descriptor,
                model.forward_from_slots(
                    descriptor.task,
                    student_unique[entity_positions],
                    entities.roles[entity_positions],
                    conditions,
                    targets,
                    target_mask,
                    teacher_unique[entity_positions],
                    lambda_teacher=config.loss.lambda_teacher,
                ),
            )
        )
    return outputs


@torch.inference_mode()
def evaluate_stage2(
    model: Stage2ObjectModel,
    valid_datasets: dict[str, Stage2TaskDataset],
    device_data: dict[str, Stage2DeviceTaskData],
    entity_dataset: Stage2EntityDataset,
    packer: MultimodalPacker,
    teacher_embeddings: torch.Tensor,
    entity_roles: torch.Tensor,
    scalers: dict[str, Any],
    config: Stage2Config,
    device: torch.device,
    *,
    backbone_frozen: bool,
) -> dict[str, float | int | str]:
    was_training = model.training
    model.eval()
    result: dict[str, float | int | str] = {
        "validation_scope": "full",
        "validation_backbone_frozen": int(backbone_frozen),
    }
    accumulators: dict[str, dict[str, torch.Tensor | int]] = {}
    descriptors: list[Stage2BatchDescriptor] = []
    for task in STAGE2_TASKS:
        dataset = valid_datasets[task]
        output_count = len(dataset.target_columns)
        accumulators[task] = {
            "normalized_absolute": torch.zeros(
                output_count, dtype=torch.float64, device=device
            ),
            "absolute": torch.zeros(
                output_count, dtype=torch.float64, device=device
            ),
            "squared": torch.zeros(
                output_count, dtype=torch.float64, device=device
            ),
            "target_sum": torch.zeros(
                output_count, dtype=torch.float64, device=device
            ),
            "target_squared": torch.zeros(
                output_count, dtype=torch.float64, device=device
            ),
            "target_counts": torch.zeros(
                output_count, dtype=torch.long, device=device
            ),
            "teacher_sum": torch.zeros((), dtype=torch.float64, device=device),
            "cosine_sum": torch.zeros((), dtype=torch.float64, device=device),
            "row_count": 0,
        }
        descriptors.extend(
            Stage2BatchDescriptor(
                task,
                torch.arange(
                    start,
                    min(len(dataset), start + config.training.batch_size),
                ),
            )
            for start in range(0, len(dataset), config.training.batch_size)
        )

    packed_iterator: Iterator[PackedStage2Window] | None = None
    if not backbone_frozen:
        packed_iterator = _ordered_packed_windows(
            [(descriptor,) for descriptor in descriptors],
            valid_datasets,
            entity_dataset,
            packer,
            workers=config.training.packing_workers,
            prefetch_windows=config.training.packing_prefetch_windows,
            pin_memory=device.type == "cuda",
        )

    for descriptor in descriptors:
        if backbone_frozen:
            output = _frozen_output(
                model,
                descriptor,
                device_data[descriptor.task],
                teacher_embeddings,
                entity_roles,
                config,
                device,
            )
        else:
            if packed_iterator is None:
                raise AssertionError("Missing Stage 2 validation packer")
            packed = next(packed_iterator)
            output = _unfrozen_outputs(
                model,
                packed,
                device_data,
                teacher_embeddings,
                config,
                device,
            )[0][1]
        dataset = valid_datasets[descriptor.task]
        task_data = device_data[descriptor.task]
        selected = descriptor.indices.to(device)
        mask = task_data.target_mask[selected]
        mask_float = mask.to(torch.float64)
        normalized_difference = (
            output.predictions.float() - task_data.targets[selected]
        ).to(torch.float64)
        raw_prediction = output.predictions.float().clone()
        for column, name in enumerate(dataset.target_columns):
            stats = scalers["targets"][name]
            raw_prediction[:, column] = (
                raw_prediction[:, column] * float(stats["scale"])
                + float(stats["mean"])
            )
        raw_target = task_data.raw_targets
        if raw_target is None:
            raise AssertionError("Stage 2 validation raw targets are missing")
        target_values = raw_target[selected].to(torch.float64)
        difference = raw_prediction.to(torch.float64) - target_values
        accumulator = accumulators[descriptor.task]
        accumulator["normalized_absolute"] += (
            torch.abs(normalized_difference) * mask_float
        ).sum(dim=0)
        accumulator["absolute"] += (
            torch.abs(difference) * mask_float
        ).sum(dim=0)
        accumulator["squared"] += (
            torch.square(difference) * mask_float
        ).sum(dim=0)
        accumulator["target_sum"] += (target_values * mask_float).sum(dim=0)
        accumulator["target_squared"] += (
            torch.square(target_values) * mask_float
        ).sum(dim=0)
        accumulator["target_counts"] += mask.sum(dim=0)
        size = int(descriptor.indices.numel())
        accumulator["teacher_sum"] += output.teacher_loss.to(torch.float64) * size
        accumulator["cosine_sum"] += F.cosine_similarity(
            output.student_slots, output.teacher_slots, dim=-1
        ).mean(dim=1).sum().to(torch.float64)
        accumulator["row_count"] = int(accumulator["row_count"]) + size

    task_normalized_maes: dict[str, float] = {}
    for task in STAGE2_TASKS:
        dataset = valid_datasets[task]
        values = accumulators[task]
        tensors = torch.cat(
            [
                values["normalized_absolute"],
                values["absolute"],
                values["squared"],
                values["target_sum"],
                values["target_squared"],
                values["target_counts"].to(torch.float64),
                values["teacher_sum"].reshape(1),
                values["cosine_sum"].reshape(1),
            ]
        ).cpu()
        count = len(dataset.target_columns)
        offset = 0
        normalized_absolute = tensors[offset : offset + count]
        offset += count
        absolute = tensors[offset : offset + count]
        offset += count
        squared = tensors[offset : offset + count]
        offset += count
        target_sum = tensors[offset : offset + count]
        offset += count
        target_squared = tensors[offset : offset + count]
        offset += count
        target_counts = tensors[offset : offset + count].to(torch.long)
        offset += count
        teacher_sum = float(tensors[offset])
        cosine_sum = float(tensors[offset + 1])
        valid_targets = target_counts > 0
        if not bool(valid_targets.any()):
            raise ValueError(f"Stage 2 validation has no targets for {task}")
        normalized_by_target = normalized_absolute / target_counts.clamp_min(1)
        task_normalized = float(normalized_by_target[valid_targets].mean())
        task_normalized_maes[task] = task_normalized
        prefix = f"valid_{task}"
        row_count = int(values["row_count"])
        result[f"{prefix}_rows"] = row_count
        result[f"{prefix}_normalized_mae"] = task_normalized
        result[f"{prefix}_teacher_mse"] = teacher_sum / row_count
        result[f"{prefix}_teacher_cosine"] = cosine_sum / row_count
        for index, name in enumerate(dataset.target_columns):
            supervised = int(target_counts[index])
            result[f"{prefix}_{name}_count"] = supervised
            if supervised == 0:
                result[f"{prefix}_{name}_mae"] = float("nan")
                result[f"{prefix}_{name}_rmse"] = float("nan")
                result[f"{prefix}_{name}_r2"] = float("nan")
                continue
            result[f"{prefix}_{name}_mae"] = float(absolute[index] / supervised)
            result[f"{prefix}_{name}_rmse"] = math.sqrt(
                float(squared[index] / supervised)
            )
            total_variance = float(
                target_squared[index]
                - torch.square(target_sum[index]) / supervised
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
    if was_training:
        model.train()
    return result


def run_stage2_training(
    config: Stage2Config,
    *,
    output_dir: str | Path,
    resume_from: str | Path | None = None,
) -> list[dict[str, Any]]:
    config.validate()
    seed_everything(config.data.seed)
    device = resolve_device(config.training.device)
    math_contract = configure_stage2_math(device)
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
    entity_dataset = Stage2EntityDataset(config.data.artifacts_dir)
    data_metadata_path = config.data.artifacts_dir / "metadata.json"
    data_metadata = json.loads(data_metadata_path.read_text(encoding="utf-8"))
    if data_metadata.get("pretrain_artifact_hash") != loaded.artifact_hash:
        raise ValueError("Stage 2 data artifact does not match the checkpoint")
    if data_metadata.get("model_contract") != model.model_contract:
        raise ValueError("Stage 2 data model contract does not match Stage 1")
    teacher_cpu, teacher_metadata = load_teacher_embeddings(
        config,
        loaded,
        data_metadata,
        math_contract,
        expected_count=len(entity_dataset),
        expected_dim=loaded.config.model.d_model,
    )
    teacher_embeddings = teacher_cpu.to(device)
    teacher_embeddings_hash = teacher_metadata["embeddings_hash"]
    teacher_cache_identity = teacher_metadata["identity"]
    train_datasets = {
        task: Stage2TaskDataset(config.data.artifacts_dir, task, "train")
        for task in STAGE2_TASKS
    }
    valid_datasets = {
        task: Stage2TaskDataset(config.data.artifacts_dir, task, "valid")
        for task in STAGE2_TASKS
    }
    train_device = {
        task: Stage2DeviceTaskData.from_dataset(dataset, device)
        for task, dataset in train_datasets.items()
    }
    valid_device = {
        task: Stage2DeviceTaskData.from_dataset(dataset, device)
        for task, dataset in valid_datasets.items()
    }
    entity_roles = torch.tensor(
        [int(entry["role_id"]) for entry in entity_dataset.entries],
        dtype=torch.long,
        device=device,
    )
    task_rows = {task: len(dataset) for task, dataset in train_datasets.items()}
    task_batches, steps_per_epoch, total_steps, unfreeze_step = _training_geometry(
        config, train_datasets
    )
    total_epoch_batches = sum(task_batches.values())
    packer = MultimodalPacker(loaded.vocabulary)
    model.set_backbone_trainable(config.training.backbone_frozen_epochs == 0)
    optimizer_implementation = _optimizer_implementation(device)
    optimizer = torch.optim.AdamW(
        stage2_optimizer_groups(
            model,
            backbone_learning_rate=config.training.backbone_learning_rate,
            object_encoder_learning_rate=(
                config.training.object_encoder_learning_rate
            ),
            task_head_learning_rate=config.training.task_head_learning_rate,
            weight_decay=config.training.weight_decay,
        ),
        fused=device.type == "cuda",
        foreach=False if device.type != "cuda" else None,
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
            raise ValueError(
                "Unsupported Stage 2 object checkpoint; Object v1 is not migrated"
            )
        checkpoint_config = stage2_config_from_checkpoint_dict(checkpoint["config"])
        if checkpoint_config.experiment_dict() != config.experiment_dict():
            raise ValueError("Stage 2 checkpoint experiment config does not match")
        expected = {
            "config_hash": _config_hash(config),
            "model_contract": model.model_contract,
            "data_metadata_hash": data_metadata_hash,
            "teacher_embeddings_hash": teacher_embeddings_hash,
            "teacher_cache_identity": teacher_cache_identity,
            "task_rows": task_rows,
            "task_batches": task_batches,
            "task_weights": config.loss.task_weights,
            "optimizer_implementation": optimizer_implementation,
            "math_contract": math_contract,
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
        windows = _schedule_windows(
            schedule, config.training.gradient_accumulation_steps
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        interval_started = epoch_started
        interval_batches = 0
        interval_values = torch.zeros(
            (len(STAGE2_TASKS), 4), dtype=torch.float64, device=device
        )
        interval_counts = torch.zeros(
            len(STAGE2_TASKS), dtype=torch.long, device=device
        )
        interval_finite = torch.ones((), dtype=torch.bool, device=device)

        packed_windows: Iterator[PackedStage2Window] | None = None
        if backbone_trainable:
            packed_windows = _ordered_packed_windows(
                windows,
                train_datasets,
                entity_dataset,
                packer,
                workers=config.training.packing_workers,
                prefetch_windows=config.training.packing_prefetch_windows,
                pin_memory=device.type == "cuda",
            )

        with reporter.bar(
            total=len(schedule),
            desc=f"Stage 2 object v2 epoch {epoch}",
            unit="batch",
        ) as progress:
            for window in windows:
                optimizer.zero_grad(set_to_none=True)
                if backbone_trainable:
                    if packed_windows is None:
                        raise AssertionError("Missing Stage 2 packed window iterator")
                    packed = next(packed_windows)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        batch_outputs = _unfrozen_outputs(
                            model,
                            packed,
                            train_device,
                            teacher_embeddings,
                            config,
                            device,
                        )
                else:
                    batch_outputs = []
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        for descriptor in window:
                            batch_outputs.append(
                                (
                                    descriptor,
                                    _frozen_output(
                                        model,
                                        descriptor,
                                        train_device[descriptor.task],
                                        teacher_embeddings,
                                        entity_roles,
                                        config,
                                        device,
                                    ),
                                )
                            )
                weighted_losses: list[torch.Tensor] = []
                for descriptor, output in batch_outputs:
                    batch_rows = int(descriptor.indices.numel())
                    compensation = task_compensation_scale(
                        config.loss.task_weights[descriptor.task],
                        total_epoch_batches,
                        batch_rows,
                        task_rows[descriptor.task],
                    )
                    weighted_loss = output.total_loss * compensation
                    weighted_losses.append(weighted_loss)
                    task_index = STAGE2_TASKS.index(descriptor.task)
                    interval_values[task_index] += torch.stack(
                        (
                            output.property_loss.detach().to(torch.float64),
                            output.teacher_loss.detach().to(torch.float64),
                            output.total_loss.detach().to(torch.float64),
                            weighted_loss.detach().to(torch.float64),
                        )
                    )
                    interval_counts[task_index] += 1
                    interval_finite &= torch.isfinite(weighted_loss.detach())
                    interval_batches += 1
                    progress.update(1)
                backward_loss = torch.stack(weighted_losses).sum() / len(window)
                scaler.scale(backward_loss).backward()
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
                used_learning_rates = tuple(
                    float(group["lr"]) for group in optimizer.param_groups
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_optimizer_step += 1

                should_log = (
                    interval_batches >= config.training.log_every_batches
                    or window is windows[-1]
                )
                if should_log:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    if not bool(interval_finite.cpu()):
                        raise RuntimeError(
                            f"Non-finite Stage 2 loss in epoch {epoch}"
                        )
                    elapsed = max(time.perf_counter() - interval_started, 1.0e-12)
                    values_cpu = interval_values.cpu()
                    counts_cpu = interval_counts.cpu()
                    row: dict[str, Any] = {
                        "event": "stage2_train_interval",
                        "phase": (
                            "unfrozen" if backbone_trainable else "frozen"
                        ),
                        "epoch": epoch,
                        "global_optimizer_step": global_optimizer_step,
                        "task_batches": interval_batches,
                        "batches_per_second": interval_batches / elapsed,
                        "backbone_trainable": int(backbone_trainable),
                        "backbone_learning_rate": used_learning_rates[0],
                        "object_encoder_learning_rate": used_learning_rates[1],
                        "task_head_learning_rate": used_learning_rates[2],
                    }
                    for task_index, task in enumerate(STAGE2_TASKS):
                        count = int(counts_cpu[task_index])
                        if count == 0:
                            continue
                        for column, name in enumerate(
                            (
                                "loss_property",
                                "loss_teacher",
                                "loss_total",
                                "loss_weighted",
                            )
                        ):
                            row[f"{task}_{name}"] = float(
                                values_cpu[task_index, column] / count
                            )
                    _append_metric(metrics_path, row)
                    results.append(row)
                    interval_started = time.perf_counter()
                    interval_batches = 0
                    interval_values.zero_()
                    interval_counts.zero_()
                    interval_finite.fill_(True)

        validation_started = time.perf_counter()
        validation = evaluate_stage2(
            model,
            valid_datasets,
            valid_device,
            entity_dataset,
            packer,
            teacher_embeddings,
            entity_roles,
            data_metadata["scalers"],
            config,
            device,
            backbone_frozen=not backbone_trainable,
        )
        validation_seconds = time.perf_counter() - validation_started
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(device)
        else:
            peak_memory = 0
        validation.update(
            {
                "event": "stage2_full_validation",
                "epoch": epoch,
                "global_optimizer_step": global_optimizer_step,
                "epoch_wall_seconds": time.perf_counter() - epoch_started,
                "validation_wall_seconds": validation_seconds,
                "peak_cuda_memory_bytes": peak_memory,
                "phase": "unfrozen" if backbone_trainable else "frozen",
                "teacher_cache_reused": 1,
                "freeze_boundary_epoch": (
                    config.training.backbone_frozen_epochs
                ),
            }
        )
        _append_metric(metrics_path, validation)
        reporter.emit_json(validation)
        checkpoint_started = time.perf_counter()
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
            teacher_cache_identity=teacher_cache_identity,
            model_contract=model.model_contract,
            task_rows=task_rows,
            task_batches=task_batches,
            validation=validation,
            optimizer_implementation=optimizer_implementation,
            math_contract=math_contract,
        )
        checkpoint_row = {
            "event": "stage2_checkpoint_complete",
            "epoch": epoch,
            "global_optimizer_step": global_optimizer_step,
            "checkpoint_wall_seconds": time.perf_counter() - checkpoint_started,
        }
        _append_metric(metrics_path, checkpoint_row)
        results.extend((validation, checkpoint_row))
        completed_epoch = epoch
    final_metrics = dict(validation)
    final_metrics["event"] = "stage2_training_complete"
    atomic_json(output_dir / "final_metrics.json", final_metrics)
    return results
