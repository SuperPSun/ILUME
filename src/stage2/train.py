from __future__ import annotations

import hashlib
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
from common.training import canonical_json_sha256, capture_rng_state, cosine_warmup, resolve_device, restore_rng_state, seed_everything
from stage1.features import ROLE_TO_ID
from stage1.masking import MultimodalPacker
from stage1.model import EncodedEntityStates, load_stage1_model
from .config import Stage2Config, stage2_config_from_checkpoint_dict
from .data import (
    PackedStage2Window, Stage2BatchDescriptor, Stage2DeviceTaskData,
    Stage2EntityDataset, Stage2TaskDataset, epoch_batch_schedule,
    load_artifact_registry, pack_stage2_window, task_batch_counts,
)
from .model import RECONSTRUCTION_MODULES, Stage2ForwardOutput, Stage2ObjectModel, stage2_optimizer_groups
from .prepare import load_teacher_embeddings
from .registry import Stage2Registry
from .runtime import configure_stage2_math


STAGE2_CHECKPOINT_VERSION = 3
STAGE2_CHECKPOINT_KIND = "ilume_stage2_object"
STAGE2_ENCODER_VERSION = 1
STAGE2_ENCODER_KIND = "ilume_stage2_encoder"


def _config_hash(config: Stage2Config) -> str:
    return canonical_json_sha256(config.experiment_dict())


def task_compensation_scale(task_weight: float, total_epoch_batches: int, batch_rows: int, task_rows: int) -> float:
    if task_weight <= 0 or total_epoch_batches <= 0 or batch_rows <= 0 or task_rows <= 0:
        raise ValueError("Stage 2 compensation inputs must be positive")
    return task_weight * total_epoch_batches * batch_rows / task_rows


def _training_geometry(config: Stage2Config, datasets: dict[str, Stage2TaskDataset]) -> tuple[dict[str, int], int, int, int]:
    if config.training.gradient_accumulation_steps != 1:
        raise ValueError("Stage 2 Object v3 requires one batch per optimizer step")
    batch_counts = task_batch_counts(datasets, config.training.batch_size)
    steps_per_epoch = sum(batch_counts.values())
    total_steps = steps_per_epoch * config.training.epochs
    return batch_counts, steps_per_epoch, total_steps, steps_per_epoch * config.training.backbone_frozen_epochs


def _scheduler_lambdas(config: Stage2Config, total_steps: int, unfreeze_step: int):
    def backbone(step: int) -> float:
        if step < unfreeze_step:
            return 0.0
        return cosine_warmup(step - unfreeze_step, total_steps - unfreeze_step, config.training.warmup_fraction)
    def modules(step: int) -> float:
        return cosine_warmup(step, total_steps, config.training.warmup_fraction)
    return backbone, modules, modules


def _ordered_packed_windows(
    windows: Sequence[tuple[Stage2BatchDescriptor, ...]], datasets: dict[str, Stage2TaskDataset],
    entities: Stage2EntityDataset, packer: MultimodalPacker, *, workers: int,
    prefetch_windows: int, pin_memory: bool,
) -> Iterator[PackedStage2Window]:
    capacity = max(1, workers * prefetch_windows)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stage2-packer") as executor:
        pending: deque[Future[PackedStage2Window]] = deque()
        iterator = iter(windows)
        for _ in range(capacity):
            try:
                window = next(iterator)
            except StopIteration:
                break
            pending.append(executor.submit(pack_stage2_window, window, datasets, entities, packer, pin_memory=pin_memory))
        while pending:
            yield pending.popleft().result()
            try:
                window = next(iterator)
            except StopIteration:
                continue
            pending.append(executor.submit(pack_stage2_window, window, datasets, entities, packer, pin_memory=pin_memory))


def _append_metric(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=True) + "\n")


def _reconcile_metrics_for_resume(path: Path, completed_epoch: int) -> None:
    if not path.is_file():
        raise FileNotFoundError("Stage 2 resume requires metrics.jsonl")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if int(row.get("epoch", 0)) <= completed_epoch:
                rows.append(json.dumps(row, sort_keys=True, allow_nan=True))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("" if not rows else "\n".join(rows) + "\n", encoding="utf-8")
    temporary.replace(path)


def _optimizer_implementation(device: torch.device) -> str:
    return "fused" if device.type == "cuda" else "single_tensor"


def _descriptor_base(descriptor: Stage2BatchDescriptor, data: Stage2DeviceTaskData, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = descriptor.indices.to(device)
    return indices, data.entity_indices[indices], data.conditions[indices]


def _selected_atom_targets(data: Stage2DeviceTaskData, indices: torch.Tensor, *, raw: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = data.raw_atom_target_values if raw else data.atom_target_values
    if values is None or data.atom_target_offsets is None or data.atom_target_mask is None:
        raise ValueError("Missing Stage 2 atom target tensors")
    chunks: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    offsets = [0]
    for index in indices.tolist():
        start = int(data.atom_target_offsets[index])
        end = int(data.atom_target_offsets[index + 1])
        chunks.append(values[start:end])
        masks.append(data.atom_target_mask[start:end])
        offsets.append(offsets[-1] + end - start)
    return torch.cat(chunks), torch.cat(masks), torch.tensor(offsets, dtype=torch.long, device=values.device)


def _batch_output(
    model: Stage2ObjectModel, registry: Stage2Registry, descriptor: Stage2BatchDescriptor,
    task_data: Stage2DeviceTaskData, teacher_embeddings: torch.Tensor,
    entity_roles: torch.Tensor, device: torch.device, config: Stage2Config,
    *, packed: PackedStage2Window | None, backbone_trainable: bool,
) -> Stage2ForwardOutput:
    spec = registry.by_id(descriptor.task)
    indices, global_slots, conditions = _descriptor_base(descriptor, task_data, device)
    teacher_slots = teacher_embeddings[global_slots]
    loss_mode = config.loss.task_loss_modes.get(spec.task_id, "element_mean")
    if not backbone_trainable and spec.target_level == "object":
        if task_data.targets is None or task_data.target_mask is None:
            raise ValueError("Missing object targets")
        return model.forward_object_from_slots(
            spec.task_id, teacher_slots, entity_roles[global_slots], conditions,
            task_data.targets[indices], task_data.target_mask[indices], teacher_slots,
            loss_mode=loss_mode, teacher_loss_is_zero=True,
        )
    if packed is None:
        raise AssertionError("Stage 2 batch requires packed entities")
    entities = packed.entities.to(device, non_blocking=device.type == "cuda")
    unique_ids = packed.unique_entity_ids.to(device, non_blocking=device.type == "cuda")
    positions = packed.entity_positions[0].to(device, non_blocking=device.type == "cuda")
    teacher_unique = teacher_embeddings[unique_ids]
    if spec.target_level == "object":
        if task_data.targets is None or task_data.target_mask is None:
            raise ValueError("Missing object targets")
        student = model.encode_entities(entities)
        return model.forward_object_from_slots(
            spec.task_id, student[positions], entities.roles[positions], conditions,
            task_data.targets[indices], task_data.target_mask[indices], teacher_unique[positions],
            loss_mode=loss_mode,
        )
    states = model.encode_entity_states(entities)
    targets, mask, offsets = _selected_atom_targets(task_data, indices)
    object_slots = teacher_unique[positions] if not backbone_trainable else states.entity_cls[positions]
    return model.forward_atom_from_states(
        spec.task_id, states, positions, entities.roles[positions], object_slots,
        teacher_unique[positions], targets, mask, offsets,
        teacher_loss_is_zero=not backbone_trainable,
    )


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _save_epoch_checkpoint(path: Path, *, model: Stage2ObjectModel, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, scaler: torch.amp.GradScaler, completed_epoch: int, global_optimizer_step: int, config: Stage2Config, registry: Stage2Registry, normalized_task_weights: dict[str, float], data_metadata_hash: str, teacher_embeddings_hash: str, teacher_cache_identity: str, task_rows: dict[str, int], task_batches: dict[str, int], scheduler_geometry: dict[str, int], validation: dict[str, Any], optimizer_implementation: str, math_contract: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Stage 2 checkpoint already exists: {path}")
    atomic_torch_save(path, {
        "format_version": STAGE2_CHECKPOINT_VERSION, "kind": STAGE2_CHECKPOINT_KIND,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "completed_epoch": completed_epoch,
        "global_optimizer_step": global_optimizer_step, "rng": capture_rng_state(),
        "config": config.to_dict(), "config_hash": _config_hash(config),
        "registry": registry.snapshot(), "registry_hash": registry.registry_hash,
        "catalog_sha256": registry.catalog_sha256, "model_contract": model.model_contract,
        "data_metadata_hash": data_metadata_hash, "teacher_embeddings_hash": teacher_embeddings_hash,
        "teacher_cache_identity": teacher_cache_identity, "task_rows": task_rows,
        "task_batches": task_batches, "normalized_task_weights": normalized_task_weights,
        "scheduler_geometry": scheduler_geometry,
        "loss_modes": {task: config.loss.task_loss_modes.get(task, "element_mean") for task in registry.task_ids},
        "optimizer_implementation": optimizer_implementation, "math_contract": math_contract,
        "validation": validation,
    })


def _export_encoder(path: Path, *, model: Stage2ObjectModel, config: Stage2Config, registry: Stage2Registry, checkpoint_path: Path, data_metadata_hash: str) -> None:
    if path.exists():
        raise FileExistsError(f"Stage 2 encoder artifact already exists: {path}")
    stage1_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.backbone.state_dict().items()
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in RECONSTRUCTION_MODULES)
    }
    object_state = {name: tensor.detach().cpu() for name, tensor in model.object_encoder.state_dict().items()}
    atomic_torch_save(path, {
        "kind": STAGE2_ENCODER_KIND, "format_version": STAGE2_ENCODER_VERSION,
        "stage1_backbone": stage1_state, "object_encoder": object_state,
        "stage1_config": model.backbone.config.to_dict(),
        "object_encoder_config": model.model_contract["object_encoder"],
        "role_to_id": dict(ROLE_TO_ID), "model_contract": model.model_contract,
        "state_hashes": {"stage1_backbone": _state_hash(stage1_state), "object_encoder": _state_hash(object_state)},
        "provenance": {
            "stage1_checkpoint_hash": sha256_file(config.initialization.checkpoint),
            "stage2_checkpoint_hash": sha256_file(checkpoint_path),
            "stage2_data_hash": data_metadata_hash,
            "task_catalog_hash": registry.catalog_sha256,
            "registry_hash": registry.registry_hash,
            "config_hash": _config_hash(config),
        },
    })


def load_stage2_encoder_artifact(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("kind") != STAGE2_ENCODER_KIND or payload.get("format_version") != STAGE2_ENCODER_VERSION:
        raise ValueError("Unsupported Stage 2 encoder artifact")
    stage1_state = payload.get("stage1_backbone")
    object_state = payload.get("object_encoder")
    if not isinstance(stage1_state, dict) or not isinstance(object_state, dict):
        raise ValueError("Stage 2 encoder artifact is missing encoding states")
    expected_hashes = payload.get("state_hashes", {})
    if expected_hashes.get("stage1_backbone") != _state_hash(stage1_state):
        raise ValueError("Stage 2 encoder Stage 1 state hash mismatch")
    if expected_hashes.get("object_encoder") != _state_hash(object_state):
        raise ValueError("Stage 2 encoder ObjectEncoder state hash mismatch")
    required = {"stage1_config", "object_encoder_config", "role_to_id", "model_contract", "provenance"}
    if not required.issubset(payload):
        raise ValueError("Stage 2 encoder artifact contract is incomplete")
    if payload["role_to_id"] != dict(ROLE_TO_ID):
        raise ValueError("Stage 2 encoder role mapping mismatch")
    return payload


@torch.inference_mode()
def evaluate_stage2(
    model: Stage2ObjectModel, registry: Stage2Registry,
    valid_datasets: dict[str, Stage2TaskDataset], device_data: dict[str, Stage2DeviceTaskData],
    entity_dataset: Stage2EntityDataset, packer: MultimodalPacker,
    teacher_embeddings: torch.Tensor, entity_roles: torch.Tensor,
    scalers: dict[str, Any], config: Stage2Config, device: torch.device,
    *, backbone_frozen: bool,
) -> dict[str, float | int | str]:
    was_training = model.training
    model.eval()
    result: dict[str, float | int | str] = {"validation_scope": "full", "validation_backbone_frozen": int(backbone_frozen)}
    task_scores: dict[str, float] = {}
    for spec in registry.tasks:
        dataset = valid_datasets[spec.task_id]
        data = device_data[spec.task_id]
        normalized_abs = 0.0
        normalized_units = 0
        raw_abs = [0.0] * len(spec.target_columns)
        raw_squared = [0.0] * len(spec.target_columns)
        raw_sum = [0.0] * len(spec.target_columns)
        raw_sum_squared = [0.0] * len(spec.target_columns)
        counts = [0] * len(spec.target_columns)
        atom_raw_molecule_abs = 0.0
        atom_molecule_count = 0
        teacher_sum = 0.0
        slot_count = 0
        for start in range(0, len(dataset), config.training.batch_size):
            descriptor = Stage2BatchDescriptor(spec.task_id, torch.arange(start, min(len(dataset), start + config.training.batch_size)))
            needs_pack = not backbone_frozen or spec.target_level == "atom"
            packed = pack_stage2_window((descriptor,), valid_datasets, entity_dataset, packer, pin_memory=device.type == "cuda") if needs_pack else None
            output = _batch_output(model, registry, descriptor, data, teacher_embeddings, entity_roles, device, config, packed=packed, backbone_trainable=not backbone_frozen)
            selected = descriptor.indices.to(device)
            teacher_sum += float(output.teacher_loss) * int(selected.numel())
            slot_count += int(selected.numel())
            task_scalers = scalers[spec.task_id]["targets"]
            if spec.target_level == "object":
                if data.targets is None or data.target_mask is None or data.raw_targets is None:
                    raise ValueError("Missing validation object targets")
                mask = data.target_mask[selected]
                normalized_diff = torch.abs(output.predictions - data.targets[selected])
                if config.loss.task_loss_modes.get(spec.task_id, "element_mean") == "masked_target_macro":
                    per_target = (normalized_diff * mask).sum(0) / mask.sum(0).clamp_min(1)
                    normalized_abs += float(per_target[mask.sum(0) > 0].mean()) * int(selected.numel())
                    normalized_units += int(selected.numel())
                else:
                    normalized_abs += float(normalized_diff[mask].sum())
                    normalized_units += int(mask.sum())
                raw_prediction = output.predictions.clone()
                for column, name in enumerate(spec.target_columns):
                    stats = task_scalers[name]
                    raw_prediction[:, column] = raw_prediction[:, column] * float(stats["scale"]) + float(stats["mean"])
                    selected_mask = mask[:, column]
                    target = data.raw_targets[selected, column][selected_mask].double()
                    difference = raw_prediction[:, column][selected_mask].double() - target
                    raw_abs[column] += float(torch.abs(difference).sum())
                    raw_squared[column] += float(torch.square(difference).sum())
                    raw_sum[column] += float(target.sum())
                    raw_sum_squared[column] += float(torch.square(target).sum())
                    counts[column] += int(selected_mask.sum())
            else:
                normalized_target, mask, offsets = _selected_atom_targets(data, selected)
                raw_target, _, _ = _selected_atom_targets(data, selected, raw=True)
                stats = task_scalers[spec.target_columns[0]]
                raw_prediction = output.predictions * float(stats["scale"]) + float(stats["mean"])
                difference = raw_prediction.double() - raw_target.double()
                for first, end in zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True):
                    normalized_abs += float(torch.abs(output.predictions[first:end] - normalized_target[first:end])[mask[first:end]].mean())
                    normalized_units += 1
                    atom_raw_molecule_abs += float(torch.abs(difference[first:end])[mask[first:end]].mean())
                    atom_molecule_count += 1
                raw_abs[0] += float(torch.abs(difference[mask]).sum())
                raw_squared[0] += float(torch.square(difference[mask]).sum())
                raw_sum[0] += float(raw_target[mask].double().sum())
                raw_sum_squared[0] += float(torch.square(raw_target[mask].double()).sum())
                counts[0] += int(mask.sum())
        score = normalized_abs / normalized_units
        task_scores[spec.task_id] = score
        prefix = f"valid_{spec.task_id}"
        result[f"{prefix}_rows"] = len(dataset)
        result[f"{prefix}_normalized_mae"] = score
        result[f"{prefix}_teacher_mse"] = teacher_sum / max(slot_count, 1)
        if spec.target_level == "atom":
            result[f"{prefix}_molecule_macro_normalized_mae"] = score
            result[f"{prefix}_molecule_macro_raw_mae"] = atom_raw_molecule_abs / atom_molecule_count
            result[f"{prefix}_atom_micro_rmse"] = math.sqrt(raw_squared[0] / counts[0])
            variance = raw_sum_squared[0] - raw_sum[0] ** 2 / counts[0]
            result[f"{prefix}_atom_micro_r2"] = 1.0 - raw_squared[0] / variance if variance > 0 else float("nan")
        for column, name in enumerate(spec.target_columns):
            count = counts[column]
            result[f"{prefix}_{name}_count"] = count
            result[f"{prefix}_{name}_mae"] = raw_abs[column] / count
            result[f"{prefix}_{name}_rmse"] = math.sqrt(raw_squared[column] / count)
            variance = raw_sum_squared[column] - raw_sum[column] ** 2 / count
            result[f"{prefix}_{name}_r2"] = 1.0 - raw_squared[column] / variance if variance > 0 else float("nan")
    normalized_weights = config.normalized_task_weights(registry)
    result["valid_macro_normalized_mae"] = float(np.mean(list(task_scores.values())))
    result["valid_weighted_macro_normalized_mae"] = sum(normalized_weights[task] * score for task, score in task_scores.items())
    if was_training:
        model.train()
    return result


def run_stage2_training(config: Stage2Config, *, output_dir: str | Path, resume_from: str | Path | None = None) -> list[dict[str, Any]]:
    config.validate()
    seed_everything(config.data.seed)
    device = resolve_device(config.training.device)
    math_contract = configure_stage2_math(device)
    loaded = load_stage1_model(config.initialization.checkpoint, config.data.pretrain_artifacts_dir, device=device, backbone_dropout=0.0)
    registry = load_artifact_registry(config.data.artifacts_dir)
    config.validate_registry(registry)
    model = Stage2ObjectModel(loaded.model, registry, object_layers=config.model.object_layers, object_ffn_dim=config.model.object_ffn_dim, dropout=config.model.dropout).to(device)
    entity_dataset = Stage2EntityDataset(config.data.artifacts_dir)
    metadata_path = config.data.artifacts_dir / "metadata.json"
    data_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if data_metadata.get("pretrain_artifact_hash") != loaded.artifact_hash or data_metadata.get("registry_hash") != registry.registry_hash or data_metadata.get("model_contract") != model.model_contract:
        raise ValueError("Stage 2 data artifact does not match model/registry")
    teacher_cpu, teacher_metadata = load_teacher_embeddings(config, loaded, data_metadata, math_contract, expected_count=len(entity_dataset), expected_dim=loaded.config.model.d_model)
    teacher_embeddings = teacher_cpu.to(device)
    train_datasets = {task: Stage2TaskDataset(config.data.artifacts_dir, task, "train") for task in registry.task_ids}
    valid_datasets = {task: Stage2TaskDataset(config.data.artifacts_dir, task, "valid") for task in registry.task_ids}
    train_device = {task: Stage2DeviceTaskData.from_dataset(dataset, device) for task, dataset in train_datasets.items()}
    valid_device = {task: Stage2DeviceTaskData.from_dataset(dataset, device) for task, dataset in valid_datasets.items()}
    entity_roles = torch.tensor([int(entry["role_id"]) for entry in entity_dataset.entries], dtype=torch.long, device=device)
    task_rows = {task: len(dataset) for task, dataset in train_datasets.items()}
    task_batches, steps_per_epoch, total_steps, unfreeze_step = _training_geometry(config, train_datasets)
    scheduler_geometry = {
        "gradient_accumulation_steps": 1,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "backbone_unfreeze_step": unfreeze_step,
    }
    total_epoch_batches = sum(task_batches.values())
    normalized_weights = config.normalized_task_weights(registry)
    packer = MultimodalPacker(loaded.vocabulary)
    model.set_backbone_trainable(config.training.backbone_frozen_epochs == 0)
    optimizer_implementation = _optimizer_implementation(device)
    optimizer = torch.optim.AdamW(stage2_optimizer_groups(model, backbone_learning_rate=config.training.backbone_learning_rate, object_encoder_learning_rate=config.training.object_encoder_learning_rate, task_head_learning_rate=config.training.task_head_learning_rate, weight_decay=config.training.weight_decay), fused=device.type == "cuda", foreach=False if device.type != "cuda" else None)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _scheduler_lambdas(config, total_steps, unfreeze_step))
    fp16 = config.training.amp_dtype == "fp16" and device.type == "cuda"
    amp_enabled = config.training.amp_dtype != "none" and device.type == "cuda"
    amp_dtype = torch.float16 if fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    data_metadata_hash = sha256_file(metadata_path)
    completed_epoch = 0
    global_step = 0
    if resume_from is None:
        if metrics_path.is_file() and metrics_path.stat().st_size:
            raise FileExistsError(f"Stage 2 output already contains metrics: {metrics_path}")
    else:
        checkpoint = torch.load(Path(resume_from), map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != STAGE2_CHECKPOINT_VERSION or checkpoint.get("kind") != STAGE2_CHECKPOINT_KIND:
            raise ValueError("Unsupported Stage 2 object checkpoint; Object v2 is not migrated")
        checkpoint_config = stage2_config_from_checkpoint_dict(checkpoint["config"])
        if checkpoint_config.experiment_dict() != config.experiment_dict():
            raise ValueError("Stage 2 checkpoint experiment config does not match")
        expected = {
            "config_hash": _config_hash(config), "registry_hash": registry.registry_hash,
            "model_contract": model.model_contract, "data_metadata_hash": data_metadata_hash,
            "teacher_embeddings_hash": teacher_metadata["embeddings_hash"],
            "teacher_cache_identity": teacher_metadata["identity"], "task_rows": task_rows,
            "task_batches": task_batches, "normalized_task_weights": normalized_weights,
            "scheduler_geometry": scheduler_geometry,
            "optimizer_implementation": optimizer_implementation, "math_contract": math_contract,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"Stage 2 checkpoint {key} does not match")
        completed_epoch = int(checkpoint["completed_epoch"])
        global_step = int(checkpoint["global_optimizer_step"])
        if not 1 <= completed_epoch <= config.training.epochs or global_step != completed_epoch * steps_per_epoch:
            raise ValueError("Stage 2 checkpoint is not a valid epoch boundary")
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
    validation: dict[str, Any] = {}
    for epoch in range(completed_epoch + 1, config.training.epochs + 1):
        backbone_trainable = epoch > config.training.backbone_frozen_epochs
        model.set_backbone_trainable(backbone_trainable)
        model.train()
        if not backbone_trainable:
            model.backbone.eval()
        schedule = epoch_batch_schedule(train_datasets, config.training.batch_size, seed=config.data.seed, epoch=epoch)
        needs_pack = [descriptor for descriptor in schedule if backbone_trainable or registry.by_id(descriptor.task).target_level == "atom"]
        packed_iterator = _ordered_packed_windows([(descriptor,) for descriptor in needs_pack], train_datasets, entity_dataset, packer, workers=config.training.packing_workers, prefetch_windows=config.training.packing_prefetch_windows, pin_memory=device.type == "cuda")
        interval_values: dict[str, list[float]] = {}
        interval_started = time.perf_counter()
        interval_batches = 0
        epoch_started = interval_started
        with reporter.bar(total=len(schedule), desc=f"Stage 2 object v3 epoch {epoch}", unit="batch") as progress:
            for descriptor in schedule:
                spec = registry.by_id(descriptor.task)
                packed = next(packed_iterator) if backbone_trainable or spec.target_level == "atom" else None
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    batch_output = _batch_output(model, registry, descriptor, train_device[descriptor.task], teacher_embeddings, entity_roles, device, config, packed=packed, backbone_trainable=backbone_trainable)
                    compensation = task_compensation_scale(normalized_weights[descriptor.task], total_epoch_batches, int(descriptor.indices.numel()), task_rows[descriptor.task])
                    loss = compensation * batch_output.physics_loss + config.loss.lambda_teacher * batch_output.teacher_loss
                if not bool(torch.isfinite(loss.detach())):
                    raise RuntimeError(f"Non-finite Stage 2 loss in epoch {epoch}")
                scaler.scale(loss).backward()
                if config.training.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_([parameter for group in optimizer.param_groups for parameter in group["params"] if parameter.requires_grad], config.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1
                interval_values.setdefault(descriptor.task, [0.0, 0.0, 0.0, 0.0])
                values = interval_values[descriptor.task]
                values[0] += float(batch_output.physics_loss.detach())
                values[1] += float(batch_output.teacher_loss.detach())
                values[2] += float(loss.detach())
                values[3] += 1
                interval_batches += 1
                progress.update(1)
                if interval_batches >= config.training.log_every_batches or descriptor is schedule[-1]:
                    row: dict[str, Any] = {"event": "stage2_train_interval", "epoch": epoch, "global_optimizer_step": global_step, "task_batches": interval_batches, "batches_per_second": interval_batches / max(time.perf_counter() - interval_started, 1e-12), "phase": "unfrozen" if backbone_trainable else "frozen", "backbone_learning_rate": optimizer.param_groups[0]["lr"], "object_encoder_learning_rate": optimizer.param_groups[1]["lr"], "task_head_learning_rate": optimizer.param_groups[2]["lr"]}
                    for task, values in interval_values.items():
                        row[f"{task}_loss_physics"] = values[0] / values[3]
                        row[f"{task}_loss_teacher"] = values[1] / values[3]
                        row[f"{task}_loss_step"] = values[2] / values[3]
                    _append_metric(metrics_path, row)
                    results.append(row)
                    interval_values = {}
                    interval_batches = 0
                    interval_started = time.perf_counter()
        validation = evaluate_stage2(model, registry, valid_datasets, valid_device, entity_dataset, packer, teacher_embeddings, entity_roles, data_metadata["scalers"], config, device, backbone_frozen=not backbone_trainable)
        validation.update({"event": "stage2_full_validation", "epoch": epoch, "global_optimizer_step": global_step, "epoch_wall_seconds": time.perf_counter() - epoch_started, "phase": "unfrozen" if backbone_trainable else "frozen"})
        _append_metric(metrics_path, validation)
        reporter.emit_json(validation)
        checkpoint_path = output / f"checkpoint_epoch_{epoch:05d}.pt"
        _save_epoch_checkpoint(checkpoint_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, completed_epoch=epoch, global_optimizer_step=global_step, config=config, registry=registry, normalized_task_weights=normalized_weights, data_metadata_hash=data_metadata_hash, teacher_embeddings_hash=teacher_metadata["embeddings_hash"], teacher_cache_identity=teacher_metadata["identity"], task_rows=task_rows, task_batches=task_batches, scheduler_geometry=scheduler_geometry, validation=validation, optimizer_implementation=optimizer_implementation, math_contract=math_contract)
        if epoch == config.training.epochs:
            _export_encoder(output / "stage2_encoder.pt", model=model, config=config, registry=registry, checkpoint_path=checkpoint_path, data_metadata_hash=data_metadata_hash)
        checkpoint_row = {"event": "stage2_checkpoint_complete", "epoch": epoch, "global_optimizer_step": global_step}
        _append_metric(metrics_path, checkpoint_row)
        results.extend((validation, checkpoint_row))
    final = dict(validation)
    final["event"] = "stage2_training_complete"
    atomic_json(output / "final_metrics.json", final)
    return results


__all__ = [
    "STAGE2_CHECKPOINT_KIND", "STAGE2_CHECKPOINT_VERSION", "evaluate_stage2",
    "load_stage2_encoder_artifact", "run_stage2_training", "task_compensation_scale",
]
