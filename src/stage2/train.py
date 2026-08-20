from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import fields, is_dataclass
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
from .config import (
    STAGE2_CHECKPOINT_KIND,
    STAGE2_CHECKPOINT_VERSION,
    Stage2Config,
    stage2_config_from_checkpoint_dict,
)
from .data import (
    PackedStage2Batch, Stage2BatchDescriptor, Stage2DeviceTaskData,
    Stage2EntityDataset, Stage2TaskDataset, epoch_batch_schedule,
    load_artifact_registry, pack_stage2_batch, task_batch_counts,
    validate_runtime_task_contract,
)
from .model import RECONSTRUCTION_MODULES, Stage2ForwardOutput, Stage2ObjectModel, stage2_optimizer_groups
from .prepare import load_teacher_embeddings
from .registry import Stage2Registry
from .runtime import configure_stage2_math


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


def _ordered_packed_batches(
    descriptors: Sequence[Stage2BatchDescriptor], datasets: dict[str, Stage2TaskDataset],
    entities: Stage2EntityDataset, packer: MultimodalPacker, registry: Stage2Registry,
    *, backbone_trainable: bool, include_raw_atom_targets: bool,
    workers: int, prefetch_batches: int, pin_memory: bool,
) -> Iterator[PackedStage2Batch]:
    capacity = max(1, prefetch_batches)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stage2-packer") as executor:
        pending: deque[Future[PackedStage2Batch]] = deque()
        iterator = iter(descriptors)

        def submit(descriptor: Stage2BatchDescriptor) -> Future[PackedStage2Batch]:
            needs_entities = backbone_trainable or registry.by_id(descriptor.task).target_level == "atom"
            return executor.submit(
                pack_stage2_batch, descriptor, datasets, entities, packer,
                needs_entities=needs_entities,
                include_raw_atom_targets=include_raw_atom_targets,
                pin_memory=pin_memory,
            )

        for _ in range(capacity):
            try:
                descriptor = next(iterator)
            except StopIteration:
                break
            pending.append(submit(descriptor))
        while pending:
            yield pending.popleft().result()
            try:
                descriptor = next(iterator)
            except StopIteration:
                continue
            pending.append(submit(descriptor))


def _record_stream(value: Any, stream: torch.cuda.Stream) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            value.record_stream(stream)
    elif is_dataclass(value):
        for field in fields(value):
            _record_stream(getattr(value, field.name), stream)
    elif isinstance(value, dict):
        for item in value.values():
            _record_stream(item, stream)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _record_stream(item, stream)


class CudaPackedBatchPrefetcher:
    def __init__(self, batches: Iterator[PackedStage2Batch], device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA packed-batch prefetch requires a CUDA device")
        self._batches = batches
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._next: PackedStage2Batch | None = None
        self._event: torch.cuda.Event | None = None
        self._closed = False
        self._preload()

    def _preload(self) -> None:
        try:
            batch = next(self._batches)
        except StopIteration:
            self._next = None
            self._event = None
            return
        with torch.cuda.stream(self._stream):
            self._next = batch.to(self._device, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self._stream)
            self._event = event

    def __iter__(self) -> "CudaPackedBatchPrefetcher":
        return self

    def __next__(self) -> PackedStage2Batch:
        if self._next is None or self._event is None:
            raise StopIteration
        current = self._next
        event = self._event
        consumer = torch.cuda.current_stream(self._device)
        consumer.wait_event(event)
        _record_stream(current, consumer)
        self._preload()
        return current

    def close(self, *, failed: bool = False) -> None:
        if self._closed:
            return
        if failed and self._next is not None:
            self._stream.synchronize()
        self._closed = True


def _device_batches(
    batches: Iterator[PackedStage2Batch], device: torch.device,
) -> Iterator[PackedStage2Batch]:
    if device.type != "cuda":
        yield from batches
        return
    prefetcher = CudaPackedBatchPrefetcher(batches, device)
    failed = True
    try:
        yield from prefetcher
        failed = False
    finally:
        prefetcher.close(failed=failed)


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


def _descriptor_base(indices: torch.Tensor, data: Stage2DeviceTaskData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return indices, data.entity_indices[indices], data.conditions[indices]


def _batch_output(
    model: Stage2ObjectModel, registry: Stage2Registry, packed: PackedStage2Batch,
    task_data: Stage2DeviceTaskData, teacher_embeddings: torch.Tensor,
    entity_roles: torch.Tensor, config: Stage2Config, *, backbone_trainable: bool,
) -> Stage2ForwardOutput:
    descriptor = packed.descriptor
    spec = registry.by_id(descriptor.task)
    indices, global_slots, conditions = _descriptor_base(packed.row_indices, task_data)
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
    if packed.entities is None or packed.unique_entity_ids is None or packed.entity_positions is None:
        raise AssertionError("Stage 2 batch requires packed entities")
    entities = packed.entities
    unique_ids = packed.unique_entity_ids
    positions = packed.entity_positions
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
    if packed.atom_targets is None:
        raise ValueError("Stage 2 atom batch is missing packed targets")
    atom_targets = packed.atom_targets
    object_slots = teacher_unique[positions] if not backbone_trainable else states.entity_cls[positions]
    return model.forward_atom_from_states(
        spec.task_id, states, positions, entities.roles[positions], object_slots,
        teacher_unique[positions], atom_targets.values, atom_targets.mask,
        atom_targets.atom_state_indices, atom_targets.atom_sample_indices,
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
        "execution_contract_version": 2,
        "execution_parameters": {
            "packing_workers": config.training.packing_workers,
            "packing_prefetch_batches": config.training.packing_prefetch_batches,
            "cuda_prefetch_batches": config.training.cuda_prefetch_batches,
            "log_every_batches": config.training.log_every_batches,
        },
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
        target_count = len(spec.target_columns)
        normalized_abs = torch.zeros((), dtype=torch.float64, device=device)
        normalized_units = torch.zeros((), dtype=torch.int64, device=device)
        raw_abs = torch.zeros(target_count, dtype=torch.float64, device=device)
        raw_squared = torch.zeros_like(raw_abs)
        raw_sum = torch.zeros_like(raw_abs)
        raw_sum_squared = torch.zeros_like(raw_abs)
        counts = torch.zeros(target_count, dtype=torch.int64, device=device)
        atom_raw_molecule_abs = torch.zeros((), dtype=torch.float64, device=device)
        atom_molecule_count = 0
        teacher_sum = torch.zeros((), dtype=torch.float64, device=device)
        slot_count = 0
        descriptors = [
            Stage2BatchDescriptor(
                spec.task_id,
                torch.arange(start, min(len(dataset), start + config.training.batch_size)),
            )
            for start in range(0, len(dataset), config.training.batch_size)
        ]
        cpu_batches = _ordered_packed_batches(
            descriptors, valid_datasets, entity_dataset, packer, registry,
            backbone_trainable=not backbone_frozen, include_raw_atom_targets=True,
            workers=config.training.packing_workers,
            prefetch_batches=config.training.packing_prefetch_batches,
            pin_memory=device.type == "cuda",
        )
        for packed in _device_batches(cpu_batches, device):
            output = _batch_output(
                model, registry, packed, data, teacher_embeddings, entity_roles,
                config, backbone_trainable=not backbone_frozen,
            )
            selected = packed.row_indices
            batch_rows = selected.numel()
            teacher_sum += output.teacher_loss.double() * batch_rows
            slot_count += batch_rows
            task_scalers = scalers[spec.task_id]["targets"]
            if spec.target_level == "object":
                if data.targets is None or data.target_mask is None or data.raw_targets is None:
                    raise ValueError("Missing validation object targets")
                mask = data.target_mask[selected]
                normalized_diff = torch.abs(output.predictions - data.targets[selected])
                if config.loss.task_loss_modes.get(spec.task_id, "element_mean") == "masked_target_macro":
                    target_counts = mask.sum(0)
                    valid = target_counts > 0
                    per_target = (normalized_diff * mask).sum(0) / target_counts.clamp_min(1)
                    normalized_abs += (
                        (per_target * valid).sum() / valid.sum().clamp_min(1)
                    ).double() * batch_rows
                    normalized_units += batch_rows
                else:
                    normalized_abs += (normalized_diff * mask).sum().double()
                    normalized_units += mask.sum()
                raw_prediction = output.predictions.clone()
                for column, name in enumerate(spec.target_columns):
                    stats = task_scalers[name]
                    raw_prediction[:, column] = raw_prediction[:, column] * float(stats["scale"]) + float(stats["mean"])
                    selected_mask = mask[:, column]
                    target = data.raw_targets[selected, column][selected_mask].double()
                    difference = raw_prediction[:, column][selected_mask].double() - target
                    raw_abs[column] += torch.abs(difference).sum()
                    raw_squared[column] += torch.square(difference).sum()
                    raw_sum[column] += target.sum()
                    raw_sum_squared[column] += torch.square(target).sum()
                    counts[column] += selected_mask.sum()
            else:
                if packed.atom_targets is None or packed.atom_targets.raw_values is None:
                    raise ValueError("Validation atom batch is missing raw targets")
                atom = packed.atom_targets
                normalized_target = atom.values
                raw_target = atom.raw_values
                mask = atom.mask
                stats = task_scalers[spec.target_columns[0]]
                raw_prediction = output.predictions * float(stats["scale"]) + float(stats["mean"])
                difference = raw_prediction.double() - raw_target.double()
                weights = mask.double()
                batch_molecules = packed.entity_positions.shape[0] if packed.entity_positions is not None else 0
                molecule_counts = torch.zeros(batch_molecules, dtype=torch.float64, device=device).index_add_(0, atom.atom_sample_indices, weights)
                normalized_molecule_abs = torch.zeros_like(molecule_counts).index_add_(
                    0, atom.atom_sample_indices,
                    torch.abs(output.predictions - normalized_target).double() * weights,
                ) / molecule_counts.clamp_min(1)
                raw_molecule_abs = torch.zeros_like(molecule_counts).index_add_(
                    0, atom.atom_sample_indices, torch.abs(difference) * weights,
                ) / molecule_counts.clamp_min(1)
                normalized_abs += normalized_molecule_abs.sum()
                normalized_units += batch_molecules
                atom_raw_molecule_abs += raw_molecule_abs.sum()
                atom_molecule_count += batch_molecules
                raw_abs[0] += (torch.abs(difference) * weights).sum()
                raw_squared[0] += (torch.square(difference) * weights).sum()
                raw_sum[0] += (raw_target.double() * weights).sum()
                raw_sum_squared[0] += (torch.square(raw_target.double()) * weights).sum()
                counts[0] += mask.sum()
        materialized = torch.cat((
            normalized_abs.reshape(1), normalized_units.double().reshape(1),
            raw_abs, raw_squared, raw_sum, raw_sum_squared, counts.double(),
            atom_raw_molecule_abs.reshape(1), teacher_sum.reshape(1),
        )).cpu().tolist()
        cursor = 0
        normalized_abs_value, normalized_units_value = materialized[cursor:cursor + 2]
        cursor += 2
        raw_abs_values = materialized[cursor:cursor + target_count]; cursor += target_count
        raw_squared_values = materialized[cursor:cursor + target_count]; cursor += target_count
        raw_sum_values = materialized[cursor:cursor + target_count]; cursor += target_count
        raw_sum_squared_values = materialized[cursor:cursor + target_count]; cursor += target_count
        count_values = [int(value) for value in materialized[cursor:cursor + target_count]]; cursor += target_count
        atom_raw_molecule_abs_value, teacher_sum_value = materialized[cursor:cursor + 2]
        score = normalized_abs_value / normalized_units_value
        task_scores[spec.task_id] = score
        prefix = f"valid_{spec.task_id}"
        result[f"{prefix}_rows"] = len(dataset)
        result[f"{prefix}_normalized_mae"] = score
        result[f"{prefix}_teacher_mse"] = teacher_sum_value / max(slot_count, 1)
        if spec.target_level == "atom":
            result[f"{prefix}_molecule_macro_normalized_mae"] = score
            result[f"{prefix}_molecule_macro_raw_mae"] = atom_raw_molecule_abs_value / atom_molecule_count
            result[f"{prefix}_atom_micro_rmse"] = math.sqrt(raw_squared_values[0] / count_values[0])
            variance = raw_sum_squared_values[0] - raw_sum_values[0] ** 2 / count_values[0]
            result[f"{prefix}_atom_micro_r2"] = 1.0 - raw_squared_values[0] / variance if variance > 0 else float("nan")
        for column, name in enumerate(spec.target_columns):
            count = count_values[column]
            result[f"{prefix}_{name}_count"] = count
            result[f"{prefix}_{name}_mae"] = raw_abs_values[column] / count
            result[f"{prefix}_{name}_rmse"] = math.sqrt(raw_squared_values[column] / count)
            variance = raw_sum_squared_values[column] - raw_sum_values[column] ** 2 / count
            result[f"{prefix}_{name}_r2"] = 1.0 - raw_squared_values[column] / variance if variance > 0 else float("nan")
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
    data_metadata = entity_dataset.metadata
    if data_metadata.get("pretrain_artifact_hash") != loaded.artifact_hash:
        raise ValueError("Stage 2 data artifact does not match Stage 1 features")
    if data_metadata.get("registry_hash") != registry.registry_hash:
        raise ValueError("Stage 2 data artifact registry mismatch")
    teacher_cpu, teacher_metadata = load_teacher_embeddings(config, loaded, data_metadata, expected_count=len(entity_dataset), expected_dim=loaded.config.model.d_model)
    teacher_embeddings = teacher_cpu.to(device)
    del teacher_cpu
    train_datasets = {task: Stage2TaskDataset(config.data.artifacts_dir, task, "train") for task in registry.task_ids}
    valid_datasets = {task: Stage2TaskDataset(config.data.artifacts_dir, task, "valid") for task in registry.task_ids}
    for task in registry.task_ids:
        mode = config.loss.task_loss_modes.get(task, "element_mean")
        validate_runtime_task_contract(train_datasets[task], entity_dataset, loss_mode=mode)
        validate_runtime_task_contract(valid_datasets[task], entity_dataset, loss_mode=mode)
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
    clip_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
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
        if checkpoint.get("execution_contract_version") != 2:
            raise ValueError("Stage 2 checkpoint predates the Object v3 efficiency contract")
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
        cpu_batches = _ordered_packed_batches(
            schedule, train_datasets, entity_dataset, packer, registry,
            backbone_trainable=backbone_trainable, include_raw_atom_targets=False,
            workers=config.training.packing_workers,
            prefetch_batches=config.training.packing_prefetch_batches,
            pin_memory=device.type == "cuda",
        )
        packed_iterator = _device_batches(cpu_batches, device)
        interval_values: dict[str, list[torch.Tensor | int]] = {}
        interval_finite = torch.ones((), dtype=torch.bool, device=device)
        interval_started = time.perf_counter()
        interval_batches = 0
        epoch_started = interval_started
        with reporter.bar(total=len(schedule), desc=f"Stage 2 object v3 epoch {epoch}", unit="batch") as progress:
            for batch_number, packed in enumerate(packed_iterator, start=1):
                descriptor = packed.descriptor
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    batch_output = _batch_output(
                        model, registry, packed, train_device[descriptor.task],
                        teacher_embeddings, entity_roles, config,
                        backbone_trainable=backbone_trainable,
                    )
                    compensation = task_compensation_scale(normalized_weights[descriptor.task], total_epoch_batches, int(descriptor.indices.numel()), task_rows[descriptor.task])
                    loss = compensation * batch_output.physics_loss + config.loss.lambda_teacher * batch_output.teacher_loss
                interval_finite = interval_finite & torch.isfinite(loss.detach())
                scaler.scale(loss).backward()
                if config.training.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(clip_parameters, config.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1
                interval_values.setdefault(
                    descriptor.task,
                    [loss.new_zeros(()), loss.new_zeros(()), loss.new_zeros(()), 0],
                )
                values = interval_values[descriptor.task]
                values[0] = values[0] + batch_output.physics_loss.detach()  # type: ignore[operator]
                values[1] = values[1] + batch_output.teacher_loss.detach()  # type: ignore[operator]
                values[2] = values[2] + loss.detach()  # type: ignore[operator]
                values[3] = int(values[3]) + 1
                interval_batches += 1
                progress.update(1)
                if interval_batches >= config.training.log_every_batches or batch_number == len(schedule):
                    ordered_tasks = sorted(interval_values)
                    device_values = torch.stack([
                        value
                        for task in ordered_tasks
                        for value in interval_values[task][:3]
                        if isinstance(value, torch.Tensor)
                    ]).reshape(len(ordered_tasks), 3)
                    materialized = torch.cat((
                        interval_finite.to(torch.float32).reshape(1),
                        device_values.flatten(),
                    )).cpu()
                    if not bool(materialized[0]):
                        raise RuntimeError(f"Non-finite Stage 2 loss in epoch {epoch}")
                    row: dict[str, Any] = {"event": "stage2_train_interval", "epoch": epoch, "global_optimizer_step": global_step, "task_batches": interval_batches, "batches_per_second": interval_batches / max(time.perf_counter() - interval_started, 1e-12), "phase": "unfrozen" if backbone_trainable else "frozen", "backbone_learning_rate": optimizer.param_groups[0]["lr"], "object_encoder_learning_rate": optimizer.param_groups[1]["lr"], "task_head_learning_rate": optimizer.param_groups[2]["lr"]}
                    for task_index, task in enumerate(ordered_tasks):
                        count = int(interval_values[task][3])
                        row[f"{task}_loss_physics"] = float(materialized[1 + task_index * 3]) / count
                        row[f"{task}_loss_teacher"] = float(materialized[2 + task_index * 3]) / count
                        row[f"{task}_loss_step"] = float(materialized[3 + task_index * 3]) / count
                    _append_metric(metrics_path, row)
                    results.append(row)
                    interval_values = {}
                    interval_finite = torch.ones((), dtype=torch.bool, device=device)
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
