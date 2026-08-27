from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from common.identity import (
    IDENTITY_CONTRACT_VERSION,
    require_compatible_identity,
    tensor_state_hash,
)
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.refinement import (
    refinement_cosine_factor,
    selection_record,
)
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
from .prepare import (
    load_teacher_embeddings,
    stage1_encoder_identity,
    teacher_cache_dir,
    teacher_cache_identity,
)
from .registry import Stage2Registry
from .runtime import configure_stage2_math
from .identity import (
    build_stage2_encoder_identity,
    build_stage2_training_identity,
    metadata_identity,
)
from stage1.identity import metadata_identity as stage1_metadata_identity


STAGE2_ENCODER_VERSION = 1
STAGE2_ENCODER_KIND = "ilume_stage2_encoder"
STAGE2_REFINED_VERSION = 2
STAGE2_REFINED_KIND = "ilume_stage2_taskwise_refined"


def _config_hash(config: Stage2Config) -> str:
    return canonical_json_sha256(config.experiment_dict())


def resolve_stage2_training_identity(config: Stage2Config) -> dict[str, Any]:
    config.validate()
    runtime_device = resolve_device(config.training.device)
    math_contract = configure_stage2_math(runtime_device)
    loaded = load_stage1_model(
        config.initialization.checkpoint,
        config.data.pretrain_artifacts_dir,
        device="cpu",
        backbone_dropout=0.0,
    )
    registry = load_artifact_registry(config.data.artifacts_dir)
    config.validate_registry(registry)
    model = Stage2ObjectModel(
        loaded.model,
        registry,
        object_layers=config.model.object_layers,
        object_ffn_dim=config.model.object_ffn_dim,
        dropout=config.model.dropout,
    )
    data_metadata = json.loads(
        (config.data.artifacts_dir / "metadata.json").read_text(encoding="utf-8")
    )
    teacher_identity = teacher_cache_identity(data_metadata, loaded)
    teacher_metadata = json.loads(
        (
            teacher_cache_dir(config, teacher_identity["hash"])
            / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    stored_teacher = dict(
        metadata_identity(
            teacher_metadata, "teacher", context="Stage 2 teacher cache"
        )
    )
    require_compatible_identity(
        teacher_identity,
        stored_teacher,
        context="Stage 2 training teacher",
    )
    return build_stage2_training_identity(
        config,
        data_identity=dict(
            metadata_identity(
                data_metadata, "data", context="Stage 2 data artifact"
            )
        ),
        teacher_identity=stored_teacher,
        stage1_encoder_identity=stage1_encoder_identity(loaded),
        registry=registry,
        model_contract=model.model_contract,
        normalized_task_weights=config.normalized_task_weights(registry),
        math_contract=math_contract,
        optimizer_implementation=_optimizer_implementation(runtime_device),
    )


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
    *, use_student_encoder: bool, include_raw_atom_targets: bool,
    workers: int, prefetch_batches: int, pin_memory: bool,
) -> Iterator[PackedStage2Batch]:
    capacity = max(1, prefetch_batches)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stage2-packer") as executor:
        pending: deque[Future[PackedStage2Batch]] = deque()
        iterator = iter(descriptors)

        def submit(descriptor: Stage2BatchDescriptor) -> Future[PackedStage2Batch]:
            needs_entities = use_student_encoder or registry.by_id(descriptor.task).target_level == "atom"
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
    entity_roles: torch.Tensor, config: Stage2Config, *,
    backbone_trainable: bool, use_student_encoder: bool | None = None,
) -> Stage2ForwardOutput:
    if use_student_encoder is None:
        use_student_encoder = backbone_trainable
    descriptor = packed.descriptor
    spec = registry.by_id(descriptor.task)
    indices, global_slots, conditions = _descriptor_base(packed.row_indices, task_data)
    teacher_slots = teacher_embeddings[global_slots]
    loss_mode = config.loss.task_loss_modes.get(spec.task_id, "element_mean")
    if not use_student_encoder and spec.target_level == "object":
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
    object_slots = states.entity_cls[positions] if use_student_encoder else teacher_unique[positions]
    return model.forward_atom_from_states(
        spec.task_id, states, positions, entities.roles[positions], object_slots,
        teacher_unique[positions], atom_targets.values, atom_targets.mask,
        atom_targets.atom_state_indices, atom_targets.atom_sample_indices,
        teacher_loss_is_zero=not use_student_encoder,
    )


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    return tensor_state_hash("stage2.encoder-state", state)


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def _shared_state(model: Stage2ObjectModel) -> dict[str, torch.Tensor]:
    return {
        **{f"backbone.{name}": value for name, value in _cpu_state(model.backbone).items()},
        **{
            f"object_encoder.{name}": value
            for name, value in _cpu_state(model.object_encoder).items()
        },
    }


def _task_selection_metric(
    task_id: str, spec: Any, validation: Mapping[str, Any]
) -> tuple[str, float]:
    prefix = f"valid_{task_id}"
    if task_id == "simulation/partial_atomic_charge":
        name = "molecule_macro_normalized_mae"
        key = f"{prefix}_{name}"
    elif task_id in {"simulation/homo", "simulation/lumo"}:
        name = "pooled_sample_micro_raw_mae"
        key = f"{prefix}_{spec.target_columns[0]}_mae"
    else:
        name = "normalized_mae"
        key = f"{prefix}_normalized_mae"
    value = float(validation[key])
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite Stage 2 refinement metric: {task_id}")
    return name, value


def _refinement_optimizers(
    model: Stage2ObjectModel,
    config: Stage2Config,
    task_batches: Mapping[str, int],
    refinement_tasks: Sequence[str],
    device: torch.device,
) -> tuple[
    dict[str, torch.optim.AdamW],
    dict[str, torch.optim.lr_scheduler.LambdaLR],
]:
    optimizers: dict[str, torch.optim.AdamW] = {}
    schedulers: dict[str, torch.optim.lr_scheduler.LambdaLR] = {}
    seen: set[int] = set()
    for task_id in refinement_tasks:
        parameters = model.task_head_parameters_for(task_id)
        identities = {id(parameter) for parameter in parameters}
        if not parameters or seen & identities:
            raise RuntimeError("Stage 2 task head parameter ownership is invalid")
        seen.update(identities)
        optimizer = torch.optim.AdamW(
            parameters,
            lr=(
                config.training.task_head_learning_rate
                * config.training.refinement_lr_multiplier
            ),
            weight_decay=config.training.weight_decay,
            fused=device.type == "cuda",
            foreach=False if device.type != "cuda" else None,
        )
        total_updates = task_batches[task_id] * config.training.refinement_epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step, total=total_updates: refinement_cosine_factor(
                min(step, total), total
            ),
        )
        optimizers[task_id] = optimizer
        schedulers[task_id] = scheduler
    return optimizers, schedulers


def _initial_refinement_state(
    model: Stage2ObjectModel,
    registry: Stage2Registry,
    refinement_tasks: Sequence[str],
    validation: Mapping[str, Any],
    boundary_epoch: int,
) -> dict[str, Any]:
    selections: dict[str, Any] = {}
    selected = set(refinement_tasks)
    for spec in registry.tasks:
        if spec.task_id not in selected:
            continue
        metric_name, metric = _task_selection_metric(
            spec.task_id, spec, validation
        )
        selections[spec.task_id] = {
            **selection_record(
                metric_name=metric_name,
                boundary_epoch=boundary_epoch,
                boundary_metric=metric,
                selected_epoch=boundary_epoch,
                best_metric=metric,
            ),
            "selected_refinement_epoch": 0,
            "candidates": [
                {
                    "global_epoch": boundary_epoch,
                    "refinement_epoch": 0,
                    "metric": metric,
                }
            ],
            "best_state": _cpu_state(model.task_head_module(spec.task_id)),
        }
    unrefined_tasks = tuple(
        task_id for task_id in registry.task_ids if task_id not in selected
    )
    return {
        "boundary_epoch": boundary_epoch,
        "refined_tasks": tuple(refinement_tasks),
        "unrefined_tasks": unrefined_tasks,
        "shared_state_hash": tensor_state_hash(
            "stage2.refinement-shared-state", _shared_state(model)
        ),
        "unrefined_task_state_hashes": {
            task_id: tensor_state_hash(
                "stage2.refinement-private-state",
                _cpu_state(model.task_head_module(task_id)),
            )
            for task_id in unrefined_tasks
        },
        "selected_tasks": selections,
        "task_updates": {task: 0 for task in refinement_tasks},
    }


def _update_refinement_selection(
    state: dict[str, Any],
    model: Stage2ObjectModel,
    registry: Stage2Registry,
    validation: Mapping[str, Any],
    epoch: int,
    refinement_epoch: int,
) -> None:
    current_hash = tensor_state_hash(
        "stage2.refinement-shared-state", _shared_state(model)
    )
    if current_hash != state["shared_state_hash"]:
        raise RuntimeError("Stage 2 shared state changed during refinement")
    for task_id in state["refined_tasks"]:
        spec = registry.by_id(task_id)
        metric_name, metric = _task_selection_metric(
            spec.task_id, spec, validation
        )
        selected = state["selected_tasks"][spec.task_id]
        selected["candidates"].append(
            {
                "global_epoch": epoch,
                "refinement_epoch": refinement_epoch,
                "metric": metric,
            }
        )
        if metric_name != selected["metric"]:
            raise RuntimeError("Stage 2 refinement selection metric changed")
        if metric < float(selected["best_metric"]):
            selected["best_metric"] = metric
            selected["selected_epoch"] = epoch
            selected["selected_refinement_epoch"] = refinement_epoch
            selected["improved"] = metric < float(selected["boundary_metric"])
            selected["best_state"] = _cpu_state(
                model.task_head_module(spec.task_id)
            )


def _publish_taskwise_refined(
    output: Path,
    model: Stage2ObjectModel,
    registry: Stage2Registry,
    state: Mapping[str, Any],
    validation: Mapping[str, Any],
    training_identity: Mapping[str, Any],
    ordinary_final_epoch: int,
) -> dict[str, Any]:
    if tensor_state_hash(
        "stage2.refinement-shared-state", _shared_state(model)
    ) != state["shared_state_hash"]:
        raise RuntimeError("Stage 2 shared state changed before stitching")
    for task_id in state["refined_tasks"]:
        model.task_head_module(task_id).load_state_dict(
            state["selected_tasks"][task_id]["best_state"], strict=True
        )
    for task_id in state["unrefined_tasks"]:
        current_hash = tensor_state_hash(
            "stage2.refinement-private-state",
            _cpu_state(model.task_head_module(task_id)),
        )
        if current_hash != state["unrefined_task_state_hashes"][task_id]:
            raise RuntimeError("Stage 2 unrefined task head changed during refinement")
    model_state = model.state_dict()
    selected_public = {}
    for task, selection in state["selected_tasks"].items():
        selected_public[task] = {
            **{key: value for key, value in selection.items() if key != "best_state"},
            "private_state_hash": tensor_state_hash(
                "stage2.refinement-private-state", selection["best_state"]
            ),
        }
    private_state_hashes = {
        task_id: tensor_state_hash(
            "stage2.refinement-private-state",
            _cpu_state(model.task_head_module(task_id)),
        )
        for task_id in registry.task_ids
    }
    artifact_path = output / "taskwise_refined.pt"
    atomic_torch_save(
        artifact_path,
        {
            "kind": STAGE2_REFINED_KIND,
            "format_version": STAGE2_REFINED_VERSION,
            "model": model_state,
            "model_state_hash": tensor_state_hash(
                "stage2.taskwise-refined-state", model_state
            ),
            "training_identity": dict(training_identity),
            "boundary_epoch": state["boundary_epoch"],
            "refined_tasks": list(state["refined_tasks"]),
            "unrefined_tasks": list(state["unrefined_tasks"]),
            "shared_state_hash": state["shared_state_hash"],
            "private_state_hashes": private_state_hashes,
            "selected_tasks": selected_public,
            "validation": dict(validation),
        },
    )
    manifest = {
        "kind": STAGE2_REFINED_KIND,
        "format_version": STAGE2_REFINED_VERSION,
        "artifact": artifact_path.name,
        "artifact_sha256": sha256_file(artifact_path),
        "boundary_epoch": state["boundary_epoch"],
        "ordinary_final_epoch": ordinary_final_epoch,
        "refined_tasks": list(state["refined_tasks"]),
        "unrefined_tasks": list(state["unrefined_tasks"]),
        "shared_state_hash": state["shared_state_hash"],
        "private_state_hashes": private_state_hashes,
        "selected_tasks": selected_public,
        "validation": dict(validation),
    }
    atomic_json(output / "taskwise_refinement.json", manifest)
    return manifest


def _save_epoch_checkpoint(path: Path, *, model: Stage2ObjectModel, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, scaler: torch.amp.GradScaler, completed_epoch: int, global_optimizer_step: int, config: Stage2Config, registry: Stage2Registry, normalized_task_weights: dict[str, float], training_identity: dict[str, Any], data_identity: dict[str, Any], teacher_embeddings_hash: str, teacher_cache_identity: dict[str, Any], task_rows: dict[str, int], task_batches: dict[str, int], scheduler_geometry: dict[str, int], validation: dict[str, Any], optimizer_implementation: str, math_contract: dict[str, Any], phase: str, refinement_state: Mapping[str, Any] | None = None, refinement_optimizers: Mapping[str, torch.optim.Optimizer] | None = None, refinement_schedulers: Mapping[str, torch.optim.lr_scheduler.LRScheduler] | None = None) -> None:
    if path.exists():
        raise FileExistsError(f"Stage 2 checkpoint already exists: {path}")
    atomic_torch_save(path, {
        "format_version": STAGE2_CHECKPOINT_VERSION, "kind": STAGE2_CHECKPOINT_KIND,
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "completed_epoch": completed_epoch,
        "global_optimizer_step": global_optimizer_step, "rng": capture_rng_state(),
        "config": config.to_dict(), "config_hash": _config_hash(config),
        "registry": registry.snapshot(), "registry_hash": registry.registry_hash,
        "catalog_sha256": registry.catalog_sha256, "model_contract": model.model_contract,
        "training_identity": training_identity, "data_identity": data_identity,
        "teacher_embeddings_hash": teacher_embeddings_hash,
        "teacher_cache_identity": teacher_cache_identity, "task_rows": task_rows,
        "task_batches": task_batches, "normalized_task_weights": normalized_task_weights,
        "scheduler_geometry": scheduler_geometry,
        "execution_contract_version": 3,
        "execution_parameters": {
            "packing_workers": config.training.packing_workers,
            "packing_prefetch_batches": config.training.packing_prefetch_batches,
            "cuda_prefetch_batches": config.training.cuda_prefetch_batches,
            "log_every_batches": config.training.log_every_batches,
        },
        "loss_modes": {task: config.loss.task_loss_modes.get(task, "element_mean") for task in registry.task_ids},
        "optimizer_implementation": optimizer_implementation, "math_contract": math_contract,
        "validation": validation, "phase": phase,
        "refinement": (
            {
                **dict(refinement_state),
                "optimizers": {
                    task: item.state_dict()
                    for task, item in (refinement_optimizers or {}).items()
                },
                "schedulers": {
                    task: item.state_dict()
                    for task, item in (refinement_schedulers or {}).items()
                },
            }
            if refinement_state is not None
            else None
        ),
    })


def _export_encoder(path: Path, *, model: Stage2ObjectModel, config: Stage2Config, registry: Stage2Registry, checkpoint_path: Path, data_identity: dict[str, Any], refinement_state: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Stage 2 encoder artifact already exists: {path}")
    stage1_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.backbone.state_dict().items()
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in RECONSTRUCTION_MODULES)
    }
    object_state = {name: tensor.detach().cpu() for name, tensor in model.object_encoder.state_dict().items()}
    feature_metadata = json.loads(
        (config.data.pretrain_artifacts_dir / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    feature_identity = dict(
        stage1_metadata_identity(
            feature_metadata, "feature", context="Stage 1 feature artifact"
        )
    )
    feature_artifacts = {
        name: json.loads(
            (config.data.pretrain_artifacts_dir / name).read_text(encoding="utf-8")
        )
        for name in (
            "tokenizer.json",
            "descriptor_schema.json",
            "descriptor_scaler.json",
        )
    }
    stage1_state_hash = _state_hash(stage1_state)
    object_state_hash = _state_hash(object_state)
    stage1_model_config = model.backbone.config.to_dict()["model"]
    stage1_contract = {
        "encoding_api": "encode-states-v1",
        "model": {
            name: stage1_model_config[name]
            for name in (
                "d_model",
                "n_heads",
                "smiles_layers",
                "graph_depth",
                "descriptor_hidden_dim",
                "descriptor_blocks",
                "fusion_layers",
                "feedforward_dim",
                "dropout",
                "role_embedding",
                "gradient_checkpointing",
            )
        },
        "feature_generation_contract": feature_metadata[
            "feature_generation_contract"
        ],
    }
    encoder_identity = build_stage2_encoder_identity(
        stage1_feature_identity=feature_identity,
        stage1_encoding_contract=stage1_contract,
        stage1_state_hash=stage1_state_hash,
        object_encoder_contract=model.model_contract["object_encoder"],
        object_encoder_state_hash=object_state_hash,
        role_to_id=ROLE_TO_ID,
    )
    atomic_torch_save(path, {
        "kind": STAGE2_ENCODER_KIND, "format_version": STAGE2_ENCODER_VERSION,
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "semantic_identity": encoder_identity,
        "stage1_backbone": stage1_state, "object_encoder": object_state,
        "stage1_config": model.backbone.config.to_dict(),
        "stage1_feature_identity": feature_identity,
        "stage1_encoding_contract": stage1_contract,
        "feature_artifacts": feature_artifacts,
        "object_encoder_config": model.model_contract["object_encoder"],
        "role_to_id": dict(ROLE_TO_ID), "model_contract": model.model_contract,
        "state_hashes": {"stage1_backbone": stage1_state_hash, "object_encoder": object_state_hash},
        "provenance": {
            "stage1_checkpoint_hash": sha256_file(config.initialization.checkpoint),
            "stage2_checkpoint_hash": sha256_file(checkpoint_path),
            "stage2_data_identity": data_identity["hash"],
            "task_catalog_hash": registry.catalog_sha256,
            "registry_hash": registry.registry_hash,
            "config_hash": _config_hash(config),
            "refinement_boundary_epoch": refinement_state["boundary_epoch"],
            "refinement_shared_state_hash": refinement_state["shared_state_hash"],
        },
    })


def load_stage2_encoder_artifact(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("kind") != STAGE2_ENCODER_KIND or payload.get("format_version") != STAGE2_ENCODER_VERSION:
        raise ValueError("Unsupported Stage 2 encoder artifact")
    if payload.get("identity_contract_version") != IDENTITY_CONTRACT_VERSION:
        raise ValueError(
            "Stage 2 encoder predates identity contract v1; retrain Stage 2"
        )
    stage1_state = payload.get("stage1_backbone")
    object_state = payload.get("object_encoder")
    if not isinstance(stage1_state, dict) or not isinstance(object_state, dict):
        raise ValueError("Stage 2 encoder artifact is missing encoding states")
    expected_hashes = payload.get("state_hashes", {})
    if expected_hashes.get("stage1_backbone") != _state_hash(stage1_state):
        raise ValueError("Stage 2 encoder Stage 1 state hash mismatch")
    if expected_hashes.get("object_encoder") != _state_hash(object_state):
        raise ValueError("Stage 2 encoder ObjectEncoder state hash mismatch")
    required = {"stage1_config", "stage1_feature_identity", "stage1_encoding_contract", "feature_artifacts", "object_encoder_config", "role_to_id", "model_contract", "provenance", "semantic_identity"}
    if not required.issubset(payload):
        raise ValueError("Stage 2 encoder artifact contract is incomplete")
    if payload["role_to_id"] != dict(ROLE_TO_ID):
        raise ValueError("Stage 2 encoder role mapping mismatch")
    expected_identity = build_stage2_encoder_identity(
        stage1_feature_identity=payload["stage1_feature_identity"],
        stage1_encoding_contract=payload["stage1_encoding_contract"],
        stage1_state_hash=expected_hashes["stage1_backbone"],
        object_encoder_contract=payload["object_encoder_config"],
        object_encoder_state_hash=expected_hashes["object_encoder"],
        role_to_id=payload["role_to_id"],
    )
    require_compatible_identity(
        expected_identity,
        payload["semantic_identity"],
        context="Stage 2 encoder artifact",
    )
    return payload


@torch.inference_mode()
def evaluate_stage2(
    model: Stage2ObjectModel, registry: Stage2Registry,
    valid_datasets: dict[str, Stage2TaskDataset], device_data: dict[str, Stage2DeviceTaskData],
    entity_dataset: Stage2EntityDataset, packer: MultimodalPacker,
    teacher_embeddings: torch.Tensor, entity_roles: torch.Tensor,
    scalers: dict[str, Any], config: Stage2Config, device: torch.device,
    *, backbone_frozen: bool, use_student_encoder: bool | None = None,
) -> dict[str, float | int | str]:
    if use_student_encoder is None:
        use_student_encoder = not backbone_frozen
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
            use_student_encoder=use_student_encoder, include_raw_atom_targets=True,
            workers=config.training.packing_workers,
            prefetch_batches=config.training.packing_prefetch_batches,
            pin_memory=device.type == "cuda",
        )
        for packed in _device_batches(cpu_batches, device):
            output = _batch_output(
                model, registry, packed, data, teacher_embeddings, entity_roles,
                config, backbone_trainable=not backbone_frozen,
                use_student_encoder=use_student_encoder,
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


def run_stage2_training(config: Stage2Config, *, output_dir: str | Path, resume_from: str | Path | None = None, expected_training_identity: dict[str, Any] | None = None) -> list[dict[str, Any]]:
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
    boundary_epoch = config.training.epochs
    refinement_epochs = config.training.refinement_epochs
    total_epochs = boundary_epoch + refinement_epochs
    refinement_tasks = config.ordered_refinement_tasks(registry)
    refinement_steps_per_epoch = sum(task_batches[task] for task in refinement_tasks)
    scheduler_geometry = {
        "gradient_accumulation_steps": 1,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "backbone_unfreeze_step": unfreeze_step,
        "joint_epochs": boundary_epoch,
        "refinement_epochs": refinement_epochs,
        "refinement_steps_per_epoch": refinement_steps_per_epoch,
        "total_epochs": total_epochs,
    }
    total_epoch_batches = sum(task_batches.values())
    normalized_weights = config.normalized_task_weights(registry)
    packer = MultimodalPacker(loaded.vocabulary)
    model.set_backbone_trainable(config.training.backbone_frozen_epochs == 0)
    optimizer_implementation = _optimizer_implementation(device)
    data_identity = dict(
        metadata_identity(data_metadata, "data", context="Stage 2 data artifact")
    )
    teacher_identity = dict(
        metadata_identity(
            teacher_metadata, "teacher", context="Stage 2 teacher cache"
        )
    )
    initial_stage1_encoder_identity = stage1_encoder_identity(loaded)
    training_identity = build_stage2_training_identity(
        config,
        data_identity=data_identity,
        teacher_identity=teacher_identity,
        stage1_encoder_identity=initial_stage1_encoder_identity,
        registry=registry,
        model_contract=model.model_contract,
        normalized_task_weights=normalized_weights,
        math_contract=math_contract,
        optimizer_implementation=optimizer_implementation,
    )
    if expected_training_identity is not None:
        require_compatible_identity(
            training_identity,
            expected_training_identity,
            context="Stage 2 script/trainer",
        )
    optimizer = torch.optim.AdamW(stage2_optimizer_groups(model, backbone_learning_rate=config.training.backbone_learning_rate, object_encoder_learning_rate=config.training.object_encoder_learning_rate, task_head_learning_rate=config.training.task_head_learning_rate, weight_decay=config.training.weight_decay), fused=device.type == "cuda", foreach=False if device.type != "cuda" else None)
    clip_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _scheduler_lambdas(config, total_steps, unfreeze_step))
    refinement_optimizers, refinement_schedulers = _refinement_optimizers(
        model, config, task_batches, refinement_tasks, device
    )
    fp16 = config.training.amp_dtype == "fp16" and device.type == "cuda"
    amp_enabled = config.training.amp_dtype != "none" and device.type == "cuda"
    amp_dtype = torch.float16 if fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    completed_epoch = 0
    global_step = 0
    refinement_state: dict[str, Any] | None = None
    if resume_from is None:
        if metrics_path.is_file() and metrics_path.stat().st_size:
            raise FileExistsError(f"Stage 2 output already contains metrics: {metrics_path}")
    else:
        checkpoint = torch.load(Path(resume_from), map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != STAGE2_CHECKPOINT_VERSION or checkpoint.get("kind") != STAGE2_CHECKPOINT_KIND:
            raise ValueError(
                "Unsupported Stage 2 checkpoint; older refinement contracts are not migrated"
            )
        if checkpoint.get("identity_contract_version") != IDENTITY_CONTRACT_VERSION:
            raise ValueError(
                "Stage 2 checkpoint predates identity contract v1; retrain Stage 2"
            )
        if checkpoint.get("execution_contract_version") != 3:
            raise ValueError("Stage 2 checkpoint predates the current refinement contract")
        checkpoint_identity = checkpoint.get("training_identity")
        if not isinstance(checkpoint_identity, dict):
            raise ValueError("Stage 2 checkpoint has no training identity")
        require_compatible_identity(
            training_identity,
            checkpoint_identity,
            context="Stage 2 resume",
        )
        expected = {
            "registry_hash": registry.registry_hash,
            "model_contract": model.model_contract,
            "teacher_embeddings_hash": teacher_metadata["embeddings_hash"],
            "teacher_cache_identity": teacher_identity, "task_rows": task_rows,
            "task_batches": task_batches, "normalized_task_weights": normalized_weights,
            "scheduler_geometry": scheduler_geometry,
            "optimizer_implementation": optimizer_implementation, "math_contract": math_contract,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"Stage 2 checkpoint {key} does not match")
        completed_epoch = int(checkpoint["completed_epoch"])
        global_step = int(checkpoint["global_optimizer_step"])
        expected_global_step = (
            completed_epoch * steps_per_epoch
            if completed_epoch <= boundary_epoch
            else boundary_epoch * steps_per_epoch
            + (completed_epoch - boundary_epoch) * refinement_steps_per_epoch
        )
        if not 1 <= completed_epoch <= total_epochs or global_step != expected_global_step:
            raise ValueError("Stage 2 checkpoint is not a valid epoch boundary")
        expected_phase = (
            "boundary" if completed_epoch == boundary_epoch
            else "refinement" if completed_epoch > boundary_epoch
            else "unfrozen" if completed_epoch > config.training.backbone_frozen_epochs
            else "frozen"
        )
        if checkpoint.get("phase") != expected_phase:
            raise ValueError("Stage 2 checkpoint phase mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng"])
        stored_refinement = checkpoint.get("refinement")
        if completed_epoch >= boundary_epoch:
            if not isinstance(stored_refinement, dict):
                raise ValueError("Stage 2 refinement checkpoint has no refinement state")
            refinement_state = {
                key: value
                for key, value in stored_refinement.items()
                if key not in {"optimizers", "schedulers"}
            }
            stored_optimizers = stored_refinement.get("optimizers", {})
            stored_schedulers = stored_refinement.get("schedulers", {})
            if completed_epoch > boundary_epoch:
                if set(stored_optimizers) != set(refinement_tasks) or set(
                    stored_schedulers
                ) != set(refinement_tasks):
                    raise ValueError("Stage 2 refinement optimizer state is incomplete")
                for task in refinement_tasks:
                    refinement_optimizers[task].load_state_dict(
                        stored_optimizers[task]
                    )
                    refinement_schedulers[task].load_state_dict(
                        stored_schedulers[task]
                    )
        _reconcile_metrics_for_resume(metrics_path, completed_epoch)
    reporter = ProgressReporter()
    results: list[dict[str, Any]] = []
    validation: dict[str, Any] = {}
    for epoch in range(completed_epoch + 1, total_epochs + 1):
        in_refinement = epoch > boundary_epoch
        refinement_epoch = max(0, epoch - boundary_epoch)
        backbone_trainable = epoch > config.training.backbone_frozen_epochs
        if in_refinement:
            model.set_backbone_trainable(False)
            for parameter in model.object_encoder_parameters():
                parameter.requires_grad_(False)
        else:
            model.set_backbone_trainable(backbone_trainable)
            for parameter in model.object_encoder_parameters():
                parameter.requires_grad_(True)
            model.train()
            if not backbone_trainable:
                model.backbone.eval()
        epoch_datasets = (
            {task: train_datasets[task] for task in refinement_tasks}
            if in_refinement
            else train_datasets
        )
        schedule = epoch_batch_schedule(
            epoch_datasets,
            config.training.batch_size,
            seed=config.data.seed,
            epoch=epoch,
        )
        cpu_batches = _ordered_packed_batches(
            schedule, epoch_datasets, entity_dataset, packer, registry,
            use_student_encoder=(in_refinement or backbone_trainable), include_raw_atom_targets=False,
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
                active_optimizer = (
                    refinement_optimizers[descriptor.task]
                    if in_refinement
                    else optimizer
                )
                if in_refinement:
                    model.set_task_refinement_mode(descriptor.task)
                active_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    batch_output = _batch_output(
                        model, registry, packed, train_device[descriptor.task],
                        teacher_embeddings, entity_roles, config,
                        backbone_trainable=backbone_trainable and not in_refinement,
                        use_student_encoder=(in_refinement or backbone_trainable),
                    )
                    if in_refinement:
                        loss = batch_output.physics_loss
                    else:
                        compensation = task_compensation_scale(normalized_weights[descriptor.task], total_epoch_batches, int(descriptor.indices.numel()), task_rows[descriptor.task])
                        loss = compensation * batch_output.physics_loss + config.loss.lambda_teacher * batch_output.teacher_loss
                interval_finite = interval_finite & torch.isfinite(loss.detach())
                scaler.scale(loss).backward()
                if config.training.max_grad_norm > 0:
                    scaler.unscale_(active_optimizer)
                    active_clip_parameters = (
                        model.task_head_parameters_for(descriptor.task)
                        if in_refinement
                        else clip_parameters
                    )
                    torch.nn.utils.clip_grad_norm_(active_clip_parameters, config.training.max_grad_norm)
                scaler.step(active_optimizer)
                scaler.update()
                if in_refinement:
                    refinement_schedulers[descriptor.task].step()
                    assert refinement_state is not None
                    refinement_state["task_updates"][descriptor.task] += 1
                else:
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
                    row: dict[str, Any] = {"event": "stage2_train_interval", "epoch": epoch, "global_optimizer_step": global_step, "task_batches": interval_batches, "batches_per_second": interval_batches / max(time.perf_counter() - interval_started, 1e-12), "phase": "refinement" if in_refinement else ("unfrozen" if backbone_trainable else "frozen"), "backbone_learning_rate": 0.0 if in_refinement else optimizer.param_groups[0]["lr"], "object_encoder_learning_rate": 0.0 if in_refinement else optimizer.param_groups[1]["lr"], "task_head_learning_rate": active_optimizer.param_groups[0]["lr"] if in_refinement else optimizer.param_groups[2]["lr"]}
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
        validation = evaluate_stage2(model, registry, valid_datasets, valid_device, entity_dataset, packer, teacher_embeddings, entity_roles, data_metadata["scalers"], config, device, backbone_frozen=(in_refinement or not backbone_trainable), use_student_encoder=(in_refinement or backbone_trainable))
        phase = (
            "boundary" if epoch == boundary_epoch
            else "refinement" if in_refinement
            else "unfrozen" if backbone_trainable
            else "frozen"
        )
        validation.update({"event": "stage2_full_validation", "epoch": epoch, "refinement_epoch": refinement_epoch if epoch >= boundary_epoch else None, "global_optimizer_step": global_step, "epoch_wall_seconds": time.perf_counter() - epoch_started, "phase": phase})
        if epoch == boundary_epoch:
            refinement_state = _initial_refinement_state(
                model, registry, refinement_tasks, validation, boundary_epoch
            )
        elif in_refinement:
            assert refinement_state is not None
            _update_refinement_selection(
                refinement_state, model, registry, validation, epoch,
                refinement_epoch,
            )
        _append_metric(metrics_path, validation)
        reporter.emit_json(validation)
        checkpoint_path = output / f"checkpoint_epoch_{epoch:05d}.pt"
        _save_epoch_checkpoint(checkpoint_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, completed_epoch=epoch, global_optimizer_step=global_step, config=config, registry=registry, normalized_task_weights=normalized_weights, training_identity=training_identity, data_identity=data_identity, teacher_embeddings_hash=teacher_metadata["embeddings_hash"], teacher_cache_identity=teacher_identity, task_rows=task_rows, task_batches=task_batches, scheduler_geometry=scheduler_geometry, validation=validation, optimizer_implementation=optimizer_implementation, math_contract=math_contract, phase=phase, refinement_state=refinement_state, refinement_optimizers=refinement_optimizers if in_refinement else None, refinement_schedulers=refinement_schedulers if in_refinement else None)
        if epoch == boundary_epoch:
            assert refinement_state is not None
            _export_encoder(output / "stage2_encoder.pt", model=model, config=config, registry=registry, checkpoint_path=checkpoint_path, data_identity=data_identity, refinement_state=refinement_state)
        checkpoint_row = {"event": "stage2_checkpoint_complete", "epoch": epoch, "global_optimizer_step": global_step}
        _append_metric(metrics_path, checkpoint_row)
        results.extend((validation, checkpoint_row))
    if refinement_state is None:
        raise RuntimeError("Stage 2 refinement boundary was not captured")
    expected_refinement_updates = {
        task: task_batches[task] * refinement_epochs for task in refinement_tasks
    }
    if refinement_state["task_updates"] != expected_refinement_updates:
        raise RuntimeError("Stage 2 refinement task update counts are incomplete")
    if not validation and completed_epoch == total_epochs:
        validation = dict(checkpoint["validation"])
    encoder_path = output / "stage2_encoder.pt"
    if not encoder_path.is_file():
        source_checkpoint_path = Path(resume_from) if resume_from is not None else (
            output / f"checkpoint_epoch_{boundary_epoch:05d}.pt"
        )
        if not source_checkpoint_path.is_file():
            raise FileNotFoundError("Stage 2 encoder export requires a checkpoint")
        _export_encoder(
            encoder_path,
            model=model,
            config=config,
            registry=registry,
            checkpoint_path=source_checkpoint_path,
            data_identity=data_identity,
            refinement_state=refinement_state,
        )
    for task_id in refinement_tasks:
        model.task_head_module(task_id).load_state_dict(
            refinement_state["selected_tasks"][task_id]["best_state"], strict=True
        )
    stitched_validation = evaluate_stage2(
        model, registry, valid_datasets, valid_device, entity_dataset, packer,
        teacher_embeddings, entity_roles, data_metadata["scalers"], config, device,
        backbone_frozen=True, use_student_encoder=True,
    )
    stitched_validation.update({
        "event": "stage2_taskwise_refined_validation",
        "epoch": total_epochs,
        "refinement_epoch": refinement_epochs,
        "phase": "taskwise_refined",
    })
    manifest = _publish_taskwise_refined(
        output, model, registry, refinement_state, stitched_validation,
        training_identity,
        total_epochs,
    )
    final = {
        "event": "stage2_training_complete",
        "ordinary_final_epoch": total_epochs,
        "ordinary_final_validation": validation,
        "taskwise_refinement": manifest,
    }
    atomic_json(output / "final_metrics.json", final)
    return results


__all__ = [
    "STAGE2_CHECKPOINT_KIND", "STAGE2_CHECKPOINT_VERSION",
    "STAGE2_REFINED_KIND", "STAGE2_REFINED_VERSION", "evaluate_stage2",
    "load_stage2_encoder_artifact", "resolve_stage2_training_identity",
    "run_stage2_training", "task_compensation_scale",
]
