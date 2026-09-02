from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from common.identity import IDENTITY_CONTRACT_VERSION, require_compatible_identity
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.progress import ProgressReporter
from common.training import capture_rng_state, resolve_device, restore_rng_state, seed_everything
from stage1.features import ROLE_TO_ID

from .config import STAGE2_CHECKPOINT_VERSION, Stage2Config
from .data import (
    STAGE2_RDKIT_ARTIFACT_KIND,
    Stage2BatchDescriptor,
    Stage2DeviceTaskData,
    Stage2EntityDataset,
    Stage2TaskDataset,
    epoch_batch_schedule,
    load_artifact_registry,
    task_batch_counts,
    validate_runtime_task_contract,
)
from .identity import (
    build_rdkit_stage2_encoder_identity,
    build_rdkit_stage2_training_identity,
    metadata_identity,
)
from .model import RDKitDescriptorBackbone, Stage2ForwardOutput, Stage2ObjectModel, stage2_optimizer_groups
from .runtime import configure_stage2_math
from .train import (
    _append_metric,
    _config_hash,
    _initial_refinement_state,
    _optimizer_implementation,
    _publish_taskwise_refined,
    _reconcile_metrics_for_resume,
    _refinement_optimizers,
    _scheduler_lambdas,
    _state_hash,
    _update_refinement_selection,
    task_compensation_scale,
)


STAGE2_RDKIT_CHECKPOINT_KIND = "ilume_stage2_rdkit_object"
STAGE2_RDKIT_REFINED_KIND = "ilume_stage2_rdkit_taskwise_refined"
STAGE2_RDKIT_ENCODER_KIND = "ilume_stage2_rdkit_encoder"
STAGE2_RDKIT_ENCODER_VERSION = 1


def _model(config: Stage2Config, registry: Any, input_dim: int) -> Stage2ObjectModel:
    representation = config.representation
    if representation is None:
        raise ValueError("RDKit Stage 2 training requires representation config")
    backbone = RDKitDescriptorBackbone(
        input_dim,
        hidden_dim=representation.hidden_dim,
        output_dim=representation.output_dim,
        dropout=representation.dropout,
    )
    return Stage2ObjectModel(
        backbone,  # type: ignore[arg-type]
        registry,
        object_layers=config.model.object_layers,
        object_ffn_dim=config.model.object_ffn_dim,
        dropout=config.model.dropout,
    )


def _batch_output(
    model: Stage2ObjectModel,
    task: str,
    descriptor: Stage2BatchDescriptor,
    task_data: Stage2DeviceTaskData,
    features: torch.Tensor,
    roles: torch.Tensor,
    config: Stage2Config,
) -> Stage2ForwardOutput:
    spec = model.specs[task]
    indices = descriptor.indices.to(features.device)
    global_slots = task_data.entity_indices[indices]
    batch_size, slot_count = global_slots.shape
    encoded = model.encode_entities(features[global_slots].reshape(-1, features.shape[1]))
    slots = encoded.reshape(batch_size, slot_count, -1)
    if task_data.targets is None or task_data.target_mask is None:
        raise ValueError("RDKit Stage 2 requires object targets")
    return model.forward_object_from_slots(
        task,
        slots,
        roles[global_slots],
        task_data.conditions[indices],
        task_data.targets[indices],
        task_data.target_mask[indices],
        slots,
        loss_mode=config.loss.task_loss_modes.get(spec.task_id, "element_mean"),
        teacher_loss_is_zero=True,
    )


@torch.inference_mode()
def evaluate_rdkit_stage2(
    model: Stage2ObjectModel,
    registry: Any,
    datasets: Mapping[str, Stage2TaskDataset],
    device_data: Mapping[str, Stage2DeviceTaskData],
    features: torch.Tensor,
    roles: torch.Tensor,
    scalers: Mapping[str, Any],
    config: Stage2Config,
    *,
    shared_frozen: bool = False,
) -> dict[str, float | int | str]:
    was_training = model.training
    model.eval()
    result: dict[str, float | int | str] = {
        "validation_scope": "full",
        "validation_backbone_frozen": int(shared_frozen),
    }
    task_scores: dict[str, float] = {}
    for spec in registry.tasks:
        dataset = datasets[spec.task_id]
        data = device_data[spec.task_id]
        target_count = len(spec.target_columns)
        normalized_abs = torch.zeros((), dtype=torch.float64, device=features.device)
        normalized_units = torch.zeros((), dtype=torch.int64, device=features.device)
        raw_abs = torch.zeros(target_count, dtype=torch.float64, device=features.device)
        raw_squared = torch.zeros_like(raw_abs)
        raw_sum = torch.zeros_like(raw_abs)
        raw_sum_squared = torch.zeros_like(raw_abs)
        counts = torch.zeros(target_count, dtype=torch.int64, device=features.device)
        for start in range(0, len(dataset), config.training.batch_size):
            selected = torch.arange(
                start,
                min(len(dataset), start + config.training.batch_size),
                device=features.device,
            )
            descriptor = Stage2BatchDescriptor(spec.task_id, selected)
            output = _batch_output(
                model, spec.task_id, descriptor, data, features, roles, config
            )
            if data.targets is None or data.target_mask is None or data.raw_targets is None:
                raise ValueError("Missing RDKit Stage 2 validation targets")
            mask = data.target_mask[selected]
            difference = torch.abs(output.predictions - data.targets[selected])
            if config.loss.task_loss_modes.get(spec.task_id, "element_mean") == "masked_target_macro":
                target_counts = mask.sum(0)
                valid = target_counts > 0
                per_target = (difference * mask).sum(0) / target_counts.clamp_min(1)
                normalized_abs += ((per_target * valid).sum() / valid.sum().clamp_min(1)).double() * len(selected)
                normalized_units += len(selected)
            else:
                normalized_abs += (difference * mask).sum().double()
                normalized_units += mask.sum()
            raw_prediction = output.predictions.clone()
            for column, name in enumerate(spec.target_columns):
                stats = scalers[spec.task_id]["targets"][name]
                raw_prediction[:, column] = raw_prediction[:, column] * float(stats["scale"]) + float(stats["mean"])
                selected_mask = mask[:, column]
                target = data.raw_targets[selected, column][selected_mask].double()
                delta = raw_prediction[:, column][selected_mask].double() - target
                raw_abs[column] += torch.abs(delta).sum()
                raw_squared[column] += torch.square(delta).sum()
                raw_sum[column] += target.sum()
                raw_sum_squared[column] += torch.square(target).sum()
                counts[column] += selected_mask.sum()
        score = float((normalized_abs / normalized_units).cpu())
        task_scores[spec.task_id] = score
        prefix = f"valid_{spec.task_id}"
        result[f"{prefix}_rows"] = len(dataset)
        result[f"{prefix}_normalized_mae"] = score
        for column, name in enumerate(spec.target_columns):
            count = int(counts[column].cpu())
            absolute = float(raw_abs[column].cpu())
            squared = float(raw_squared[column].cpu())
            target_sum = float(raw_sum[column].cpu())
            target_squared = float(raw_sum_squared[column].cpu())
            variance = target_squared - target_sum * target_sum / count
            result[f"{prefix}_{name}_count"] = count
            result[f"{prefix}_{name}_mae"] = absolute / count
            result[f"{prefix}_{name}_rmse"] = math.sqrt(squared / count)
            result[f"{prefix}_{name}_r2"] = 1.0 - squared / variance if variance > 0 else float("nan")
    weights = config.normalized_task_weights(registry)
    result["valid_macro_normalized_mae"] = float(np.mean(list(task_scores.values())))
    result["valid_weighted_macro_normalized_mae"] = sum(
        weights[task] * score for task, score in task_scores.items()
    )
    if was_training:
        model.train()
    return result


def _training_context(config: Stage2Config) -> tuple[Any, Stage2EntityDataset, Stage2ObjectModel]:
    registry = load_artifact_registry(config.data.artifacts_dir)
    config.validate_registry(registry)
    entities = Stage2EntityDataset(config.data.artifacts_dir)
    if entities.metadata.get("kind") != STAGE2_RDKIT_ARTIFACT_KIND or entities.features is None:
        raise ValueError("RDKit Stage 2 training requires RDKit prepared artifact")
    model = _model(config, registry, int(entities.features.shape[1]))
    return registry, entities, model


def resolve_rdkit_stage2_training_identity(config: Stage2Config) -> dict[str, Any]:
    config.validate()
    device = resolve_device(config.training.device)
    math_contract = configure_stage2_math(device)
    registry, entities, model = _training_context(config)
    data_identity = metadata_identity(
        entities.metadata, "data", context="RDKit Stage 2 artifact"
    )
    return build_rdkit_stage2_training_identity(
        config,
        data_identity=data_identity,
        registry=registry,
        model_contract=model.model_contract,
        normalized_task_weights=config.normalized_task_weights(registry),
        math_contract=math_contract,
        optimizer_implementation=_optimizer_implementation(device),
    )


def _checkpoint_payload(
    *,
    model: Stage2ObjectModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    config: Stage2Config,
    registry: Any,
    training_identity: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    task_rows: Mapping[str, int],
    task_batches: Mapping[str, int],
    normalized_weights: Mapping[str, float],
    scheduler_geometry: Mapping[str, int],
    validation: Mapping[str, Any],
    optimizer_implementation: str,
    math_contract: Mapping[str, Any],
    phase: str,
    refinement_state: Mapping[str, Any] | None,
    refinement_optimizers: Mapping[str, torch.optim.Optimizer] | None,
    refinement_schedulers: Mapping[str, torch.optim.lr_scheduler.LRScheduler] | None,
) -> dict[str, Any]:
    return {
        "format_version": STAGE2_CHECKPOINT_VERSION,
        "kind": STAGE2_RDKIT_CHECKPOINT_KIND,
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "completed_epoch": epoch,
        "global_optimizer_step": global_step,
        "rng": capture_rng_state(),
        "config": config.to_dict(),
        "config_hash": _config_hash(config),
        "representation": config.to_dict()["representation"],
        "registry": registry.snapshot(),
        "registry_hash": registry.registry_hash,
        "catalog_sha256": registry.catalog_sha256,
        "model_contract": model.model_contract,
        "training_identity": dict(training_identity),
        "data_identity": dict(data_identity),
        "task_rows": dict(task_rows),
        "task_batches": dict(task_batches),
        "normalized_task_weights": dict(normalized_weights),
        "scheduler_geometry": dict(scheduler_geometry),
        "optimizer_implementation": optimizer_implementation,
        "math_contract": dict(math_contract),
        "validation": dict(validation),
        "phase": phase,
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
    }


def _export_encoder(
    path: Path,
    *,
    model: Stage2ObjectModel,
    config: Stage2Config,
    registry: Any,
    checkpoint_path: Path,
    data_identity: Mapping[str, Any],
    descriptor_contract: Mapping[str, Any],
    refinement_state: Mapping[str, Any],
) -> None:
    if path.exists():
        raise FileExistsError(f"Stage 2 encoder artifact already exists: {path}")
    descriptor_state = {
        name: value.detach().cpu() for name, value in model.backbone.state_dict().items()
    }
    object_state = {
        name: value.detach().cpu()
        for name, value in model.object_encoder.state_dict().items()
    }
    descriptor_hash = _state_hash(descriptor_state)
    object_hash = _state_hash(object_state)
    identity = build_rdkit_stage2_encoder_identity(
        descriptor_contract=descriptor_contract,
        descriptor_encoder_state_hash=descriptor_hash,
        object_encoder_contract=model.model_contract["object_encoder"],
        object_encoder_state_hash=object_hash,
        role_to_id=ROLE_TO_ID,
    )
    atomic_torch_save(
        path,
        {
            "kind": STAGE2_RDKIT_ENCODER_KIND,
            "format_version": STAGE2_RDKIT_ENCODER_VERSION,
            "identity_contract_version": IDENTITY_CONTRACT_VERSION,
            "semantic_identity": identity,
            "descriptor_contract": dict(descriptor_contract),
            "descriptor_encoder": descriptor_state,
            "object_encoder": object_state,
            "object_encoder_config": model.model_contract["object_encoder"],
            "role_to_id": dict(ROLE_TO_ID),
            "model_contract": model.model_contract,
            "state_hashes": {
                "descriptor_encoder": descriptor_hash,
                "object_encoder": object_hash,
            },
            "provenance": {
                "stage2_checkpoint_hash": sha256_file(checkpoint_path),
                "stage2_data_identity": data_identity["hash"],
                "task_catalog_hash": registry.catalog_sha256,
                "registry_hash": registry.registry_hash,
                "config_hash": _config_hash(config),
                "refinement_boundary_epoch": refinement_state["boundary_epoch"],
                "refinement_shared_state_hash": refinement_state["shared_state_hash"],
            },
        },
    )


def load_rdkit_stage2_encoder_artifact(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if (
        payload.get("kind") != STAGE2_RDKIT_ENCODER_KIND
        or payload.get("format_version") != STAGE2_RDKIT_ENCODER_VERSION
        or payload.get("identity_contract_version") != IDENTITY_CONTRACT_VERSION
    ):
        raise ValueError("Unsupported RDKit Stage 2 encoder artifact")
    required = {
        "semantic_identity",
        "descriptor_contract",
        "descriptor_encoder",
        "object_encoder",
        "object_encoder_config",
        "role_to_id",
        "model_contract",
        "state_hashes",
        "provenance",
    }
    if not required.issubset(payload):
        raise ValueError("RDKit Stage 2 encoder artifact contract is incomplete")
    if payload["role_to_id"] != ROLE_TO_ID:
        raise ValueError("RDKit Stage 2 encoder role mapping mismatch")
    descriptor_state = payload.get("descriptor_encoder")
    object_state = payload.get("object_encoder")
    if not isinstance(descriptor_state, dict) or not isinstance(object_state, dict):
        raise ValueError("RDKit Stage 2 encoder states are missing")
    hashes = payload.get("state_hashes", {})
    if hashes.get("descriptor_encoder") != _state_hash(descriptor_state):
        raise ValueError("RDKit Stage 2 descriptor encoder state hash mismatch")
    if hashes.get("object_encoder") != _state_hash(object_state):
        raise ValueError("RDKit Stage 2 ObjectEncoder state hash mismatch")
    identity = build_rdkit_stage2_encoder_identity(
        descriptor_contract=payload["descriptor_contract"],
        descriptor_encoder_state_hash=hashes["descriptor_encoder"],
        object_encoder_contract=payload["object_encoder_config"],
        object_encoder_state_hash=hashes["object_encoder"],
        role_to_id=payload["role_to_id"],
    )
    require_compatible_identity(
        identity, payload["semantic_identity"], context="RDKit Stage 2 encoder"
    )
    return payload


def run_rdkit_stage2_training(
    config: Stage2Config,
    *,
    output_dir: str | Path,
    resume_from: str | Path | None = None,
    expected_training_identity: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    config.validate()
    seed_everything(config.data.seed)
    device = resolve_device(config.training.device)
    math_contract = configure_stage2_math(device)
    registry, entities, model = _training_context(config)
    model = model.to(device)
    assert entities.features is not None
    features = entities.features.to(device)
    roles = torch.tensor(
        [int(entry["role_id"]) for entry in entities.entries],
        dtype=torch.long,
        device=device,
    )
    train_datasets = {
        task: Stage2TaskDataset(config.data.artifacts_dir, task, "train")
        for task in registry.task_ids
    }
    valid_datasets = {
        task: Stage2TaskDataset(config.data.artifacts_dir, task, "valid")
        for task in registry.task_ids
    }
    for task in registry.task_ids:
        validate_runtime_task_contract(
            train_datasets[task], entities, loss_mode=config.loss.task_loss_modes.get(task, "element_mean")
        )
        validate_runtime_task_contract(
            valid_datasets[task], entities, loss_mode=config.loss.task_loss_modes.get(task, "element_mean")
        )
    train_device = {
        task: Stage2DeviceTaskData.from_dataset(dataset, device)
        for task, dataset in train_datasets.items()
    }
    valid_device = {
        task: Stage2DeviceTaskData.from_dataset(dataset, device)
        for task, dataset in valid_datasets.items()
    }
    task_rows = {task: len(dataset) for task, dataset in train_datasets.items()}
    task_batches = task_batch_counts(train_datasets, config.training.batch_size)
    steps_per_epoch = sum(task_batches.values())
    boundary_epoch = config.training.epochs
    refinement_epochs = config.training.refinement_epochs
    total_epochs = boundary_epoch + refinement_epochs
    total_steps = steps_per_epoch * boundary_epoch
    refinement_tasks = config.ordered_refinement_tasks(registry)
    refinement_steps = sum(task_batches[task] for task in refinement_tasks)
    scheduler_geometry = {
        "gradient_accumulation_steps": 1,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "backbone_unfreeze_step": 0,
        "joint_epochs": boundary_epoch,
        "refinement_epochs": refinement_epochs,
        "refinement_steps_per_epoch": refinement_steps,
        "total_epochs": total_epochs,
    }
    normalized_weights = config.normalized_task_weights(registry)
    optimizer_implementation = _optimizer_implementation(device)
    representation = config.representation
    assert representation is not None
    optimizer = torch.optim.AdamW(
        stage2_optimizer_groups(
            model,
            backbone_learning_rate=representation.learning_rate,
            object_encoder_learning_rate=config.training.object_encoder_learning_rate,
            task_head_learning_rate=config.training.task_head_learning_rate,
            weight_decay=config.training.weight_decay,
        ),
        fused=device.type == "cuda",
        foreach=False if device.type != "cuda" else None,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _scheduler_lambdas(config, total_steps, 0)
    )
    refinement_optimizers, refinement_schedulers = _refinement_optimizers(
        model, config, task_batches, refinement_tasks, device
    )
    fp16 = config.training.amp_dtype == "fp16" and device.type == "cuda"
    amp_enabled = config.training.amp_dtype != "none" and device.type == "cuda"
    amp_dtype = torch.float16 if fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    data_identity = dict(
        metadata_identity(entities.metadata, "data", context="RDKit Stage 2 artifact")
    )
    training_identity = build_rdkit_stage2_training_identity(
        config,
        data_identity=data_identity,
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
            context="RDKit Stage 2 script/trainer",
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    completed_epoch = 0
    global_step = 0
    refinement_state: dict[str, Any] | None = None
    checkpoint: Mapping[str, Any] = {}
    if resume_from is None:
        if metrics_path.is_file() and metrics_path.stat().st_size:
            raise FileExistsError(f"Stage 2 output already contains metrics: {metrics_path}")
    else:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        if checkpoint.get("kind") != STAGE2_RDKIT_CHECKPOINT_KIND or checkpoint.get("format_version") != STAGE2_CHECKPOINT_VERSION:
            raise ValueError("Unsupported RDKit Stage 2 checkpoint")
        require_compatible_identity(
            training_identity,
            checkpoint["training_identity"],
            context="RDKit Stage 2 resume",
        )
        expected = {
            "registry_hash": registry.registry_hash,
            "model_contract": model.model_contract,
            "data_identity": data_identity,
            "task_rows": task_rows,
            "task_batches": task_batches,
            "normalized_task_weights": normalized_weights,
            "scheduler_geometry": scheduler_geometry,
            "optimizer_implementation": optimizer_implementation,
            "math_contract": math_contract,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"RDKit Stage 2 checkpoint {key} does not match")
        completed_epoch = int(checkpoint["completed_epoch"])
        global_step = int(checkpoint["global_optimizer_step"])
        expected_step = (
            completed_epoch * steps_per_epoch
            if completed_epoch <= boundary_epoch
            else boundary_epoch * steps_per_epoch + (completed_epoch - boundary_epoch) * refinement_steps
        )
        if not 1 <= completed_epoch <= total_epochs or global_step != expected_step:
            raise ValueError("RDKit Stage 2 checkpoint is not an epoch boundary")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng"])
        stored_refinement = checkpoint.get("refinement")
        expected_phase = (
            "boundary"
            if completed_epoch == boundary_epoch
            else "refinement"
            if completed_epoch > boundary_epoch
            else "unfrozen"
        )
        if checkpoint.get("phase") != expected_phase:
            raise ValueError("RDKit Stage 2 checkpoint phase mismatch")
        if completed_epoch >= boundary_epoch:
            if not isinstance(stored_refinement, dict):
                raise ValueError("RDKit Stage 2 refinement state is missing")
            refinement_state = {
                key: value
                for key, value in stored_refinement.items()
                if key not in {"optimizers", "schedulers"}
            }
            if completed_epoch > boundary_epoch:
                if set(stored_refinement.get("optimizers", {})) != set(refinement_tasks):
                    raise ValueError("RDKit Stage 2 refinement optimizer set mismatch")
                if set(stored_refinement.get("schedulers", {})) != set(refinement_tasks):
                    raise ValueError("RDKit Stage 2 refinement scheduler set mismatch")
                for task in refinement_tasks:
                    refinement_optimizers[task].load_state_dict(stored_refinement["optimizers"][task])
                    refinement_schedulers[task].load_state_dict(stored_refinement["schedulers"][task])
        _reconcile_metrics_for_resume(metrics_path, completed_epoch)
    reporter = ProgressReporter()
    results: list[dict[str, Any]] = []
    validation: dict[str, Any] = {}
    total_epoch_batches = sum(task_batches.values())
    clip_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    for epoch in range(completed_epoch + 1, total_epochs + 1):
        started = time.perf_counter()
        in_refinement = epoch > boundary_epoch
        refinement_epoch = max(0, epoch - boundary_epoch)
        if in_refinement:
            model.set_backbone_trainable(False)
            for parameter in model.object_encoder_parameters():
                parameter.requires_grad_(False)
        else:
            model.set_backbone_trainable(True)
            for parameter in model.object_encoder_parameters():
                parameter.requires_grad_(True)
            model.train()
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
        loss_sum = 0.0
        with reporter.bar(
            total=len(schedule), desc=f"Stage 2 RDKit epoch {epoch}", unit="batch"
        ) as progress:
            for descriptor in schedule:
                active_optimizer = refinement_optimizers[descriptor.task] if in_refinement else optimizer
                if in_refinement:
                    model.set_task_refinement_mode(descriptor.task)
                active_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    output_value = _batch_output(
                        model,
                        descriptor.task,
                        descriptor,
                        train_device[descriptor.task],
                        features,
                        roles,
                        config,
                    )
                    if in_refinement:
                        loss = output_value.physics_loss
                    else:
                        compensation = task_compensation_scale(
                            normalized_weights[descriptor.task],
                            total_epoch_batches,
                            int(descriptor.indices.numel()),
                            task_rows[descriptor.task],
                        )
                        loss = compensation * output_value.physics_loss
                if not bool(torch.isfinite(loss.detach())):
                    raise RuntimeError(f"Non-finite RDKit Stage 2 loss in epoch {epoch}")
                scaler.scale(loss).backward()
                if config.training.max_grad_norm > 0:
                    scaler.unscale_(active_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.task_head_parameters_for(descriptor.task)
                        if in_refinement
                        else clip_parameters,
                        config.training.max_grad_norm,
                    )
                scaler.step(active_optimizer)
                scaler.update()
                if in_refinement:
                    refinement_schedulers[descriptor.task].step()
                    assert refinement_state is not None
                    refinement_state["task_updates"][descriptor.task] += 1
                else:
                    scheduler.step()
                global_step += 1
                loss_sum += float(loss.detach().cpu())
                progress.update(1)
        validation = dict(
            evaluate_rdkit_stage2(
                model,
                registry,
                valid_datasets,
                valid_device,
                features,
                roles,
                entities.metadata["scalers"],
                config,
                shared_frozen=in_refinement,
            )
        )
        phase = "boundary" if epoch == boundary_epoch else "refinement" if in_refinement else "unfrozen"
        validation.update(
            {
                "event": "stage2_rdkit_full_validation",
                "epoch": epoch,
                "refinement_epoch": refinement_epoch if epoch >= boundary_epoch else None,
                "global_optimizer_step": global_step,
                "epoch_wall_seconds": time.perf_counter() - started,
                "phase": phase,
                "train_loss_mean": loss_sum / len(schedule),
            }
        )
        if epoch == boundary_epoch:
            refinement_state = _initial_refinement_state(
                model, registry, refinement_tasks, validation, boundary_epoch
            )
        elif in_refinement:
            assert refinement_state is not None
            _update_refinement_selection(
                refinement_state, model, registry, validation, epoch, refinement_epoch
            )
        _append_metric(metrics_path, validation)
        checkpoint_path = output / f"checkpoint_epoch_{epoch:05d}.pt"
        if checkpoint_path.exists():
            raise FileExistsError(
                f"Stage 2 checkpoint already exists: {checkpoint_path}"
            )
        atomic_torch_save(
            checkpoint_path,
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                config=config,
                registry=registry,
                training_identity=training_identity,
                data_identity=data_identity,
                task_rows=task_rows,
                task_batches=task_batches,
                normalized_weights=normalized_weights,
                scheduler_geometry=scheduler_geometry,
                validation=validation,
                optimizer_implementation=optimizer_implementation,
                math_contract=math_contract,
                phase=phase,
                refinement_state=refinement_state,
                refinement_optimizers=refinement_optimizers if in_refinement else None,
                refinement_schedulers=refinement_schedulers if in_refinement else None,
            ),
        )
        if epoch == boundary_epoch:
            assert refinement_state is not None
            _export_encoder(
                output / "stage2_encoder.pt",
                model=model,
                config=config,
                registry=registry,
                checkpoint_path=checkpoint_path,
                data_identity=data_identity,
                descriptor_contract=entities.metadata["descriptor_contract"],
                refinement_state=refinement_state,
            )
        checkpoint_row = {
            "event": "stage2_checkpoint_complete",
            "epoch": epoch,
            "global_optimizer_step": global_step,
        }
        _append_metric(metrics_path, checkpoint_row)
        results.extend((validation, checkpoint_row))
    if refinement_state is None:
        raise RuntimeError("RDKit Stage 2 refinement boundary was not captured")
    expected_updates = {
        task: task_batches[task] * refinement_epochs for task in refinement_tasks
    }
    if refinement_state["task_updates"] != expected_updates:
        raise RuntimeError("RDKit Stage 2 refinement updates are incomplete")
    if not validation and completed_epoch == total_epochs:
        validation = dict(checkpoint["validation"])
    encoder_path = output / "stage2_encoder.pt"
    if not encoder_path.is_file():
        source = Path(resume_from) if resume_from else output / f"checkpoint_epoch_{boundary_epoch:05d}.pt"
        _export_encoder(
            encoder_path,
            model=model,
            config=config,
            registry=registry,
            checkpoint_path=source,
            data_identity=data_identity,
            descriptor_contract=entities.metadata["descriptor_contract"],
            refinement_state=refinement_state,
        )
    for task in refinement_tasks:
        model.task_head_module(task).load_state_dict(
            refinement_state["selected_tasks"][task]["best_state"], strict=True
        )
    stitched = dict(
        evaluate_rdkit_stage2(
            model,
            registry,
            valid_datasets,
            valid_device,
            features,
            roles,
            entities.metadata["scalers"],
            config,
            shared_frozen=True,
        )
    )
    stitched.update(
        {
            "event": "stage2_taskwise_refined_validation",
            "epoch": total_epochs,
            "refinement_epoch": refinement_epochs,
            "phase": "taskwise_refined",
        }
    )
    manifest = _publish_taskwise_refined(
        output,
        model,
        registry,
        refinement_state,
        stitched,
        training_identity,
        total_epochs,
        artifact_kind=STAGE2_RDKIT_REFINED_KIND,
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
    "STAGE2_RDKIT_CHECKPOINT_KIND",
    "STAGE2_RDKIT_ENCODER_KIND",
    "STAGE2_RDKIT_REFINED_KIND",
    "evaluate_rdkit_stage2",
    "load_rdkit_stage2_encoder_artifact",
    "resolve_rdkit_stage2_training_identity",
    "run_rdkit_stage2_training",
]
