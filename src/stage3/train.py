from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from common.progress import ProgressReporter
from common.identity import (
    IDENTITY_CONTRACT_VERSION,
    require_compatible_identity,
    tensor_state_hash,
)
from common.io import atomic_json, atomic_torch_save
from common.training import (
    canonical_json_sha256,
    capture_rng_state,
    resolve_device,
    restore_rng_state,
    seed_everything,
)
from .config import Stage3Config
from .data import (
    Stage3TaskDataset,
    balanced_virtual_indices,
    composite_steps_per_epoch,
    resolve_group_registry,
    resolve_batch_allocation,
    stable_seed,
)
from .model import GLOBAL, Ownership, Stage3SparseModel, group_owner, private_owner
from .pcgrad import GradientMap, HierarchicalPCGradResult, hierarchical_pcgrad
from .prepare import load_prepared_stage3
from .identity import (
    build_stage3_training_identity,
    metadata_identity,
)


STAGE3_CHECKPOINT_VERSION = 1
STAGE3_CHECKPOINT_KIND = "ilume_stage3_sparse_model"


def checkpoint_epochs(total_epochs: int, interval: int) -> tuple[int, ...]:
    if total_epochs <= 0 or interval <= 0:
        raise ValueError("Checkpoint epoch geometry must be positive")
    epochs = list(range(interval, total_epochs + 1, interval))
    if not epochs or epochs[-1] != total_epochs:
        epochs.append(total_epochs)
    return tuple(epochs)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, allow_nan=True, sort_keys=True) + "\n")


def _resolved_widths(d_model: int, config: Stage3Config) -> dict[str, int]:
    return {
        "d_model": d_model,
        "expert_hidden": max(1, round(d_model * config.model.expert_hidden_ratio)),
        "interaction_hidden": max(
            1, round(d_model * config.model.interaction_hidden_ratio)
        ),
        "film_hidden": max(1, round(d_model * config.model.film_hidden_ratio)),
        "tower_hidden": max(1, round(d_model * config.model.tower_hidden_ratio)),
    }


def _active_tasks(
    config: Stage3Config,
    enabled: Sequence[str],
    source_registry: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    requested = config.training.active_tasks
    if isinstance(requested, tuple):
        active = requested
    elif source_registry is not None and requested in {"auto", "auto_new"}:
        active = tuple(task for task in enabled if task not in source_registry)
    elif requested == "auto_new":
        raise ValueError("training.active_tasks=auto_new requires plugin initialization")
    else:
        active = tuple(enabled)
    if not active:
        raise ValueError("Stage 3 training has no active tasks")
    invalid = set(active) - set(enabled)
    if invalid:
        raise ValueError("Inactive Stage 3 tasks selected: " + ", ".join(sorted(invalid)))
    return tuple(active)


def _scope_matches(scope: str, ownership: str) -> bool:
    if scope == ownership:
        return True
    return scope.endswith(":*") and ownership.startswith(scope[:-1])


def _load_plugin(
    config: Stage3Config,
    model: Stage3SparseModel,
    stage2_encoder_identity: str,
    *,
    fold: int | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    plugin = config.initialization.plugin
    if plugin is None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return None, {"mode": "scratch", "loaded_parameters": []}
    source = torch.load(plugin.checkpoint, map_location="cpu", weights_only=False)
    if (
        source.get("kind") != STAGE3_CHECKPOINT_KIND
        or source.get("format_version") != STAGE3_CHECKPOINT_VERSION
        or source.get("stage") != "stage3"
    ):
        raise ValueError("Plugin requires a Stage 3 v1 sparse checkpoint")
    if source.get("identity_contract_version") != IDENTITY_CONTRACT_VERSION:
        raise ValueError("Plugin checkpoint predates identity contract v1; retrain it")
    if source.get("stage2_encoder_identity") != stage2_encoder_identity:
        raise ValueError("Plugin source and target Stage 2 encoder identities differ")
    if fold is not None and source.get("fold") != fold:
        raise ValueError("Plugin source and target fold differ")
    source_plan = source.get("resolved_training_plan")
    if not isinstance(source_plan, dict):
        raise ValueError("Plugin checkpoint lacks its resolved plan")
    source_training_identity = source.get("training_identity")
    if not isinstance(source_training_identity, Mapping):
        raise ValueError("Plugin checkpoint lacks its training identity")
    require_compatible_identity(
        source_training_identity,
        build_stage3_training_identity(source_plan),
        context="Stage 3 plugin source training identity",
    )
    source_model_config = source_plan.get("model")
    if not isinstance(source_model_config, dict) or any(
        source_model_config.get(name) != value
        for name, value in asdict(config.model).items()
    ):
        raise ValueError("Plugin model structure signature mismatch")
    source_registry = source.get("resolved_registry")
    if not isinstance(source_registry, dict):
        raise ValueError("Plugin checkpoint lacks resolved registry")
    target_registry = {
        task_id: spec.to_dict() for task_id, spec in model.task_specs.items()
    }
    for task_id in set(source_registry) & set(target_registry):
        fields = (
            "system_type", "primary_slots", "partner_slots", "partner_mode",
            "condition_columns", "meta_group",
        )
        if any(source_registry[task_id].get(name) != target_registry[task_id].get(name) for name in fields):
            raise ValueError(f"Plugin task structure mismatch: {task_id}")
    if set(source_registry) - set(target_registry):
        raise ValueError("Plugin target registry cannot remove source tasks")
    source_manifest = source.get("ownership_manifest")
    target_manifest = model.ownership_manifest()
    if not isinstance(source_manifest, dict):
        raise ValueError("Plugin checkpoint lacks ownership manifest")
    source_state = source.get("model")
    if not isinstance(source_state, dict):
        raise ValueError("Plugin checkpoint lacks model state")
    if source.get("model_state_hash") != tensor_state_hash(
        "stage3.model-state", source_state
    ):
        raise ValueError("Plugin checkpoint model state hash mismatch")
    selected_names = {
        name
        for name, owner in source_manifest.items()
        if any(_scope_matches(scope, owner) for scope in plugin.load_scopes)
    }
    for scope in plugin.load_scopes:
        if not any(_scope_matches(scope, owner) for owner in source_manifest.values()):
            raise ValueError(f"Plugin load scope does not exist: {scope}")
    target_state = model.state_dict()
    for name in sorted(selected_names):
        if name not in source_state or name not in target_state:
            raise ValueError(f"Plugin parameter key mismatch: {name}")
        if source_manifest[name] != target_manifest.get(name):
            raise ValueError(f"Plugin ownership mismatch: {name}")
        if source_state[name].shape != target_state[name].shape:
            raise ValueError(f"Plugin parameter shape mismatch: {name}")
        target_state[name] = source_state[name]
    model.load_state_dict(target_state, strict=True)

    source_groups = {value["meta_group"] for value in source_registry.values()}
    target_groups = {value["meta_group"] for value in target_registry.values()}
    new_groups = target_groups - source_groups
    new_tasks = set(target_registry) - set(source_registry)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    ownership = model.parameter_ownership()
    for parameter, owner in ownership.items():
        if (
            (owner.scope == "GROUP" and owner.owner_id in new_groups)
            or (owner.scope == "PRIVATE" and owner.owner_id in new_tasks)
        ):
            parameter.requires_grad_(True)

    adaptation = plugin.adaptation
    requested_owners: list[Ownership] = []
    if adaptation.global_scope:
        requested_owners.append(GLOBAL)
    requested_owners.extend(group_owner(group) for group in adaptation.groups)
    requested_owners.extend(private_owner(task) for task in adaptation.private_tasks)
    selected_owners = {source_manifest[name] for name in selected_names}
    for owner in requested_owners:
        if owner.label not in selected_owners:
            raise ValueError(f"Plugin adaptation scope was not loaded: {owner.label}")
        for parameter in model.parameters_for_owner(owner):
            parameter.requires_grad_(True)
    return source, {
        "mode": "plugin",
        "source_training_identity": source_training_identity["hash"],
        "source_model_state_hash": source["model_state_hash"],
        "load_scopes": list(plugin.load_scopes),
        "adaptation": {
            "global": adaptation.global_scope,
            "groups": list(adaptation.groups),
            "private_tasks": list(adaptation.private_tasks),
        },
        "new_groups": sorted(new_groups),
        "new_tasks": sorted(new_tasks),
        "loaded_parameters": sorted(selected_names),
    }


def _validate_adaptation(
    config: Stage3Config,
    model: Stage3SparseModel,
    active_tasks: Sequence[str],
) -> None:
    active_groups = {model.task_specs[task].meta_group for task in active_tasks}
    for parameter, owner in model.parameter_ownership().items():
        if (
            (owner.scope == "PRIVATE" and owner.owner_id not in active_tasks)
            or (owner.scope == "GROUP" and owner.owner_id not in active_groups)
        ):
            parameter.requires_grad_(False)
    plugin = config.initialization.plugin
    if plugin is None:
        return
    adaptation = plugin.adaptation
    unused_groups = set(adaptation.groups) - active_groups
    unused_private = set(adaptation.private_tasks) - set(active_tasks)
    if unused_groups or unused_private:
        raise ValueError("Plugin adaptation scope is unused by active tasks")
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Plugin training has no trainable parameters")


def _normalization_for_run(
    prepared: Mapping[str, Any],
    fold: int,
    source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    current = prepared["normalization"][f"fold{fold}"]
    if source is None:
        return current
    inherited = source.get("normalization")
    source_registry = source["resolved_registry"]
    if not isinstance(inherited, dict):
        raise ValueError("Plugin checkpoint lacks normalization")
    return {
        task: inherited[task] if task in source_registry else stats
        for task, stats in current.items()
    }


def build_resolved_training_plan(
    config: Stage3Config,
    fold: int,
    model: Stage3SparseModel,
    datasets: Mapping[str, Stage3TaskDataset],
    active_tasks: Sequence[str],
    prepared: Mapping[str, Any],
    plugin_plan: Mapping[str, Any],
    normalizations: Mapping[str, Any],
) -> dict[str, Any]:
    counts = {task: len(datasets[task]) for task in active_tasks}
    allocation = resolve_batch_allocation(
        counts, config.training.composite_batch_size, config.training.virtual_min_size
    )
    steps = composite_steps_per_epoch(
        counts, allocation, config.training.virtual_min_size
    )
    total_steps = config.training.epochs * steps
    warmup_steps = math.ceil(config.training.warmup_ratio * total_steps)
    virtual = {task: max(counts[task], config.training.virtual_min_size) for task in active_tasks}
    return {
        "format_version": 1,
        "fold": fold,
        "active_tasks": list(active_tasks),
        "resolved_registry": {
            task_id: spec.to_dict() for task_id, spec in model.task_specs.items()
        },
        "groups": {
            group: spec.to_dict()
            for group, spec in resolve_group_registry(config).items()
        },
        "data": {
            "N_t": counts,
            "N_prime_t": virtual,
            "B_t": allocation,
            "K": steps,
            "padded_sizes": {task: steps * allocation[task] for task in active_tasks},
            "replication_ratios": {
                task: steps * allocation[task] / counts[task] for task in active_tasks
            },
        },
        "model": {**asdict(config.model), "resolved_widths": _resolved_widths(model.d_model, config)},
        "optimizer": {
            "name": "AdamW", "implementation": config.training.optimizer_implementation,
            "lr": config.training.learning_rate, "weight_decay": config.training.weight_decay,
            "betas": list(config.training.betas), "eps": config.training.eps,
        },
        "scheduler": {
            "name": "linear_warmup_cosine", "warmup_steps": warmup_steps,
            "total_steps": total_steps, "min_lr_ratio": config.training.min_lr_ratio,
        },
        "math": {
            "precision": config.training.amp_dtype,
            "smooth_l1_beta": config.training.smooth_l1_beta,
            "max_grad_norm": config.training.max_grad_norm,
            "microbatch_size": config.training.microbatch_size,
            "pcgrad": "hierarchical_ownership_blocks_v1",
        },
        "stage2_encoder_identity": metadata_identity(
            prepared["metadata"], "stage2_encoder", context="Stage 3 prepared artifact"
        )["hash"],
        "prepared_identity": metadata_identity(
            prepared["metadata"], "prepared", context="Stage 3 prepared artifact"
        )["hash"],
        "normalization_hash": canonical_json_sha256(normalizations),
        "ownership_manifest": model.ownership_manifest(),
        "plugin": dict(plugin_plan),
        "trainable_parameters": sorted(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ),
        "frozen_parameters": sorted(
            name
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        ),
        "execution": {
            "checkpoint_interval_epochs": config.training.checkpoint_interval_epochs,
            "device": config.training.device,
            "cpu_threads": config.training.cpu_threads,
            "cpu_interop_threads": config.training.cpu_interop_threads,
            "debug_pcgrad_traces": config.training.debug_pcgrad_traces,
        },
    }


def _optimizer(model: nn.Module, config: Stage3Config) -> torch.optim.AdamW:
    decay = []
    no_decay = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    if not decay and not no_decay:
        raise ValueError("Stage 3 optimizer has no trainable parameters")
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.training.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.training.learning_rate,
        betas=config.training.betas,
        eps=config.training.eps,
        foreach=False,
        fused=False,
    )


def _lr_factor(step: int, warmup: int, total: int, floor: float) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return floor + (1.0 - floor) * cosine


def _batch(
    dataset: Stage3TaskDataset,
    indices: torch.Tensor,
    embeddings: torch.Tensor,
    normalization: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    cpu_indices = indices.cpu()
    primary = embeddings[dataset.primary_object_ids[cpu_indices]].to(device)
    partner_ids = dataset.partner_object_ids[cpu_indices]
    partner = (
        embeddings[partner_ids].to(device)
        if len(partner_ids) and bool((partner_ids >= 0).all())
        else None
    )
    conditions = dataset.conditions[cpu_indices].to(device)
    target_stats = normalization["target"]
    targets = (
        dataset.raw_targets[cpu_indices].to(device) - float(target_stats["mean"])
    ) / float(target_stats["scale"])
    return primary, conditions, partner, targets


def compute_task_gradient(
    model: Stage3SparseModel,
    task_id: str,
    dataset: Stage3TaskDataset,
    indices: torch.Tensor,
    embeddings: torch.Tensor,
    normalization: Mapping[str, Any],
    config: Stage3Config,
    device: torch.device,
) -> tuple[GradientMap, float]:
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    accumulated: GradientMap = {}
    loss_total = 0.0
    task_batch_size = len(indices)
    for start in range(0, task_batch_size, config.training.microbatch_size):
        micro = indices[start : start + config.training.microbatch_size]
        primary, conditions, partner, targets = _batch(
            dataset, micro, embeddings, normalization, device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=config.training.amp_dtype == "bf16",
        ):
            predictions = model(
                task_id, primary, conditions, partner_embedding=partner
            ).predictions
            if not torch.isfinite(predictions).all():
                raise RuntimeError(f"Non-finite Stage 3 prediction: {task_id}")
            loss_sum = F.smooth_l1_loss(
                predictions, targets, beta=config.training.smooth_l1_beta, reduction="sum"
            )
            loss = loss_sum / task_batch_size
        gradients = torch.autograd.grad(
            loss, parameters, allow_unused=True, materialize_grads=False
        )
        for parameter, gradient in zip(parameters, gradients, strict=True):
            if gradient is not None:
                value = gradient.detach().float()
                accumulated[parameter] = accumulated.get(parameter, torch.zeros_like(value)) + value
        loss_total += float(loss_sum.detach().float().cpu())
    return accumulated, loss_total / task_batch_size


def regression_metrics(
    normalized_predictions: torch.Tensor,
    normalized_targets: torch.Tensor,
    normalization: Mapping[str, Any],
) -> dict[str, Any]:
    predictions = normalized_predictions.double()
    targets = normalized_targets.double()
    count = int(targets.numel())
    if count == 0:
        return {"count": 0, "reason": "no_samples"}
    delta = predictions - targets
    normalized_mae = float(delta.abs().mean())
    normalized_rmse = float(delta.square().mean().sqrt())
    scale = float(normalization["target"]["scale"])
    raw_predictions = predictions * scale + float(normalization["target"]["mean"])
    raw_targets = targets * scale + float(normalization["target"]["mean"])
    raw_delta = raw_predictions - raw_targets
    centered_targets = raw_targets - raw_targets.mean()
    denominator = float(centered_targets.square().sum())
    r2 = float("nan") if denominator == 0.0 else 1.0 - float(raw_delta.square().sum()) / denominator
    if count < 2 or float(raw_predictions.std(unbiased=False)) == 0.0 or float(raw_targets.std(unbiased=False)) == 0.0:
        pearson = float("nan")
        pearson_reason = "insufficient_or_constant_samples"
    else:
        pearson = float(torch.corrcoef(torch.stack((raw_predictions, raw_targets)))[0, 1])
        pearson_reason = None
    return {
        "count": count,
        "mae": float(raw_delta.abs().mean()), "rmse": float(raw_delta.square().mean().sqrt()),
        "r2": r2, "r2_reason": "constant_target" if math.isnan(r2) else None,
        "pearson_r": pearson, "pearson_reason": pearson_reason,
        "normalized_mae": normalized_mae, "normalized_rmse": normalized_rmse,
    }


@torch.no_grad()
def validate_tasks(
    model: Stage3SparseModel,
    datasets: Mapping[str, Stage3TaskDataset],
    embeddings: torch.Tensor,
    normalizations: Mapping[str, Any],
    config: Stage3Config,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    per_task: dict[str, Any] = {}
    for task_id, dataset in datasets.items():
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for start in range(0, len(dataset), config.training.microbatch_size):
            indices = torch.arange(start, min(len(dataset), start + config.training.microbatch_size))
            primary, conditions, partner, target = _batch(
                dataset, indices, embeddings, normalizations[task_id], device
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.training.amp_dtype == "bf16"):
                prediction = model(task_id, primary, conditions, partner_embedding=partner).predictions
            if not torch.isfinite(prediction).all():
                raise RuntimeError(
                    f"Non-finite Stage 3 validation prediction: {task_id}"
                )
            predictions.append(prediction.float().cpu())
            targets.append(target.float().cpu())
        per_task[task_id] = regression_metrics(
            torch.cat(predictions), torch.cat(targets), normalizations[task_id]
        )
    metrics = ("mae", "rmse", "r2", "pearson_r", "normalized_mae", "normalized_rmse")
    macro_task: dict[str, Any] = {}
    macro_group: dict[str, Any] = {}
    for metric in metrics:
        valid = {task: value[metric] for task, value in per_task.items() if metric in value and math.isfinite(value[metric])}
        macro_task[metric] = {
            "value": sum(valid.values()) / len(valid) if valid else float("nan"),
            "valid_tasks": len(valid), "total_tasks": len(per_task),
        }
        group_values = []
        for group in sorted({model.task_specs[task].meta_group for task in per_task}):
            values = [valid[task] for task in valid if model.task_specs[task].meta_group == group]
            if values:
                group_values.append(sum(values) / len(values))
        macro_group[metric] = {
            "value": sum(group_values) / len(group_values) if group_values else float("nan"),
            "valid_groups": len(group_values),
            "total_groups": len({model.task_specs[task].meta_group for task in per_task}),
        }
    return {"tasks": per_task, "macro_task_equal": macro_task, "macro_group_equal": macro_group}


def _pair_matrix(names: Sequence[str], diagnostics: Mapping[tuple[str, str], Any]) -> dict[str, Any]:
    cosine = [[float("nan") for _ in names] for _ in names]
    conflict = [[float("nan") for _ in names] for _ in names]
    projection = [[float("nan") for _ in names] for _ in names]
    applicable = [[False for _ in names] for _ in names]
    positions = {name: index for index, name in enumerate(names)}
    for (left, right), value in diagnostics.items():
        row, column = positions[left], positions[right]
        cosine[row][column] = value.cosine
        conflict[row][column] = float(value.conflict)
        projection[row][column] = value.projection_norm
        applicable[row][column] = True
    return {"names": list(names), "cosine": cosine, "conflict": conflict, "projection_norm": projection, "applicable": applicable}


def _checkpoint_payload(
    config: Stage3Config,
    fold: int,
    epoch: int,
    global_step: int,
    model: Stage3SparseModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    plan: Mapping[str, Any],
    normalizations: Mapping[str, Any],
    pcgrad_rng: random.Random,
    task_order_rng: random.Random,
) -> dict[str, Any]:
    model_state = model.state_dict()
    return {
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "format_version": STAGE3_CHECKPOINT_VERSION, "kind": STAGE3_CHECKPOINT_KIND,
        "stage": "stage3",
        "config": config.to_dict(), "fold": fold, "completed_epoch": epoch,
        "global_step": global_step, "model": model_state,
        "model_state_hash": tensor_state_hash("stage3.model-state", model_state),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "rng": capture_rng_state(), "pcgrad_rng": pcgrad_rng.getstate(),
        "task_order_rng": task_order_rng.getstate(),
        "stage2_encoder_identity": plan["stage2_encoder_identity"],
        "training_identity": build_stage3_training_identity(plan),
        "resolved_registry": plan["resolved_registry"], "resolved_training_plan": dict(plan),
        "normalization": dict(normalizations),
        "ownership_manifest": model.ownership_manifest(),
        "data_provenance": plan["data"], "math_contract": plan["math"],
        "optimizer_contract": plan["optimizer"], "scheduler_contract": plan["scheduler"],
    }


def run_stage3_training(
    config: Stage3Config,
    fold: int,
    *,
    output_dir: str | Path,
    resume_from: str | Path | None = None,
    expected_training_identity: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if fold not in range(1, 6):
        raise ValueError("Stage 3 fold must be in 1..5")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Stage 3 v1 training supports one process only")
    device = resolve_device(config.training.device)
    if config.training.amp_dtype == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Stage 3 BF16 requires a BF16-capable CUDA GPU")
    seed_everything(config.data.seed + fold)
    prepared = load_prepared_stage3(config)
    embedding_matrix = prepared["objects"]["embeddings"].float()
    d_model = int(embedding_matrix.shape[1])
    registry = prepared["registry"]
    model = Stage3SparseModel(config.model, registry, d_model).to(device)
    source, plugin_plan = _load_plugin(
        config,
        model,
        metadata_identity(
            prepared["metadata"], "stage2_encoder", context="Stage 3 prepared artifact"
        )["hash"],
        fold=fold,
    )
    enabled = tuple(task for task, spec in registry.items() if spec.enabled)
    active = _active_tasks(config, enabled, source.get("resolved_registry") if source else None)
    _validate_adaptation(config, model, active)
    train_data = {task: Stage3TaskDataset(config.data.artifacts_dir, fold, task, "train") for task in active}
    valid_data = {task: Stage3TaskDataset(config.data.artifacts_dir, fold, task, "valid") for task in active}
    normalizations = _normalization_for_run(prepared, fold, source)
    plan = build_resolved_training_plan(
        config, fold, model, train_data, active, prepared, plugin_plan,
        normalizations,
    )
    training_identity = build_stage3_training_identity(plan)
    if expected_training_identity is not None:
        require_compatible_identity(
            expected_training_identity,
            training_identity,
            context="Stage 3 run-directory training identity",
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if resume_from is None and (
        (output / "metrics.jsonl").exists()
        or (output / "diagnostics.jsonl").exists()
        or any(output.glob("checkpoint_epoch_*.pt"))
    ):
        raise FileExistsError("Stage 3 training output already contains run state")
    plan_path = output / "resolved_training_plan.json"
    if plan_path.exists():
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        require_compatible_identity(
            build_stage3_training_identity(existing_plan),
            training_identity,
            context="Existing Stage 3 resolved training plan",
        )
    else:
        atomic_json(plan_path, plan)
    optimizer = _optimizer(model, config)
    scheduler_info = plan["scheduler"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_factor(step, scheduler_info["warmup_steps"], scheduler_info["total_steps"], scheduler_info["min_lr_ratio"]),
    )
    pcgrad_rng = random.Random(stable_seed(config.data.seed, fold, "pcgrad"))
    task_order_rng = random.Random(
        stable_seed(config.data.seed, fold, "task_order")
    )
    start_epoch = 1
    global_step = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        expected = {
            "identity_contract_version": IDENTITY_CONTRACT_VERSION,
            "kind": STAGE3_CHECKPOINT_KIND, "format_version": STAGE3_CHECKPOINT_VERSION,
            "stage": "stage3",
            "fold": fold,
            "ownership_manifest": model.ownership_manifest(),
            "stage2_encoder_identity": plan["stage2_encoder_identity"],
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"Stage 3 resume contract mismatch: {key}")
        checkpoint_identity = checkpoint.get("training_identity")
        if not isinstance(checkpoint_identity, Mapping):
            raise ValueError(
                "Stage 3 checkpoint predates identity contract v1; retrain it"
            )
        require_compatible_identity(
            training_identity,
            checkpoint_identity,
            context="Stage 3 resume training identity",
        )
        if checkpoint.get("model_state_hash") != tensor_state_hash(
            "stage3.model-state", checkpoint["model"]
        ):
            raise ValueError("Stage 3 resume model state hash mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng_state(checkpoint["rng"])
        pcgrad_rng.setstate(checkpoint["pcgrad_rng"])
        task_order_rng.setstate(checkpoint["task_order_rng"])
        start_epoch = int(checkpoint["completed_epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        metrics_path = output / "metrics.jsonl"
        if not metrics_path.is_file():
            raise FileNotFoundError("Stage 3 resume requires metrics.jsonl")
        recorded = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
        if [row["epoch"] for row in recorded] != list(range(1, start_epoch)):
            raise ValueError("Stage 3 resume metrics history is incomplete or has future rows")
        diagnostics_path = output / "diagnostics.jsonl"
        if not diagnostics_path.is_file():
            raise FileNotFoundError("Stage 3 resume requires diagnostics.jsonl")
        diagnostics_rows = [
            json.loads(line)
            for line in diagnostics_path.read_text().splitlines()
            if line
        ]
        if [row["epoch"] for row in diagnostics_rows] != list(range(1, start_epoch)):
            raise ValueError(
                "Stage 3 resume diagnostics history is incomplete or has future rows"
            )
    metrics_path = output / "metrics.jsonl"
    diagnostics_path = output / "diagnostics.jsonl"
    rows: list[dict[str, Any]] = []
    counts = plan["data"]["N_t"]
    allocation = plan["data"]["B_t"]
    steps_per_epoch = plan["data"]["K"]
    group_weights = {
        group: config.groups[group].group_weight
        for group in config.groups
    }

    total_steps = config.training.epochs * steps_per_epoch
    completed_steps = (start_epoch - 1) * steps_per_epoch

    progress = ProgressReporter().bar(
        total=total_steps,
        initial=completed_steps,
        desc=f"Stage3 fold{fold}",
        unit="step",
    )

    try:
        for epoch in range(start_epoch, config.training.epochs + 1):
            progress.set_description(
                f"Stage3 fold{fold} epoch {epoch}/{config.training.epochs}"
            )
            model.train()
            sequences = {
                task: balanced_virtual_indices(
                    counts[task], steps_per_epoch * allocation[task],
                    seed=config.data.seed, epoch=epoch, task_id=task,
                )
                for task in active
            }
            epoch_loss = {task: 0.0 for task in active}
            latest_pcgrad: HierarchicalPCGradResult | None = None
            pre_norm = post_norm = 0.0
            for step in range(steps_per_epoch):
                order = list(active)
                task_order_rng.shuffle(order)
                task_gradients: dict[str, GradientMap] = {}
                for task in order:
                    begin = step * allocation[task]
                    indices = sequences[task][begin : begin + allocation[task]]
                    gradient, loss = compute_task_gradient(
                        model, task, train_data[task], indices, embedding_matrix,
                        normalizations[task], config, device,
                    )
                    task_gradients[task] = gradient
                    epoch_loss[task] += loss
                latest_pcgrad = hierarchical_pcgrad(
                    model, task_gradients, registry, group_weights, pcgrad_rng
                )
                optimizer.zero_grad(set_to_none=True)
                for parameter, gradient in latest_pcgrad.gradients.items():
                    if parameter.requires_grad:
                        parameter.grad = gradient.to(parameter.device, dtype=parameter.dtype)
                trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
                pre_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        trainable, float("inf"), error_if_nonfinite=True
                    )
                )
                if config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable,
                        config.training.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                post_norm = min(pre_norm, config.training.max_grad_norm) if config.training.max_grad_norm > 0 else pre_norm
                optimizer.step()
                scheduler.step()
                global_step += 1

                mean_train_loss = sum(epoch_loss.values()) / (
                    len(active) * (step + 1)
                )

                progress.set_postfix(
                    {
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        "loss": f"{mean_train_loss:.4f}",
                    }
                )
                progress.update(1)
            validation = validate_tasks(
                model,
                valid_data,
                embedding_matrix,
                normalizations,
                config,
                device,
            )

            val_mae = validation["macro_task_equal"]["mae"]["value"]

            mean_epoch_loss = sum(
                epoch_loss[task] / steps_per_epoch
                for task in active
            ) / len(active)

            progress.set_postfix(
                {
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "loss": f"{mean_epoch_loss:.4f}",
                    "val_mae": f"{val_mae:.4f}",
                }
            )
            row = {
                "epoch": epoch, "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "training_loss": {task: epoch_loss[task] / steps_per_epoch for task in active},
                "validation": validation,
            }
            _append_jsonl(metrics_path, row)
            rows.append(row)
            assert latest_pcgrad is not None
            groups = sorted({registry[task].meta_group for task in active})
            _append_jsonl(
                diagnostics_path,
                {
                    "epoch": epoch,
                    "task_level_global": _pair_matrix(list(active), latest_pcgrad.task_global),
                    "task_level_group": _pair_matrix(list(active), latest_pcgrad.task_group),
                    "group_level_global": _pair_matrix(groups, latest_pcgrad.group_global),
                    "task_gradient_norms": latest_pcgrad.task_norms,
                    "task_global_norms": latest_pcgrad.task_global_norms,
                    "task_group_norms": latest_pcgrad.task_group_norms,
                    "private_norms": latest_pcgrad.private_norms,
                    "group_global_norms": latest_pcgrad.group_global_norms,
                    "assembled_owner_norms": latest_pcgrad.assembled_owner_norms,
                    "clip_pre_norm": pre_norm, "clip_post_norm": post_norm,
                },
            )
            if epoch in checkpoint_epochs(
                config.training.epochs, config.training.checkpoint_interval_epochs
            ):
                path = output / f"checkpoint_epoch_{epoch:05d}.pt"
                if path.exists():
                    raise FileExistsError(f"Stage 3 checkpoint already exists: {path}")
                atomic_torch_save(
                    path,
                    _checkpoint_payload(
                        config, fold, epoch, global_step, model, optimizer,
                        scheduler, plan, normalizations, pcgrad_rng,
                        task_order_rng,
                    ),
                )
    finally:
        progress.close()
    return rows


def resolve_stage3_training_identity(
    config: Stage3Config, fold: int
) -> dict[str, Any]:
    """Resolve the exact semantic identity used by ``run_stage3_training``."""
    if fold not in range(1, 6):
        raise ValueError("Stage 3 fold must be in 1..5")
    prepared = load_prepared_stage3(config)
    d_model = int(prepared["objects"]["embeddings"].shape[1])
    model = Stage3SparseModel(config.model, prepared["registry"], d_model)
    encoder_identity = metadata_identity(
        prepared["metadata"], "stage2_encoder", context="Stage 3 prepared artifact"
    )["hash"]
    source, plugin_plan = _load_plugin(
        config, model, encoder_identity, fold=fold
    )
    enabled = tuple(
        task for task, spec in prepared["registry"].items() if spec.enabled
    )
    active = _active_tasks(
        config, enabled, source.get("resolved_registry") if source else None
    )
    _validate_adaptation(config, model, active)
    datasets = {
        task: Stage3TaskDataset(config.data.artifacts_dir, fold, task, "train")
        for task in active
    }
    normalizations = _normalization_for_run(prepared, fold, source)
    plan = build_resolved_training_plan(
        config,
        fold,
        model,
        datasets,
        active,
        prepared,
        plugin_plan,
        normalizations,
    )
    return build_stage3_training_identity(plan)


__all__ = [
    "STAGE3_CHECKPOINT_KIND",
    "STAGE3_CHECKPOINT_VERSION",
    "build_resolved_training_plan",
    "checkpoint_epochs",
    "compute_task_gradient",
    "regression_metrics",
    "resolve_stage3_training_identity",
    "run_stage3_training",
    "validate_tasks",
]
