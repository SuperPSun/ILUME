from __future__ import annotations

import json
import math
import os
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
from common.io import sha256_file
from common.refinement import (
    TASKWISE_REFINED_FORMAT_VERSION,
    refinement_geometry,
)
from common.training import (
    canonical_json_sha256,
    resolve_device,
    seed_everything,
)
from .config import Stage3Config, effective_training_seed
from .data import (
    STAGE3_ARTIFACT_KIND,
    Stage3RepresentationStore,
    Stage3TaskDataset,
    composite_steps_per_epoch,
    raw_task_steps,
    resolve_group_registry,
    resolve_batch_allocation,
    resolve_raw_batch_allocation,
)
from .model import (
    GLOBAL,
    Ownership,
    Stage3SparseModel,
    group_owner,
    private_owner,
    summarize_task_gate_observations,
    task_gate_observations,
)
from .gradient_assembly import GradientMap
from .prepare import load_prepared_stage3
from .identity import (
    build_stage3_training_identity,
    metadata_identity,
)


STAGE3_CHECKPOINT_VERSION = 2
STAGE3_CHECKPOINT_KIND = "ilume_stage3_sparse_model"
STAGE3_REFINED_KIND = "ilume_stage3_taskwise_refined"
STAGE3_RDKIT_CHECKPOINT_KIND = "ilume_stage3_rdkit_home_model"
STAGE3_RDKIT_REFINED_KIND = "ilume_stage3_rdkit_home_taskwise_refined"


def checkpoint_epochs(total_epochs: int, interval: int) -> tuple[int, ...]:
    if total_epochs <= 0 or interval <= 0:
        raise ValueError("Checkpoint epoch geometry must be positive")
    epochs = list(range(interval, total_epochs + 1, interval))
    if not epochs or epochs[-1] != total_epochs:
        epochs.append(total_epochs)
    return tuple(epochs)


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


def _resolved_private_recipes(config: Stage3Config) -> dict[str, Any]:
    if config.training.schedule_mode != "three_phase":
        return {}
    return {
        task: config.resolved_private_recipe(task)
        for task, task_config in config.tasks.items()
        if task_config.enabled
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
    if config.training.sampling_mode == "raw":
        allocation = resolve_raw_batch_allocation(
            counts, config.training.composite_batch_size
        )
        task_steps = raw_task_steps(counts, allocation)
        steps = max(task_steps.values())
        data_plan = {
            "sampling": "raw_without_replacement_v1",
            "N_t": counts,
            "B_t": allocation,
            "task_steps": task_steps,
            "K": steps,
            "epoch_exposures": dict(counts),
            "effective_composite_batch_size": sum(allocation.values()),
        }
    else:
        allocation = resolve_batch_allocation(
            counts,
            config.training.composite_batch_size,
            config.training.virtual_min_size,
        )
        steps = composite_steps_per_epoch(
            counts, allocation, config.training.virtual_min_size
        )
        virtual = {
            task: max(counts[task], config.training.virtual_min_size)
            for task in active_tasks
        }
        data_plan = {
            "N_t": counts,
            "N_prime_t": virtual,
            "B_t": allocation,
            "K": steps,
            "padded_sizes": {
                task: steps * allocation[task] for task in active_tasks
            },
            "replication_ratios": {
                task: steps * allocation[task] / counts[task]
                for task in active_tasks
            },
        }
    three_phase = config.training.schedule_mode == "three_phase"
    if three_phase:
        recipe = config.training.three_phase
        assert recipe is not None
        task_steps = data_plan["task_steps"]
        active_groups = sorted(
            {model.task_specs[task].meta_group for task in active_tasks}
        )
        group_steps = {
            group: max(
                int(task_steps[task])
                for task in active_tasks
                if model.task_specs[task].meta_group == group
            )
            for group in active_groups
        }

        def owner_recipe(
            *, lr: float, nominal_epochs: int, effective_epochs: int,
            updates_per_epoch: int, floor: float, warmup_updates: int = 0,
            **extra: Any,
        ) -> dict[str, Any]:
            return {
                "nominal_lr": lr,
                "terminal_lr": lr * floor,
                "nominal_epochs": nominal_epochs,
                "effective_epochs": effective_epochs,
                "freeze_epoch": effective_epochs,
                "updates_per_epoch": updates_per_epoch,
                "actual_update_budget": effective_epochs * updates_per_epoch,
                "warmup_updates": warmup_updates,
                **extra,
            }

        phase1_owners: dict[str, Any] = {
            "GLOBAL": owner_recipe(
                lr=recipe.global_scope.lr,
                nominal_epochs=recipe.global_scope.epochs,
                effective_epochs=recipe.global_scope.epochs,
                updates_per_epoch=steps,
                floor=recipe.global_scope.min_lr_ratio,
                warmup_updates=math.ceil(
                    recipe.global_scope.warmup_ratio
                    * recipe.global_scope.epochs
                    * steps
                ),
                capacity={
                    "experts": config.model.global_experts,
                    "expert_hidden_ratio": config.model.expert_hidden_ratio,
                },
            )
        }
        for group in active_groups:
            budget = config.groups[group].phase1
            assert budget is not None
            phase1_owners[f"GROUP:{group}"] = owner_recipe(
                lr=budget.lr, nominal_epochs=budget.epochs,
                effective_epochs=budget.epochs,
                updates_per_epoch=group_steps[group],
                floor=recipe.phase1_min_lr_ratio,
                capacity=model.resolved_capacity_recipe()["groups"][group],
            )
        for task in active_tasks:
            task_config = config.tasks[task]
            private_recipe = config.resolved_private_recipe(task)
            phase1_owners[f"PRIVATE:{task}"] = owner_recipe(
                lr=private_recipe.phase1_lr,
                nominal_epochs=private_recipe.phase1_epochs,
                effective_epochs=private_recipe.phase1_epochs,
                updates_per_epoch=int(task_steps[task]),
                floor=recipe.phase1_min_lr_ratio,
                size_class=task_config.size_class,
                unique_systems=task_config.unique_systems,
                capacity=model.resolved_capacity_recipe()["tasks"][task],
            )
        phase2_branches: dict[str, Any] = {}
        for group in active_groups:
            group_budget = config.groups[group].phase2
            assert group_budget is not None
            owners = {
                f"GROUP:{group}": owner_recipe(
                    lr=group_budget.lr,
                    nominal_epochs=group_budget.epochs,
                    effective_epochs=group_budget.epochs,
                    updates_per_epoch=group_steps[group],
                    floor=recipe.phase2_min_lr_ratio,
                    capacity=model.resolved_capacity_recipe()["groups"][group],
                )
            }
            for task in active_tasks:
                if model.task_specs[task].meta_group != group:
                    continue
                task_config = config.tasks[task]
                private_recipe = config.resolved_private_recipe(task)
                effective = min(private_recipe.phase2_epochs, group_budget.epochs)
                owners[f"PRIVATE:{task}"] = owner_recipe(
                    lr=private_recipe.phase2_lr,
                    nominal_epochs=private_recipe.phase2_epochs,
                    effective_epochs=effective,
                    updates_per_epoch=int(task_steps[task]),
                    floor=recipe.phase2_min_lr_ratio,
                    size_class=task_config.size_class,
                    unique_systems=task_config.unique_systems,
                    capacity=model.resolved_capacity_recipe()["tasks"][task],
                )
            phase2_branches[group] = {
                "epochs": group_budget.epochs,
                "steps_per_epoch": group_steps[group],
                "owners": owners,
            }
        phase3_branches = {}
        for task in active_tasks:
            task_config = config.tasks[task]
            private_recipe = config.resolved_private_recipe(task)
            phase3_branches[task] = {
                "epochs": private_recipe.phase3_epochs,
                "steps_per_epoch": int(task_steps[task]),
                "carried_from_anchor": private_recipe.phase3_epochs == 0,
                "owners": {
                    f"PRIVATE:{task}": owner_recipe(
                        lr=private_recipe.phase3_lr,
                        nominal_epochs=private_recipe.phase3_epochs,
                        effective_epochs=private_recipe.phase3_epochs,
                        updates_per_epoch=int(task_steps[task]),
                        floor=recipe.phase3_min_lr_ratio,
                        size_class=task_config.size_class,
                        unique_systems=task_config.unique_systems,
                        capacity=model.resolved_capacity_recipe()["tasks"][task],
                    )
                },
            }
        phase_plan = {
            "phase1": {
                "epochs": recipe.global_scope.epochs,
                "steps_per_epoch": steps,
                "owners": phase1_owners,
            },
            "phase2": {"branches": phase2_branches},
            "phase3": {"branches": phase3_branches},
        }
    else:
        boundary_epoch, refinement_epochs = refinement_geometry(
            config.training.epochs, config.training.refinement_ratio
        )
        total_steps = boundary_epoch * steps
        warmup_steps = math.ceil(config.training.warmup_ratio * total_steps)
    plan = {
        "format_version": 4 if three_phase else 1,
        "fold": fold,
        "active_tasks": list(active_tasks),
        "resolved_registry": {
            task_id: spec.to_dict() for task_id, spec in model.task_specs.items()
        },
        "groups": {
            group: spec.to_dict()
            for group, spec in resolve_group_registry(config).items()
        },
        "data": data_plan,
        "model": {**asdict(config.model), "resolved_widths": _resolved_widths(model.d_model, config)},
        "optimizer": {
            "name": "AdamW", "implementation": config.training.optimizer_implementation,
            "weight_decay": config.training.weight_decay,
            "betas": list(config.training.betas), "eps": config.training.eps,
        },
        "math": {
            "precision": config.training.amp_dtype,
            "smooth_l1_beta": config.training.smooth_l1_beta,
            "max_grad_norm": config.training.max_grad_norm,
            "microbatch_size": config.training.microbatch_size,
        },
        "prepared_identity": metadata_identity(
            prepared["metadata"], "prepared", context="Stage 3 prepared artifact"
        )["hash"],
        "normalization_hash": canonical_json_sha256(normalizations),
        "ownership_manifest": model.ownership_manifest(),
        "plugin": dict(plugin_plan),
        "trainable_parameters": sorted(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ),
        "parameter_counts": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
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
        },
    }
    if three_phase:
        plan["model"].pop("group_experts")
        plan["schedule_mode"] = "three_phase"
        plan["phases"] = phase_plan
        plan["model"]["capacity_recipe"] = model.resolved_capacity_recipe()
        plan["optimizer"]["parameter_groups"] = "ownership_decay_split"
        plan["math"]["gradient_aggregation"] = "weighted_owner_raw_v1"
    else:
        plan["optimizer"]["lr"] = config.training.learning_rate
        plan["scheduler"] = {
            "name": "linear_warmup_cosine", "warmup_steps": warmup_steps,
            "total_steps": total_steps, "min_lr_ratio": config.training.min_lr_ratio,
        }
        plan["refinement"] = {
            "boundary_epoch": boundary_epoch,
            "epochs": refinement_epochs,
            "lr_multiplier": config.training.refinement_lr_multiplier,
            "scheduler": "task-local-no-warmup-cosine",
            "min_lr_ratio": config.training.min_lr_ratio,
            "selection": "task-validation-normalized-mae-min",
        }
    if config.training.joint_gradient_clip_mode != "global":
        plan["math"]["joint_gradient_clip_mode"] = (
            config.training.joint_gradient_clip_mode
        )
    if prepared["metadata"].get("kind") == STAGE3_ARTIFACT_KIND:
        plan["stage2_encoder_identity"] = metadata_identity(
            prepared["metadata"],
            "stage2_encoder",
            context="Stage 3 prepared artifact",
        )["hash"]
    else:
        contract = prepared["metadata"].get("descriptor_contract")
        if not isinstance(contract, dict):
            raise ValueError("RDKit Stage 3 artifact lacks descriptor contract")
        input_dims = dict(contract["fold_input_dims"][f"fold{fold}"])
        actual_dims = {
            "il": int(model.descriptor_adapters["il"][0].in_features),
            "single": int(
                model.descriptor_adapters["molecule"][0].in_features
            ),
        }
        if input_dims != actual_dims:
            raise ValueError("RDKit Stage 3 adapter/artifact input widths differ")
        plan["representation"] = {
            "kind": "rdkit_2d_adapter",
            "contract_sha256": canonical_json_sha256(contract),
            "input_dims": input_dims,
            "output_dim": 512,
        }
    if config.training.seed is not None:
        plan["training_seed"] = effective_training_seed(config)
    return plan


def _clip_joint_gradients(
    model: Stage3SparseModel,
    max_grad_norm: float,
    mode: str,
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    pre_norm = float(
        torch.nn.utils.clip_grad_norm_(
            trainable, float("inf"), error_if_nonfinite=True
        )
    )
    owner_pre_norms: dict[str, float] = {}
    owner_post_norms: dict[str, float] = {}
    if mode == "global":
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                trainable, max_grad_norm, error_if_nonfinite=True
            )
        post_norm = min(pre_norm, max_grad_norm) if max_grad_norm > 0 else pre_norm
        return pre_norm, post_norm, owner_pre_norms, owner_post_norms
    elif mode == "ownership":
        owned: dict[Ownership, list[nn.Parameter]] = {}
        for parameter, owner in model.parameter_ownership().items():
            if parameter.requires_grad and parameter.grad is not None:
                owned.setdefault(owner, []).append(parameter)
        for owner in sorted(owned):
            parameters = owned[owner]
            owner_pre_norms[owner.label] = float(
                torch.nn.utils.clip_grad_norm_(
                    parameters, float("inf"), error_if_nonfinite=True
                )
            )
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters, max_grad_norm, error_if_nonfinite=True
                )
            owner_post_norms[owner.label] = float(
                torch.nn.utils.clip_grad_norm_(
                    parameters, float("inf"), error_if_nonfinite=True
                )
            )
    else:
        raise ValueError(f"Unknown Stage 3 joint gradient clip mode: {mode}")
    post_norm = float(
        torch.nn.utils.clip_grad_norm_(
            trainable, float("inf"), error_if_nonfinite=True
        )
    )
    return pre_norm, post_norm, owner_pre_norms, owner_post_norms


def _lr_factor(step: int, warmup: int, total: int, floor: float) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return floor + (1.0 - floor) * cosine


def _batch(
    dataset: Stage3TaskDataset,
    indices: torch.Tensor,
    representations: Stage3RepresentationStore | torch.Tensor,
    task_spec: Any,
    normalization: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    cpu_indices = indices.cpu()
    primary_topology = (
        "il" if tuple(task_spec.primary_slots) == ("cation", "anion") else "molecule"
    )
    primary = (
        representations[dataset.primary_object_ids[cpu_indices]]
        if isinstance(representations, torch.Tensor)
        else representations.values(
            dataset.primary_object_ids[cpu_indices], primary_topology
        )
    ).to(device)
    partner_ids = dataset.partner_object_ids[cpu_indices]
    partner = (
        (
            representations[partner_ids]
            if isinstance(representations, torch.Tensor)
            else representations.values(partner_ids, "molecule")
        ).to(device)
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
    representations: Stage3RepresentationStore | torch.Tensor,
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
            dataset,
            micro,
            representations,
            model.task_specs[task_id],
            normalization,
            device,
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
    representations: Stage3RepresentationStore | torch.Tensor,
    normalizations: Mapping[str, Any],
    config: Stage3Config,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    per_task: dict[str, Any] = {}
    gate_diagnostics: dict[str, dict[str, float]] = {}
    for task_id, dataset in datasets.items():
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        gate_observations: list[torch.Tensor] = []
        for start in range(0, len(dataset), config.training.microbatch_size):
            indices = torch.arange(start, min(len(dataset), start + config.training.microbatch_size))
            primary, conditions, partner, target = _batch(
                dataset,
                indices,
                representations,
                model.task_specs[task_id],
                normalizations[task_id],
                device,
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.training.amp_dtype == "bf16"):
                output = model(task_id, primary, conditions, partner_embedding=partner)
                prediction = output.predictions
            if not torch.isfinite(prediction).all():
                raise RuntimeError(
                    f"Non-finite Stage 3 validation prediction: {task_id}"
                )
            predictions.append(prediction.float().cpu())
            targets.append(target.float().cpu())
            if config.training.schedule_mode == "three_phase":
                gate_observations.append(
                    task_gate_observations(output.diagnostics).cpu()
                )
        per_task[task_id] = regression_metrics(
            torch.cat(predictions), torch.cat(targets), normalizations[task_id]
        )
        if gate_observations:
            gate_diagnostics[task_id] = summarize_task_gate_observations(
                torch.cat(gate_observations)
            )
    metrics = ("mae", "rmse", "r2", "pearson_r", "normalized_mae", "normalized_rmse")
    macro_task: dict[str, Any] = {}
    macro_group: dict[str, Any] = {}
    per_group: dict[str, dict[str, float]] = {
        group: {}
        for group in sorted(
            {model.task_specs[task].meta_group for task in per_task}
        )
    }
    for metric in metrics:
        valid = {task: value[metric] for task, value in per_task.items() if metric in value and math.isfinite(value[metric])}
        macro_task[metric] = {
            "value": sum(valid.values()) / len(valid) if valid else float("nan"),
            "valid_tasks": len(valid), "total_tasks": len(per_task),
        }
        group_values = []
        for group in per_group:
            values = [valid[task] for task in valid if model.task_specs[task].meta_group == group]
            if values:
                value = sum(values) / len(values)
                per_group[group][metric] = value
                group_values.append(value)
        macro_group[metric] = {
            "value": sum(group_values) / len(group_values) if group_values else float("nan"),
            "valid_groups": len(group_values),
            "total_groups": len({model.task_specs[task].meta_group for task in per_task}),
        }
    result = {
        "tasks": per_task,
        "groups": per_group,
        "macro_task_equal": macro_task,
        "macro_group_equal": macro_group,
    }
    if config.training.schedule_mode == "three_phase":
        result["gate_diagnostics"] = gate_diagnostics
    return result


def run_stage3_training(
    config: Stage3Config,
    fold: int,
    *,
    output_dir: str | Path,
    resume_from: str | Path | None = None,
    expected_training_identity: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if config.training.schedule_mode != "three_phase":
        raise ValueError("Legacy Stage 3 training and resume are retired; historical final artifacts remain read-only")
    if fold not in range(1, 6):
        raise ValueError("Stage 3 fold must be in 1..5")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Stage 3 training supports one process per fold only")
    device = resolve_device(config.training.device)
    if config.training.amp_dtype == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Stage 3 BF16 requires a BF16-capable CUDA GPU")
    training_seed = effective_training_seed(config)
    seed_everything(training_seed + fold)
    prepared = load_prepared_stage3(config)
    representations = Stage3RepresentationStore(
        config.data.artifacts_dir,
        fold,
        prepared["objects"],
        str(prepared["metadata"]["kind"]),
    )
    d_model = representations.output_dim
    registry = prepared["registry"]
    model = Stage3SparseModel(
        config.model,
        registry,
        d_model,
        group_configs=config.groups,
        task_configs=config.tasks,
        task_private_recipes=_resolved_private_recipes(config),
        descriptor_input_dims=representations.input_dims,
    ).to(device)
    representation_source_identity = (
        metadata_identity(
            prepared["metadata"],
            "stage2_encoder",
            context="Stage 3 prepared artifact",
        )["hash"]
        if prepared["metadata"].get("kind") == STAGE3_ARTIFACT_KIND
        else metadata_identity(
            prepared["metadata"], "prepared", context="RDKit Stage 3 artifact"
        )["hash"]
    )
    source, plugin_plan = _load_plugin(
        config,
        model,
        representation_source_identity,
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
    from .three_phase import run_three_phase_training

    return run_three_phase_training(
        config=config, fold=fold, output_dir=output_dir, resume_from=resume_from,
        model=model, registry=registry, active=active, train_data=train_data,
        valid_data=valid_data, representations=representations,
        normalizations=normalizations, plan=plan, device=device,
    )



def resolve_stage3_training_identity(
    config: Stage3Config, fold: int
) -> dict[str, Any]:
    """Resolve the exact semantic identity used by ``run_stage3_training``."""
    if config.training.schedule_mode != "three_phase":
        raise ValueError("Legacy Stage 3 training and resume are retired; historical final artifacts remain read-only")
    if fold not in range(1, 6):
        raise ValueError("Stage 3 fold must be in 1..5")
    prepared = load_prepared_stage3(config)
    representations = Stage3RepresentationStore(
        config.data.artifacts_dir,
        fold,
        prepared["objects"],
        str(prepared["metadata"]["kind"]),
    )
    model = Stage3SparseModel(
        config.model,
        prepared["registry"],
        representations.output_dim,
        group_configs=config.groups,
        task_configs=config.tasks,
        task_private_recipes=_resolved_private_recipes(config),
        descriptor_input_dims=representations.input_dims,
    )
    encoder_identity = (
        metadata_identity(
            prepared["metadata"],
            "stage2_encoder",
            context="Stage 3 prepared artifact",
        )["hash"]
        if prepared["metadata"].get("kind") == STAGE3_ARTIFACT_KIND
        else metadata_identity(
            prepared["metadata"], "prepared", context="RDKit Stage 3 artifact"
        )["hash"]
    )
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
    "STAGE3_RDKIT_CHECKPOINT_KIND",
    "STAGE3_RDKIT_REFINED_KIND",
    "STAGE3_REFINED_KIND",
    "build_resolved_training_plan",
    "checkpoint_epochs",
    "compute_task_gradient",
    "regression_metrics",
    "resolve_stage3_training_identity",
    "run_stage3_training",
    "validate_tasks",
]
