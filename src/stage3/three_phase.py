from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from common.identity import require_compatible_identity, tensor_state_hash
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.training import capture_rng_state, restore_rng_state, seed_everything
from .config import Stage3Config, effective_training_seed
from .data import (
    Stage3RepresentationStore,
    Stage3TaskDataset,
    sanitize_task,
    shuffled_epoch_indices,
    stable_seed,
)
from .identity import build_stage3_training_identity
from .model import GLOBAL, Ownership, Stage3SparseModel, group_owner, private_owner
from .pcgrad import HierarchicalPCGradResult, hierarchical_pcgrad


THREE_PHASE_CHECKPOINT_VERSION = 1
THREE_PHASE_FINAL_FORMAT_VERSION = 1
THREE_PHASE_FINAL_KIND = "ilume_stage3_three_phase_final"
THREE_PHASE_RDKIT_FINAL_KIND = "ilume_stage3_rdkit_home_three_phase_final"


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, allow_nan=True, sort_keys=True) + "\n")


def _checkpoint_epochs(total_epochs: int, interval: int) -> tuple[int, ...]:
    epochs = list(range(interval, total_epochs + 1, interval))
    if not epochs or epochs[-1] != total_epochs:
        epochs.append(total_epochs)
    return tuple(epochs)


def _checkpoint_files(root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in root.glob("checkpoint_epoch_*.pt"):
        match = re.fullmatch(r"checkpoint_epoch_(\d{5})\.pt", path.name)
        if match is not None:
            result[int(match.group(1))] = path
    return result


def _history_epochs(path: Path) -> list[int]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if any(not isinstance(row, dict) or not isinstance(row.get("epoch"), int) for row in rows):
        raise ValueError(f"Malformed Stage 3 three-phase history: {path}")
    return [int(row["epoch"]) for row in rows]


def _history_last(path: Path) -> Mapping[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows or not isinstance(rows[-1], dict):
        raise ValueError(f"Malformed Stage 3 three-phase history: {path}")
    return rows[-1]


def _resume_epoch(root: Path, total_epochs: int, resume: bool) -> tuple[int, Path | None]:
    metrics = _history_epochs(root / "metrics.jsonl")
    diagnostics = _history_epochs(root / "diagnostics.jsonl")
    if metrics != diagnostics or metrics != list(range(1, len(metrics) + 1)):
        raise ValueError(f"Stage 3 three-phase histories are inconsistent: {root}")
    checkpoints = _checkpoint_files(root)
    if not metrics and not checkpoints:
        return 1, None
    if not resume:
        raise FileExistsError(f"Stage 3 three-phase scope already has run state: {root}")
    if not metrics or not checkpoints or max(checkpoints) != metrics[-1]:
        raise ValueError(
            f"Stage 3 three-phase scope has no legal resume point: {root}"
        )
    if metrics[-1] > total_epochs:
        raise ValueError(f"Stage 3 three-phase history exceeds its budget: {root}")
    return metrics[-1] + 1, checkpoints[metrics[-1]]


def _model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _model_hash(state: Mapping[str, torch.Tensor]) -> str:
    return tensor_state_hash("stage3.three-phase-model-state", state)


def _owner_state(
    model: Stage3SparseModel, owners: Sequence[Ownership]
) -> dict[str, torch.Tensor]:
    selected = set(owners)
    ownership = model.parameter_ownership()
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if ownership[parameter] in selected
    }


def _owner_hash(state: Mapping[str, torch.Tensor]) -> str:
    return tensor_state_hash("stage3.three-phase-owner-state", state)


def _per_owner_hashes(
    model: Stage3SparseModel,
    state: Mapping[str, torch.Tensor],
    owners: Sequence[Ownership],
) -> dict[str, str]:
    ownership = model.parameter_ownership()
    names = {
        name: ownership[parameter]
        for name, parameter in model.named_parameters()
    }
    return {
        owner.label: _owner_hash(
            {name: value for name, value in state.items() if names[name] == owner}
        )
        for owner in owners
    }


def _load_owner_state(
    model: Stage3SparseModel,
    state: Mapping[str, torch.Tensor],
    owners: Sequence[Ownership],
) -> None:
    expected = set(_owner_state(model, owners))
    if set(state) != expected:
        raise ValueError("Stage 3 branch delta contains the wrong owner parameters")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            if value.shape != parameters[name].shape:
                raise ValueError(f"Stage 3 branch delta shape mismatch: {name}")
            parameters[name].copy_(value.to(parameters[name].device))


def _require_anchor_outside_owners(
    model: Stage3SparseModel,
    anchor_state: Mapping[str, torch.Tensor],
    owners: Sequence[Ownership],
) -> None:
    owned = set(_owner_state(model, owners))
    current = model.state_dict()
    if set(current) != set(anchor_state):
        raise ValueError("Stage 3 branch anchor state keys changed")
    changed = [
        name
        for name in current
        if name not in owned
        and not torch.equal(current[name].detach().cpu(), anchor_state[name])
    ]
    if changed:
        raise RuntimeError(
            "Stage 3 branch changed parameters outside its owners: "
            + ", ".join(changed[:5])
        )


def _stitch_owner_deltas(
    model: Stage3SparseModel,
    anchor_state: Mapping[str, torch.Tensor],
    deltas: Mapping[
        str, tuple[Sequence[Ownership], Mapping[str, torch.Tensor], str]
    ],
) -> dict[str, torch.Tensor]:
    model.load_state_dict(anchor_state, strict=True)
    seen: set[str] = set()
    for scope, (owners, state, state_hash) in deltas.items():
        labels = {owner.label for owner in owners}
        if seen & labels:
            raise RuntimeError(f"Stage 3 stitched owners overlap at scope: {scope}")
        if _owner_hash(state) != state_hash:
            raise ValueError(f"Stage 3 owner delta hash mismatch: {scope}")
        seen.update(labels)
        _load_owner_state(model, state, owners)
    return _model_state(model)


def _representation_fields(plan: Mapping[str, Any]) -> dict[str, Any]:
    if "representation" in plan:
        return {"representation": dict(plan["representation"])}
    return {"stage2_encoder_identity": plan["stage2_encoder_identity"]}


def _final_kind(plan: Mapping[str, Any]) -> str:
    return (
        THREE_PHASE_RDKIT_FINAL_KIND
        if "representation" in plan
        else THREE_PHASE_FINAL_KIND
    )


def _set_trainable(
    model: Stage3SparseModel, owners: Sequence[Ownership]
) -> None:
    model.set_trainable_owners(owners)


def _optimizer(
    model: Stage3SparseModel,
    config: Stage3Config,
    owner_lrs: Mapping[Ownership, float],
) -> torch.optim.AdamW:
    groups: list[dict[str, Any]] = []
    seen: set[int] = set()
    for owner in sorted(owner_lrs):
        parameters = tuple(
            parameter
            for parameter in model.parameters_for_owner(owner)
            if parameter.requires_grad
        )
        if not parameters:
            raise ValueError(f"Stage 3 optimizer owner has no parameters: {owner.label}")
        identities = {id(parameter) for parameter in parameters}
        if seen & identities:
            raise RuntimeError("Stage 3 optimizer parameter appears in multiple owners")
        seen.update(identities)
        for decay, selected in (
            (True, [parameter for parameter in parameters if parameter.ndim >= 2]),
            (False, [parameter for parameter in parameters if parameter.ndim < 2]),
        ):
            if selected:
                groups.append(
                    {
                        "params": selected,
                        "lr": owner_lrs[owner],
                        "weight_decay": config.training.weight_decay if decay else 0.0,
                        "owner": owner.label,
                        "decay": decay,
                    }
                )
    return torch.optim.AdamW(
        groups,
        betas=config.training.betas,
        eps=config.training.eps,
        foreach=False,
        fused=False,
    )


def _lr_factor(step: int, warmup: int, total: int, floor: float) -> float:
    from .train import _lr_factor as legacy_factor

    return legacy_factor(step, warmup, total, floor)


class _OwnerScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        recipes: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.optimizer = optimizer
        self.recipes = {
            owner: {
                "lr": float(recipe["nominal_lr"]),
                "total": int(recipe["actual_update_budget"]),
                "warmup": int(recipe.get("warmup_updates", 0)),
                "floor": float(recipe["terminal_lr"]) / float(recipe["nominal_lr"]),
            }
            for owner, recipe in recipes.items()
        }
        self.updates = {owner: 0 for owner in self.recipes}
        for group in self.optimizer.param_groups:
            owner = str(group["owner"])
            if owner not in self.recipes:
                raise ValueError(f"Missing scheduler recipe for owner: {owner}")

    def step(self, active_owners: Sequence[str]) -> None:
        active = set(active_owners)
        for group in self.optimizer.param_groups:
            owner = str(group["owner"])
            recipe = self.recipes[owner]
            if owner in active:
                step = self.updates[owner]
                if step >= recipe["total"]:
                    raise RuntimeError(f"Owner scheduler budget exhausted: {owner}")
                group["lr"] = recipe["lr"] * _lr_factor(
                    step, recipe["warmup"], recipe["total"], recipe["floor"]
                )
        self.optimizer.step()
        for owner in active:
            self.updates[owner] += 1

    def state_dict(self) -> dict[str, Any]:
        return {"recipes": self.recipes, "updates": self.updates}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("recipes") != self.recipes:
            raise ValueError("Stage 3 owner scheduler recipe mismatch")
        updates = state.get("updates")
        if not isinstance(updates, dict) or set(updates) != set(self.updates):
            raise ValueError("Stage 3 owner scheduler state mismatch")
        self.updates = {owner: int(value) for owner, value in updates.items()}


def _gradient_owners(optimizer: torch.optim.Optimizer) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(group["owner"])
                for group in optimizer.param_groups
                if any(parameter.grad is not None for parameter in group["params"])
            }
        )
    )


def _learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    result: dict[str, float] = {}
    for group in optimizer.param_groups:
        owner = str(group["owner"])
        result[owner] = float(group["lr"])
    return result


def _assign_gradients(
    model: Stage3SparseModel, gradients: Mapping[nn.Parameter, torch.Tensor]
) -> None:
    for parameter, gradient in gradients.items():
        if parameter.requires_grad:
            parameter.grad = gradient.to(parameter.device, dtype=parameter.dtype)


def _clip(
    model: Stage3SparseModel, config: Stage3Config
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    from .train import _clip_joint_gradients

    return _clip_joint_gradients(
        model, config.training.max_grad_norm, "ownership"
    )


def _phase_checkpoint(
    *,
    phase: str,
    anchor_hash: str,
    epoch: int,
    updates: int,
    model: Stage3SparseModel,
    optimizer: torch.optim.Optimizer,
    scheduler: _OwnerScheduler,
    pcgrad_rng: random.Random,
    task_order_rng: random.Random,
    config: Stage3Config,
    fold: int,
    plan: Mapping[str, Any],
    normalizations: Mapping[str, Any],
) -> dict[str, Any]:
    state = _model_state(model)
    return {
        "kind": "ilume_stage3_three_phase_checkpoint",
        "format_version": THREE_PHASE_CHECKPOINT_VERSION,
        "stage": "stage3",
        "fold": fold,
        "phase": phase,
        "anchor_model_state_hash": anchor_hash,
        "completed_epoch": epoch,
        "updates": updates,
        "model": state,
        "model_state_hash": _model_hash(state),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "owner_updates": dict(scheduler.updates),
        "frozen_owners": sorted(
            label
            for label in scheduler.recipes
            if not any(
                parameter.requires_grad
                for parameter in model.parameters_for_owner(
                    GLOBAL
                    if label == "GLOBAL"
                    else group_owner(label.removeprefix("GROUP:"))
                    if label.startswith("GROUP:")
                    else private_owner(label.removeprefix("PRIVATE:"))
                )
            )
        ),
        "rng": capture_rng_state(),
        "pcgrad_rng": pcgrad_rng.getstate(),
        "task_order_rng": task_order_rng.getstate(),
        "config": config.to_dict(),
        "training_identity": build_stage3_training_identity(plan),
        "resolved_training_plan": dict(plan),
        "resolved_registry": plan["resolved_registry"],
        "normalization": dict(normalizations),
        "ownership_manifest": model.ownership_manifest(),
        **_representation_fields(plan),
    }


def _delta_checkpoint(
    *,
    phase: str,
    scope: str,
    epoch: int,
    updates: int,
    anchor_hash: str,
    owners: Sequence[Ownership],
    model: Stage3SparseModel,
    optimizer: torch.optim.Optimizer,
    scheduler: _OwnerScheduler,
    pcgrad_rng: random.Random,
    task_order_rng: random.Random,
    fold: int,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    state = _owner_state(model, owners)
    return {
        "kind": "ilume_stage3_three_phase_owner_delta",
        "format_version": THREE_PHASE_CHECKPOINT_VERSION,
        "stage": "stage3",
        "fold": fold,
        "phase": phase,
        "scope": scope,
        "completed_epoch": epoch,
        "updates": updates,
        "anchor_model_state_hash": anchor_hash,
        "owners": [owner.label for owner in sorted(owners)],
        "owner_state": state,
        "owner_state_hash": _owner_hash(state),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "owner_updates": dict(scheduler.updates),
        "frozen_owners": sorted(
            owner.label
            for owner in owners
            if not any(parameter.requires_grad for parameter in model.parameters_for_owner(owner))
        ),
        "rng": capture_rng_state(),
        "pcgrad_rng": pcgrad_rng.getstate(),
        "task_order_rng": task_order_rng.getstate(),
        "training_identity": build_stage3_training_identity(plan),
        "ownership_manifest": model.ownership_manifest(),
    }


def _validate_checkpoint_common(
    checkpoint: Mapping[str, Any],
    *,
    phase: str,
    fold: int,
    plan: Mapping[str, Any],
) -> None:
    expected = {
        "format_version": THREE_PHASE_CHECKPOINT_VERSION,
        "stage": "stage3",
        "fold": fold,
        "phase": phase,
        "training_identity": build_stage3_training_identity(plan),
    }
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            raise ValueError(f"Stage 3 three-phase checkpoint mismatch: {name}")


def _joint_epoch(
    *,
    model: Stage3SparseModel,
    tasks: Sequence[str],
    epoch: int,
    phase_seed: int,
    steps_per_epoch: int,
    allocation: Mapping[str, int],
    counts: Mapping[str, int],
    train_data: Mapping[str, Stage3TaskDataset],
    representations: Stage3RepresentationStore | torch.Tensor,
    normalizations: Mapping[str, Any],
    config: Stage3Config,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: _OwnerScheduler,
    registry: Mapping[str, Any],
    group_weights: Mapping[str, float],
    pcgrad_rng: random.Random,
    task_order_rng: random.Random,
) -> tuple[dict[str, float], dict[str, Any]]:
    from .train import compute_task_gradient

    sequences = {
        task: shuffled_epoch_indices(
            counts[task], seed=phase_seed, epoch=epoch, task_id=task
        )
        for task in tasks
    }
    loss_sums = {task: 0.0 for task in tasks}
    sample_counts = {task: 0 for task in tasks}
    latest: HierarchicalPCGradResult | None = None
    clip_values: tuple[float, float, dict[str, float], dict[str, float]] = (
        0.0, 0.0, {}, {}
    )
    for step in range(steps_per_epoch):
        order = list(tasks)
        task_order_rng.shuffle(order)
        gradients = {}
        for task in order:
            begin = step * allocation[task]
            indices = sequences[task][begin : begin + allocation[task]]
            if not len(indices):
                continue
            gradient, loss = compute_task_gradient(
                model, task, train_data[task], indices, representations,
                normalizations[task], config, device,
            )
            gradients[task] = gradient
            loss_sums[task] += loss * len(indices)
            sample_counts[task] += len(indices)
        if not gradients:
            raise RuntimeError("Stage 3 three-phase joint step has no tasks")
        latest = hierarchical_pcgrad(
            model, gradients, registry, group_weights, pcgrad_rng
        )
        optimizer.zero_grad(set_to_none=True)
        _assign_gradients(model, latest.gradients)
        clip_values = _clip(model, config)
        scheduler.step(_gradient_owners(optimizer))
    if sample_counts != {task: counts[task] for task in tasks}:
        raise RuntimeError("Stage 3 three-phase raw epoch coverage is incomplete")
    assert latest is not None
    pre, post, owner_pre, owner_post = clip_values
    return (
        {task: loss_sums[task] / counts[task] for task in tasks},
        {
            "pcgrad_applied": True,
            "pcgrad_scope": (
                "group" if len({registry[t].meta_group for t in tasks}) == 1 else "hierarchical"
            ),
            "task_gradient_norms": latest.task_norms,
            "assembled_owner_norms": latest.assembled_owner_norms,
            "clip_pre_norm": pre,
            "clip_post_norm": post,
            "clip_owner_pre_norms": owner_pre,
            "clip_owner_post_norms": owner_post,
        },
    )


def _task_epoch(
    *,
    model: Stage3SparseModel,
    task: str,
    epoch: int,
    phase_seed: int,
    allocation: int,
    count: int,
    dataset: Stage3TaskDataset,
    representations: Stage3RepresentationStore | torch.Tensor,
    normalization: Mapping[str, Any],
    config: Stage3Config,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: _OwnerScheduler,
) -> tuple[float, dict[str, Any]]:
    from .train import compute_task_gradient

    sequence = shuffled_epoch_indices(
        count, seed=phase_seed, epoch=epoch, task_id=task
    )
    loss_sum = 0.0
    samples = 0
    pre_norm = post_norm = 0.0
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    for begin in range(0, count, allocation):
        indices = sequence[begin : begin + allocation]
        gradient, loss = compute_task_gradient(
            model, task, dataset, indices, representations,
            normalization, config, device,
        )
        optimizer.zero_grad(set_to_none=True)
        _assign_gradients(model, gradient)
        pre_norm = float(
            torch.nn.utils.clip_grad_norm_(
                parameters, float("inf"), error_if_nonfinite=True
            )
        )
        if config.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                parameters, config.training.max_grad_norm, error_if_nonfinite=True
            )
        post_norm = float(
            torch.nn.utils.clip_grad_norm_(
                parameters, float("inf"), error_if_nonfinite=True
            )
        )
        scheduler.step(_gradient_owners(optimizer))
        loss_sum += loss * len(indices)
        samples += len(indices)
    if samples != count:
        raise RuntimeError("Stage 3 Phase 3 raw epoch coverage is incomplete")
    return loss_sum / count, {
        "pcgrad_applied": False,
        "gradient_norm": pre_norm,
        "clip_post_norm": post_norm,
    }


def _run_phase1(
    *,
    root: Path,
    model: Stage3SparseModel,
    tasks: Sequence[str],
    owner_recipes: Mapping[str, Mapping[str, Any]],
    epochs: int,
    steps_per_epoch: int,
    config: Stage3Config,
    fold: int,
    plan: Mapping[str, Any],
    train_data: Mapping[str, Stage3TaskDataset],
    valid_data: Mapping[str, Stage3TaskDataset],
    representations: Stage3RepresentationStore | torch.Tensor,
    normalizations: Mapping[str, Any],
    registry: Mapping[str, Any],
    device: torch.device,
    resume: bool,
    anchor_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any], str]:
    from .train import validate_tasks

    root.mkdir(parents=True, exist_ok=True)
    anchor_hash = _model_hash(anchor_state)
    model.load_state_dict(anchor_state, strict=True)
    owners = {
        owner.label: owner
        for owner in (
            GLOBAL,
            *(group_owner(group) for group in sorted(config.groups)),
            *(private_owner(task) for task in sorted(tasks)),
        )
        if owner.label in owner_recipes
    }
    _set_trainable(model, tuple(owners.values()))
    optimizer = _optimizer(
        model,
        config,
        {owners[label]: float(recipe["nominal_lr"]) for label, recipe in owner_recipes.items()},
    )
    scheduler = _OwnerScheduler(optimizer, owner_recipes)
    phase = "phase_1"
    phase_seed = stable_seed(effective_training_seed(config), fold, phase)
    seed_everything(phase_seed % (2**32))
    pcgrad_rng = random.Random(stable_seed(phase_seed, "pcgrad"))
    task_order_rng = random.Random(stable_seed(phase_seed, "task_order"))
    start, checkpoint_path = _resume_epoch(root, epochs, resume)
    updates = 0
    validation: dict[str, Any] | None = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _validate_checkpoint_common(checkpoint, phase=phase, fold=fold, plan=plan)
        completed_epoch = start - 1
        expected_updates = completed_epoch * steps_per_epoch
        if checkpoint.get("kind") != "ilume_stage3_three_phase_checkpoint":
            raise ValueError("Stage 3 three-phase full checkpoint kind mismatch")
        if (
            checkpoint.get("anchor_model_state_hash") != anchor_hash
            or checkpoint.get("completed_epoch") != completed_epoch
            or checkpoint.get("updates") != expected_updates
            or checkpoint.get("ownership_manifest") != model.ownership_manifest()
            or checkpoint.get("model_state_hash") != _model_hash(checkpoint["model"])
            or _history_last(root / "metrics.jsonl").get("updates")
            != expected_updates
        ):
            raise ValueError("Stage 3 three-phase full checkpoint mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        expected_owner_updates = {
            label: min(completed_epoch, int(recipe["effective_epochs"]))
            * int(recipe["updates_per_epoch"])
            for label, recipe in owner_recipes.items()
        }
        if scheduler.updates != expected_owner_updates:
            raise ValueError("Stage 3 three-phase owner scheduler update mismatch")
        expected_frozen = sorted(
            label
            for label, recipe in owner_recipes.items()
            if completed_epoch >= int(recipe["effective_epochs"])
        )
        if (
            checkpoint.get("owner_updates") != expected_owner_updates
            or checkpoint.get("frozen_owners") != expected_frozen
            or _history_last(root / "metrics.jsonl").get("owner_updates")
            != expected_owner_updates
        ):
            raise ValueError("Stage 3 three-phase freeze/update state mismatch")
        _set_trainable(
            model,
            tuple(
                owners[label]
                for label in owner_recipes
                if label not in expected_frozen
            ),
        )
        restore_rng_state(checkpoint["rng"])
        pcgrad_rng.setstate(checkpoint["pcgrad_rng"])
        task_order_rng.setstate(checkpoint["task_order_rng"])
        updates = int(checkpoint["updates"])
        if start > epochs:
            rows = [
                json.loads(line)
                for line in (root / "metrics.jsonl").read_text().splitlines()
                if line
            ]
            validation = rows[-1]["validation"]
    allocation = plan["data"]["B_t"]
    counts = plan["data"]["N_t"]
    group_weights = {group: config.groups[group].group_weight for group in config.groups}
    for epoch in range(start, epochs + 1):
        active_owners = tuple(
            owners[label]
            for label, recipe in owner_recipes.items()
            if epoch <= int(recipe["effective_epochs"])
        )
        _set_trainable(model, active_owners)
        losses, diagnostics = _joint_epoch(
            model=model, tasks=tasks, epoch=epoch, phase_seed=phase_seed,
            steps_per_epoch=steps_per_epoch, allocation=allocation, counts=counts,
            train_data=train_data, representations=representations,
            normalizations=normalizations, config=config, device=device,
            optimizer=optimizer, scheduler=scheduler, registry=registry,
            group_weights=group_weights, pcgrad_rng=pcgrad_rng,
            task_order_rng=task_order_rng,
        )
        updates += steps_per_epoch
        remaining_owners = tuple(
            owners[label]
            for label, recipe in owner_recipes.items()
            if epoch < int(recipe["effective_epochs"])
        )
        _set_trainable(model, remaining_owners)
        validation = validate_tasks(
            model,
            {task: valid_data[task] for task in tasks},
            representations,
            {task: normalizations[task] for task in tasks},
            config,
            device,
        )
        _append_jsonl(
            root / "metrics.jsonl",
            {
                "epoch": epoch, "phase": phase, "updates": updates,
                "learning_rates": _learning_rates(optimizer),
                "owner_updates": dict(scheduler.updates),
                "frozen_owners": sorted(set(owners) - {owner.label for owner in remaining_owners}),
                "training_loss": losses, "validation": validation,
            },
        )
        _append_jsonl(
            root / "diagnostics.jsonl",
            {
                "epoch": epoch,
                "phase": phase,
                "owner_updates": dict(scheduler.updates),
                **diagnostics,
            },
        )
        if epoch in _checkpoint_epochs(epochs, config.training.checkpoint_interval_epochs):
            path = root / f"checkpoint_epoch_{epoch:05d}.pt"
            if path.exists():
                raise FileExistsError(f"Stage 3 checkpoint already exists: {path}")
            atomic_torch_save(
                path,
                _phase_checkpoint(
                    phase=phase, anchor_hash=anchor_hash, epoch=epoch,
                    updates=updates, model=model,
                    optimizer=optimizer, scheduler=scheduler,
                    pcgrad_rng=pcgrad_rng, task_order_rng=task_order_rng,
                    config=config, fold=fold, plan=plan,
                    normalizations=normalizations,
                ),
            )
    if validation is None:
        raise RuntimeError(f"Stage 3 phase has no validation: {phase}")
    state = _model_state(model)
    return state, validation, _model_hash(state)


def _run_delta_branch(
    *,
    phase: str,
    scope: str,
    root: Path,
    model: Stage3SparseModel,
    tasks: Sequence[str],
    owners: Sequence[Ownership],
    owner_recipes: Mapping[str, Mapping[str, Any]],
    epochs: int,
    steps_per_epoch: int,
    config: Stage3Config,
    fold: int,
    plan: Mapping[str, Any],
    train_data: Mapping[str, Stage3TaskDataset],
    valid_data: Mapping[str, Stage3TaskDataset],
    representations: Stage3RepresentationStore | torch.Tensor,
    normalizations: Mapping[str, Any],
    registry: Mapping[str, Any],
    device: torch.device,
    resume: bool,
    anchor_state: Mapping[str, torch.Tensor],
    anchor_hash: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], str]:
    from .train import validate_tasks

    root.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(anchor_state, strict=True)
    owners_by_label = {owner.label: owner for owner in owners}
    trainable_owners = tuple(
        owners_by_label[label]
        for label, recipe in owner_recipes.items()
        if int(recipe["effective_epochs"]) > 0
    )
    _set_trainable(model, trainable_owners)
    optimizer = _optimizer(
        model,
        config,
        {
            owners_by_label[label]: float(recipe["nominal_lr"])
            for label, recipe in owner_recipes.items()
            if int(recipe["effective_epochs"]) > 0
        },
    )
    scheduler = _OwnerScheduler(optimizer, owner_recipes)
    phase_seed = stable_seed(effective_training_seed(config), fold, phase, scope)
    seed_everything(phase_seed % (2**32))
    pcgrad_rng = random.Random(stable_seed(phase_seed, "pcgrad"))
    task_order_rng = random.Random(stable_seed(phase_seed, "task_order"))
    start, checkpoint_path = _resume_epoch(root, epochs, resume)
    updates = 0
    validation: dict[str, Any] | None = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _validate_checkpoint_common(checkpoint, phase=phase, fold=fold, plan=plan)
        completed_epoch = start - 1
        expected_updates = completed_epoch * steps_per_epoch
        if (
            checkpoint.get("kind") != "ilume_stage3_three_phase_owner_delta"
            or checkpoint.get("scope") != scope
            or checkpoint.get("anchor_model_state_hash") != anchor_hash
            or checkpoint.get("completed_epoch") != completed_epoch
            or checkpoint.get("updates") != expected_updates
            or checkpoint.get("owners") != [owner.label for owner in sorted(owners)]
            or checkpoint.get("ownership_manifest") != model.ownership_manifest()
            or checkpoint.get("owner_state_hash")
            != _owner_hash(checkpoint["owner_state"])
            or _history_last(root / "metrics.jsonl").get("updates")
            != expected_updates
        ):
            raise ValueError("Stage 3 three-phase branch checkpoint mismatch")
        _load_owner_state(model, checkpoint["owner_state"], owners)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        expected_owner_updates = {
            label: min(completed_epoch, int(recipe["effective_epochs"]))
            * int(recipe["updates_per_epoch"])
            for label, recipe in owner_recipes.items()
        }
        if scheduler.updates != expected_owner_updates:
            raise ValueError("Stage 3 three-phase owner scheduler update mismatch")
        expected_frozen = sorted(
            label
            for label, recipe in owner_recipes.items()
            if completed_epoch >= int(recipe["effective_epochs"])
        )
        if (
            checkpoint.get("owner_updates") != expected_owner_updates
            or checkpoint.get("frozen_owners") != expected_frozen
            or _history_last(root / "metrics.jsonl").get("owner_updates")
            != expected_owner_updates
        ):
            raise ValueError("Stage 3 three-phase branch freeze/update mismatch")
        _set_trainable(
            model,
            tuple(
                owners_by_label[label]
                for label in owner_recipes
                if label not in expected_frozen
            ),
        )
        restore_rng_state(checkpoint["rng"])
        pcgrad_rng.setstate(checkpoint["pcgrad_rng"])
        task_order_rng.setstate(checkpoint["task_order_rng"])
        updates = int(checkpoint["updates"])
        if start > epochs:
            rows = [
                json.loads(line)
                for line in (root / "metrics.jsonl").read_text().splitlines()
                if line
            ]
            validation = rows[-1]["validation"]
    allocation = plan["data"]["B_t"]
    counts = plan["data"]["N_t"]
    group_weights = {group: config.groups[group].group_weight for group in config.groups}
    for epoch in range(start, epochs + 1):
        active_owners = tuple(
            owners_by_label[label]
            for label, recipe in owner_recipes.items()
            if epoch <= int(recipe["effective_epochs"])
        )
        _set_trainable(model, active_owners)
        if phase == "phase_2":
            losses, diagnostics = _joint_epoch(
                model=model, tasks=tasks, epoch=epoch, phase_seed=phase_seed,
                steps_per_epoch=steps_per_epoch, allocation=allocation, counts=counts,
                train_data=train_data, representations=representations,
                normalizations=normalizations, config=config, device=device,
                optimizer=optimizer, scheduler=scheduler, registry=registry,
                group_weights=group_weights, pcgrad_rng=pcgrad_rng,
                task_order_rng=task_order_rng,
            )
        else:
            task = tasks[0]
            loss, diagnostics = _task_epoch(
                model=model, task=task, epoch=epoch, phase_seed=phase_seed,
                allocation=int(allocation[task]), count=int(counts[task]),
                dataset=train_data[task], representations=representations,
                normalization=normalizations[task], config=config, device=device,
                optimizer=optimizer, scheduler=scheduler,
            )
            losses = {task: loss}
        updates += steps_per_epoch
        remaining_owners = tuple(
            owners_by_label[label]
            for label, recipe in owner_recipes.items()
            if epoch < int(recipe["effective_epochs"])
        )
        _set_trainable(model, remaining_owners)
        validation = validate_tasks(
            model,
            {task: valid_data[task] for task in tasks},
            representations,
            {task: normalizations[task] for task in tasks},
            config,
            device,
        )
        _append_jsonl(
            root / "metrics.jsonl",
            {
                "epoch": epoch, "phase": phase, "scope": scope,
                "updates": updates, "learning_rates": _learning_rates(optimizer),
                "owner_updates": dict(scheduler.updates),
                "frozen_owners": sorted(set(owners_by_label) - {owner.label for owner in remaining_owners}),
                "training_loss": losses, "validation": validation,
            },
        )
        _append_jsonl(
            root / "diagnostics.jsonl",
            {
                "epoch": epoch,
                "phase": phase,
                "scope": scope,
                "owner_updates": dict(scheduler.updates),
                **diagnostics,
            },
        )
        if epoch in _checkpoint_epochs(epochs, config.training.checkpoint_interval_epochs):
            _require_anchor_outside_owners(model, anchor_state, owners)
            path = root / f"checkpoint_epoch_{epoch:05d}.pt"
            if path.exists():
                raise FileExistsError(f"Stage 3 checkpoint already exists: {path}")
            atomic_torch_save(
                path,
                _delta_checkpoint(
                    phase=phase, scope=scope, epoch=epoch, updates=updates,
                    anchor_hash=anchor_hash, owners=owners, model=model,
                    optimizer=optimizer, scheduler=scheduler,
                    pcgrad_rng=pcgrad_rng, task_order_rng=task_order_rng,
                    fold=fold, plan=plan,
                ),
            )
    if validation is None:
        raise RuntimeError(f"Stage 3 branch has no validation: {phase}/{scope}")
    _require_anchor_outside_owners(model, anchor_state, owners)
    state = _owner_state(model, owners)
    return state, validation, _owner_hash(state)


def _load_or_publish_stitched(
    *,
    root: Path,
    model: Stage3SparseModel,
    anchor_state: Mapping[str, torch.Tensor],
    anchor_hash: str,
    deltas: Mapping[str, tuple[Sequence[Ownership], Mapping[str, torch.Tensor], str]],
    fold: int,
    plan: Mapping[str, Any],
    validation: Mapping[str, Any],
    normalizations: Mapping[str, Any],
    resume: bool,
) -> tuple[dict[str, torch.Tensor], str]:
    expected_labels = {
        label
        for branch in plan["phases"]["phase2"]["branches"].values()
        for label in branch["owners"]
    }
    actual_labels = {
        owner.label
        for owners, _, _ in deltas.values()
        for owner in owners
    }
    if actual_labels != expected_labels:
        raise ValueError("Stage 3 Phase 2 stitched owner set is incomplete")
    artifact_path = root / "stitched.pt"
    manifest_path = root / "stitched.json"
    if artifact_path.exists() or manifest_path.exists():
        if not resume or not artifact_path.is_file() or not manifest_path.is_file():
            raise FileExistsError("Stage 3 Phase 2 stitched artifact is incomplete")
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        manifest = json.loads(manifest_path.read_text())
        expected_identity = build_stage3_training_identity(plan)
        expected_stitched_hash = _model_hash(_model_state(model))
        if (
            artifact.get("kind") != "ilume_stage3_three_phase_2_stitched"
            or artifact.get("format_version") != THREE_PHASE_FINAL_FORMAT_VERSION
            or artifact.get("fold") != fold
            or manifest.get("artifact_sha256") != sha256_file(artifact_path)
            or artifact.get("model_state_hash") != _model_hash(artifact["model"])
            or artifact.get("model_state_hash") != expected_stitched_hash
            or artifact.get("anchor_model_state_hash") != anchor_hash
            or artifact.get("ownership_manifest") != model.ownership_manifest()
        ):
            raise ValueError("Stage 3 Phase 2 stitched artifact is corrupt")
        require_compatible_identity(
            expected_identity,
            artifact.get("training_identity", {}),
            context="Stage 3 Phase 2 stitched artifact",
        )
        if (
            manifest.get("kind") != artifact["kind"]
            or manifest.get("format_version") != artifact["format_version"]
            or manifest.get("fold") != fold
            or manifest.get("anchor_model_state_hash") != anchor_hash
            or manifest.get("model_state_hash") != artifact["model_state_hash"]
            or manifest.get("ownership_manifest") != model.ownership_manifest()
            or set(manifest.get("groups", {})) != set(deltas)
        ):
            raise ValueError("Stage 3 Phase 2 stitched manifest is corrupt")
        require_compatible_identity(
            expected_identity,
            manifest.get("training_identity", {}),
            context="Stage 3 Phase 2 stitched manifest",
        )
        model.load_state_dict(artifact["model"], strict=True)
        return artifact["model"], artifact["model_state_hash"]
    state = _stitch_owner_deltas(model, anchor_state, deltas)
    public: dict[str, Any] = {}
    for scope in sorted(deltas):
        owners, state, state_hash = deltas[scope]
        labels = {owner.label for owner in owners}
        public[scope] = {
            "owners": sorted(labels),
            "owner_state_hash": state_hash,
            "per_owner_state_hashes": _per_owner_hashes(model, state, owners),
        }
    state = _model_state(model)
    state_hash = _model_hash(state)
    atomic_torch_save(
        artifact_path,
        {
            "kind": "ilume_stage3_three_phase_2_stitched",
            "format_version": THREE_PHASE_FINAL_FORMAT_VERSION,
            "fold": fold,
            "anchor_model_state_hash": anchor_hash,
            "model": state,
            "model_state_hash": state_hash,
            "training_identity": build_stage3_training_identity(plan),
            "validation": dict(validation),
            "resolved_training_plan": dict(plan),
            "normalization": dict(normalizations),
            "ownership_manifest": model.ownership_manifest(),
            **_representation_fields(plan),
        },
    )
    atomic_json(
        manifest_path,
        {
            "kind": "ilume_stage3_three_phase_2_stitched",
            "format_version": THREE_PHASE_FINAL_FORMAT_VERSION,
            "fold": fold,
            "artifact": artifact_path.name,
            "artifact_sha256": sha256_file(artifact_path),
            "anchor_model_state_hash": anchor_hash,
            "model_state_hash": state_hash,
            "training_identity": build_stage3_training_identity(plan),
            "ownership_manifest": model.ownership_manifest(),
            "groups": public,
            "validation": dict(validation),
        },
    )
    return state, state_hash


def run_three_phase_training(
    *,
    config: Stage3Config,
    fold: int,
    output_dir: str | Path,
    resume_from: str | Path | None,
    model: Stage3SparseModel,
    registry: Mapping[str, Any],
    active: Sequence[str],
    train_data: Mapping[str, Stage3TaskDataset],
    valid_data: Mapping[str, Stage3TaskDataset],
    representations: Stage3RepresentationStore | torch.Tensor,
    normalizations: Mapping[str, Any],
    plan: Mapping[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    from .train import validate_tasks

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resume = resume_from is not None
    plan_path = output / "resolved_training_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        require_compatible_identity(
            build_stage3_training_identity(plan),
            build_stage3_training_identity(existing),
            context="Existing Stage 3 three-phase resolved plan",
        )
    else:
        atomic_json(plan_path, plan)
    phases = plan["phases"]
    groups = sorted({registry[task].meta_group for task in active})
    capacity = plan["model"]["capacity_recipe"]

    initial_state = _model_state(model)
    phase1 = phases["phase1"]
    phase1_state, phase1_validation, phase1_hash = _run_phase1(
        root=output / "phase_1", model=model, tasks=active,
        owner_recipes=phase1["owners"], epochs=int(phase1["epochs"]),
        steps_per_epoch=int(phase1["steps_per_epoch"]),
        config=config, fold=fold, plan=plan, train_data=train_data,
        valid_data=valid_data, representations=representations,
        normalizations=normalizations, registry=registry, device=device,
        resume=resume, anchor_state=initial_state,
    )

    phase2 = phases["phase2"]
    phase2_deltas: dict[str, tuple[Sequence[Ownership], Mapping[str, torch.Tensor], str]] = {}
    phase2_records: dict[str, Any] = {}
    for group in groups:
        tasks = tuple(task for task in active if registry[task].meta_group == group)
        owners = (group_owner(group), *(private_owner(task) for task in tasks))
        branch = phase2["branches"][group]
        state, validation, state_hash = _run_delta_branch(
            phase="phase_2", scope=group, root=output / "phase_2" / group,
            model=model, tasks=tasks, owners=owners,
            owner_recipes=branch["owners"],
            epochs=int(branch["epochs"]),
            steps_per_epoch=int(branch["steps_per_epoch"]),
            config=config, fold=fold, plan=plan, train_data=train_data, valid_data=valid_data,
            representations=representations, normalizations=normalizations,
            registry=registry, device=device, resume=resume,
            anchor_state=phase1_state, anchor_hash=phase1_hash,
        )
        phase2_deltas[group] = (owners, state, state_hash)
        phase2_records[group] = {
            "epochs": int(branch["epochs"]),
            "owner_recipes": branch["owners"],
            "capacity": capacity["groups"][group],
            "anchor_model_state_hash": phase1_hash,
            "owner_state_hash": state_hash,
            "per_owner_state_hashes": _per_owner_hashes(model, state, owners),
            "validation": validation,
        }
    if set(phase2_deltas) != set(groups):
        raise RuntimeError("Stage 3 Phase 2 branch set is incomplete")
    _stitch_owner_deltas(model, phase1_state, phase2_deltas)
    phase2_validation = validate_tasks(
        model, valid_data, representations, normalizations, config, device
    )
    phase2_state, phase2_hash = _load_or_publish_stitched(
        root=output / "phase_2", model=model, anchor_state=phase1_state,
        anchor_hash=phase1_hash, deltas=phase2_deltas, fold=fold, plan=plan,
        validation=phase2_validation, normalizations=normalizations, resume=resume,
    )

    phase3 = phases["phase3"]
    phase3_deltas: dict[str, tuple[Sequence[Ownership], Mapping[str, torch.Tensor], str]] = {}
    phase3_records: dict[str, Any] = {}
    for task in sorted(active):
        owner = private_owner(task)
        branch = phase3["branches"][task]
        carried_from_anchor = int(branch["epochs"]) == 0
        if carried_from_anchor:
            model.load_state_dict(phase2_state, strict=True)
            _set_trainable(model, ())
            state = _owner_state(model, (owner,))
            state_hash = _owner_hash(state)
            validation = validate_tasks(
                model,
                {task: valid_data[task]},
                representations,
                {task: normalizations[task]},
                config,
                device,
            )
        else:
            state, validation, state_hash = _run_delta_branch(
                phase="phase_3", scope=task,
                root=output / "phase_3" / sanitize_task(task), model=model,
                tasks=(task,), owners=(owner,), owner_recipes=branch["owners"],
                epochs=int(branch["epochs"]),
                steps_per_epoch=int(branch["steps_per_epoch"]),
                config=config, fold=fold, plan=plan, train_data=train_data,
                valid_data=valid_data, representations=representations,
                normalizations=normalizations, registry=registry, device=device,
                resume=resume, anchor_state=phase2_state, anchor_hash=phase2_hash,
            )
        phase3_deltas[task] = ((owner,), state, state_hash)
        phase3_records[task] = {
            "epochs": int(branch["epochs"]),
            "owner_recipe": branch["owners"][owner.label],
            "capacity": capacity["tasks"][task],
            "anchor_model_state_hash": phase2_hash,
            "private_state_hash": state_hash,
            "carried_from_anchor": carried_from_anchor,
            "validation": validation,
        }

    if set(phase3_deltas) != set(active):
        raise RuntimeError("Stage 3 Phase 3 branch set is incomplete")
    _stitch_owner_deltas(model, phase2_state, phase3_deltas)
    final_validation = validate_tasks(
        model, valid_data, representations, normalizations, config, device
    )
    final_state = _model_state(model)
    final_hash = _model_hash(final_state)
    artifact_path = output / "three_phase_final.pt"
    manifest_path = output / "three_phase_final.json"
    if artifact_path.exists() or manifest_path.exists():
        if not resume or not artifact_path.is_file() or not manifest_path.is_file():
            raise FileExistsError("Stage 3 three-phase final artifact is incomplete")
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        manifest = json.loads(manifest_path.read_text())
        expected_identity = build_stage3_training_identity(plan)
        if (
            artifact.get("kind") != _final_kind(plan)
            or artifact.get("format_version") != THREE_PHASE_FINAL_FORMAT_VERSION
            or artifact.get("fold") != fold
            or manifest.get("artifact_sha256") != sha256_file(artifact_path)
            or artifact.get("model_state_hash") != _model_hash(artifact["model"])
            or artifact.get("model_state_hash") != final_hash
            or artifact.get("ownership_manifest") != model.ownership_manifest()
            or artifact.get("normalization_hash") != plan["normalization_hash"]
        ):
            raise ValueError("Stage 3 three-phase final artifact is corrupt")
        require_compatible_identity(
            expected_identity,
            artifact.get("training_identity", {}),
            context="Stage 3 three-phase final artifact",
        )
        if (
            manifest.get("kind") != artifact["kind"]
            or manifest.get("format_version") != artifact["format_version"]
            or manifest.get("fold") != fold
            or manifest.get("model_state_hash") != artifact["model_state_hash"]
        ):
            raise ValueError("Stage 3 three-phase final manifest is corrupt")
        require_compatible_identity(
            expected_identity,
            manifest.get("training_identity", {}),
            context="Stage 3 three-phase final manifest",
        )
        return [{"phase": "three_phase_final", "validation": artifact["validation"]}]
    training_identity = build_stage3_training_identity(plan)
    artifact = {
        "kind": _final_kind(plan),
        "format_version": THREE_PHASE_FINAL_FORMAT_VERSION,
        "fold": fold,
        "model": final_state,
        "model_state_hash": final_hash,
        "training_identity": training_identity,
        "phase1_model_state_hash": phase1_hash,
        "phase2_model_state_hash": phase2_hash,
        "group_state_hashes": {
            group: record["per_owner_state_hashes"][f"GROUP:{group}"]
            for group, record in phase2_records.items()
        },
        "private_state_hashes": {
            task: record["private_state_hash"] for task, record in phase3_records.items()
        },
        "validation": final_validation,
        "resolved_training_plan": dict(plan),
        "resolved_registry": plan["resolved_registry"],
        "normalization": dict(normalizations),
        "normalization_hash": plan["normalization_hash"],
        "ownership_manifest": model.ownership_manifest(),
        **_representation_fields(plan),
    }
    atomic_torch_save(artifact_path, artifact)
    manifest = {
        "kind": _final_kind(plan),
        "format_version": THREE_PHASE_FINAL_FORMAT_VERSION,
        "fold": fold,
        "artifact": artifact_path.name,
        "artifact_sha256": sha256_file(artifact_path),
        "model_state_hash": final_hash,
        "training_identity": training_identity,
        "phases": {
            "phase1": {
                "recipe": dict(phase1),
                "model_state_hash": phase1_hash,
                "validation": phase1_validation,
            },
            "phase2": {
                "anchor_model_state_hash": phase1_hash,
                "stitched_model_state_hash": phase2_hash,
                "groups": phase2_records,
                "validation": phase2_validation,
            },
            "phase3": {
                "anchor_model_state_hash": phase2_hash,
                "tasks": phase3_records,
            },
        },
        "capacity_recipe": capacity,
        "validation": final_validation,
    }
    atomic_json(manifest_path, manifest)
    return [{"phase": "three_phase_final", "validation": final_validation}]


__all__ = [
    "THREE_PHASE_CHECKPOINT_VERSION",
    "THREE_PHASE_FINAL_FORMAT_VERSION",
    "THREE_PHASE_FINAL_KIND",
    "THREE_PHASE_RDKIT_FINAL_KIND",
    "run_three_phase_training",
]
