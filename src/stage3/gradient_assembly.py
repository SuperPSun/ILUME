from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from .data import ResolvedTaskSpec
from .model import GLOBAL, Stage3SparseModel, group_owner, private_owner


GradientMap = dict[nn.Parameter, torch.Tensor]
BATCH_SAMPLE_AGGREGATION = "batch_sample_weighted_owner_raw_v1"
BATCH_SAMPLE_WEIGHTING = {
    "weight_source": "actual_task_batch_size",
    "normalization": "participating_experimental_samples",
    "global": "cross_task_sample_mean",
    "group": "within_group_sample_mean",
    "private": "raw_task_mean_gradient",
    "simulation": "private_only",
}


@dataclass(frozen=True)
class OwnerGradientResult:
    gradients: GradientMap
    task_norms: dict[str, float]
    assembled_owner_norms: dict[str, float]


def _norm(gradients: GradientMap, parameters: Sequence[nn.Parameter]) -> float:
    values = [gradients[parameter].float().square().sum()
              for parameter in parameters if parameter in gradients]
    return math.sqrt(max(0.0, float(torch.stack(values).sum().detach().cpu()))) if values else 0.0


def _weighted_mean(
    gradients: Mapping[str, GradientMap],
    weights: Mapping[str, float],
    parameters: Sequence[nn.Parameter],
    *, divisor: float | None = None,
) -> GradientMap:
    divisor = float(len(gradients)) if divisor is None else divisor
    result: GradientMap = {}
    for parameter in parameters:
        values = [gradients[name][parameter].float() * weights[name]
                  for name in gradients if parameter in gradients[name]]
        if values:
            result[parameter] = torch.stack(values).sum(dim=0) / divisor
    return result


def assemble_owner_gradients(
    model: Stage3SparseModel,
    task_gradients: Mapping[str, GradientMap],
    task_specs: Mapping[str, ResolvedTaskSpec],
    group_weights: Mapping[str, float],
    *, task_batch_sizes: Mapping[str, int] | None = None,
) -> OwnerGradientResult:
    """Use actual batch sample means for Entity-HoME; preserve historical weighting."""
    if set(task_gradients) - set(task_specs):
        raise ValueError("Gradient assembly received unknown Stage 3 tasks")
    batch_weighted = model.entity_inputs
    if batch_weighted and (
        task_batch_sizes is None or set(task_batch_sizes) != set(task_gradients)
        or any(not isinstance(size, int) or size <= 0 for size in task_batch_sizes.values())
    ):
        raise ValueError("Entity-HoME gradient assembly requires positive actual task batch sizes")
    global_parameters = model.parameters_for_owner(GLOBAL)
    upstream_owners = tuple(getattr(model, "joint_upstream_owners", ()))
    upstream_parameters = tuple(
        parameter
        for owner in upstream_owners
        for parameter in model.parameters_for_owner(owner)
        if parameter.requires_grad
    )
    shared_parameters = (*global_parameters, *upstream_parameters)
    final: GradientMap = {}
    group_global: dict[str, GradientMap] = {}
    task_norms: dict[str, float] = {}
    groups = sorted({task_specs[task].meta_group for task in task_gradients})
    private_only = (
        set(getattr(model, "simulation_tasks", ()))
        if batch_weighted else set()
    )
    for group in groups:
        tasks = tuple(task for task in task_gradients if task_specs[task].meta_group == group)
        group_parameters = model.parameters_for_owner(group_owner(group))
        shared_tasks = tuple(task for task in tasks if task not in private_only)
        raw = {task: task_gradients[task] for task in shared_tasks}
        weights = (
            {task: float(task_batch_sizes[task]) for task in shared_tasks}
            if batch_weighted else {task: task_specs[task].task_weight for task in shared_tasks}
        )
        weight_sum = sum(weights.values())
        normalized = (
            {} if batch_weighted else
            {task: len(shared_tasks) * weights[task] / weight_sum for task in shared_tasks}
        )
        if shared_tasks:
            if batch_weighted:
                final.update(_weighted_mean(raw, weights, group_parameters, divisor=weight_sum))
            else:
                group_global[group] = _weighted_mean(raw, normalized, shared_parameters)
                final.update(_weighted_mean(raw, normalized, group_parameters))
        for task in tasks:
            private_parameters = model.parameters_for_owner(private_owner(task))
            for parameter in private_parameters:
                if parameter in task_gradients[task]:
                    value = task_gradients[task][parameter].float()
                    final[parameter] = value if batch_weighted else value * normalized.get(task, 1.0)
            task_norms[task] = _norm(
                task_gradients[task], (*shared_parameters, *group_parameters, *private_parameters)
            )
    if batch_weighted:
        raw = {task: gradient for task, gradient in task_gradients.items() if task not in private_only}
        if raw:
            weights = {task: float(task_batch_sizes[task]) for task in raw}
            final.update(_weighted_mean(raw, weights, shared_parameters, divisor=sum(weights.values())))
    else:
        group_weight_sum = sum(group_weights[group] for group in group_global)
        for parameter in shared_parameters:
            values = [group_global[group][parameter] * group_weights[group]
                      for group in group_global if parameter in group_global[group]]
            if values:
                final[parameter] = torch.stack(values).sum(dim=0) / group_weight_sum
    owner_norms = {"GLOBAL": _norm(final, global_parameters)}
    for owner in upstream_owners:
        owner_norms[owner.label] = _norm(final, model.parameters_for_owner(owner))
    for group in groups:
        owner_norms[f"GROUP:{group}"] = _norm(final, model.parameters_for_owner(group_owner(group)))
    for task in task_gradients:
        owner_norms[f"PRIVATE:{task}"] = _norm(final, model.parameters_for_owner(private_owner(task)))
    return OwnerGradientResult(final, task_norms, owner_norms)
