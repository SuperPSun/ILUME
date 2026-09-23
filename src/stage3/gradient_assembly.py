from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from .data import ResolvedTaskSpec
from .model import GLOBAL, Stage3SparseModel, group_owner, private_owner


GradientMap = dict[nn.Parameter, torch.Tensor]


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
) -> GradientMap:
    divisor = float(len(gradients))
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
) -> OwnerGradientResult:
    """Aggregate raw gradients by task within groups, then by group for GLOBAL."""
    if set(task_gradients) - set(task_specs):
        raise ValueError("Gradient assembly received unknown Stage 3 tasks")
    global_parameters = model.parameters_for_owner(GLOBAL)
    final: GradientMap = {}
    group_global: dict[str, GradientMap] = {}
    task_norms: dict[str, float] = {}
    groups = sorted({task_specs[task].meta_group for task in task_gradients})
    for group in groups:
        tasks = tuple(task for task in task_gradients if task_specs[task].meta_group == group)
        group_parameters = model.parameters_for_owner(group_owner(group))
        raw = {task: task_gradients[task] for task in tasks}
        weight_sum = sum(task_specs[task].task_weight for task in tasks)
        normalized = {
            task: len(tasks) * task_specs[task].task_weight / weight_sum for task in tasks
        }
        group_global[group] = _weighted_mean(raw, normalized, global_parameters)
        final.update(_weighted_mean(raw, normalized, group_parameters))
        for task in tasks:
            private_parameters = model.parameters_for_owner(private_owner(task))
            for parameter in private_parameters:
                if parameter in task_gradients[task]:
                    final[parameter] = task_gradients[task][parameter].float() * normalized[task]
            task_norms[task] = _norm(
                task_gradients[task], (*global_parameters, *group_parameters, *private_parameters)
            )
    group_weight_sum = sum(group_weights[group] for group in groups)
    for parameter in global_parameters:
        values = [group_global[group][parameter] * group_weights[group]
                  for group in groups if parameter in group_global[group]]
        if values:
            final[parameter] = torch.stack(values).sum(dim=0) / group_weight_sum
    owner_norms = {"GLOBAL": _norm(final, global_parameters)}
    for group in groups:
        owner_norms[f"GROUP:{group}"] = _norm(final, model.parameters_for_owner(group_owner(group)))
    for task in task_gradients:
        owner_norms[f"PRIVATE:{task}"] = _norm(final, model.parameters_for_owner(private_owner(task)))
    return OwnerGradientResult(final, task_norms, owner_norms)
