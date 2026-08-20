from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from .data import ResolvedTaskSpec
from .model import GLOBAL, Stage3SparseModel, group_owner, private_owner


GradientMap = dict[nn.Parameter, torch.Tensor]


@dataclass(frozen=True)
class PairDiagnostic:
    cosine: float
    conflict: bool
    projection_norm: float


@dataclass(frozen=True)
class HierarchicalPCGradResult:
    gradients: GradientMap
    task_global: dict[tuple[str, str], PairDiagnostic]
    task_group: dict[tuple[str, str], PairDiagnostic]
    group_global: dict[tuple[str, str], PairDiagnostic]
    task_norms: dict[str, float]
    task_global_norms: dict[str, float]
    task_group_norms: dict[str, float]
    private_norms: dict[str, float]
    group_global_norms: dict[str, float]
    assembled_owner_norms: dict[str, float]


def _dot(
    left: GradientMap,
    right: GradientMap,
    parameters: Sequence[nn.Parameter],
) -> torch.Tensor:
    terms = [
        left[parameter].float().reshape(-1).dot(right[parameter].float().reshape(-1))
        for parameter in parameters
        if parameter in left and parameter in right
    ]
    if not terms:
        device = next(iter(left.values()), next(iter(right.values()), torch.tensor(0.0))).device
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack(terms).sum()


def _norm_sq(gradient: GradientMap, parameters: Sequence[nn.Parameter]) -> torch.Tensor:
    terms = [
        gradient[parameter].float().square().sum()
        for parameter in parameters
        if parameter in gradient
    ]
    if not terms:
        device = next(iter(gradient.values()), torch.tensor(0.0)).device
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack(terms).sum()


def gradient_norm(gradient: GradientMap, parameters: Sequence[nn.Parameter]) -> float:
    return math.sqrt(max(0.0, float(_norm_sq(gradient, parameters).detach().cpu())))


def pcgrad_block(
    raw: Mapping[str, GradientMap],
    parameters: Sequence[nn.Parameter],
    rng: random.Random,
) -> tuple[dict[str, GradientMap], dict[tuple[str, str], PairDiagnostic]]:
    names = tuple(raw)
    parameter_set = set(parameters)
    projected = {
        name: {
            parameter: value.detach().float().clone()
            for parameter, value in raw[name].items()
            if parameter in parameter_set
        }
        for name in names
    }
    diagnostics: dict[tuple[str, str], PairDiagnostic] = {}
    for left_name in names:
        left_raw = raw[left_name]
        for right_name in names:
            if left_name == right_name:
                continue
            dot = _dot(left_raw, raw[right_name], parameters)
            left_norm = _norm_sq(left_raw, parameters).sqrt()
            right_norm = _norm_sq(raw[right_name], parameters).sqrt()
            denominator = left_norm * right_norm
            cosine = (
                float((dot / denominator).detach().cpu())
                if float(denominator.detach().cpu()) > 0.0
                else float("nan")
            )
            diagnostics[(left_name, right_name)] = PairDiagnostic(
                cosine=cosine,
                conflict=bool(float(dot.detach().cpu()) < 0.0),
                projection_norm=0.0,
            )
    for left_name in names:
        references = [name for name in names if name != left_name]
        rng.shuffle(references)
        current = projected[left_name]
        for right_name in references:
            reference = raw[right_name]
            common = tuple(
                parameter
                for parameter in parameters
                if parameter in current and parameter in reference
            )
            if not common:
                continue
            dot = _dot(current, reference, common)
            denominator = _norm_sq(reference, common)
            if float(dot.detach().cpu()) >= 0.0 or float(denominator.detach().cpu()) == 0.0:
                continue
            scale = dot / denominator
            projection_sq = torch.zeros((), dtype=torch.float32, device=dot.device)
            for parameter in common:
                delta = scale * reference[parameter].float()
                current[parameter] = current[parameter] - delta
                projection_sq = projection_sq + delta.square().sum()
            prior = diagnostics[(left_name, right_name)]
            diagnostics[(left_name, right_name)] = PairDiagnostic(
                cosine=prior.cosine,
                conflict=True,
                projection_norm=math.sqrt(
                    max(0.0, float(projection_sq.detach().cpu()))
                ),
            )
    return projected, diagnostics


def _weighted_mean(
    gradients: Mapping[str, GradientMap],
    weights: Mapping[str, float],
    parameters: Sequence[nn.Parameter],
) -> GradientMap:
    divisor = float(len(gradients))
    result: GradientMap = {}
    for parameter in parameters:
        values = [
            gradients[name][parameter] * weights[name]
            for name in gradients
            if parameter in gradients[name]
        ]
        if values:
            result[parameter] = torch.stack(values).sum(dim=0) / divisor
    return result


def hierarchical_pcgrad(
    model: Stage3SparseModel,
    task_gradients: Mapping[str, GradientMap],
    task_specs: Mapping[str, ResolvedTaskSpec],
    group_weights: Mapping[str, float],
    rng: random.Random,
) -> HierarchicalPCGradResult:
    if set(task_gradients) - set(task_specs):
        raise ValueError("PCGrad received unknown Stage 3 tasks")
    global_parameters = model.parameters_for_owner(GLOBAL)
    final: GradientMap = {}
    task_global_diagnostics: dict[tuple[str, str], PairDiagnostic] = {}
    task_group_diagnostics: dict[tuple[str, str], PairDiagnostic] = {}
    group_global_raw: dict[str, GradientMap] = {}
    task_norms: dict[str, float] = {}
    task_global_norms: dict[str, float] = {}
    task_group_norms: dict[str, float] = {}
    private_norms: dict[str, float] = {}
    group_global_norms: dict[str, float] = {}
    groups = sorted({task_specs[task].meta_group for task in task_gradients})
    for group in groups:
        tasks = tuple(
            task for task in task_gradients if task_specs[task].meta_group == group
        )
        group_parameters = model.parameters_for_owner(group_owner(group))
        raw_for_group = {task: task_gradients[task] for task in tasks}
        projected_global, global_diag = pcgrad_block(
            raw_for_group,
            global_parameters,
            random.Random(rng.getrandbits(64)),
        )
        projected_group, group_diag = pcgrad_block(
            raw_for_group,
            group_parameters,
            random.Random(rng.getrandbits(64)),
        )
        task_global_diagnostics.update(global_diag)
        task_group_diagnostics.update(group_diag)
        weight_sum = sum(task_specs[task].task_weight for task in tasks)
        normalized = {
            task: len(tasks) * task_specs[task].task_weight / weight_sum
            for task in tasks
        }
        group_global_raw[group] = _weighted_mean(
            projected_global, normalized, global_parameters
        )
        final.update(
            _weighted_mean(projected_group, normalized, group_parameters)
        )
        for task in tasks:
            private_parameters = model.parameters_for_owner(private_owner(task))
            for parameter in private_parameters:
                if parameter in task_gradients[task]:
                    final[parameter] = (
                        task_gradients[task][parameter].float() * normalized[task]
                    )
            task_norms[task] = gradient_norm(
                task_gradients[task],
                (*global_parameters, *group_parameters, *private_parameters),
            )
            task_global_norms[task] = gradient_norm(
                task_gradients[task], global_parameters
            )
            task_group_norms[task] = gradient_norm(
                task_gradients[task], group_parameters
            )
            private_norms[task] = gradient_norm(
                task_gradients[task], private_parameters
            )
        group_global_norms[group] = gradient_norm(
            group_global_raw[group], global_parameters
        )
    projected_groups, group_global_diagnostics = pcgrad_block(
        group_global_raw,
        global_parameters,
        random.Random(rng.getrandbits(64)),
    )
    group_weight_sum = sum(group_weights[group] for group in groups)
    for parameter in global_parameters:
        values = [
            projected_groups[group][parameter] * group_weights[group]
            for group in groups
            if parameter in projected_groups[group]
        ]
        if values:
            final[parameter] = torch.stack(values).sum(dim=0) / group_weight_sum
    assembled_owner_norms = {"GLOBAL": gradient_norm(final, global_parameters)}
    for group in groups:
        assembled_owner_norms[f"GROUP:{group}"] = gradient_norm(
            final, model.parameters_for_owner(group_owner(group))
        )
    for task in task_gradients:
        assembled_owner_norms[f"PRIVATE:{task}"] = gradient_norm(
            final, model.parameters_for_owner(private_owner(task))
        )
    return HierarchicalPCGradResult(
        gradients=final,
        task_global=task_global_diagnostics,
        task_group=task_group_diagnostics,
        group_global=group_global_diagnostics,
        task_norms=task_norms,
        task_global_norms=task_global_norms,
        task_group_norms=task_group_norms,
        private_norms=private_norms,
        group_global_norms=group_global_norms,
        assembled_owner_norms=assembled_owner_norms,
    )
