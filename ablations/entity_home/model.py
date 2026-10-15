from copy import deepcopy

import torch
from torch import nn

from common.training import seeded_initialization
from stage3.model import Expert
from stage3.simulation import SimulationEntityModel


class AblationSimulationModel(SimulationEntityModel):
    def group_delta(self, group, values):
        if self.ablation_variant in {'no_group', 'capacity_no_group'}:
            return values.new_zeros((len(values), self.d_model)), values.new_empty((len(values), 0))
        return super().group_delta(group, values)

    def task_gate_weights(self, key, values):
        if self.ablation_variant == 'uniform_gate':
            count = self.task_gates[key].out_features
            return values.new_full((len(values), count), 1.0 / count)
        return super().task_gate_weights(key, values)

    def set_trainable_owners(self, owners):
        super().set_trainable_owners(owners)
        if self.ablation_variant == 'uniform_gate':
            self.task_gates.requires_grad_(False).eval()

    def resolved_capacity_recipe(self):
        result = deepcopy(super().resolved_capacity_recipe())
        for group, recipe in result['groups'].items():
            experts = self.l1_group_experts[group]
            recipe['experts'] = len(experts)
            if experts:
                recipe['expert_hidden'] = experts[0].layers[0].out_features
                recipe['expert_hidden_ratio'] = recipe['expert_hidden'] / self.d_model
            else:
                recipe['expert_hidden'] = 0
                recipe['expert_hidden_ratio'] = 0.0
        for task, recipe in result['tasks'].items():
            key = task.replace('/', '__')
            experts = self.private_experts[key]
            recipe['private_experts'] = len(experts)
            if experts:
                recipe['private_hidden'] = experts[0].layers[0].out_features
                recipe['private_hidden_ratio'] = recipe['private_hidden'] / self.d_model
            else:
                recipe['private_hidden'] = 0
                recipe['private_hidden_ratio'] = 0.0
            recipe['candidate_count'] = self.task_gates[key].out_features
        return result


def expert_parameter_count(expert, hidden):
    first, last = expert.layers[0], expert.layers[3]
    return hidden * (first.in_features + 1 + last.out_features) + last.out_features


def matched_scale(model, removed):
    """Search every distinct integer-width combination, using parameters only."""
    total = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in model.simulation_backbone.parameters())
    removed_parameters = set()
    modules = (model.l1_global_experts, model.l2_global_experts, model.l1_global_gate) if removed == 'global' else (
        model.l1_group_experts, model.l2_group_experts, model.l1_group_gates)
    for module in modules:
        if module is not None:
            removed_parameters.update(module.parameters())
    reduction = sum(p.numel() for p in removed_parameters)
    # Deletion also shortens the output rows of each task gate.
    for task, spec in model.task_specs.items():
        count = len(model.l2_global_experts) if removed == 'global' else len(model.l2_group_experts[spec.meta_group])
        gate = model.task_gates[task.replace('/', '__')]
        reduction += count * (gate.in_features + 1)
    target = total - reduction
    experts = [m for m in model.modules() if isinstance(m, Expert)]
    fixed = total - sum(expert_parameter_count(e, e.layers[0].out_features) for e in experts)
    widths = sorted({e.layers[0].out_features for e in experts})
    boundaries = sorted({0.0, 1.0, *( (h + .5) / width for width in widths for h in range(1, width))})
    candidates = [(a + b) / 2 for a, b in zip(boundaries, boundaries[1:])]
    candidates.append(1.0)
    def count(scale):
        return fixed + sum(expert_parameter_count(e, max(1, round(e.layers[0].out_features * scale))) for e in experts)
    scale = min(candidates, key=lambda value: (abs(count(value) - target), -value))
    result = {'parameter_count_scope': 'downstream_home', 'target': removed, 'target_parameters': target, 'matched_parameters': count(scale),
              'relative_error': abs(count(scale) - target) / target, 'expert_width_scale': scale}
    if result['relative_error'] > .01:
        raise ValueError('Cannot match ablation parameter count within 1 percent')
    return result


def transform_model(model, variant, *, seed):
    # Change behavior after audited full initialization, preserving every retained tensor and RNG stream.
    model.__class__ = AblationSimulationModel
    model.ablation_variant = variant
    if variant.startswith('capacity_match_'):
        model.capacity_match = matched_scale(model, variant.removeprefix('capacity_match_'))
        scale = model.capacity_match['expert_width_scale']
        for name, module in list(model.named_modules()):
            if not isinstance(module, Expert):
                continue
            first = module.layers[0]
            owner = model.parameter_ownership()[first.weight]
            width = max(1, round(first.out_features * scale))
            with seeded_initialization(seed, 'capacity:' + name):
                module.layers[0] = nn.Linear(first.in_features, width).to(first.weight.device)
                module.layers[3] = nn.Linear(width, model.d_model).to(first.weight.device)
            model.register_owner_modules(owner, module)
    elif variant in {'no_global', 'no_group', 'no_private', 'capacity_no_global', 'capacity_no_group'}:
        scope = variant.removeprefix('capacity_').removeprefix('no_')
        for task, spec in model.task_specs.items():
            key = task.replace('/', '__')
            counts = [len(model.l2_global_experts), len(model.l2_group_experts[spec.meta_group]), len(model.private_experts[key])]
            position = {'global': 0, 'group': 1, 'private': 2}[scope]
            start = sum(counts[:position])
            indices = [i for i in range(sum(counts)) if not start <= i < start + counts[position]]
            old = model.task_gates[key]
            if not indices:
                raise ValueError('Ablation removed every task-gate candidate')
            with torch.random.fork_rng(devices=[]):
                gate = nn.Linear(old.in_features, len(indices), device=old.weight.device)
            with torch.no_grad():
                gate.weight.copy_(old.weight[indices]); gate.bias.copy_(old.bias[indices])
            model.task_gates[key] = gate
            # New gates retain their original PRIVATE owner.
            owner = next(o for p, o in model.parameter_ownership().items() if p is old.weight)
            model.register_owner_modules(owner, gate)
        if scope == 'global':
            model.l1_global_experts = nn.ModuleList()
            model.l2_global_experts = nn.ModuleList()
            model.l1_global_gate = None
        elif scope == 'group':
            for group in model.groups:
                model.l1_group_experts[group] = nn.ModuleList()
                model.l2_group_experts[group] = nn.ModuleList()
            model.l1_group_gates = nn.ModuleDict()
        else:
            for key in model.private_experts:
                model.private_experts[key] = nn.ModuleList()
    model.refresh_ownership()
    return model
