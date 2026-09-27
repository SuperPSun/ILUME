from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn

from stage3.data import sanitize_task
from stage3.model import GLOBAL, Ownership, Stage3ForwardOutput, group_owner, private_owner
from stage3.object_phase1 import OBJECT_ENCODER_OWNER, ObjectPhase1Model


SIM_GLOBAL = Ownership("SIM_GLOBAL")


def pool(experts: nn.ModuleList, values: torch.Tensor, gate: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.softmax(gate(values), dim=-1)
    outputs = torch.stack([expert(values) for expert in experts], dim=1)
    return (outputs * weights.unsqueeze(-1)).sum(dim=1), weights


def entropy(weights: torch.Tensor) -> torch.Tensor:
    if weights.shape[1] == 1:
        return weights.new_zeros(weights.shape[0])
    return torch.special.entr(weights.float()).sum(dim=1) / math.log(weights.shape[1])


class CrossDomainHoME(ObjectPhase1Model):
    def __init__(self, *args, simulation_home: nn.Module, atom_adapter: nn.Module, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulation_home = simulation_home
        self.simulation_atom_adapter = atom_adapter
        self.sim_groups = tuple(sorted(simulation_home.groups))
        # Register source modules once under their new owners; never duplicate parameters.
        for source_owner, modules in simulation_home._modules_by_owner.items():
            if source_owner.scope == "GLOBAL":
                owner = SIM_GLOBAL
            elif source_owner.scope == "GROUP":
                owner = Ownership("SIM_GROUP", source_owner.owner_id)
            else:
                task = source_owner.owner_id.split("::target_", 1)[0]
                owner = Ownership("SIM_PRIVATE", task)
            self._own_modules(owner, *modules)
        self._own_modules(Ownership("SIM_PRIVATE", "simulation/partial_atomic_charge"), atom_adapter)
        self.joint_upstream_owners = (
            OBJECT_ENCODER_OWNER, SIM_GLOBAL,
            *(Ownership("SIM_GROUP", group) for group in self.sim_groups),
        )
        d = self.d_model
        self.l1_global_domain = nn.Linear(d, 2)
        self.l2_global_domain = nn.Linear(2 * d, 2)
        self.l2_exp_global_selector = nn.Linear(2 * d, len(self.l2_global_experts))
        self.l2_sim_global_selector = nn.Linear(2 * d, len(simulation_home.l2_global_experts))
        self._own_modules(GLOBAL, self.l1_global_domain, self.l2_global_domain,
                          self.l2_exp_global_selector, self.l2_sim_global_selector)
        self.l1_sim_group_selectors = nn.ModuleDict()
        self.l2_sim_group_selectors = nn.ModuleDict()
        self.l1_group_domains = nn.ModuleDict()
        self.l2_group_domains = nn.ModuleDict()
        self.l2_exp_group_selectors = nn.ModuleDict()
        sim_count = sum(len(simulation_home.l1_group_experts[group]) for group in self.sim_groups)
        for group in self.groups:
            self.l1_sim_group_selectors[group] = nn.Linear(d, sim_count)
            self.l2_sim_group_selectors[group] = nn.Linear(2 * d, sim_count)
            self.l1_group_domains[group] = nn.Linear(d, 2)
            self.l2_group_domains[group] = nn.Linear(2 * d, 2)
            self.l2_exp_group_selectors[group] = nn.Linear(2 * d, len(self.l2_group_experts[group]))
            self._own_modules(group_owner(group), self.l1_sim_group_selectors[group],
                              self.l2_sim_group_selectors[group], self.l1_group_domains[group],
                              self.l2_group_domains[group], self.l2_exp_group_selectors[group])
        for task in self.task_specs:
            if not self.task_specs[task].enabled:
                continue
            key = sanitize_task(task)
            old = self.task_gates[key]
            owner = private_owner(task)
            self._modules_by_owner[owner].remove(old)
            for parameter in old.parameters():
                del self._ownership_by_parameter[parameter]
            self.task_gates[key] = nn.Linear(2 * d, 2 + len(self.private_experts[key]))
            self._own_modules(owner, self.task_gates[key])
        self._validate_ownership()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for owner, modules in self._modules_by_owner.items():
                if owner.scope.startswith("SIM_") and not any(
                    parameter.requires_grad for parameter in self.parameters_for_owner(owner)
                ):
                    for module in modules:
                        module.eval()
        return self

    def resolved_capacity_recipe(self):
        recipe = super().resolved_capacity_recipe()
        for task, values in recipe["tasks"].items():
            values["candidate_count"] = 2 + len(self.private_experts[sanitize_task(task)])
        return recipe

    def _sim_group_pool(self, level: int, values: torch.Tensor, routing: torch.Tensor):
        modules = getattr(self.simulation_home, f"l{level}_group_experts")
        outputs = torch.stack([expert(values) for group in self.sim_groups for expert in modules[group]], dim=1)
        weights = torch.softmax(routing, dim=-1)
        return (outputs * weights.unsqueeze(-1)).sum(dim=1), weights

    def forward(self, task_id, primary_embedding, conditions, *, partner_embedding=None,
                primary_knowledge=None, partner_knowledge=None):
        spec = self.task_specs[task_id]
        group, key = spec.meta_group, sanitize_task(task_id)
        exp_global, _ = pool(self.l1_global_experts, primary_embedding, self.l1_global_gate)
        sim_global, _ = pool(self.simulation_home.l1_global_experts, primary_embedding,
                             self.simulation_home.l1_global_gate)
        a_global1 = torch.softmax(self.l1_global_domain(primary_embedding), dim=-1)
        z_global = a_global1[:, :1] * sim_global + a_global1[:, 1:] * exp_global
        exp_group, _ = pool(self.l1_group_experts[group], primary_embedding, self.l1_group_gates[group])
        sim_group, group_weights1 = self._sim_group_pool(1, primary_embedding,
                                                       self.l1_sim_group_selectors[group](primary_embedding))
        a_group1 = torch.softmax(self.l1_group_domains[group](primary_embedding), dim=-1)
        delta = a_group1[:, :1] * sim_group + a_group1[:, 1:] * exp_group
        local = self.l1_group_normalizations[group](primary_embedding + delta)
        if spec.condition_columns:
            if conditions.shape[-1] != len(spec.condition_columns):
                raise ValueError("Cross-domain condition width mismatch")
            local = self.condition_films[key](local, conditions)
        elif conditions.shape[-1] != 0:
            raise ValueError("Cross-domain condition-free task received conditions")
        if spec.partner_mode == "interaction":
            if partner_embedding is None:
                raise ValueError("Cross-domain interaction requires partner embedding")
            local = self.interactions[group](local, partner_embedding)
        elif partner_embedding is not None:
            raise ValueError("Cross-domain non-interaction received partner embedding")
        routing = torch.cat((z_global, local), dim=-1)
        global_exp_outputs = torch.stack([expert(z_global) for expert in self.l2_global_experts], dim=1)
        global_sim_outputs = torch.stack([expert(z_global) for expert in self.simulation_home.l2_global_experts], dim=1)
        wg_exp = torch.softmax(self.l2_exp_global_selector(routing), dim=-1)
        wg_sim = torch.softmax(self.l2_sim_global_selector(routing), dim=-1)
        exp_global2 = (global_exp_outputs * wg_exp.unsqueeze(-1)).sum(dim=1)
        sim_global2 = (global_sim_outputs * wg_sim.unsqueeze(-1)).sum(dim=1)
        a_global2 = torch.softmax(self.l2_global_domain(routing), dim=-1)
        global2 = a_global2[:, :1] * sim_global2 + a_global2[:, 1:] * exp_global2
        exp_outputs = torch.stack([expert(local) for expert in self.l2_group_experts[group]], dim=1)
        w_exp = torch.softmax(self.l2_exp_group_selectors[group](routing), dim=-1)
        exp_group2 = (exp_outputs * w_exp.unsqueeze(-1)).sum(dim=1)
        sim_group2, group_weights2 = self._sim_group_pool(2, local, self.l2_sim_group_selectors[group](routing))
        a_group2 = torch.softmax(self.l2_group_domains[group](routing), dim=-1)
        group2 = a_group2[:, :1] * sim_group2 + a_group2[:, 1:] * exp_group2
        private = torch.stack([expert(local) for expert in self.private_experts[key]], dim=1)
        candidates = torch.cat((global2.unsqueeze(1), group2.unsqueeze(1), private), dim=1)
        gate = torch.softmax(self.task_gates[key](routing), dim=-1)
        mixed = (gate.unsqueeze(-1) * candidates).sum(dim=1)
        representation = self.task_normalizations[key](local + mixed) if self.model_config.l2_residual else mixed
        extra = [a_global1, a_group1, a_global2, a_group2]
        scalars = [entropy(value).unsqueeze(1) for value in extra]
        scalars += [entropy(group_weights1).unsqueeze(1), entropy(group_weights2).unsqueeze(1)]
        effective = torch.stack((gate[:, 0] * a_global2[:, 0], gate[:, 1] * a_group2[:, 0]), dim=1)
        group_masses = []
        for weights in (group_weights1, group_weights2):
            start = 0
            for sim_group_id in self.sim_groups:
                count = len(self.simulation_home.l1_group_experts[sim_group_id])
                group_masses.append(weights[:, start:start + count].sum(dim=1, keepdim=True))
                start += count
        return Stage3ForwardOutput(self.towers[key](representation), {
            "task_gate": gate, "l2_global_candidates": global2.unsqueeze(1),
            "l2_group_candidates": group2.unsqueeze(1), "l2_private_candidates": private,
            "z_global": z_global, "z_group": delta,
            "cross_domain_gate": torch.cat((*extra, *scalars, effective, *group_masses), dim=1),
        })

    def diagnostic_names(self):
        names = [f"mean_l{level}_{family}_{domain}_weight"
                 for level in (1, 2) for family in ("global", "group") for domain in ("sim", "exp")]
        names += [f"routing_entropy_l{level}_{family}_domain" for level in (1, 2) for family in ("global", "group")]
        names += ["routing_entropy_l1_sim_group_pool", "routing_entropy_l2_sim_group_pool",
                  "effective_sim_global_contribution", "effective_sim_group_contribution"]
        names += [f"mean_l{level}_sim_group_{group}_weight" for level in (1, 2) for group in self.sim_groups]
        return names

    def gate_observations(self, diagnostics: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat((super().gate_observations(diagnostics), diagnostics["cross_domain_gate"].detach().float()), dim=1)

    @property
    def gate_observation_width(self):
        return 4 + len(self.diagnostic_names())

    def summarize_gate_observations(self, observations):
        names = self.diagnostic_names()
        if observations.shape[1] != 4 + len(names) or not torch.isfinite(observations).all():
            raise ValueError("Invalid cross-domain gate observations")
        result = super().summarize_gate_observations(observations[:, :4])
        result.update(zip(names, observations[:, 4:].double().mean(dim=0).tolist(), strict=True))
        return result
