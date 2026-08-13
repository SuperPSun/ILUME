from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .stage3_config import AUX6_TASKS, IL21_TASKS, Stage3Config
from .stage3_data import PHASE_TOKENS, TASK_REGISTRY, sanitize_task


class Expert(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class FeatureGate(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        rank = max(1, d_model // 2)
        self.down = nn.ModuleList(
            [nn.Linear(d_model, rank, bias=False) for _ in range(2)]
        )
        self.up = nn.ModuleList(
            [nn.Linear(rank, d_model, bias=False) for _ in range(2)]
        )
        self.mix = nn.Linear(d_model, 2)
        for layer in self.up:
            nn.init.zeros_(layer.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.mix(values), dim=-1)
        residual = sum(
            weights[:, index : index + 1]
            * up(torch.nn.functional.silu(down(values)))
            for index, (down, up) in enumerate(
                zip(self.down, self.up, strict=True)
            )
        )
        return values + residual


class SoluteInteraction(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.normalization = nn.LayerNorm(d_model)

    def forward(
        self, group: torch.Tensor, solute: torch.Tensor
    ) -> torch.Tensor:
        interaction = torch.cat(
            (group, solute, torch.abs(group - solute), group * solute),
            dim=-1,
        )
        return self.normalization(group + self.project(interaction))


class ConditionFusion(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.condition = nn.Linear(8, d_model)
        self.phase = nn.Embedding(len(PHASE_TOKENS), d_model)
        self.normalization = nn.LayerNorm(d_model)

    def forward(
        self,
        base: torch.Tensor,
        conditions: torch.Tensor,
        phase_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.normalization(
            base + self.condition(conditions) + self.phase(phase_ids)
        )


def _mixture(
    experts: nn.ModuleList,
    values: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    outputs = torch.stack([expert(values) for expert in experts], dim=1)
    return (
        torch.softmax(logits, dim=-1).unsqueeze(-1) * outputs
    ).sum(dim=1)


@dataclass(frozen=True)
class Stage3ForwardOutput:
    predictions: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class IL21Model(nn.Module):
    def __init__(self, config: Stage3Config, d_model: int) -> None:
        super().__init__()
        self.config = config
        self.tasks = IL21_TASKS
        self.groups = tuple(
            sorted({TASK_REGISTRY[task].meta_group for task in self.tasks})
        )
        self.condition_fusion = ConditionFusion(d_model)
        self.solute_project = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model)
        )
        self.solute_interaction = SoluteInteraction(d_model)
        if config.model.architecture == "shared_bottom":
            self.shared_bottom = nn.Sequential(
                Expert(d_model), nn.Dropout(config.model.dropout)
            )
        elif config.model.architecture == "mmoe":
            expert_count = (
                config.model.global_experts + config.model.group_experts
            )
            self.mmoe_experts = nn.ModuleList(
                [Expert(d_model) for _ in range(expert_count)]
            )
            self.mmoe_gates = nn.ModuleDict(
                {
                    sanitize_task(task): nn.Linear(d_model, expert_count)
                    for task in self.tasks
                }
            )
        else:
            self.l1_global = nn.ModuleList(
                [Expert(d_model) for _ in range(config.model.global_experts)]
            )
            self.l1_global_gate = nn.Linear(
                d_model, config.model.global_experts
            )
            self.l1_groups = nn.ModuleDict(
                {
                    group: nn.ModuleList(
                        [
                            Expert(d_model)
                            for _ in range(config.model.group_experts)
                        ]
                    )
                    for group in self.groups
                }
            )
            self.l1_group_gates = nn.ModuleDict(
                {
                    group: nn.Linear(d_model, config.model.group_experts)
                    for group in self.groups
                }
            )
            self.l2_global = nn.ModuleList(
                [Expert(d_model) for _ in range(config.model.global_experts)]
            )
            self.l2_group = nn.ModuleDict(
                {
                    group: nn.ModuleList(
                        [
                            Expert(d_model)
                            for _ in range(config.model.group_experts)
                        ]
                    )
                    for group in self.groups
                }
            )
            self.private = nn.ModuleDict(
                {
                    sanitize_task(task): nn.ModuleList(
                        [
                            Expert(d_model)
                            for _ in range(config.model.private_experts)
                        ]
                    )
                    for task in self.tasks
                }
            )
            self.task_gates = nn.ModuleDict(
                {
                    sanitize_task(task): nn.Linear(
                        d_model
                        * (3 if TASK_REGISTRY[task].uses_solute else 2),
                        config.model.global_experts
                        + config.model.group_experts
                        + config.model.private_experts,
                    )
                    for task in self.tasks
                }
            )
            self.feature_gates = nn.ModuleDict(
                {
                    sanitize_task(task): FeatureGate(d_model)
                    for task in self.tasks
                }
            )
            self.self_gates = nn.ModuleDict(
                {
                    sanitize_task(task): nn.Linear(d_model, d_model)
                    for task in self.tasks
                }
            )
        self.towers = nn.ModuleDict(
            {
                sanitize_task(task): nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.SiLU(),
                    nn.Dropout(config.model.dropout),
                    nn.Linear(d_model, 1),
                )
                for task in self.tasks
            }
        )

    def _home(
        self,
        task: str,
        values: torch.Tensor,
        solute_cls: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        spec = TASK_REGISTRY[task]
        key = sanitize_task(task)
        group = spec.meta_group
        projected_solute = (
            self.solute_project(solute_cls)
            if solute_cls is not None
            else None
        )
        l1_values = values
        if spec.uses_solute and self.config.model.solute_injection == "early":
            if projected_solute is None:
                raise ValueError(f"Stage 3 task requires solute CLS: {task}")
            l1_values = self.solute_interaction(values, projected_solute)
        z_shared = _mixture(
            self.l1_global,
            l1_values,
            self.l1_global_gate(l1_values),
        )
        z_group = _mixture(
            self.l1_groups[group],
            l1_values,
            self.l1_group_gates[group](l1_values),
        )
        local_input = z_group
        if spec.uses_solute and self.config.model.solute_injection == "late":
            if projected_solute is None:
                raise ValueError(f"Stage 3 task requires solute CLS: {task}")
            local_input = self.solute_interaction(z_group, projected_solute)
        elif not spec.uses_solute and solute_cls is not None:
            raise ValueError(
                f"Direct Stage 3 task must not receive solute CLS: {task}"
            )
        global_outputs = torch.stack(
            [expert(z_shared) for expert in self.l2_global], dim=1
        )
        local_outputs = torch.stack(
            [expert(local_input) for expert in self.l2_group[group]], dim=1
        )
        private_outputs = torch.stack(
            [expert(local_input) for expert in self.private[key]], dim=1
        )
        gate_inputs = [z_shared, local_input]
        if spec.uses_solute:
            gate_inputs.append(projected_solute)
        logits = self.task_gates[key](torch.cat(gate_inputs, dim=-1))
        expert_outputs = torch.cat(
            (global_outputs, local_outputs, private_outputs), dim=1
        )
        mixed = (
            torch.softmax(logits, dim=-1).unsqueeze(-1) * expert_outputs
        ).sum(dim=1)
        if self.config.model.feature_gate:
            mixed = self.feature_gates[key](mixed)
        if self.config.model.self_gate:
            mixed = mixed * torch.sigmoid(self.self_gates[key](mixed))
        return mixed, {
            "first_layer_shared": z_shared,
            "first_layer_group": z_group,
            "second_layer_global": global_outputs,
            "second_layer_local": local_outputs,
            "second_layer_private": private_outputs,
            "task_gate": torch.softmax(logits, dim=-1),
        }

    def forward(
        self,
        task: str,
        base_embedding: torch.Tensor,
        conditions: torch.Tensor,
        phase_ids: torch.Tensor,
        *,
        solute_cls: torch.Tensor | None = None,
    ) -> Stage3ForwardOutput:
        if task not in self.tasks:
            raise ValueError(f"Inactive IL21 task: {task}")
        values = self.condition_fusion(
            base_embedding, conditions, phase_ids
        )
        architecture = self.config.model.architecture
        if architecture == "home":
            representation, diagnostics = self._home(
                task, values, solute_cls
            )
        else:
            spec = TASK_REGISTRY[task]
            projected_solute = None
            if spec.uses_solute and solute_cls is None:
                raise ValueError(
                    f"Stage 3 task requires solute CLS: {task}"
                )
            if spec.uses_solute:
                projected_solute = self.solute_project(solute_cls)
            elif solute_cls is not None:
                raise ValueError(
                    f"Direct Stage 3 task must not receive solute CLS: {task}"
                )
            if (
                spec.uses_solute
                and self.config.model.solute_injection == "early"
            ):
                values = self.solute_interaction(values, projected_solute)
            if architecture == "shared_bottom":
                representation = self.shared_bottom(values)
                diagnostics = {"shared_bottom": representation}
            else:
                logits = self.mmoe_gates[sanitize_task(task)](values)
                representation = _mixture(
                    self.mmoe_experts, values, logits
                )
                diagnostics = {"task_gate": torch.softmax(logits, dim=-1)}
            if (
                spec.uses_solute
                and self.config.model.solute_injection == "late"
            ):
                representation = self.solute_interaction(
                    representation, projected_solute
                )
        predictions = self.towers[sanitize_task(task)](
            representation
        ).squeeze(-1)
        return Stage3ForwardOutput(predictions, diagnostics)


class IndependentTaskHead(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.adapter = nn.Linear(d_model, d_model)
        self.condition = nn.Linear(8, d_model)
        self.phase = nn.Embedding(len(PHASE_TOKENS), d_model)
        self.normalization = nn.LayerNorm(d_model)
        self.expert = Expert(d_model)
        self.tower = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        base_embedding: torch.Tensor,
        conditions: torch.Tensor,
        phase_ids: torch.Tensor,
    ) -> torch.Tensor:
        values = self.normalization(
            self.adapter(base_embedding)
            + self.condition(conditions)
            + self.phase(phase_ids)
        )
        return self.tower(self.expert(values)).squeeze(-1)


class Aux6Model(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.tasks = AUX6_TASKS
        self.heads = nn.ModuleDict(
            {
                sanitize_task(task): IndependentTaskHead(d_model, dropout)
                for task in self.tasks
            }
        )

    def forward(
        self,
        task: str,
        base_embedding: torch.Tensor,
        conditions: torch.Tensor,
        phase_ids: torch.Tensor,
        *,
        solute_cls: torch.Tensor | None = None,
    ) -> Stage3ForwardOutput:
        if task not in self.tasks:
            raise ValueError(f"Inactive Aux6 task: {task}")
        if solute_cls is not None:
            raise ValueError("Aux6 independent heads do not accept solute CLS")
        predictions = self.heads[sanitize_task(task)](
            base_embedding, conditions, phase_ids
        )
        return Stage3ForwardOutput(predictions, {})


class Stage3MultiDomainModel(nn.Module):
    def __init__(
        self,
        config: Stage3Config,
        d_model: int,
        *,
        seed: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.active_domains = config.active_domains
        if "il21" in self.active_domains:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 11000)
                self.il21 = IL21Model(config, d_model)
        if "aux6" in self.active_domains:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 22000)
                self.aux6 = Aux6Model(d_model, config.model.dropout)

    def domain_module(self, domain: str) -> nn.Module:
        if domain not in self.active_domains:
            raise ValueError(f"Inactive Stage 3 domain: {domain}")
        return getattr(self, domain)

    def forward(
        self,
        task: str,
        base_embedding: torch.Tensor,
        conditions: torch.Tensor,
        phase_ids: torch.Tensor,
        *,
        solute_cls: torch.Tensor | None = None,
    ) -> Stage3ForwardOutput:
        domain = "il21" if task in IL21_TASKS else "aux6"
        return self.domain_module(domain)(
            task,
            base_embedding,
            conditions,
            phase_ids,
            solute_cls=solute_cls,
        )
