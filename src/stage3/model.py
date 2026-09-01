from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import nn

from .config import Stage3ModelConfig
from .data import ResolvedTaskSpec, sanitize_task


@dataclass(frozen=True, order=True)
class Ownership:
    scope: str
    owner_id: str | None = None

    @property
    def label(self) -> str:
        return self.scope if self.owner_id is None else f"{self.scope}:{self.owner_id}"


GLOBAL = Ownership("GLOBAL")


def group_owner(group_id: str) -> Ownership:
    return Ownership("GROUP", group_id)


def private_owner(task_id: str) -> Ownership:
    return Ownership("PRIVATE", task_id)


def _activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported Stage 3 activation: {name}")


def _width(d_model: int, ratio: float) -> int:
    return max(1, round(d_model * ratio))


class Expert(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        hidden_ratio: float,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        hidden = _width(d_model, hidden_ratio)
        self.layers = nn.Sequential(
            nn.Linear(d_model, hidden),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            _activation(activation),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class ConditionFiLM(nn.Module):
    def __init__(
        self,
        condition_width: int,
        d_model: int,
        *,
        hidden_ratio: float,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        hidden = _width(d_model, hidden_ratio)
        self.network = nn.Sequential(
            nn.Linear(condition_width, hidden),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 * d_model),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.normalization = nn.LayerNorm(d_model)

    def forward(self, values: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.network(conditions).chunk(2, dim=-1)
        return self.normalization(values * (1.0 + gamma) + beta)


class PartnerInteraction(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        hidden_ratio: float,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        hidden = _width(d_model, hidden_ratio)
        self.phi = nn.Sequential(
            nn.Linear(4 * d_model, hidden),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.normalization = nn.LayerNorm(d_model)

    def forward(self, primary: torch.Tensor, partner: torch.Tensor) -> torch.Tensor:
        interaction = torch.cat(
            (primary, partner, torch.abs(primary - partner), primary * partner),
            dim=-1,
        )
        return self.normalization(primary + self.phi(interaction))


class TaskTower(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        hidden_ratio: float,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        hidden = _width(d_model, hidden_ratio)
        self.layers = nn.Sequential(
            nn.Linear(d_model, hidden),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


def _mixture(
    experts: Iterable[nn.Module],
    values: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.softmax(logits, dim=-1)
    outputs = torch.stack([expert(values) for expert in experts], dim=1)
    return (weights.unsqueeze(-1) * outputs).sum(dim=1), weights


@dataclass(frozen=True)
class Stage3ForwardOutput:
    predictions: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class Stage3SparseModel(nn.Module):
    def __init__(
        self,
        model_config: Stage3ModelConfig,
        task_specs: Mapping[str, ResolvedTaskSpec],
        d_model: int,
        *,
        descriptor_input_dims: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.task_specs = dict(task_specs)
        self.d_model = d_model
        self.groups = tuple(
            sorted({spec.meta_group for spec in self.task_specs.values() if spec.enabled})
        )
        self._ownership_by_parameter: dict[nn.Parameter, Ownership] = {}
        self.descriptor_adapters = nn.ModuleDict()
        if descriptor_input_dims is not None:
            if d_model != 512 or set(descriptor_input_dims) != {"il", "molecule"}:
                raise ValueError("RDKit Stage 3 adapter contract is invalid")
            if any(int(width) <= 0 for width in descriptor_input_dims.values()):
                raise ValueError("RDKit Stage 3 adapter inputs must be positive")
            for topology in ("il", "molecule"):
                self.descriptor_adapters[topology] = nn.Sequential(
                    nn.Linear(int(descriptor_input_dims[topology]), d_model),
                    nn.LayerNorm(d_model),
                )
            self._own_modules(GLOBAL, self.descriptor_adapters)

        expert_kwargs = {
            "hidden_ratio": model_config.expert_hidden_ratio,
            "dropout": model_config.dropout,
            "activation": model_config.activation,
        }
        self.l1_global_experts = nn.ModuleList(
            [Expert(d_model, **expert_kwargs) for _ in range(model_config.global_experts)]
        )
        self.l1_global_gate = nn.Linear(d_model, model_config.global_experts)
        self.l2_global_experts = nn.ModuleList(
            [Expert(d_model, **expert_kwargs) for _ in range(model_config.global_experts)]
        )
        self._own_modules(
            GLOBAL,
            self.l1_global_experts,
            self.l1_global_gate,
            self.l2_global_experts,
        )

        self.l1_group_experts = nn.ModuleDict()
        self.l1_group_gates = nn.ModuleDict()
        self.l1_group_normalizations = nn.ModuleDict()
        self.l2_group_experts = nn.ModuleDict()
        self.interactions = nn.ModuleDict()
        partner_groups = {
            spec.meta_group
            for spec in self.task_specs.values()
            if spec.enabled and spec.partner_mode == "interaction"
        }
        for group in self.groups:
            self.l1_group_experts[group] = nn.ModuleList(
                [Expert(d_model, **expert_kwargs) for _ in range(model_config.group_experts)]
            )
            self.l1_group_gates[group] = nn.Linear(d_model, model_config.group_experts)
            self.l1_group_normalizations[group] = nn.LayerNorm(d_model)
            self.l2_group_experts[group] = nn.ModuleList(
                [Expert(d_model, **expert_kwargs) for _ in range(model_config.group_experts)]
            )
            modules: list[nn.Module] = [
                self.l1_group_experts[group],
                self.l1_group_gates[group],
                self.l1_group_normalizations[group],
                self.l2_group_experts[group],
            ]
            if group in partner_groups:
                self.interactions[group] = PartnerInteraction(
                    d_model,
                    hidden_ratio=model_config.interaction_hidden_ratio,
                    dropout=model_config.dropout,
                    activation=model_config.activation,
                )
                modules.append(self.interactions[group])
            self._own_modules(group_owner(group), *modules)

        self.private_experts = nn.ModuleDict()
        self.task_gates = nn.ModuleDict()
        self.condition_films = nn.ModuleDict()
        self.task_normalizations = nn.ModuleDict()
        self.towers = nn.ModuleDict()
        candidate_count = (
            model_config.global_experts
            + model_config.group_experts
            + model_config.private_experts
        )
        for task_id, spec in self.task_specs.items():
            if not spec.enabled:
                continue
            key = sanitize_task(task_id)
            self.private_experts[key] = nn.ModuleList(
                [Expert(d_model, **expert_kwargs) for _ in range(model_config.private_experts)]
            )
            self.task_gates[key] = nn.Linear(2 * d_model, candidate_count)
            if spec.condition_columns:
                self.condition_films[key] = ConditionFiLM(
                    len(spec.condition_columns),
                    d_model,
                    hidden_ratio=model_config.film_hidden_ratio,
                    dropout=model_config.dropout,
                    activation=model_config.activation,
                )
            self.task_normalizations[key] = (
                nn.LayerNorm(d_model) if model_config.l2_residual else nn.Identity()
            )
            self.towers[key] = TaskTower(
                d_model,
                hidden_ratio=model_config.tower_hidden_ratio,
                dropout=model_config.dropout,
                activation=model_config.activation,
            )
            modules = [
                self.private_experts[key],
                self.task_gates[key],
                self.task_normalizations[key],
                self.towers[key],
            ]
            if key in self.condition_films:
                modules.append(self.condition_films[key])
            self._own_modules(private_owner(task_id), *modules)
        self._validate_ownership()

    def _own_modules(self, owner: Ownership, *modules: nn.Module) -> None:
        for module in modules:
            for parameter in module.parameters():
                existing = self._ownership_by_parameter.get(parameter)
                if existing is not None and existing != owner:
                    raise RuntimeError(
                        f"Stage 3 parameter has multiple owners: {existing} and {owner}"
                    )
                self._ownership_by_parameter[parameter] = owner

    def _validate_ownership(self) -> None:
        missing = [
            name
            for name, parameter in self.named_parameters()
            if parameter not in self._ownership_by_parameter
        ]
        if missing:
            raise RuntimeError("Stage 3 parameters lack ownership: " + ", ".join(missing))

    def parameter_ownership(self) -> dict[nn.Parameter, Ownership]:
        return dict(self._ownership_by_parameter)

    def ownership_manifest(self) -> dict[str, str]:
        return {
            name: self._ownership_by_parameter[parameter].label
            for name, parameter in self.named_parameters()
        }

    def parameters_for_owner(self, owner: Ownership) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for parameter, candidate in self._ownership_by_parameter.items()
            if candidate == owner
        )

    def private_modules_for_task(self, task_id: str) -> tuple[nn.Module, ...]:
        if task_id not in self.task_specs or not self.task_specs[task_id].enabled:
            raise KeyError(f"Unknown Stage 3 private task: {task_id}")
        key = sanitize_task(task_id)
        modules: list[nn.Module] = [
            self.private_experts[key],
            self.task_gates[key],
            self.task_normalizations[key],
            self.towers[key],
        ]
        if key in self.condition_films:
            modules.append(self.condition_films[key])
        return tuple(modules)

    def set_task_refinement_mode(self, task_id: str) -> None:
        self.eval()
        for module in self.private_modules_for_task(task_id):
            module.train()

    def forward(
        self,
        task_id: str,
        primary_embedding: torch.Tensor,
        conditions: torch.Tensor,
        *,
        partner_embedding: torch.Tensor | None = None,
    ) -> Stage3ForwardOutput:
        spec = self.task_specs.get(task_id)
        if spec is None or not spec.enabled:
            raise ValueError(f"Inactive Stage 3 task: {task_id}")
        if self.descriptor_adapters:
            primary_topology = (
                "il"
                if tuple(spec.primary_slots) == ("cation", "anion")
                else "molecule"
            )
            primary_embedding = self.descriptor_adapters[primary_topology](
                primary_embedding
            )
            if partner_embedding is not None:
                partner_embedding = self.descriptor_adapters["molecule"](
                    partner_embedding
                )
        key = sanitize_task(task_id)
        z_global, l1_global_weights = _mixture(
            self.l1_global_experts,
            primary_embedding,
            self.l1_global_gate(primary_embedding),
        )
        z_group_delta, l1_group_weights = _mixture(
            self.l1_group_experts[spec.meta_group],
            primary_embedding,
            self.l1_group_gates[spec.meta_group](primary_embedding),
        )
        local = self.l1_group_normalizations[spec.meta_group](
            primary_embedding + z_group_delta
        )
        if spec.condition_columns:
            if conditions.shape[-1] != len(spec.condition_columns):
                raise ValueError(f"Stage 3 condition width mismatch: {task_id}")
            local = self.condition_films[key](local, conditions)
        elif conditions.shape[-1] != 0:
            raise ValueError(f"Condition-free Stage 3 task received conditions: {task_id}")
        if spec.partner_mode == "interaction":
            if partner_embedding is None:
                raise ValueError(f"Stage 3 task requires partner embedding: {task_id}")
            local = self.interactions[spec.meta_group](local, partner_embedding)
        elif partner_embedding is not None:
            raise ValueError(f"Stage 3 task must not receive partner embedding: {task_id}")

        global_outputs = torch.stack(
            [expert(z_global) for expert in self.l2_global_experts], dim=1
        )
        group_outputs = torch.stack(
            [expert(local) for expert in self.l2_group_experts[spec.meta_group]], dim=1
        )
        private_outputs = torch.stack(
            [expert(local) for expert in self.private_experts[key]], dim=1
        )
        task_gate = torch.softmax(
            self.task_gates[key](torch.cat((z_global, local), dim=-1)), dim=-1
        )
        candidates = torch.cat((global_outputs, group_outputs, private_outputs), dim=1)
        mixed = (task_gate.unsqueeze(-1) * candidates).sum(dim=1)
        representation = (
            self.task_normalizations[key](local + mixed)
            if self.model_config.l2_residual
            else mixed
        )
        return Stage3ForwardOutput(
            predictions=self.towers[key](representation),
            diagnostics={
                "z_global": z_global,
                "z_group": z_group_delta,
                "l1_global_gate": l1_global_weights,
                "l1_group_gate": l1_group_weights,
                "task_gate": task_gate,
                "l2_global_candidates": global_outputs,
                "l2_group_candidates": group_outputs,
                "l2_private_candidates": private_outputs,
            },
        )
