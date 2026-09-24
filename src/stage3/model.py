from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import nn

from .config import (
    ResolvedStage3PrivateRecipe,
    Stage3GroupConfig,
    Stage3ModelConfig,
    Stage3TaskConfig,
    Stage3TransferKnowledgeConfig,
)
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


class KnowledgeMixer(nn.Module):
    def __init__(self, sources: tuple[str, ...]) -> None:
        super().__init__()
        self.sources = sources
        self.gamma = nn.Parameter(torch.zeros(()))
        self.logits = nn.Parameter(torch.zeros(len(sources)))

    def forward(self, anchor: torch.Tensor, deltas: Mapping[str, torch.Tensor]) -> torch.Tensor:
        weights = torch.softmax(self.logits, dim=0)
        combined = sum(weights[index] * deltas[source] for index, source in enumerate(self.sources))
        return anchor + self.gamma * combined


def _mixture(
    experts: Iterable[nn.Module],
    values: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.softmax(logits, dim=-1)
    outputs = torch.stack([expert(values) for expert in experts], dim=1)
    return (weights.unsqueeze(-1) * outputs).sum(dim=1), weights


def _expert_outputs(
    experts: Iterable[nn.Module], values: torch.Tensor
) -> torch.Tensor:
    outputs = [expert(values) for expert in experts]
    if outputs:
        return torch.stack(outputs, dim=1)
    return values.new_empty((values.shape[0], 0, values.shape[-1]))


@dataclass(frozen=True)
class Stage3ForwardOutput:
    predictions: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def task_gate_observations(
    diagnostics: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return per-sample GLOBAL/GROUP/PRIVATE mass and normalized entropy."""
    task_gate = diagnostics["task_gate"].detach().float()
    candidate_counts = tuple(
        int(diagnostics[name].shape[1])
        for name in (
            "l2_global_candidates",
            "l2_group_candidates",
            "l2_private_candidates",
        )
    )
    if task_gate.ndim != 2 or sum(candidate_counts) != task_gate.shape[1]:
        raise ValueError("Stage 3 task-gate candidate partition mismatch")
    masses = []
    offset = 0
    for count in candidate_counts:
        masses.append(task_gate[:, offset : offset + count].sum(dim=1))
        offset += count
    if task_gate.shape[1] > 1:
        entropy = torch.special.entr(task_gate).sum(dim=1) / math.log(
            task_gate.shape[1]
        )
    else:
        entropy = task_gate.new_zeros(task_gate.shape[0])
    return torch.stack((*masses, entropy), dim=1)


def summarize_task_gate_observations(
    observations: torch.Tensor,
) -> dict[str, float]:
    """Summarize rows produced by :func:`task_gate_observations`."""
    if observations.ndim != 2 or observations.shape[1] != 4:
        raise ValueError("Stage 3 task-gate observations must have four columns")
    values = observations.detach().double().cpu()
    if not torch.isfinite(values).all():
        raise RuntimeError("Non-finite Stage 3 task-gate diagnostics")
    if values.shape[0] == 0:
        return {
            name: float("nan")
            for name in (
                "mean_global_gate_weight",
                "mean_group_gate_weight",
                "mean_private_gate_weight",
                "task_gate_entropy",
                "private_gate_weight_p10",
                "private_gate_weight_p50",
                "private_gate_weight_p90",
            )
        }
    private_quantiles = torch.quantile(
        values[:, 2], torch.tensor((0.1, 0.5, 0.9), dtype=torch.float64)
    )
    return {
        "mean_global_gate_weight": float(values[:, 0].mean()),
        "mean_group_gate_weight": float(values[:, 1].mean()),
        "mean_private_gate_weight": float(values[:, 2].mean()),
        "task_gate_entropy": float(values[:, 3].mean()),
        "private_gate_weight_p10": float(private_quantiles[0]),
        "private_gate_weight_p50": float(private_quantiles[1]),
        "private_gate_weight_p90": float(private_quantiles[2]),
    }


class Stage3SparseModel(nn.Module):
    def __init__(
        self,
        model_config: Stage3ModelConfig,
        task_specs: Mapping[str, ResolvedTaskSpec],
        d_model: int,
        *,
        group_configs: Mapping[str, Stage3GroupConfig] | None = None,
        task_configs: Mapping[str, Stage3TaskConfig] | None = None,
        task_private_recipes: Mapping[str, ResolvedStage3PrivateRecipe] | None = None,
        descriptor_input_dims: Mapping[str, int] | None = None,
        transfer_knowledge: Stage3TransferKnowledgeConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.task_specs = dict(task_specs)
        self.group_configs = dict(group_configs or {})
        self.task_configs = dict(task_configs or {})
        self.task_private_recipes = dict(task_private_recipes or {})
        self.d_model = d_model
        self.transfer_knowledge = transfer_knowledge
        self.groups = tuple(
            sorted({spec.meta_group for spec in self.task_specs.values() if spec.enabled})
        )
        self._ownership_by_parameter: dict[nn.Parameter, Ownership] = {}
        self._modules_by_owner: dict[Ownership, list[nn.Module]] = {}
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
        self.l1_global_gate = (
            nn.Linear(d_model, model_config.global_experts)
            if model_config.global_experts
            else None
        )
        self.l2_global_experts = nn.ModuleList(
            [Expert(d_model, **expert_kwargs) for _ in range(model_config.global_experts)]
        )
        global_modules: list[nn.Module] = [
            self.l1_global_experts,
            self.l2_global_experts,
        ]
        if self.l1_global_gate is not None:
            global_modules.append(self.l1_global_gate)
        self._own_modules(GLOBAL, *global_modules)
        if transfer_knowledge is not None:
            self.global_knowledge_mixer = KnowledgeMixer(transfer_knowledge.global_sources)
            self._own_modules(GLOBAL, self.global_knowledge_mixer)

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
            group_config = self.group_configs.get(group)
            group_experts = (
                model_config.group_experts
                if group_config is None or group_config.experts is None
                else group_config.experts
            )
            group_hidden_ratio = (
                model_config.expert_hidden_ratio
                if group_config is None or group_config.expert_hidden_ratio is None
                else group_config.expert_hidden_ratio
            )
            group_expert_kwargs = {
                **expert_kwargs,
                "hidden_ratio": group_hidden_ratio,
            }
            self.l1_group_experts[group] = nn.ModuleList(
                [Expert(d_model, **group_expert_kwargs) for _ in range(group_experts)]
            )
            self.l1_group_gates[group] = nn.Linear(d_model, group_experts)
            self.l1_group_normalizations[group] = nn.LayerNorm(d_model)
            self.l2_group_experts[group] = nn.ModuleList(
                [Expert(d_model, **group_expert_kwargs) for _ in range(group_experts)]
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

        self.group_knowledge_mixers = nn.ModuleDict()
        if transfer_knowledge is not None:
            for group, entry in transfer_knowledge.group_sources.items():
                self.group_knowledge_mixers[group] = KnowledgeMixer(entry.sources)
                self._own_modules(group_owner(group), self.group_knowledge_mixers[group])

        self.private_experts = nn.ModuleDict()
        self.task_gates = nn.ModuleDict()
        self.condition_films = nn.ModuleDict()
        self.task_normalizations = nn.ModuleDict()
        self.towers = nn.ModuleDict()
        self.private_knowledge_mixers = nn.ModuleDict()
        for task_id, spec in self.task_specs.items():
            if not spec.enabled:
                continue
            key = sanitize_task(task_id)
            task_config = self.task_configs.get(task_id)
            overrides = task_config.model_overrides if task_config is not None else {}
            private_recipe = self.task_private_recipes.get(task_id)
            private_experts = int(
                overrides.get("private_experts", model_config.private_experts)
            )
            private_hidden_ratio = float(
                private_recipe.private_hidden_ratio
                if private_recipe is not None
                else overrides.get("private_hidden_ratio", model_config.expert_hidden_ratio)
            )
            tower_hidden_ratio = float(
                private_recipe.tower_hidden_ratio
                if private_recipe is not None
                else overrides.get("tower_hidden_ratio", model_config.tower_hidden_ratio)
            )
            film_hidden_ratio = float(
                private_recipe.film_hidden_ratio
                if private_recipe is not None
                else overrides.get("film_hidden_ratio", model_config.film_hidden_ratio)
            )
            private_dropout = float(
                private_recipe.private_dropout
                if private_recipe is not None
                else overrides.get("private_dropout", model_config.dropout)
            )
            group_config = self.group_configs.get(spec.meta_group)
            group_experts = (
                model_config.group_experts
                if group_config is None or group_config.experts is None
                else group_config.experts
            )
            candidate_count = (
                model_config.global_experts + group_experts + private_experts
            )
            self.private_experts[key] = nn.ModuleList(
                [
                    Expert(
                        d_model,
                        hidden_ratio=private_hidden_ratio,
                        dropout=private_dropout,
                        activation=model_config.activation,
                    )
                    for _ in range(private_experts)
                ]
            )
            self.task_gates[key] = nn.Linear(2 * d_model, candidate_count)
            if spec.condition_columns:
                self.condition_films[key] = ConditionFiLM(
                    len(spec.condition_columns),
                    d_model,
                    hidden_ratio=film_hidden_ratio,
                    dropout=private_dropout,
                    activation=model_config.activation,
                )
            self.task_normalizations[key] = (
                nn.LayerNorm(d_model) if model_config.l2_residual else nn.Identity()
            )
            self.towers[key] = TaskTower(
                d_model,
                hidden_ratio=tower_hidden_ratio,
                dropout=private_dropout,
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
            if transfer_knowledge is not None and task_id in transfer_knowledge.private_sources:
                self.private_knowledge_mixers[key] = KnowledgeMixer(
                    transfer_knowledge.private_sources[task_id]
                )
                self._own_modules(private_owner(task_id), self.private_knowledge_mixers[key])
        self._validate_ownership()

    def resolved_capacity_recipe(self) -> dict[str, object]:
        groups: dict[str, dict[str, int | float]] = {}
        for group in self.groups:
            config = self.group_configs.get(group)
            experts = (
                self.model_config.group_experts
                if config is None or config.experts is None
                else config.experts
            )
            ratio = (
                self.model_config.expert_hidden_ratio
                if config is None or config.expert_hidden_ratio is None
                else config.expert_hidden_ratio
            )
            groups[group] = {
                "experts": experts,
                "expert_hidden_ratio": ratio,
                "expert_hidden": _width(self.d_model, ratio),
            }
        tasks: dict[str, object] = {}
        for task_id, spec in self.task_specs.items():
            if not spec.enabled:
                continue
            config = self.task_configs.get(task_id)
            overrides = config.model_overrides if config is not None else {}
            private_recipe = self.task_private_recipes.get(task_id)
            private_experts = int(
                overrides.get("private_experts", self.model_config.private_experts)
            )
            private_ratio = float(
                private_recipe.private_hidden_ratio
                if private_recipe is not None
                else overrides.get(
                    "private_hidden_ratio", self.model_config.expert_hidden_ratio
                )
            )
            tower_ratio = float(
                private_recipe.tower_hidden_ratio
                if private_recipe is not None
                else overrides.get("tower_hidden_ratio", self.model_config.tower_hidden_ratio)
            )
            film_ratio = float(
                private_recipe.film_hidden_ratio
                if private_recipe is not None
                else overrides.get("film_hidden_ratio", self.model_config.film_hidden_ratio)
            )
            private_dropout = float(
                private_recipe.private_dropout
                if private_recipe is not None
                else overrides.get("private_dropout", self.model_config.dropout)
            )
            tasks[task_id] = {
                "private_experts": private_experts,
                "private_hidden_ratio": private_ratio,
                "private_hidden": _width(self.d_model, private_ratio),
                "tower_hidden_ratio": tower_ratio,
                "tower_hidden": _width(self.d_model, tower_ratio),
                "film_hidden_ratio": film_ratio,
                "film_hidden": _width(self.d_model, film_ratio),
                "private_dropout": private_dropout,
                "candidate_count": (
                    self.model_config.global_experts
                    + int(groups[spec.meta_group]["experts"])
                    + private_experts
                ),
            }
        return {"groups": groups, "tasks": tasks}

    def _own_modules(self, owner: Ownership, *modules: nn.Module) -> None:
        owned_modules = self._modules_by_owner.setdefault(owner, [])
        for module in modules:
            if module not in owned_modules:
                owned_modules.append(module)
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

    def set_trainable_owners(self, owners: Iterable[Ownership]) -> None:
        selected = set(owners)
        unknown = selected - set(self._modules_by_owner)
        if unknown:
            raise KeyError(
                "Unknown Stage 3 owners: "
                + ", ".join(owner.label for owner in sorted(unknown))
            )
        for parameter, owner in self._ownership_by_parameter.items():
            parameter.requires_grad_(owner in selected)
        self.eval()
        for owner in selected:
            for module in self._modules_by_owner[owner]:
                module.train()

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

    def knowledge_sources(self, task_id: str) -> tuple[str, ...]:
        config = self.transfer_knowledge
        if config is None:
            return ()
        group = self.task_specs[task_id].meta_group
        sources = list(config.global_sources)
        entry = config.group_sources.get(group)
        if entry is not None and task_id in entry.tasks:
            sources.extend(entry.sources)
        sources.extend(config.private_sources.get(task_id, ()))
        return tuple(dict.fromkeys(sources))

    def forward(
        self,
        task_id: str,
        primary_embedding: torch.Tensor,
        conditions: torch.Tensor,
        *,
        partner_embedding: torch.Tensor | None = None,
        primary_knowledge: Mapping[str, torch.Tensor] | None = None,
        partner_knowledge: Mapping[str, torch.Tensor] | None = None,
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
        group = spec.meta_group
        if self.transfer_knowledge is not None:
            if primary_knowledge is None:
                raise ValueError("Transfer knowledge primary deltas are required")
            group_enabled = (
                group in self.transfer_knowledge.group_sources
                and task_id in self.transfer_knowledge.group_sources[group].tasks
            )
            global_primary = self.global_knowledge_mixer(primary_embedding, primary_knowledge)
            group_primary = (
                self.group_knowledge_mixers[group](global_primary, primary_knowledge)
                if group_enabled else global_primary
            )
            private_primary = (
                self.private_knowledge_mixers[key](group_primary, primary_knowledge)
                if key in self.private_knowledge_mixers else group_primary
            )
            if partner_embedding is not None:
                if partner_knowledge is None:
                    raise ValueError("Transfer knowledge partner deltas are required")
                global_partner = self.global_knowledge_mixer(partner_embedding, partner_knowledge)
                group_partner = (
                    self.group_knowledge_mixers[group](global_partner, partner_knowledge)
                    if group_enabled else global_partner
                )
                private_partner = (
                    self.private_knowledge_mixers[key](group_partner, partner_knowledge)
                    if key in self.private_knowledge_mixers else group_partner
                )
            else:
                group_partner = private_partner = None
        else:
            global_primary = group_primary = private_primary = primary_embedding
            group_partner = private_partner = partner_embedding
        if self.l1_global_gate is None:
            z_global = global_primary
            l1_global_weights = primary_embedding.new_empty(
                (primary_embedding.shape[0], 0)
            )
        else:
            z_global, l1_global_weights = _mixture(
                self.l1_global_experts,
                global_primary,
                self.l1_global_gate(global_primary),
            )
        z_group_delta, l1_group_weights = _mixture(
            self.l1_group_experts[spec.meta_group],
            group_primary,
            self.l1_group_gates[spec.meta_group](group_primary),
        )
        local = self.l1_group_normalizations[spec.meta_group](
            group_primary + z_group_delta
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
            local = self.interactions[spec.meta_group](local, group_partner)
        elif partner_embedding is not None:
            raise ValueError(f"Stage 3 task must not receive partner embedding: {task_id}")

        global_outputs = _expert_outputs(self.l2_global_experts, z_global)
        group_outputs = _expert_outputs(
            self.l2_group_experts[spec.meta_group], local
        )
        private_local = local
        if self.transfer_knowledge is not None and key in self.private_knowledge_mixers:
            private_delta, _ = _mixture(
                self.l1_group_experts[group], private_primary,
                self.l1_group_gates[group](private_primary),
            )
            private_local = self.l1_group_normalizations[group](private_primary + private_delta)
            if spec.condition_columns:
                private_local = self.condition_films[key](private_local, conditions)
            if spec.partner_mode == "interaction":
                private_local = self.interactions[group](private_local, private_partner)
        private_outputs = _expert_outputs(self.private_experts[key], private_local)
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
