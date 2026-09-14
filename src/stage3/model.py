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

ROUTING_MODES = (
    "learned_gate",
    "no_private",
    "global_only",
    "global_floor_025",
    "global_floor_050",
)


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


def apply_routing_ablation(
    task_gate: torch.Tensor,
    global_count: int,
    group_count: int,
    private_count: int,
    routing_mode: str,
) -> torch.Tensor:
    """Apply an evaluation-only intervention to L2 task-gate weights."""
    if routing_mode not in ROUTING_MODES:
        raise ValueError(f"Unsupported Stage 3 routing mode: {routing_mode}")
    counts = (global_count, group_count, private_count)
    if (
        task_gate.ndim != 2
        or not task_gate.is_floating_point()
        or any(count < 0 for count in counts)
    ):
        raise ValueError("Stage 3 routing candidate counts are invalid")
    if sum(counts) != task_gate.shape[1] or global_count <= 0:
        raise ValueError("Stage 3 routing candidate partition mismatch")
    if not torch.isfinite(task_gate).all() or bool((task_gate < 0).any()):
        raise ValueError("Stage 3 routing weights must be finite and non-negative")
    if not torch.allclose(
        task_gate.sum(dim=1),
        torch.ones(task_gate.shape[0], device=task_gate.device, dtype=task_gate.dtype),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("Stage 3 routing weights must sum to one")
    if routing_mode == "learned_gate":
        return task_gate

    local_count = group_count + private_count
    if routing_mode == "no_private":
        kept_count = global_count + group_count
        if kept_count <= 0:
            raise ValueError("Stage 3 no_private routing has no candidates")
        kept = task_gate[:, :kept_count]
        mass = kept.sum(dim=1, keepdim=True)
        normalized = kept / mass.clamp_min(torch.finfo(task_gate.dtype).tiny)
        fallback = torch.full_like(kept, 1.0 / kept_count)
        return torch.cat(
            (
                torch.where(mass > 0, normalized, fallback),
                torch.zeros_like(task_gate[:, kept_count:]),
            ),
            dim=1,
        )
    if routing_mode == "global_only":
        global_weights = task_gate[:, :global_count]
        mass = global_weights.sum(dim=1, keepdim=True)
        normalized = global_weights / mass.clamp_min(
            torch.finfo(task_gate.dtype).tiny
        )
        fallback = torch.full_like(global_weights, 1.0 / global_count)
        return torch.cat(
            (
                torch.where(mass > 0, normalized, fallback),
                torch.zeros_like(task_gate[:, global_count:]),
            ),
            dim=1,
        )

    floor = 0.25 if routing_mode == "global_floor_025" else 0.50
    if local_count == 0:
        return task_gate
    global_weights = task_gate[:, :global_count]
    local_weights = task_gate[:, global_count:]
    global_mass = global_weights.sum(dim=1, keepdim=True)
    local_mass = local_weights.sum(dim=1, keepdim=True)
    scaled_global = global_weights * (
        floor / global_mass.clamp_min(torch.finfo(task_gate.dtype).tiny)
    )
    scaled_global = torch.where(
        global_mass > 0,
        scaled_global,
        torch.full_like(global_weights, floor / global_count),
    )
    scaled_local = local_weights * (
        (1.0 - floor) / local_mass.clamp_min(torch.finfo(task_gate.dtype).tiny)
    )
    scaled_local = torch.where(
        local_mass > 0,
        scaled_local,
        torch.full_like(local_weights, (1.0 - floor) / local_count),
    )
    adjusted = torch.cat((scaled_global, scaled_local), dim=1)
    return torch.where(global_mass < floor, adjusted, task_gate)


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
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.task_specs = dict(task_specs)
        self.group_configs = dict(group_configs or {})
        self.task_configs = dict(task_configs or {})
        self.task_private_recipes = dict(task_private_recipes or {})
        self.d_model = d_model
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

        self.private_experts = nn.ModuleDict()
        self.task_gates = nn.ModuleDict()
        self.condition_films = nn.ModuleDict()
        self.task_normalizations = nn.ModuleDict()
        self.towers = nn.ModuleDict()
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

    def forward(
        self,
        task_id: str,
        primary_embedding: torch.Tensor,
        conditions: torch.Tensor,
        *,
        partner_embedding: torch.Tensor | None = None,
        routing_mode: str = "learned_gate",
    ) -> Stage3ForwardOutput:
        if routing_mode != "learned_gate" and self.training:
            raise ValueError("Stage 3 routing ablations are evaluation-only")
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
        if self.l1_global_gate is None:
            z_global = primary_embedding
            l1_global_weights = primary_embedding.new_empty(
                (primary_embedding.shape[0], 0)
            )
        else:
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

        global_outputs = _expert_outputs(self.l2_global_experts, z_global)
        group_outputs = _expert_outputs(
            self.l2_group_experts[spec.meta_group], local
        )
        private_outputs = _expert_outputs(self.private_experts[key], local)
        learned_task_gate = torch.softmax(
            self.task_gates[key](torch.cat((z_global, local), dim=-1)), dim=-1
        )
        candidates = torch.cat((global_outputs, group_outputs, private_outputs), dim=1)
        learned_predictions = None
        if routing_mode == "learned_gate":
            task_gate = learned_task_gate
        else:
            learned_mixed = (
                learned_task_gate.unsqueeze(-1) * candidates
            ).sum(dim=1)
            learned_representation = (
                self.task_normalizations[key](local + learned_mixed)
                if self.model_config.l2_residual
                else learned_mixed
            )
            learned_predictions = self.towers[key](learned_representation)
            task_gate = apply_routing_ablation(
                learned_task_gate,
                global_outputs.shape[1],
                group_outputs.shape[1],
                private_outputs.shape[1],
                routing_mode,
            )
        mixed = (task_gate.unsqueeze(-1) * candidates).sum(dim=1)
        representation = (
            self.task_normalizations[key](local + mixed)
            if self.model_config.l2_residual
            else mixed
        )
        diagnostics = {
            "z_global": z_global,
            "z_group": z_group_delta,
            "l1_global_gate": l1_global_weights,
            "l1_group_gate": l1_group_weights,
            "task_gate": task_gate,
            "l2_global_candidates": global_outputs,
            "l2_group_candidates": group_outputs,
            "l2_private_candidates": private_outputs,
        }
        if learned_predictions is not None:
            diagnostics["learned_gate_predictions"] = learned_predictions
        return Stage3ForwardOutput(
            predictions=self.towers[key](representation),
            diagnostics=diagnostics,
        )
