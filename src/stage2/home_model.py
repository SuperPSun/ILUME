from __future__ import annotations

from typing import Any

import torch
from torch import nn

from stage2.model import (ObjectEncoder, RECONSTRUCTION_MODULES, encode_object_entities,
                          build_object_encoder, object_encoder_contract)
from common.training import seeded_initialization
from stage3.model import Stage3SparseModel

from .home_contract import model_task_ids, source_task_specs


class SimulationHoME(nn.Module):
    """Simulation objectives with a shared Stage3-shaped HoME decoder."""

    def __init__(self, backbone: nn.Module, registry: Any, base_config: Any, stage2_config: Any) -> None:
        super().__init__()
        self.backbone = backbone
        if getattr(backbone.config, "is_dual_view", False):
            backbone.requires_grad_(False).eval()
        for name, parameter in backbone.named_parameters():
            if any(name == prefix or name.startswith(prefix + ".") for prefix in RECONSTRUCTION_MODULES):
                parameter.requires_grad_(False)
        self.stage2_config = stage2_config
        if stage2_config.is_v5:
            with seeded_initialization(stage2_config.data.seed, "OBJECT"):
                self.object_encoder = build_object_encoder(backbone.entity_dim, backbone.config.model.n_heads,
                    object_encoder_contract(stage2_config.model, backbone.entity_dim + 217))
        else:
            self.object_encoder = ObjectEncoder(
                backbone.entity_dim, backbone.config.model.n_heads,
                num_layers=stage2_config.model.object_layers,
                feedforward_dim=stage2_config.model.object_ffn_dim,
                dropout=stage2_config.model.dropout,
                input_dim=backbone.entity_dim + 217 if getattr(backbone.config, "is_dual_view", False) else None,
            )
        group_configs = {
            name: base_config.groups[name] for name in ("thermophysical", "solvation")
        }
        if "simulation/homo" in registry.task_ids:
            group_configs["electronic_structure"] = base_config.groups["thermophysical"]
        self.home = Stage3SparseModel(
            base_config.model, source_task_specs(registry, role_policy="formal_charge_v1" if stage2_config.is_v5 else "legacy_slot_v1"), backbone.entity_dim,
            group_configs=group_configs,
            initialization_seed=stage2_config.data.seed if stage2_config.is_v5 else None,
        )
        self.atom_adapter = None
        if "simulation/partial_atomic_charge" in registry.task_ids:
            with seeded_initialization(stage2_config.data.seed if stage2_config.is_v5 else None, "ATOM_ADAPTER"):
                self.atom_adapter = nn.Sequential(
                    nn.Linear(backbone.atom_dim + backbone.entity_dim, backbone.entity_dim),
                    nn.SiLU(), nn.LayerNorm(backbone.entity_dim),
                )
        self.registry = registry

    @property
    def model_contract(self) -> dict[str, Any]:
        return {"d_model": self.backbone.entity_dim,
                "object_encoder": {**object_encoder_contract(self.stage2_config.model, self.backbone.entity_dim + 217),
                                   "source_tasks": sorted(self.registry.task_ids)}}

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self.backbone.config, "is_dual_view", False):
            self.backbone.eval()
        return self

    def backbone_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(parameter for parameter in self.backbone.parameters() if parameter.requires_grad)

    def home_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.home.parameters()) + (tuple(self.atom_adapter.parameters()) if self.atom_adapter is not None else ())

    def predict(self, task_id: str, packed: Any, dataset: Any) -> torch.Tensor:
        return predict_simulation_task(
            self.backbone, self.object_encoder, self.home, self.atom_adapter,
            self.registry, task_id, packed, dataset,
        )


def predict_simulation_task(
    backbone: nn.Module, object_encoder: nn.Module, home: Stage3SparseModel,
    atom_adapter: nn.Module | None, registry: Any, task_id: str, packed: Any,
    dataset: Any,
) -> torch.Tensor:
    """Run a simulation task through a HoME decoder and its frozen feature path."""
    if packed.entities is None or packed.entity_positions is None:
        raise ValueError("Stage2-HoME needs live Stage1 entities for every task")
    spec = registry.by_id(task_id)
    positions = packed.entity_positions
    encoded = encode_object_entities(backbone, packed.entities)
    slots = encoded.entity_embedding[positions]
    roles = packed.entities.roles[positions]
    conditions = dataset.conditions[packed.row_indices]
    if spec.target_level == "atom":
        atoms = packed.atom_targets
        if atoms is None or atom_adapter is None:
            raise ValueError("Stage2-HoME atom task is missing targets or its adapter")
        objects = object_encoder(slots, roles)
        atom_values = encoded.atom_states[atoms.atom_state_indices]
        object_values = objects[atoms.atom_sample_indices]
        embedding = atom_adapter(torch.cat((atom_values, object_values), dim=-1))
        empty_conditions = embedding.new_empty((len(embedding), 0))
        return home(task_id, embedding, empty_conditions).predictions
    if spec.topology == "interaction":
        primary = object_encoder(slots[:, :1], roles[:, :1])
        partner = object_encoder(slots[:, 1:], roles[:, 1:])
    else:
        primary = object_encoder(slots, roles)
        partner = None
    model_tasks = model_task_ids(task_id, len(spec.target_columns))
    columns = [
        home(item, primary, conditions, partner_embedding=partner).predictions
        for item in model_tasks
    ]
    return torch.stack(columns, dim=-1)
