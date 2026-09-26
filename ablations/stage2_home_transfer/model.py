from __future__ import annotations

from typing import Any

import torch
from torch import nn

from stage2.model import ObjectEncoder, RECONSTRUCTION_MODULES
from stage3.model import Stage3SparseModel

from .contract import model_task_ids, source_task_specs


class SimulationHoME(nn.Module):
    """Nine simulation objectives with a shared Stage3-shaped HoME decoder."""

    def __init__(self, backbone: nn.Module, registry: Any, base_config: Any, stage2_config: Any) -> None:
        super().__init__()
        self.backbone = backbone
        for name, parameter in backbone.named_parameters():
            if any(name == prefix or name.startswith(prefix + ".") for prefix in RECONSTRUCTION_MODULES):
                parameter.requires_grad_(False)
        self.object_encoder = ObjectEncoder(
            backbone.entity_dim, backbone.config.model.n_heads,
            num_layers=stage2_config.model.object_layers,
            feedforward_dim=stage2_config.model.object_ffn_dim,
            dropout=stage2_config.model.dropout,
        )
        group_configs = {
            name: base_config.groups[name] for name in ("thermophysical", "solvation")
        }
        group_configs["electronic_structure"] = base_config.groups["thermophysical"]
        self.home = Stage3SparseModel(
            base_config.model, source_task_specs(registry), backbone.entity_dim,
            group_configs=group_configs,
        )
        self.atom_adapter = nn.Sequential(
            nn.Linear(backbone.atom_dim + backbone.entity_dim, backbone.entity_dim),
            nn.SiLU(), nn.LayerNorm(backbone.entity_dim),
        )
        self.registry = registry

    def backbone_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(parameter for parameter in self.backbone.parameters() if parameter.requires_grad)

    def home_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.home.parameters()) + tuple(self.atom_adapter.parameters())

    def predict(self, task_id: str, packed: Any, dataset: Any) -> torch.Tensor:
        if packed.entities is None or packed.entity_positions is None:
            raise ValueError("Stage2-HoME needs live Stage1 entities for every task")
        spec = self.registry.by_id(task_id)
        positions = packed.entity_positions
        encoded = self.backbone.encode_entity(packed.entities)
        slots = encoded.entity_embedding[positions]
        roles = packed.entities.roles[positions]
        conditions = dataset.conditions[packed.row_indices]
        if spec.target_level == "atom":
            atoms = packed.atom_targets
            if atoms is None:
                raise ValueError("Stage2-HoME atom task is missing packed atom targets")
            objects = self.object_encoder(slots, roles)
            atom_values = encoded.atom_states[atoms.atom_state_indices]
            object_values = objects[atoms.atom_sample_indices]
            embedding = self.atom_adapter(torch.cat((atom_values, object_values), dim=-1))
            empty_conditions = embedding.new_empty((len(embedding), 0))
            return self.home(task_id, embedding, empty_conditions).predictions
        if spec.topology == "interaction":
            primary = self.object_encoder(slots[:, :1], roles[:, :1])
            partner = self.object_encoder(slots[:, 1:], roles[:, 1:])
        else:
            primary = self.object_encoder(slots, roles)
            partner = None
        model_tasks = model_task_ids(task_id, len(spec.target_columns))
        columns = [
            self.home(item, primary, conditions, partner_embedding=partner).predictions
            for item in model_tasks
        ]
        return torch.stack(columns, dim=-1)
