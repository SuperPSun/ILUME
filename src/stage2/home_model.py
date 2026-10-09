from __future__ import annotations
from typing import Any
import torch
from torch import nn
from common.entity_inputs import EntityInputs, ENTITY_INPUT_CONTRACT
from stage2.model import encode_object_entities
from stage3.model import Stage3SparseModel
from .home_contract import source_task_specs

class SimulationHoME(nn.Module):
    """Five-task HoME with permanently frozen Stage1 entity inputs."""
    def __init__(self, backbone, registry, base_config, stage2_config):
        super().__init__()
        if not stage2_config.is_entity_home or not backbone.config.is_dual_view:
            raise ValueError("Historical ObjectEncoder models require their historical Git revision")
        self.backbone = backbone.requires_grad_(False).eval()
        self.stage2_config = stage2_config
        self.registry = registry
        self.home = Stage3SparseModel(base_config.model,
            source_task_specs(registry, role_policy="formal_charge_v1"), 1024,
            group_configs=base_config.groups, initialization_seed=stage2_config.data.seed,
            entity_inputs=True)

    @property
    def model_contract(self):
        return {**ENTITY_INPUT_CONTRACT, "source_tasks": sorted(self.registry.task_ids)}

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def backbone_parameters(self):
        return ()

    def home_parameters(self):
        return tuple(self.home.parameters())

    def predict(self, task_id, packed, dataset):
        return predict_simulation_task(self.backbone, self.home, self.registry, task_id, packed, dataset)

def predict_simulation_task(backbone, home, registry, task_id, packed, dataset):
    if packed.entities is None or packed.entity_positions is None:
        raise ValueError("HoME requires frozen Stage1 entities")
    spec = registry.by_id(task_id)
    if spec.target_level == "atom" or len(spec.target_columns) != 1:
        raise ValueError("Entity HoME only supports the five scalar simulation tasks")
    positions = packed.entity_positions
    encoded = encode_object_entities(backbone, packed.entities)
    slots, roles = encoded.entity_embedding[positions], packed.entities.roles[positions]
    if spec.topology == "interaction":
        primary = EntityInputs.from_slots(slots[:, :1], roles[:, :1])
        partner = EntityInputs.from_slots(slots[:, 1:], roles[:, 1:])
    else:
        primary, partner = EntityInputs.from_slots(slots, roles), None
    conditions = dataset.conditions[packed.row_indices]
    return home(task_id, primary, conditions, partner_embedding=partner).predictions[:, None]
