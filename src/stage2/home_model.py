from __future__ import annotations
from typing import Any
import json
from stage3.token_attention import token_model
import torch
from torch import nn
from common.entity_inputs import EntityInputs
from stage2.model import encode_object_entities
from stage3.model import Stage3SparseModel
from .home_contract import source_task_specs

class SimulationHoME(nn.Module):
    """Five-task HoME with permanently frozen Stage1 entity inputs."""
    def __init__(self, backbone, registry, base_config, stage2_config, *, scalers=None):
        super().__init__()
        if not stage2_config.is_entity_home or (stage2_config.frozen_encoder is None and not backbone.config.is_dual_view):
            raise ValueError("Historical ObjectEncoder models require their historical Git revision")
        self.backbone = backbone.requires_grad_(False).eval()
        self.stage2_config = stage2_config
        self.registry = registry
        self.home = Stage3SparseModel(base_config.model,
            source_task_specs(registry, role_policy="formal_charge_v1"), 1024,
            group_configs=base_config.groups, initialization_seed=stage2_config.data.seed if stage2_config.training.seed is None else stage2_config.training.seed,
            entity_inputs=True, entity_input_dim=(1241 if stage2_config.frozen_encoder is not None
                else 1024 + backbone.descriptor_schema.selected_dim))
        if token_model(base_config.model):
            if scalers is None:
                scalers = json.loads((stage2_config.data.artifacts_dir / "scalers.json").read_text())
            self.home.bind_condition_statistics(scalers)

    @property
    def model_contract(self):
        contract = {**self.home.entity_input_contract, "source_tasks": sorted(self.registry.task_ids)}
        if self.stage2_config.frozen_encoder is not None:
            contract["frozen_encoder"] = self.backbone.encoder_identity["hash"]
        ablation = self.backbone.config.to_dict().get("ablation")
        if ablation is not None:
            contract["stage1_ablation"] = ablation
        return contract

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
