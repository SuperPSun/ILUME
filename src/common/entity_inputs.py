"""Frozen entity slots consumed directly by GLOBAL and GROUP experts."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F

ENTITY_INPUT_CONTRACT = {
    "architecture_kind": "entity_home_v1", "representation_contract": "entity_home_v4",
    "slot_contract": "ordered_entity_slots_v1", "entity_dim": 1241,
    "learned_dim": 1024, "packed_dim": 2490, "role_policy": "formal_charge_v1",
    "role_to_id": {"cation": 0, "anion": 1, "neutral": 2},
    "residual": "valid_learned_mean", "initialization": "seed_owner_v1",
}

@dataclass(frozen=True)
class EntityInputs:
    values: torch.Tensor
    roles: torch.Tensor
    mask: torch.Tensor

    def to(self, device):
        return EntityInputs(self.values.to(device), self.roles.to(device), self.mask.to(device))

    @classmethod
    def from_slots(cls, values, roles):
        if values.ndim != 3 or values.shape[1] not in (1, 2):
            raise ValueError("Entity inputs require one entity or an ordered ion pair")
        count = values.shape[1]
        if count == 1:
            values = F.pad(values, (0, 0, 0, 1))
            roles = F.pad(roles, (0, 1))
        mask = torch.arange(2, device=values.device)[None, :].expand(len(values), -1) < count
        result = cls(values, roles, mask)
        result.pack()
        return result

    def pack(self):
        if (self.values.ndim != 3 or self.values.shape[1:] != (2, 1241)
                or self.roles.shape != self.values.shape[:2] or self.mask.shape != self.roles.shape
                or self.mask.dtype != torch.bool or self.roles.dtype != torch.long):
            raise ValueError("Entity slot tensor contract mismatch")
        if not bool(self.mask[:, 0].all()) or not bool(((self.roles >= 0) & (self.roles < 3)).all()):
            raise ValueError("Invalid entity mask or role")
        pairs = self.roles[self.mask[:, 1]]
        if not torch.equal(pairs, pairs.new_tensor([0, 1]).expand_as(pairs)):
            raise ValueError("Ionic liquid requires ordered cation and anion roles")
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError("Non-finite entity input")
        valid = self.mask.unsqueeze(-1)
        values = self.values * valid
        roles = F.one_hot(self.roles, 3).to(values.dtype) * valid
        packed = torch.cat((values.flatten(1), roles.flatten(1), self.mask.to(values.dtype)), -1)
        anchor = values[..., :1024].sum(1) / valid.sum(1)
        return packed, anchor
