"""Explicit, entry-point-registered frozen entity sources for Stage2 and Stage3."""
from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch


_FACTORIES: dict[str, Callable[..., Any]] = {}


@dataclass(frozen=True)
class FrozenEncoderConfig:
    kind: str
    variant: str
    repo_id: str
    revision: str
    snapshot_dir: str
    pooling: str = "last_hidden_cls"
    include_rdkit: bool = False

    @property
    def is_dual_view(self):
        return False

    def to_dict(self):
        return asdict(self)


def register_frozen_encoder(kind: str, factory: Callable[..., Any]) -> None:
    if kind in _FACTORIES and _FACTORIES[kind] is not factory:
        raise ValueError(f"Frozen encoder already registered: {kind}")
    _FACTORIES[kind] = factory


def load_frozen_encoder(spec: FrozenEncoderConfig, *, bundle=None, device="cpu"):
    if spec.kind not in _FACTORIES:
        raise ValueError(f"Unregistered frozen encoder: {spec.kind}; use the public Stage entry point")
    source = _FACTORIES[spec.kind](spec, bundle=bundle)
    return source.requires_grad_(False).to(device).eval()


@dataclass(frozen=True)
class FrozenEntityBatch:
    roles: torch.Tensor
    frozen_entity_embedding: torch.Tensor

    def to(self, device, *, non_blocking=False):
        return FrozenEntityBatch(self.roles.to(device, non_blocking=non_blocking),
                                 self.frozen_entity_embedding.to(device, non_blocking=non_blocking))

    def pin_memory(self):
        return FrozenEntityBatch(self.roles.pin_memory(), self.frozen_entity_embedding.pin_memory())


class FrozenEntityPacker:
    def __call__(self, samples):
        return FrozenEntityBatch(torch.tensor([s["role_id"] for s in samples], dtype=torch.long),
                                 torch.stack([s["frozen_entity_embedding"] for s in samples]))
