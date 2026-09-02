"""Catalog-driven Stage 2 physics representation training."""

from .frozen import (
    FrozenObjectSpec,
    FrozenRDKitStage2ObjectEncoder,
    FrozenStage2ObjectEncoder,
    load_frozen_object_encoder,
    load_stage2_encoder_identity,
)

__all__ = [
    "FrozenObjectSpec",
    "FrozenRDKitStage2ObjectEncoder",
    "FrozenStage2ObjectEncoder",
    "load_frozen_object_encoder",
    "load_stage2_encoder_identity",
]
