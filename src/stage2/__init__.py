"""Catalog-driven Stage 2 physics representation training."""

from .frozen import (
    FrozenObjectSpec,
    FrozenStage2ObjectEncoder,
    load_frozen_object_encoder,
)

__all__ = [
    "FrozenObjectSpec",
    "FrozenStage2ObjectEncoder",
    "load_frozen_object_encoder",
]
