"""Catalog-driven Stage2 physics supervision."""

# Keep the registry accessible to isolated baseline environments without eagerly loading Stage1 graph dependencies.
__all__ = (
    "FrozenObjectSpec",
    "load_frozen_stage1_entities",
    "load_frozen_object_encoder",
    "load_stage2_encoder_identity",
)


def __getattr__(name: str):
    if name in __all__:
        from . import frozen

        value = getattr(frozen, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
