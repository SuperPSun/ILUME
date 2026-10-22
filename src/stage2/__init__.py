"""Catalog-driven Stage2 physics supervision."""

# 注册表供基线独立环境读取，避免提前加载Stage1图特征依赖。
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
