from __future__ import annotations

from .config import DataConfig, PretrainConfig
from .data import _preparation_source_paths, prepare_corpus


def preparation_source_paths(config: PretrainConfig | DataConfig):
    data = config.data if isinstance(config, PretrainConfig) else config
    return _preparation_source_paths(data)


__all__ = ["prepare_corpus", "preparation_source_paths"]
