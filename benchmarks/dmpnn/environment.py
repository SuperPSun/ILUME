from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
from typing import Any
from benchmarks.common.config import BenchmarkConfig
from benchmarks.common.environment import environment_snapshot, validate_lock


def validate_dmpnn_environment(config: BenchmarkConfig) -> dict[str, Any]:
    if config.name != "dmpnn" or config.environment is None:
        raise ValueError("D-MPNN environment validation requires a D-MPNN config")
    try:
        import chemprop
        import lightning
        from rdkit import rdBase
        import torch
    except ImportError as error:
        raise RuntimeError("D-MPNN environment cannot import its locked runtime") from error
    direct = {
        "python": platform.python_version(),
        "pip": importlib.metadata.version("pip"),
        "chemprop": importlib.metadata.version("chemprop"),
        "lightning": lightning.__version__,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "rdkit": rdBase.rdkitVersion,
    }
    expected_direct = {
        "python": "3.12.12",
        "pip": "25.2",
        "chemprop": "2.3.1",
        "pytorch": "2.9.0+cu128",
        "cuda": "12.8",
        "rdkit": "2026.03.5",
    }
    definition, lock, installed = validate_lock(
        config, expected_direct=expected_direct, direct=direct
    )
    return {
        **environment_snapshot(config, definition, lock, installed, direct, torch),
    }
