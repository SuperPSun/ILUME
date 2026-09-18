from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
from typing import Any
from common.io import sha256_file
from common.outputs import repository_path, repository_relative
from benchmarks.common.config import BenchmarkConfig
from benchmarks.common.environment import environment_snapshot, validate_lock


def aifc_asset_snapshot(config: BenchmarkConfig) -> dict[str, Any]:
    if config.name != "aifc":
        raise ValueError("AIFC asset validation requires an AIFC config")
    from benchmarks.aifc.preprocessing import (
        FRAGMENT_BLOB,
        FRAGMENT_COMMIT,
        FRAGMENT_SHA256,
        FragmentScheme,
    )

    path = repository_path(config.model["fragment_scheme"])
    if not path.is_file():
        raise FileNotFoundError(f"AIFC pinned fragment dictionary is missing: {path}")
    scheme = FragmentScheme.load(path)
    reference = repository_path(config.model["legacy_reference"])
    if (
        not reference.is_file()
        or sha256_file(reference) != config.model["legacy_reference_sha256"]
    ):
        raise ValueError("AIFC legacy parity reference hash differs from the registered contract")
    if (
        config.model["fragment_commit"] != FRAGMENT_COMMIT
        or config.model["fragment_blob"] != FRAGMENT_BLOB
        or config.model["fragment_sha256"] != FRAGMENT_SHA256
    ):
        raise ValueError("AIFC fragment provenance differs from the registered contract")
    return {
        "repository": config.model["repository"],
        "revision": config.model["revision"],
        "fragment_scheme": {
            "path": repository_relative(path),
            "commit": FRAGMENT_COMMIT,
            "blob": FRAGMENT_BLOB,
            "sha256": scheme.sha256,
            "size": path.stat().st_size,
            "entries": len(scheme.names),
            "order_sha256": scheme.order_sha256,
        },
        "graph_backend": config.model["graph_backend"],
        "legacy_parity": config.model["legacy_parity"],
        "legacy_reference": {
            "path": repository_relative(reference),
            "sha256": sha256_file(reference),
            "size": reference.stat().st_size,
        },
    }


def validate_aifc_environment(config: BenchmarkConfig) -> dict[str, Any]:
    if config.name != "aifc" or config.environment is None:
        raise ValueError("AIFC environment validation requires an AIFC config")
    try:
        import numpy
        from rdkit import rdBase
        import torch
    except ImportError as error:
        raise RuntimeError("AIFC environment cannot import its locked runtime") from error
    direct = {
        "python": platform.python_version(),
        "pip": importlib.metadata.version("pip"),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": numpy.__version__,
        "rdkit": rdBase.rdkitVersion,
    }
    expected_direct = {
        "python": "3.10.14",
        "pip": "25.2",
        "pytorch": "2.9.0+cu128",
        "cuda": "12.8",
        "numpy": "1.26.4",
        "rdkit": "2023.09.6",
    }
    definition, lock, installed = validate_lock(
        config, expected_direct=expected_direct, direct=direct
    )
    if not torch.cuda.is_available():
        raise RuntimeError("AIFC requires CUDA; no silent CPU fallback")
    from benchmarks.aifc.parity import validate_legacy_parity

    assets = aifc_asset_snapshot(config)
    parity = validate_legacy_parity(
        repository_path(config.model["legacy_reference"]),
        repository_path(config.model["fragment_scheme"]),
    )
    return {
        **environment_snapshot(config, definition, lock, installed, direct, torch),
        "pretrained_snapshot": {**assets, "pytorch_legacy_dgl_parity": parity},
    }
