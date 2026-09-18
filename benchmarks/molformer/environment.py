from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
from pathlib import Path
from typing import Any
from common.io import sha256_file
from benchmarks.common.config import BenchmarkConfig
from benchmarks.common.environment import environment_snapshot, validate_lock


def validate_molformer_environment(config: BenchmarkConfig) -> dict[str, Any]:
    if config.name != "molformer" or config.environment is None:
        raise ValueError("MoLFormer environment validation requires a MoLFormer config")
    try:
        from huggingface_hub import snapshot_download
        from rdkit import rdBase
        import torch
        import transformers
    except ImportError as error:
        raise RuntimeError("MoLFormer environment cannot import its locked runtime") from error
    direct = {
        "python": platform.python_version(),
        "pip": importlib.metadata.version("pip"),
        "transformers": transformers.__version__,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "rdkit": rdBase.rdkitVersion,
    }
    expected_direct = {
        "python": "3.12.12",
        "pip": "25.2",
        "transformers": "5.12.1",
        "pytorch": "2.9.0+cu128",
        "cuda": "12.8",
        "rdkit": "2026.03.5",
    }
    definition, lock, installed = validate_lock(
        config, expected_direct=expected_direct, direct=direct
    )
    repository = str(config.model["repository"])
    revision = str(config.model["revision"])
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=repository,
                revision=revision,
                local_files_only=True,
            )
        )
    except Exception as error:
        raise RuntimeError(
            f"MoLFormer snapshot {repository}@{revision} is not available locally"
        ) from error
    required = {
        "config.json",
        "configuration_molformer.py",
        "model.safetensors",
        "modeling_molformer.py",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    missing = sorted(name for name in required if not (snapshot / name).is_file())
    if missing:
        raise FileNotFoundError(
            "MoLFormer snapshot is incomplete: " + ", ".join(missing)
        )
    files = {
        name: {
            "sha256": sha256_file(snapshot / name),
            "size": (snapshot / name).stat().st_size,
        }
        for name in sorted(required)
    }
    return {
        **environment_snapshot(config, definition, lock, installed, direct, torch),
        "pretrained_snapshot": {
            "repository": repository,
            "revision": revision,
            "files": files,
        },
    }
