from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import subprocess
from pathlib import Path
from typing import Any
from common.io import sha256_file
from common.outputs import repository_path
from benchmarks.common.config import BenchmarkConfig
from benchmarks.common.environment import environment_snapshot, validate_lock


def _load_ilbert_tokenizer_class(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "ilume_pinned_ilbert_tokenizer", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load pinned ILBERT tokenizer source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SMILES_Atomwise_Tokenizer


def ilbert_asset_snapshot(config: BenchmarkConfig) -> dict[str, Any]:
    if config.name != "ilbert":
        raise ValueError("ILBERT asset validation requires an ILBERT config")
    checkout = repository_path(str(config.model["checkout"]))
    model_source = checkout / "ILBERT" / "model.py"
    tokenizer_source = checkout / "ILBERT" / "ILtokenizer.py"
    vocab = checkout / "ILBERT" / "merged_vocab.txt"
    checkpoint = repository_path(str(config.model["pretrained_checkpoint"]))
    required = {
        "model.py": (model_source, str(config.model["model_source_sha256"])),
        "ILtokenizer.py": (
            tokenizer_source,
            str(config.model["tokenizer_source_sha256"]),
        ),
        "merged_vocab.txt": (vocab, str(config.model["vocab_sha256"])),
        "pretrained_model.pth": (
            checkpoint,
            str(config.model["pretrained_sha256"]),
        ),
    }
    missing = sorted(name for name, (path, _) in required.items() if not path.is_file())
    if missing:
        raise FileNotFoundError("ILBERT local assets are incomplete: " + ", ".join(missing))
    mismatches = {
        name: {"expected": expected, "actual": sha256_file(path)}
        for name, (path, expected) in required.items()
        if sha256_file(path) != expected
    }
    if mismatches:
        raise RuntimeError(
            "ILBERT local asset hash mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    actual_revision = revision.stdout.strip()
    if revision.returncode != 0 or actual_revision != str(config.model["revision"]):
        raise RuntimeError(
            "ILBERT checkout revision mismatch: "
            f"expected {config.model['revision']}, got {actual_revision or 'unavailable'}"
        )
    tokenizer_class = _load_ilbert_tokenizer_class(tokenizer_source)
    tokenizer = tokenizer_class(str(vocab))
    special_ids = {
        "pad": tokenizer.pad_token_id,
        "unk": tokenizer.unk_token_id,
        "cls": tokenizer.cls_token_id,
        "sep": tokenizer.sep_token_id,
        "mask": tokenizer.mask_token_id,
    }
    if int(tokenizer.vocab_size) != 2000 or special_ids != {
        "pad": 0,
        "unk": 1,
        "cls": 2,
        "sep": 3,
        "mask": 4,
    }:
        raise RuntimeError("ILBERT tokenizer vocabulary or special IDs differ from upstream")
    return {
        "repository": str(config.model["repository"]),
        "revision": actual_revision,
        "files": {
            name: {"sha256": expected, "size": path.stat().st_size}
            for name, (path, expected) in sorted(required.items())
        },
        "tokenizer": {"vocab_size": 2000, "special_token_ids": special_ids},
    }


def validate_ilbert_environment(config: BenchmarkConfig) -> dict[str, Any]:
    if config.name != "ilbert" or config.environment is None:
        raise ValueError("ILBERT environment validation requires an ILBERT config")
    try:
        import atomInSmiles
        import numpy
        from rdkit import rdBase
        import tokenizers
        import torch
        import transformers
    except ImportError as error:
        raise RuntimeError("ILBERT environment cannot import its locked runtime") from error
    direct = {
        "python": platform.python_version(),
        "pip": importlib.metadata.version("pip"),
        "atominsmiles": importlib.metadata.version("atomInSmiles"),
        "numpy": numpy.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "rdkit": rdBase.rdkitVersion,
    }
    expected_direct = {
        "python": "3.11.9",
        "pip": "25.2",
        "atominsmiles": "1.0.2",
        "numpy": "1.26.4",
        "transformers": "4.39.1",
        "tokenizers": "0.15.2",
        "pytorch": "2.9.0+cu128",
        "cuda": "12.8",
        "rdkit": "2023.09.5",
    }
    definition, lock, installed = validate_lock(
        config, expected_direct=expected_direct, direct=direct
    )
    return {
        **environment_snapshot(config, definition, lock, installed, direct, torch),
        "pretrained_snapshot": ilbert_asset_snapshot(config),
    }
