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


def spmm_asset_snapshot(config: BenchmarkConfig) -> dict[str, Any]:
    if config.name != "spmm":
        raise ValueError("SPMM asset validation requires an SPMM config")
    checkout = repository_path(str(config.model["checkout"]))
    checkpoint = repository_path(str(config.model["pretrained_checkpoint"]))
    required = {
        "SPMM_models.py": (
            checkout / "SPMM_models.py", str(config.model["spmm_source_sha256"])
        ),
        "xbert.py": (
            checkout / "xbert.py", str(config.model["xbert_source_sha256"])
        ),
        "d_regression.py": (
            checkout / "d_regression.py",
            str(config.model["regression_source_sha256"]),
        ),
        "vocab_bpe_300.txt": (
            checkout / "vocab_bpe_300.txt", str(config.model["vocab_sha256"])
        ),
        "config_bert.json": (
            checkout / "config_bert.json", str(config.model["bert_config_sha256"])
        ),
        "checkpoint_SPMM.ckpt": (
            checkpoint, str(config.model["pretrained_sha256"])
        ),
    }
    missing = sorted(name for name, (path, _) in required.items() if not path.is_file())
    if missing:
        raise FileNotFoundError("SPMM local assets are incomplete: " + ", ".join(missing))
    if checkpoint.stat().st_size != int(config.model["pretrained_size"]):
        raise RuntimeError(
            "SPMM pretrained checkpoint size mismatch: "
            f"expected {config.model['pretrained_size']}, got {checkpoint.stat().st_size}"
        )
    actual_hashes = {name: sha256_file(path) for name, (path, _) in required.items()}
    mismatches = {
        name: {"expected": expected, "actual": actual_hashes[name]}
        for name, (_, expected) in required.items()
        if actual_hashes[name] != expected
    }
    if mismatches:
        raise RuntimeError(
            "SPMM local asset hash mismatch: " + json.dumps(mismatches, sort_keys=True)
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
            "SPMM checkout revision mismatch: "
            f"expected {config.model['revision']}, got {actual_revision or 'unavailable'}"
        )
    vocab = checkout / "vocab_bpe_300.txt"
    max_input_chars = int(config.model["wordpiece_max_input_chars_per_word"])
    tokenizer = _spmm_tokenizer(vocab, max_input_chars)
    special_ids = {
        "pad": tokenizer.pad_token_id,
        "unk": tokenizer.unk_token_id,
        "cls": tokenizer.cls_token_id,
        "sep": tokenizer.sep_token_id,
        "mask": tokenizer.mask_token_id,
    }
    if int(tokenizer.vocab_size) != 300 or special_ids != {
        "pad": 0,
        "unk": 1,
        "cls": 2,
        "sep": 3,
        "mask": 1,
    }:
        raise RuntimeError("SPMM tokenizer vocabulary or special IDs differ from upstream")
    return {
        "repository": str(config.model["repository"]),
        "revision": actual_revision,
        "files": {
            name: {"sha256": actual_hashes[name], "size": path.stat().st_size}
            for name, (path, _) in sorted(required.items())
        },
        "tokenizer": {
            "vocab_size": 300,
            "special_token_ids": special_ids,
            "wordpiece_max_input_chars_per_word": max_input_chars,
        },
        "checkpoint_trust": "pinned_official_lightning_pickle",
    }


def _spmm_tokenizer(vocab: Path, max_input_chars_per_word: int) -> Any:
    from transformers import BertTokenizer, WordpieceTokenizer

    tokenizer = BertTokenizer(
        vocab_file=str(vocab), do_lower_case=False, do_basic_tokenize=False
    )
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(
        vocab=tokenizer.vocab,
        unk_token=tokenizer.unk_token,
        max_input_chars_per_word=max_input_chars_per_word,
    )
    return tokenizer


def validate_spmm_environment(config: BenchmarkConfig) -> dict[str, Any]:
    if config.name != "spmm" or config.environment is None:
        raise ValueError("SPMM environment validation requires an SPMM config")
    try:
        import numpy
        from rdkit import rdBase
        import tokenizers
        import torch
        import transformers
    except ImportError as error:
        raise RuntimeError("SPMM environment cannot import its locked runtime") from error
    direct = {
        "python": platform.python_version(),
        "pip": importlib.metadata.version("pip"),
        "numpy": numpy.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "rdkit": rdBase.rdkitVersion,
    }
    expected_direct = {
        "python": "3.10.14",
        "pip": "25.2",
        "numpy": "1.24.3",
        "transformers": "4.30.1",
        "tokenizers": "0.13.3",
        "pytorch": "2.9.0+cu128",
        "cuda": "12.8",
        "rdkit": "2023.03.1",
    }
    definition, lock, installed = validate_lock(
        config, expected_direct=expected_direct, direct=direct
    )
    return {
        **environment_snapshot(config, definition, lock, installed, direct, torch),
        "pretrained_snapshot": spmm_asset_snapshot(config),
    }
