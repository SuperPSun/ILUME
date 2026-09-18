from __future__ import annotations

import importlib.metadata
import importlib.util
import hashlib
import json
import os
import platform
from typing import Any, Mapping
from common.io import sha256_file
from common.outputs import repository_path
from benchmarks.common.config import BenchmarkConfig
from benchmarks.common.environment import environment_snapshot, validate_lock

LLASMOL_ASSET_MARKER = "ILUME_LLASMOL_ASSETS"


def llasmol_asset_snapshot(config: BenchmarkConfig) -> dict[str, Any]:
    if config.name != "llasmol":
        raise ValueError("LlaSMol asset validation requires a LlaSMol config")
    base = repository_path(str(config.model["base_snapshot"]))
    adapter = repository_path(str(config.model["adapter_snapshot"]))
    shard_hashes = list(config.model["base_shard_sha256"])
    shard_sizes = list(config.model["base_shard_size"])
    required = {
        "base/config.json": (
            base / "config.json", str(config.model["base_config_sha256"]), 571
        ),
        "base/model.safetensors.index.json": (
            base / "model.safetensors.index.json",
            str(config.model["base_index_sha256"]),
            25125,
        ),
        "base/model-00001-of-00002.safetensors": (
            base / "model-00001-of-00002.safetensors",
            str(shard_hashes[0]),
            int(shard_sizes[0]),
        ),
        "base/model-00002-of-00002.safetensors": (
            base / "model-00002-of-00002.safetensors",
            str(shard_hashes[1]),
            int(shard_sizes[1]),
        ),
        "base/tokenizer.json": (
            base / "tokenizer.json", str(config.model["tokenizer_json_sha256"]), 1795188
        ),
        "base/tokenizer.model": (
            base / "tokenizer.model", str(config.model["tokenizer_model_sha256"]), 493443
        ),
        "base/tokenizer_config.json": (
            base / "tokenizer_config.json",
            str(config.model["tokenizer_config_sha256"]),
            996,
        ),
        "base/special_tokens_map.json": (
            base / "special_tokens_map.json",
            str(config.model["special_tokens_sha256"]),
            414,
        ),
        "adapter/adapter_config.json": (
            adapter / "adapter_config.json",
            str(config.model["adapter_config_sha256"]),
            653,
        ),
        "adapter/adapter_model.bin": (
            adapter / "adapter_model.bin",
            str(config.model["adapter_model_sha256"]),
            int(config.model["adapter_model_size"]),
        ),
    }
    missing = sorted(name for name, (path, _, _) in required.items() if not path.is_file())
    if missing:
        raise FileNotFoundError("LlaSMol local assets are incomplete: " + ", ".join(missing))
    sizes = {name: path.stat().st_size for name, (path, _, _) in required.items()}
    size_mismatches = {
        name: {"expected": expected, "actual": sizes[name]}
        for name, (_, _, expected) in required.items()
        if sizes[name] != expected
    }
    if size_mismatches:
        raise RuntimeError(
            "LlaSMol local asset size mismatch: "
            + json.dumps(size_mismatches, sort_keys=True)
        )
    marker_payload = {
        "base_revision": config.model["base_revision"],
        "adapter_revision": config.model["adapter_revision"],
        "files": {
            name: {"sha256": expected, "size": size}
            for name, (_, expected, size) in sorted(required.items())
        },
    }
    expected_marker = hashlib.sha256(
        json.dumps(marker_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    inherited_validation = os.environ.get(LLASMOL_ASSET_MARKER) == expected_marker
    large_shards = {
        "base/model-00001-of-00002.safetensors",
        "base/model-00002-of-00002.safetensors",
    }
    hashes = {
        name: (
            expected
            if inherited_validation and name in large_shards
            else sha256_file(path)
        )
        for name, (path, expected, _) in required.items()
    }
    hash_mismatches = {
        name: {"expected": expected, "actual": hashes[name]}
        for name, (_, expected, _) in required.items()
        if hashes[name] != expected
    }
    if hash_mismatches:
        raise RuntimeError(
            "LlaSMol local asset hash mismatch: "
            + json.dumps(hash_mismatches, sort_keys=True)
        )
    import torch

    adapter_state = torch.load(
        adapter / "adapter_model.bin", map_location="cpu", weights_only=True
    )
    if not isinstance(adapter_state, Mapping):
        raise RuntimeError("LlaSMol official adapter is not a tensor state mapping")
    target_modules = tuple(str(value) for value in config.model["lora_target_modules"])
    module_counts = {
        module: sum(f".{module}." in str(name) for name in adapter_state)
        for module in target_modules
    }
    if (
        len(adapter_state) != int(config.model["adapter_state_entries"])
        or any(not isinstance(value, torch.Tensor) for value in adapter_state.values())
        or any(value.dtype != torch.bfloat16 for value in adapter_state.values())
        or any(count != 64 for count in module_counts.values())
    ):
        raise RuntimeError("LlaSMol official adapter tensor contract mismatch")
    base_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    if {
        "model_type": base_config.get("model_type"),
        "hidden_size": base_config.get("hidden_size"),
        "num_hidden_layers": base_config.get("num_hidden_layers"),
        "vocab_size": base_config.get("vocab_size"),
    } != {
        "model_type": "mistral",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "vocab_size": 32000,
    }:
        raise RuntimeError("LlaSMol base Mistral config differs from the registered model")
    adapter_config = json.loads(
        (adapter / "adapter_config.json").read_text(encoding="utf-8")
    )
    if (
        adapter_config.get("base_model_name_or_path")
        != str(config.model["base_repository"])
        or adapter_config.get("peft_type") != "LORA"
        or adapter_config.get("task_type") != "CAUSAL_LM"
        or int(adapter_config.get("r", -1)) != int(config.model["lora_rank"])
        or int(adapter_config.get("lora_alpha", -1))
        != int(config.model["lora_alpha"])
        or float(adapter_config.get("lora_dropout", -1.0))
        != float(config.model["lora_dropout"])
        or set(adapter_config.get("target_modules", ()))
        != set(config.model["lora_target_modules"])
    ):
        raise RuntimeError("LlaSMol official adapter config differs from contract")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(base), local_files_only=True, use_fast=True
    )
    special_ids = {
        "pad": tokenizer.pad_token_id,
        "unk": tokenizer.unk_token_id,
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
    }
    if int(tokenizer.vocab_size) != 32000 or special_ids != {
        "pad": None,
        "unk": 0,
        "bos": 1,
        "eos": 2,
    }:
        raise RuntimeError("LlaSMol tokenizer vocabulary or special IDs differ from base")
    snapshot = {
        "base": {
            "repository": str(config.model["base_repository"]),
            "revision": str(config.model["base_revision"]),
        },
        "adapter": {
            "repository": str(config.model["adapter_repository"]),
            "revision": str(config.model["adapter_revision"]),
        },
        "files": {
            name: {"sha256": hashes[name], "size": sizes[name]}
            for name in sorted(required)
        },
        "tokenizer": {"vocab_size": 32000, "special_token_ids": special_ids},
        "checkpoint_trust": "pinned_official_weights_only_pickle",
        "adapter_state": {
            "entries": len(adapter_state),
            "dtype": "torch.bfloat16",
            "module_entry_counts": module_counts,
        },
    }
    os.environ[LLASMOL_ASSET_MARKER] = expected_marker
    return snapshot


def validate_llasmol_environment(config: BenchmarkConfig) -> dict[str, Any]:
    if config.name != "llasmol" or config.environment is None:
        raise ValueError("LlaSMol environment validation requires a LlaSMol config")
    try:
        import accelerate
        import bitsandbytes
        from bitsandbytes.cextension import lib as bitsandbytes_library
        import numpy
        import peft
        from rdkit import rdBase
        import tokenizers
        import torch
        import transformers
    except ImportError as error:
        raise RuntimeError("LlaSMol environment cannot import its locked runtime") from error
    direct = {
        "python": platform.python_version(),
        "pip": importlib.metadata.version("pip"),
        "accelerate": accelerate.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "numpy": numpy.__version__,
        "peft": peft.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "rdkit": rdBase.rdkitVersion,
    }
    expected_direct = {
        "python": "3.12.12",
        "pip": "25.2",
        "accelerate": "1.12.0",
        "bitsandbytes": "0.49.2",
        "numpy": "2.5.2",
        "peft": "0.18.1",
        "transformers": "4.57.6",
        "tokenizers": "0.22.2",
        "pytorch": "2.9.0+cu128",
        "cuda": "12.8",
        "rdkit": "2026.03.5",
    }
    definition, lock, installed = validate_lock(
        config, expected_direct=expected_direct, direct=direct
    )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("LlaSMol QLoRA requires CUDA BF16 support")
    if not getattr(bitsandbytes_library, "compiled_with_cuda", False):
        raise RuntimeError("LlaSMol QLoRA requires the bitsandbytes CUDA backend")
    return {
        **environment_snapshot(config, definition, lock, installed, direct, torch),
        "pretrained_snapshot": llasmol_asset_snapshot(config),
    }
