from __future__ import annotations

import importlib.metadata
import importlib.util
import hashlib
import json
import os
import platform
from typing import Any
from common.io import sha256_file
from common.outputs import repository_path
from benchmarks.common.config import BenchmarkConfig
from benchmarks.common.environment import environment_snapshot, validate_lock

AIONOPEDIA_ASSET_MARKER = "ILUME_AIONOPEDIA_ASSETS"


def aionopedia_asset_snapshot(config: BenchmarkConfig) -> dict[str, Any]:
    if config.name != "aionopedia":
        raise ValueError("AIonopedia asset snapshot requires an AIonopedia config")
    marker_payload = {
        "base_revision": config.model["base_revision"],
        "pretrained_revision": config.model["pretrained_revision"],
        "adapter_config_provenance": config.model["adapter_config_provenance"],
        "base_files": config.model["base_files"],
        "pretrained_files": config.model["pretrained_files"],
    }
    expected_marker = hashlib.sha256(
        json.dumps(marker_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    inherited_validation = os.environ.get(AIONOPEDIA_ASSET_MARKER) == expected_marker
    groups = {}
    for group, root_field in (
        ("base_files", "base_snapshot"),
        ("pretrained_files", "pretrained_snapshot"),
    ):
        root = repository_path(config.model[root_field])
        if not root.is_dir():
            raise FileNotFoundError(f"AIonopedia asset directory is missing: {root}")
        files = {}
        for filename, expected in config.model[group].items():
            path = root / filename
            if not path.is_file():
                raise FileNotFoundError(f"AIonopedia asset is missing: {path}")
            actual = {
                "sha256": expected["sha256"] if inherited_validation else sha256_file(path),
                "size": path.stat().st_size,
            }
            if actual != expected:
                raise ValueError(
                    f"AIonopedia asset integrity mismatch for {group}/{filename}: "
                    f"expected {expected}, got {actual}"
                )
            files[filename] = actual
        groups[group] = files
    structure = {
        "base_model_type": "qwen3",
        "base_hidden_size": 1024,
        "pretraining_outputs_ignored": 71,
        "released_module_files": sorted(config.model["pretrained_files"]),
        "base_tensor_width_validated": True,
        "released_lora_only_validated": True,
        "released_modules_strictly_loaded": True,
        "adapter_config_provenance": config.model["adapter_config_provenance"],
        "adapter_config_byte_identical_to_current_hf_revision": False,
    }
    if not inherited_validation:
        import torch
        from safetensors import safe_open

        from benchmarks.aionopedia.adapter import OFFICIAL_MODULE_FILES, OFFICIAL_SEGMENTS
        from benchmarks.aionopedia.model import MultiModalRegressor

        base = repository_path(config.model["base_snapshot"])
        pretrained = repository_path(config.model["pretrained_snapshot"])
        base_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
        if {
            "model_type": base_config.get("model_type"),
            "hidden_size": base_config.get("hidden_size"),
        } != {"model_type": "qwen3", "hidden_size": 1024}:
            raise RuntimeError("AIonopedia Qwen base config differs from contract")
        with safe_open(base / "model.safetensors", framework="pt", device="cpu") as state:
            embedding_shape = tuple(state.get_slice("model.embed_tokens.weight").get_shape())
        if embedding_shape[-1] != 1024:
            raise RuntimeError("AIonopedia Qwen tensor width differs from contract")

        adapter_config = json.loads(
            (pretrained / "adapter_config.json").read_text(encoding="utf-8")
        )
        adapter_semantics = {
            "peft_type": adapter_config.get("peft_type"),
            "task_type": adapter_config.get("task_type"),
            "r": adapter_config.get("r"),
            "lora_alpha": adapter_config.get("lora_alpha"),
            "lora_dropout": adapter_config.get("lora_dropout"),
            "target_modules": sorted(adapter_config.get("target_modules", [])),
            "bias": adapter_config.get("bias"),
            "inference_mode": adapter_config.get("inference_mode"),
            "auto_mapping": adapter_config.get("auto_mapping"),
        }
        expected_adapter_semantics = {
            "peft_type": "LORA",
            "task_type": None,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["k_proj", "o_proj", "q_proj", "v_proj"],
            "bias": "none",
            "inference_mode": True,
            "auto_mapping": {
                "base_model_class": "Qwen3ForCausalLM",
                "parent_library": "transformers.models.qwen3.modeling_qwen3",
            },
        }
        if adapter_semantics != expected_adapter_semantics:
            raise RuntimeError("AIonopedia local LoRA config differs from pinned contract")
        with safe_open(
            pretrained / "adapter_model.safetensors", framework="pt", device="cpu"
        ) as state:
            adapter_keys = tuple(state.keys())
        if not adapter_keys or any("lora_" not in name.lower() for name in adapter_keys):
            raise RuntimeError("AIonopedia released adapter contains non-LoRA tensors")

        model = MultiModalRegressor(torch.nn.Identity(), llm_dim=1024)
        for filename, attribute in OFFICIAL_MODULE_FILES.items():
            getattr(model, attribute).load_state_dict(
                torch.load(
                    pretrained / filename, map_location="cpu", weights_only=True
                ),
                strict=True,
            )
        segments = torch.load(
            pretrained / "segment_embeddings.pt", map_location="cpu", weights_only=True
        )
        if set(segments) != set(OFFICIAL_SEGMENTS):
            raise RuntimeError("AIonopedia released segment tensors differ from contract")
        for name in OFFICIAL_SEGMENTS:
            if tuple(segments[name].shape) != tuple(getattr(model, name).shape):
                raise RuntimeError(f"AIonopedia released segment shape differs: {name}")
        pretraining_head = torch.nn.Sequential(
            torch.nn.Linear(512, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 71)
        )
        pretraining_head.load_state_dict(
            torch.load(
                pretrained / "fc_out_state_dict.pt",
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        os.environ[AIONOPEDIA_ASSET_MARKER] = expected_marker
    return {
        "base_repository": config.model["base_repository"],
        "base_revision": config.model["base_revision"],
        "pretrained_repository": config.model["pretrained_repository"],
        "pretrained_revision": config.model["pretrained_revision"],
        "upstream_repository": config.model["upstream_repository"],
        "upstream_revision": config.model["upstream_revision"],
        "structure": structure,
        **groups,
    }


def validate_aionopedia_environment(config: BenchmarkConfig) -> dict[str, Any]:
    if config.name != "aionopedia" or config.environment is None:
        raise ValueError("AIonopedia environment validation requires an AIonopedia config")
    try:
        import peft
        from rdkit import rdBase
        import torch
        import torch_geometric
        import transformers
    except ImportError as error:
        raise RuntimeError("AIonopedia environment cannot import its locked runtime") from error
    direct = {
        "python": platform.python_version(),
        "pip": importlib.metadata.version("pip"),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "torch_geometric": torch_geometric.__version__,
        "rdkit": rdBase.rdkitVersion,
    }
    expected_direct = {
        "python": "3.12.9",
        "pytorch": "2.9.0+cu128",
        "cuda": "12.8",
        "transformers": "4.52.4",
        "peft": "0.15.2",
        "torch_geometric": "2.6.1",
        "rdkit": "2023.09.5",
    }
    definition, lock, installed = validate_lock(
        config, expected_direct=expected_direct, direct=direct
    )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("AIonopedia requires CUDA BF16 support")
    return {
        **environment_snapshot(config, definition, lock, installed, direct, torch),
        "pretrained_snapshot": aionopedia_asset_snapshot(config),
    }
