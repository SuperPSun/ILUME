from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
from typing import Any
from common.io import sha256_file
from common.outputs import repository_path, repository_relative
from benchmarks.common.config import BenchmarkConfig
from benchmarks.common.environment import environment_snapshot, validate_lock


def iltransr_asset_snapshot(config: BenchmarkConfig) -> dict[str, Any]:
    if config.name != "iltransr":
        raise ValueError("ILTransR asset validation requires an ILTransR config")
    paths = {
        "generic_checkpoint": repository_path(config.model["generic_checkpoint"]),
        "source_vocab": repository_path(config.model["source_vocab"]),
        "target_vocab": repository_path(config.model["target_vocab"]),
        "converted_checkpoint": repository_path(config.model["converted_checkpoint"]),
        "parity_reference": repository_path(config.model["parity_reference"]),
        "conversion_manifest": repository_path(config.model["conversion_manifest"]),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        command = (
            "download the three pinned GitHub files with the README ILTransR curl commands, then run: "
            "PYTHONPATH=src:. conda run --no-capture-output -n ilume-iltransr-convert "
            "python -m benchmarks.iltransr.conversion "
            "--checkpoint artifacts/benchmarks/iltransr/pretraining/valid_best.params "
            "--source-vocab artifacts/benchmarks/iltransr/pretraining/vocab.random_smiles.json "
            "--target-vocab artifacts/benchmarks/iltransr/pretraining/vocab.rdkit_canonical_smiles.json "
            "--output artifacts/benchmarks/iltransr/pretraining/encoder.safetensors "
            "--parity-reference artifacts/benchmarks/iltransr/pretraining/parity_reference.safetensors "
            "--manifest artifacts/benchmarks/iltransr/pretraining/conversion_manifest.json"
        )
        raise FileNotFoundError(
            "ILTransR pretrained assets are missing: " + ", ".join(missing) + "; " + command
        )
    expected_hashes = {
        "generic_checkpoint": config.model["generic_checkpoint_sha256"],
        "source_vocab": config.model["source_vocab_sha256"],
        "target_vocab": config.model["target_vocab_sha256"],
        "converted_checkpoint": config.model["converted_checkpoint_sha256"],
        "parity_reference": config.model["parity_reference_sha256"],
        "conversion_manifest": config.model["conversion_manifest_sha256"],
    }
    actual_hashes = {name: sha256_file(paths[name]) for name in expected_hashes}
    mismatch = {
        name: {"expected": expected_hashes[name], "actual": actual}
        for name, actual in actual_hashes.items()
        if actual != expected_hashes[name]
    }
    if mismatch:
        raise ValueError("ILTransR pretrained asset hash mismatch: " + json.dumps(mismatch, sort_keys=True))
    manifest = json.loads(paths["conversion_manifest"].read_text(encoding="utf-8"))
    required_manifest = {
        "format_version": 1,
        "upstream_revision": config.model["revision"],
        "source_sha256": config.model["generic_checkpoint_sha256"],
        "converted_sha256": config.model["converted_checkpoint_sha256"],
        "tensor_state_sha256": config.model["converted_tensor_state_sha256"],
        "parity_reference_sha256": config.model["parity_reference_sha256"],
    }
    if any(manifest.get(key) != value for key, value in required_manifest.items()):
        raise ValueError("ILTransR conversion manifest differs from the pinned conversion contract")
    conversion_environment = manifest.get("conversion_environment")
    expected_conversion = {
        "mxnet": "1.9.1",
        "gluonnlp": "0.10.0",
        "numpy": "1.23.5",
        "safetensors": "0.4.5",
        "device": "cpu",
    }
    if not isinstance(conversion_environment, dict) or any(
        conversion_environment.get(key) != value for key, value in expected_conversion.items()
    ) or conversion_environment.get("python") != "3.8.20":
        raise ValueError("ILTransR conversion environment differs from the pinned CPU contract")
    from benchmarks.iltransr.model import ILTransRTransformer
    from safetensors import safe_open

    with safe_open(paths["converted_checkpoint"], framework="pt", device="cpu") as handle:
        converted_names = set(handle.keys())
    expected_names = set(ILTransRTransformer(72).state_dict())
    if converted_names != expected_names or set(manifest.get("converted_tensors", {})) != expected_names:
        raise ValueError("ILTransR converted checkpoint tensor structure mismatch")
    ignored = manifest.get("ignored_pretraining_tensors", [])
    if not ignored or any(
        not str(name).startswith(("decoder.", "one_step_ahead_decoder.", "tgt_embed.", "tgt_proj."))
        for name in ignored
    ):
        raise ValueError("ILTransR conversion manifest has invalid ignored pretraining tensors")
    from safetensors.torch import load_file
    from benchmarks.iltransr.model import load_converted_transformer

    reference = load_file(str(paths["parity_reference"]), device="cpu")
    transformer, _ = load_converted_transformer(
        str(paths["converted_checkpoint"]), vocab_size=72
    )
    transformer.eval()
    import torch

    with torch.inference_mode():
        actual = transformer(reference["token_ids"], reference["valid_lengths"])
    difference = (actual - reference["mxnet_output"]).abs()
    parity = {
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "max_abs_threshold": 1.0e-5,
        "mean_abs_threshold": 1.0e-6,
    }
    if (
        parity["max_abs_error"] > parity["max_abs_threshold"]
        or parity["mean_abs_error"] > parity["mean_abs_threshold"]
    ):
        raise RuntimeError("ILTransR MXNet/PyTorch encoder parity check failed")
    return {
        "repository": config.model["repository"],
        "revision": config.model["revision"],
        "assets": {
            name: {
                "path": repository_relative(path),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "structure": {
            "converted_tensors": len(expected_names),
            "tensor_state_sha256": manifest["tensor_state_sha256"],
            "source_embedding_and_encoder_only": True,
            "property_specific_weights_loaded": False,
            "ignored_pretraining_tensor_count": len(ignored),
            "mxnet_pytorch_parity": parity,
        },
        "conversion_environment": conversion_environment,
        "official_historical_environment": {
            "python": "3.8",
            "mxnet": "mxnet-cu112==1.9.1",
            "cuda": "11.2",
            "rdkit_pypi": "2022.3.2.1",
        },
    }


def validate_iltransr_environment(config: BenchmarkConfig) -> dict[str, Any]:
    if config.name != "iltransr" or config.environment is None:
        raise ValueError("ILTransR environment validation requires an ILTransR config")
    try:
        import numpy
        from rdkit import rdBase
        import safetensors
        import torch
    except ImportError as error:
        raise RuntimeError("ILTransR environment cannot import its locked runtime") from error
    direct = {
        "python": platform.python_version(),
        "pip": importlib.metadata.version("pip"),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": numpy.__version__,
        "safetensors": safetensors.__version__,
        "rdkit": rdBase.rdkitVersion,
    }
    expected_direct = {
        "python": "3.10.14",
        "pip": "25.2",
        "pytorch": "2.9.0+cu128",
        "cuda": "12.8",
        "numpy": "1.26.4",
        "safetensors": "0.4.5",
        "rdkit": "2022.03.2",
    }
    definition, lock, installed = validate_lock(
        config, expected_direct=expected_direct, direct=direct
    )
    return {
        **environment_snapshot(config, definition, lock, installed, direct, torch),
        "pretrained_snapshot": iltransr_asset_snapshot(config),
    }
