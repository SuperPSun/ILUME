from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from common.io import atomic_json, sha256_file
from common.outputs import REPOSITORY_ROOT, repository_path, repository_relative

from .config import BenchmarkConfig


ENVIRONMENT_MARKER = "ILUME_BENCHMARK_ENVIRONMENT"
_LOCKED_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==([^ ;\\]+)")


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _locked_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = _LOCKED_REQUIREMENT.match(raw.strip())
        if match:
            versions[_canonical_name(match.group(1))] = match.group(2)
    if not versions:
        raise ValueError(f"Benchmark dependency lock has no pinned packages: {path}")
    return versions


def environment_command(
    config: BenchmarkConfig, argv: Sequence[str], *, conda: str | None = None
) -> list[str]:
    if config.name not in {"dmpnn", "molformer", "ilbert", "spmm"} or config.environment is None:
        raise ValueError("Environment dispatch is only defined for advanced baselines")
    executable = conda or shutil.which("conda")
    if executable is None:
        raise RuntimeError(f"{config.display_name} requires conda; executable was not found")
    if not argv:
        raise ValueError("Environment dispatch requires an entrypoint argv")
    return [
        executable,
        "run",
        "--no-capture-output",
        "-n",
        config.environment.name,
        "python",
        str(Path(argv[0]).resolve()),
        *argv[1:],
    ]


def ensure_benchmark_environment(
    config: BenchmarkConfig, argv: Sequence[str] | None = None
) -> dict[str, Any] | None:
    if config.name not in {"dmpnn", "molformer", "ilbert", "spmm"}:
        return None
    if config.environment is None:
        raise ValueError(f"{config.display_name} environment contract is missing")
    marker = os.environ.get(ENVIRONMENT_MARKER)
    if marker is None:
        environment = os.environ.copy()
        environment[ENVIRONMENT_MARKER] = config.environment.name
        result = subprocess.run(
            environment_command(config, tuple(argv or sys.argv)),
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
        )
        raise SystemExit(result.returncode)
    if marker != config.environment.name:
        raise RuntimeError(
            f"Benchmark environment marker mismatch: expected {config.environment.name}, got {marker}"
        )
    if config.name == "dmpnn":
        return validate_dmpnn_environment(config)
    if config.name == "molformer":
        return validate_molformer_environment(config)
    if config.name == "ilbert":
        return validate_ilbert_environment(config)
    return validate_spmm_environment(config)


def _installed_versions() -> dict[str, str]:
    return {
        _canonical_name(distribution.metadata["Name"]): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }


def _validate_lock(
    config: BenchmarkConfig,
    *,
    expected_direct: dict[str, Any],
    direct: dict[str, Any],
) -> tuple[Path, Path, dict[str, str]]:
    assert config.environment is not None
    definition = repository_path(config.environment.definition)
    lock = repository_path(config.environment.lock)
    if not definition.is_file() or not lock.is_file():
        raise FileNotFoundError("Benchmark environment definition or dependency lock is missing")
    locked = _locked_versions(lock)
    installed = _installed_versions()
    mismatches = {
        name: {"expected": version, "installed": installed.get(name)}
        for name, version in locked.items()
        if installed.get(name) != version
    }
    direct_mismatches = {
        name: {"expected": expected, "installed": direct.get(name)}
        for name, expected in expected_direct.items()
        if direct.get(name) != expected
    }
    if mismatches or direct_mismatches:
        details = json.dumps(
            {"locked_packages": mismatches, "direct_runtime": direct_mismatches},
            sort_keys=True,
        )
        raise RuntimeError(
            f"{config.display_name} environment does not match its lock: {details}"
        )
    return definition, lock, installed


def _gpu_snapshot(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Advanced baseline requires CUDA; no silent CPU fallback")
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "model": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "cudnn": torch.backends.cudnn.version(),
        "driver_versions": sorted(
            {line.strip() for line in driver.stdout.splitlines() if line.strip()}
        ),
    }


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
    definition, lock, installed = _validate_lock(
        config, expected_direct=expected_direct, direct=direct
    )
    return {
        "environment_name": config.environment.name,
        "environment_definition": repository_relative(definition),
        "environment_lock": repository_relative(lock),
        "environment_lock_sha256": sha256_file(lock),
        "direct_versions": direct,
        "resolved_packages": dict(sorted(installed.items())),
        "gpu": _gpu_snapshot(torch),
    }


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
    definition, lock, installed = _validate_lock(
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
        "environment_name": config.environment.name,
        "environment_definition": repository_relative(definition),
        "environment_lock": repository_relative(lock),
        "environment_lock_sha256": sha256_file(lock),
        "direct_versions": direct,
        "resolved_packages": dict(sorted(installed.items())),
        "gpu": _gpu_snapshot(torch),
        "pretrained_snapshot": {
            "repository": repository,
            "revision": revision,
            "files": files,
        },
    }


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
        "pytorch": "2.2.1+cu121",
        "cuda": "12.1",
        "rdkit": "2023.09.5",
    }
    definition, lock, installed = _validate_lock(
        config, expected_direct=expected_direct, direct=direct
    )
    return {
        "environment_name": config.environment.name,
        "environment_definition": repository_relative(definition),
        "environment_lock": repository_relative(lock),
        "environment_lock_sha256": sha256_file(lock),
        "direct_versions": direct,
        "resolved_packages": dict(sorted(installed.items())),
        "gpu": _gpu_snapshot(torch),
        "pretrained_snapshot": ilbert_asset_snapshot(config),
    }


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
    tokenizer = _spmm_tokenizer(vocab)
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
        "tokenizer": {"vocab_size": 300, "special_token_ids": special_ids},
        "checkpoint_trust": "pinned_official_lightning_pickle",
    }


def _spmm_tokenizer(vocab: Path) -> Any:
    from transformers import BertTokenizer, WordpieceTokenizer

    tokenizer = BertTokenizer(
        vocab_file=str(vocab), do_lower_case=False, do_basic_tokenize=False
    )
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(
        vocab=tokenizer.vocab,
        unk_token=tokenizer.unk_token,
        max_input_chars_per_word=250,
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
        "pytorch": "1.13.1+cu117",
        "cuda": "11.7",
        "rdkit": "2023.03.1",
    }
    definition, lock, installed = _validate_lock(
        config, expected_direct=expected_direct, direct=direct
    )
    return {
        "environment_name": config.environment.name,
        "environment_definition": repository_relative(definition),
        "environment_lock": repository_relative(lock),
        "environment_lock_sha256": sha256_file(lock),
        "direct_versions": direct,
        "resolved_packages": dict(sorted(installed.items())),
        "gpu": _gpu_snapshot(torch),
        "pretrained_snapshot": spmm_asset_snapshot(config),
    }


def environment_run_details(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if snapshot is None:
        return {}
    details = {
        "benchmark_environment": snapshot["environment_name"],
        "environment_lock_sha256": snapshot["environment_lock_sha256"],
    }
    direct = snapshot["direct_versions"]
    if "chemprop" in direct:
        details["chemprop_version"] = direct["chemprop"]
    if "transformers" in direct and snapshot["environment_name"] == "ilume-molformer":
        details["transformers_version"] = direct["transformers"]
        details["hf_revision"] = snapshot["pretrained_snapshot"]["revision"]
    if snapshot["environment_name"] == "ilume-ilbert":
        details["transformers_version"] = direct["transformers"]
        details["upstream_revision"] = snapshot["pretrained_snapshot"]["revision"]
    if snapshot["environment_name"] == "ilume-spmm":
        details["transformers_version"] = direct["transformers"]
        details["upstream_revision"] = snapshot["pretrained_snapshot"]["revision"]
    return details


def write_environment_snapshot(path: str | Path, snapshot: dict[str, Any]) -> None:
    atomic_json(path, snapshot)


__all__ = [
    "ENVIRONMENT_MARKER",
    "ensure_benchmark_environment",
    "environment_run_details",
    "environment_command",
    "ilbert_asset_snapshot",
    "spmm_asset_snapshot",
    "validate_dmpnn_environment",
    "validate_ilbert_environment",
    "validate_molformer_environment",
    "validate_spmm_environment",
    "write_environment_snapshot",
]
