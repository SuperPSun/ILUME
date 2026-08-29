from __future__ import annotations

import importlib.metadata
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
    if config.name not in {"dmpnn", "molformer"} or config.environment is None:
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
    if config.name not in {"dmpnn", "molformer"}:
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
    return (
        validate_dmpnn_environment(config)
        if config.name == "dmpnn"
        else validate_molformer_environment(config)
    )


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
    if "transformers" in direct:
        details["transformers_version"] = direct["transformers"]
        details["hf_revision"] = snapshot["pretrained_snapshot"]["revision"]
    return details


def write_environment_snapshot(path: str | Path, snapshot: dict[str, Any]) -> None:
    atomic_json(path, snapshot)


__all__ = [
    "ENVIRONMENT_MARKER",
    "ensure_benchmark_environment",
    "environment_run_details",
    "environment_command",
    "validate_dmpnn_environment",
    "validate_molformer_environment",
    "write_environment_snapshot",
]
