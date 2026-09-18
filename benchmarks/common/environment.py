from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from common.io import atomic_json, sha256_file
from common.outputs import REPOSITORY_ROOT, repository_path, repository_relative

from .config import BenchmarkConfig
from .registry import ADAPTERS


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
    adapter = ADAPTERS.get(config.name)
    if adapter is None or not adapter.isolated_environment or config.environment is None:
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
    adapter = ADAPTERS.get(config.name)
    if adapter is None or not adapter.isolated_environment:
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
    return adapter.load("environment")(config)


def _installed_versions() -> dict[str, str]:
    return {
        _canonical_name(distribution.metadata["Name"]): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }


def validate_lock(
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


def environment_snapshot(
    config: BenchmarkConfig,
    definition: Path,
    lock: Path,
    installed: dict[str, str],
    direct: dict[str, Any],
    torch: Any,
) -> dict[str, Any]:
    assert config.environment is not None
    return {
        "environment_name": config.environment.name,
        "environment_definition": repository_relative(definition),
        "environment_lock": repository_relative(lock),
        "environment_lock_sha256": sha256_file(lock),
        "direct_versions": direct,
        "resolved_packages": dict(sorted(installed.items())),
        "gpu": _gpu_snapshot(torch),
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
    if snapshot["environment_name"] == "ilume-llasmol":
        details["transformers_version"] = direct["transformers"]
        details["peft_version"] = direct["peft"]
        details["bitsandbytes_version"] = direct["bitsandbytes"]
        details["base_revision"] = snapshot["pretrained_snapshot"]["base"]["revision"]
        details["adapter_revision"] = snapshot["pretrained_snapshot"]["adapter"]["revision"]
    if snapshot["environment_name"] == "ilume-aionopedia":
        details["transformers_version"] = direct["transformers"]
        details["peft_version"] = direct["peft"]
        details["base_revision"] = snapshot["pretrained_snapshot"]["base_revision"]
        details["pretrained_revision"] = snapshot["pretrained_snapshot"]["pretrained_revision"]
    if snapshot["environment_name"] == "ilume-iltransr":
        details["upstream_revision"] = snapshot["pretrained_snapshot"]["revision"]
        details["conversion_tensor_state_sha256"] = snapshot["pretrained_snapshot"]["structure"]["tensor_state_sha256"]
    if snapshot["environment_name"] == "ilume-aifc":
        details["upstream_revision"] = snapshot["pretrained_snapshot"]["revision"]
        details["fragment_scheme_sha256"] = snapshot["pretrained_snapshot"]["fragment_scheme"]["sha256"]
    return details


def write_environment_snapshot(path: str | Path, snapshot: dict[str, Any]) -> None:
    atomic_json(path, snapshot)


__all__ = [
    "ENVIRONMENT_MARKER",
    "ensure_benchmark_environment",
    "environment_command",
    "environment_run_details",
    "environment_snapshot",
    "validate_lock",
    "write_environment_snapshot",
]
