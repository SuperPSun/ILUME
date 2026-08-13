from __future__ import annotations

import platform
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rdkit import rdBase

from .io import atomic_json, atomic_yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def repository_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Path must be repository-relative: {path}")
    resolved = (REPOSITORY_ROOT / path).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError(f"Path escapes the repository: {path}") from error
    return resolved


def repository_relative(value: str | Path) -> str:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"Public run identity cannot contain an external path: {path}") from error


def _validate_public_paths(value: Any, key: str = "") -> None:
    if isinstance(value, dict):
        for name, item in value.items():
            _validate_public_paths(item, name)
    elif isinstance(value, list):
        for item in value:
            _validate_public_paths(item, key)
    elif isinstance(value, str) and any(
        marker in key for marker in ("path", "dir", "checkpoint", "resume", "output")
    ):
        if Path(value).is_absolute():
            raise ValueError(f"Config path must be repository-relative: {key}={value}")


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, dirty


def _runtime_metadata() -> dict[str, Any]:
    commit, dirty = _git_state()
    cuda_available = torch.cuda.is_available()
    return {
        "repository_commit": commit,
        "repository_dirty": dirty,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "rdkit_version": rdBase.rdkitVersion,
        "gpu_model": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_capability": (
            list(torch.cuda.get_device_capability(0)) if cuda_available else None
        ),
        "cudnn_version": torch.backends.cudnn.version(),
        "float32_matmul_precision": (
            torch.backends.cuda.matmul.fp32_precision if cuda_available else None
        ),
        "cuda_matmul_fp32_precision": (
            torch.backends.cuda.matmul.fp32_precision if cuda_available else None
        ),
        "cudnn_conv_fp32_precision": (
            torch.backends.cudnn.conv.fp32_precision if cuda_available else None
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _without_fields(payload: dict[str, Any], fields: set[str]) -> dict[str, Any]:
    import copy

    result = copy.deepcopy(payload)
    for field in fields:
        parts = field.split(".")
        target: Any = result
        for part in parts[:-1]:
            if not isinstance(target, dict) or part not in target:
                break
            target = target[part]
        else:
            if isinstance(target, dict):
                target.pop(parts[-1], None)
    return result


@dataclass
class RunDirectory:
    root: Path
    metadata: dict[str, Any]

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    def complete(self, summary: Any) -> None:
        atomic_json(self.root / "summary.json", summary)
        self.metadata["status"] = "completed"
        atomic_json(self.root / "metadata.json", self.metadata)

    def fail(self) -> None:
        self.metadata["status"] = "failed"
        atomic_json(self.root / "metadata.json", self.metadata)


def open_run_directory(
    *,
    stage: str,
    operation: str,
    config_path: str | Path,
    config_payload: dict[str, Any],
    output: str | Path,
    seed: int,
    reusable: bool = False,
    resume: str | Path | None = None,
    details: dict[str, Any] | None = None,
    ignored_config_sections: set[str] | None = None,
    ignored_config_fields: set[str] | None = None,
) -> RunDirectory:
    _validate_public_paths(config_payload)
    root = repository_path(output)
    snapshot = root / "run_config.yaml"
    if resume is not None:
        if not root.is_dir() or not snapshot.is_file() or not (root / "metadata.json").is_file():
            raise FileNotFoundError("Resume output must contain run_config.yaml and metadata.json")
    elif root.exists() and not reusable:
        raise FileExistsError(f"Output already exists: {repository_relative(root)}")
    elif root.exists() and reusable and not snapshot.is_file():
        raise FileExistsError(f"Existing prepare output has no run_config.yaml: {repository_relative(root)}")
    root.mkdir(parents=True, exist_ok=True)
    if snapshot.is_file():
        import yaml

        existing = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError("Existing run_config.yaml must contain a mapping")
        ignored = ignored_config_sections or set()
        existing_identity = _without_fields({
            key: value for key, value in existing.items() if key not in ignored
        }, ignored_config_fields or set())
        requested_identity = _without_fields({
            key: value for key, value in config_payload.items() if key not in ignored
        }, ignored_config_fields or set())
        if existing_identity != requested_identity:
            raise ValueError("Existing run_config.yaml does not match the effective config")
        if reusable and existing != config_payload:
            atomic_yaml(snapshot, config_payload)
    else:
        atomic_yaml(snapshot, config_payload)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "stage": stage,
        "operation": operation,
        "status": "running",
        "config_path": repository_relative(config_path),
        "data_metadata": f"data/{stage}/metadata.json",
        "seed": seed,
        "attempt_id": uuid.uuid4().hex,
        **_runtime_metadata(),
    }
    if resume is not None:
        metadata["resume"] = repository_relative(resume)
    if details:
        metadata.update(details)
    atomic_json(root / "metadata.json", metadata)
    return RunDirectory(root=root, metadata=metadata)
