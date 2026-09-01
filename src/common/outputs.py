from __future__ import annotations

import platform
import json
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rdkit import rdBase

from .io import atomic_json, atomic_yaml
from .identity import (
    IDENTITY_CONTRACT_VERSION,
    require_compatible_identity,
    validate_semantic_identity,
)


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
            _validate_public_paths(item, str(name))
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
    matmul_precision = None
    cudnn_precision = None
    if cuda_available:
        matmul = torch.backends.cuda.matmul
        matmul_precision = getattr(matmul, "fp32_precision", None)
        if matmul_precision is None:
            matmul_precision = "tf32" if matmul.allow_tf32 else "ieee"
        cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
        cudnn_precision = getattr(cudnn_conv, "fp32_precision", None)
        if cudnn_precision is None:
            cudnn_precision = "tf32" if torch.backends.cudnn.allow_tf32 else "ieee"
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
        "float32_matmul_precision": matmul_precision,
        "cuda_matmul_fp32_precision": matmul_precision,
        "cudnn_conv_fp32_precision": cudnn_precision,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


@dataclass
class RunDirectory:
    root: Path
    metadata: dict[str, Any]

    def _append_attempt(self, event: str, **details: Any) -> None:
        row = {
            "event": event,
            "attempt_id": self.metadata["attempt_id"],
            **details,
        }
        with (self.root / "attempts.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    def complete(self, summary: Any) -> None:
        atomic_json(self.root / "summary.json", summary)
        self.metadata["status"] = "completed"
        atomic_json(self.root / "metadata.json", self.metadata)
        self._append_attempt("completed")

    def fail(self) -> None:
        self.metadata["status"] = "failed"
        atomic_json(self.root / "metadata.json", self.metadata)
        self._append_attempt("failed")


def open_run_directory(
    *,
    stage: str,
    operation: str,
    config_path: str | Path,
    config_payload: dict[str, Any],
    semantic_identity: dict[str, Any],
    output: str | Path,
    seed: int,
    reusable: bool = False,
    resume: str | Path | None = None,
    details: dict[str, Any] | None = None,
    data_metadata: str | list[str] | None = None,
) -> RunDirectory:
    _validate_public_paths(config_payload)
    validate_semantic_identity(semantic_identity)
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
        metadata_path = root / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError("Existing run output has no metadata.json")
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing_identity = previous.get("semantic_identity")
        if not isinstance(existing_identity, dict):
            raise ValueError(
                "Existing run predates identity contract v1; regenerate the run"
            )
        require_compatible_identity(
            semantic_identity,
            existing_identity,
            context=f"Existing {stage}/{operation} run",
        )
        atomic_yaml(snapshot, config_payload)
    else:
        atomic_yaml(snapshot, config_payload)
    previous_attempts = 0
    previous_metadata = root / "metadata.json"
    if previous_metadata.is_file():
        previous_attempts = int(
            json.loads(previous_metadata.read_text(encoding="utf-8")).get(
                "attempt_count", 0
            )
        )
    runtime = _runtime_metadata()
    attempt_id = uuid.uuid4().hex
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "stage": stage,
        "operation": operation,
        "status": "running",
        "locator": {
            "config_path": repository_relative(config_path),
            "output": repository_relative(root),
        },
        "data_metadata": data_metadata if data_metadata is not None else f"data/{stage}/metadata.json",
        "seed": seed,
        "attempt_id": attempt_id,
        "attempt_count": previous_attempts + 1,
        "semantic_identity": semantic_identity,
        "provenance": runtime,
    }
    if resume is not None:
        metadata["locator"]["resume"] = repository_relative(resume)
    if details:
        metadata["provenance"].update(details)
    atomic_json(root / "metadata.json", metadata)
    run = RunDirectory(root=root, metadata=metadata)
    run._append_attempt(
        "started",
        semantic_identity=semantic_identity,
        locator=metadata["locator"],
        provenance=metadata["provenance"],
    )
    return run
