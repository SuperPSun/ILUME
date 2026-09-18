"""Publication shared by baselines with model/history/input-audit artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from common.io import atomic_json, atomic_torch_save, sha256_file


def write_model_artifacts(
    root: Path,
    state: Mapping[str, torch.Tensor],
    state_hash: str,
    history: Sequence[Mapping[str, Any]],
    input_audit: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    model_path = root / "model.pt"
    history_path = root / "history.json"
    audit_path = root / "input_audit.json"
    atomic_torch_save(model_path, {"state_dict": state, "state_hash": state_hash})
    atomic_json(history_path, history)
    atomic_json(audit_path, input_audit)
    return {
        path.name: {"sha256": sha256_file(path), "size": path.stat().st_size}
        for path in (model_path, history_path, audit_path)
    }
