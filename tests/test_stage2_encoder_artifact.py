from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from stage1.features import ROLE_TO_ID
from stage2.train import load_stage2_encoder_artifact


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def test_encoder_artifact_reloads_without_physics_heads(tmp_path: Path) -> None:
    stage1 = {"fusion.weight": torch.arange(4, dtype=torch.float32)}
    objects = {"encoder.weight": torch.ones(2, dtype=torch.float32)}
    payload = {
        "kind": "ilume_stage2_encoder",
        "format_version": 1,
        "stage1_backbone": stage1,
        "object_encoder": objects,
        "stage1_config": {"model": {"d_model": 2}},
        "object_encoder_config": {"layers": 2},
        "role_to_id": dict(ROLE_TO_ID),
        "model_contract": {"d_model": 2},
        "state_hashes": {
            "stage1_backbone": _state_hash(stage1),
            "object_encoder": _state_hash(objects),
        },
        "provenance": {"registry_hash": "test"},
    }
    path = tmp_path / "stage2_encoder.pt"
    torch.save(payload, path)
    loaded = load_stage2_encoder_artifact(path)
    assert set(loaded["stage1_backbone"]) == {"fusion.weight"}
    assert "physics_heads" not in loaded


def test_encoder_artifact_detects_state_tampering(tmp_path: Path) -> None:
    path = tmp_path / "tampered.pt"
    torch.save(
        {
            "kind": "ilume_stage2_encoder",
            "format_version": 1,
            "stage1_backbone": {"value": torch.ones(1)},
            "object_encoder": {"value": torch.ones(1)},
            "state_hashes": {
                "stage1_backbone": "bad",
                "object_encoder": "bad",
            },
        },
        path,
    )
    with pytest.raises(ValueError, match="Stage 1 state hash mismatch"):
        load_stage2_encoder_artifact(path)
