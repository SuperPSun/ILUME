from __future__ import annotations

from pathlib import Path

import torch

from stage1.features import ROLE_TO_ID
from common.identity import IDENTITY_CONTRACT_VERSION, tensor_state_hash
from stage2.identity import build_stage2_encoder_identity
from stage2.train import load_stage2_encoder_artifact


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    return tensor_state_hash("stage2.encoder-state", state)


def test_encoder_artifact_reloads_without_physics_heads(tmp_path: Path) -> None:
    stage1 = {"fusion.weight": torch.arange(4, dtype=torch.float32)}
    objects = {"encoder.weight": torch.ones(2, dtype=torch.float32)}
    feature_identity = {"hash": "feature-test"}
    encoding_contract = {"encoding_api": "encode-states-v1"}
    object_config = {"layers": 2}
    state_hashes = {
        "stage1_backbone": _state_hash(stage1),
        "object_encoder": _state_hash(objects),
    }
    payload = {
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "kind": "ilume_stage2_encoder",
        "format_version": 1,
        "stage1_backbone": stage1,
        "object_encoder": objects,
        "stage1_config": {"model": {"d_model": 2}},
        "stage1_feature_identity": feature_identity,
        "stage1_encoding_contract": encoding_contract,
        "feature_artifacts": {},
        "object_encoder_config": object_config,
        "role_to_id": dict(ROLE_TO_ID),
        "model_contract": {"d_model": 2},
        "state_hashes": state_hashes,
        "semantic_identity": build_stage2_encoder_identity(
            stage1_feature_identity=feature_identity,
            stage1_encoding_contract=encoding_contract,
            stage1_state_hash=state_hashes["stage1_backbone"],
            object_encoder_contract=object_config,
            object_encoder_state_hash=state_hashes["object_encoder"],
            role_to_id=ROLE_TO_ID,
        ),
        "provenance": {"registry_hash": "test"},
    }
    path = tmp_path / "stage2_encoder.pt"
    torch.save(payload, path)
    loaded = load_stage2_encoder_artifact(path)
    assert set(loaded["stage1_backbone"]) == {"fusion.weight"}
    assert "physics_heads" not in loaded
