from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from common.identity import (
    compare_semantic_identity,
    semantic_hash,
    semantic_identity,
    tensor_state_hash,
    verify_integrity,
)
from common.io import sha256_file


def test_semantic_hash_is_domain_separated_and_canonical() -> None:
    left = {"b": [2, 3], "a": {"value": 1}}
    right = {"a": {"value": 1}, "b": [2, 3]}
    assert semantic_hash("example.one", left) == semantic_hash("example.one", right)
    assert semantic_hash("example.one", left) != semantic_hash("example.two", left)
    with pytest.raises(ValueError, match="must be finite"):
        semantic_hash("example.one", {"bad": float("nan")})
    with pytest.raises(TypeError, match="Path"):
        semantic_hash("example.one", {"path": Path("data.csv")})


def test_tensor_state_hash_binds_names_dtype_shape_and_bytes() -> None:
    state = {
        "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "empty": torch.empty((0, 4), dtype=torch.int64),
    }
    digest = tensor_state_hash("model.state", state)
    clone = {name: value.clone() for name, value in state.items()}
    assert tensor_state_hash("model.state", clone) == digest
    clone["weight"][0, 0] = 1
    assert tensor_state_hash("model.state", clone) != digest


def test_semantic_comparison_validates_self_hash_and_reports_field() -> None:
    expected = semantic_identity("example.identity", {"model": {"width": 16}})
    actual = semantic_identity("example.identity", {"model": {"width": 32}})
    differences = compare_semantic_identity(expected, actual)
    assert differences == ("payload.model.width: 32 != 16",)
    tampered = json.loads(json.dumps(expected))
    tampered["payload"]["model"]["width"] = 32
    with pytest.raises(ValueError, match="self-hash mismatch"):
        compare_semantic_identity(tampered, actual)


def test_verify_integrity_checks_safe_locator_size_and_sha(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"identity-contract-v1")
    locator = {"payload": "payload.bin"}
    manifest = {
        "payload": {"size": payload.stat().st_size, "sha256": sha256_file(payload)}
    }
    assert verify_integrity(tmp_path, locator, manifest) is None
    payload.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size mismatch|SHA256 mismatch"):
        verify_integrity(tmp_path, locator, manifest)
    with pytest.raises(ValueError, match="escapes root"):
        verify_integrity(tmp_path, {"payload": "../payload.bin"}, manifest)
    with pytest.raises(ValueError, match="logical IDs"):
        verify_integrity(tmp_path, {"other": "payload.bin"}, manifest)
