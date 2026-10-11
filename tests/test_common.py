from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from common.identity import semantic_hash, tensor_state_hash, verify_integrity
from common.data_identity import write_data_identity
from common.io import sha256_file
from common.refinement import (
    refinement_cosine_factor,
    refinement_geometry,
    selection_record,
)
from common.training import capture_rng_state, restore_rng_state, seed_everything

def test_identity_hashes_and_integrity_manifest_are_stable(tmp_path: Path) -> None:
    left = {"b": [2, 3], "a": {"value": 1}}
    right = {"a": {"value": 1}, "b": [2, 3]}
    assert semantic_hash("example.one", left) == semantic_hash("example.one", right)
    assert semantic_hash("example.one", left) != semantic_hash("example.two", left)

    state = {
        "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "empty": torch.empty((0, 4), dtype=torch.int64),
    }
    digest = tensor_state_hash("model.state", state)
    clone = {name: value.clone() for name, value in state.items()}
    assert tensor_state_hash("model.state", clone) == digest
    clone["weight"][0, 0] = 1
    assert tensor_state_hash("model.state", clone) != digest

    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"identity-contract-v1")
    locator = {"payload": "payload.bin"}
    manifest = {
        "payload": {"size": payload.stat().st_size, "sha256": sha256_file(payload)}
    }
    assert verify_integrity(tmp_path, locator, manifest) is None

@pytest.mark.parametrize("stage", ["stage2", "stage3"])
def test_data_identity_isolated_by_split_root(tmp_path: Path, stage: str) -> None:
    identities = []
    for split_seed in (42, 123):
        data_dir = tmp_path / "data" / str(split_seed) / stage
        data_dir.mkdir(parents=True)
        source = data_dir / "train.csv"
        source.write_text("SMILES,value\nC,1\n")
        identities.append(write_data_identity(tmp_path, stage, [source], data_dir=data_dir))
        assert (data_dir / "metadata.json").is_file()
        assert identities[-1]["locator"]["files"]["source_00000"] == f"data/{split_seed}/{stage}/train.csv"
    first_metadata = tmp_path / "data" / "42" / stage / "metadata.json"
    original = first_metadata.read_bytes()
    source.write_text("SMILES,value\nC,2\n")
    changed = write_data_identity(tmp_path, stage, [source], data_dir=data_dir)
    assert first_metadata.read_bytes() == original
    assert identities[0]["semantic"] == identities[1]["semantic"]
    assert changed["semantic"] != identities[1]["semantic"]
    assert not (tmp_path / "data" / stage).exists()
    write_data_identity(tmp_path, "stage1", [source])
    assert (tmp_path / "data" / "stage1" / "metadata.json").is_file()

def test_refinement_geometry_and_selection_contract() -> None:
    for epochs, expected in ((5, (4, 1)), (10, (8, 2)), (100, (80, 20))):
        assert refinement_geometry(epochs, 0.20) == expected
    assert refinement_cosine_factor(0, 10, 0.05) == pytest.approx(1.0)
    assert refinement_cosine_factor(5, 10, 0.05) == pytest.approx(0.525)
    assert refinement_cosine_factor(10, 10, 0.05) == pytest.approx(0.05)
    assert selection_record(
        metric_name="normalized_mae",
        boundary_epoch=8,
        boundary_metric=0.2,
        selected_epoch=8,
        best_metric=0.2,
    )["improved"] is False
    assert selection_record(
        metric_name="normalized_mae",
        boundary_epoch=8,
        boundary_metric=0.2,
        selected_epoch=9,
        best_metric=0.1,
    )["improved"] is True

def test_rng_state_round_trip_is_exact() -> None:
    seed_everything(17)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual == expected
