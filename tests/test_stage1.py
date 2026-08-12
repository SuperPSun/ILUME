from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pytest

from stage1.config import (
    _config_from_checkpoint_dict,
    config_from_dict,
    load_config,
)
from stage1.sampler import coverage_epoch_plan
import common.outputs as outputs_module
from common.data_identity import write_data_identity
from common.outputs import open_run_directory


ROOT = Path(__file__).resolve().parents[1]


def test_formal_stage1_profiles_preserve_capacity_and_coverage() -> None:
    configs = {
        name: load_config(ROOT / f"configs/formal/stage1/{name}.yaml")
        for name in ("base", "large", "xlarge")
    }
    base, large, xlarge = (configs[name] for name in ("base", "large", "xlarge"))
    assert (base.model.d_model, large.model.d_model, xlarge.model.d_model) == (384, 512, 640)
    assert (base.training.batch_size, large.training.batch_size, xlarge.training.batch_size) == (256, 256, 128)
    assert (base.training.gradient_accumulation_steps, large.training.gradient_accumulation_steps, xlarge.training.gradient_accumulation_steps) == (1, 1, 2)
    assert all(config.training.epochs == 5 for config in configs.values())
    assert all(config.training.learning_rate == pytest.approx(1.0e-4) for config in configs.values())
    assert len({config.data.artifacts_dir for config in configs.values()}) == 1
    assert all(config.training.save_every_n_epochs == 1 for config in configs.values())

    plan = coverage_epoch_plan((24908, 27907, 56532), 256, 1)
    assert (plan.steps_per_epoch, plan.draws_per_epoch) == (2209, 565504)
    assert plan.role_quotas == (254477, 254477, 56550)
    assert plan == coverage_epoch_plan((24908, 27907, 56532), 128, 2)


def test_stage1_config_round_trip_and_legacy_checkpoint_fields() -> None:
    config = load_config(ROOT / "configs/formal/stage1/base.yaml")
    assert config_from_dict(config.to_dict()) == config
    legacy = config.to_dict()
    legacy["smoke"] = {"steps": 2}
    legacy["training"].pop("save_every_n_epochs")
    legacy["training"].update(
        checkpoint_interval_epochs=1,
        keep_last_checkpoints=3,
        output_dir="artifacts/old",
        resume_from=None,
    )
    with pytest.raises(ValueError, match="Unknown config sections: smoke"):
        config_from_dict(legacy)
    assert _config_from_checkpoint_dict(legacy) == config


def test_run_directory_is_non_overwriting_self_describing_and_private(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(outputs_module, "REPOSITORY_ROOT", tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("data:\n  seed: 42\n", encoding="utf-8")
    payload = {"data": {"stage1_dir": "data/stage1", "seed": 42}}
    run = open_run_directory(
        stage="stage1", operation="train", config_path="config.yaml",
        config_payload=payload, output="outputs/run", seed=42,
    )
    run.complete({"loss": 1.0})
    assert (run.root / "run_config.yaml").is_file()
    metadata = json.loads((run.root / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["config_path"] == "config.yaml"
    assert not any(str(tmp_path) in str(value) for value in metadata.values())
    with pytest.raises(FileExistsError):
        open_run_directory(
            stage="stage1", operation="train", config_path="config.yaml",
            config_payload=payload, output="outputs/run", seed=42,
        )


def test_data_identity_records_relative_hash_size_and_rows(tmp_path) -> None:
    source = tmp_path / "data" / "stage1" / "cation.csv"
    source.parent.mkdir(parents=True)
    source.write_text("SMILES\n[Na+]\nC[NH3+]\n", encoding="utf-8")
    identity = write_data_identity(tmp_path, "stage1", [source])
    record = identity["files"][0]
    assert record["path"] == "data/stage1/cation.csv"
    assert record["rows"] == 2
    assert record["size"] == source.stat().st_size
    assert record["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert identity["source_repository_commit"] is None

import csv

import pytest
import torch

from stage1.config import (
    DataConfig,
    DescriptorConfig,
    FingerprintConfig,
    ModelConfig,
    PretrainConfig,
    TrainingConfig,
)
from stage1.data import prepare_corpus
from stage1.train import run_training


def _write_smiles(path, values) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["SMILES"])
        writer.writeheader()
        writer.writerows({"SMILES": value} for value in values)


def test_stage1_checkpoint_cadence_last_and_exact_resume(tmp_path, capsys) -> None:
    source = tmp_path / "stage1"
    source.mkdir()
    _write_smiles(source / "cation.csv", ["[Na+]", "C[NH3+]"])
    _write_smiles(source / "anion.csv", ["[Cl-]", "C(=O)[O-]"])
    _write_smiles(source / "molecule.csv", ["O", "CCO"])
    artifacts = tmp_path / "prepared"
    output = tmp_path / "train"
    config = PretrainConfig(
        data=DataConfig(
            stage1_dir=source, artifacts_dir=artifacts, valid_fraction=0.5,
            max_smiles_tokens=64, shard_size=2,
        ),
        descriptor=DescriptorConfig(mode="clean", token_count=8),
        fingerprint=FingerprintConfig(kind="both"),
        model=ModelConfig(
            d_model=16, n_heads=4, smiles_layers=1, graph_depth=2,
            descriptor_hidden_dim=32, descriptor_blocks=1, fusion_layers=1,
            feedforward_dim=32, dropout=0.0,
        ),
        training=TrainingConfig(
            batch_size=10, epochs=2, gradient_accumulation_steps=1,
            learning_rate=1.0e-3, num_workers=0, device="cpu",
            amp_dtype="none", validation_interval_epochs=1,
            validation_batches=1, save_every_n_epochs=1,
        ),
    )
    prepare_corpus(config)
    capsys.readouterr()
    rows = run_training(config, output_dir=output)
    assert [row["global_step"] for row in rows] == [1, 2]
    assert (output / "checkpoint_epoch_00001.pt").is_file()
    assert (output / "checkpoint_epoch_00002.pt").is_file()
    assert (output / "last.pt").is_file()
    last = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert last["completed_epochs"] == 2
    assert last["sampler"]["start_offset"] == 10

    assert run_training(
        config, output_dir=output, resume_from=output / "last.pt"
    ) == []
    replay = run_training(
        config, output_dir=output,
        resume_from=output / "checkpoint_epoch_00001.pt",
    )
    assert replay == rows[1:]

    invalid = dict(last)
    invalid["format_version"] = 2
    legacy = output / "legacy.pt"
    torch.save(invalid, legacy)
    with pytest.raises(ValueError, match="format v2 is incompatible"):
        run_training(config, output_dir=output, resume_from=legacy)
