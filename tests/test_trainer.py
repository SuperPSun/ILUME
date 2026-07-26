from __future__ import annotations

import csv
import json
from dataclasses import replace

import pytest
import torch

from ilume_pretrain.cli import train as train_module
from ilume_pretrain.cli.smoke import run_smoke
from ilume_pretrain.cli.train import run_training
from ilume_pretrain.config import (
    DataConfig,
    DescriptorConfig,
    FingerprintConfig,
    ModelConfig,
    PretrainConfig,
    SmokeConfig,
    TrainingConfig,
)
from ilume_pretrain.data import prepare_corpus


def _write_smiles(path, values):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["SMILES"])
        writer.writeheader()
        for value in values:
            writer.writerow({"SMILES": value})


def test_single_device_trainer_validates_checkpoints_and_resumes(
    tmp_path, capsys, monkeypatch
):
    stage1 = tmp_path / "stage1"
    artifacts = tmp_path / "corpus"
    output = tmp_path / "training"
    stage1.mkdir()
    _write_smiles(stage1 / "cation.csv", ["[Na+]", "C[NH3+]"])
    _write_smiles(stage1 / "anion.csv", ["[Cl-]", "C(=O)[O-]"])
    _write_smiles(stage1 / "molecule.csv", ["O", "CCO"])

    config = PretrainConfig(
        data=DataConfig(
            stage1_dir=stage1,
            artifacts_dir=artifacts,
            valid_fraction=0.5,
            max_smiles_tokens=64,
            shard_size=2,
        ),
        descriptor=DescriptorConfig(mode="clean", token_count=8),
        fingerprint=FingerprintConfig(kind="both"),
        model=ModelConfig(
            d_model=16,
            n_heads=4,
            smiles_layers=1,
            graph_depth=2,
            descriptor_hidden_dim=32,
            descriptor_blocks=1,
            fusion_layers=1,
            feedforward_dim=32,
            dropout=0.0,
        ),
        smoke=SmokeConfig(
            batch_size=10,
            steps=2,
            device="cpu",
            amp=False,
        ),
        training=TrainingConfig(
            batch_size=10,
            epochs=2,
            gradient_accumulation_steps=1,
            learning_rate=1.0e-3,
            num_workers=0,
            device="cpu",
            amp_dtype="none",
            validation_interval_epochs=1,
            validation_batches=1,
            checkpoint_interval_epochs=1,
            keep_last_checkpoints=2,
            output_dir=output,
        ),
    )
    prepare_corpus(config)
    capsys.readouterr()
    smoke_metrics = run_smoke(config)
    smoke_output = [
        json.loads(line) for line in capsys.readouterr().out.splitlines()
    ]
    assert [row["step"] for row in smoke_output] == [1, 2]
    assert smoke_metrics == smoke_output

    metrics = run_training(config)
    output_rows = [
        json.loads(line) for line in capsys.readouterr().out.splitlines()
    ]
    assert [row["epoch"] for row in output_rows] == [1, 2]
    assert [row["epoch_step"] for row in output_rows] == [1, 1]
    assert [row["global_step"] for row in output_rows] == [1, 2]
    assert len(metrics) == 2
    assert all("loss_fingerprint" in row for row in metrics)
    for role in ("cation", "anion", "molecule"):
        assert f"valid_{role}_loss" in metrics[-1]

    checkpoint = output / "checkpoint_epoch_00002.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 3
    assert payload["completed_epochs"] == 2
    assert payload["global_step"] == 2
    assert payload["steps_per_epoch"] == 1
    assert payload["draws_per_epoch"] == 10
    assert payload["sampler"]["start_offset"] == 10
    assert payload["config"]["sampling"]["role_probabilities"] == [0.45, 0.45, 0.10]
    assert payload["source_hashes"]

    resumed = replace(
        config,
        training=replace(config.training, resume_from=checkpoint),
    )
    assert run_training(resumed) == []

    resumed_from_epoch_one = replace(
        config,
        training=replace(
            config.training,
            resume_from=output / "checkpoint_epoch_00001.pt",
        ),
    )

    class RecordingBar:
        def __init__(self, total, initial):
            self.total = total
            self.initial = initial
            self.updates = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def update(self, count):
            self.updates += count

        def set_description_str(self, value):
            pass

        def set_postfix(self, values, refresh=False):
            pass

    class RecordingReporter:
        def __init__(self):
            self.progress = None

        def bar(self, *, total, initial=0, **kwargs):
            self.progress = RecordingBar(total, initial)
            return self.progress

        def emit_json(self, payload):
            pass

    reporter = RecordingReporter()
    monkeypatch.setattr(train_module, "ProgressReporter", lambda: reporter)
    assert run_training(resumed_from_epoch_one) == metrics[1:]
    assert reporter.progress.total == 1
    assert reporter.progress.initial == 0
    assert reporter.progress.updates == 1

    legacy_checkpoint = output / "checkpoint_step_legacy.pt"
    legacy_payload = dict(payload)
    legacy_payload["format_version"] = 2
    torch.save(legacy_payload, legacy_checkpoint)
    legacy_resume = replace(
        config,
        training=replace(config.training, resume_from=legacy_checkpoint),
    )
    with pytest.raises(ValueError, match="format v2 is incompatible"):
        run_training(legacy_resume)
