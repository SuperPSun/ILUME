from __future__ import annotations

import csv
from dataclasses import replace

import torch

from ilume_pretrain.cli.train import run_training
from ilume_pretrain.config import (
    DataConfig,
    DescriptorConfig,
    FingerprintConfig,
    ModelConfig,
    PretrainConfig,
    TrainingConfig,
)
from ilume_pretrain.data import prepare_corpus


def _write_smiles(path, values):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["SMILES"])
        writer.writeheader()
        for value in values:
            writer.writerow({"SMILES": value})


def test_single_device_trainer_validates_checkpoints_and_resumes(tmp_path):
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
        training=TrainingConfig(
            batch_size=10,
            max_steps=2,
            gradient_accumulation_steps=1,
            learning_rate=1.0e-3,
            num_workers=0,
            device="cpu",
            amp_dtype="none",
            validation_interval=1,
            validation_batches=1,
            checkpoint_interval=1,
            keep_last_checkpoints=2,
            output_dir=output,
        ),
    )
    prepare_corpus(config)
    metrics = run_training(config)
    assert len(metrics) == 2
    assert all("loss_fingerprint" in row for row in metrics)
    for role in ("cation", "anion", "molecule"):
        assert f"valid_{role}_loss" in metrics[-1]

    checkpoint = output / "checkpoint_step_00000002.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 2
    assert payload["sampler"]["start_offset"] == 20
    assert payload["config"]["sampling"]["role_probabilities"] == [0.45, 0.45, 0.10]
    assert payload["source_hashes"]

    resumed = replace(
        config,
        training=replace(config.training, resume_from=checkpoint),
    )
    assert run_training(resumed) == []

    resumed_from_step_one = replace(
        config,
        training=replace(
            config.training,
            resume_from=output / "checkpoint_step_00000001.pt",
        ),
    )
    assert run_training(resumed_from_step_one) == metrics[1:]
