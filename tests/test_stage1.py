from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import hashlib
import json

import pytest

from stage1.config import (
    STAGE1_CHECKPOINT_KIND,
    STAGE1_CHECKPOINT_VERSION,
    config_from_dict,
    load_config,
)
import common.outputs as outputs_module
from common.data_identity import write_data_identity
from common.identity import semantic_identity
from common.outputs import open_run_directory


ROOT = Path(__file__).resolve().parents[1]


def test_formal_stage1_has_one_large_capacity_base_profile() -> None:
    assert sorted(path.name for path in (ROOT / "configs/v1/stage1").glob("*.yaml")) == [
        "base.yaml"
    ]
    base = load_config(ROOT / "configs/v1/stage1/base.yaml")
    assert (
        base.model.d_model,
        base.model.n_heads,
        base.model.smiles_layers,
        base.model.graph_depth,
        base.model.descriptor_hidden_dim,
        base.model.fusion_layers,
        base.model.feedforward_dim,
    ) == (512, 8, 8, 6, 1024, 8, 2048)
    assert base.model.role_embedding is True
    assert base.model.gradient_checkpointing is False
    assert base.data.include_augmentation is True
    assert base.loss.role_weights == (2.0, 2.0, 1.0)
    assert base.training.batch_size == 128
    assert base.training.num_workers == 8
    assert base.training.epochs == 5
    assert base.training.learning_rate == pytest.approx(1.0e-4)
    assert base.training.compile is False
    assert base.training.validation_interval_steps == 5000
    assert base.tokenizer.min_frequency == 1
    assert (
        base.preparation.workers,
        base.preparation.catalog_batch_size,
        base.preparation.qc_batch_size,
        base.preparation.tokenizer_batch_size,
        base.preparation.descriptor_batch_size,
    ) == (16, 10000, 2048, 2048, 512)
    assert "preparation" in base.to_dict()
    assert "preparation" not in base.experiment_dict()


def test_stage1_config_round_trip() -> None:
    config = load_config(ROOT / "configs/v1/stage1/base.yaml")
    assert config_from_dict(config.to_dict()) == config


def test_run_directory_is_self_describing_and_private(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(outputs_module, "REPOSITORY_ROOT", tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("data:\n  seed: 42\n", encoding="utf-8")
    payload = {"data": {"stage1_dir": "data/stage1", "seed": 42}}
    identity = semantic_identity("test.run", {"seed": 42})
    run = open_run_directory(
        stage="stage1", operation="train", config_path="config.yaml",
        config_payload=payload, semantic_identity=identity,
        output="outputs/run", seed=42,
    )
    run.complete({"loss": 1.0})
    assert (run.root / "run_config.yaml").is_file()
    metadata = json.loads((run.root / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["locator"]["config_path"] == "config.yaml"
    assert not any(str(tmp_path) in str(value) for value in metadata.values())
    attempts = [
        json.loads(line)
        for line in (run.root / "attempts.jsonl").read_text().splitlines()
    ]
    assert [row["event"] for row in attempts] == ["started", "completed"]
    assert attempts[0]["attempt_id"] == attempts[1]["attempt_id"]


def test_run_directory_reuses_after_execution_change(tmp_path, monkeypatch):
    monkeypatch.setattr(outputs_module, "REPOSITORY_ROOT", tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("preparation:\n  workers: 1\n", encoding="utf-8")
    original = {"tokenizer": {"min_frequency": 1}, "preparation": {"workers": 1}}
    identity = semantic_identity("test.prepare", {"min_frequency": 1})
    run = open_run_directory(
        stage="stage1",
        operation="prepare",
        config_path="config.yaml",
        config_payload=original,
        semantic_identity=identity,
        output="outputs/prepare",
        seed=42,
        reusable=True,
    )
    run.complete({"total": 1})
    changed_workers = {
        "tokenizer": {"min_frequency": 1},
        "preparation": {"workers": 4},
    }
    open_run_directory(
        stage="stage1",
        operation="prepare",
        config_path="config.yaml",
        config_payload=changed_workers,
        semantic_identity=identity,
        output="outputs/prepare",
        seed=42,
        reusable=True,
    )
    import yaml

    assert yaml.safe_load(
        (tmp_path / "outputs/prepare/run_config.yaml").read_text()
    ) == changed_workers


def test_run_directory_allows_execution_change_on_resume(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(outputs_module, "REPOSITORY_ROOT", tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("training:\n  compile: true\n", encoding="utf-8")
    original = {"training": {"epochs": 2, "compile": True}}
    identity = semantic_identity("test.train", {"epochs": 2})
    run = open_run_directory(
        stage="stage1",
        operation="train",
        config_path="config.yaml",
        config_payload=original,
        semantic_identity=identity,
        output="outputs/train",
        seed=42,
    )
    run.fail()
    resumed = open_run_directory(
        stage="stage1",
        operation="train",
        config_path="config.yaml",
        config_payload={"training": {"epochs": 2, "compile": False}},
        semantic_identity=identity,
        output="outputs/train",
        seed=42,
        resume="outputs/train/last.pt",
    )
    assert resumed.metadata["attempt_id"] != run.metadata["attempt_id"]
    assert resumed.metadata["locator"]["resume"] == "outputs/train/last.pt"


def test_data_identity_records_relative_hash_size_and_rows(tmp_path) -> None:
    source = tmp_path / "data" / "stage1" / "cation.csv"
    source.parent.mkdir(parents=True)
    source.write_text("SMILES\n[Na+]\nC[NH3+]\n", encoding="utf-8")
    identity = write_data_identity(tmp_path, "stage1", [source])
    logical_id = next(iter(identity["locator"]["files"]))
    assert identity["locator"]["files"][logical_id] == "data/stage1/cation.csv"
    source_payload = identity["semantic"]["identities"]["source"]["payload"]
    assert source_payload["sources"][logical_id]["rows"] == 2
    record = identity["integrity"]["files"][logical_id]
    assert record["size"] == source.stat().st_size
    assert record["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert identity["provenance"]["source_repository_commit"] is None

import csv

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from stage1.config import (
    DataConfig,
    DescriptorConfig,
    FingerprintConfig,
    ModelConfig,
    PreparationConfig,
    PretrainConfig,
    TrainingConfig,
)
from stage1.prepare import prepare_corpus
from stage1.model import LossStatistics, PretrainOutput
import stage1.train as train_module
from stage1.train import (
    _DistributedContext,
    _global_training_losses,
    run_training,
)


def _write_smiles(path, values) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["SMILES"])
        writer.writeheader()
        writer.writerows({"SMILES": value} for value in values)


def _ddp_training_worker(
    rank: int,
    world_size: int,
    init_path: str,
    config: PretrainConfig,
    output_dir: str,
    resume_from: str | None,
    stop_after_first_epoch: bool,
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        parameter = torch.tensor(1.0, requires_grad=True)
        local_numerator = parameter * (2.0 if rank == 0 else 9.0)
        statistics = LossStatistics(
            numerators=local_numerator.reshape(1),
            denominators=torch.tensor([2.0 if rank == 0 else 3.0]),
            role_numerators=torch.zeros((1, 3)),
            role_denominators=torch.zeros((1, 3)),
        )
        reduced_loss, _ = _global_training_losses(
            PretrainOutput(
                loss=local_numerator,
                losses={"smiles": local_numerator},
                loss_statistics={"smiles": statistics},
                logits={},
                fused_cls=torch.empty(0),
            ),
            _DistributedContext(rank, world_size, rank),
            config,
        )
        reduced_loss.backward()
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad /= world_size
        assert parameter.grad.item() == pytest.approx(11.0 / 5.0)

        if stop_after_first_epoch:
            real_save = train_module._save_checkpoint

            class StopAfterCheckpoint(RuntimeError):
                pass

            def save_then_stop(paths, **kwargs):
                real_save(paths, **kwargs)
                if kwargs["completed_epoch"] == 1:
                    raise StopAfterCheckpoint

            train_module._save_checkpoint = save_then_stop
            try:
                run_training(config, output_dir=output_dir)
            except StopAfterCheckpoint:
                pass
        else:
            run_training(
                config,
                output_dir=output_dir,
                resume_from=resume_from,
            )
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_checkpoint_ownership_and_cross_world_resume(tmp_path, capsys) -> None:
    source = tmp_path / "stage1"
    source.mkdir()
    _write_smiles(source / "cation.csv", ["[Na+]", "[K+]", "C[NH3+]", "C[NH2+]C"])
    _write_smiles(source / "anion.csv", ["[Cl-]", "[Br-]", "[I-]", "C(=O)[O-]"])
    _write_smiles(source / "molecule.csv", ["O", "N", "CC", "CCO"])
    artifacts = tmp_path / "prepared"
    output = tmp_path / "ddp_train"
    config = PretrainConfig(
        data=DataConfig(
            stage1_dir=source,
            artifacts_dir=artifacts,
            valid_fraction=0.5,
            max_smiles_tokens=64,
            shard_size=2,
        ),
        descriptor=DescriptorConfig(mode="clean", token_count=1),
        fingerprint=FingerprintConfig(kind="maccs"),
        model=ModelConfig(
            d_model=8,
            n_heads=2,
            smiles_layers=1,
            graph_depth=1,
            descriptor_hidden_dim=16,
            descriptor_blocks=1,
            fusion_layers=1,
            feedforward_dim=16,
            dropout=0.0,
        ),
        training=TrainingConfig(
            batch_size=2,
            epochs=1,
            learning_rate=1.0e-3,
            num_workers=0,
            device="cpu",
            amp_dtype="none",
            compile=False,
            validation_interval_steps=100,
            quick_validation_samples_per_role=1,
        ),
    )
    prepare_corpus(config)
    capsys.readouterr()

    world_size = 2
    mp.spawn(
        _ddp_training_worker,
        args=(
            world_size,
            str(tmp_path / "first_init"),
            config,
            str(output),
            None,
            True,
        ),
        nprocs=world_size,
        join=True,
    )
    mid = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert mid["world_size_at_save"] == 2
    assert mid["global_step"] == 3
    assert mid["completed_epoch"] == 1
    assert "rank_rng" not in mid
    assert "epoch_cursor" not in mid

    mp.spawn(
        _ddp_training_worker,
        args=(
            world_size,
            str(tmp_path / "resume_init"),
            config,
            str(output),
            str(output / "last.pt"),
            False,
        ),
        nprocs=world_size,
        join=True,
    )
    completed = torch.load(
        output / "last.pt", map_location="cpu", weights_only=False
    )
    assert completed["completed_epoch"] == 1
    assert sorted(path.name for path in output.glob("*.pt")) == [
        "checkpoint_epoch_00001.pt",
        "last.pt",
    ]
    metric_steps = [
        json.loads(line)["global_step"]
        for line in (output / "metrics.jsonl").read_text().splitlines()
        if json.loads(line).get("event") != "attempt_start"
    ]
    assert metric_steps == [3]

    assert run_training(
        config, output_dir=output, resume_from=output / "last.pt",
        attempt_id="single-rank-resume",
    ) == []


def test_stage1_epoch_checkpoint_resume_and_attempt_log_preservation(
    tmp_path, capsys, monkeypatch
) -> None:
    source = tmp_path / "stage1"
    source.mkdir()
    _write_smiles(source / "cation.csv", ["[Na+]", "[K+]", "C[NH3+]", "C[NH2+]C"])
    _write_smiles(source / "anion.csv", ["[Cl-]", "[Br-]", "[I-]", "C(=O)[O-]"])
    _write_smiles(source / "molecule.csv", ["O", "N", "CC", "CCO"])
    artifacts = tmp_path / "prepared"
    baseline_output = tmp_path / "baseline"
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
            feedforward_dim=32, dropout=0.1,
        ),
        training=TrainingConfig(
            batch_size=2, epochs=2,
            learning_rate=1.0e-3, num_workers=0, device="cpu",
            amp_dtype="none", compile=False,
            validation_interval_steps=2, quick_validation_samples_per_role=1,
        ),
    )
    prepare_corpus(config)
    capsys.readouterr()
    validation_calls: list[bool] = []
    real_validate = train_module._validate

    def record_validation(*args, **kwargs):
        validation_calls.append(bool(kwargs["quick"]))
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(train_module, "_validate", record_validation)
    baseline = run_training(config, output_dir=baseline_output, attempt_id="baseline")
    monkeypatch.setattr(train_module, "_validate", real_validate)
    assert [row["global_step"] for row in baseline] == [2, 3, 4, 6]
    assert validation_calls == [True, False, True, False]

    real_save = train_module._save_checkpoint

    class Interrupted(RuntimeError):
        pass

    def interrupt_after_epoch(paths, **kwargs):
        real_save(paths, **kwargs)
        if kwargs["completed_epoch"] == 1:
            raise Interrupted

    monkeypatch.setattr(train_module, "_save_checkpoint", interrupt_after_epoch)
    try:
        run_training(config, output_dir=output, attempt_id="attempt-1")
    except Interrupted:
        pass
    monkeypatch.setattr(train_module, "_save_checkpoint", real_save)

    mid = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert mid["kind"] == STAGE1_CHECKPOINT_KIND
    assert mid["format_version"] == STAGE1_CHECKPOINT_VERSION
    assert mid["completed_epoch"] == 1
    assert mid["global_step"] == 3
    assert set(mid).isdisjoint({"epoch_index", "epoch_cursor", "rank_rng", "micro_step"})
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"global_step": 999, "loss": 0}\n')

    rows = run_training(
        config, output_dir=output, resume_from=output / "last.pt",
        attempt_id="attempt-2",
    )
    assert [row["global_step"] for row in rows] == [4, 6]
    assert [row["loss"] for row in rows] == pytest.approx(
        [row["loss"] for row in baseline[2:]]
    )
    assert (output / "checkpoint_epoch_00001.pt").is_file()
    assert (output / "checkpoint_epoch_00002.pt").is_file()
    assert (output / "last.pt").is_file()
    last = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert last["completed_epoch"] == 2
    metric_rows = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    attempt_rows = [row for row in metric_rows if row.get("event") == "attempt_start"]
    assert attempt_rows == [
        {
            "event": "attempt_start",
            "attempt_id": "attempt-2",
            "resumed_from_attempt_id": "attempt-1",
            "completed_epoch": 1,
            "global_step": 3,
            "world_size": 1,
            "compile": False,
        }
    ]
    training_rows = [row for row in metric_rows if row.get("event") != "attempt_start"]
    assert [row["global_step"] for row in training_rows] == [2, 3, 999, 4, 6]
    assert [row.get("attempt_id") for row in training_rows] == [
        "attempt-1", "attempt-1", None, "attempt-2", "attempt-2"
    ]

    assert run_training(
        config, output_dir=output, resume_from=output / "last.pt"
    ) == []
    changed_preparation = replace(
        config, preparation=PreparationConfig(workers=4)
    )
    assert run_training(
        changed_preparation, output_dir=output, resume_from=output / "last.pt"
    ) == []
    changed_compile = replace(config, training=replace(config.training, compile=True))
    assert run_training(
        changed_compile, output_dir=output, resume_from=output / "last.pt"
    ) == []
    changed_tokenizer = replace(
        config, tokenizer=replace(config.tokenizer, min_frequency=2)
    )
    assert run_training(
        changed_tokenizer, output_dir=output, resume_from=output / "last.pt"
    ) == []
