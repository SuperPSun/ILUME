from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.common.config import benchmark_config_from_dict, load_benchmark_config
from benchmarks.common.data import BenchmarkTask, RawDataset, configured_tasks
from benchmarks.common.engine import TargetStats
from benchmarks.common.environment import environment_command, environment_run_details
from benchmarks.common.environment import ilbert_asset_snapshot
from benchmarks.ilbert.adapter import (
    EpochBatchSampler,
    SharedILBERTRegressor,
    _collate,
    _prepare_split,
    build_ilbert_model,
    ilbert_model_sequences,
)
from scripts.benchmarks.sweep import _scientific_config


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def encode(self, sequence: str, **kwargs):
        truncated = bool(kwargs.get("truncation"))
        self.calls.append((sequence, truncated))
        length = 103 if sequence.startswith("long") else len(sequence) + 2
        values = [2, *range(5, 5 + max(0, length - 2)), 3]
        if truncated:
            maximum = int(kwargs["max_length"])
            values = values[: maximum - 1] + [3] if len(values) > maximum else values
            values += [0] * (maximum - len(values))
        return values


def _task(slots=("cation", "anion"), conditions=("temperature_K",)) -> BenchmarkTask:
    return BenchmarkTask(
        benchmark="stage3",
        task_id="experiment/tiny",
        slots=slots,
        condition_columns=conditions,
        target_columns=("value",),
        audit_columns=(),
        train_paths=(),
        valid_paths=(),
        test_path=None,  # type: ignore[arg-type]
        fold=1,
        meta_group="tiny",
        registry_payload={"task_id": "experiment/tiny"},
    )


def _raw() -> RawDataset:
    return RawDataset(
        components=(("C[N+]", "[Cl-]"), ("long-cation", "[Br-]")),
        component_count=2,
        conditions=np.asarray([[300.0], [350.0]]),
        targets=np.asarray([[1.0], [3.0]]),
        source_rows=("fold1.csv:2", "fold1.csv:3"),
        audit_rows=({}, {}),
    )


def test_formal_ilbert_config_resolves_108_training_jobs() -> None:
    config = load_benchmark_config("configs/benchmarks/ilbert.yaml")
    stage3 = configured_tasks(config, "stage3")
    stage2 = configured_tasks(config, "stage2_physics")
    assert len(stage3) == 21
    assert stage2 == (
        "simulation/heat_of_vaporization",
        "simulation/homo",
        "simulation/lumo",
    )
    assert len(stage3) * len(config.stage3.folds) + len(stage2) == 108
    assert config.training["batch_size"] == 16
    assert config.training["condition_transform"] == "raw_physical_units"
    old_recipe = copy.deepcopy(config.to_dict())
    old_recipe["training"]["tf32"] = False
    with pytest.raises(ValueError, match="registered fine-tuning recipe"):
        benchmark_config_from_dict(old_recipe)
    runtime_variant = replace(config, runtime={**config.runtime, "num_workers": 8})
    assert _scientific_config(runtime_variant) == _scientific_config(config)


def test_ilbert_environment_dispatch_and_public_details() -> None:
    config = load_benchmark_config("configs/benchmarks/ilbert.yaml")
    command = environment_command(
        config,
        ("scripts/benchmarks/train.py", "--config", "configs/benchmarks/ilbert.yaml"),
        conda="/conda",
    )
    assert command[:6] == [
        "/conda", "run", "--no-capture-output", "-n", "ilume-ilbert", "python"
    ]
    assert environment_run_details(
        {
            "environment_name": "ilume-ilbert",
            "environment_lock_sha256": "lock",
            "direct_versions": {"transformers": "4.39.1"},
            "pretrained_snapshot": {"revision": "commit"},
        }
    ) == {
        "benchmark_environment": "ilume-ilbert",
        "environment_lock_sha256": "lock",
        "transformers_version": "4.39.1",
        "upstream_revision": "commit",
    }


def test_ilbert_asset_snapshot_checks_revision_hashes_and_special_ids(
    tmp_path, monkeypatch
) -> None:
    config = load_benchmark_config("configs/benchmarks/ilbert.yaml")
    checkout = tmp_path / "upstream"
    source = checkout / "ILBERT"
    source.mkdir(parents=True)
    checkpoint = tmp_path / "pretrained_model.pth"
    for path in (
        source / "model.py",
        source / "ILtokenizer.py",
        source / "merged_vocab.txt",
        checkpoint,
    ):
        path.write_text("fixture", encoding="utf-8")

    def repository_path(value):
        return checkpoint if str(value).endswith("pretrained_model.pth") else checkout

    hashes = {
        "model.py": config.model["model_source_sha256"],
        "ILtokenizer.py": config.model["tokenizer_source_sha256"],
        "merged_vocab.txt": config.model["vocab_sha256"],
        "pretrained_model.pth": config.model["pretrained_sha256"],
    }
    monkeypatch.setattr("benchmarks.common.environment.repository_path", repository_path)
    monkeypatch.setattr(
        "benchmarks.common.environment.sha256_file", lambda path: hashes[path.name]
    )
    monkeypatch.setattr(
        "benchmarks.common.environment.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=f"{config.model['revision']}\n"
        ),
    )

    class Tokenizer:
        vocab_size = 2000
        pad_token_id = 0
        unk_token_id = 1
        cls_token_id = 2
        sep_token_id = 3
        mask_token_id = 4

        def __init__(self, path):
            pass

    monkeypatch.setattr(
        "benchmarks.common.environment._load_ilbert_tokenizer_class",
        lambda path: Tokenizer,
    )
    snapshot = ilbert_asset_snapshot(config)
    assert snapshot["revision"] == config.model["revision"]
    assert snapshot["tokenizer"]["special_token_ids"]["sep"] == 3
    hashes["model.py"] = "wrong"
    with pytest.raises(RuntimeError, match="asset hash mismatch"):
        ilbert_asset_snapshot(config)


def test_ilbert_topology_cache_truncation_and_raw_conditions() -> None:
    task = _task()
    assert ilbert_model_sequences(task, ("cat", "anion")) == (
        ("cat.anion",), ("ionic_liquid",)
    )
    solvation = replace(task, slots=("cation", "anion", "solute"))
    assert ilbert_model_sequences(solvation, ("cat", "anion", "solute")) == (
        ("cat.anion", "solute"), ("ionic_liquid", "solute")
    )
    tokenizer = FakeTokenizer()
    cache = {}
    prepared = _prepare_split(
        _raw(), task, "train", tokenizer, cache, max_length=100
    )
    assert np.array_equal(prepared.raw_conditions, _raw().conditions.astype(np.float32))
    assert prepared.audit["truncated_row_count"] == 1
    assert prepared.audit["truncated_rows"] == ["fold1.csv:3"]
    assert prepared.audit["truncated_sequences"][0]["source_slots"] == [
        "cation", "anion"
    ]
    calls = len(tokenizer.calls)
    _prepare_split(_raw(), task, "valid", tokenizer, cache, max_length=100)
    assert len(tokenizer.calls) == calls
    collate = _collate(prepared, cache, TargetStats.fit(_raw().targets))
    input_ids, conditions, targets = collate([0, 1])
    assert input_ids.shape == (2, 100)
    assert conditions.tolist() == [[300.0], [350.0]]
    assert targets.shape == (2, 1)


class FakeRoberta(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        states = input_ids.float().unsqueeze(-1).repeat(1, 1, 4) * self.weight
        return SimpleNamespace(last_hidden_state=states)


class FakeTextCNN(torch.nn.Module):
    def forward(self, states):
        return states.permute(1, 0, 2).mean(dim=1)


class FakeOfficialILBERT(torch.nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()
        self.roberta = torch.nn.Module()
        self.roberta.encoder = torch.nn.Linear(2, 2)
        self.roberta.pooler = torch.nn.Linear(2, 2)
        self.CNN = torch.nn.Linear(2, 2)
        self.pred_head = torch.nn.Sequential(
            torch.nn.Linear(512, 256), torch.nn.Softplus(), torch.nn.Linear(256, 1)
        )


def test_ilbert_checkpoint_load_audit_and_predictor_extension(monkeypatch) -> None:
    config = load_benchmark_config("configs/benchmarks/ilbert.yaml")
    reference = FakeOfficialILBERT()
    checkpoint = {
        key: value.clone()
        for key, value in reference.state_dict().items()
        if key.startswith("roberta.encoder.")
    }
    checkpoint["lm_head.decoder.weight"] = torch.ones((2, 2))
    monkeypatch.setattr(
        "benchmarks.ilbert.adapter._load_module",
        lambda path, name: SimpleNamespace(ILBERT=FakeOfficialILBERT),
    )
    monkeypatch.setattr(
        "benchmarks.ilbert.adapter._upstream_paths",
        lambda config: (Path("a"), Path("b"), Path("c"), Path("d")),
    )
    monkeypatch.setattr("benchmarks.ilbert.adapter.torch.load", lambda *args, **kwargs: checkpoint)
    bundle = SimpleNamespace(
        train=SimpleNamespace(view_count=2, raw_conditions=np.empty((1, 1)))
    )
    model = build_ilbert_model(config, bundle)
    assert model.predictor[0].in_features == 1025
    assert model.load_audit["unexpected_keys"] == ["lm_head.decoder.weight"]
    assert all(
        key.startswith(("CNN.", "pred_head.", "roberta.pooler."))
        for key in model.load_audit["missing_keys"]
    )

    checkpoint.pop("roberta.encoder.weight")
    with pytest.raises(RuntimeError, match="checkpoint load contract mismatch"):
        build_ilbert_model(config, bundle)


def test_ilbert_multiview_uses_one_shared_backbone_forward_in_order() -> None:
    backbone = FakeRoberta()
    model = SharedILBERTRegressor(
        backbone,
        FakeTextCNN(),
        torch.nn.Linear(9, 1),
        view_count=2,
        condition_dim=1,
        hidden_dim=4,
        load_audit={},
    )
    input_ids = torch.cat(
        (torch.ones((2, 3), dtype=torch.long), torch.ones((2, 3), dtype=torch.long) * 2)
    )
    output = model(input_ids, torch.asarray([[10.0], [20.0]]))
    assert output.shape == (2, 1)
    assert backbone.calls == 1
    assert model.roberta is backbone


def test_ilbert_epoch_sampler_is_deterministic_and_complete() -> None:
    sampler = EpochBatchSampler(37, batch_size=16, seed=42)
    first = list(sampler)
    assert first == list(sampler)
    assert sorted(index for batch in first for index in batch) == list(range(37))
    assert sorted(map(len, first)) == [5, 16, 16]
    sampler.set_epoch(1)
    assert list(sampler) != first
