from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.common.config import benchmark_config_from_dict, load_benchmark_config
from benchmarks.common.data import BenchmarkTask, RawDataset, configured_tasks
from benchmarks.common.environment import environment_command, environment_run_details
from benchmarks.common.engine import TargetStats
from benchmarks.molformer.adapter import (
    ConditionStats,
    SharedMolFormerRegressor,
    SortishBatchSampler,
    _collate,
    _prepare_split,
    model_input_smiles,
)
from scripts.benchmarks.sweep import _scientific_config


class FakeTokenizer:
    def __init__(self, lengths: dict[str, int]) -> None:
        self.lengths = lengths
        self.calls: list[dict[str, object]] = []
        self.values: list[tuple[str, ...]] = []
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.truncation_side = "right"

    def __call__(self, values, **kwargs):
        self.calls.append(dict(kwargs))
        single = isinstance(values, str)
        items = [values] if single else list(values)
        self.values.append(tuple(items))
        lengths = [self.lengths.get(value, 4) for value in items]
        if kwargs.get("truncation"):
            lengths = [min(length, int(kwargs["max_length"])) for length in lengths]
        if kwargs.get("return_tensors") == "pt":
            width = max(lengths, default=0)
            input_ids = torch.zeros((len(items), width), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for index, length in enumerate(lengths):
                values = torch.arange(length)
                if length:
                    values[0] = 0
                    values[-1] = 1
                input_ids[index, :length] = values
                attention_mask[index, :length] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        encoded = []
        for length in lengths:
            values = list(range(length))
            if length:
                values[0] = 0
                values[-1] = 1
            encoded.append(values)
        return {"input_ids": encoded[0] if single else encoded}


def _task() -> BenchmarkTask:
    return BenchmarkTask(
        benchmark="stage2_physics",
        task_id="simulation/tiny",
        slots=("cation", "anion"),
        condition_columns=("temperature_K",),
        target_columns=("value",),
        audit_columns=(),
        train_paths=(),
        valid_paths=(),
        test_path=None,  # type: ignore[arg-type]
        fold=None,
        meta_group=None,
        registry_payload={"task_id": "simulation/tiny"},
    )


def _raw() -> RawDataset:
    return RawDataset(
        components=(
            ("F/C=C/F", "[Cl-]"),
            ("F/C=C\\F", "[Br-]"),
            ("C[N+](C)(C)C", "[I-]"),
            ("CC[N+](C)(C)C", "[Cl-]"),
        ),
        component_count=2,
        conditions=np.asarray([[300.0], [310.0], [320.0], [330.0]]),
        targets=np.asarray([[1.0], [2.0], [4.0], [6.0]]),
        source_rows=("train.csv:2", "train.csv:3", "train.csv:4", "train.csv:5"),
        audit_rows=({}, {}, {}, {}),
    )


def test_formal_molformer_config_resolves_108_training_jobs() -> None:
    config = load_benchmark_config("configs/benchmarks/molformer.yaml")
    stage3 = configured_tasks(config, "stage3")
    stage2 = configured_tasks(config, "stage2_physics")
    assert len(stage3) == 21
    assert stage2 == (
        "simulation/heat_of_vaporization",
        "simulation/homo",
        "simulation/lumo",
    )
    assert len(stage3) * len(config.stage3.folds) + len(stage2) == 108
    assert config.training["batch_size"] == 256
    assert config.training["max_epochs"] == 50
    assert config.training["early_stopping_patience"] == 8
    assert config.training["tf32"] is True
    assert config.runtime == {
        "num_workers": 4,
        "prefetch_factor": 2,
        "persistent_workers": True,
        "pin_memory": True,
        "non_blocking_transfer": True,
    }
    old_recipe = copy.deepcopy(config.to_dict())
    old_recipe["training"]["batch_size"] = 32
    with pytest.raises(ValueError, match="registered fine-tuning recipe"):
        benchmark_config_from_dict(old_recipe)
    runtime_variant = replace(config, runtime={**config.runtime, "num_workers": 8})
    assert runtime_variant.to_dict()["runtime"]["num_workers"] == 8
    assert _scientific_config(runtime_variant) == _scientific_config(config)


def test_molformer_environment_dispatch_and_public_details() -> None:
    config = load_benchmark_config("configs/benchmarks/molformer.yaml")
    command = environment_command(
        config,
        ("scripts/benchmarks/train.py", "--config", "configs/benchmarks/molformer.yaml"),
        conda="/conda",
    )
    assert command[:6] == [
        "/conda", "run", "--no-capture-output", "-n", "ilume-molformer", "python"
    ]
    details = environment_run_details(
        {
            "environment_name": "ilume-molformer",
            "environment_lock_sha256": "lock",
            "direct_versions": {"transformers": "5.12.1"},
            "pretrained_snapshot": {"revision": "revision"},
        }
    )
    assert details == {
        "benchmark_environment": "ilume-molformer",
        "environment_lock_sha256": "lock",
        "transformers_version": "5.12.1",
        "hf_revision": "revision",
    }


def test_stereochemistry_collapse_and_train_overlength_filtering() -> None:
    first = model_input_smiles("F/C=C/F")
    second = model_input_smiles("F/C=C\\F")
    assert first == second
    tokenizer = FakeTokenizer({first: 203})
    token_cache = {}
    prepared = _prepare_split(
        _raw(), _task(), "train", tokenizer, token_cache, None, max_tokens=202
    )
    assert prepared.raw.source_rows == ("train.csv:4", "train.csv:5")
    assert prepared.audit["collision_group_count"] == 1
    assert prepared.audit["collision_affected_rows"] == 2
    assert prepared.audit["overlength_row_count"] == 2
    assert prepared.audit["skipped_rows"] == ["train.csv:2", "train.csv:3"]
    stats = TargetStats.fit(prepared.raw.targets)
    assert stats.mean == (5.0,)
    assert stats.scale == (1.0,)
    calls = len(tokenizer.calls)
    _prepare_split(
        _raw(), _task(), "valid", tokenizer, token_cache, None, max_tokens=202
    )
    assert len(tokenizer.calls) == calls


def test_valid_overlength_is_retained_and_explicitly_truncated() -> None:
    first = model_input_smiles("F/C=C/F")
    tokenizer = FakeTokenizer({first: 203})
    stats = ConditionStats.fit(_raw().conditions)
    token_cache = {}
    prepared = _prepare_split(
        _raw(), _task(), "valid", tokenizer, token_cache, stats, max_tokens=202
    )
    assert len(prepared.raw) == 4
    assert prepared.audit["truncated_rows"] == ["train.csv:2", "train.csv:3"]
    collate = _collate(
        prepared,
        token_cache,
        SimpleNamespace(normalize=lambda values: values.astype(np.float32)),
        truncate=True,
        max_tokens=202,
        pad_token_id=0,
    )
    components, conditions, targets = collate([0, 1, 2, 3])
    assert components["input_ids"].shape == (8, 202)
    assert conditions.shape == (4, 1)
    assert targets.shape == (4, 1)
    assert not any(call.get("truncation") is True for call in tokenizer.calls)
    direct = tokenizer(
        [first], truncation=True, max_length=202, return_tensors="pt"
    )
    assert torch.equal(components["input_ids"][0], direct["input_ids"][0])


class FakeBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls = 0
        self.last_input_ids = None

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        self.last_input_ids = input_ids.clone()
        pooled = input_ids.float().mean(dim=1, keepdim=True).repeat(1, 4)
        return SimpleNamespace(pooler_output=pooled * self.weight)


def test_multicomponent_model_reuses_one_backbone_and_one_linear_fusion() -> None:
    backbone = FakeBackbone()
    classifier = torch.nn.Linear(4, 1)
    model = SharedMolFormerRegressor(
        backbone,
        classifier,
        component_count=3,
        condition_dim=2,
        hidden_dim=4,
        initializer_range=0.02,
    )
    input_ids = torch.cat(
        [torch.ones((2, 3), dtype=torch.long) * value for value in (1, 2, 3)]
    )
    inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    output = model(inputs, torch.zeros((2, 2)))
    assert output.shape == (2, 1)
    assert backbone.calls == 1
    assert torch.equal(backbone.last_input_ids, input_ids)
    assert model.backbone is backbone
    assert isinstance(model.fusion, torch.nn.Linear)
    assert model.fusion.in_features == 14


def test_single_component_without_conditions_has_no_fusion() -> None:
    model = SharedMolFormerRegressor(
        FakeBackbone(),
        torch.nn.Linear(4, 1),
        component_count=1,
        condition_dim=0,
        hidden_dim=4,
        initializer_range=0.02,
    )
    assert model.fusion is None


def test_sortish_sampler_is_epoch_deterministic_and_complete() -> None:
    sampler = SortishBatchSampler(
        tuple(range(37)), batch_size=8, window_batches=2, seed=42
    )
    first = list(sampler)
    assert first == list(sampler)
    assert sorted(index for batch in first for index in batch) == list(range(37))
    assert sorted(map(len, first)) == [5, 8, 8, 8, 8]
    sampler.set_epoch(1)
    assert list(sampler) != first
