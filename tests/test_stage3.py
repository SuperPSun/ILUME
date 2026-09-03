from __future__ import annotations

import csv

import json

import math

import shutil

from dataclasses import asdict, replace

from pathlib import Path

from unittest.mock import patch

import pytest

import numpy as np
import torch
from rdkit import Chem

from common.descriptor_preprocessing import FeaturePreprocessor
from common.identity import IDENTITY_CONTRACT_VERSION, semantic_identity, tensor_state_hash

from stage3.config import (
    BASE_GROUP_TASKS,
    Stage3Config,
    Stage3DataConfig,
    Stage3GroupConfig,
    Stage3InitializationConfig,
    Stage3ModelConfig,
    Stage3PluginAdaptationConfig,
    Stage3PluginConfig,
    Stage3PreparationConfig,
    Stage3RepresentationConfig,
    Stage3TaskConfig,
    Stage3TrainingConfig,
    effective_training_seed,
    load_stage3_config,
)

from stage3.data import (
    ObjectKey,
    ResolvedTaskSpec,
    Stage3TaskDataset,
    Stage3RepresentationStore,
    balanced_virtual_indices,
    composite_steps_per_epoch,
    resolve_batch_allocation,
    resolve_task_registry,
    source_path,
)

from stage3.model import GLOBAL, Stage3SparseModel, group_owner, private_owner

from stage3.pcgrad import hierarchical_pcgrad

from stage3.prepare import (
    load_prepared_stage3,
    materialize_object_embeddings,
    prepare_stage3,
)

from stage3.evaluate import evaluate_checkpoints

from stage3.train import (
    STAGE3_CHECKPOINT_KIND,
    STAGE3_CHECKPOINT_VERSION,
    STAGE3_RDKIT_CHECKPOINT_KIND,
    STAGE3_RDKIT_REFINED_KIND,
    _load_plugin,
    checkpoint_epochs,
    compute_task_gradient,
    resolve_stage3_training_identity,
    run_stage3_training,
)

from stage3.identity import build_stage3_training_identity, metadata_identity

from dataclasses import replace

import yaml

from common.identity import semantic_identity

from stage3.capacity import (
    CapacityStudyConfig,
    aggregate_fold_summaries,
    config_for_trial,
    confirmation_trial_numbers,
    load_capacity_study_config,
    materialize_final_recipe_configs,
    select_probe_winners,
    summarize_capacity_manifest,
    refined_validation_summary,
    validate_anchor_decision,
)

from stage3.search import (
    ALL_TASKS,
    EVALUATION_WEIGHTS,
    EVALUATION_WEIGHT_SUM,
    TIER_1,
    TIER_2,
    TIER_3,
    WEAK_TASKS,
    aggregate_search_trial,
    config_for_search_c,
    expert_candidates,
    grouping_candidates,
    load_search_config,
    rank_trials,
)

from stage3.config import load_stage3_config

import scripts.stage3.train as train_launcher
import scripts.stage3.search as search_launcher

from stage1.config import load_config
from stage1.descriptors import calculate_descriptors, rdkit_descriptor_names

from stage1.identity import build_stage1_corpus_identity

from stage2.config import load_stage2_config

from common.io import sha256_file

import hashlib

from typing import Any

from stage3.config import validate_stage3_folds

import argparse

from types import SimpleNamespace

import scripts.stage3.evaluate as evaluate_launcher

from stage3.config import Stage3Config

from stage3.evaluate import resolve_stage3_reporting_study_id

# --- Sparse-label model, training, and resume contracts ---

TEST_ENCODER_IDENTITY = semantic_identity(
    "stage2.encoder", {"contract_version": 1, "test": True}
)

def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def _catalog_row(
    task: str,
    target: str,
    identities: str,
    conditions: str,
    system_type: str,
    strategies: str,
) -> dict[str, object]:
    return {
        "catalog_schema_version": 1,
        "stage": 3,
        "task_id": task,
        "target_columns": target,
        "identity_columns": identities,
        "condition_columns": conditions,
        "system_type": system_type,
        "materialized_path": f"stage3/{task}",
        "strategies": strategies,
    }

def _tiny_config(tmp_path: Path) -> Stage3Config:
    catalog = tmp_path / "task_catalog.csv"
    rows = [
        _catalog_row("experiment/a", "a", "cation;anion", "", "il", "random;il;cation"),
        _catalog_row("experiment/b", "b", "cation;anion", "temperature_K", "il", "il;anion"),
        _catalog_row(
            "experiment/c", "c", "solute;solvent", "temperature_K",
            "solute_solvent", "random;solute_solvent;solute;solvent",
        ),
    ]
    _write_csv(catalog, list(rows[0]), rows)
    stage3 = tmp_path / "stage3"
    for task, directory, fields in (
        ("experiment/a", "IL", ["cation", "anion", "a"]),
        ("experiment/b", "IL", ["cation", "anion", "temperature_K", "b"]),
        ("experiment/c", "solute-solvent", ["solute", "solvent", "temperature_K", "c"]),
    ):
        for fold in range(1, 6):
            if task == "experiment/c":
                data = [
                    {"solute": "C", "solvent": "O", "temperature_K": 290 + fold, "c": fold},
                    {"solute": "CC", "solvent": "CO", "temperature_K": 300 + fold, "c": fold + 0.5},
                ]
            else:
                target = task.rsplit("/", 1)[1]
                data = [
                    {"cation": "[Na+]", "anion": "[Cl-]", target: fold},
                    {"cation": "[K+]", "anion": "[Br-]", target: fold + 0.5},
                ]
                if task == "experiment/b":
                    data[0]["temperature_K"] = 290 + fold
                    data[1]["temperature_K"] = 300 + fold
            _write_csv(stage3 / task / directory / f"fold{fold}.csv", fields, data)
        _write_csv(stage3 / task / "test.csv", fields, data)
    checkpoint = tmp_path / "stage2.pt"
    checkpoint.write_bytes(b"stage2-object-v3-test")
    return Stage3Config(
        data=Stage3DataConfig(
            stage3_dir=stage3,
            task_catalog=catalog,
            artifacts_dir=tmp_path / "artifacts",
            seed=13,
        ),
        preparation=Stage3PreparationConfig(
            encoding_batch_size=2, cache_dir=tmp_path / "cache"
        ),
        initialization=Stage3InitializationConfig(stage2_encoder=checkpoint),
        model=Stage3ModelConfig(
            global_experts=1, group_experts=1, private_experts=1,
            dropout=0.0, expert_hidden_ratio=1.0,
            interaction_hidden_ratio=1.0, film_hidden_ratio=1.0,
            tower_hidden_ratio=1.0,
        ),
        groups={
            "g1": Stage3GroupConfig(),
            "g2": Stage3GroupConfig(),
        },
        tasks={
            "experiment/a": Stage3TaskConfig(meta_group="g1"),
            "experiment/b": Stage3TaskConfig(meta_group="g1"),
            "experiment/c": Stage3TaskConfig(
                meta_group="g2", partner_mode="interaction",
                primary_slots=("solute",), partner_slots=("solvent",),
            ),
        },
        training=Stage3TrainingConfig(
            composite_batch_size=8, microbatch_size=2, virtual_min_size=4,
            epochs=2, checkpoint_interval_epochs=1, amp_dtype="none",
            device="cpu", cpu_threads=1, cpu_interop_threads=1,
        ),
    )

@pytest.fixture()
def tiny_prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Stage3Config:
    config = _tiny_config(tmp_path)
    monkeypatch.setattr(
        "stage3.prepare.load_stage2_encoder_identity",
        lambda path: TEST_ENCODER_IDENTITY,
    )

    def fake_materialize(config, object_keys, reporter=None):
        del reporter
        values = torch.arange(len(object_keys) * 4, dtype=torch.float32).reshape(-1, 4) / 10
        return values, TEST_ENCODER_IDENTITY, {
            "hits": 0, "misses": len(object_keys)
        }

    with patch("stage3.prepare.materialize_object_embeddings", side_effect=fake_materialize):
        summary = prepare_stage3(config)
    assert summary["task_count"] == 3
    return config


@pytest.fixture()
def tiny_rdkit_prepared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Stage3Config:
    config = _tiny_config(tmp_path)
    config = replace(
        config,
        data=replace(config.data, artifacts_dir=tmp_path / "rdkit-artifacts"),
        preparation=replace(
            config.preparation, cache_dir=tmp_path / "rdkit-cache"
        ),
        initialization=Stage3InitializationConfig(
            stage2_encoder=None, plugin=None
        ),
        representation=Stage3RepresentationConfig(
            kind="rdkit_2d_adapter",
            descriptor_family="rdkit_2d",
            adapter="linear_layernorm",
            output_dim=512,
        ),
    )
    monkeypatch.setattr(
        "stage3.prepare.load_stage2_encoder_identity",
        lambda *_: pytest.fail("RDKit prepare loaded a Stage 2 identity"),
    )
    monkeypatch.setattr(
        "stage3.prepare.load_frozen_object_encoder",
        lambda *_args, **_kwargs: pytest.fail("RDKit prepare loaded Stage 2"),
    )
    summary = prepare_stage3(config)
    assert summary["artifact_kind"] == "ilume_stage3_rdkit_sparse_data"
    return config

def test_base_registry_and_config_defaults_are_explicit() -> None:
    config = load_stage3_config("configs/v1/stage3/base.yaml")
    assert sum(map(len, BASE_GROUP_TASKS.values())) == 21
    assert len(config.tasks) == 21
    assert len(config.groups) == 6
    assert config.data.split_policy == "prefer_il"
    assert config.training.microbatch_size == 1024
    assert config.training.checkpoint_interval_epochs == 10
    assert config.training.seed is None
    assert effective_training_seed(config) == config.data.seed
    assert config.model.dropout == 0.10
    assert config.model.expert_hidden_ratio == 2.0
    assert config.representation is None
    assert "representation" not in config.to_dict()
    assert checkpoint_epochs(100, 10) == tuple(range(10, 101, 10))
    assert checkpoint_epochs(23, 10) == (10, 20, 23)

    ablation = load_stage3_config(
        "configs/ablations/stage1_stage2_rdkit_home.yaml"
    )
    assert len(ablation.enabled_task_ids) == 21
    assert ablation.representation == Stage3RepresentationConfig(
        kind="rdkit_2d_adapter",
        descriptor_family="rdkit_2d",
        adapter="linear_layernorm",
        output_dim=512,
    )
    assert ablation.initialization.stage2_encoder is None
    assert ablation.initialization.plugin is None


def test_v2_native_split_configs_match_materialized_task_subsets() -> None:
    expected = {
        "il": ({"il", "solute_solvent"}, 21),
        "random": ({"random"}, 21),
        "cation": ({"cation"}, 20),
        "anion": ({"anion"}, 20),
        "il_solute": ({"il_solute"}, 2),
        "solute": ({"solute"}, 3),
        "solvent": ({"solvent"}, 1),
    }
    root = Path("configs/v2/stage3/splits")
    for name, (strategies, task_count) in expected.items():
        config = load_stage3_config(root / f"{name}.yaml")
        registry = resolve_task_registry(config)
        enabled = {
            task_id: spec for task_id, spec in registry.items() if spec.enabled
        }
        assert len(enabled) == task_count
        assert {spec.split_strategy for spec in enabled.values()} == set(strategies)
        assert config.data.artifacts_dir == Path(
            f"outputs/v2/stage3/splits/{name}/prepare/artifacts"
        )
        assert config.preparation.cache_dir == Path(
            f"outputs/v2/stage3/splits/{name}/prepare/object_cache"
        )
        for spec in enabled.values():
            for fold in range(1, 6):
                assert source_path(config, spec, fold).is_file()


def test_training_seed_changes_training_identity_not_prepared_artifact(
    tiny_prepared: Stage3Config,
) -> None:
    original_metadata = json.loads(
        (tiny_prepared.data.artifacts_dir / "metadata.json").read_text()
    )
    changed = replace(
        tiny_prepared,
        training=replace(tiny_prepared.training, seed=10042),
    )

    assert effective_training_seed(tiny_prepared) == tiny_prepared.data.seed
    assert effective_training_seed(changed) == 10042
    assert changed.data.artifacts_dir == tiny_prepared.data.artifacts_dir
    assert json.loads(
        (changed.data.artifacts_dir / "metadata.json").read_text()
    ) == original_metadata
    assert resolve_stage3_training_identity(changed, 1) != (
        resolve_stage3_training_identity(tiny_prepared, 1)
    )


def test_grouping_and_task_weights_change_training_not_prepared_identity(
    tiny_prepared: Stage3Config,
) -> None:
    regrouped = replace(
        tiny_prepared,
        groups={"merged": Stage3GroupConfig(group_weight=2.0)},
        tasks={
            task: replace(spec, meta_group="merged", task_weight=2.0)
            for task, spec in tiny_prepared.tasks.items()
        },
    )
    variants = (
        regrouped,
        replace(
            tiny_prepared,
            groups={
                name: replace(spec, group_weight=2.0)
                for name, spec in tiny_prepared.groups.items()
            },
        ),
        replace(
            tiny_prepared,
            model=replace(tiny_prepared.model, group_experts=2),
        ),
    )
    original_identity = resolve_stage3_training_identity(tiny_prepared, 1)
    for changed in variants:
        prepared = load_prepared_stage3(changed)
        assert set(prepared["registry"]) == set(tiny_prepared.tasks)
        assert resolve_stage3_training_identity(changed, 1) != original_identity
    assert all(
        spec.meta_group == "merged"
        for spec in load_prepared_stage3(regrouped)["registry"].values()
    )

def test_registry_catalog_precedence_split_and_topology(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    registry = resolve_task_registry(config)
    assert registry["experiment/a"].split_strategy == "il"
    assert registry["experiment/c"].split_strategy == "solute_solvent"
    assert registry["experiment/c"].primary_slots == ("solute",)
    assert registry["experiment/c"].partner_slots == ("solvent",)
    override = replace(
        config,
        data=replace(config.data, split_strategies={"experiment/a": "random"}),
    )
    assert resolve_task_registry(override)["experiment/a"].split_strategy == "random"

def test_ownership_is_complete_and_isolated(tiny_prepared: Stage3Config) -> None:
    model = Stage3SparseModel(tiny_prepared.model, resolve_task_registry(tiny_prepared), 4)
    ownership = model.parameter_ownership()
    assert all(owner.scope in {"GLOBAL", "GROUP", "PRIVATE"} for owner in ownership.values())
    assert set(model.parameters_for_owner(private_owner("experiment/a"))).isdisjoint(
        model.parameters_for_owner(private_owner("experiment/b"))
    )
    assert set(model.parameters_for_owner(group_owner("g1"))).isdisjoint(
        model.parameters_for_owner(group_owner("g2"))
    )
    assert model.parameters_for_owner(GLOBAL)


@pytest.mark.parametrize("global_count,private_count", [(0, 0), (0, 1), (1, 0)])
def test_zero_global_or_private_experts_preserve_forward_and_ownership(
    tiny_prepared: Stage3Config,
    global_count: int,
    private_count: int,
) -> None:
    config = replace(
        tiny_prepared,
        model=replace(
            tiny_prepared.model,
            global_experts=global_count,
            private_experts=private_count,
        ),
    )
    config.validate()
    registry = resolve_task_registry(config)
    model = Stage3SparseModel(config.model, registry, 4)
    task = "experiment/a"
    result = model(
        task,
        torch.randn(3, 4),
        torch.empty(3, len(registry[task].condition_columns)),
    )
    assert result.predictions.shape == (3,)
    assert result.diagnostics["l1_global_gate"].shape == (3, global_count)
    assert result.diagnostics["l2_global_candidates"].shape == (
        3, global_count, 4
    )
    assert result.diagnostics["l2_private_candidates"].shape == (
        3, private_count, 4
    )
    if global_count == 0:
        assert model.parameters_for_owner(GLOBAL) == ()
    assert model.parameters_for_owner(private_owner(task))


def test_v2_stage3_search_candidate_and_metric_contracts() -> None:
    spec = load_search_config("configs/v2/stage3/search.yaml")
    assert spec.folds == (1, 2)
    assert spec.epochs == 20
    groupings = grouping_candidates(spec.sampler_seed)
    assert len(groupings) == 50
    assert {source: sum(item.source == source for item in groupings) for source in (
        "anchor", "manual", "combination"
    )} == {"anchor": 5, "manual": 20, "combination": 25}
    assert {
        count: sum(item.group_count == count for item in groupings)
        for count in (2, 3, 6, 9, 12)
    } == {2: 10, 3: 10, 6: 10, 9: 10, 12: 10}
    for candidate in groupings:
        assert set(candidate.assignments) == set(ALL_TASKS)
        assert len(set(candidate.assignments.values())) == candidate.group_count

    experts = expert_candidates()
    assert len(experts) == 30
    assert {source: sum(item.source == source for item in experts) for source in (
        "local", "ablation", "higher_capacity"
    )} == {"local": 10, "ablation": 10, "higher_capacity": 10}
    assert {
        (item.global_experts, item.group_experts, item.private_experts)
        for item in experts
    } == {
        (global_count, group_count, private_count)
        for global_count in (0, 1, 2)
        for group_count in (1, 2, 3, 4, 6)
        for private_count in (0, 1)
    }
    assert EVALUATION_WEIGHT_SUM == pytest.approx(33.0)
    assert sum(EVALUATION_WEIGHTS.values()) == pytest.approx(33.0)
    config = config_for_search_c(
        load_stage3_config(spec.base_config),
        groupings[0],
        experts[0],
        {
            "tier1_weight": 5.0,
            "tier2_weight": 4.0,
            "tier3_weight": 3.0,
            "learning_rate": 1.0e-4,
            "dropout": 0.20,
            "weight_decay": 3.0e-2,
        },
    )
    assert {config.tasks[task].task_weight for task in TIER_1} == {5.0}
    assert {config.tasks[task].task_weight for task in TIER_2} == {4.0}
    assert {config.tasks[task].task_weight for task in TIER_3} == {3.0}
    assert {
        config.tasks[task].task_weight
        for task in ALL_TASKS
        if task not in WEAK_TASKS
    } == {1.0}


def test_stage3_search_phase_catalogs_bind_top3_and_balance_groupings(
    tmp_path: Path,
) -> None:
    groupings = grouping_candidates()
    a_root = tmp_path / "search_a"
    a_root.mkdir()
    (a_root / "result.json").write_text(
        json.dumps({"ranking": [{"candidate": item.to_dict()} for item in groupings[:3]]}),
        encoding="utf-8",
    )
    b_catalog, prerequisites = search_launcher._candidate_catalog("b", tmp_path)
    assert set(prerequisites) == {"search_a"}
    assert len(b_catalog) == 30
    assert {
        grouping.candidate_id: sum(
            row["grouping"]["candidate_id"] == grouping.candidate_id
            for row in b_catalog
        )
        for grouping in groupings[:3]
    } == {grouping.candidate_id: 10 for grouping in groupings[:3]}

    b_root = tmp_path / "search_b"
    b_root.mkdir()
    (b_root / "result.json").write_text(
        json.dumps({"ranking": [{"candidate": item} for item in b_catalog[:3]]}),
        encoding="utf-8",
    )
    c_catalog, prerequisites = search_launcher._candidate_catalog("c", tmp_path)
    assert set(prerequisites) == {"search_a", "search_b"}
    assert len(c_catalog) == 9
    assert len({row["grouping"]["candidate_id"] for row in c_catalog}) == 3
    assert len({row["experts"]["candidate_id"] for row in c_catalog}) == 3


def test_stage3_search_two_fold_aggregation_and_tie_break() -> None:
    weak = {task: 0.5 for task in EVALUATION_WEIGHTS if EVALUATION_WEIGHTS[task] > 1}
    fold = {
        "weighted_normalized_mae": 0.4,
        "original_macro_task_score": 0.6,
        "weak_task_scores": weak,
        "training_cost": {
            "wall_seconds": 2.0,
            "gpu_seconds": 2.0,
            "peak_allocated_bytes": 100,
            "total_parameters": 10,
            "trainable_parameters": 9,
        },
    }
    row = aggregate_search_trial(
        {1: fold, 2: {**fold, "weighted_normalized_mae": 0.6}},
        trial_number=7,
        candidate={"candidate_id": "x"},
    )
    assert row["score"] == pytest.approx(0.5)
    assert row["fold_sample_sd"] == pytest.approx(math.sqrt(0.02))
    assert row["training_cost"]["gpu_seconds"] == pytest.approx(4.0)
    slower = {
        **row,
        "trial_number": 8,
        "training_cost": {**row["training_cost"], "gpu_seconds": 5.0},
    }
    assert [item["trial_number"] for item in rank_trials([slower, row])] == [7, 8]


def test_stage3_search_a_runs_fixed_budget_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stage3.search as search_module

    spec = load_search_config("configs/v2/stage3/search.yaml")
    base = search_module.search_base_config(
        load_stage3_config(spec.base_config), spec
    )
    calls = 0

    def fake_wave(trials, *, phase, folds, devices, max_retries):
        nonlocal calls
        calls += 1
        assert phase == "screen"
        assert folds == (1, 2)
        assert max_retries == 1
        return {
            number: {
                fold: trial_root / phase / "attempt0" / f"fold{fold}"
                for fold in folds
            }
            for number, _, trial_root in trials
        }

    def fake_fold_summary(root, *, expected_epochs):
        number = int(next(part for part in Path(root).parts if part.startswith("trial_")).split("_")[1])
        score = 0.1 + number / 1000
        return {
            "weighted_normalized_mae": score,
            "original_macro_task_score": score + 0.1,
            "weak_task_scores": {
                task: score for task in search_module.WEAK_TASKS
            },
            "training_cost": {
                "wall_seconds": 1.0,
                "gpu_seconds": 1.0,
                "peak_allocated_bytes": 10,
                "total_parameters": 100,
                "trainable_parameters": 90,
            },
        }

    monkeypatch.setattr(train_launcher, "_run_capacity_wave", fake_wave)
    monkeypatch.setattr(search_module, "fold_search_summary", fake_fold_summary)
    monkeypatch.setattr(search_launcher, "ROOT", tmp_path)
    output = tmp_path / "search_a"
    result = search_launcher._run_study(
        phase="a",
        base=base,
        spec=spec,
        catalog=[item.to_dict() for item in grouping_candidates()],
        phase_root=output,
        resume=False,
        devices=("cuda:0", "cuda:1"),
        max_parallel=2,
    )
    assert result["attempted_trials"] == 50
    assert result["completed_trials"] == 50
    assert [row["trial_number"] for row in result["top3"]] == [0, 1, 2]
    first_calls = calls
    resumed = search_launcher._run_study(
        phase="a",
        base=base,
        spec=spec,
        catalog=[item.to_dict() for item in grouping_candidates()],
        phase_root=output,
        resume=True,
        devices=("cuda:0", "cuda:1"),
        max_parallel=2,
    )
    assert resumed == result
    assert calls == first_calls


def test_rdkit_prepare_adapter_refinement_and_reporting_contract(
    tiny_rdkit_prepared: Stage3Config,
) -> None:
    metadata = json.loads(
        (tiny_rdkit_prepared.data.artifacts_dir / "metadata.json").read_text()
    )
    assert metadata["kind"] == "ilume_stage3_rdkit_sparse_data"
    assert "stage2_encoder_identity" not in metadata
    contract = metadata["descriptor_contract"]
    assert contract["fit_scope"] == "joint_training_rows"
    assert contract["clip"] == [-10.0, 10.0]
    assert set(contract["fold_preprocessing"]) == {
        f"fold{fold}" for fold in range(1, 6)
    }
    names = rdkit_descriptor_names()
    descriptor = lambda smiles: calculate_descriptors(
        Chem.MolFromSmiles(smiles), names
    )
    expected_il = FeaturePreprocessor.fit(
        np.stack(
            [
                np.concatenate((descriptor("[Na+]"), descriptor("[Cl-]"))),
                np.concatenate((descriptor("[K+]"), descriptor("[Br-]"))),
            ]
            * 8
        )
    )
    expected_single = FeaturePreprocessor.fit(
        np.stack(
            [descriptor(smiles) for smiles in ("C", "O", "CC", "CO")] * 4
        )
    )
    fold1 = contract["fold_preprocessing"]["fold1"]
    assert FeaturePreprocessor.from_dict(fold1["il"]) == expected_il
    assert FeaturePreprocessor.from_dict(fold1["single"]) == expected_single
    assert len(fold1["il"]["finite_mask"]) == 2 * len(names)
    assert len(fold1["single"]["finite_mask"]) == len(names)

    prepared_objects = {
        "objects": json.loads(
            (tiny_rdkit_prepared.data.artifacts_dir / "objects.json").read_text()
        )
    }
    store = Stage3RepresentationStore(
        tiny_rdkit_prepared.data.artifacts_dir,
        1,
        prepared_objects,
        metadata["kind"],
    )
    registry = resolve_task_registry(tiny_rdkit_prepared)
    model = Stage3SparseModel(
        tiny_rdkit_prepared.model,
        registry,
        store.output_dim,
        descriptor_input_dims=store.input_dims,
    )
    assert list(model.descriptor_adapters) == ["il", "molecule"]
    for adapter in model.descriptor_adapters.values():
        assert [type(layer) for layer in adapter] == [
            torch.nn.Linear,
            torch.nn.LayerNorm,
        ]
        assert all(
            model.parameter_ownership()[parameter] == GLOBAL
            for parameter in adapter.parameters()
        )

    output = tiny_rdkit_prepared.data.artifacts_dir.parent / "rdkit-train"
    run_stage3_training(tiny_rdkit_prepared, 1, output_dir=output)
    boundary = torch.load(
        output / "checkpoint_epoch_00001.pt",
        map_location="cpu",
        weights_only=False,
    )
    final = torch.load(
        output / "checkpoint_epoch_00002.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert boundary["kind"] == STAGE3_RDKIT_CHECKPOINT_KIND
    assert "stage2_encoder_identity" not in boundary
    assert boundary["representation"]["kind"] == "rdkit_2d_adapter"
    adapter_names = [
        name for name in boundary["model"] if name.startswith("descriptor_adapters.")
    ]
    assert adapter_names
    assert all(
        torch.equal(boundary["model"][name], final["model"][name])
        for name in adapter_names
    )
    refined = torch.load(
        output / "taskwise_refined.pt", map_location="cpu", weights_only=False
    )
    assert refined["kind"] == STAGE3_RDKIT_REFINED_KIND
    resumed = tiny_rdkit_prepared.data.artifacts_dir.parent / "rdkit-resumed"
    resumed.mkdir()
    shutil.copy(output / "resolved_training_plan.json", resumed)
    (resumed / "metrics.jsonl").write_text(
        (output / "metrics.jsonl").read_text().splitlines()[0] + "\n"
    )
    (resumed / "diagnostics.jsonl").write_text(
        (output / "diagnostics.jsonl").read_text().splitlines()[0] + "\n"
    )
    run_stage3_training(
        tiny_rdkit_prepared,
        1,
        output_dir=resumed,
        resume_from=output / "checkpoint_epoch_00001.pt",
    )
    resumed_final = torch.load(
        resumed / "checkpoint_epoch_00002.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert all(
        torch.equal(final["model"][name], resumed_final["model"][name])
        for name in final["model"]
    )
    evaluation = evaluate_checkpoints(
        tiny_rdkit_prepared,
        output,
        split="valid",
        ensemble_folds=False,
        task_subset=("experiment/a",),
        fold=1,
    )
    assert evaluation["reporting"]["model_id"] == "rdkit_2d_home"
    assert evaluation["reporting"]["model_display_name"] == "RDKit 2D + HoME"

    object_config = replace(
        tiny_rdkit_prepared,
        representation=None,
        initialization=Stage3InitializationConfig(
            stage2_encoder=tiny_rdkit_prepared.data.task_catalog,
            plugin=None,
        ),
    )
    with pytest.raises(ValueError, match="requires RDKit representation config"):
        load_prepared_stage3(object_config)


def test_no_stage1_stage2_encoder_keeps_object_home_reporting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config(tmp_path)
    encoder_identity = semantic_identity(
        "stage2.rdkit-encoder", {"contract_version": 1, "test": True}
    )
    monkeypatch.setattr(
        "stage3.prepare.load_stage2_encoder_identity", lambda _: encoder_identity
    )

    def fake_materialize(config, object_keys, reporter=None):
        del config, reporter
        values = torch.arange(
            len(object_keys) * 4, dtype=torch.float32
        ).reshape(-1, 4) / 10
        return values, encoder_identity, {"hits": 0, "misses": len(object_keys)}

    with patch(
        "stage3.prepare.materialize_object_embeddings",
        side_effect=fake_materialize,
    ):
        prepare_stage3(config)
    metadata = json.loads(
        (config.data.artifacts_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["kind"] == "ilume_stage3_sparse_data"
    assert metadata["provenance"]["representation"] == "rdkit_2d_stage2"

    output = tmp_path / "no-stage1-stage3-train"
    run_stage3_training(config, 1, output_dir=output)
    evaluation = evaluate_checkpoints(
        config,
        output,
        split="valid",
        ensemble_folds=False,
        task_subset=("experiment/a",),
        fold=1,
    )
    assert evaluation["reporting"]["model_id"] == "rdkit_2d_stage2_home"
    assert evaluation["reporting"]["model_display_name"] == (
        "RDKit 2D MLP + Stage2 + HoME"
    )

def test_microbatch_accumulation_matches_full_task_batch(tiny_prepared: Stage3Config) -> None:
    registry = resolve_task_registry(tiny_prepared)
    dataset = Stage3TaskDataset(tiny_prepared.data.artifacts_dir, 1, "experiment/a", "train")
    embeddings = torch.load(
        tiny_prepared.data.artifacts_dir / "object_embeddings.pt",
        map_location="cpu", weights_only=True,
    )["embeddings"]
    normalization = json.loads(
        (tiny_prepared.data.artifacts_dir / "normalization.json").read_text()
    )["fold1"]["experiment/a"]
    indices = torch.arange(len(dataset))
    first = Stage3SparseModel(tiny_prepared.model, registry, 4)
    second = Stage3SparseModel(tiny_prepared.model, registry, 4)
    second.load_state_dict(first.state_dict())
    micro = replace(tiny_prepared, training=replace(tiny_prepared.training, microbatch_size=1))
    full = replace(tiny_prepared, training=replace(tiny_prepared.training, microbatch_size=len(dataset)))
    gradients_micro, _ = compute_task_gradient(
        first, "experiment/a", dataset, indices, embeddings, normalization,
        micro, torch.device("cpu"),
    )
    gradients_full, _ = compute_task_gradient(
        second, "experiment/a", dataset, indices, embeddings, normalization,
        full, torch.device("cpu"),
    )
    first_named = dict(first.named_parameters())
    second_named = dict(second.named_parameters())
    for name in first_named:
        left = gradients_micro.get(first_named[name])
        right = gradients_full.get(second_named[name])
        assert (left is None) == (right is None)
        if left is not None:
            assert torch.allclose(left, right, atol=1e-6, rtol=1e-5)

def test_pcgrad_keeps_global_and_group_as_separate_blocks(
    tiny_prepared: Stage3Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stage3.pcgrad as module

    registry = resolve_task_registry(tiny_prepared)
    model = Stage3SparseModel(tiny_prepared.model, registry, 4)
    gradients = {}
    for task in ("experiment/a", "experiment/b", "experiment/c"):
        owners = (GLOBAL, group_owner(registry[task].meta_group), private_owner(task))
        gradients[task] = {
            parameter: torch.ones_like(parameter, dtype=torch.float32)
            for owner in owners
            for parameter in model.parameters_for_owner(owner)
        }
    calls: list[set[torch.nn.Parameter]] = []
    original = module.pcgrad_block

    def wrapped(raw, parameters, rng):
        calls.append(set(parameters))
        return original(raw, parameters, rng)

    monkeypatch.setattr(module, "pcgrad_block", wrapped)
    result = hierarchical_pcgrad(
        model, gradients, registry, {"g1": 1.0, "g2": 1.0},
        __import__("random").Random(3),
    )
    global_parameters = set(model.parameters_for_owner(GLOBAL))
    group_parameters = set(model.parameters_for_owner(group_owner("g1")))
    assert global_parameters in calls
    assert group_parameters in calls
    assert global_parameters | group_parameters not in calls
    private = model.parameters_for_owner(private_owner("experiment/a"))[0]
    assert torch.equal(result.gradients[private], gradients["experiment/a"][private])


def test_pcgrad_accepts_empty_global_expert_block(
    tiny_prepared: Stage3Config,
) -> None:
    config = replace(
        tiny_prepared,
        model=replace(
            tiny_prepared.model, global_experts=0, private_experts=0
        ),
    )
    registry = resolve_task_registry(config)
    model = Stage3SparseModel(config.model, registry, 4)
    gradients = {}
    for task in registry:
        owners = (group_owner(registry[task].meta_group), private_owner(task))
        gradients[task] = {
            parameter: torch.ones_like(parameter, dtype=torch.float32)
            for owner in owners
            for parameter in model.parameters_for_owner(owner)
        }
    result = hierarchical_pcgrad(
        model,
        gradients,
        registry,
        {"g1": 1.0, "g2": 1.0},
        __import__("random").Random(3),
    )
    assert result.assembled_owner_norms["GLOBAL"] == 0.0
    assert result.gradients

def test_short_training_checkpoint_and_resume_are_exact(tiny_prepared: Stage3Config) -> None:
    continuous = tiny_prepared.data.artifacts_dir.parent / "continuous"
    rows = run_stage3_training(tiny_prepared, 1, output_dir=continuous)
    assert [row["epoch"] for row in rows] == [1, 2]
    assert sorted(path.name for path in continuous.glob("checkpoint_*.pt")) == [
        "checkpoint_epoch_00001.pt", "checkpoint_epoch_00002.pt"
    ]
    assert (continuous / "taskwise_refined.pt").is_file()
    assert (continuous / "taskwise_refinement.json").is_file()
    boundary_checkpoint = torch.load(
        continuous / "checkpoint_epoch_00001.pt", map_location="cpu", weights_only=False
    )
    assert boundary_checkpoint["optimizer"]["state"]
    assert boundary_checkpoint["refinement"]["optimizers"] == {}
    refined_payload = torch.load(
        continuous / "taskwise_refined.pt", map_location="cpu", weights_only=False
    )
    assert set(refined_payload["private_state_hashes"]) == set(
        refined_payload["selected_tasks"]
    )
    resumed = tiny_prepared.data.artifacts_dir.parent / "resumed"
    resumed.mkdir()
    shutil.copy(continuous / "resolved_training_plan.json", resumed)
    first_metric = (continuous / "metrics.jsonl").read_text().splitlines()[0]
    (resumed / "metrics.jsonl").write_text(first_metric + "\n")
    first_diag = (continuous / "diagnostics.jsonl").read_text().splitlines()[0]
    (resumed / "diagnostics.jsonl").write_text(first_diag + "\n")
    resumed_rows = run_stage3_training(
        tiny_prepared, 1, output_dir=resumed,
        resume_from=continuous / "checkpoint_epoch_00001.pt",
    )
    assert [row["epoch"] for row in resumed_rows] == [2]
    expected = torch.load(
        continuous / "checkpoint_epoch_00002.pt", map_location="cpu", weights_only=False
    )["model"]
    actual = torch.load(
        resumed / "checkpoint_epoch_00002.pt", map_location="cpu", weights_only=False
    )["model"]
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[name], actual[name]) for name in expected)
    assert (resumed / "taskwise_refined.pt").is_file()

    prediction_dir = tiny_prepared.data.artifacts_dir.parent / "evaluation-predictions"
    evaluation = evaluate_checkpoints(
        tiny_prepared,
        continuous,
        split="valid",
        ensemble_folds=False,
        checkpoint_epoch=2,
        task_subset=("experiment/a",),
        fold=1,
        predictions_dir=prediction_dir,
    )
    assert evaluation["checkpoint_epoch"] == 2
    assert set(evaluation["tasks"]) == {"experiment/a"}
    prediction_path = prediction_dir / "experiment__a.csv"
    with prediction_path.open(newline="", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))
    assert prediction_rows
    spec = resolve_task_registry(tiny_prepared)["experiment/a"]
    assert set(prediction_rows[0]) == {
        "source_row", "source_fold", *spec.identity_columns,
        *spec.condition_columns, "target", "prediction", "absolute_error",
    }
    assert evaluation["reporting"]["predictions"][0]["rows"] == len(
        prediction_rows
    )
    refined = evaluate_checkpoints(
        tiny_prepared,
        continuous,
        split="valid",
        ensemble_folds=False,
        task_subset=("experiment/a",),
        fold=1,
    )
    assert refined["checkpoint_epoch"] is None
    assert refined["model_selector"] == "taskwise_refined"
    (continuous / "taskwise_refined.pt").unlink()
    (continuous / "taskwise_refinement.json").unlink()
    with pytest.raises(FileNotFoundError, match="taskwise_refined"):
        evaluate_checkpoints(
            tiny_prepared,
            continuous,
            split="valid",
            ensemble_folds=False,
            task_subset=("experiment/a",),
            fold=1,
        )
    epoch_only = evaluate_checkpoints(
        tiny_prepared,
        continuous,
        split="valid",
        ensemble_folds=False,
        checkpoint_epoch=2,
        task_subset=("experiment/a",),
        fold=1,
    )
    assert epoch_only["model_selector"] == "epoch_checkpoint"


def test_zero_global_private_experts_train_and_checkpoint(
    tiny_prepared: Stage3Config,
) -> None:
    config = replace(
        tiny_prepared,
        model=replace(
            tiny_prepared.model, global_experts=0, private_experts=0
        ),
    )
    output = tiny_prepared.data.artifacts_dir.parent / "zero-expert-train"
    rows = run_stage3_training(config, 1, output_dir=output)
    assert [row["phase"] for row in rows] == ["joint", "refinement"]
    checkpoint = torch.load(
        output / "checkpoint_epoch_00002.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["config"]["model"]["global_experts"] == 0
    assert checkpoint["config"]["model"]["private_experts"] == 0
    assert (output / "taskwise_refined.pt").is_file()

# --- Capacity v1 selection contract ---

def _write_metrics(path: Path, values: list[float]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for epoch, value in enumerate(values, start=1):
        rows.append(
            {
                "epoch": epoch,
                "validation": {
                    "tasks": {
                        "experiment/a": {"normalized_mae": value + 0.1},
                        "experiment/b": {"normalized_mae": value + 0.2},
                    },
                    "groups": {
                        "group-a": {"normalized_mae": value + 0.15},
                    },
                    "macro_task_equal": {
                        "normalized_mae": {"value": value}
                    },
                    "macro_group_equal": {
                        "normalized_mae": {"value": value + 0.05}
                    },
                },
            }
        )
    (path / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    artifact = path / "taskwise_refined.pt"
    artifact.write_bytes(b"fake-refined-artifact")
    (path / "taskwise_refinement.json").write_text(
        json.dumps({
            "kind": "ilume_stage3_taskwise_refined",
            "format_version": 1,
            "artifact": artifact.name,
            "artifact_sha256": sha256_file(artifact),
            "ordinary_final_epoch": len(values),
            "validation": rows[-1]["validation"],
        }),
        encoding="utf-8",
    )

def test_refined_score_uses_stitched_validation(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_metrics(root, [1.0] * 19 + [0.25])
    summary = refined_validation_summary(root, expected_epochs=20)
    assert summary["score"] == pytest.approx(0.25)
    assert summary["model_selector"] == "taskwise_refined"

def test_capacity_study_and_trial_config_are_strict(tmp_path: Path) -> None:
    study = tmp_path / "study.yaml"
    study.write_text(
        """
schema_version: 2
study_name: test
anchor_decision: outputs/test/anchor.yaml
attempted_trials: 40
startup_trials: 10
trials_per_wave: 2
folds: [1, 2]
confirmation_folds: [3, 4, 5]
top_k: 5
max_retries: 1
sampler_seed: 42
global_experts: [1, 2, 3, 4]
group_experts: [1, 2, 3, 4]
private_experts: [1, 2]
expert_hidden_ratio: [1.0, 1.5, 2.0, 3.0, 4.0]
dropout: [0.0, 0.3]
learning_rate: [0.0001, 0.001]
weight_decay: [0.0001, 0.1]
baseline:
  global_experts: 2
  group_experts: 2
  private_experts: 1
  expert_hidden_ratio: 2.0
  dropout: 0.1
  learning_rate: 0.0003
  weight_decay: 0.01
""".lstrip(),
        encoding="utf-8",
    )
    spec = load_capacity_study_config(study)
    base = load_stage3_config("configs/v1/stage3/base.yaml")
    base = replace(base, training=replace(base.training, epochs=20, seed=42))
    trial = config_for_trial(base, spec.baseline)
    assert trial.model.global_experts == 2
    assert trial.model.expert_hidden_ratio == 2.0
    assert trial.training.learning_rate == pytest.approx(3.0e-4)
    assert trial.training.seed == 42

def test_capacity_study_runs_synchronous_waves_and_resumes(tmp_path: Path) -> None:
    baseline = {
        "global_experts": 2,
        "group_experts": 2,
        "private_experts": 1,
        "expert_hidden_ratio": 2.0,
        "dropout": 0.1,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-2,
    }
    spec = CapacityStudyConfig(
        study_name="capacity-test",
        anchor_decision="outputs/test/anchor.yaml",
        attempted_trials=4,
        startup_trials=2,
        trials_per_wave=2,
        folds=(1, 2),
        confirmation_folds=(3, 4, 5),
        top_k=2,
        max_retries=1,
        sampler_seed=42,
        global_experts=(1, 2),
        group_experts=(1, 2),
        private_experts=(1, 2),
        expert_hidden_ratio=(1.0, 2.0),
        dropout=(0.0, 0.3),
        learning_rate=(1.0e-4, 1.0e-3),
        weight_decay=(1.0e-4, 1.0e-1),
        baseline=baseline,
    )
    spec.validate()
    base = load_stage3_config("configs/v1/stage3/base.yaml")
    base = replace(base, training=replace(base.training, epochs=20, seed=42))
    calls: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def run_wave(trials, *, phase, folds, devices, max_retries):
        assert max_retries == 1
        calls.append(
            (
                phase,
                tuple(number for number, _, _ in trials),
                tuple(folds),
            )
        )
        result = {}
        for number, _, trial_root in trials:
            result[number] = {}
            for fold in folds:
                root = trial_root / phase / "attempt0" / f"fold{fold}"
                _write_metrics(root, [0.1 + number / 100] * 20)
                result[number][fold] = root
        return result

    output = tmp_path / "study"
    report = train_launcher._run_capacity_study(
        base_config=base,
        study_config=spec,
        output=output,
        resume=False,
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        run_wave=run_wave,
    )
    assert calls[:2] == [
        ("search", (0, 1), (1, 2)),
        ("search", (2, 3), (1, 2)),
    ]
    assert report["shortlist"] == [0, 1]
    assert [row["trial_number"] for row in report["ranking"]] == [0, 1]
    calls.clear()
    resumed = train_launcher._run_capacity_study(
        base_config=base,
        study_config=spec,
        output=output,
        resume=True,
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        run_wave=run_wave,
    )
    assert resumed == report
    assert calls == []

    decision = tmp_path / "final-recipe.yaml"
    probe_config = "configs/experiments_v1/stage3/probe/base-r4.yaml"
    decision.write_text(
        __import__("yaml").safe_dump(
            {
                    "schema_version": 1,
                "kind": "final_recipe",
                "hpo_output": str(output),
                "trial_number": 0,
                "reason": "test confirmed baseline",
                "scale_configs": {
                    scale: probe_config for scale in ("s", "base", "l", "xl")
                },
                "seed_output_root": "outputs/test/seeds",
                "formal_output_root": "outputs/test/formal",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    materialized = materialize_final_recipe_configs(
        decision, tmp_path / "materialized"
    )
    assert materialized["trial_number"] == 0
    seed_config = load_stage3_config(
        tmp_path / "materialized/seed/seed10042.yaml"
    )
    formal_config = load_stage3_config(tmp_path / "materialized/formal/xl.yaml")
    assert (seed_config.training.seed, seed_config.training.epochs) == (10042, 20)
    assert (formal_config.training.seed, formal_config.training.epochs) == (42, 50)

def test_anchor_decision_requires_selected_probe_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import common.outputs as outputs_module

    monkeypatch.setattr(outputs_module, "REPOSITORY_ROOT", tmp_path)
    config_path = tmp_path / "configs/anchor.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("training: {}\n", encoding="utf-8")
    report_path = tmp_path / "outputs/probe/summary.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps({"scale_winners": [{"id": "l-default"}]}), encoding="utf-8"
    )
    decision_path = tmp_path / "outputs/anchor.yaml"
    decision_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "kind": "anchor",
                "selected_candidate": "l-default",
                "selected_config": "configs/anchor.yaml",
                "probe_report": "outputs/probe/summary.json",
                "reason": "Pareto evidence",
            }
        ),
        encoding="utf-8",
    )
    base_spec = load_capacity_study_config(
        "configs/experiments_v1/stage3/hpo.yaml"
    )
    spec = replace(base_spec, anchor_decision="outputs/anchor.yaml")
    decision = validate_anchor_decision(spec, config_path)
    assert decision["selected_candidate"] == "l-default"

# --- Multi-fold training launcher contract ---

TRAINING_IDENTITY = semantic_identity(
    "stage3.training", {"contract_version": 1, "microbatch_size": 1024}
)

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")

def _write_history(path: Path, epochs: list[int]) -> None:
    path.write_text(
        "".join(json.dumps({"epoch": epoch}) + "\n" for epoch in epochs),
        encoding="utf-8",
    )

def _write_run(
    root: Path,
    *,
    status: str,
    epochs: list[int],
    checkpoints: list[int],
    identity: dict[str, Any] = TRAINING_IDENTITY,
) -> None:
    root.mkdir()
    (root / "run_config.yaml").write_text("training: {}\n", encoding="utf-8")
    _write_json(
        root / "metadata.json",
        {
            "stage": "stage3",
            "operation": "train",
            "status": status,
            "provenance": {"fold": 2},
            "semantic_identity": identity,
        },
    )
    _write_history(root / "metrics.jsonl", epochs)
    _write_history(root / "diagnostics.jsonl", epochs)
    for epoch in checkpoints:
        (root / f"checkpoint_epoch_{epoch:05d}.pt").write_bytes(b"checkpoint")

def test_completed_run_is_skipped_after_identity_and_history_checks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fold2"
    _write_run(root, status="completed", epochs=[1, 2], checkpoints=[2])
    artifact = root / "taskwise_refined.pt"
    artifact.write_bytes(b"refined")
    refinement = {"artifact_sha256": hashlib.sha256(b"refined").hexdigest()}
    _write_json(root / "taskwise_refinement.json", refinement)
    _write_json(root / "summary.json", {
        "fold": 2,
        "final_epoch": {"epoch": 2},
        "taskwise_refinement": refinement,
    })
    assert train_launcher._resume_action(
        root, fold=2, total_epochs=2, training_identity=TRAINING_IDENTITY
    ) == ("skipped", None)

def test_resume_uses_latest_complete_checkpoint_and_matching_history(
    tmp_path: Path,
) -> None:
    legal = tmp_path / "legal"
    _write_run(legal, status="failed", epochs=[1, 2], checkpoints=[1, 2])
    assert train_launcher._resume_action(
        legal, fold=2, total_epochs=4, training_identity=TRAINING_IDENTITY
    ) == ("resume", legal / "checkpoint_epoch_00002.pt")

class _FakeQueue:
    def __init__(self) -> None:
        self.items: list[tuple[str, str | None, str | None]] = []
        self.closed = False

    def put(self, value: tuple[str, str | None, str | None]) -> None:
        self.items.append(value)

    def get_nowait(self) -> tuple[str, str | None, str | None]:
        if not self.items:
            raise train_launcher.queue.Empty
        return self.items.pop(0)

    def close(self) -> None:
        self.closed = True

    def join_thread(self) -> None:
        return

class _FakeProcess:
    created: list["_FakeProcess"] = []

    def __init__(self, *, target, args, name: str) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.sentinel = object()
        self.exitcode: int | None = None
        self.created.append(self)

    def start(self) -> None:
        try:
            self.target(*self.args)
        except SystemExit as error:
            self.exitcode = int(error.code)
        else:
            self.exitcode = 0

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def is_alive(self) -> bool:
        return self.exitcode is None

class _FakeContext:
    Process = _FakeProcess

    @staticmethod
    def Queue() -> _FakeQueue:
        return _FakeQueue()

def test_scheduler_binds_slots_for_successful_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeProcess.created = []
    calls: list[tuple[int, str | None, bool]] = []

    def worker(config, fold, output, resume, device, progress, result_queue):
        del config, output, resume
        calls.append((fold, device, progress))
        result_queue.put(("completed", None, None))

    monkeypatch.setattr(train_launcher.multiprocessing, "get_context", lambda mode: _FakeContext())
    monkeypatch.setattr("multiprocessing.connection.wait", lambda sentinels: sentinels)
    monkeypatch.setattr(train_launcher, "_worker_entry", worker)
    results = train_launcher._run_schedule(
        config_path="config.yaml",
        folds=(1, 2, 3),
        output_root="outputs/test",
        resume=False,
        max_parallel=2,
        devices=("cuda:0", "cuda:1"),
    )
    assert results == {1: "completed", 2: "completed", 3: "completed"}
    assert calls == [
        (1, "cuda:0", True),
        (2, "cuda:1", False),
        (3, "cuda:0", False),
    ]

# --- Evaluation launcher contract ---

EVALUATION_IDENTITY = semantic_identity("stage3.evaluation", {"contract_version": 1})

class _Progress:
    class _Status:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    def status(self, message: str) -> _Status:
        del message
        return self._Status()

class _Run:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.completed: dict[str, Any] | None = None
        self.failed = False

    def complete(self, result: dict[str, Any]) -> None:
        self.completed = result

    def fail(self) -> None:
        self.failed = True

def test_single_validation_fold_uses_fold_directory_and_run_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs: list[_Run] = []
    open_calls: list[dict[str, Any]] = []
    evaluate_calls: list[dict[str, Any]] = []

    def open_run(**kwargs: Any) -> _Run:
        open_calls.append(kwargs)
        run = _Run(Path(f"/repo/{kwargs['output']}"))
        runs.append(run)
        return run

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        evaluate_calls.append(kwargs)
        return {"split": "valid"}

    monkeypatch.setattr(evaluate_launcher, "open_run_directory", open_run)
    evaluate_launcher._run_fold(
        config=Stage3Config(),
        config_path="base.yaml",
        checkpoint_dir=evaluate_launcher.ROOT / "train",
        output_root="evaluate",
        fold=3,
        checkpoint_epoch=10,
        tasks=["task/a"],
        study_id="study-a",
        progress=_Progress(),
        resolve_identity=lambda *args, **kwargs: EVALUATION_IDENTITY,
        evaluate_checkpoints=evaluate,
    )
    assert open_calls[0]["output"] == Path("evaluate/fold3")
    assert open_calls[0]["details"]["reporting_study_id"] == "study-a"
    assert evaluate_calls[0]["fold"] == 3
    assert evaluate_calls[0]["predictions_dir"] == Path(
        "/repo/evaluate/fold3/predictions"
    )
    assert runs[0].completed == {"split": "valid"}
    assert runs[0].failed is False

def test_test_path_remains_one_root_ensemble_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    open_calls: list[dict[str, Any]] = []
    evaluate_calls: list[dict[str, Any]] = []
    run = _Run(Path("/repo/evaluate_test"))

    def open_run(**kwargs: Any) -> _Run:
        open_calls.append(kwargs)
        return run

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        evaluate_calls.append(kwargs)
        return {"split": "test"}

    monkeypatch.setattr(evaluate_launcher, "open_run_directory", open_run)
    args = SimpleNamespace(
        config="base.yaml",
        output="evaluate_test",
        checkpoint_epoch=100,
        tasks=None,
        study_id=None,
    )
    evaluate_launcher._run_test(
        args=args,
        config=Stage3Config(),
        checkpoint_dir=evaluate_launcher.ROOT / "train",
        progress=_Progress(),
        resolve_identity=lambda *args, **kwargs: EVALUATION_IDENTITY,
        evaluate_checkpoints=evaluate,
    )
    assert open_calls[0]["output"] == "evaluate_test"
    assert evaluate_calls == [
        {
            "split": "test",
            "ensemble_folds": True,
            "checkpoint_epoch": 100,
            "task_subset": None,
            "fold": None,
            "predictions_dir": Path("/repo/evaluate_test/predictions"),
            "reporting_study_id": None,
            "expected_evaluation_identity": EVALUATION_IDENTITY,
        }
    ]
    assert run.completed == {"split": "test"}
