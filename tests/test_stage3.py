from __future__ import annotations

import csv

import json

import math

import shutil

from dataclasses import replace

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
    raw_task_steps,
    resolve_batch_allocation,
    resolve_raw_batch_allocation,
    resolve_task_registry,
    shuffled_epoch_indices,
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
    _clip_joint_gradients,
    _load_plugin,
    checkpoint_epochs,
    compute_task_gradient,
    resolve_stage3_training_identity,
    run_stage3_training,
)

from stage3.identity import build_stage3_training_identity, metadata_identity

from dataclasses import replace


from common.identity import semantic_identity

from stage3.capacity import refined_validation_summary, summarize_capacity_manifest

from stage3.config import load_stage3_config

import scripts.stage3.train as train_launcher

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
    assert config.training.sampling_mode == "virtual"
    assert config.training.joint_gradient_clip_mode == "global"
    assert "sampling_mode" not in config.to_dict()["training"]
    assert "joint_gradient_clip_mode" not in config.to_dict()["training"]
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
    assert ablation.training.sampling_mode == "raw"
    assert ablation.training.joint_gradient_clip_mode == "ownership"

    v2 = load_stage3_config("configs/v2/stage3/base.yaml")
    assert v2.training.sampling_mode == "raw"
    assert v2.training.joint_gradient_clip_mode == "ownership"
    assert "virtual_min_size" not in v2.to_dict()["training"]


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
        assert config.training.sampling_mode == "raw"
        assert config.training.joint_gradient_clip_mode == "ownership"
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

    changed_sampling = replace(
        tiny_prepared,
        training=replace(
            tiny_prepared.training,
            sampling_mode="raw",
        ),
    )
    assert resolve_stage3_training_identity(changed_sampling, 1) != (
        resolve_stage3_training_identity(tiny_prepared, 1)
    )


def test_raw_sampling_uses_every_index_once_without_padding() -> None:
    counts = {"tiny": 10, "medium": 400, "large": 2000}
    allocation = resolve_raw_batch_allocation(counts, 30)
    task_steps = raw_task_steps(counts, allocation)
    steps = max(task_steps.values())
    sequences = {
        task: shuffled_epoch_indices(
            count, seed=42, epoch=1, task_id=task
        )
        for task, count in counts.items()
    }

    assert sum(allocation.values()) == 30
    assert all(1 <= allocation[task] <= counts[task] for task in counts)
    for task, count in counts.items():
        batches = [
            sequences[task][
                step * allocation[task] : (step + 1) * allocation[task]
            ]
            for step in range(steps)
        ]
        observed = torch.cat([batch for batch in batches if len(batch)])
        assert len(observed) == count
        assert sorted(observed.tolist()) == list(range(count))
        assert sum(bool(len(batch)) for batch in batches) == task_steps[task]
    assert all(
        sum(
            len(
                sequences[task][
                    step * allocation[task] : (step + 1) * allocation[task]
                ]
            )
            for task in counts
        )
        <= 30
        for step in range(steps)
    )

    legacy = resolve_batch_allocation(counts, 30, 1000)
    legacy_steps = composite_steps_per_epoch(counts, legacy, 1000)
    assert legacy_steps * legacy["tiny"] > counts["tiny"]


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


def test_joint_gradient_clipping_isolated_by_owner(
    tiny_prepared: Stage3Config,
) -> None:
    registry = resolve_task_registry(tiny_prepared)
    model = Stage3SparseModel(tiny_prepared.model, registry, 4)
    requested_norms = {
        "GLOBAL": 0.25,
        "GROUP:g1": 0.50,
        "GROUP:g2": 0.75,
        "PRIVATE:experiment/a": 10.0,
        "PRIVATE:experiment/b": 20.0,
        "PRIVATE:experiment/c": 30.0,
    }
    ownership = model.parameter_ownership()
    for owner in sorted(set(ownership.values())):
        parameters = model.parameters_for_owner(owner)
        element_count = sum(parameter.numel() for parameter in parameters)
        value = requested_norms[owner.label] / math.sqrt(element_count)
        for parameter in parameters:
            parameter.grad = torch.full_like(parameter, value)

    pre, post, owner_pre, owner_post = _clip_joint_gradients(
        model, 1.0, "ownership"
    )

    assert pre > 30.0
    assert post == pytest.approx(
        math.sqrt(0.25**2 + 0.50**2 + 0.75**2 + 3.0), rel=1e-5
    )
    assert owner_pre == pytest.approx(requested_norms, rel=1e-5)
    assert owner_post["GLOBAL"] == pytest.approx(0.25, rel=1e-5)
    assert owner_post["GROUP:g1"] == pytest.approx(0.50, rel=1e-5)
    assert owner_post["GROUP:g2"] == pytest.approx(0.75, rel=1e-5)
    assert owner_post["PRIVATE:experiment/a"] == pytest.approx(1.0, rel=1e-5)
    assert owner_post["PRIVATE:experiment/b"] == pytest.approx(1.0, rel=1e-5)
    assert owner_post["PRIVATE:experiment/c"] == pytest.approx(1.0, rel=1e-5)


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


def test_pcgrad_accepts_only_tasks_present_in_raw_step(
    tiny_prepared: Stage3Config,
) -> None:
    registry = resolve_task_registry(tiny_prepared)
    model = Stage3SparseModel(tiny_prepared.model, registry, 4)
    task = "experiment/a"
    owners = (GLOBAL, group_owner(registry[task].meta_group), private_owner(task))
    gradients = {
        task: {
            parameter: torch.ones_like(parameter, dtype=torch.float32)
            for owner in owners
            for parameter in model.parameters_for_owner(owner)
        }
    }

    result = hierarchical_pcgrad(
        model,
        gradients,
        registry,
        {"g1": 1.0, "g2": 1.0},
        __import__("random").Random(3),
    )

    assert set(result.private_norms) == {task}
    absent_private = {
        parameter
        for absent in ("experiment/b", "experiment/c")
        for parameter in model.parameters_for_owner(private_owner(absent))
    }
    assert absent_private.isdisjoint(result.gradients)


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
    tiny_prepared = replace(
        tiny_prepared,
        training=replace(
            tiny_prepared.training,
            sampling_mode="raw",
            joint_gradient_clip_mode="ownership",
        ),
    )
    continuous = tiny_prepared.data.artifacts_dir.parent / "continuous"
    rows = run_stage3_training(tiny_prepared, 1, output_dir=continuous)
    assert [row["epoch"] for row in rows] == [1, 2]
    assert sorted(path.name for path in continuous.glob("checkpoint_*.pt")) == [
        "checkpoint_epoch_00001.pt", "checkpoint_epoch_00002.pt"
    ]
    assert (continuous / "taskwise_refined.pt").is_file()
    assert (continuous / "taskwise_refinement.json").is_file()
    plan = json.loads((continuous / "resolved_training_plan.json").read_text())
    assert plan["data"]["sampling"] == "raw_without_replacement_v1"
    assert plan["data"]["epoch_exposures"] == plan["data"]["N_t"]
    assert not {
        "N_prime_t", "padded_sizes", "replication_ratios"
    } & plan["data"].keys()
    assert plan["math"]["joint_gradient_clip_mode"] == "ownership"
    joint_diagnostics = json.loads(
        (continuous / "diagnostics.jsonl").read_text().splitlines()[0]
    )
    assert joint_diagnostics["clip_owner_pre_norms"]
    assert joint_diagnostics["clip_owner_post_norms"]
    assert all(
        norm <= tiny_prepared.training.max_grad_norm + 1e-5
        for norm in joint_diagnostics["clip_owner_post_norms"].values()
    )
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
    final_checkpoint = torch.load(
        continuous / "checkpoint_epoch_00002.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert final_checkpoint["refinement"]["task_updates"] == {
        task: plan["data"]["task_steps"][task]
        for task in plan["active_tasks"]
    }
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


@pytest.mark.parametrize("scale", ("s", "base", "l", "xl"))
def test_capacity_formal_configs_remain_loadable(scale: str) -> None:
    config = load_stage3_config(f"configs/experiments_v1/stage3/formal/{scale}.yaml")
    assert config.training.seed == 42
    assert config.training.epochs == 100


def test_capacity_probe_report_remains_supported(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_metrics(first, [0.4] * 20)
    _write_metrics(second, [0.2] * 20)
    manifest = tmp_path / "probe.yaml"
    manifest.write_text(
        __import__("yaml").safe_dump(
            {
                "schema_version": 2,
                "kind": "probe",
                "expected_epochs": 20,
                "candidates": [
                    {
                        "id": "base-r4",
                        "scale": "Base",
                        "recipe": "r4",
                        "folds": {1: str(first)},
                    },
                    {
                        "id": "base-r6",
                        "scale": "Base",
                        "recipe": "r6",
                        "folds": {1: str(second)},
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    result = summarize_capacity_manifest(manifest)
    assert [row["id"] for row in result["ranking"]] == ["base-r6", "base-r4"]
    assert result["scale_winners"][0]["id"] == "base-r6"


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
