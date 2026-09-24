from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from rdkit import Chem
import torch

from common.descriptor_preprocessing import FeaturePreprocessor
from common.identity import semantic_identity, tensor_state_hash
from common.io import sha256_file
from common.training import canonical_json_sha256, seed_everything
import scripts.stage3.evaluate as evaluate_launcher
import scripts.stage3.full_finetune as full_finetune_launcher
import scripts.stage3.transfer as transfer_launcher
import scripts.stage3.train as train_launcher
from stage1.descriptors import calculate_descriptors, rdkit_descriptor_names
from stage1.features import ROLE_TO_ID
from stage2.model import ObjectEncoder
from stage3.capacity import refined_validation_summary, summarize_capacity_manifest
from stage3.config import (
    BASE_GROUP_TASKS,
    Stage3Config,
    Stage3DataConfig,
    Stage3GlobalBudgetConfig,
    Stage3GroupConfig,
    Stage3InitializationConfig,
    Stage3ModelConfig,
    Stage3OwnerBudgetConfig,
    Stage3PreparationConfig,
    Stage3PrivateClassConfig,
    Stage3RepresentationConfig,
    Stage3TaskConfig,
    Stage3ThreePhaseConfig,
    Stage3TrainingConfig,
    Stage3KnowledgeGroupConfig,
    Stage3TransferKnowledgeConfig,
    effective_training_seed,
    load_stage3_config,
    stage3_config_from_dict,
)
from stage3.data import (
    ResolvedTaskSpec,
    Stage3TaskDataset,
    Stage3RepresentationStore,
    composite_steps_per_epoch,
    raw_task_steps,
    resolve_batch_allocation,
    resolve_raw_batch_allocation,
    resolve_task_registry,
    shuffled_epoch_indices,
    source_path,
)
from stage3.evaluate import _load_model, evaluate_checkpoints
from stage3.identity import build_stage3_prepared_identity, build_stage3_training_identity, metadata_identity
from stage3.model import (
    GLOBAL,
    Stage3SparseModel,
    group_owner,
    private_owner,
    summarize_task_gate_observations,
    task_gate_observations,
)
from stage3.transfer_knowledge import (
    KNOWLEDGE_KIND, KNOWLEDGE_VERSION, SOURCES, TransferKnowledgeBank,
)
from stage3.gradient_assembly import assemble_owner_gradients
from stage3.prepare import load_prepared_stage3, prepare_stage3
from stage3.three_phase import (
    _OwnerScheduler,
    _lr_factor as _three_phase_lr_factor,
    _model_state as _three_phase_model_state,
    _optimizer as _three_phase_optimizer,
    _owner_hash as _three_phase_owner_hash,
    _owner_state as _three_phase_owner_state,
    _stitch_owner_deltas,
)
from stage3.train import (
    _clip_joint_gradients,
    build_resolved_training_plan,
    checkpoint_epochs,
    compute_task_gradient,
    resolve_stage3_training_identity,
    run_stage3_training,
)
from ablations.stage2_stage3_transfer.config import (
    Stage2TransferConfig,
    Stage3TransferConfig,
    TransferExperimentConfig,
    load_transfer_config,
    transfer_config_from_dict,
)
from ablations.stage2_stage3_transfer.summary import (
    summarize_transfer_matrix,
    transfer_gain,
)
from ablations.stage2_stage3_transfer.stage3 import (
    MODEL_KIND as TRANSFER_MODEL_KIND,
    MODEL_VERSION as TRANSFER_MODEL_VERSION,
    encode_transfer_objects,
    load_representation_bank,
    prepare_representation_bank,
    train_transfer_job,
    transfer_training_seed,
)
from ablations.stage3_full_finetune.representation import (
    FinetuneRecipe, FinetuneStage3Model, LiveRepresentationStore, STAGE1_OWNER, STAGE2_OWNER,
    load_config as load_full_finetune_config, object_keys as finetune_object_keys,
    prepare_features as prepare_finetune_features,
    load_features as load_finetune_features,
)
from ablations.stage3_full_finetune.train import (
    _encoder_owner_recipe, resolved_plan, validate_initial_representation,
)
from ablations.stage3_full_finetune.evaluate import compare_historical_base, evaluate_finetuned
from stage3.three_phase import run_three_phase_training


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
    unique_systems: int = 2,
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
        "unique_systems": unique_systems,
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


def _tiny_three_phase(config: Stage3Config) -> Stage3Config:
    private_class = Stage3PrivateClassConfig(
        width_ratio=1.0,
        phase1=Stage3OwnerBudgetConfig(lr=8.0e-5, epochs=1),
        phase2_epochs=1,
        phase3_epochs=1,
    )
    return replace(
        config,
        groups={
            "g1": Stage3GroupConfig(
                experts=2,
                expert_hidden_ratio=1.5,
                phase1=Stage3OwnerBudgetConfig(lr=1.25e-4, epochs=1),
                phase2=Stage3OwnerBudgetConfig(lr=6.25e-5, epochs=2),
            ),
            "g2": Stage3GroupConfig(
                experts=1,
                expert_hidden_ratio=0.5,
                phase1=Stage3OwnerBudgetConfig(lr=1.25e-4, epochs=1),
                phase2=Stage3OwnerBudgetConfig(lr=6.25e-5, epochs=1),
            ),
        },
        tasks={
            task: replace(
                spec,
                unique_systems=2,
                size_class="medium",
                phase3_private_epochs=1,
                model_overrides={},
            )
            for task, spec in config.tasks.items()
        },
        training=replace(
            config.training,
            sampling_mode="raw",
            joint_gradient_clip_mode="ownership",
            schedule_mode="three_phase",
            three_phase=Stage3ThreePhaseConfig(
                global_scope=Stage3GlobalBudgetConfig(
                    lr=2.5e-4, epochs=2, warmup_ratio=0.05,
                    min_lr_ratio=0.1,
                ),
                private_classes={
                    name: private_class
                    for name in ("tiny", "small", "medium", "large")
                },
                phase1_min_lr_ratio=0.5,
                phase2_min_lr_ratio=0.5,
                phase3_min_lr_ratio=0.2,
            ),
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
    assert config.training.schedule_mode == "legacy_joint_refinement"
    assert all(group.experts is None for group in config.groups.values())
    assert all(task.phase3_private_epochs is None for task in config.tasks.values())
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
    assert len(ablation.enabled_task_ids) == 20
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
    assert ablation.training.schedule_mode == "three_phase"

    no_stage1 = load_stage3_config(
        "configs/ablations/no_stage1_rdkit_stage3.yaml"
    )
    assert no_stage1.training.schedule_mode == "three_phase"

    v2 = load_stage3_config("configs/v2/stage3/base.yaml")
    assert v2.model == config.model
    assert all(group.experts is not None for group in v2.groups.values())
    assert all(v2.resolved_private_recipe(task).phase3_epochs >= 0 for task in v2.tasks)
    assert v2.training.sampling_mode == "raw"
    assert v2.training.joint_gradient_clip_mode == "ownership"
    assert v2.training.schedule_mode == "three_phase"
    assert "virtual_min_size" not in v2.to_dict()["training"]
    for path in (
        "configs/v2/stage3/splits/random.yaml",
        "configs/v2/stage3/splits/system.yaml",
        "configs/v2/stage3/splits/individual.yaml",
        "configs/ablations/stage1_stage2_rdkit_home.yaml",
        "configs/ablations/no_stage1_rdkit_stage3.yaml",
    ):
        active = load_stage3_config(path)
        assert active.groups == v2.groups
        assert active.tasks == v2.tasks
        assert active.training.three_phase == v2.training.three_phase


def test_knowledge_graph_grouping_config_is_yaml_driven() -> None:
    base = load_stage3_config("configs/v2/stage3/base.yaml")
    candidate = load_stage3_config("configs/v2/stage3/base1.yaml")
    expected = {
        "transport_dynamics": {
            "experiment/electrical_conductivity",
            "experiment/self_diffusion_coefficient",
            "experiment/viscosity",
        },
        "thermophysical_interfacial_response": {
            "experiment/density",
            "experiment/heat_capacity",
            "experiment/speed_of_sound",
            "experiment/surface_tension",
            "experiment/thermal_conductivity",
            "experiment/refractive_index",
            "experiment/dynamic_relative_permittivity",
            "experiment/x_co2",
        },
        "phase_stability": {
            "experiment/equilibrium_pressure",
            "experiment/glass_transition_temperature",
            "experiment/melting_point",
            "experiment/thermal_decomposition_temperature",
        },
        "solvation_transfer": {
            "experiment/solvation",
            "experiment/transfer",
            "experiment/transfer_organic",
        },
        "biological": {"experiment/pec50"},
        "static_dielectric": {"experiment/static_relative_permittivity"},
    }
    actual = {
        group: {
            task_id
            for task_id, task in candidate.tasks.items()
            if task.meta_group == group
        }
        for group in candidate.groups
    }
    assert actual == expected
    assert set().union(*actual.values()) == set(base.tasks)
    assert sum(map(len, actual.values())) == len(base.tasks) == 20

    inherited_groups = {
        "transport_dynamics": "transport",
        "thermophysical_interfacial_response": "thermophysical",
        "phase_stability": "phase_stability",
        "solvation_transfer": "solvation",
        "biological": "biological",
        "static_dielectric": "dielectric_optical",
    }
    for group, source in inherited_groups.items():
        assert candidate.groups[group] == base.groups[source]
    assert candidate.model == base.model
    assert candidate.training == base.training
    assert candidate.data == base.data
    assert candidate.preparation == base.preparation
    assert candidate.initialization == base.initialization
    for task_id, task in candidate.tasks.items():
        assert replace(task, meta_group=base.tasks[task_id].meta_group) == base.tasks[task_id]

    base_registry = resolve_task_registry(base)
    candidate_registry = resolve_task_registry(candidate)
    assert {
        task_id: spec.prepared_dict() for task_id, spec in base_registry.items()
    } == {
        task_id: spec.prepared_dict()
        for task_id, spec in candidate_registry.items()
    }
    assert candidate_registry != base_registry

    model = Stage3SparseModel(
        candidate.model,
        candidate_registry,
        8,
        group_configs=candidate.groups,
        task_configs=candidate.tasks,
        task_private_recipes={
            task_id: candidate.resolved_private_recipe(task_id)
            for task_id in candidate.tasks
        },
    )
    assert set(model.groups) == set(expected)
    assert model.task_gates[
        "experiment__dynamic_relative_permittivity"
    ].out_features == 5
    assert model.task_gates[
        "experiment__static_relative_permittivity"
    ].out_features == 4
    assert model.task_gates["experiment__solvation"].out_features == 6
    group_owners = {
        owner
        for owner in model.ownership_manifest().values()
        if owner.startswith("GROUP:")
    }
    assert group_owners == {f"GROUP:{group}" for group in expected}


def test_knowledge_graph_budget_candidates_preserve_experiment_boundaries() -> None:
    base = load_stage3_config("configs/v2/stage3/base1.yaml")
    group = "thermophysical_interfacial_response"
    static = "experiment/static_relative_permittivity"
    base_registry = resolve_task_registry(base)
    identities = set()
    baseline_shapes = None
    baseline_count = None
    prepared_hash = None
    for index in range(6):
        config = load_stage3_config(
            f"configs/v2/stage3/base1{'_' + str(index) if index else ''}.yaml"
        )
        expected = base.to_dict()
        if index in (1, 5):
            expected["groups"][group]["experts"] = 3
        if index in (2, 5):
            expected["groups"][group]["phase1"]["epochs"] = 15
        if index in (3, 5):
            expected["groups"][group]["phase1"]["lr"] = 2e-4
            expected["groups"][group]["phase2"]["lr"] = 1e-4
        if index in (4, 5):
            expected["groups"]["static_dielectric"]["phase1"]["lr"] = 5e-5
            expected["groups"]["static_dielectric"]["phase2"] = {
                "lr": 2.5e-5, "epochs": 1,
            }
            expected["tasks"][static]["phase1_private_lr"] = 2e-5
        assert config.to_dict() == expected
        assert stage3_config_from_dict(config.to_dict()).to_dict() == expected
        registry = resolve_task_registry(config)
        assert registry == base_registry
        # Hold source contents, embeddings and normalization fixed; exercise the
        # actual prepared identity builder without production artifact I/O.
        with patch("stage3.identity._source_content", return_value={}):
            prepared_identity = build_stage3_prepared_identity(
                config, registry, [], {}, {"hash": "fixed-stage2-encoder"}
            )
        if index == 0:
            prepared_hash = prepared_identity["hash"]
        assert prepared_identity["hash"] == prepared_hash
        model = Stage3SparseModel(
            config.model, registry, 8, group_configs=config.groups,
            task_configs=config.tasks,
            task_private_recipes={
                task: config.resolved_private_recipe(task) for task in registry
            },
        )
        shapes = {name: tuple(t.shape) for name, t in model.state_dict().items()}
        count = sum(p.numel() for p in model.parameters())
        if index == 0:
            baseline_shapes, baseline_count = shapes, count
        if index in (1, 5):
            assert count > baseline_count
            changed = {
                name for name in shapes.keys() | baseline_shapes.keys()
                if shapes.get(name) != baseline_shapes.get(name)
            }
            allowed = {
                name for name in shapes
                if name.startswith((f"l1_group_experts.{group}.2.",
                                    f"l2_group_experts.{group}.2.",
                                    f"l1_group_gates.{group}."))
            }
            for task, spec in registry.items():
                if spec.meta_group == group:
                    key = task.replace("/", "__")
                    allowed.update({f"task_gates.{key}.weight", f"task_gates.{key}.bias"})
                    assert model.task_gates[key].out_features == 6
            assert changed == allowed
        else:
            assert shapes == baseline_shapes
            assert count == baseline_count
        phases = config.training.three_phase
        assert phases is not None
        for task, spec in config.tasks.items():
            budget = config.groups[spec.meta_group]
            private = config.resolved_private_recipe(task)
            assert phases.global_scope.lr > budget.phase1.lr > private.phase1_lr
            assert budget.phase2.lr == budget.phase1.lr * phases.phase1_min_lr_ratio
        prepared = {"metadata": {
            "kind": "ilume_stage3_sparse_data",
            "semantic": {"identities": {
                "prepared": prepared_identity,
                "stage2_encoder": {"hash": "fixed-stage2-encoder"},
            }},
        }}
        plan = build_resolved_training_plan(
            config, 1, model, {task: range(10) for task in registry},
            tuple(registry), prepared, {}, {},
        )
        identities.add(build_stage3_training_identity(plan)["hash"])
        assert plan["phases"]["phase2"]["branches"][group]["epochs"] == 4
        if index in (4, 5):
            recipe = config.resolved_private_recipe(static)
            assert (recipe.phase1_lr, recipe.phase2_lr, recipe.phase3_lr) == (
                2e-5, 1e-5, 5e-6,
            )
            assert (recipe.phase1_epochs, recipe.phase2_epochs, recipe.phase3_epochs) == (4, 1, 0)
    assert len(identities) == 6


def test_knowledge_graph_targeted_candidates_match_base1_5_contract() -> None:
    anchor = load_stage3_config("configs/v2/stage3/base1_5.yaml")
    group = "thermophysical_interfacial_response"
    static_group = "static_dielectric"
    speed = "experiment/speed_of_sound"
    diffusion = "experiment/self_diffusion_coefficient"
    registry = resolve_task_registry(anchor)
    anchor_model = Stage3SparseModel(
        anchor.model, registry, 8, group_configs=anchor.groups,
        task_configs=anchor.tasks,
        task_private_recipes={
            task: anchor.resolved_private_recipe(task) for task in registry
        },
    )
    anchor_shapes = {
        name: tuple(tensor.shape)
        for name, tensor in anchor_model.state_dict().items()
    }
    anchor_count = sum(parameter.numel() for parameter in anchor_model.parameters())
    with patch("stage3.identity._source_content", return_value={}):
        anchor_prepared = build_stage3_prepared_identity(
            anchor, registry, [], {}, {"hash": "fixed-stage2-encoder"}
        )
    training_identities = set()

    for index in range(1, 6):
        config = load_stage3_config(f"configs/v2/stage3/base2_{index}.yaml")
        expected = anchor.to_dict()
        if index in (1, 5):
            expected["groups"][static_group]["expert_hidden_ratio"] = 0.25
        if index in (2, 5):
            expected["tasks"][speed]["phase3_private_epochs"] = 2
        if index in (3, 5):
            expected["tasks"][diffusion]["phase3_private_epochs"] = 2
        if index in (4, 5):
            expected["groups"][group]["experts"] = 2
        assert config.to_dict() == expected
        assert stage3_config_from_dict(config.to_dict()).to_dict() == expected
        candidate_registry = resolve_task_registry(config)
        assert candidate_registry == registry
        with patch("stage3.identity._source_content", return_value={}):
            prepared_identity = build_stage3_prepared_identity(
                config, candidate_registry, [], {},
                {"hash": "fixed-stage2-encoder"},
            )
        assert prepared_identity["hash"] == anchor_prepared["hash"]

        model = Stage3SparseModel(
            config.model, candidate_registry, 8, group_configs=config.groups,
            task_configs=config.tasks,
            task_private_recipes={
                task: config.resolved_private_recipe(task)
                for task in candidate_registry
            },
        )
        shapes = {
            name: tuple(tensor.shape)
            for name, tensor in model.state_dict().items()
        }
        changed = {
            name for name in shapes.keys() | anchor_shapes.keys()
            if shapes.get(name) != anchor_shapes.get(name)
        }
        if index in (2, 3):
            assert not changed
            assert sum(p.numel() for p in model.parameters()) == anchor_count
        if index in (1, 5):
            static_changes = {
                name for name in changed
                if name.startswith((
                    f"l1_group_experts.{static_group}.",
                    f"l2_group_experts.{static_group}.",
                ))
            }
            assert len(static_changes) == 6
            if index == 1:
                assert changed == static_changes
            assert sum(p.numel() for p in model.parameters()) < anchor_count
        if index in (4, 5):
            large_group_changes = changed - {
                name for name in changed
                if name.startswith((
                    f"l1_group_experts.{static_group}.",
                    f"l2_group_experts.{static_group}.",
                ))
            }
            assert large_group_changes
            assert all(
                name.startswith((
                    f"l1_group_experts.{group}.2.",
                    f"l2_group_experts.{group}.2.",
                    f"l1_group_gates.{group}.",
                    "task_gates.experiment__",
                ))
                for name in large_group_changes
            )
            for task, spec in candidate_registry.items():
                if spec.meta_group == group:
                    assert model.task_gates[task.replace("/", "__")].out_features == 5
            assert sum(p.numel() for p in model.parameters()) < anchor_count
        if index == 1:
            static_task = "experiment/static_relative_permittivity"
            assert (
                model.task_gates[static_task.replace("/", "__")].out_features
                == anchor_model.task_gates[static_task.replace("/", "__")].out_features
            )

        prepared = {"metadata": {
            "kind": "ilume_stage3_sparse_data",
            "semantic": {"identities": {
                "prepared": prepared_identity,
                "stage2_encoder": {"hash": "fixed-stage2-encoder"},
            }},
        }}
        plan = build_resolved_training_plan(
            config, 1, model,
            {task: range(10) for task in candidate_registry},
            tuple(candidate_registry), prepared, {}, {},
        )
        training_identities.add(build_stage3_training_identity(plan)["hash"])
        assert config.groups[group].phase1.epochs == 15
        assert config.groups[group].phase1.lr == 2e-4
        assert config.groups[group].phase2.epochs == 4
        assert config.groups[group].phase2.lr == 1e-4
        phases = config.training.three_phase
        assert phases is not None
        for task, task_config in config.tasks.items():
            budget = config.groups[task_config.meta_group]
            private = config.resolved_private_recipe(task)
            assert phases.global_scope.lr > budget.phase1.lr > private.phase1_lr
            assert budget.phase2.lr == budget.phase1.lr * phases.phase1_min_lr_ratio
    assert len(training_identities) == 5


def test_v2_native_split_configs_match_materialized_task_subsets() -> None:
    expected = {
        "system": ({"il", "il_solute", "solute_solvent"}, 20),
        "random": ({"random"}, 20),
        "individual": ({"cation", "solvent"}, 20),
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
        assert config.training.schedule_mode == "three_phase"
        if name == "system":
            assert {spec.system_type for spec in enabled.values()} == strategies
        elif name == "individual":
            assert sum(spec.split_strategy == "cation" for spec in enabled.values()) == 19
            assert enabled["experiment/transfer_organic"].split_strategy == "solvent"
        for spec in enabled.values():
            for fold in range(1, 6):
                assert source_path(config, spec, fold).is_file()


def test_three_phase_config_and_task_specific_gate_contract() -> None:
    config = load_stage3_config("configs/v2/stage3/base.yaml")
    assert config.training.schedule_mode == "three_phase"
    assert config.training.three_phase is not None
    serialized = config.to_dict()
    assert "epochs" not in serialized["training"]
    assert "refinement_ratio" not in serialized["training"]
    assert "group_experts" not in serialized["model"]
    mixed = {**serialized, "training": {**serialized["training"], "epochs": 20}}
    with pytest.raises(ValueError, match="forbids legacy fields"):
        stage3_config_from_dict(mixed)

    with pytest.raises(ValueError, match="retired"):
        stage3_config_from_dict(
            {**serialized, "training": {"schedule_mode": "four_phase"}}
        )
    retired_task = {
        **serialized,
        "tasks": {
            **serialized["tasks"],
            "experiment/pec50": {
                **serialized["tasks"]["experiment/pec50"],
                "phase3_epochs": 3,
            },
        },
    }
    with pytest.raises(ValueError, match="phase3_epochs is retired"):
        stage3_config_from_dict(retired_task)

    task_a = Stage3TaskConfig(meta_group="g1")
    task_b = Stage3TaskConfig(meta_group="g2")
    tiny = Stage3Config(
        model=Stage3ModelConfig(global_experts=1, group_experts=9),
        groups={
            "g1": Stage3GroupConfig(experts=3, expert_hidden_ratio=1.0),
            "g2": Stage3GroupConfig(experts=1, expert_hidden_ratio=1.0),
        },
        tasks={"experiment/a": task_a, "experiment/b": task_b},
    )
    specs = {
        task: ResolvedTaskSpec(
            task_id=task,
            target_column="value",
            identity_columns=("cation", "anion"),
            condition_columns=(),
            system_type="il",
            materialized_path=task,
            split_strategy="il",
            cv_repeat=1,
            meta_group=spec.meta_group,
            partner_mode="none",
            primary_slots=("cation", "anion"),
            partner_slots=(),
            enabled=True,
            task_weight=1.0,
            catalog_schema_version=1,
            provenance={},
        )
        for task, spec in tiny.tasks.items()
    }
    flat_model_config = replace(tiny.model, l2_residual=False)
    model = Stage3SparseModel(
        flat_model_config,
        specs,
        4,
        group_configs=tiny.groups,
        task_configs=tiny.tasks,
    )
    assert model.task_gates["experiment__a"].out_features == 5
    assert model.task_gates["experiment__b"].out_features == 3

    model.eval()
    primary = torch.randn(4, 4)
    output = model("experiment/a", primary, torch.empty(4, 0))
    weights = output.diagnostics["task_gate"]
    candidates = torch.cat(
        (
            output.diagnostics["l2_global_candidates"],
            output.diagnostics["l2_group_candidates"],
            output.diagnostics["l2_private_candidates"],
        ),
        dim=1,
    )
    assert weights.shape == (4, 5)
    assert output.diagnostics["l2_group_candidates"].shape[1] == 3
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4))
    expected_mixed = (weights.unsqueeze(-1) * candidates).sum(dim=1)
    expected_prediction = model.towers["experiment__a"](expected_mixed)
    assert torch.equal(output.predictions, expected_prediction)


def test_transfer_knowledge_config_model_and_zero_gamma_flat_parity() -> None:
    base = load_stage3_config("configs/v2/stage3/base.yaml")
    variant = load_stage3_config("configs/ablations/stage3_transfer_knowledge.yaml")
    assert stage3_config_from_dict(variant.to_dict()) == variant
    assert replace(variant, transfer_knowledge=None) == base
    registry = resolve_task_registry(base)
    torch.manual_seed(17)
    flat = Stage3SparseModel(base.model, registry, 4, group_configs=base.groups,
                             task_configs=base.tasks)
    torch.manual_seed(17)
    knowledge = Stage3SparseModel(
        variant.model, registry, 4, group_configs=variant.groups,
        task_configs=variant.tasks, transfer_knowledge=variant.transfer_knowledge,
    )
    assert set(knowledge.state_dict()) - set(flat.state_dict()) == {
        name for name in knowledge.state_dict() if "knowledge_mixer" in name
    }
    assert all(torch.equal(value, knowledge.state_dict()[name])
               for name, value in flat.state_dict().items())
    ownership = knowledge.ownership_manifest()
    assert ownership["global_knowledge_mixer.gamma"] == "GLOBAL"
    assert ownership["group_knowledge_mixers.solvation.gamma"] == "GROUP:solvation"
    assert ownership["private_knowledge_mixers.experiment__density.gamma"] == "PRIVATE:experiment/density"
    knowledge.set_trainable_owners((GLOBAL,))
    assert knowledge.global_knowledge_mixer.gamma.requires_grad
    assert not knowledge.group_knowledge_mixers["solvation"].gamma.requires_grad
    assert not knowledge.private_knowledge_mixers["experiment__density"].gamma.requires_grad
    assert "simulation/transfer_organic" in knowledge.knowledge_sources("experiment/solvation")
    assert knowledge.knowledge_sources("experiment/x_co2") == variant.transfer_knowledge.global_sources
    flat.eval()
    knowledge.eval()
    primary = torch.randn(2, 4)
    for task in (
        "experiment/x_co2", "experiment/density",
        "experiment/dynamic_relative_permittivity", "experiment/solvation",
    ):
        conditions = torch.zeros(2, len(registry[task].condition_columns))
        deltas = {source: torch.randn_like(primary) for source in knowledge.knowledge_sources(task)}
        partner = primary if registry[task].partner_mode == "interaction" else None
        expected = flat(task, primary, conditions, partner_embedding=partner)
        actual = knowledge(
            task, primary, conditions, partner_embedding=partner,
            primary_knowledge=deltas,
            partner_knowledge=deltas if partner is not None else None,
        )
        assert torch.equal(expected.predictions, actual.predictions)
        assert torch.equal(expected.diagnostics["task_gate"], actual.diagnostics["task_gate"])
        assert task_gate_observations(expected.diagnostics).equal(task_gate_observations(actual.diagnostics))


def test_transfer_knowledge_group_task_mask_and_mixer_math() -> None:
    variant = load_stage3_config("configs/ablations/stage3_transfer_knowledge.yaml")
    registry = resolve_task_registry(variant)
    model = Stage3SparseModel(
        variant.model, registry, 4, group_configs=variant.groups,
        task_configs=variant.tasks, transfer_knowledge=variant.transfer_knowledge,
    ).eval()
    with torch.no_grad():
        model.group_knowledge_mixers["solvation"].gamma.fill_(1)
    anchor = torch.zeros(2, 4)
    conditions = lambda task: torch.zeros(2, len(registry[task].condition_columns))
    deltas = {source: torch.full_like(anchor, float(index + 1))
              for index, source in enumerate(SOURCES)}
    observed = []
    partners = []
    hook = model.l1_group_experts["solvation"][0].register_forward_pre_hook(
        lambda _module, args: observed.append(args[0].detach().clone())
    )
    partner_hook = model.interactions["solvation"].register_forward_pre_hook(
        lambda _module, args: partners.append(args[1].detach().clone())
    )
    model("experiment/x_co2", anchor, conditions("experiment/x_co2"),
          primary_knowledge=deltas)
    assert torch.equal(observed[-1], anchor)
    model("experiment/solvation", anchor, conditions("experiment/solvation"),
          partner_embedding=anchor, primary_knowledge=deltas, partner_knowledge=deltas)
    expected = sum(deltas[source] for source in variant.transfer_knowledge.group_sources["solvation"].sources) / 3
    assert torch.equal(observed[-1], expected)
    assert torch.equal(partners[-1], expected)
    hook.remove()
    partner_hook.remove()
    with torch.no_grad():
        model.private_knowledge_mixers["experiment__density"].gamma.fill_(1)
    density_inputs = []
    density_hook = model.l1_group_experts["thermophysical"][0].register_forward_pre_hook(
        lambda _module, args: density_inputs.append(args[0].detach().clone())
    )
    model("experiment/density", anchor, conditions("experiment/density"),
          primary_knowledge=deltas)
    assert len(density_inputs) == 2
    assert torch.equal(density_inputs[0], anchor)
    assert torch.equal(density_inputs[1], deltas["simulation/heat_of_vaporization"])
    density_hook.remove()


def test_transfer_knowledge_bank_rejects_wrong_order_and_corruption(tmp_path: Path) -> None:
    objects = {"objects": [{"topology": "molecule", "slots": [["a"]]},
                           {"topology": "molecule", "slots": [["b"]]}]}
    embeddings = {name: torch.full((2, 4), float(index), dtype=torch.float32)
                  for index, name in enumerate(("baseline", *SOURCES))}
    source_artifacts = {name: {"sha256": name} for name in ("baseline", *SOURCES)}
    object_hash = canonical_json_sha256(objects["objects"])
    tensor_hash = tensor_state_hash("stage3.transfer-knowledge-embeddings.v1", embeddings)
    identity = semantic_identity("stage3.transfer-knowledge", {
        "contract_version": KNOWLEDGE_VERSION, "prepared_identity": "prepared-hash",
        "object_list_hash": object_hash, "source_artifacts": source_artifacts,
        "tensor_hash": tensor_hash, "shape": [2, 4],
    })
    payload = {
        "kind": KNOWLEDGE_KIND, "format_version": KNOWLEDGE_VERSION,
        "identity": identity,
        "prepared_identity": "prepared-hash",
        "object_list_hash": object_hash,
        "tensor_hash": tensor_hash,
        "source_artifacts": source_artifacts,
        "embeddings": embeddings,
    }
    path = tmp_path / "knowledge_bank.pt"
    torch.save(payload, path)
    manifest = {name: value for name, value in payload.items() if name != "embeddings"}
    manifest["artifact_sha256"] = sha256_file(path)
    path.with_suffix(".json").write_text(json.dumps(manifest))
    bank = TransferKnowledgeBank(path, prepared_identity="prepared-hash", objects=objects, d_model=4)
    delta = bank.deltas(torch.tensor([1]), (SOURCES[0],))[SOURCES[0]]
    assert torch.equal(delta, torch.ones((1, 4)))
    with pytest.raises(ValueError, match="ObjectKey order"):
        TransferKnowledgeBank(path, prepared_identity="prepared-hash",
                              objects={"objects": list(reversed(objects["objects"]))}, d_model=4)
    with pytest.raises(ValueError, match="prepared identity"):
        TransferKnowledgeBank(path, prepared_identity="other", objects=objects, d_model=4)
    payload["embeddings"][SOURCES[0]][0, 0] = float("nan")
    torch.save(payload, path)
    manifest["artifact_sha256"] = sha256_file(path)
    path.with_suffix(".json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="malformed"):
        TransferKnowledgeBank(path, prepared_identity="prepared-hash", objects=objects, d_model=4)


def test_transfer_knowledge_train_evaluate_and_checkpoint_isolation(
    tiny_prepared: Stage3Config,
) -> None:
    prepared = load_prepared_stage3(tiny_prepared)
    objects = prepared["objects"]
    prepared_hash = metadata_identity(prepared["metadata"], "prepared", context="test")["hash"]
    path = tiny_prepared.data.artifacts_dir.parent / "knowledge.pt"
    count = len(objects["objects"])
    embeddings = {name: torch.full((count, 4), float(index) / 10)
                  for index, name in enumerate(("baseline", *SOURCES))}
    source_artifacts = {name: {"sha256": name} for name in embeddings}
    object_hash = canonical_json_sha256(objects["objects"])
    tensor_hash = tensor_state_hash("stage3.transfer-knowledge-embeddings.v1", embeddings)
    identity = semantic_identity("stage3.transfer-knowledge", {
        "contract_version": KNOWLEDGE_VERSION, "prepared_identity": prepared_hash,
        "object_list_hash": object_hash, "source_artifacts": source_artifacts,
        "tensor_hash": tensor_hash, "shape": [count, 4],
    })
    payload = {
        "kind": KNOWLEDGE_KIND, "format_version": KNOWLEDGE_VERSION,
        "identity": identity, "prepared_identity": prepared_hash,
        "object_list_hash": object_hash, "source_artifacts": source_artifacts,
        "tensor_hash": tensor_hash, "embeddings": embeddings,
    }
    torch.save(payload, path)
    manifest = {name: value for name, value in payload.items() if name != "embeddings"}
    manifest["artifact_sha256"] = sha256_file(path)
    path.with_suffix(".json").write_text(json.dumps(manifest))
    config = replace(
        _tiny_three_phase(tiny_prepared),
        transfer_knowledge=Stage3TransferKnowledgeConfig(
            bank=path, global_sources=(SOURCES[0],),
            group_sources={"g1": Stage3KnowledgeGroupConfig(
                sources=(SOURCES[1],), tasks=("experiment/a",)
            )},
            private_sources={"experiment/c": (SOURCES[2],)},
        ),
    )
    config.validate()
    output = path.parent / "knowledge-train"
    run_stage3_training(config, 1, output_dir=output)
    checkpoint = torch.load(output / "phase_1/checkpoint_epoch_00002.pt",
                            map_location="cpu", weights_only=False)
    final = torch.load(output / "three_phase_final.pt", map_location="cpu", weights_only=False)
    assert checkpoint["kind"] == "ilume_stage3_transfer_knowledge_three_phase_checkpoint"
    assert final["kind"] == "ilume_stage3_transfer_knowledge_three_phase_final"
    assert final["resolved_training_plan"]["transfer_knowledge"]["bank_identity"] == identity["hash"]
    evaluated = evaluate_checkpoints(config, output, split="valid", ensemble_folds=False,
                                     task_subset=("experiment/a",), fold=1)
    assert "gate_diagnostics" in evaluated
    with pytest.raises(ValueError, match="checkpoint mismatch: kind"):
        evaluate_checkpoints(_tiny_three_phase(tiny_prepared), output, split="valid",
                             ensemble_folds=False, task_subset=("experiment/a",), fold=1)


def test_three_phase_private_capacity_ratios_follow_size_class() -> None:
    config = load_stage3_config("configs/v2/stage3/base.yaml")
    fallback_task = "experiment/dynamic_relative_permittivity"
    fallback_config = replace(
        config,
        tasks={
            **config.tasks,
            fallback_task: replace(
                config.tasks[fallback_task],
                phase1_private_epochs=5,
                model_overrides={},
            ),
        },
    )
    fallback_config.validate()
    fallback = fallback_config.resolved_private_recipe(fallback_task)
    assert (
        fallback.private_hidden_ratio,
        fallback.tower_hidden_ratio,
        fallback.film_hidden_ratio,
    ) == (0.5, 0.5, 0.5)
    assert fallback.phase1_epochs == 5
    task_id = "experiment/static_relative_permittivity"
    task = config.tasks[task_id]
    assert task.size_class == "tiny"

    spec = resolve_task_registry(config)[task_id]
    model = Stage3SparseModel(
        config.model,
        {task_id: spec},
        1024,
        group_configs=config.groups,
        task_configs={task_id: task},
        task_private_recipes={task_id: config.resolved_private_recipe(task_id)},
    )
    key = task_id.replace("/", "__")
    recipe = model.resolved_capacity_recipe()["tasks"][task_id]
    assert recipe["private_hidden"] == 256
    assert recipe["tower_hidden"] == 256
    assert recipe["film_hidden"] == 256
    assert model.private_experts[key][0].layers[0].out_features == 256
    assert model.towers[key].layers[0].out_features == 256
    assert model.condition_films[key].network[0].out_features == 256
    assert model.task_gates[key].in_features == 2048

    dropout_task = "experiment/refractive_index"
    dropout_model = Stage3SparseModel(
        config.model,
        {dropout_task: resolve_task_registry(config)[dropout_task]},
        16,
        group_configs=config.groups,
        task_configs={dropout_task: config.tasks[dropout_task]},
        task_private_recipes={
            dropout_task: config.resolved_private_recipe(dropout_task)
        },
    )
    dropout_key = dropout_task.replace("/", "__")
    private_dropouts = [
        module.p
        for owner_module in (
            dropout_model.private_experts[dropout_key],
            dropout_model.towers[dropout_key],
            dropout_model.condition_films[dropout_key],
        )
        for module in owner_module.modules()
        if isinstance(module, torch.nn.Dropout)
    ]
    assert private_dropouts == [0.15, 0.15, 0.15]
    assert dropout_model.resolved_capacity_recipe()["tasks"][dropout_task][
        "private_dropout"
    ] == 0.15

def test_task_gate_diagnostics_partition_entropy_and_pooled_quantiles() -> None:
    task_gate = torch.tensor(
        (
            (0.10, 0.10, 0.10, 0.10, 0.10, 0.50),
            (1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6),
        ),
        dtype=torch.float64,
    )
    diagnostics = {
        "task_gate": task_gate,
        "l2_global_candidates": torch.empty((2, 2, 3)),
        "l2_group_candidates": torch.empty((2, 3, 3)),
        "l2_private_candidates": torch.empty((2, 1, 3)),
    }
    observations = task_gate_observations(diagnostics)
    assert observations[:, :3] == pytest.approx(
        torch.tensor(((0.2, 0.3, 0.5), (1 / 3, 0.5, 1 / 6)))
    )
    expected_entropy = torch.special.entr(task_gate).sum(dim=1) / math.log(6)
    assert observations[:, 3] == pytest.approx(expected_entropy)

    pooled = summarize_task_gate_observations(
        torch.cat((observations, observations.flip(0)))
    )
    assert pooled["mean_global_gate_weight"] == pytest.approx(4 / 15)
    assert pooled["mean_group_gate_weight"] == pytest.approx(0.4)
    assert pooled["mean_private_gate_weight"] == pytest.approx(1 / 3)
    assert pooled["task_gate_entropy"] == pytest.approx(expected_entropy.mean())
    assert pooled["private_gate_weight_p10"] == pytest.approx(1 / 6)
    assert pooled["private_gate_weight_p50"] == pytest.approx(1 / 3)
    assert pooled["private_gate_weight_p90"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("phase1_private_lr", 1.5e-4, "nominal LR ordering"),
        ("phase1_private_epochs", 16, "exceed GLOBAL budget"),
        ("phase2_private_epochs", -1, "non-negative integer"),
        ("phase3_private_epochs", -1, "non-negative integer"),
    ),
)
def test_three_phase_task_budget_overrides_are_strict(
    field: str, value: object, message: str
) -> None:
    payload = load_stage3_config("configs/v2/stage3/base.yaml").to_dict()
    payload = json.loads(json.dumps(payload))
    payload["tasks"]["experiment/density"][field] = value
    with pytest.raises(ValueError, match=message):
        stage3_config_from_dict(payload)


@pytest.mark.parametrize("value", (True, -0.01, 0.151, "0.15"))
def test_three_phase_private_dropout_override_is_bounded(value: object) -> None:
    payload = load_stage3_config("configs/v2/stage3/base.yaml").to_dict()
    payload = json.loads(json.dumps(payload))
    payload["tasks"]["experiment/density"]["model_overrides"] = {
        "private_dropout": value
    }
    with pytest.raises(ValueError, match="private_dropout must be in"):
        stage3_config_from_dict(payload)


def test_three_phase_training_publishes_fixed_final_state(
    tiny_prepared: Stage3Config,
) -> None:
    config = _tiny_three_phase(tiny_prepared)
    config = replace(
        config,
        tasks={
            **config.tasks,
            "experiment/a": replace(
                config.tasks["experiment/a"],
                phase2_private_epochs=0,
                phase3_private_epochs=0,
                model_overrides={"private_hidden_ratio": 0.5},
            ),
            "experiment/b": replace(
                config.tasks["experiment/b"], phase2_private_epochs=3
            ),
        },
    )
    output = config.data.artifacts_dir.parent / "three-phase-train"
    rows = run_stage3_training(config, 1, output_dir=output)

    assert rows[-1]["phase"] == "three_phase_final"
    assert (output / "phase_1/checkpoint_epoch_00002.pt").is_file()
    assert (output / "phase_2/g1/checkpoint_epoch_00002.pt").is_file()
    assert not (output / "phase_3/experiment__a").exists()
    assert (
        output / "phase_3/experiment__b/checkpoint_epoch_00001.pt"
    ).is_file()
    artifact = torch.load(
        output / "three_phase_final.pt", map_location="cpu", weights_only=False
    )
    phase_checkpoint = torch.load(
        output / "phase_1/checkpoint_epoch_00002.pt", map_location="cpu", weights_only=False
    )
    assert phase_checkpoint["format_version"] == 2
    assert "pcgrad_rng" not in phase_checkpoint
    historical = dict(artifact)
    historical_plan = json.loads(json.dumps(artifact["resolved_training_plan"]))
    historical_plan["format_version"] = 3
    historical_plan["math"].pop("gradient_aggregation")
    historical_plan["math"]["pcgrad"] = {
        "phase1": "hierarchical_ownership_blocks_v1",
        "phase2": "group_block_only_v1", "phase3": "off",
    }
    historical["resolved_training_plan"] = historical_plan
    old_payload = json.loads(json.dumps(artifact["training_identity"]["payload"]))
    old_payload["contract_version"] = 5
    old_payload["plan"]["math"] = historical_plan["math"]
    historical["training_identity"] = semantic_identity("stage3.training", old_payload)
    historical_path = output / "historical-three-phase.pt"
    torch.save(historical, historical_path)
    with pytest.raises(ValueError, match="identity"):
        _load_model(
            config, load_prepared_stage3(config), historical_path, 1, 0,
            torch.device("cpu"), three_phase_final=True,
        )
    manifest = json.loads((output / "three_phase_final.json").read_text())
    assert artifact["kind"] == "ilume_stage3_three_phase_final"
    assert manifest["artifact_sha256"] == sha256_file(
        output / "three_phase_final.pt"
    )
    assert set(manifest["phases"]["phase2"]["groups"]) == {
        "g1", "g2"
    }
    assert set(manifest["phases"]["phase3"]["tasks"]) == set(
        config.tasks
    )
    assert manifest["phases"]["phase3"]["tasks"]["experiment/a"][
        "carried_from_anchor"
    ] is True
    gate_fields = {
        "mean_global_gate_weight",
        "mean_group_gate_weight",
        "mean_private_gate_weight",
        "task_gate_entropy",
        "private_gate_weight_p10",
        "private_gate_weight_p50",
        "private_gate_weight_p90",
    }
    assert set(rows[-1]["validation"]["gate_diagnostics"]) == set(config.tasks)
    assert set(
        rows[-1]["validation"]["gate_diagnostics"]["experiment/a"]
    ) == gate_fields
    phase1_metric = json.loads(
        (output / "phase_1/metrics.jsonl").read_text().splitlines()[0]
    )
    phase2_metric = json.loads(
        (output / "phase_2/g1/metrics.jsonl").read_text().splitlines()[0]
    )
    phase2_manifest = json.loads((output / "phase_2/stitched.json").read_text())
    assert "gate_diagnostics" in phase1_metric["validation"]
    assert "gate_diagnostics" in phase2_metric["validation"]
    assert "gate_diagnostics" in phase2_manifest["validation"]
    assert "gate_diagnostics" in manifest["validation"]
    assert "best_metric" not in json.dumps(manifest)
    assert "selected_epoch" not in json.dumps(manifest)
    assert "best_state" not in json.dumps(manifest)
    plan = json.loads((output / "resolved_training_plan.json").read_text())
    assert plan["math"]["gradient_aggregation"] == "weighted_owner_raw_v1"
    assert "pcgrad" not in json.dumps(plan).lower()
    assert plan["format_version"] == 4
    assert artifact["training_identity"]["payload"]["contract_version"] == 6
    for scope in ("phase_1", "phase_2/g1", "phase_2/g2"):
        diagnostics = (output / scope / "diagnostics.jsonl").read_text()
        assert "pcgrad" not in diagnostics.lower()
    assert plan["phases"]["phase1"]["owners"]["PRIVATE:experiment/a"][
        "freeze_epoch"
    ] == 1
    assert plan["phases"]["phase1"]["owners"]["PRIVATE:experiment/a"][
        "capacity"
    ]["private_hidden_ratio"] == 0.5
    assert plan["phases"]["phase1"]["owners"]["PRIVATE:experiment/a"][
        "capacity"
    ]["tower_hidden_ratio"] == 1.0
    assert plan["phases"]["phase1"]["owners"]["PRIVATE:experiment/a"][
        "capacity"
    ]["private_dropout"] == config.model.dropout
    phase2_private = plan["phases"]["phase2"]["branches"]["g1"]["owners"][
        "PRIVATE:experiment/a"
    ]
    assert phase2_private["nominal_epochs"] == 0
    assert phase2_private["effective_epochs"] == 0
    assert phase2_private["actual_update_budget"] == 0
    assert phase2_private["terminal_lr"] == pytest.approx(2.0e-5)
    phase2_truncated = plan["phases"]["phase2"]["branches"]["g1"]["owners"][
        "PRIVATE:experiment/b"
    ]
    assert phase2_truncated["nominal_epochs"] == 3
    assert phase2_truncated["effective_epochs"] == 2
    assert plan["phases"]["phase3"]["branches"]["experiment/a"]["owners"][
        "PRIVATE:experiment/a"
    ]["nominal_lr"] == pytest.approx(2.0e-5)
    zero_branch = plan["phases"]["phase3"]["branches"]["experiment/a"]
    assert zero_branch["carried_from_anchor"] is True
    assert zero_branch["owners"]["PRIVATE:experiment/a"][
        "actual_update_budget"
    ] == 0

    phase2 = torch.load(
        output / "phase_2/stitched.pt", map_location="cpu", weights_only=False
    )
    private_names = {
        name for name, owner in artifact["ownership_manifest"].items()
        if owner == "PRIVATE:experiment/a"
    }
    assert all(
        torch.equal(phase2["model"][name], artifact["model"][name])
        for name in private_names
    )

    phase1_epoch1 = torch.load(
        output / "phase_1/checkpoint_epoch_00001.pt",
        map_location="cpu", weights_only=False,
    )
    phase1_epoch2 = torch.load(
        output / "phase_1/checkpoint_epoch_00002.pt",
        map_location="cpu", weights_only=False,
    )
    private_names = {
        name for name, owner in phase1_epoch1["ownership_manifest"].items()
        if owner == "PRIVATE:experiment/a"
    }
    global_names = {
        name for name, owner in phase1_epoch1["ownership_manifest"].items()
        if owner == "GLOBAL"
    }
    assert all(
        torch.equal(phase1_epoch1["model"][name], phase1_epoch2["model"][name])
        for name in private_names
    )
    assert any(
        not torch.equal(phase1_epoch1["model"][name], phase1_epoch2["model"][name])
        for name in global_names
    )

    phase2_epoch1 = torch.load(
        output / "phase_2/g1/checkpoint_epoch_00001.pt",
        map_location="cpu", weights_only=False,
    )
    phase2_epoch2 = torch.load(
        output / "phase_2/g1/checkpoint_epoch_00002.pt",
        map_location="cpu", weights_only=False,
    )
    private_delta_names = {
        name for name, owner in phase2_epoch1["ownership_manifest"].items()
        if owner == "PRIVATE:experiment/a"
    }
    group_delta_names = {
        name for name, owner in phase2_epoch1["ownership_manifest"].items()
        if owner == "GROUP:g1"
    }
    assert "PRIVATE:experiment/a" not in {
        group["owner"] for group in phase2_epoch1["optimizer"]["param_groups"]
    }
    assert phase2_epoch1["owner_updates"]["PRIVATE:experiment/a"] == 0
    assert all(
        torch.equal(
            phase1_epoch2["model"][name],
            phase2_epoch2["owner_state"][name],
        )
        for name in private_delta_names
    )
    assert all(
        torch.equal(
            phase2_epoch1["owner_state"][name],
            phase2_epoch2["owner_state"][name],
        )
        for name in private_delta_names
    )
    assert any(
        not torch.equal(
            phase2_epoch1["owner_state"][name],
            phase2_epoch2["owner_state"][name],
        )
        for name in group_delta_names
    )

    resumed = run_stage3_training(
        config, 1, output_dir=output, resume_from=output
    )
    assert resumed[-1]["phase"] == "three_phase_final"
    evaluated = evaluate_checkpoints(
        config,
        output,
        split="valid",
        ensemble_folds=False,
        task_subset=("experiment/a",),
        fold=1,
    )
    assert evaluated["model_selector"] == "three_phase_final"
    assert set(evaluated["gate_diagnostics"]) == {"experiment/a"}
    gate = evaluated["gate_diagnostics"]["experiment/a"]
    assert set(gate) == gate_fields
    assert gate["mean_global_gate_weight"] + gate[
        "mean_group_gate_weight"
    ] + gate["mean_private_gate_weight"] == pytest.approx(1.0)
    assert 0.0 <= gate["task_gate_entropy"] <= 1.0
    with pytest.raises(ValueError, match="only supported by legacy"):
        evaluate_checkpoints(
            config,
            output,
            split="valid",
            ensemble_folds=False,
            checkpoint_epoch=1,
            task_subset=("experiment/a",),
            fold=1,
        )


def test_three_phase_test_reports_fold_and_pooled_gate_diagnostics(
    tiny_prepared: Stage3Config,
) -> None:
    config = _tiny_three_phase(tiny_prepared)
    output = config.data.artifacts_dir.parent / "three-phase-ensemble"
    for fold in range(1, 6):
        run_stage3_training(config, fold, output_dir=output / f"fold{fold}")
    predictions = output / "evaluation-predictions"
    evaluated = evaluate_checkpoints(
        config,
        output,
        split="test",
        ensemble_folds=True,
        task_subset=("experiment/a",),
        predictions_dir=predictions,
    )
    gate_fields = {
        "mean_global_gate_weight",
        "mean_group_gate_weight",
        "mean_private_gate_weight",
        "task_gate_entropy",
        "private_gate_weight_p10",
        "private_gate_weight_p50",
        "private_gate_weight_p90",
    }
    assert set(evaluated["folds"]) == {f"fold{fold}" for fold in range(1, 6)}
    for fold_result in evaluated["folds"].values():
        assert set(fold_result["gate_diagnostics"]["experiment/a"]) == gate_fields
    aggregate = evaluated["ensemble"]["gate_diagnostics"]["experiment/a"]
    assert set(aggregate) == gate_fields
    assert aggregate["mean_global_gate_weight"] + aggregate[
        "mean_group_gate_weight"
    ] + aggregate["mean_private_gate_weight"] == pytest.approx(1.0)
    assert 0.0 <= aggregate["task_gate_entropy"] <= 1.0

    with (predictions / "experiment__a.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    spec = resolve_task_registry(config)["experiment/a"]
    assert set(rows[0]) == {
        "source_row",
        *spec.identity_columns,
        *spec.condition_columns,
        "target",
        *(f"prediction_fold{fold}" for fold in range(1, 6)),
        "prediction_ensemble",
        "absolute_error_ensemble",
    }


def test_three_phase_optimizer_groups_follow_ownership(
    tiny_prepared: Stage3Config,
) -> None:
    config = _tiny_three_phase(tiny_prepared)
    model = Stage3SparseModel(
        config.model,
        resolve_task_registry(config),
        4,
        group_configs=config.groups,
        task_configs=config.tasks,
    )
    owners = (group_owner("g1"), private_owner("experiment/a"))
    model.set_trainable_owners(owners)
    optimizer = _three_phase_optimizer(
        model,
        config,
        {owners[0]: 7.5e-5, owners[1]: 5.0e-5},
    )

    assert optimizer.state == {}
    assert {group["owner"] for group in optimizer.param_groups} == {
        "GROUP:g1", "PRIVATE:experiment/a"
    }
    assert {
        (group["owner"], group["lr"])
        for group in optimizer.param_groups
    } == {
        ("GROUP:g1", 7.5e-5),
        ("PRIVATE:experiment/a", 5.0e-5),
    }
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected = {
        id(parameter)
        for parameter, owner in model.parameter_ownership().items()
        if owner in owners
    }
    assert optimized == expected
    assert all(
        parameter.requires_grad == (owner in owners)
        for parameter, owner in model.parameter_ownership().items()
    )


def test_three_phase_scheduler_recipes_reach_owner_local_floors() -> None:
    assert _three_phase_lr_factor(0, 5, 100, 0.5) == pytest.approx(0.2)
    assert _three_phase_lr_factor(4, 5, 100, 0.5) == pytest.approx(1.0)
    assert _three_phase_lr_factor(99, 5, 100, 0.1) == pytest.approx(0.1)
    assert _three_phase_lr_factor(99, 0, 100, 0.5) == pytest.approx(0.5)
    assert _three_phase_lr_factor(99, 0, 100, 0.2) == pytest.approx(0.2)
    left = torch.nn.Parameter(torch.ones(()))
    right = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW(
        [
            {"params": [left], "lr": 1.0, "owner": "LEFT"},
            {"params": [right], "lr": 1.0, "owner": "RIGHT"},
        ]
    )
    scheduler = _OwnerScheduler(
        optimizer,
        {
            owner: {
                "nominal_lr": 1.0,
                "terminal_lr": 0.5,
                "actual_update_budget": 2,
            }
            for owner in ("LEFT", "RIGHT")
        },
    )
    left.grad = torch.ones_like(left)
    scheduler.step(("LEFT",))
    assert scheduler.updates == {"LEFT": 1, "RIGHT": 0}
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1.0)
    left.grad = torch.ones_like(left)
    scheduler.step(("LEFT",))
    assert scheduler.updates == {"LEFT": 2, "RIGHT": 0}
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)


def test_three_phase_owner_delta_stitch_is_order_independent(
    tiny_prepared: Stage3Config,
) -> None:
    config = _tiny_three_phase(tiny_prepared)
    registry = resolve_task_registry(config)
    first = Stage3SparseModel(
        config.model, registry, 4,
        group_configs=config.groups, task_configs=config.tasks,
    )
    second = Stage3SparseModel(
        config.model, registry, 4,
        group_configs=config.groups, task_configs=config.tasks,
    )
    second.load_state_dict(first.state_dict())
    anchor = _three_phase_model_state(first)
    owners = (group_owner("g1"), private_owner("experiment/c"))
    deltas = {}
    for index, owner in enumerate(owners, start=1):
        state = _three_phase_owner_state(first, (owner,))
        state = {name: value + index for name, value in state.items()}
        deltas[owner.label] = ((owner,), state, _three_phase_owner_hash(state))

    forward = _stitch_owner_deltas(first, anchor, deltas)
    reverse = _stitch_owner_deltas(
        second, anchor, dict(reversed(list(deltas.items())))
    )
    assert set(forward) == set(reverse)
    assert all(torch.equal(forward[name], reverse[name]) for name in forward)


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


def test_training_variants_preserve_prepared_data_but_change_identity(
    tiny_prepared: Stage3Config,
) -> None:
    config = _tiny_three_phase(tiny_prepared)
    metadata_path = config.data.artifacts_dir / "metadata.json"
    original_metadata = metadata_path.read_bytes()
    original = resolve_stage3_training_identity(config, 1)
    variants = (
        replace(config, training=replace(config.training, seed=10042)),
        replace(config, groups={name: replace(spec, group_weight=2.0)
                                for name, spec in config.groups.items()}),
        replace(config, model=replace(config.model, global_experts=0)),
    )
    for changed in variants:
        assert changed.data.artifacts_dir == config.data.artifacts_dir
        assert set(load_prepared_stage3(changed)["registry"]) == set(config.tasks)
        assert metadata_path.read_bytes() == original_metadata
        assert resolve_stage3_training_identity(changed, 1) != original


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


def test_joint_gradient_clipping_isolated_by_owner(
    tiny_prepared: Stage3Config,
) -> None:
    registry = resolve_task_registry(tiny_prepared)
    model = Stage3SparseModel(tiny_prepared.model, registry, 4)
    ownership = model.parameter_ownership()
    assert all(owner.scope in {"GLOBAL", "GROUP", "PRIVATE"} for owner in ownership.values())
    assert set(model.parameters_for_owner(private_owner("experiment/a"))).isdisjoint(
        model.parameters_for_owner(private_owner("experiment/b"))
    )
    assert set(model.parameters_for_owner(group_owner("g1"))).isdisjoint(
        model.parameters_for_owner(group_owner("g2"))
    )
    assert model.parameters_for_owner(GLOBAL)

    requested_norms = {
        "GLOBAL": 0.25,
        "GROUP:g1": 0.50,
        "GROUP:g2": 0.75,
        "PRIVATE:experiment/a": 10.0,
        "PRIVATE:experiment/b": 20.0,
        "PRIVATE:experiment/c": 30.0,
    }
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

    config = _tiny_three_phase(tiny_rdkit_prepared)
    output = config.data.artifacts_dir.parent / "rdkit-train"
    run_stage3_training(config, 1, output_dir=output)
    final = torch.load(output / "three_phase_final.pt", map_location="cpu", weights_only=False)
    assert final["kind"] == "ilume_stage3_rdkit_home_three_phase_final"
    assert final["representation"]["kind"] == "rdkit_2d_adapter"
    assert any(name.startswith("descriptor_adapters.") for name in final["model"])
    evaluation = evaluate_checkpoints(
        config, output, split="valid", ensemble_folds=False,
        task_subset=("experiment/a",), fold=1,
    )
    assert evaluation["reporting"]["model_id"] == "rdkit_2d_home"

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

    config = _tiny_three_phase(config)
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

def test_raw_gradient_config_and_identity(tiny_prepared: Stage3Config) -> None:
    base = load_stage3_config("configs/v2/stage3/base.yaml")
    assert "pcgrad_mode" not in base.to_dict()["training"]
    assert "debug_pcgrad_traces" not in base.to_dict()["training"]
    assert stage3_config_from_dict(base.to_dict()) == base
    for name in ("pcgrad_mode", "debug_pcgrad_traces"):
        payload = base.to_dict()
        payload["training"][name] = "off"
        with pytest.raises(ValueError):
            stage3_config_from_dict(payload)
    config = _tiny_three_phase(tiny_prepared)
    identity = resolve_stage3_training_identity(config, 1)
    plan = identity["payload"]["plan"]
    assert plan["math"]["gradient_aggregation"] == "weighted_owner_raw_v1"
    assert plan["prepared_identity"]
    assert identity["payload"]["contract_version"] == 6


@pytest.mark.parametrize("tasks, frozen_global", (
    (("experiment/a", "experiment/b", "experiment/c"), False),
    (("experiment/a", "experiment/c"), False),
    (("experiment/b",), False),
    (("experiment/a", "experiment/b"), True),
))
def test_no_pcgrad_preserves_raw_weighted_owner_gradients(
    tiny_prepared: Stage3Config, monkeypatch: pytest.MonkeyPatch,
    tasks: tuple[str, ...], frozen_global: bool,
) -> None:
    registry = resolve_task_registry(tiny_prepared)
    weights = {"experiment/a": 1.0, "experiment/b": 3.0, "experiment/c": 2.0}
    registry = {task: replace(spec, task_weight=weights[task]) for task, spec in registry.items()}
    model = Stage3SparseModel(tiny_prepared.model, registry, 4)
    raw_values = {"experiment/a": 2.0, "experiment/b": -1.0, "experiment/c": 4.0}
    group_weights = {"g1": 2.0, "g2": 5.0}
    gradients = {}
    for task in tasks:
        owners = (group_owner(registry[task].meta_group), private_owner(task))
        if not frozen_global:
            owners = (GLOBAL, *owners)
        gradients[task] = {
            parameter: torch.full_like(parameter, raw_values[task])
            for owner in owners for parameter in model.parameters_for_owner(owner)
        }
    result = assemble_owner_gradients(model, gradients, registry, group_weights)
    groups = {registry[task].meta_group for task in tasks}
    means = {}
    for group in groups:
        members = [task for task in tasks if registry[task].meta_group == group]
        weight_sum = sum(weights[task] for task in members)
        means[group] = sum(weights[task] * raw_values[task] for task in members) / weight_sum
        for parameter in model.parameters_for_owner(group_owner(group)):
            torch.testing.assert_close(result.gradients[parameter], torch.full_like(parameter, means[group]))
        for task in members:
            expected = raw_values[task] * len(members) * weights[task] / weight_sum
            for parameter in model.parameters_for_owner(private_owner(task)):
                torch.testing.assert_close(result.gradients[parameter], torch.full_like(parameter, expected))
    expected_global = sum(means[group] * group_weights[group] for group in groups) / sum(group_weights[group] for group in groups)
    for parameter in model.parameters_for_owner(GLOBAL):
        if frozen_global:
            assert parameter not in result.gradients
        else:
            torch.testing.assert_close(result.gradients[parameter], torch.full_like(parameter, expected_global))
    assert set(result.gradients) == {parameter for raw in gradients.values() for parameter in raw}
    for task, raw in gradients.items():
        assert all(torch.equal(value, torch.full_like(value, raw_values[task])) for value in raw.values())


@pytest.mark.parametrize("scope", ("phase_1", "phase_2/g1"))
def test_no_pcgrad_partial_resume_is_exact(tiny_prepared: Stage3Config, scope: str) -> None:
    config = _tiny_three_phase(tiny_prepared)
    config = replace(
        config,
        model=replace(config.model, dropout=0.1),
    )
    continuous = config.data.artifacts_dir.parent / "continuous-no-pcgrad"
    resumed = config.data.artifacts_dir.parent / "resumed-no-pcgrad"
    run_stage3_training(config, 1, output_dir=continuous)
    (resumed / scope).mkdir(parents=True)
    shutil.copy(continuous / "resolved_training_plan.json", resumed)
    if scope.startswith("phase_2"):
        shutil.copytree(continuous / "phase_1", resumed / "phase_1")
    shutil.copy(continuous / scope / "checkpoint_epoch_00001.pt", resumed / scope)
    for filename in ("metrics.jsonl", "diagnostics.jsonl"):
        first = (continuous / scope / filename).read_text().splitlines()[0]
        (resumed / scope / filename).write_text(first + "\n")
    run_stage3_training(config, 1, output_dir=resumed, resume_from=resumed)
    expected = torch.load(continuous / "three_phase_final.pt", map_location="cpu", weights_only=False)
    actual = torch.load(resumed / "three_phase_final.pt", map_location="cpu", weights_only=False)
    assert expected["training_identity"] == actual["training_identity"]
    assert expected["model"].keys() == actual["model"].keys()
    assert all(torch.equal(value, actual["model"][name]) for name, value in expected["model"].items())
    for filename in ("metrics.jsonl", "diagnostics.jsonl"):
        assert (continuous / scope / filename).read_text() == (resumed / scope / filename).read_text()


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

    result = assemble_owner_gradients(
        model,
        gradients,
        registry,
        {"g1": 1.0, "g2": 1.0},
    )

    assert set(result.task_norms) == {task}
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
    result = assemble_owner_gradients(
        model,
        gradients,
        registry,
        {"g1": 1.0, "g2": 1.0},
    )
    assert result.assembled_owner_norms["GLOBAL"] == 0.0
    assert result.gradients

def test_legacy_training_is_retired(tiny_prepared: Stage3Config) -> None:
    with pytest.raises(ValueError, match="Legacy Stage 3 training and resume are retired"):
        run_stage3_training(tiny_prepared, 1, output_dir=tiny_prepared.data.artifacts_dir.parent / "retired")
    with pytest.raises(ValueError, match="Legacy Stage 3 training and resume are retired"):
        resolve_stage3_training_identity(tiny_prepared, 1)


def test_legacy_final_artifact_remains_readable(tiny_prepared: Stage3Config) -> None:
    from stage3.train import _normalization_for_run

    config = tiny_prepared
    prepared = load_prepared_stage3(config)
    representations = Stage3RepresentationStore(
        config.data.artifacts_dir, 1, prepared["objects"], prepared["metadata"]["kind"]
    )
    model = Stage3SparseModel(config.model, prepared["registry"], representations.output_dim)
    tasks = tuple(prepared["registry"])
    datasets = {task: Stage3TaskDataset(config.data.artifacts_dir, 1, task, "train") for task in tasks}
    normalization = _normalization_for_run(prepared, 1, None)
    plan = build_resolved_training_plan(
        config, 1, model, datasets, tasks, prepared, {}, normalization
    )
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    artifact = {
        "kind": "ilume_stage3_taskwise_refined", "format_version": 1, "fold": 1,
        "resolved_registry": plan["resolved_registry"],
        "resolved_training_plan": plan,
        "training_identity": build_stage3_training_identity(plan),
        "normalization": normalization,
        "stage2_encoder_identity": metadata_identity(
            prepared["metadata"], "stage2_encoder", context="test"
        )["hash"],
        "ownership_manifest": model.ownership_manifest(),
        "model": state,
        "model_state_hash": tensor_state_hash("stage3.taskwise-refined-state", state),
    }
    path = config.data.artifacts_dir.parent / "historical-taskwise-refined.pt"
    torch.save(artifact, path)
    loaded, _, _ = _load_model(
        config, prepared, path, 1, 0, torch.device("cpu"), taskwise_refined=True
    )
    assert set(loaded.state_dict()) == set(state)


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


def test_full_finetune_scheduler_reuses_cuda_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeProcess.created = []
    calls: list[tuple[int, str | None, bool]] = []

    def worker(config, features, output, fold, resume, device, progress, result_queue):
        del config, features, output, resume
        calls.append((fold, device, progress))
        result_queue.put(("failed" if fold == 2 else "completed", None, None))

    monkeypatch.setattr(full_finetune_launcher.multiprocessing, "get_context", lambda mode: _FakeContext())
    monkeypatch.setattr("multiprocessing.connection.wait", lambda sentinels: sentinels)
    monkeypatch.setattr(full_finetune_launcher, "_worker_entry", worker)
    results = full_finetune_launcher._run_schedule(
        config_path="config.yaml", feature_dir="features", folds=(1, 2, 3, 4, 5),
        output_root="outputs/test", resume=True, max_parallel=4,
        devices=("cuda:0", "cuda:1"),
    )
    assert results == {1: "completed", 2: "failed", 3: "completed", 4: "completed", 5: "completed"}
    assert calls == [
        (1, "cuda:0", True), (2, "cuda:1", False),
        (3, "cuda:0", False), (4, "cuda:1", False),
        (5, "cuda:0", False),
    ]


def test_full_finetune_resume_starts_missing_fold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        "ablations.stage3_full_finetune.representation.load_config",
        lambda _: (object(), object()),
    )
    monkeypatch.setattr("stage3.config.configure_process_runtime", lambda _: None)
    monkeypatch.setattr(
        "ablations.stage3_full_finetune.train.run_finetuning",
        lambda _config, _recipe, **kwargs: calls.append((kwargs["fold"], kwargs["resume"])),
    )
    assert full_finetune_launcher._run_fold(
        "config.yaml", "features", str(tmp_path), 2, True, None, True
    ) == "completed"
    assert calls == [(2, False)]

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


def test_stage2_stage3_transfer_config_covers_full_matrix() -> None:
    config = load_transfer_config("configs/ablations/stage2_stage3_transfer.yaml")
    assert len(config.stage2.sources) == 9
    assert len(config.stage3.targets) == 20
    assert config.stage3.folds == (1, 2, 3, 4, 5)
    assert config.stage2.physics_only is True
    assert config.stage2.epochs == config.stage2.final_epoch == 10
    assert config.stage3.epochs == 10
    assert config.stage3.selection == "final"
    assert config.stage3.metric == "validation_raw_mae"
    assert "signature" not in json.dumps(config.to_dict())
    assert transfer_config_from_dict(config.to_dict()) == config
    assert canonical_json_sha256(config.to_dict()) == (
        "b9d7905885d1329c315aab50b1a1826a0f73d892b86cfec3d610fcfdcd2857c8"
    )
    retired = config.to_dict()
    retired["stage2"]["sampling_mode"] = "balanced_rows"
    with pytest.raises(ValueError, match="Balanced transfer is retired"):
        transfer_config_from_dict(retired)


def test_stage2_stage3_transfer_seed_is_numpy_compatible() -> None:
    seed = transfer_training_seed(
        42, "experiment/thermal_conductivity", 5
    )
    assert 0 <= seed < 2**32
    assert seed == transfer_training_seed(
        42, "experiment/thermal_conductivity", 5
    )
    seed_everything(seed)


def test_transfer_representation_rejects_retired_balanced_artifact(tmp_path: Path) -> None:
    path = tmp_path / "balanced.pt"
    path.with_suffix(".json").write_text(
        json.dumps({"balanced_experiment_identity": "retired"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Balanced transfer artifacts are retired"):
        load_representation_bank(path, expected_prepared_identity={})
    with pytest.raises(ValueError, match="Balanced transfer artifacts are retired"):
        prepare_representation_bank(
            _tiny_transfer_config(), variant="baseline", encoder_path=tmp_path / "encoder.pt",
            encoder_manifest_path=path.with_suffix(".json"), destination=tmp_path / "output.pt",
            expected_initial_shared_state_hash="initial",
        )


def test_stage3_transfer_joint_object_encoder_is_trainable_for_mixed_topologies() -> None:
    torch.manual_seed(7)
    encoder = ObjectEncoder(
        8, 2, num_layers=1, feedforward_dim=16, dropout=0.0
    )
    bank = {
        "entity_slots": torch.randn(3, 2, 8),
        "entity_roles": torch.tensor(
            [
                [ROLE_TO_ID["neutral"], 0],
                [ROLE_TO_ID["cation"], ROLE_TO_ID["anion"]],
                [ROLE_TO_ID["neutral"], 0],
            ]
        ),
        "slot_counts": torch.tensor([1, 2, 1]),
    }
    object_ids = torch.tensor([1, 0, 1, 2])
    before = {
        name: value.detach().clone() for name, value in encoder.state_dict().items()
    }
    values = encode_transfer_objects(
        encoder, bank, object_ids, device=torch.device("cpu")
    )
    assert values.shape == (4, 8)
    assert torch.equal(values[0], values[2])
    loss = values.square().mean()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-2)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert any(parameter.grad is not None for parameter in encoder.parameters())
    optimizer.step()
    assert any(
        not torch.equal(before[name], value)
        for name, value in encoder.state_dict().items()
    )


def test_stage3_transfer_job_updates_object_encoder_and_mlp_together(
    tmp_path: Path,
) -> None:
    class Dataset:
        def __init__(self, split: str) -> None:
            self.primary_object_ids = torch.tensor([0, 1, 0, 1])
            self.partner_object_ids = torch.full((4,), -1, dtype=torch.long)
            self.conditions = torch.tensor([[0.0], [0.5], [1.0], [1.5]])
            self.targets = torch.tensor([0.2, -0.1, 0.4, -0.3])
            self.raw_targets = self.targets.clone()
            self.source_rows = torch.arange(4) + (100 if split == "valid" else 0)

        def __len__(self) -> int:
            return len(self.targets)

    config = _tiny_transfer_config()
    config = replace(
        config,
        stage3=replace(
            config.stage3,
            batch_size=2,
            epochs=1,
            warmup_fraction=0.0,
            amp_dtype="none",
        ),
    )
    initial_encoder = ObjectEncoder(
        8, 2, num_layers=1, feedforward_dim=16, dropout=0.0
    )
    initial_encoder_state = {
        name: value.detach().clone()
        for name, value in initial_encoder.state_dict().items()
    }
    bank = {
        "identity": {"hash": "representation"},
        "variant": "baseline",
        "objects": [{"topology": "molecule", "slots": [["neutral", "C"]]}] * 2,
        "entity_slots": torch.randn(2, 2, 8),
        "entity_roles": torch.tensor(
            [[ROLE_TO_ID["neutral"], 0], [ROLE_TO_ID["neutral"], 0]]
        ),
        "slot_counts": torch.ones(2, dtype=torch.long),
        "object_encoder_contract": {
            "d_model": 8,
            "n_heads": 2,
            "layers": 1,
            "ffn_dim": 16,
            "dropout": 0.0,
        },
        "object_encoder_state": initial_encoder_state,
        "object_encoder_state_hash": tensor_state_hash(
            "stage2-stage3-transfer-object-encoder-initial.v2",
            initial_encoder_state,
        ),
    }
    prepared = {
        "metadata": {},
        "objects": {"objects": bank["objects"]},
        "registry": {"target/a": SimpleNamespace(partner_slots=())},
        "normalization": {
            "fold1": {"target/a": {"target": {"mean": 0.0, "scale": 1.0}}}
        },
    }
    authority = SimpleNamespace(data=SimpleNamespace(artifacts_dir=tmp_path))
    with (
        patch(
            "ablations.stage2_stage3_transfer.stage3.load_stage3_config",
            return_value=authority,
        ),
        patch(
            "ablations.stage2_stage3_transfer.stage3.load_prepared_stage3",
            return_value=prepared,
        ),
        patch(
            "ablations.stage2_stage3_transfer.stage3.metadata_identity",
            return_value={"hash": "prepared"},
        ),
        patch(
            "ablations.stage2_stage3_transfer.stage3.load_representation_bank",
            return_value=bank,
        ),
        patch(
            "ablations.stage2_stage3_transfer.stage3.Stage3TaskDataset",
            side_effect=lambda _root, _fold, _task, split: Dataset(split),
        ),
    ):
        manifest = train_transfer_job(
            config,
            variant="baseline",
            representation_path=tmp_path / "representation.pt",
            task_id="target/a",
            fold=1,
            output_dir=tmp_path / "job",
            device_name="cpu",
            reporter=SimpleNamespace(
                bar=lambda **_kwargs: SimpleNamespace(
                    set_postfix=lambda **_values: None,
                    update=lambda _value: None,
                    close=lambda: None,
                )
            ),
        )
    artifact = torch.load(tmp_path / "job/final.pt", weights_only=False)
    assert manifest["downstream_training"] == "joint_object_encoder_mlp"
    assert any(
        not torch.equal(initial_encoder_state[name], value)
        for name, value in artifact["object_encoder"].items()
    )
    assert artifact["initial_state_hash"] != tensor_state_hash(
        "stage2-stage3-transfer-mlp-initial.v2", artifact["model"]
    )


def test_stage3_transfer_parallel_slots_support_multiple_jobs_per_device() -> None:
    assert transfer_launcher._parallel_slots(6, ("cuda:0",)) == 6
    assert transfer_launcher._parallel_slots(
        8, ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    ) == 2
    with pytest.raises(ValueError, match="positive multiple"):
        transfer_launcher._parallel_slots(
            6, ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
        )


def test_stage3_transfer_reports_completed_training_jobs(tmp_path: Path) -> None:
    updates: list[int] = []
    closed: list[bool] = []
    dispatched: list[tuple[Any, ...]] = []

    class Bar:
        def update(self, value: int) -> None:
            updates.append(value)

        def close(self) -> None:
            closed.append(True)

    class Reporter:
        def bar(self, **kwargs):
            assert kwargs == {
                "total": 4,
                "desc": "Stage3 transfer matrix",
                "unit": "train-job",
            }
            return Bar()

    config = SimpleNamespace(
        stage2=SimpleNamespace(sources=("source/a",)),
        stage3=SimpleNamespace(targets=("experiment/a",), folds=(1, 2)),
    )
    with (
        patch.object(transfer_launcher, "load_transfer_config", return_value=config),
        patch.object(
            transfer_launcher,
            "_train_worker",
            side_effect=lambda *args: dispatched.append(args),
        ),
        patch.object(transfer_launcher, "ProgressReporter", return_value=Reporter()),
        patch(
            "sys.argv",
            [
                "transfer.py", "train", "--config", "unused.yaml",
                "--representations", str(tmp_path / "representations"),
                "--output", str(tmp_path / "stage3"),
                "--max-parallel", "1",
            ],
        ),
    ):
        transfer_launcher.main()
    assert len(dispatched) == 4
    assert all(job[-2] is True for job in dispatched)
    assert updates == [1, 1, 1, 1]
    assert closed == [True]


def _tiny_transfer_config() -> TransferExperimentConfig:
    return TransferExperimentConfig(
        name="tiny-transfer",
        seed=42,
        stage2=Stage2TransferConfig(
            authority_config=Path("unused-stage2.yaml"),
            stage1_checkpoint=Path("unused-stage1.pt"),
            prepared_artifacts=Path("unused-stage2-data"),
            sources=("source/a", "source/b"),
            object_layers=2,
            object_ffn_dim=2048,
            dropout=0.1,
            batch_size=256,
            epochs=10,
            backbone_frozen_epochs=1,
            backbone_learning_rate=1e-5,
            object_encoder_learning_rate=3e-5,
            task_head_learning_rate=1e-4,
            weight_decay=0.01,
            warmup_fraction=0.05,
            max_grad_norm=1.0,
            amp_dtype="bf16",
            optimizer="AdamW",
            physics_only=True,
            final_epoch=10,
        ),
        stage3=Stage3TransferConfig(
            authority_config=Path("unused-stage3.yaml"),
            prepared_artifacts=Path("unused-stage3-data"),
            targets=("target/a", "target/b"),
            folds=(1, 2),
            hidden_dims=(512, 256),
            dropout=0.1,
            batch_size=128,
            epochs=10,
            learning_rate=3e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
            smooth_l1_beta=1.0,
            warmup_fraction=0.05,
            min_lr_ratio=0.05,
            max_grad_norm=1.0,
            amp_dtype="none",
            selection="final",
            metric="validation_raw_mae",
        ),
    )


def _write_transfer_job(
    root: Path, *, variant: str, task: str, fold: int, mae: float,
) -> None:
    root.mkdir(parents=True)
    artifact = root / "final.pt"
    predictions = root / "validation_predictions.csv"
    artifact.write_bytes(b"model")
    predictions.write_text("source_row,target,prediction\n1,1,1\n", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({
            "kind": TRANSFER_MODEL_KIND,
            "format_version": TRANSFER_MODEL_VERSION,
            "downstream_training": "joint_object_encoder_mlp",
            "variant": variant,
            "task": task,
            "fold": fold,
            "artifact": artifact.name,
            "artifact_sha256": sha256_file(artifact),
            "predictions_sha256": sha256_file(predictions),
            "final_epoch": 10,
            "validation_raw_mae": mae,
            "row_target_hash": f"rows-{task}-{fold}",
            "initial_state_hash": f"init-{task}-{fold}",
            "permutation_hashes": [f"perm-{task}-{fold}"],
        }),
        encoding="utf-8",
    )


def test_transfer_summary_uses_fold_gain_and_rejects_alignment_drift(
    tmp_path: Path,
) -> None:
    config = _tiny_transfer_config()
    stage3_root = tmp_path / "stage3"
    baseline_mae = {1: 10.0, 2: 20.0}
    for target in config.stage3.targets:
        for fold in config.stage3.folds:
            _write_transfer_job(
                stage3_root / "baseline" / target.replace("/", "__") / f"fold{fold}",
                variant="baseline", task=target, fold=fold,
                mae=baseline_mae[fold],
            )
            for source in config.stage2.sources:
                transfer_mae = (
                    {1: 8.0, 2: 22.0}[fold]
                    if source == "source/a" and target == "target/a"
                    else baseline_mae[fold]
                )
                _write_transfer_job(
                    stage3_root / "sources" / source.replace("/", "__") / target.replace("/", "__") / f"fold{fold}",
                    variant=source, task=target, fold=fold, mae=transfer_mae,
                )
    output = tmp_path / "summary"
    summary = summarize_transfer_matrix(
        config, stage3_root=stage3_root, output_dir=output
    )
    assert summary["pair_count"] == 4
    assert "sampling_mode" not in summary
    with (output / "transfer_gain_pairs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next(item for item in rows if item["source"] == "source/a" and item["target"] == "target/a")
    assert float(row["TG_fold1"]) == pytest.approx(0.2)
    assert float(row["TG_fold2"]) == pytest.approx(-0.1)
    assert float(row["TG_mean"]) == pytest.approx(0.05)
    assert float(row["TG_median"]) == pytest.approx(0.05)
    assert int(row["positive_folds"]) == 1
    with (output / "transfer_gain_matrix.csv").open(newline="", encoding="utf-8") as handle:
        matrix = list(csv.DictReader(handle))
    assert [item["source"] for item in matrix] == ["source/a", "source/b"]
    assert float(matrix[0]["target/a"]) == pytest.approx(0.05)
    assert "5.0%" in (output / "transfer_gain_heatmap.svg").read_text(encoding="utf-8")
    assert transfer_gain(10.0, 8.0) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="finite and positive"):
        transfer_gain(0.0, 0.0)
    contaminated = stage3_root / "sources/source__a/target__a/fold1/manifest.json"
    payload = json.loads(contaminated.read_text(encoding="utf-8"))
    payload["balanced_experiment_identity"] = "retired"
    contaminated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Balanced transfer artifacts are retired"):
        summarize_transfer_matrix(
            config, stage3_root=stage3_root, output_dir=tmp_path / "mixed-summary"
        )
    payload.pop("balanced_experiment_identity")
    payload["row_target_hash"] = "wrong-order"
    contaminated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="row_target_hash mismatch"):
        summarize_transfer_matrix(
            config, stage3_root=stage3_root, output_dir=tmp_path / "bad-summary"
        )


def test_full_finetune_config_only_changes_base_microbatch() -> None:
    config, recipe = load_full_finetune_config(
        "configs/ablations/stage3_full_finetune.yaml"
    )
    base = load_stage3_config("configs/v2/stage3/base.yaml")
    expected = base.to_dict()
    expected["training"]["microbatch_size"] = 8
    assert config.to_dict() == expected
    assert recipe.stage1_lr == pytest.approx(5e-6)
    assert recipe.stage2_lr == pytest.approx(1.5e-5)
    assert recipe.epochs == 15
    owner = _encoder_owner_recipe(5e-6, 15, 7, 0.05, 0.1)
    assert owner["actual_update_budget"] == 105
    assert owner["warmup_updates"] == 6
    assert owner["terminal_lr"] == pytest.approx(5e-7)


def test_full_finetune_features_bind_base_prepared_and_encoder(
    tiny_prepared: Stage3Config, tmp_path: Path,
) -> None:
    config = _tiny_three_phase(tiny_prepared)
    prepared = load_prepared_stage3(config)
    encoder = SimpleNamespace(
        encoder_identity=TEST_ENCODER_IDENTITY,
        input_sample=lambda role, smiles: {
            "token_ids": torch.tensor([len(role), len(smiles)])
        },
    )
    root = tmp_path / "finetune-features"
    with patch(
        "ablations.stage3_full_finetune.representation.load_frozen_object_encoder",
        return_value=encoder,
    ):
        manifest = prepare_finetune_features(config, root)
    payload = load_finetune_features(config, root, prepared)
    assert len(payload["samples"]) == manifest["sample_count"]
    assert payload["artifact_sha256"] == manifest["artifact_sha256"]
    assert payload["object_keys_hash"] == canonical_json_sha256([
        key.to_dict() for key in finetune_object_keys(prepared)
    ])
    (root / "features.json").write_text(
        json.dumps({**manifest, "artifact_sha256": "wrong"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="integrity mismatch"):
        load_finetune_features(config, root, prepared)


def test_full_finetune_upstream_gradients_use_global_weighting(
    tiny_prepared: Stage3Config,
) -> None:
    registry = resolve_task_registry(tiny_prepared)
    weights = {"experiment/a": 1.0, "experiment/b": 3.0, "experiment/c": 2.0}
    registry = {task: replace(spec, task_weight=weights[task]) for task, spec in registry.items()}
    model = FinetuneStage3Model(
        tiny_prepared.model, registry, 4,
        backbone=torch.nn.Linear(4, 4),
        object_encoder=torch.nn.Linear(4, 4),
    )
    values = {"experiment/a": 2.0, "experiment/b": -1.0, "experiment/c": 4.0}
    gradients = {
        task: {
            parameter: torch.full_like(parameter, values[task])
            for owner in (GLOBAL, STAGE1_OWNER, STAGE2_OWNER,
                          group_owner(spec.meta_group), private_owner(task))
            for parameter in model.parameters_for_owner(owner)
        }
        for task, spec in registry.items()
    }
    result = assemble_owner_gradients(model, gradients, registry, {"g1": 2.0, "g2": 5.0})
    expected = ((2.0 - 3.0) / 4.0 * 2.0 + 4.0 * 5.0) / 7.0
    for owner in (GLOBAL, STAGE1_OWNER, STAGE2_OWNER):
        for parameter in model.parameters_for_owner(owner):
            torch.testing.assert_close(
                result.gradients[parameter], torch.full_like(parameter, expected)
            )
    assert result.assembled_owner_norms[STAGE1_OWNER.label] > 0
    assert result.assembled_owner_norms[STAGE2_OWNER.label] > 0


def test_full_finetune_live_gradients_and_phase1_freeze(
    tiny_prepared: Stage3Config,
) -> None:
    config = _tiny_three_phase(tiny_prepared)
    prepared = load_prepared_stage3(config)
    keys = finetune_object_keys(prepared)

    class Batch:
        def __init__(self, values: torch.Tensor) -> None:
            self.values = values

        def to(self, device: torch.device) -> "Batch":
            return Batch(self.values.to(device))

    class Packer:
        def __call__(self, samples: list[dict[str, torch.Tensor]]) -> Batch:
            return Batch(torch.stack([sample["values"] for sample in samples]))

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.project = torch.nn.Linear(4, 4)

        def encode_entity(self, batch: Batch) -> SimpleNamespace:
            return SimpleNamespace(entity_embedding=self.project(batch.values))

    samples = {
        (role, smiles): {"values": torch.tensor(
            [float(len(smiles)), float(ROLE_TO_ID[role]), 1.0, 0.5]
        )}
        for key in keys for role, smiles in key.slots
    }
    model = FinetuneStage3Model(
        config.model, prepared["registry"], 4,
        group_configs=config.groups, task_configs=config.tasks,
        task_private_recipes={
            task: config.resolved_private_recipe(task) for task in config.tasks
        },
        backbone=Backbone(),
        object_encoder=ObjectEncoder(4, 2, num_layers=1, feedforward_dim=8, dropout=0.0),
    )
    store = LiveRepresentationStore(model, Packer(), keys, samples)
    model.set_trainable_owners(tuple(set(model.parameter_ownership().values())))
    task = "experiment/a"
    dataset = Stage3TaskDataset(config.data.artifacts_dir, 1, task, "train")
    stats = prepared["normalization"]["fold1"][task]
    gradients, _ = compute_task_gradient(
        model, task, dataset, torch.arange(len(dataset)), store,
        stats, config, torch.device("cpu"),
    )
    assert any(parameter in gradients for parameter in model.parameters_for_owner(STAGE1_OWNER))
    assert any(parameter in gradients for parameter in model.parameters_for_owner(STAGE2_OWNER))
    shared = [*model.parameters_for_owner(STAGE1_OWNER), *model.parameters_for_owner(STAGE2_OWNER)]
    before = [parameter.detach().clone() for parameter in shared]
    optimizer = torch.optim.AdamW(shared, lr=1e-3)
    for parameter in shared:
        parameter.grad = gradients[parameter]
    optimizer.step()
    assert any(not torch.equal(old, parameter) for old, parameter in zip(before, shared))
    model.set_trainable_owners((GLOBAL, group_owner("g1"), private_owner(task)))
    frozen = [parameter.detach().clone() for parameter in shared]
    gradients, _ = compute_task_gradient(
        model, task, dataset, torch.arange(len(dataset)), store,
        stats, config, torch.device("cpu"),
    )
    assert all(parameter not in gradients for parameter in shared)
    assert all(torch.equal(old, parameter) for old, parameter in zip(frozen, shared))
    store.freeze_after_phase1(model, "phase1-test-hash")
    assert store._embeddings is not None
    assert not store.values(torch.tensor([0]), keys[0].topology).requires_grad


def test_full_finetune_three_phase_artifact_freezes_encoders(
    tiny_prepared: Stage3Config, tmp_path: Path,
) -> None:
    config = _tiny_three_phase(tiny_prepared)
    prepared = load_prepared_stage3(config)
    keys = finetune_object_keys(prepared)

    class Batch:
        def __init__(self, values: torch.Tensor) -> None:
            self.values = values

        def to(self, device: torch.device) -> "Batch":
            return Batch(self.values.to(device))

    class Packer:
        def __call__(self, samples: list[dict[str, torch.Tensor]]) -> Batch:
            return Batch(torch.stack([sample["values"] for sample in samples]))

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.project = torch.nn.Linear(4, 4)

        def encode_entity(self, batch: Batch) -> SimpleNamespace:
            return SimpleNamespace(entity_embedding=self.project(batch.values))

    seed_everything(11)
    model = FinetuneStage3Model(
        config.model, prepared["registry"], 4,
        group_configs=config.groups, task_configs=config.tasks,
        task_private_recipes={task: config.resolved_private_recipe(task) for task in config.tasks},
        backbone=Backbone(),
        object_encoder=ObjectEncoder(4, 2, num_layers=1, feedforward_dim=8, dropout=0.0),
    )
    model.set_trainable_owners(tuple(set(model.parameter_ownership().values())))
    samples = {
        (role, smiles): {"values": torch.tensor([
            float(len(smiles)), float(ROLE_TO_ID[role]), 1.0, 0.5
        ])}
        for key in keys for role, smiles in key.slots
    }
    store = LiveRepresentationStore(model, Packer(), keys, samples)
    model.eval()
    with torch.no_grad():
        reference = torch.stack([
            store.values(torch.tensor([index]), key.topology)[0]
            for index, key in enumerate(keys)
        ])
    matched = {**prepared, "objects": {**prepared["objects"], "embeddings": reference}}
    assert validate_initial_representation(model, store, matched) <= 1e-5
    mismatched = reference.clone()
    mismatched[0] += 1
    with pytest.raises(ValueError, match="differs from Base prepared"):
        validate_initial_representation(
            model, store,
            {**prepared, "objects": {**prepared["objects"], "embeddings": mismatched}},
        )
    active = tuple(config.tasks)
    train_data = {
        task: Stage3TaskDataset(config.data.artifacts_dir, 1, task, "train")
        for task in active
    }
    valid_data = {
        task: Stage3TaskDataset(config.data.artifacts_dir, 1, task, "valid")
        for task in active
    }
    features = {
        "artifact_sha256": "feature-hash",
        "stage2_encoder_sha256": "encoder-hash",
        "stage2_encoder_identity": TEST_ENCODER_IDENTITY["hash"],
    }
    recipe = FinetuneRecipe(5e-6, 1.5e-5, 2, 0.05, 0.1)
    plan = resolved_plan(config, recipe, 1, model, prepared, train_data, features)
    output = tmp_path / "fine-tune-train"
    result = run_three_phase_training(
        config=config, fold=1, output_dir=output, resume_from=None,
        model=model, registry=prepared["registry"], active=active,
        train_data=train_data, valid_data=valid_data,
        representations=store, normalizations=prepared["normalization"]["fold1"],
        plan=plan, device=torch.device("cpu"),
    )
    assert result[0]["phase"] == "three_phase_final"
    phase1 = torch.load(output / "phase_1/checkpoint_epoch_00002.pt", weights_only=False)
    final = torch.load(output / "three_phase_final.pt", weights_only=False)
    manifest = json.loads((output / "three_phase_final.json").read_text())
    assert final["kind"] == "ilume_stage3_encoder_finetune_three_phase_final"
    assert final["training_identity"]["payload"]["contract_version"] == 8
    assert manifest["encoder_state_hashes"] == final["encoder_state_hashes"]
    for name in phase1["model"]:
        if name.startswith(("stage1_encoder.", "stage2_object_encoder.")):
            assert torch.equal(phase1["model"][name], final["model"][name])
    assert (output / "phase_2/stitched.pt").is_file()
    assert (output / "phase_3/experiment__a/checkpoint_epoch_00001.pt").is_file()
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        _load_model(
            config, prepared, output / "three_phase_final.pt", 1, 2,
            torch.device("cpu"), taskwise_refined=False, three_phase_final=True,
        )

    def rebuild(*_args: Any, **_kwargs: Any) -> tuple[FinetuneStage3Model, LiveRepresentationStore]:
        fresh = FinetuneStage3Model(
            config.model, prepared["registry"], 4,
            group_configs=config.groups, task_configs=config.tasks,
            task_private_recipes={task: config.resolved_private_recipe(task) for task in config.tasks},
            backbone=Backbone(),
            object_encoder=ObjectEncoder(4, 2, num_layers=1, feedforward_dim=8, dropout=0.0),
        )
        return fresh, LiveRepresentationStore(fresh, Packer(), keys, samples)

    with patch(
        "ablations.stage3_full_finetune.evaluate.load_features", return_value=features
    ), patch(
        "ablations.stage3_full_finetune.evaluate.build_model_and_store", side_effect=rebuild
    ):
        evaluated = evaluate_finetuned(
            config, recipe, feature_dir=tmp_path, checkpoint_dir=output,
            split="valid", fold=1, predictions_dir=tmp_path / "predictions",
        )
    assert set(evaluated["tasks"]) == set(active)
    assert set(evaluated["gate_diagnostics"]) == set(active)
    assert evaluated["ablation"] == "stage3_full_finetune"

    historical = tmp_path / "historical-base"
    (historical / "evaluate_valid/fold1").mkdir(parents=True)
    (historical / "train/fold1").mkdir(parents=True)
    (historical / "evaluate_valid/fold1/summary.json").write_text(
        json.dumps(evaluated), encoding="utf-8"
    )
    (historical / "train/fold1/three_phase_final.json").write_text(
        json.dumps({"training_identity": {"payload": {
            "contract_version": 6,
            "plan": {"math": {"gradient_aggregation": "weighted_owner_raw_v1"},
                     "active_tasks": list(active),
                     "prepared_identity": plan["prepared_identity"]},
        }}}), encoding="utf-8"
    )
    comparison = compare_historical_base(
        evaluated, historical, split="valid", fold=1, training_tasks=active,
        prepared_identity=plan["prepared_identity"],
    )
    assert comparison["macro_normalized_mae"]["delta"] == 0.0
    contaminated = json.loads((historical / "evaluate_valid/fold1/summary.json").read_text())
    contaminated["reporting"]["comparison_identity"]["hash"] = "wrong-data"
    (historical / "evaluate_valid/fold1/summary.json").write_text(
        json.dumps(contaminated), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="evaluation data contract"):
        compare_historical_base(
            evaluated, historical, split="valid", fold=1, training_tasks=active,
            prepared_identity=plan["prepared_identity"],
        )

    seed_everything(11)
    resumed_model, resumed_store = rebuild()
    resumed_model.set_trainable_owners(tuple(set(resumed_model.parameter_ownership().values())))
    resumed = run_three_phase_training(
        config=config, fold=1, output_dir=output, resume_from=output,
        model=resumed_model, registry=prepared["registry"], active=active,
        train_data=train_data, valid_data=valid_data,
        representations=resumed_store,
        normalizations=prepared["normalization"]["fold1"],
        plan=plan, device=torch.device("cpu"),
    )
    assert resumed[0]["validation"] == result[0]["validation"]

    partial = tmp_path / "fine-tune-phase1-resume"
    partial.mkdir()
    shutil.copy2(output / "resolved_training_plan.json", partial)
    shutil.copytree(output / "phase_1", partial / "phase_1")
    seed_everything(11)
    partial_model, partial_store = rebuild()
    partial_model.set_trainable_owners(tuple(set(partial_model.parameter_ownership().values())))
    run_three_phase_training(
        config=config, fold=1, output_dir=partial, resume_from=partial,
        model=partial_model, registry=prepared["registry"], active=active,
        train_data=train_data, valid_data=valid_data,
        representations=partial_store,
        normalizations=prepared["normalization"]["fold1"],
        plan=plan, device=torch.device("cpu"),
    )
    partial_final = torch.load(partial / "three_phase_final.pt", weights_only=False)
    assert partial_final["model_state_hash"] == final["model_state_hash"]
