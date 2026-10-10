from __future__ import annotations

import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
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
import scripts.stage3.train as train_launcher
from stage1.descriptors import calculate_descriptors, rdkit_descriptor_names
from stage1.features import ROLE_TO_ID
from stage2.model import ObjectEncoder
from stage2.home_contract import SOURCE_GROUPS, state_hash
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
    Stage3ObjectEncoderPhase1Config,
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
    STAGE3_ARTIFACT_KIND,
    ResolvedTaskSpec,
    Stage3TaskDataset,
    Stage3RepresentationStore,
    collect_object_keys,
    composite_steps_per_epoch,
    raw_task_steps,
    resolve_batch_allocation,
    resolve_raw_batch_allocation,
    resolve_task_registry,
    shuffled_epoch_indices,
    source_path,
)
from stage3.evaluate import _load_model, evaluate_checkpoints, resolve_stage3_evaluation_identity
from stage3.evaluate import _configured_study_id, _reporting_model
from stage3.identity import build_stage3_prepared_identity, build_stage3_training_identity, metadata_identity, resolve_stage3_prepared_identity
from stage3.home import load_source as load_home_source
from stage3.model import (
    GLOBAL,
    Stage3SparseModel,
    group_owner,
    private_owner,
    summarize_task_gate_observations,
    task_gate_observations,
)
from stage3.object_phase1 import (
    OBJECT_ENCODER_OWNER, ObjectPhase1Model, ObjectPhase1Representations,
    validate_encoder_source, validate_initial_object_embeddings,
)
from stage3.gradient_assembly import assemble_owner_gradients
from stage3.prepare import load_prepared_stage3, materialize_object_slots, prepare_stage3
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
from stage3.three_phase import run_three_phase_training


def test_stage2_home_source_mapping_and_state_boundary() -> None:
    from stage2.home_contract import (
        SOURCE_GROUPS, load_transferable_state, source_task_specs,
        state_hash, transferable_state,
    )

    tasks = tuple(
        SimpleNamespace(
            task_id=task,
            target_columns=tuple(f"y{index}" for index in range(11))
            if task == "simulation/simulated_qm_elec_hf" else ("y",),
            condition_columns=("temperature_K",) if task in {
                "simulation/density", "simulation/heat_capacity",
                "simulation/thermal_expansion", "simulation/heat_of_vaporization",
            } else (),
            topology="interaction" if task == "simulation/transfer_organic" else
                "ionic_liquid" if task in {
                    "simulation/density", "simulation/heat_capacity",
                    "simulation/thermal_expansion", "simulation/heat_of_vaporization",
                } else "single_entity",
        )
        for task in SOURCE_GROUPS
    )
    registry = SimpleNamespace(tasks=tasks, task_ids=tuple(task.task_id for task in tasks))
    specs = source_task_specs(registry)
    assert len(specs) == 19
    assert {spec.meta_group for spec in specs.values()} == {
        "thermophysical", "solvation", "electronic_structure",
    }
    config = load_stage3_config("configs/v3/stage3/base.yaml")
    source = Stage3SparseModel(
        config.model, specs, 16,
        group_configs={
            **{name: config.groups[name] for name in ("thermophysical", "solvation")},
            "electronic_structure": config.groups["thermophysical"],
        },
    )
    state = transferable_state(source)
    assert state
    assert all("electronic_structure" not in name for name in state)
    assert all(not name.startswith(("private_experts.", "task_gates.", "towers.")) for name in state)
    target = Stage3SparseModel(
        config.model,
        {**specs, "experiment/new": replace(next(iter(specs.values())), task_id="experiment/new", meta_group="transport")},
        16,
        group_configs={
            **{name: config.groups[name] for name in ("thermophysical", "solvation", "transport")},
            "electronic_structure": config.groups["thermophysical"],
        },
    )
    before = {name: value.clone() for name, value in target.state_dict().items()}
    names = load_transferable_state(target, state, state_hash(state))
    assert set(names) == set(state)
    for name, value in target.state_dict().items():
        assert torch.equal(value, state[name] if name in state else before[name])
    with pytest.raises(ValueError, match="hash mismatch"):
        load_transferable_state(target, state, "wrong")
    incomplete = {name: value for name, value in state.items() if name != names[0]}
    with pytest.raises(ValueError, match="incomplete"):
        load_transferable_state(target, incomplete, state_hash(incomplete))



@pytest.mark.parametrize("number", [1, 2, 3, 4])
def test_home_candidate_transfer_shapes(number: int) -> None:
    from stage2.home_config import load_home_recipe
    from stage2.home_contract import SOURCE_GROUPS, load_transferable_state, source_task_specs, state_hash, transferable_state
    from stage2.home_model import SimulationHoME

    name = f"base1-{number}"
    stage2 = load_home_recipe(f"configs/v3/stage2/candidates/{name}.yaml")
    stage3 = load_stage3_config(f"configs/v3/stage3/candidates/{name}.yaml")
    tasks = tuple(SimpleNamespace(
        task_id=task, target_columns=("y",), condition_columns=(),
        topology="single_entity",
    ) for task in SOURCE_GROUPS)
    registry = SimpleNamespace(tasks=tasks, task_ids=tuple(task.task_id for task in tasks))
    backbone = torch.nn.Module()
    backbone.entity_dim = 16
    backbone.atom_dim = 8
    backbone.config = SimpleNamespace(model=SimpleNamespace(n_heads=8))
    source = Stage3SparseModel(stage3.model, source_task_specs(registry), 16, group_configs={**stage3.groups, "electronic_structure":stage3.groups["thermophysical"]})
    state = transferable_state(source)
    target = Stage3SparseModel(stage3.model, source_task_specs(registry), 16,
                               group_configs=stage3.groups)
    names = load_transferable_state(target, state, state_hash(state))
    assert set(names) == set(state)
    assert all(torch.equal(target.state_dict()[key], value) for key, value in state.items())


def test_formal_home_source_rejects_old_kind(tmp_path):
    from stage2.home_artifact import load_home_final
    artifact=tmp_path / 'stage2_final.pt'
    torch.save({'kind':'ilume_stage2_home_final_v5','format_version':5},artifact)
    artifact.with_suffix('.json').write_text(json.dumps({'artifact':artifact.name,'artifact_sha256':sha256_file(artifact),'kind':'ilume_stage2_home_final_v5'}))
    with pytest.raises(ValueError,match='kind'): load_home_final(artifact)


@pytest.mark.parametrize("private_only", [True])
def test_simulation_phase2_phase3_plan_and_identity(private_only: bool) -> None:
    from stage3.simulation import SIMULATION_TASKS, extend_simulation_plan
    from stage3.three_phase import _final_kind, _scope_kind

    config = load_stage3_config("configs/v4/stage3/base.yaml" if private_only else "configs/v3/stage3/base.yaml")
    plan = {
        "data": {
            "N_t": {"experiment/density": 3}, "B_t": {"experiment/density": 2},
            "task_steps": {"experiment/density": 2},
            "epoch_exposures": {"experiment/density": 3},
            "effective_composite_batch_size": 2,
        },
        "phases": {
            "phase1": {"epochs": 15, "owners": {"GLOBAL": {"actual_update_budget": 30}}},
            "phase2": {"branches": {"thermophysical": {
                "epochs": 4, "steps_per_epoch": 2,
                "owners": {"GROUP:thermophysical": {
                    "updates_per_epoch": 2, "actual_update_budget": 8,
                }},
            }}},
            "phase3": {"branches": {}},
        },
    }
    tasks = SIMULATION_TASKS[:2] if private_only else SIMULATION_TASKS
    if private_only:
        plan["representation_contract"] = "entity_home_v4"
    model = SimpleNamespace(
        task_specs={task: SimpleNamespace(meta_group=SOURCE_GROUPS[task]) for task in tasks},
        resolved_capacity_recipe=lambda: {
            "groups": {group: {} for group in ("thermophysical", "electronic_structure")},
            "tasks": {task: {} for task in SIMULATION_TASKS},
        },
    )
    data = SimpleNamespace(train={task: range(1025 if private_only else 300 if task == "simulation/homo" else 2) for task in tasks}, data_identity="data-hash")
    extend_simulation_plan(plan, config, model, data, {"full_model_state_hash": "source-hash"})
    assert plan["phases"]["phase1"]["owners"]["GLOBAL"]["actual_update_budget"] == 30
    if private_only:
        branch = plan["phases"]["phase2"]["branches"]["thermophysical"]
        assert branch["steps_per_epoch"] == 5
        assert branch["owners"]["GROUP:thermophysical"]["updates_per_epoch"] == 2
        assert branch["owners"]["GROUP:thermophysical"]["actual_update_budget"] == 8
        assert "shared_group_task_weight" not in plan["simulation_training"]["recipe"]
        assert plan["simulation_training"]["phase2_gradient_policy"] == "simulation_private_only_v1"
        assert plan["format_version"] == 14
        assert _final_kind(plan) == "ilume_stage3_entity_home_three_phase_final_v4"
    else:
        assert plan["phases"]["phase2"]["branches"]["electronic_structure"]["epochs"] == 4
        assert plan["phases"]["phase2"]["branches"]["electronic_structure"]["owners"]["GROUP:electronic_structure"]["nominal_lr"] == 7.5e-5
        assert plan["data"]["task_steps"]["simulation/homo"] == 2
        assert plan["format_version"] == 10
        assert plan["simulation_training"]["recipe"]["shared_group_task_weight"] == 0.1
        assert _final_kind(plan) == "ilume_stage3_home_simulation_three_phase_final_v2"
        assert _scope_kind(plan, "owner_delta") == "ilume_stage3_home_simulation_three_phase_v2_owner_delta"
    assert all(plan["phases"]["phase3"]["branches"][task]["epochs"] == 8 for task in tasks)




def test_stage2_home_masked_micro_loss_and_identity() -> None:
    from stage2.home_config import load_home_recipe
    from stage2.home_train import _loss_for_micro

    experiment = load_home_recipe("configs/v3/stage2/base.yaml")
    assert experiment.stage2_microbatch_size == 256
    assert experiment.stage2_epochs == 10
    assert experiment.stage2.training.backbone_frozen_epochs == 0
    assert experiment.stage2.loss.lambda_teacher == 0
    predictions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets = torch.zeros_like(predictions)
    mask = torch.tensor([[True, False], [False, True]])
    data = SimpleNamespace(targets=targets, target_mask=mask)
    parts = (
        SimpleNamespace(row_indices=torch.tensor([0])),
        SimpleNamespace(row_indices=torch.tensor([1])),
    )
    observed = sum(
        _loss_for_micro(
            "simulation/simulated_qm_elec_hf", predictions[part.row_indices],
            part, data, torch.tensor([0, 1]),
        )
        for part in parts
    )
    expected = (torch.nn.functional.smooth_l1_loss(predictions[0, 0], targets[0, 0])
                + torch.nn.functional.smooth_l1_loss(predictions[1, 1], targets[1, 1])) / 2
    assert torch.equal(observed, expected)
    plan = {"schedule_mode": "three_phase"}
    from stage3.three_phase import _final_kind, _scope_kind
    assert _final_kind(plan) == "ilume_stage3_three_phase_final"
    plan["stage2_pretraining"] = {"source": "x"}
    assert _final_kind(plan) == "ilume_stage3_home_three_phase_final_v1"
    assert _scope_kind(plan, "1_full") == "ilume_stage3_home_three_phase_v1_1_full"


def test_home_microbatch_config_and_identity(tmp_path: Path) -> None:
    import yaml
    from stage2.home_config import load_home_recipe
    from stage2.home_train import training_identity

    raw = yaml.safe_load(Path("configs/v3/stage2/base.yaml").read_text())
    identities = []
    for size in (8, 64, 256):
        raw["home"]["microbatch_size"] = size
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump(raw))
        experiment = load_home_recipe(path)
        assert experiment.stage2_microbatch_size == size
        with patch("stage2.home_train.sha256_file", return_value="source"):
            identities.append(training_identity(experiment, {"hash": "data"}, {})["hash"])
    assert len(set(identities)) == 3
    for size in (0, 257, True, 8.5, "8"):
        raw["home"]["microbatch_size"] = size
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="nine-task|microbatch|home"):
            load_home_recipe(path)


@pytest.mark.parametrize("task", ["simulation/density", "simulation/simulated_qm_elec_hf", "simulation/partial_atomic_charge"])
def test_home_logical_batch_updates_once(task: str) -> None:
    from stage2.home_train import _train_batch
    from stage2.data import Stage2BatchDescriptor

    class Packed:
        def __init__(self, indices):
            self.row_indices = indices
            self.atom_targets = SimpleNamespace(
                values=torch.zeros(len(indices)), mask=torch.ones(len(indices), dtype=torch.bool),
                atom_sample_indices=torch.arange(len(indices)),
            )

        def to(self, device, *, non_blocking):
            assert device.type == "cpu" and not non_blocking
            return self

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def predict(self, task_id, packed, data):
            result = self.weight * (packed.row_indices.float() + 1)
            return result if task_id.endswith("partial_atomic_charge") else result[:, None].expand(-1, 2)

    mask = torch.tensor([[True, False], [False, True], [True, True], [True, False], [True, True]])
    if task == "simulation/density":
        mask[:] = True
    data = SimpleNamespace(targets=torch.zeros((5, 2)), target_mask=mask)
    config = SimpleNamespace(training=SimpleNamespace(amp_dtype="fp32", max_grad_norm=1.0))
    states = []
    losses = []
    for size in (1, 2, 5):
        model = Model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        descriptor = Stage2BatchDescriptor(task, torch.arange(5))
        losses.append(_train_batch(
            model, descriptor, tuple(Packed(part) for part in descriptor.indices.split(size)),
            data, torch.device("cpu"), optimizer, scheduler, tuple(model.parameters()), config, 0.7,
        ))
        assert scheduler.last_epoch == 1
        assert optimizer.state[model.weight]["step"].item() == 1
        states.append(model.weight.detach().clone())
    assert losses == pytest.approx([losses[0]] * 3, abs=1e-6)
    assert all(torch.allclose(states[0], value, atol=1e-7) for value in states)
    model = Model()
    model.weight.data.fill_(float("nan"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    with pytest.raises(RuntimeError, match="Non-finite"):
        _train_batch(model, descriptor, (Packed(descriptor.indices),), data, torch.device("cpu"),
                     optimizer, scheduler, tuple(model.parameters()), config, 0.7)
    assert not optimizer.state and scheduler.last_epoch == 0


def test_home_prefetch_preserves_order_and_propagates_packing_errors(monkeypatch) -> None:
    from stage2.home_train import _prefetched_batches
    from stage2.data import Stage2BatchDescriptor

    schedule = [Stage2BatchDescriptor(f"task{index}", torch.arange(5)) for index in range(4)]
    calls = []
    def pack(descriptor, *_args, **kwargs):
        assert kwargs["pin_memory"]
        calls.append((descriptor.task, descriptor.indices.tolist()))
        return descriptor.indices.clone()
    monkeypatch.setattr("stage2.home_train.pack_stage2_batch", pack)
    rng_before = torch.random.get_rng_state().clone()
    with _prefetched_batches(schedule, {}, None, None, 2, pin_memory=True) as batches:
        observed = list(batches)
    assert [item[0].task for item in observed] == [item.task for item in schedule]
    assert all(torch.equal(torch.cat(item[1]), item[0].indices) for item in observed)
    assert [task for task, _ in calls] == [task.task for task in schedule for _ in range(3)]
    assert torch.equal(rng_before, torch.random.get_rng_state())
    def fail(*_args, **_kwargs):
        raise ValueError("packing failure")
    monkeypatch.setattr("stage2.home_train.pack_stage2_batch", fail)
    with pytest.raises(ValueError, match="packing failure"):
        with _prefetched_batches(schedule, {}, None, None, 256, pin_memory=False) as batches:
            next(batches)


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

    v2 = load_stage3_config("configs/v3/stage3/base.yaml")
    no_stage1 = load_stage3_config("configs/ablations/no_stage1_stage3.yaml")
    no_stage2 = load_stage3_config("configs/ablations/no_stage2_stage3.yaml")
    assert len(v2.enabled_task_ids) == 20
    assert v2.data.split_policy == no_stage1.data.split_policy == no_stage2.data.split_policy == "system"
    assert v2.model == config.model
    assert all(group.experts is not None for group in v2.groups.values())
    assert all(v2.resolved_private_recipe(task).phase3_epochs >= 0 for task in v2.tasks)
    assert v2.training.sampling_mode == "raw"
    assert v2.training.joint_gradient_clip_mode == "ownership"
    assert v2.training.schedule_mode == "three_phase"
    assert v2.initialization.home_mode == no_stage1.initialization.home_mode == "trained"
    assert no_stage2.initialization.home_mode == "no_stage2"
    assert "virtual_min_size" not in v2.to_dict()["training"]
    for path in (
        "configs/v2/stage3/splits/random.yaml",
        "configs/v2/stage3/splits/system.yaml",
        "configs/v2/stage3/splits/individual.yaml",
        "configs/ablations/no_stage1_stage3.yaml",
        "configs/ablations/no_stage2_stage3.yaml",
    ):
        active = load_stage3_config(path)
        assert all(active.groups[group] == v2.groups[group] for group in active.groups)
        assert ("electronic_structure" in active.groups) == ("no_stage1" in path)
        assert active.tasks == v2.tasks
        assert active.training.three_phase == v2.training.three_phase








def test_v2_native_split_configs_match_materialized_task_subsets(tmp_path: Path) -> None:
    expected = {
        "system": ({"il", "il_solute", "solute_solvent", "random"}, 20),
        "random": ({"random"}, 20),
        "individual": ({"cation", "solvent", "random"}, 20),
    }
    root = Path("configs/v2/stage3/splits")
    for name, (strategies, task_count) in expected.items():
        config = load_stage3_config(root / f"{name}.yaml")
        # Historical task contracts use isolated fixtures, not the evolving live catalog.
        rows = []
        for task_id, task in config.tasks.items():
            slots = task.primary_slots + task.partner_slots
            system = ("il_solute" if slots == ("cation", "anion", "solute") else
                      "il" if slots == ("cation", "anion") else
                      "solute_solvent" if slots == ("solute", "solvent") else "solute")
            rows.append(_catalog_row(task_id, "value", ";".join(slots), "", system,
                                     "random;il;il_solute;solute_solvent;cation;anion;solute;solvent",
                                     task.unique_systems or 2))
        catalog = tmp_path / "historical_catalog.csv"
        _write_csv(catalog, list(rows[0]), rows)
        original_config = config
        config = replace(config, data=replace(config.data, task_catalog=catalog, stage3_dir=tmp_path / "stage3"))
        registry = resolve_task_registry(config)
        for spec in registry.values():
            directory = {"il": "IL", "il_solute": "IL-solute", "solute_solvent": "solute-solvent"}.get(spec.split_strategy, spec.split_strategy)
            for fold in range(1, 6):
                _write_csv(config.data.stage3_dir / spec.task_id / directory / f"fold{fold}.csv", ["value"], [{"value": 1}])
        enabled = {
            task_id: spec for task_id, spec in registry.items() if spec.enabled
        }
        assert len(enabled) == task_count
        assert {spec.split_strategy for spec in enabled.values()} == set(strategies)
        assert original_config.data.artifacts_dir == Path(
            f"outputs/v2/stage3/splits/{name}/prepare/artifacts"
        )
        assert config.preparation.cache_dir == Path(
            f"outputs/v2/stage3/splits/{name}/prepare/object_cache"
        )
        assert config.training.sampling_mode == "raw"
        assert config.training.joint_gradient_clip_mode == "ownership"
        assert config.training.schedule_mode == "three_phase"
        if name == "system":
            assert {spec.system_type for spec in enabled.values()} == {"il", "il_solute", "solute_solvent", "solute"}
        elif name == "individual":
            assert sum(spec.split_strategy == "cation" for spec in enabled.values()) == 18
            assert enabled["experiment/transfer_organic"].split_strategy == "solvent"
        for spec in enabled.values():
            for fold in range(1, 6):
                assert source_path(config, spec, fold).is_file()


def test_three_phase_config_and_task_specific_gate_contract() -> None:
    config = load_stage3_config("configs/v3/stage3/base.yaml")
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










def test_entity_home_electrochemical_base_capacity() -> None:
    from common.entity_inputs import EntityInputs
    from stage3.three_phase import _checkpoint_format, _validate_checkpoint_common

    config = load_stage3_config("configs/v4/stage3/base.yaml")
    tasks = ("experiment/anodic_potential_limit", "experiment/cathodic_potential_limit")
    registry = resolve_task_registry(config)
    model = Stage3SparseModel(
        config.model, {task: registry[task] for task in tasks}, 1024,
        group_configs=config.groups, task_configs=config.tasks,
        task_private_recipes={task: config.resolved_private_recipe(task) for task in tasks},
        entity_inputs=True, initialization_seed=42,
    )
    group = config.groups["electrochemical"]
    assert (group.phase1.epochs, group.phase1.lr) == (15, 2e-4)
    assert (group.phase2.epochs, group.phase2.lr) == (4, 1e-4)
    previous = replace(config, groups={**config.groups, "electrochemical": replace(
        group, phase2=replace(group.phase2, epochs=20),
    )})
    prepared = {"metadata": {
        "kind": "ilume_stage3_entity_sparse_data_v4",
        "semantic": {"identities": {
            "prepared": semantic_identity("test.prepared", {}),
            "stage1_encoder": semantic_identity("test.encoder", {}),
        }},
    }}
    plans = [build_resolved_training_plan(
        recipe, 1, model, {task: range(2) for task in tasks}, tasks, prepared, {}, {},
    ) for recipe in (previous, config)]
    before, after = [plan["phases"]["phase2"]["branches"]["electrochemical"] for plan in plans]
    assert before["epochs"] == 20 and after["epochs"] == 4
    for task in tasks:
        assert config.resolved_private_recipe(task) == previous.resolved_private_recipe(task)
        owner = f"PRIVATE:{task}"
        assert before["owners"][owner] == after["owners"][owner]
        assert after["owners"][owner]["effective_epochs"] == 4
    assert plans[0]["phases"]["phase1"] == plans[1]["phases"]["phase1"]
    assert plans[0]["phases"]["phase3"] == plans[1]["phases"]["phase3"]
    checkpoint = {
        "format_version": _checkpoint_format(plans[0]), "stage": "stage3",
        "fold": 1, "phase": "phase2",
        "training_identity": build_stage3_training_identity(plans[0]),
    }
    _validate_checkpoint_common(checkpoint, phase="phase2", fold=1, plan=plans[0])
    with pytest.raises(ValueError, match="training_identity"):
        _validate_checkpoint_common(checkpoint, phase="phase2", fold=1, plan=plans[1])
    assert len(model.l1_group_experts["electrochemical"]) == 2
    assert len(model.l2_group_experts["electrochemical"]) == 2
    for experts, input_dim in (
        (model.l1_group_experts["electrochemical"], 2490),
        (model.l2_group_experts["electrochemical"], 1024),
    ):
        for expert in experts:
            assert expert.layers[0].in_features == input_dim
            assert expert.layers[0].out_features == 768
            assert expert.layers[3].out_features == 1024
    assert sum(p.numel() for p in model.parameters_for_owner(group_owner("electrochemical"))) == 8_557_430
    primary = EntityInputs.from_slots(torch.randn(2, 2, 1241), torch.tensor([[0, 1], [0, 1]]))
    model.eval()
    for task in tasks:
        key = task.replace("/", "__")
        assert registry[task].condition_columns == ("temperature_K",)
        assert len(model.private_experts[key]) == 1
        assert model.private_experts[key][0].layers[0].in_features == 1024
        assert model.private_experts[key][0].layers[0].out_features == 512
        assert model.private_experts[key][0].layers[3].out_features == 1024
        assert model.towers[key].layers[0].out_features == 256
        assert model.condition_films[key].network[0].in_features == 1
        assert model.condition_films[key].network[0].out_features == 128
        assert (model.task_gates[key].in_features, model.task_gates[key].out_features) == (2048, 5)
        for module in (model.private_experts[key], model.towers[key], model.condition_films[key]):
            assert all(layer.p == 0.1 for layer in module.modules() if isinstance(layer, torch.nn.Dropout))
            assert all(model.parameter_ownership()[p] == private_owner(task) for p in module.parameters())
        assert all(model.parameter_ownership()[p] == private_owner(task) for p in model.task_gates[key].parameters())
        assert sum(p.numel() for p in model.parameters_for_owner(private_owner(task))) == 1_591_558
        with torch.no_grad():
            predictions = model(task, primary, torch.ones(2, 1)).predictions
        assert predictions.shape == (2,) and torch.isfinite(predictions).all()


def test_three_phase_private_capacity_ratios_follow_size_class() -> None:
    config = load_stage3_config("configs/v3/stage3/base.yaml")
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

    selected_config = replace(config, data=replace(config.data, split_strategies={}, cv_repeats={}),
                              tasks={task_id: task, "experiment/refractive_index": config.tasks["experiment/refractive_index"]})
    spec = resolve_task_registry(selected_config)[task_id]
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
        {dropout_task: resolve_task_registry(selected_config)[dropout_task]},
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
    payload = load_stage3_config("configs/v3/stage3/base.yaml").to_dict()
    payload = json.loads(json.dumps(payload))
    payload["tasks"]["experiment/density"][field] = value
    with pytest.raises(ValueError, match=message):
        stage3_config_from_dict(payload)


@pytest.mark.parametrize("value", (True, -0.01, 0.151, "0.15"))
def test_three_phase_private_dropout_override_is_bounded(value: object) -> None:
    payload = load_stage3_config("configs/v3/stage3/base.yaml").to_dict()
    payload = json.loads(json.dumps(payload))
    payload["tasks"]["experiment/density"]["model_overrides"] = {
        "private_dropout": value
    }
    with pytest.raises(ValueError, match="private_dropout must be in"):
        stage3_config_from_dict(payload)


def test_simulation_branches_stitch_and_resume(tiny_prepared: Stage3Config) -> None:
    from stage3.simulation import SIMULATION_TASKS, extend_simulation_plan

    config = _tiny_three_phase(tiny_prepared)
    config = replace(config, groups={
        **config.groups,
        "thermophysical": config.groups["g1"],
        "electronic_structure": config.groups["g1"],
    })
    prepared = load_prepared_stage3(config)
    representations = Stage3RepresentationStore(
        config.data.artifacts_dir, 1, prepared["objects"], prepared["metadata"]["kind"],
    )
    experimental = tuple(task for task, spec in prepared["registry"].items() if spec.enabled)
    template = prepared["registry"][experimental[0]]
    simulation_specs = {
        task: replace(
            template, task_id=task, target_column="value", condition_columns=(),
            meta_group=SOURCE_GROUPS[task], partner_mode="none",
            primary_slots=("molecule",), partner_slots=(),
        )
        for task in SIMULATION_TASKS
    }
    model = Stage3SparseModel(
        config.model, {**prepared["registry"], **simulation_specs},
        representations.output_dim, group_configs=config.groups,
        task_configs=config.tasks,
        task_private_recipes={task: config.resolved_private_recipe(task) for task in experimental},
    )
    initial_state = {name: value.clone() for name, value in model.state_dict().items()}
    train = {task: Stage3TaskDataset(config.data.artifacts_dir, 1, task, "train") for task in experimental}
    valid = {task: Stage3TaskDataset(config.data.artifacts_dir, 1, task, "valid") for task in experimental}
    normalization = prepared["normalization"]["fold1"]
    plan = build_resolved_training_plan(
        config, 1, model, train, experimental, prepared, {}, normalization,
    )
    config = replace(config, training=replace(
        config.training,
        simulation=load_stage3_config("configs/v3/stage3/base.yaml").training.simulation,
    ))

    class TinySimulationData:
        train = {task: range(2) for task in SIMULATION_TASKS}
        data_identity = "tiny-simulation"

        def compute_gradient(self, model, task, indices, device):
            parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
            values = torch.ones((len(indices), model.d_model), device=device)
            conditions = torch.empty((len(indices), 0), device=device)
            loss = (model(task, values, conditions).predictions - 0.25).square().mean()
            derivatives = torch.autograd.grad(loss, parameters, allow_unused=True)
            return {
                parameter: gradient.detach().float()
                for parameter, gradient in zip(parameters, derivatives, strict=True)
                if gradient is not None
            }, float(loss.detach())

        def validate_tasks(self, model, tasks, device):
            del model, device
            return {"simulation_tasks": {task: {"normalized_mae": 0.25} for task in tasks}} if tasks else {}

    simulation_data = TinySimulationData()
    extend_simulation_plan(plan, config, model, simulation_data, {"full_model_state_hash": "source"})
    output = config.data.artifacts_dir.parent / "simulation-three-phase"
    rows = run_three_phase_training(
        config=config, fold=1, output_dir=output, resume_from=None,
        model=model, registry=model.task_specs, active=experimental,
        train_data=train, valid_data=valid, representations=representations,
        normalizations=normalization, plan=plan, device=torch.device("cpu"),
        simulation_data=simulation_data,
    )
    assert rows[-1]["phase"] == "three_phase_final"
    final = torch.load(output / "three_phase_final.pt", map_location="cpu", weights_only=False)
    assert final["kind"] == "ilume_stage3_home_simulation_three_phase_final_v2"
    assert set(final["private_state_hashes"]) == set(experimental) | set(SIMULATION_TASKS)
    assert (output / "phase_2/g1/metrics.jsonl").is_file()
    assert (output / "phase_3/simulation__heat_of_vaporization/metrics.jsonl").is_file()
    model.load_state_dict(initial_state)
    resumed = run_three_phase_training(
        config=config, fold=1, output_dir=output, resume_from=output,
        model=model, registry=model.task_specs, active=experimental,
        train_data=train, valid_data=valid, representations=representations,
        normalizations=normalization, plan=plan, device=torch.device("cpu"),
        simulation_data=simulation_data,
    )
    assert resumed[-1]["phase"] == "three_phase_final"
    assert torch.load(output / "three_phase_final.pt", map_location="cpu", weights_only=False)["model_state_hash"] == final["model_state_hash"]


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
    base = load_stage3_config("configs/v3/stage3/base.yaml")
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
















def test_hydration_single_solute_preparation_and_train_only_scaler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _tiny_config(tmp_path)
    task = "experiment/hydration"
    row = _catalog_row(task, "hydration_kcal/mol", "solute", "temperature_K", "solute", "random")
    _write_csv(config.data.task_catalog, list(row), [row])
    config = replace(config, data=replace(config.data, split_policy="system", split_strategies={task: "random"}),
                     tasks={task: Stage3TaskConfig(meta_group="g1", primary_slots=("solute",))})
    fields = ["solute", "temperature_K", "hydration_kcal/mol"]
    for fold in range(1, 6):
        _write_csv(config.data.stage3_dir / task / "random" / "cv1" / f"fold{fold}.csv", fields,
                   [{"solute": "C", "temperature_K": 290 + fold, "hydration_kcal/mol": fold * 10},
                    {"solute": "CC", "temperature_K": 300 + fold, "hydration_kcal/mol": fold * 10 + 1}])
    monkeypatch.setattr("stage3.prepare.load_stage2_encoder_identity", lambda _: TEST_ENCODER_IDENTITY)
    def fake_materialize(config, object_keys, reporter=None):
        return torch.zeros(len(object_keys), 4), TEST_ENCODER_IDENTITY, {"hits": 0, "misses": len(object_keys)}
    with patch("stage3.prepare.materialize_object_embeddings", side_effect=fake_materialize):
        prepare_stage3(config)
    prepared = load_prepared_stage3(config)
    assert prepared["registry"][task].primary_slots == ("solute",)
    assert prepared["registry"][task].partner_slots == ()
    assert prepared["normalization"]["fold1"][task]["target"]["mean"] == pytest.approx(35.5)
    assert len(Stage3TaskDataset(config.data.artifacts_dir, 1, task, "train")) == 8
    assert len(Stage3TaskDataset(config.data.artifacts_dir, 1, task, "valid")) == 2
    assert len(Stage3TaskDataset(config.data.artifacts_dir, 1, task, "test")) == 0
    old = replace(config, tasks={"experiment/transfer": config.tasks[task]})
    with pytest.raises(ValueError):
        load_prepared_stage3(old)
    formal = load_stage3_config("configs/v3/stage3/base.yaml")
    assert formal.tasks[task].size_class == "small"
    assert formal.resolved_private_recipe(task).phase3_epochs == formal.training.three_phase.private_classes["small"].phase3_epochs
