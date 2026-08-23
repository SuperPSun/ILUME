from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

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
    Stage3TaskConfig,
    Stage3TrainingConfig,
    load_stage3_config,
)
from stage3.data import (
    ObjectKey,
    ResolvedTaskSpec,
    Stage3TaskDataset,
    balanced_virtual_indices,
    composite_steps_per_epoch,
    resolve_batch_allocation,
    resolve_task_registry,
)
from stage3.model import GLOBAL, Stage3SparseModel, group_owner, private_owner
from stage3.pcgrad import hierarchical_pcgrad
from stage3.prepare import materialize_object_embeddings, prepare_stage3
from stage3.evaluate import evaluate_checkpoints
from stage3.train import (
    STAGE3_CHECKPOINT_KIND,
    STAGE3_CHECKPOINT_VERSION,
    _load_plugin,
    checkpoint_epochs,
    compute_task_gradient,
    run_stage3_training,
)
from stage3.identity import build_stage3_training_identity, metadata_identity


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


def test_base_registry_and_config_defaults_are_explicit() -> None:
    config = load_stage3_config("configs/v1/stage3/base.yaml")
    assert sum(map(len, BASE_GROUP_TASKS.values())) == 21
    assert len(config.tasks) == 21
    assert len(config.groups) == 6
    assert config.data.split_policy == "prefer_il"
    assert config.training.microbatch_size == 1024
    assert config.training.checkpoint_interval_epochs == 10
    assert config.model.dropout == 0.10
    assert config.model.expert_hidden_ratio == 2.0
    assert checkpoint_epochs(100, 10) == tuple(range(10, 101, 10))
    assert checkpoint_epochs(23, 10) == (10, 20, 23)


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
    illegal = replace(
        config,
        data=replace(config.data, split_strategies={"experiment/a": "solvent"}),
    )
    with pytest.raises(ValueError, match="Illegal split strategy"):
        resolve_task_registry(illegal)
    missing_repeat = replace(
        config,
        data=replace(config.data, cv_repeats={"experiment/a": 2}),
    )
    with pytest.raises(FileNotFoundError, match="Missing Stage 3 split"):
        from stage3.data import source_path

        source_path(
            missing_repeat,
            resolve_task_registry(missing_repeat)["experiment/a"],
            1,
        )


def test_condition_missing_fails_before_frozen_encoding(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    path = config.data.stage3_dir / "experiment/b" / "IL" / "fold1.csv"
    rows = [
        {"cation": "[Na+]", "anion": "[Cl-]", "temperature_K": "", "b": 1},
        {"cation": "[K+]", "anion": "[Br-]", "temperature_K": 300, "b": 2},
    ]
    _write_csv(path, list(rows[0]), rows)
    with patch("stage3.prepare.materialize_object_embeddings") as encode:
        with pytest.raises(ValueError, match="Missing Stage 3 value"):
            prepare_stage3(config)
    encode.assert_not_called()


def test_cache_hit_miss_and_corruption(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    keys = (
        ObjectKey("il", (("cation", "[Na+]"), ("anion", "[Cl-]"))),
        ObjectKey("molecule", (("neutral", "C"),)),
    )

    class Encoder:
        encoder_identity = TEST_ENCODER_IDENTITY

        def encode(self, specs):
            return torch.ones(len(specs), 4)

    with patch(
        "stage3.prepare.load_stage2_encoder_identity",
        return_value=TEST_ENCODER_IDENTITY,
    ), patch("stage3.prepare.load_frozen_object_encoder", return_value=Encoder()) as load:
        first, _, audit = materialize_object_embeddings(config, keys)
        second, _, second_audit = materialize_object_embeddings(config, keys)
    assert torch.equal(first, second)
    assert audit == {"hits": 0, "misses": 2}
    assert second_audit == {"hits": 2, "misses": 0}
    assert load.call_count == 1
    cache_file = next(config.preparation.cache_dir.rglob("*.pt"))
    cache_file.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt Stage 3 object cache"):
        with patch(
            "stage3.prepare.load_stage2_encoder_identity",
            return_value=TEST_ENCODER_IDENTITY,
        ):
            materialize_object_embeddings(config, keys)


def test_model_has_unified_task_gate_and_no_l2_global_gate(tiny_prepared: Stage3Config) -> None:
    registry = resolve_task_registry(tiny_prepared)
    model = Stage3SparseModel(tiny_prepared.model, registry, 4)
    names = tuple(model.state_dict())
    assert any("l1_global_gate" in name for name in names)
    assert not hasattr(model, "l2_global_gate")
    assert not any("l2_global_gate" in name for name in names)
    assert set(model.parameter_ownership()) == set(model.parameters())
    assert len(model.condition_films) == 2
    final = model.condition_films["experiment__b"].network[-1]
    assert torch.count_nonzero(final.weight) == 0
    result = model(
        "experiment/c", torch.randn(2, 4), torch.randn(2, 1),
        partner_embedding=torch.randn(2, 4),
    )
    assert result.diagnostics["task_gate"].shape[-1] == 3
    assert torch.allclose(result.diagnostics["task_gate"].sum(-1), torch.ones(2))


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


def test_virtual_allocation_and_replication_are_exact() -> None:
    counts = {"a": 2, "b": 20, "c": 3}
    allocation = resolve_batch_allocation(counts, 8, 10)
    assert sum(allocation.values()) == 8
    assert allocation == {"a": 2, "b": 4, "c": 2}
    steps = composite_steps_per_epoch(counts, allocation, 10)
    assert steps == 5
    first = balanced_virtual_indices(3, 20, seed=7, epoch=2, task_id="a")
    second = balanced_virtual_indices(3, 20, seed=7, epoch=2, task_id="a")
    assert torch.equal(first, second)
    frequencies = torch.bincount(first, minlength=3)
    assert int(frequencies.max() - frequencies.min()) <= 1


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


def _plugin_checkpoint(
    path: Path, model: Stage3SparseModel, stage2_encoder_identity: str
) -> None:
    plan = {
        "fold": 1,
        "active_tasks": list(model.task_specs),
        "resolved_registry": {
            task: spec.to_dict() for task, spec in model.task_specs.items()
        },
        "groups": {},
        "data": {},
        "model": asdict(model.model_config),
        "optimizer": {},
        "scheduler": {},
        "math": {},
        "stage2_encoder_identity": stage2_encoder_identity,
        "prepared_identity": "test-prepared",
        "normalization_hash": "test-normalization",
        "ownership_manifest": model.ownership_manifest(),
        "plugin": {"mode": "scratch", "loaded_parameters": []},
        "trainable_parameters": sorted(name for name, _ in model.named_parameters()),
        "frozen_parameters": [],
    }
    model_state = model.state_dict()
    torch.save(
        {
            "identity_contract_version": IDENTITY_CONTRACT_VERSION,
            "kind": STAGE3_CHECKPOINT_KIND,
            "format_version": STAGE3_CHECKPOINT_VERSION,
            "stage": "stage3",
            "stage2_encoder_identity": stage2_encoder_identity,
            "resolved_registry": {
                task: spec.to_dict() for task, spec in model.task_specs.items()
            },
            "ownership_manifest": model.ownership_manifest(),
            "model": model_state,
            "model_state_hash": tensor_state_hash("stage3.model-state", model_state),
            "normalization": {task: {} for task in model.task_specs},
            "resolved_training_plan": plan,
            "training_identity": build_stage3_training_identity(plan),
        },
        path,
    )


def test_plugin_defaults_and_explicit_adaptation(tiny_prepared: Stage3Config) -> None:
    registry = resolve_task_registry(tiny_prepared)
    source_registry = {task: registry[task] for task in ("experiment/a", "experiment/b")}
    source_model = Stage3SparseModel(tiny_prepared.model, source_registry, 4)
    checkpoint = tiny_prepared.data.artifacts_dir.parent / "plugin.pt"
    stage2_identity = metadata_identity(
        json.loads((tiny_prepared.data.artifacts_dir / "metadata.json").read_text()),
        "stage2_encoder",
        context="test Stage 3 artifact",
    )["hash"]
    _plugin_checkpoint(checkpoint, source_model, stage2_identity)
    target = Stage3SparseModel(tiny_prepared.model, registry, 4)
    plugin = Stage3PluginConfig(checkpoint=checkpoint)
    config = replace(
        tiny_prepared,
        initialization=replace(tiny_prepared.initialization, plugin=plugin),
    )
    _load_plugin(config, target, stage2_identity)
    assert not any(parameter.requires_grad for parameter in target.parameters_for_owner(GLOBAL))
    assert not any(parameter.requires_grad for parameter in target.parameters_for_owner(group_owner("g1")))
    assert all(parameter.requires_grad for parameter in target.parameters_for_owner(group_owner("g2")))
    assert all(parameter.requires_grad for parameter in target.parameters_for_owner(private_owner("experiment/c")))

    adapted_model = Stage3SparseModel(tiny_prepared.model, registry, 4)
    adapted = replace(
        plugin,
        adaptation=Stage3PluginAdaptationConfig(
            global_scope=True, groups=("g1",), private_tasks=("experiment/a",)
        ),
    )
    adapted_config = replace(
        tiny_prepared,
        initialization=replace(tiny_prepared.initialization, plugin=adapted),
    )
    _load_plugin(adapted_config, adapted_model, stage2_identity)
    assert all(parameter.requires_grad for parameter in adapted_model.parameters_for_owner(GLOBAL))
    assert all(parameter.requires_grad for parameter in adapted_model.parameters_for_owner(group_owner("g1")))
    assert all(parameter.requires_grad for parameter in adapted_model.parameters_for_owner(private_owner("experiment/a")))


def test_short_training_checkpoint_and_resume_are_exact(tiny_prepared: Stage3Config) -> None:
    continuous = tiny_prepared.data.artifacts_dir.parent / "continuous"
    rows = run_stage3_training(tiny_prepared, 1, output_dir=continuous)
    assert [row["epoch"] for row in rows] == [1, 2]
    assert sorted(path.name for path in continuous.glob("checkpoint_*.pt")) == [
        "checkpoint_epoch_00001.pt", "checkpoint_epoch_00002.pt"
    ]
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
    evaluation = evaluate_checkpoints(
        tiny_prepared,
        continuous,
        split="valid",
        ensemble_folds=False,
        checkpoint_epoch=2,
        task_subset=("experiment/a",),
        fold=1,
    )
    assert evaluation["checkpoint_epoch"] == 2
    assert set(evaluation["tasks"]) == {"experiment/a"}

    incompatible = replace(
        tiny_prepared,
        training=replace(tiny_prepared.training, microbatch_size=1),
    )
    incompatible_output = tiny_prepared.data.artifacts_dir.parent / "incompatible"
    incompatible_output.mkdir()
    shutil.copy(continuous / "resolved_training_plan.json", incompatible_output)
    shutil.copy(continuous / "metrics.jsonl", incompatible_output)
    shutil.copy(continuous / "diagnostics.jsonl", incompatible_output)
    with pytest.raises(ValueError, match="semantic identity mismatch"):
        run_stage3_training(
            incompatible,
            1,
            output_dir=incompatible_output,
            resume_from=continuous / "checkpoint_epoch_00002.pt",
        )

    corrupt_output = tiny_prepared.data.artifacts_dir.parent / "corrupt-resume"
    corrupt_output.mkdir()
    shutil.copy(continuous / "resolved_training_plan.json", corrupt_output)
    (corrupt_output / "metrics.jsonl").write_text(first_metric + "\n")
    (corrupt_output / "diagnostics.jsonl").write_text(first_diag + "\n")
    corrupt_checkpoint = torch.load(
        continuous / "checkpoint_epoch_00001.pt",
        map_location="cpu",
        weights_only=False,
    )
    first_parameter = next(iter(corrupt_checkpoint["model"].values()))
    first_parameter.view(-1)[0] += 1
    corrupt_path = corrupt_output / "checkpoint_epoch_00001.pt"
    torch.save(corrupt_checkpoint, corrupt_path)
    with pytest.raises(ValueError, match="model state hash mismatch"):
        run_stage3_training(
            tiny_prepared,
            1,
            output_dir=corrupt_output,
            resume_from=corrupt_path,
        )


def test_stage3_scripts_configure_runtime_before_loading_operation() -> None:
    root = Path(__file__).resolve().parents[1]
    operations = {
        "prepare.py": "from stage3.prepare import prepare_stage3",
        "train.py": "from stage3.train import ",
        "evaluate.py": "from stage3.evaluate import (",
    }
    for filename, operation_import in operations.items():
        source = (root / "scripts" / "stage3" / filename).read_text()
        assert source.index("configure_process_runtime(config)") < source.index(
            operation_import
        )
