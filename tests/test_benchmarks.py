from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data
import yaml

from ablations.stage3_single_task_mlp.adapter import _manifest, _run_training_epochs, build_input_features
from ablations.stage3_single_task_mlp.model import Stage3SingleTaskMLP
from benchmarks.aifc.adapter import (
    ConditionStats as AIFCConditionStats,
    aifc_model_views,
    resolve_aifc_architecture,
)
from benchmarks.aifc.model import AIFCRegressor
from benchmarks.aifc.parity import validate_legacy_parity
from benchmarks.aifc.preprocessing import (
    FRAGMENT_BLOB,
    FRAGMENT_COMMIT,
    FRAGMENT_SHA256,
    FragmentScheme as AIFCFragmentScheme,
    batch_aifc_graphs,
    smiles_to_aifc_graph,
)
from benchmarks.aionopedia.adapter import (
    SampleStats as AIonopediaSampleStats,
    _prepare_split as prepare_aionopedia_split,
    _scheduled_factor as aionopedia_scheduled_factor,
)
from benchmarks.aionopedia.graph import smiles_to_graph as aionopedia_graph
from benchmarks.aionopedia.model import MultiModalRegressor as AIonopediaRegressor
from benchmarks.common.config import benchmark_config_from_dict, load_benchmark_config
from benchmarks.common.data import (
    configured_tasks,
    load_split,
    resolve_task,
    BenchmarkTask,
    RawDataset,
)
from benchmarks.common.engine import (
    EvaluationResult,
    TargetStats,
    ensemble_evaluation,
    evaluate_checkpoint,
    prepare_training,
    train_bundle,
)
from benchmarks.common.environment import (
    ENVIRONMENT_MARKER,
    ensure_benchmark_environment,
    environment_command,
)
from benchmarks.spmm.environment import spmm_asset_snapshot
from benchmarks.common.features import (
    BASIC_FEATURE_NAMES,
    BASIC_FEATURE_SCHEMA_VERSION,
    FeatureCache,
    FeaturePreprocessor,
    basic_molecular_statistics,
    feature_schema,
    raw_feature_matrix,
)
from benchmarks.common.summary import (
    SUMMARY_FILES,
    _ordered_radar_tasks,
    publish_summary,
)
from benchmarks.iltransr.adapter import (
    CharacterVocabulary as ILTransRCharacterVocabulary,
    ConditionStats as ILTransRConditionStats,
    TwoBucketBatchSampler as ILTransRTwoBucketBatchSampler,
    _condition_population as iltransr_condition_population,
    iltransr_model_views,
    resolve_iltransr_recipe,
)
from benchmarks.iltransr.model import (
    ILTransRRegressor,
    ILTransRTransformer,
    load_converted_transformer,
)
from benchmarks.llasmol.adapter import ConditionStats as LlaSMolConditionStats, SharedLlaSMolRegressor, SortishBatchSampler as LlaSMolSortishBatchSampler, _collate as llasmol_collate, _official_adapter_state, _prepare_split as prepare_llasmol_split, _scheduled_factor as llasmol_scheduled_factor, llasmol_model_views, llasmol_task_prefix
from benchmarks.spmm.adapter import ConditionStats as SPMMConditionStats, SharedSPMMRegressor, SortishBatchSampler as SPMMSortishBatchSampler, _collate as spmm_collate, _load_pretrained_encoder, _prepare_split as prepare_spmm_split, _row_token_lengths as spmm_row_token_lengths, _scheduled_learning_rate
from common.identity import require_compatible_identity, semantic_identity
from common.reporting import (
    REPORTING_SCHEMA_VERSION,
    comparison_identity,
    sanitize_task_id,
    write_prediction_csv,
)
import scripts.benchmarks.evaluate as benchmark_evaluate_launcher
from scripts.benchmarks.sweep import (
    _JobResult,
    _SweepState,
    _aggregate,
    _build_jobs,
    _schedule,
)
import scripts.benchmarks.sweep as sweep_module
import scripts.benchmarks.train as benchmark_train_launcher
from stage3.config import load_stage3_config


try:
    import chemprop  # noqa: F401
    from benchmarks.dmpnn.adapter import (
        ConditionStats,
        DMPNNTrainingBundle,
        _predict,
        _prepare_scalar,
        _scalar_dataset,
        build_dmpnn_model,
        train_dmpnn_bundle,
    )
except ModuleNotFoundError:
    HAS_CHEMPROP = False
else:
    HAS_CHEMPROP = True
DMPNN_ONLY = pytest.mark.skipif(
    not HAS_CHEMPROP, reason="chemprop is unavailable"
)

# --- Shared baseline and sweep contracts ---

CATALOG_FIELDS = (
    "catalog_schema_version", "stage", "task_id", "task_kind", "target_level",
    "source_file", "target_columns", "identity_columns", "condition_columns",
    "system_type", "simulation_method", "materialized_path", "label_source",
    "resource_manifest", "strategies",
)

class RecordingBar:
    def __init__(self, *, total: int, desc: str, unit: str, initial: int = 0):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.n = initial
        self.postfixes: list[dict[str, object]] = []
        self.descriptions = [desc]
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def update(self, amount: int) -> None:
        self.n += amount

    def set_postfix(self, values: dict[str, object]) -> None:
        self.postfixes.append(dict(values))

    def set_description(self, value: str) -> None:
        self.descriptions.append(value)

    def close(self) -> None:
        self.closed = True

class RecordingReporter:
    def __init__(self):
        self.bars: list[RecordingBar] = []

    def bar(
        self, *, total: int, desc: str, unit: str, initial: int = 0
    ) -> RecordingBar:
        bar = RecordingBar(total=total, desc=desc, unit=unit, initial=initial)
        self.bars.append(bar)
        return bar

def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def _tiny_config(tmp_path: Path, *, name: str = "mlp", targets: str = "value"):
    catalog = tmp_path / "task_catalog.csv"
    with catalog.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOG_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "catalog_schema_version": 1,
                "stage": 3,
                "task_id": "experiment/tiny",
                "task_kind": "observation",
                "target_level": "object",
                "source_file": "experiment/tiny.csv",
                "target_columns": targets,
                "identity_columns": "cation;anion",
                "condition_columns": "temperature_K",
                "system_type": "il",
                "simulation_method": "test",
                "materialized_path": "stage3/experiment/tiny",
                "label_source": "materialized_csv",
                "resource_manifest": "",
                "strategies": "il",
            }
        )
    fields = ["cation", "anion", "temperature_K", *targets.split(";")]
    molecules = [
        ("[Li+]", "[F-]"), ("[Na+]", "[Cl-]"), ("[K+]", "[Br-]"),
        ("[Rb+]", "[I-]"), ("C[N+](C)(C)C", "[Cl-]"), ("CC[N+](C)(C)C", "[Br-]"),
        ("CCC[N+](C)(C)C", "[I-]"), ("CCCC[N+](C)(C)C", "[Cl-]"),
    ]
    values = []
    for index, (cation, anion) in enumerate(molecules):
        row: dict[str, object] = {"cation": cation, "anion": anion, "temperature_K": 290 + index}
        for column, target in enumerate(targets.split(";")):
            row[target] = float(index + column * 0.5)
        values.append(row)
    for fold in range(1, 6):
        _write_csv(
            tmp_path / f"stage3/experiment/tiny/IL/fold{fold}.csv",
            fields,
            values[(fold - 1) % 4 : (fold - 1) % 4 + 2],
        )
    _write_csv(tmp_path / "stage3/experiment/tiny/test.csv", fields, values[6:])
    authority = tmp_path / "stage3.yaml"
    authority.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "stage3_dir": str(tmp_path / "stage3"),
                    "task_catalog": str(catalog),
                    "artifacts_dir": str(tmp_path / "unused-artifacts"),
                },
                "preparation": {"cache_dir": str(tmp_path / "unused-cache")},
                "initialization": {"stage2_encoder": str(tmp_path / "unused.pt")},
                "groups": {"tiny": {"enabled": True, "group_weight": 1.0}},
                "tasks": {"experiment/tiny": {"meta_group": "tiny"}},
                "training": {"device": "cpu", "amp_dtype": "none"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    if name == "mlp":
        formal = load_benchmark_config("configs/benchmarks/mlp.yaml")
        return replace(
            formal,
            data=replace(
                formal.data,
                data_root=tmp_path,
                task_catalog=catalog,
                stage3_authority_config=authority,
                feature_cache=tmp_path / "features.sqlite3",
            ),
            training={
                **formal.training,
                "batch_size": 2,
                "device": "cpu",
            },
        )
    else:
        features = {"kind": "ecfp4", "radius": 2, "n_bits": 64}
        model = {
            "n_estimators": 8, "max_depth": 2, "learning_rate": 0.1,
            "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0,
            "objective": "reg:squarederror", "eval_metric": "mae", "tree_method": "hist",
        }
        training = {
            "model_selection": "final_training_state",
            "n_jobs": 1,
            "device": "cpu",
            "target_space": "raw",
        }
    return benchmark_config_from_dict(
        {
            "name": name,
            "seed": 42,
            "data": {
                "data_root": str(tmp_path), "task_catalog": str(catalog),
                "stage3_authority_config": str(authority),
                "feature_cache": str(tmp_path / "features.sqlite3"),
            },
            "features": features,
            "model": model,
            "training": training,
            "stage3": {"enabled": True, "tasks": "all", "folds": [1, 2, 3, 4, 5]},
        }
    )


def test_formal_configs_and_registry_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_benchmark_config("configs/benchmarks/mlp.yaml")
    assert len(configured_tasks(config, "stage3")) == 20
    retired = config.to_dict()
    retired["stage2_physics"] = {"enabled": True, "tasks": ["simulation/homo"]}
    with pytest.raises(ValueError, match="Unknown benchmark config fields: stage2_physics"):
        benchmark_config_from_dict(retired)
    for launcher in (
        benchmark_train_launcher,
        benchmark_evaluate_launcher,
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(launcher.__file__), "--config", "config.yaml",
                "--benchmark", "stage2_physics", "--task", "simulation/homo",
                "--output", "output",
            ],
        )
        with pytest.raises(SystemExit):
            launcher.main()
    solvation = resolve_task(config, "stage3", "experiment/solvation", 1)
    organic = resolve_task(config, "stage3", "experiment/transfer_organic", 1)
    assert solvation.slots == ("cation", "anion", "solute")
    assert organic.slots == ("solute", "solvent")
    missing_test = resolve_task(config, "stage3", "experiment/self_diffusion_coefficient", 1)
    empty = load_split(missing_test, "test")
    reporter = RecordingReporter()
    with FeatureCache(tmp_path / "empty-cache.sqlite3") as cache:
        matrix = raw_feature_matrix(
            empty, feature_schema(config.features), cache, reporter=reporter
        )
    assert matrix.shape == (0, 2 * 21 + 2)
    assert reporter.bars == []


def test_basic_molecular_statistics_schema_and_golden_values() -> None:
    assert BASIC_FEATURE_NAMES == (
        "molecular_weight",
        "heavy_atom_count",
        "total_atom_count_with_implicit_hydrogens",
        "C_count", "N_count", "O_count", "F_count", "P_count", "S_count",
        "Cl_count", "Br_count", "I_count",
        "formal_charge", "bond_count", "ring_count", "aromatic_atom_count",
        "aromatic_ring_count", "rotatable_bond_count", "h_bond_donor_count",
        "h_bond_acceptor_count", "fraction_c_sp3",
    )
    aromatic = basic_molecular_statistics(Chem.MolFromSmiles("c1ccccc1Cl"))
    assert aromatic.tolist() == pytest.approx([
        112.559, 7, 12, 6, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 7, 1, 6, 1, 0, 0, 0, 0,
    ])
    halogens = basic_molecular_statistics(Chem.MolFromSmiles("FC(Cl)(Br)I"))
    assert halogens[3:12].tolist() == [1, 0, 0, 1, 0, 0, 1, 1, 1]
    charged = basic_molecular_statistics(Chem.MolFromSmiles("[NH4+]"))
    assert charged[1:3].tolist() == [1, 5]
    assert charged[12] == 1
    butane = basic_molecular_statistics(Chem.MolFromSmiles("CCCC"))
    assert butane[17] == 1
    ethanol = basic_molecular_statistics(Chem.MolFromSmiles("CCO"))
    assert ethanol[18:20].tolist() == [1, 1]


@pytest.mark.parametrize(
    "components",
    (
        ("[Na+]", "[Cl-]"),
        ("C[N+](C)(C)C", "[Cl-]", "O"),
        ("CCO", "O"),
    ),
)
def test_basic_features_follow_component_order_then_conditions(
    tmp_path: Path, components: tuple[str, ...],
) -> None:
    conditions = np.asarray([[300.0, 101.325]], dtype=np.float64)
    dataset = RawDataset(
        components=(components,),
        component_count=len(components),
        conditions=conditions,
        targets=np.asarray([[1.0]], dtype=np.float64),
        source_rows=("synthetic:2",),
        audit_rows=({},),
    )
    schema = feature_schema(load_benchmark_config("configs/benchmarks/mlp.yaml").features)
    with FeatureCache(tmp_path / "features.sqlite3") as cache:
        matrix = raw_feature_matrix(dataset, schema, cache)
    expected = np.concatenate([
        *(basic_molecular_statistics(Chem.MolFromSmiles(value)) for value in components),
        conditions[0],
    ])
    assert matrix.shape == (1, len(components) * 21 + 2)
    assert matrix[0].tolist() == pytest.approx(expected.tolist())


def test_basic_mlp_schema_versions_cache_and_training_identity() -> None:
    config = load_benchmark_config("configs/benchmarks/mlp.yaml")
    schema = feature_schema(config.features)
    payload = schema.to_dict()
    assert payload["schema_version"] == BASIC_FEATURE_SCHEMA_VERSION
    assert payload["feature_names"] == BASIC_FEATURE_NAMES
    assert payload["component_width"] == 21
    current = semantic_identity("benchmark.training.v1", {"feature": payload})
    legacy = semantic_identity(
        "benchmark.training.v1",
        {"feature": {"kind": "rdkit_2d", "component_width": 217}},
    )
    with pytest.raises(ValueError, match="semantic identity mismatch"):
        require_compatible_identity(
            current, legacy, context="Basic-MLP legacy checkpoint"
        )


def test_mlp_configs_lock_basic_feature_model_and_training_contract() -> None:
    paths = [
        Path("configs/benchmarks/mlp.yaml"),
        *(Path("configs/benchmarks/splits") / f"mlp__{split}.yaml"
          for split in ("random", "system", "individual")),
    ]
    for path in paths:
        config = load_benchmark_config(path)
        assert config.features.kind == "basic_molecular_statistics"
        assert config.features.radius is None and config.features.n_bits is None
        assert config.model == {"hidden_dims": [128, 64], "dropout": 0.2}
        assert config.training == {
            "optimizer": "adamw",
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-3,
            "batch_size": 128,
            "max_epochs": 10,
            "loss": "normalized_mse",
            "model_selection": "final_training_state",
            "device": "cuda",
            "precision": "fp32",
        }
    legacy = load_benchmark_config("configs/benchmarks/mlp.yaml").to_dict()
    legacy["features"] = {"kind": "rdkit_2d"}
    with pytest.raises(ValueError, match="requires basic molecular statistics"):
        benchmark_config_from_dict(legacy)


@pytest.mark.parametrize(
    ("name", "fixed_budget"),
    (
        ("mlp", 10),
        ("ecfp_xgboost", 1000),
        ("dmpnn", 10),
        ("molformer", 10),
        ("ilbert", 10),
        ("spmm", 10),
        ("llasmol", 10),
        ("aionopedia", 10),
        ("iltransr", None),
        ("aifc", 10),
    ),
)
def test_formal_baseline_configs_use_fixed_final_state(
    name: str, fixed_budget: int | None,
) -> None:
    config = load_benchmark_config(Path("configs/benchmarks") / f"{name}.yaml")
    assert len(configured_tasks(config, "stage3")) == 20
    assert tuple(config.stage3.folds) == (1, 2, 3, 4, 5)
    assert config.data.stage3_authority_config == Path(
        "configs/v2/stage3/splits/system.yaml"
    )
    assert config.training["model_selection"] == "final_training_state"
    assert {
        "early_stopping_patience", "early_stopping_rounds", "selection_metric",
    }.isdisjoint(config.training)
    if name == "ecfp_xgboost":
        assert config.model["n_estimators"] == fixed_budget
    elif fixed_budget is not None:
        assert config.training["max_epochs"] == fixed_budget
    if name == "ilbert":
        assert config.training["scheduler"] == "constant"
        assert config.training["learning_rate"] == 3.0e-5
        assert {
            "scheduler_metric", "scheduler_patience", "scheduler_factor",
            "minimum_learning_rate",
        }.isdisjoint(config.training)


def test_aionopedia_head128_variant_preserves_the_official_training_recipe() -> None:
    official = load_benchmark_config("configs/benchmarks/aionopedia.yaml")
    variant = load_benchmark_config("configs/benchmarks/aionopedia_head128.yaml")
    assert official.model.get("scalar_head_hidden_dim", 1024) == 1024
    assert variant.model["scalar_head_hidden_dim"] == 128
    assert variant.model["scalar_head"] == "linear_512_128_relu_linear_128_1"
    assert variant.training == official.training
    assert variant.data == official.data
    assert variant.model["base_files"] == official.model["base_files"]
    assert variant.model["pretrained_files"] == official.model["pretrained_files"]


def test_aionopedia_regression_head_uses_configured_hidden_width() -> None:
    model = AIonopediaRegressor(torch.nn.Identity(), head_hidden_dim=128)
    assert model.fc_out[0].in_features == 512
    assert model.fc_out[0].out_features == 128
    assert model.fc_out[2].in_features == 128
    assert model.fc_out[2].out_features == 1


@pytest.mark.parametrize(
    "name",
    (
        "dmpnn", "molformer", "ilbert", "spmm", "llasmol", "aionopedia",
        "iltransr", "aifc",
    ),
)
def test_readme_validator_examples_import_model_specific_modules(name: str) -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert (
        f"from benchmarks.{name}.environment import validate_{name}_environment"
        in readme
    )
    assert (
        f"from benchmarks.common.environment import validate_{name}_environment"
        not in readme
    )


@pytest.mark.parametrize(
    "name,section,field,value,message",
    (
        ("dmpnn", "model", "multicomponent_shared", False, "registered Chemprop recipe"),
        ("molformer", "training", "batch_size", 32, "registered fine-tuning recipe"),
        ("ilbert", "training", "tf32", False, "registered fine-tuning recipe"),
        ("spmm", "model", "wordpiece_max_input_chars_per_word", 351, "registered upstream recipe"),
        ("spmm", "training", "batch_size", 8, "registered fine-tuning recipe"),
        ("llasmol", "training", "batch_size", 8, "registered QLoRA recipe"),
        ("aionopedia", "model", "pretrained_snapshot", "artifacts/property-specific/density", "registered multimodal recipe"),
    ),
)
def test_baseline_rejects_unregistered_recipe(name, section, field, value, message) -> None:
    payload = load_benchmark_config(f"configs/benchmarks/{name}.yaml").to_dict()
    payload[section][field] = value
    with pytest.raises(ValueError, match=message):
        benchmark_config_from_dict(payload)


def test_baseline_config_rejects_validation_driven_training() -> None:
    config = load_benchmark_config("configs/benchmarks/mlp.yaml")
    for field in ("selection_metric", "early_stopping_patience", "early_stopping_rounds"):
        retired = config.to_dict()
        retired["training"][field] = 1
        with pytest.raises(ValueError, match="forbid validation-driven"):
            benchmark_config_from_dict(retired)


def test_native_split_benchmark_configs_follow_v2_authorities() -> None:
    splits = ("system", "random", "individual")
    benchmarks = ("mlp", "ecfp_xgboost", "dmpnn", "molformer", "ilbert", "spmm")
    root = Path("configs/benchmarks/splits")
    for split in splits:
        authority = load_stage3_config(
            Path("configs/v2/stage3/splits") / f"{split}.yaml"
        )
        expected = tuple(authority.enabled_task_ids)
        for benchmark in benchmarks:
            config = load_benchmark_config(root / f"{benchmark}__{split}.yaml")
            assert config.data.stage3_authority_config == Path(
                f"configs/v2/stage3/splits/{split}.yaml"
            )
            assert configured_tasks(config, "stage3") == expected
            assert config.training["model_selection"] == "final_training_state"
            assert {
                "early_stopping_patience", "early_stopping_rounds", "selection_metric",
            }.isdisjoint(config.training)
            if benchmark == "ecfp_xgboost":
                assert config.model["n_estimators"] == 1000
            else:
                assert config.training["max_epochs"] == 10
            if benchmark == "ilbert":
                assert config.training["scheduler"] == "constant"
            if benchmark == "dmpnn":
                assert config.model["multicomponent_shared"] is True


def test_stage3_single_task_mlp_config_and_ordered_concat() -> None:
    config = load_benchmark_config(
        "configs/ablations/ilume_stage3_single_task_mlp.yaml"
    )
    assert config.display_name == "ILUME Stage3 Single-task MLP"
    assert config.data.stage3_authority_config == Path("configs/v1/stage3/base.yaml")
    with pytest.raises(ValueError, match="isobaric_coefficient_of_volume_expansion"):
        configured_tasks(config, "stage3")
    assert config.data.feature_cache is None and config.features is None
    with pytest.raises(ValueError, match="registered recipe"):
        replace(config, model={**config.model, "dropout": 0.2}).validate()

    embeddings = torch.arange(4 * 512, dtype=torch.float32).reshape(4, 512)
    class Dataset:
        conditions = torch.tensor([[0.25, -0.5], [1.0, 2.0]])
        primary_object_ids = torch.tensor([0, 1])
        partner_object_ids = torch.tensor([2, 3])

        def __len__(self) -> int:
            return 2

    dataset = Dataset()
    spec = SimpleNamespace(
        task_id="experiment/partner",
        condition_columns=("temperature_K", "pressure_kPa"),
        partner_slots=("solute",),
    )
    features = build_input_features(dataset, embeddings, spec)
    assert features.shape == (2, 1026)
    torch.testing.assert_close(features[:, :512], embeddings[[0, 1]])
    torch.testing.assert_close(features[:, 512:1024], embeddings[[2, 3]])
    torch.testing.assert_close(features[:, 1024:], dataset.conditions)
    first_model = Stage3SingleTaskMLP(features.shape[1])
    second_model = Stage3SingleTaskMLP(features.shape[1])
    assert {
        id(parameter) for parameter in first_model.parameters()
    }.isdisjoint(id(parameter) for parameter in second_model.parameters())

    condition_only = SimpleNamespace(
        task_id="experiment/condition",
        condition_columns=("temperature_K", "pressure_kPa"),
        partner_slots=(),
    )
    dataset.partner_object_ids = torch.full((2,), -1)
    assert build_input_features(dataset, embeddings, condition_only).shape == (2, 514)
    with pytest.raises(ValueError, match="partner embedding is missing"):
        build_input_features(dataset, embeddings, spec)

    ordinary = SimpleNamespace(
        task_id="experiment/plain", condition_columns=(), partner_slots=()
    )
    dataset.conditions = torch.empty((2, 0))
    dataset.partner_object_ids = torch.full((2,), -1)
    assert build_input_features(dataset, embeddings, ordinary).shape == (2, 512)


def test_stage3_single_task_mlp_runs_full_budget_and_selects_best() -> None:
    config = load_benchmark_config(
        "configs/ablations/ilume_stage3_single_task_mlp.yaml"
    )
    config = replace(
        config,
        training={**config.training, "batch_size": 2, "max_epochs": 3},
    )
    torch.manual_seed(7)
    model = Stage3SingleTaskMLP(4)
    train_features = torch.randn(6, 4)
    train_targets = torch.linspace(-1.0, 1.0, 6)
    valid_features = torch.randn(3, 4)
    valid_targets = torch.tensor([-0.5, 0.0, 0.5])
    reporter = RecordingReporter()
    history, best_state, best_epoch, best_score = _run_training_epochs(
        model,
        train_features,
        train_targets,
        valid_features,
        valid_targets,
        config,
        training_seed=123,
        device=torch.device("cpu"),
        use_bf16=False,
        reporter=reporter,
    )
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert best_score == min(row["valid_normalized_mae"] for row in history)
    assert best_epoch == next(
        row["epoch"] for row in history
        if row["valid_normalized_mae"] == best_score
    )
    restored = Stage3SingleTaskMLP(4)
    restored.load_state_dict(best_state, strict=True)
    assert reporter.bars[0].n == 3 and reporter.bars[0].closed


def test_stage3_single_task_mlp_v2_config_features_and_final_state(tmp_path: Path) -> None:
    legacy = load_benchmark_config("configs/ablations/ilume_stage3_single_task_mlp.yaml")
    config = load_benchmark_config("configs/ablations/ilume_stage3_single_task_mlp_v2.yaml")
    assert config.data.stage3_authority_config == Path("configs/v2/stage3/base.yaml")
    assert len(configured_tasks(config, "stage3")) == 20
    assert config.training["max_epochs"] == 10
    assert config.training["model_selection"] == "final_training_state"
    with pytest.raises(ValueError, match="registered recipe"):
        replace(config, model={**config.model, "hidden_dims": [512, 256]}).validate()

    embeddings = torch.arange(4 * 1024, dtype=torch.float32).reshape(4, 1024)
    class Rows:
        conditions = torch.tensor([[0.25], [-0.5]])
        primary_object_ids = torch.tensor([0, 1])
        partner_object_ids = torch.tensor([2, 3])

        def __len__(self) -> int:
            return 2

    features = build_input_features(
        Rows(), embeddings,
        SimpleNamespace(task_id="example", condition_columns=("temperature_K",), partner_slots=("solute",)),
    )
    assert features.shape == (2, 2049)
    torch.testing.assert_close(features[:, :1024], embeddings[[0, 1]])
    torch.testing.assert_close(features[:, 1024:2048], embeddings[[2, 3]])
    torch.testing.assert_close(features[:, 2048:], Rows.conditions)

    small = replace(config, training={**config.training, "batch_size": 2, "max_epochs": 2})
    model = Stage3SingleTaskMLP(4, (1024, 512))
    history, state, epoch, score = _run_training_epochs(
        model, torch.randn(4, 4), torch.randn(4), torch.randn(2, 4), torch.randn(2),
        small, training_seed=123, device=torch.device("cpu"), use_bf16=False,
        reporter=RecordingReporter(),
    )
    assert epoch == 2 and score == history[-1]["valid_normalized_mae"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(state[name], value.cpu())

    (tmp_path / "checkpoint.json").write_text(
        json.dumps({"format_version": 1, "kind": "ilume_stage3_single_task_mlp_model_v2", "integrity": {}}),
        encoding="utf-8",
    )
    assert _manifest(tmp_path, config)["kind"].endswith("_v2")
    with pytest.raises(ValueError, match="Unsupported"):
        _manifest(tmp_path, legacy)
    (tmp_path / "checkpoint.json").write_text(
        json.dumps({"format_version": 1, "kind": "ilume_stage3_single_task_mlp_model", "integrity": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported"):
        _manifest(tmp_path, config)


@pytest.mark.parametrize("name", ("dmpnn", "molformer", "ilbert", "spmm", "llasmol", "aionopedia", "iltransr", "aifc"))
def test_baseline_environment_command(name: str) -> None:
    config_path = f"configs/benchmarks/{name}.yaml"
    config = load_benchmark_config(config_path)
    assert environment_command(
        config, ("scripts/benchmarks/train.py", "--config", config_path), conda="/conda"
    ) == [
        "/conda", "run", "--no-capture-output", "-n", f"ilume-{name}", "python",
        str((Path.cwd() / "scripts/benchmarks/train.py").resolve()), "--config", config_path,
    ]


def test_runtime_options_do_not_change_benchmark_scientific_identity() -> None:
    config = load_benchmark_config("configs/benchmarks/molformer.yaml")
    runtime_variant = replace(config, runtime={**config.runtime, "num_workers": 8})
    assert sweep_module._scientific_config(runtime_variant) == sweep_module._scientific_config(config)


def test_dmpnn_environment_dispatches_once_before_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_benchmark_config("configs/benchmarks/dmpnn.yaml")
    monkeypatch.delenv(ENVIRONMENT_MARKER, raising=False)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return type("Result", (), {"returncode": 7})()

    monkeypatch.setattr("benchmarks.common.environment.shutil.which", lambda _: "conda")
    monkeypatch.setattr("benchmarks.common.environment.subprocess.run", fake_run)
    with pytest.raises(SystemExit, match="7"):
        ensure_benchmark_environment(config, ("scripts/benchmarks/train.py",))
    assert calls[0][1]["env"][ENVIRONMENT_MARKER] == "ilume-dmpnn"

def test_preprocessor_uses_train_mask_median_and_population_zscore() -> None:
    train = np.asarray([[1.0, np.nan, np.nan, 4.0], [3.0, 7.0, np.inf, 4.0]])
    preprocessor = FeaturePreprocessor.fit(train)
    assert preprocessor.finite_mask == (True, True, False, True)
    transformed = preprocessor.transform(np.asarray([[np.nan, 9.0, 123.0, 4.0]]))
    assert transformed.shape == (1, 3)
    assert transformed[0].tolist() == pytest.approx([0.0, 2.0, 0.0])


def test_mlp_train_checkpoint_and_test_evaluation(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    test_path = tmp_path / "stage3/experiment/tiny/test.csv"
    test_csv = test_path.read_bytes()
    test_path.unlink()
    bundle = prepare_training(config, "stage3", "experiment/tiny", 1)
    assert bundle.train_features.shape[0] == 8
    assert bundle.valid_features.shape[0] == 2
    test_path.write_bytes(test_csv)
    output = tmp_path / "mlp_run"
    reporter = RecordingReporter()
    summary = train_bundle(config, bundle, output, reporter=reporter)
    reference_output = tmp_path / "mlp_reference"
    reference_summary = train_bundle(config, bundle, reference_output)
    assert summary == reference_summary
    assert json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))[
        "model_state_hash"
    ] == json.loads(
        (reference_output / "checkpoint.json").read_text(encoding="utf-8")
    )["model_state_hash"]
    assert summary["final_epoch"] == 10
    assert summary["epochs_ran"] == 10
    assert "best_epoch" not in summary
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["format_version"] == 2
    assert checkpoint["final_epoch"] == 10
    assert checkpoint["feature_schema"]["schema_version"] == BASIC_FEATURE_SCHEMA_VERSION
    assert checkpoint["feature_schema"]["feature_names"] == list(BASIC_FEATURE_NAMES)
    assert "best_valid_raw_macro_mae" not in checkpoint
    assert len(reporter.bars) == 1
    assert reporter.bars[0].n == summary["epochs_ran"]
    assert reporter.bars[0].closed
    evaluation_reporter = RecordingReporter()
    result = evaluate_checkpoint(
        config,
        "stage3",
        "experiment/tiny",
        1,
        output,
        "test",
        reporter=evaluation_reporter,
    )
    assert result.predictions.shape == (2, 1)
    assert set(result.metrics) == {"value"}
    assert "normalized_mae" in result.metrics["value"]
    assert "normalized_rmse" in result.metrics["value"]

    valid = evaluate_checkpoint(config, "stage3", "experiment/tiny", 1, output, "valid")
    assert valid.predictions.shape == (2, 1)
    assert "normalized_mae" in valid.metrics["value"]


def test_xgboost_uses_independent_models_and_fixed_budget(tmp_path: Path) -> None:
    pytest.importorskip("xgboost")
    config = _tiny_config(tmp_path, name="ecfp_xgboost")
    bundle = prepare_training(config, "stage3", "experiment/tiny", 1)
    output = tmp_path / "xgb_run"
    reporter = RecordingReporter()
    summary = train_bundle(config, bundle, output, reporter=reporter)
    reference_summary = train_bundle(config, bundle, tmp_path / "xgb_reference")
    assert summary == reference_summary
    assert set(summary["targets"]) == {"value"}
    assert summary["targets"]["value"]["trained_rounds"] == 8
    assert "best_iteration" not in summary["targets"]["value"]
    assert len(list(output.glob("model_*.json"))) == 1
    assert len(reporter.bars) == 1
    assert all(0 < bar.n <= bar.total and bar.closed for bar in reporter.bars)
    assert all(set(bar.postfixes[-1]) == {"val_mae"} for bar in reporter.bars)
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["format_version"] == 2
    assert checkpoint["models"][0]["trained_rounds"] == 8
    result = evaluate_checkpoint(config, "stage3", "experiment/tiny", 1, output, "test")
    assert result.predictions.shape == (2, 1)

def test_five_fold_ensemble_averages_predictions_before_metrics() -> None:
    results = [
        EvaluationResult(
            predictions=np.asarray([[float(fold)], [float(fold + 2)]]),
            targets=np.asarray([[3.0], [5.0]]),
            source_rows=("test:2", "test:3"),
            metrics={},
            target_stats=TargetStats((0.0,), (float(fold),)),
            training_identity={},
        )
        for fold in range(1, 6)
    ]
    predictions, metrics = ensemble_evaluation(results, ("value",))
    assert predictions[:, 0].tolist() == pytest.approx([3.0, 5.0])
    assert metrics["value"]["mae"] == 0.0
    assert metrics["value"]["normalized_mae"] == 0.0

def test_sweep_scheduler_is_bounded_and_preserves_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs, ensembles = _build_jobs(
        root=tmp_path,
        stage3_tasks=("experiment/one", "experiment/two"),
        folds=(1, 2),
        devices=(),
    )
    lock = threading.Lock()
    active = 0
    maximum_active = 0
    finished: set[tuple[str, str, int | None]] = set()
    ensemble_dependencies: dict[str, set[int]] = {}

    def execute(job, **_kwargs):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            if job.kind == "stage3_ensemble":
                ensemble_dependencies[job.task] = {
                    fold
                    for kind, task, fold in finished
                    if kind == "stage3_fold" and task == job.task and fold is not None
                }
        time.sleep(0.03)
        with lock:
            active -= 1
            finished.add(job.key)
        return _JobResult(job, job.kind != "stage3_ensemble")

    monkeypatch.setattr(sweep_module, "_execute_job", execute)
    state = _SweepState(rows=[], status_path=tmp_path / "status.tsv")
    _schedule(
        jobs=jobs,
        ensembles=ensembles,
        folds=(1, 2),
        max_workers=3,
        state=state,
        config_path="unused.yaml",
        root=tmp_path,
        train_script=tmp_path / "train.py",
        evaluate_script=tmp_path / "evaluate.py",
    )
    assert maximum_active == 3
    assert ensemble_dependencies == {
        "experiment/one": {1, 2},
        "experiment/two": {1, 2},
    }
    assert {job.key for job in [*jobs, *ensembles.values()]} == finished

def test_sweep_scheduler_caps_concurrency_per_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs, ensembles = _build_jobs(
        root=tmp_path,
        stage3_tasks=("experiment/one", "experiment/two"),
        folds=(1, 2),
        devices=("cuda:0", "cuda:1"),
    )
    lock = threading.Lock()
    active = 0
    maximum_active = 0
    active_by_device: dict[str, int] = {"cuda:0": 0, "cuda:1": 0}
    maximum_by_device: dict[str, int] = {"cuda:0": 0, "cuda:1": 0}

    def execute(job, **_kwargs):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            assert job.device is not None
            active_by_device[job.device] += 1
            maximum_by_device[job.device] = max(
                maximum_by_device[job.device], active_by_device[job.device]
            )
        time.sleep(0.03)
        with lock:
            active -= 1
            active_by_device[job.device] -= 1
        return _JobResult(job, job.kind != "stage3_ensemble")

    monkeypatch.setattr(sweep_module, "_execute_job", execute)
    state = _SweepState(rows=[], status_path=tmp_path / "status.tsv")
    _schedule(
        jobs=jobs,
        ensembles=ensembles,
        folds=(1, 2),
        devices=("cuda:0", "cuda:1"),
        max_workers=4,
        state=state,
        config_path="unused.yaml",
        root=tmp_path,
        train_script=tmp_path / "train.py",
        evaluate_script=tmp_path / "evaluate.py",
    )
    assert maximum_active == 4
    assert maximum_by_device == {"cuda:0": 2, "cuda:1": 2}

@pytest.mark.parametrize(
    "config_name, expected_selector",
    [
        ("ilume_stage3_single_task_mlp", "validation_best"),
        ("ilume_stage3_single_task_mlp_v2", "final_training_state"),
    ],
)
def test_stage3_only_ablation_aggregate_has_one_model_and_no_stage2_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_name: str, expected_selector: str,
) -> None:
    config = load_benchmark_config(
        f"configs/ablations/{config_name}.yaml"
    )
    task = "experiment/example"
    study_id = f"{config.name}-{semantic_identity(
        'benchmark.reporting-study.v1',
        {'model': config.name, 'config': sweep_module._scientific_config(config)},
    )['hash']}"
    monkeypatch.setattr(
        sweep_module,
        "configured_tasks",
        lambda _config, benchmark: (task,) if benchmark == "stage3" else (),
    )
    monkeypatch.setattr(sweep_module, "resolve_task", lambda *_args: object())
    monkeypatch.setattr(sweep_module, "has_test_rows", lambda _task: True)
    monkeypatch.setattr(
        sweep_module, "repository_relative", lambda value: Path(value).as_posix()
    )
    sources = {
        **{f"{task}:fold{fold}": f"source-{fold}" for fold in range(1, 6)},
        f"{task}:test": "test-source",
    }
    metric = {
        "count": 2, "mae": 1.0, "rmse": 1.0, "r2": 0.0,
        "normalized_mae": 0.5, "normalized_rmse": 0.5,
    }

    def completed(path: Path, payload: dict[str, object]) -> None:
        run = path / "attempt-001"
        run.mkdir(parents=True)
        (run / "metadata.json").write_text(
            '{"status":"completed"}\n', encoding="utf-8"
        )
        (run / "summary.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    task_root = tmp_path / "stage3" / "experiment__example"
    for fold in range(1, 6):
        comparison = comparison_identity(
            "stage3_property",
            split="valid",
            expected=[task],
            sources={key: value for key, value in sources.items() if not key.endswith(":test")},
            normalization={f"{task}:fold{fold}": {"scale": 2.0}},
            folds=range(1, 6),
        )
        completed(
            task_root / f"evaluate_valid_fold{fold}",
            {
                "targets": {"value": metric},
                "reporting": {
                    "schema_version": 1,
                    "study_id": study_id,
                    "protocol": {"expected_tasks": [task]},
                    "comparison_identity": comparison,
                },
            },
        )
    completed(
        task_root / "evaluate_test",
        {
            "ensemble": {"targets": {"value": metric}},
            "reporting": {
                "schema_version": 1,
                "study_id": study_id,
                "protocol": {"expected_tasks": [task]},
                "comparison_identity": comparison_identity(
                    "stage3_property",
                    split="test",
                    expected=[task],
                    sources=sources,
                    normalization={
                        f"{task}:fold{fold}": {"scale": 2.0}
                        for fold in range(1, 6)
                    },
                    folds=range(1, 6),
                    ensemble=True,
                ),
            },
        },
    )
    summary = _aggregate(tmp_path, config)
    assert set(summary["reporting"]["benchmarks"]) == {
        "stage3_test", "stage3_validation"
    }
    assert summary["reporting"]["model_id"] == config.name
    assert summary["reporting"]["model_display_name"] == config.display_name
    assert summary["model_selector"] == expected_selector
    assert summary["checkpoint_epoch"] is None
    _write_run(tmp_path / "published", summary, stage="benchmark")
    published = publish_summary(
        tmp_path / "published", tmp_path / "leaderboard", tmp_path
    )
    assert len(published["leaderboards"]["stage3_test"]) == 1
    assert len(published["leaderboards"]["stage3_validation"]) == 1
    assert "legacy_stage2_reporting_contract" not in published["health"][0]["issues"]

# --- Reporting contract ---

def _write_run(
    root: Path, summary: dict[str, object], *, stage: str = "benchmark"
) -> None:
    root.mkdir(parents=True)
    reporting = summary.get("reporting", {})
    if stage == "stage3" and reporting.get("model_id") == "ilume":
        protocol = reporting["protocol"]
        prediction_field = (
            "prediction_ensemble"
            if protocol["split"] == "test"
            else "prediction"
        )
        manifests = []
        for task in protocol["expected_tasks"]:
            path = root / "predictions" / f"{sanitize_task_id(task)}.csv"
            manifest = write_prediction_csv(
                path,
                [
                    {
                        "source_row": protocol.get("fold", 0),
                        "target": 1.0,
                        prediction_field: 1.25,
                    }
                ],
                ("source_row", "target", prediction_field),
            )
            manifest["path"] = f"predictions/{path.name}"
            manifest["task"] = task
            manifests.append(manifest)
        reporting["predictions"] = manifests
    metadata = {
        "schema_version": 1,
        "stage": stage,
        "operation": "sweep" if stage == "benchmark" else "evaluate",
        "status": "completed",
        "semantic_identity": semantic_identity(
            "test.reporting-run", {"root": root.name}
        ),
        "provenance": {"reporting_schema_version": 1},
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _stage3_benchmark_summary(
    model: str, *, scale: float, source: str = "shared"
) -> dict[str, object]:
    task = "experiment/example"
    sources = {
        **{f"{task}:fold{fold}": source for fold in range(1, 6)},
        f"{task}:test": source,
    }
    normalization = {
        f"{task}:fold{fold}": {"scale": scale} for fold in range(1, 6)
    }
    test_comparison = comparison_identity(
        "stage3_property", split="test", expected=[task], sources=sources,
        normalization=normalization, folds=range(1, 6), ensemble=True,
    )
    valid_comparison = comparison_identity(
        "stage3_property", split="valid", expected=[task],
        sources={
            key: value
            for key, value in sources.items()
            if not key.endswith(":test")
        },
        normalization=normalization, folds=range(1, 6),
    )
    metrics = {
        "count": 5, "mae": scale, "rmse": scale, "r2": 0.5,
        "normalized_mae": 1.0, "normalized_rmse": 1.0,
    }
    aggregate = {
        name: {"count": 5, "mean": value, "std": 0.0}
        for name, value in (
            ("mae", scale), ("rmse", scale), ("r2", 0.5),
            ("normalized_mae", 1.0), ("normalized_rmse", 1.0),
        )
    }
    return {
        "jobs": {"failed": 0},
        "stage3_property_benchmark": {
            "test_ensemble": {task: metrics},
            "validation_five_fold": {task: aggregate},
        },
        "stage2_physics_benchmark": {"test": {}},
        "reporting": {
            "schema_version": REPORTING_SCHEMA_VERSION,
            "model_id": model,
            "model_display_name": model.upper(),
            "study_id": f"{model}-study",
            "source_runs": {},
            "benchmarks": {
                "stage3_test": {
                    "status": "complete",
                    "protocol": {
                        "expected_tasks": [task], "enabled_tasks": [task],
                        "folds": list(range(1, 6)), "ensemble": True,
                    },
                    "comparison_identity": test_comparison,
                },
                "stage3_validation": {
                    "status": "complete",
                    "protocol": {
                        "expected_tasks": [task],
                        "folds": list(range(1, 6)), "ensemble": False,
                    },
                    "comparison_identity": valid_comparison,
                },
            },
        },
    }


def _stage3_validation_summary(
    study_id: str, fold: int, *, mae: float
) -> dict[str, object]:
    task = "experiment/example"
    comparison = comparison_identity(
        "stage3_property",
        split="valid",
        expected=[task],
        sources={f"{task}:fold{index}": "shared" for index in range(1, 6)},
        normalization={
            f"{task}:fold{index}": {"scale": 1.0}
            for index in range(1, 6)
        },
        folds=range(1, 6),
    )
    metrics = {
        "count": 1, "mae": mae, "rmse": mae, "r2": 0.5,
        "normalized_mae": mae, "normalized_rmse": mae,
    }
    return {
        "split": "valid",
        "checkpoint_epoch": None,
        "tasks": {task: metrics},
        "reporting": {
            "schema_version": REPORTING_SCHEMA_VERSION,
            "model_id": "ilume",
            "model_display_name": "ILUME",
            "study_id": study_id,
            "protocol": {
                "split": "valid", "fold": fold,
                "folds": list(range(1, 6)), "ensemble": False,
                "expected_tasks": [task],
            },
            "comparison_identity": comparison,
        },
    }


def test_stage3_summary_ignores_normalization_but_requires_shared_sources(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    _write_run(
        inputs / "one", _stage3_benchmark_summary("one", scale=1.0),
        stage="benchmark",
    )
    _write_run(
        inputs / "two", _stage3_benchmark_summary("two", scale=2.0),
        stage="benchmark",
    )

    payload = publish_summary(inputs, tmp_path / "summary", tmp_path)
    assert len(payload["leaderboards"]["stage3_test"]) == 2
    assert len(payload["leaderboards"]["stage3_validation"]) == 2
    assert not any((tmp_path / "summary" / "ilume_scatter" / "test").iterdir())
    assert not any(
        (tmp_path / "summary" / "ilume_scatter" / "validation").iterdir()
    )
    assert "stage2" not in json.dumps(payload).lower()
    assert all("stage2" not in name for name in SUMMARY_FILES)
    with (tmp_path / "summary" / "stage3_test_task_mae.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        test_mae_rows = list(csv.DictReader(handle))
    with (tmp_path / "summary" / "stage3_test_task_rank.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        test_rank_rows = list(csv.DictReader(handle))
    with (tmp_path / "summary" / "stage3_validation_task_mae.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        validation_mae_rows = list(csv.DictReader(handle))
    with (tmp_path / "summary" / "stage3_validation_task_rank.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        validation_rank_rows = list(csv.DictReader(handle))
    for rows in (test_mae_rows, validation_mae_rows):
        assert [row["model"] for row in rows] == ["ONE", "TWO"]
        assert [float(row["experiment/example"]) for row in rows] == [1.0, 2.0]
    for rows in (test_rank_rows, validation_rank_rows):
        assert [int(row["experiment/example"]) for row in rows] == [1, 2]

    incompatible = tmp_path / "incompatible"
    _write_run(
        incompatible / "one", _stage3_benchmark_summary("one", scale=1.0),
        stage="benchmark",
    )
    _write_run(
        incompatible / "two",
        _stage3_benchmark_summary("two", scale=2.0, source="different"),
        stage="benchmark",
    )
    with pytest.raises(ValueError, match="incompatible comparison identities"):
        publish_summary(incompatible, tmp_path / "bad-summary", tmp_path)


def test_stage3_summary_separates_ilume_variants_by_output_directory(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "outputs"
    variants = (
        ("base", inputs / "v2" / "stage3" / "base", 1.0),
        (
            "stage3_transfer_knowledge",
            inputs / "ablations" / "stage3_transfer_knowledge",
            2.0,
        ),
    )
    for variant, output_root, mae in variants:
        for fold in range(1, 6):
            _write_run(
                output_root / "validation" / f"fold{fold}",
                _stage3_validation_summary("shared-study", fold, mae=mae),
                stage="stage3",
            )

    payload = publish_summary(inputs, tmp_path / "summary", tmp_path)
    rows = payload["leaderboards"]["stage3_validation"]
    assert [row["model"] for row in rows] == [
        "ILUME (base)", "ILUME (stage3_transfer_knowledge)"
    ]
    assert [row["macro_normalized_mae"] for row in rows] == [1.0, 2.0]
    assert all("duplicate_folds" not in row["issues"] for row in payload["health"])
    scatter = (
        tmp_path / "summary" / "ilume_scatter" / "validation"
        / "experiment__example.svg"
    )
    svg = scatter.read_text(encoding="utf-8")
    assert "ILUME (base) · validation · n=5" in svg
    assert 'class="identity-line"' in svg
    assert 'class="scatter-points"' in svg
    assert 'stroke="#003f88"' in svg

    original = svg
    publish_summary(inputs, tmp_path / "summary", tmp_path)
    assert scatter.read_text(encoding="utf-8") == original
    prediction = (
        inputs / "v2" / "stage3" / "base" / "validation" / "fold1"
        / "predictions" / "experiment__example.csv"
    )
    prediction.write_text(
        prediction.read_text(encoding="utf-8") + "2,1,2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        publish_summary(inputs, tmp_path / "summary", tmp_path)
    assert scatter.read_text(encoding="utf-8") == original


def test_stage3_summary_accepts_full_finetune_reporting(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "outputs" / "ablations" / "stage3_full_finetune"
    for fold in range(1, 6):
        summary = _stage3_validation_summary(
            "ilume-stage3-full-finetune-v1", fold, mae=0.5
        )
        summary["ablation"] = "stage3_full_finetune"
        summary["reporting"]["model_display_name"] = "ILUME (full fine-tune)"
        _write_run(inputs / "evaluate_valid" / f"fold{fold}", summary, stage="stage3")

    payload = publish_summary(inputs, tmp_path / "summary", tmp_path)
    rows = payload["leaderboards"]["stage3_validation"]
    assert len(rows) == 1
    assert rows[0]["model"] == "ILUME (full fine-tune)"
    assert rows[0]["macro_normalized_mae"] == 0.5


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("manifest", "lacks prediction manifests"),
        ("rows", "row count mismatch"),
        ("columns", "lacks.*prediction"),
        ("nonfinite", "non-finite point"),
    ),
)
def test_ilume_scatter_rejects_malformed_prediction_artifacts(
    tmp_path: Path, mutation: str, message: str
) -> None:
    inputs = tmp_path / "outputs"
    for fold in range(1, 6):
        _write_run(
            inputs / "v2" / "stage3" / "base" / "validation" / f"fold{fold}",
            _stage3_validation_summary("shared-study", fold, mae=1.0),
            stage="stage3",
        )
    run = inputs / "v2" / "stage3" / "base" / "validation" / "fold1"
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if mutation == "manifest":
        summary["reporting"].pop("predictions")
    elif mutation == "rows":
        summary["reporting"]["predictions"][0]["rows"] = 2
    else:
        prediction = run / "predictions" / "experiment__example.csv"
        if mutation == "columns":
            prediction.write_text(
                "source_row,target,wrong\n1,1,1.25\n", encoding="utf-8"
            )
        else:
            prediction.write_text(
                "source_row,target,prediction\n1,nan,1.25\n", encoding="utf-8"
            )
        manifest = summary["reporting"]["predictions"][0]
        manifest["sha256"] = hashlib.sha256(prediction.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        publish_summary(inputs, tmp_path / "summary", tmp_path)


def test_prediction_csv_is_atomic_and_records_integrity(tmp_path: Path) -> None:
    path = tmp_path / "predictions" / "task.csv"
    manifest = write_prediction_csv(
        path,
        [{"source_row": 2, "target": 1.0, "prediction": 1.25}],
        ("source_row", "target", "prediction"),
    )
    assert manifest["rows"] == 1
    assert len(manifest["sha256"]) == 64
    with path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [
            {"source_row": "2", "target": "1", "prediction": "1.25"}
        ]
    assert not path.with_suffix(".csv.tmp").exists()

def test_summary_labels_capacity_v1_stage3_scales(tmp_path: Path) -> None:
    inputs = tmp_path / "outputs"
    task = "experiment/example"
    comparison = comparison_identity(
        "stage3_property",
        split="test",
        expected=[task],
        sources={
            **{f"{task}:fold{fold}": "shared" for fold in range(1, 6)},
            f"{task}:test": "shared",
        },
        normalization={
            f"{task}:fold{fold}": {"scale": 1.0} for fold in range(1, 6)
        },
        folds=range(1, 6),
        ensemble=True,
    )
    metric = {
        "count": 5, "mae": 1.0, "rmse": 1.0, "r2": 0.5,
        "normalized_mae": 1.0, "normalized_rmse": 1.0,
    }
    for scale in ("s", "base", "l", "xl"):
        root = inputs / "experiments_v1" / "stage3" / "formal" / scale
        reporting = {
            "schema_version": REPORTING_SCHEMA_VERSION,
            "model_id": "ilume",
            "model_display_name": "ILUME",
            "study_id": f"ilume-capacity-v1-{scale}",
            "comparison_identity": comparison,
        }
        _write_run(
            root / "test",
            {
                "split": "test",
                "checkpoint_epoch": None,
                "ensemble": {"tasks": {task: metric}},
                "reporting": {
                    **reporting,
                    "protocol": {
                        "split": "test", "folds": list(range(1, 6)),
                        "ensemble": True, "expected_tasks": [task],
                    },
                },
            },
            stage="stage3",
        )
        for fold in range(1, 6):
            _write_run(
                root / "validation" / f"fold{fold}",
                {
                    "split": "valid",
                    "checkpoint_epoch": None,
                    "tasks": {task: metric},
                    "reporting": {
                        **reporting,
                        "protocol": {
                            "split": "valid", "fold": fold,
                            "folds": list(range(1, 6)), "ensemble": False,
                            "expected_tasks": [task],
                        },
                    },
                },
                stage="stage3",
            )

    payload = publish_summary(inputs, tmp_path / "summary", tmp_path)

    expected = {
        "ILUME Capacity v1 (S)", "ILUME Capacity v1 (Base)",
        "ILUME Capacity v1 (L)", "ILUME Capacity v1 (XL)",
    }
    assert {row["model"] for row in payload["leaderboards"]["stage3_test"]} == expected
    assert {row["model"] for row in payload["leaderboards"]["stage3_validation"]} == expected
    assert all("alternative_run" not in row["issues"] for row in payload["health"])
    overview = (tmp_path / "summary" / "overview.md").read_text(encoding="utf-8")
    radar = (tmp_path / "summary" / "radar.svg").read_text(encoding="utf-8")
    for label in expected:
        assert label in overview
        assert label in radar
    validation_scatter = (
        tmp_path / "summary" / "ilume_scatter" / "validation"
        / "experiment__example.svg"
    ).read_text(encoding="utf-8")
    test_scatter = (
        tmp_path / "summary" / "ilume_scatter" / "test"
        / "experiment__example.svg"
    ).read_text(encoding="utf-8")
    assert "validation · n=5" in validation_scatter
    assert "test · n=1" in test_scatter
    assert "ILUME Capacity v1 (Base)" in validation_scatter
    assert "ILUME Capacity v1 (Base)" in test_scatter
    validation_size = float(
        re.search(r'data-point-size="([0-9.]+)"', validation_scatter).group(1)
    )
    test_size = float(
        re.search(r'data-point-size="([0-9.]+)"', test_scatter).group(1)
    )
    assert test_size > validation_size
    assert test_size == 6.0
    assert validation_size > 4.0


def test_radar_task_order_groups_related_properties_and_sorts_unknowns():
    tasks = (
        "experiment/z_unknown",
        "experiment/static_relative_permittivity",
        "experiment/dynamic_relative_permittivity",
        "experiment/a_unknown",
        "experiment/density",
        "experiment/electrical_conductivity",
    )

    assert _ordered_radar_tasks(tasks) == (
        "experiment/electrical_conductivity",
        "experiment/density",
        "experiment/dynamic_relative_permittivity",
        "experiment/static_relative_permittivity",
        "experiment/a_unknown",
        "experiment/z_unknown",
    )


# --- D-MPNN runtime smoke ---

def _task(component_count: int) -> BenchmarkTask:
    return BenchmarkTask(
        benchmark="stage3",
        task_id="experiment/tiny",
        slots=tuple(f"component_{index}" for index in range(component_count)),
        condition_columns=("temperature_K", "pressure_kPa"),
        target_columns=("value",),
        audit_columns=(),
        train_paths=(Path("train.csv"),),
        valid_paths=(Path("valid.csv"),),
        test_path=Path("test.csv"),
        fold=1,
        meta_group="tiny",
        registry_payload={"test": True},
    )

def _scalar_bundle(component_count: int) -> DMPNNTrainingBundle:
    smiles = ("CC", "O", "[Na+]")[:component_count]
    rows = tuple(tuple(smiles) for _ in range(4))
    targets = np.asarray([[-1.0], [0.0], [1.0], [2.0]], dtype=np.float64)
    conditions = np.asarray(
        [[290.0, 100.0], [300.0, 110.0], [310.0, 120.0], [320.0, 130.0]],
        dtype=np.float64,
    )
    raw = RawDataset(
        components=rows,
        component_count=component_count,
        conditions=conditions,
        targets=targets,
        source_rows=tuple(f"tiny:{index}" for index in range(2, 6)),
        audit_rows=({}, {}, {}, {}),
    )
    target_stats = TargetStats.fit(targets)
    condition_stats = ConditionStats.fit(conditions)
    dataset = _scalar_dataset(raw, target_stats, condition_stats)
    return DMPNNTrainingBundle(
        task=_task(component_count),
        train_dataset=dataset,
        valid_dataset=dataset,
        target_stats=target_stats,
        condition_stats=condition_stats,
        source_hashes={},
        training_identity=semantic_identity(
            "benchmark.training.v1", {"synthetic_components": component_count}
        ),
        target_level="molecule",
        component_count=component_count,
    )

@DMPNN_ONLY
def test_one_epoch_scalar_and_multicomponent_save_reload_smoke(
    tmp_path: Path,
) -> None:
    component_count = 2
    from chemprop.models.utils import load_model

    config = load_benchmark_config("configs/benchmarks/dmpnn.yaml")
    config = replace(
        config,
        training={
            **config.training,
            "batch_size": 2,
            "max_epochs": 1,
            "warmup_epochs": 0,
        },
    )
    for count in (2, 3):
        candidate = build_dmpnn_model(config, _scalar_bundle(count))
        message_passing = candidate.message_passing
        assert message_passing.shared is True
        assert len(message_passing.blocks) == count
        assert all(block is message_passing.blocks[0] for block in message_passing.blocks)
        assert {id(parameter) for parameter in message_passing.parameters()} == {
            id(parameter) for parameter in message_passing.blocks[0].parameters()
        }
        assert message_passing.output_dim == count * config.model["message_hidden_dim"]
    output = tmp_path / f"components-{component_count}"
    summary = train_dmpnn_bundle(config, _scalar_bundle(component_count), output)
    assert summary["epochs_ran"] == 1
    assert (output / "model.pt").is_file()
    first = load_model(
        output / "model.pt", multicomponent=component_count > 1
    )
    second = load_model(
        output / "model.pt", multicomponent=component_count > 1
    )
    dataset = _scalar_bundle(component_count).valid_dataset
    np.testing.assert_allclose(
        _predict(first, dataset),
        _predict(second, dataset),
        rtol=0,
        atol=0,
    )
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["format_version"] == 2
    assert checkpoint["final_epoch"] == 1
    assert checkpoint["final_valid_raw_mae"] == pytest.approx(
        checkpoint["final_valid_normalized_mae"]
        * checkpoint["target_statistics"]["scale"][0]
    )


# --- SPMM baseline contracts ---


class FakeSPMMTokenizer:
    pad_token_id = 0
    cls_token_id = 2

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def encode(self, sequence: str, **kwargs):
        truncated = bool(kwargs.get("truncation"))
        self.calls.append((sequence, truncated))
        smiles = sequence.removeprefix("[CLS]")
        content = 101 if len(smiles) > 100 else max(1, len(smiles))
        values = [2, 2, *([5] * content), 3]
        if truncated and len(values) > int(kwargs["max_length"]):
            values = values[: int(kwargs["max_length"]) - 1] + [3]
        return values


def _spmm_task() -> BenchmarkTask:
    return BenchmarkTask(
        benchmark="stage3",
        task_id="experiment/spmm_tiny",
        slots=("cation", "anion"),
        condition_columns=("temperature_K",),
        target_columns=("value",),
        audit_columns=(),
        train_paths=(),
        valid_paths=(),
        test_path=Path("test.csv"),
        fold=1,
        meta_group="tiny",
        registry_payload={"task_id": "experiment/spmm_tiny"},
    )


def _spmm_raw() -> RawDataset:
    return RawDataset(
        components=(
            ("F[C@H](Cl)Br", "[Cl-]"),
            ("F[C@@H](Cl)Br", "[Cl-]"),
            ("C" * 101, "[Cl-]"),
        ),
        component_count=2,
        conditions=np.asarray([[290.0], [300.0], [310.0]]),
        targets=np.asarray([[1.0], [2.0], [4.0]]),
        source_rows=("tiny.csv:2", "tiny.csv:3", "tiny.csv:4"),
        audit_rows=({}, {}, {}),
    )


def test_spmm_asset_snapshot_rejects_hash_mismatch(tmp_path: Path, monkeypatch) -> None:
    config = load_benchmark_config("configs/benchmarks/spmm.yaml")
    checkout = tmp_path / "upstream"
    checkout.mkdir()
    checkpoint = tmp_path / "checkpoint_SPMM.ckpt"
    checkpoint.write_bytes(b"x")
    names = (
        "SPMM_models.py",
        "xbert.py",
        "d_regression.py",
        "vocab_bpe_300.txt",
        "config_bert.json",
    )
    for name in names:
        (checkout / name).write_text("fixture", encoding="utf-8")
    config = replace(
        config,
        model={
            **config.model,
            "pretrained_checkpoint": checkpoint.as_posix(),
            "pretrained_size": 1,
        },
    )
    expected = {
        "SPMM_models.py": config.model["spmm_source_sha256"],
        "xbert.py": config.model["xbert_source_sha256"],
        "d_regression.py": config.model["regression_source_sha256"],
        "vocab_bpe_300.txt": config.model["vocab_sha256"],
        "config_bert.json": config.model["bert_config_sha256"],
        "checkpoint_SPMM.ckpt": config.model["pretrained_sha256"],
    }
    monkeypatch.setattr(
        "benchmarks.spmm.environment.repository_path",
        lambda value: checkpoint if str(value).endswith(".ckpt") else checkout,
    )
    monkeypatch.setattr(
        "benchmarks.spmm.environment.sha256_file", lambda path: expected[path.name]
    )
    monkeypatch.setattr(
        "benchmarks.spmm.environment.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=f"{config.model['revision']}\n"
        ),
    )

    class Tokenizer:
        vocab_size = 300
        pad_token_id = 0
        unk_token_id = 1
        cls_token_id = 2
        sep_token_id = 3
        mask_token_id = 1

    monkeypatch.setattr(
        "benchmarks.spmm.environment._spmm_tokenizer",
        lambda path, max_input_chars: Tokenizer(),
    )
    assert spmm_asset_snapshot(config)["revision"] == config.model["revision"]
    expected["xbert.py"] = "wrong"
    with pytest.raises(RuntimeError, match="asset hash mismatch"):
        spmm_asset_snapshot(config)


def test_spmm_official_token_path_cache_truncation_collision_and_conditions() -> None:
    raw = _spmm_raw()
    task = _spmm_task()
    tokenizer = FakeSPMMTokenizer()
    cache = {}
    condition_stats = SPMMConditionStats.fit(raw.conditions)
    prepared = prepare_spmm_split(
        raw,
        task,
        "train",
        tokenizer,
        cache,
        condition_stats,
        max_length=100,
    )
    assert prepared.audit["collision_group_count"] == 1
    assert prepared.audit["collision_affected_rows"] == 2
    assert prepared.audit["truncated_rows"] == ["tiny.csv:4"]
    assert max(len(values) for values, _ in cache.values()) == 99
    assert all(int(values[0]) == 2 for values, _ in cache.values())
    assert spmm_row_token_lengths(prepared, cache)[2] == 99
    call_count = len(tokenizer.calls)
    prepare_spmm_split(
        raw,
        task,
        "valid",
        tokenizer,
        cache,
        condition_stats,
        max_length=100,
    )
    assert len(tokenizer.calls) == call_count
    collate = spmm_collate(
        prepared, cache, TargetStats.fit(raw.targets), pad_token_id=0
    )
    input_ids, attention_mask, conditions, targets = collate([0, 1, 2])
    assert input_ids.shape == attention_mask.shape == (6, 99)
    assert conditions[:, 0].tolist() == pytest.approx([-1.2247449, 0.0, 1.2247449])
    assert targets.shape == (3, 1)


def test_spmm_sortish_sampler_is_deterministic_and_covers_each_epoch() -> None:
    lengths = tuple(range(37))
    sampler = SPMMSortishBatchSampler(
        lengths, batch_size=4, window_batches=2, seed=42
    )
    epoch0 = list(sampler)
    assert epoch0 == list(sampler)
    assert len(epoch0) == 10
    assert sorted(index for batch in epoch0 for index in batch) == list(range(37))
    assert all(
        [lengths[index] for index in batch]
        == sorted(lengths[index] for index in batch)
        for batch in epoch0
    )
    sampler.set_epoch(1)
    epoch1 = list(sampler)
    assert epoch1 != epoch0
    assert sorted(index for batch in epoch1 for index in batch) == list(range(37))


class FakeSPMMBert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls = 0

    def forward(self, input_ids, attention_mask, return_dict, mode):
        self.calls += 1
        assert mode == "text"
        states = input_ids.float().unsqueeze(-1).repeat(1, 1, 4) * self.weight
        return SimpleNamespace(last_hidden_state=states)


class FakeSPMMTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert = FakeSPMMBert()


def test_spmm_multicomponent_uses_one_shared_forward_in_registry_order() -> None:
    encoder = FakeSPMMTextEncoder()
    predictor = torch.nn.Linear(9, 1)
    model = SharedSPMMRegressor(
        encoder,
        predictor,
        component_count=2,
        condition_dim=1,
        hidden_dim=4,
        load_audit={"loaded": True},
    )
    input_ids = torch.cat(
        (
            torch.ones((2, 3), dtype=torch.long),
            torch.ones((2, 3), dtype=torch.long) * 2,
        )
    )
    output = model(
        input_ids,
        torch.ones_like(input_ids),
        torch.asarray([[10.0], [20.0]]),
    )
    assert output.shape == (2, 1)
    assert encoder.bert.calls == 1
    assert model.text_encoder is encoder


def test_spmm_checkpoint_filters_exact_used_text_encoder_state(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_benchmark_config("configs/benchmarks/spmm.yaml")
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    checkpoint_path.write_bytes(b"x")
    config = replace(
        config,
        model={
            **config.model,
            "pretrained_checkpoint": checkpoint_path.as_posix(),
            "pretrained_sha256": "trusted",
            "pretrained_size": 1,
        },
    )
    target = {f"bert.used.{index}": torch.zeros(1) for index in range(102)}

    class Encoder:
        def state_dict(self):
            return target

        def load_state_dict(self, values, strict):
            assert strict is True
            assert set(values) == set(target)

    source = {
        f"text_encoder.{key}": value.clone() for key, value in target.items()
    }
    source.update(
        {f"text_encoder_m.ignored.{index}": torch.zeros(1) for index in range(656)}
    )
    monkeypatch.setattr(
        "benchmarks.spmm.adapter._upstream_paths",
        lambda config: (Path("xbert"), Path("vocab"), Path("config"), checkpoint_path),
    )
    monkeypatch.setattr("benchmarks.spmm.adapter.sha256_file", lambda path: "trusted")
    monkeypatch.setattr(
        "benchmarks.spmm.adapter.torch.load", lambda *args, **kwargs: {"state_dict": source}
    )
    audit = _load_pretrained_encoder(config, Encoder())
    assert audit["source_state_entries"] == 758
    assert audit["loaded_text_encoder_entries"] == 102
    source.pop("text_encoder.bert.used.0")
    source["other.ignored"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="load contract mismatch"):
        _load_pretrained_encoder(config, Encoder())


def test_spmm_scheduler_has_exact_warmup_peak_and_cosine_floor() -> None:
    values = [
        _scheduled_learning_rate(
            step,
            total_steps=100,
            warmup_steps=10,
            warmup_learning_rate=5.0e-6,
            peak_learning_rate=5.0e-5,
            minimum_learning_rate=3.0e-6,
        )
        for step in range(100)
    ]
    assert values[0] == pytest.approx(5.0e-6)
    assert values[9] == pytest.approx(5.0e-5)
    assert values[10] == pytest.approx(5.0e-5)
    assert values[-1] == pytest.approx(3.0e-6)
    assert all(left <= right for left, right in zip(values[:9], values[1:10]))
    assert all(left >= right for left, right in zip(values[10:-1], values[11:]))


# --- LlaSMol baseline contracts ---


def _llasmol_task(*, slots=("cation", "anion")) -> BenchmarkTask:
    return BenchmarkTask(
        benchmark="stage3",
        task_id="experiment/llasmol_tiny",
        slots=slots,
        condition_columns=("temperature_K",),
        target_columns=("value",),
        audit_columns=(),
        train_paths=(),
        valid_paths=(),
        test_path=Path("test.csv"),
        fold=1,
        meta_group="tiny",
        registry_payload={"task_id": "experiment/llasmol_tiny"},
    )


class FakeLlaSMolTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, sequence: str, **kwargs):
        truncated = bool(kwargs.get("truncation"))
        self.calls.append((sequence, truncated))
        length = 520 if "LONG" in sequence else 2 + len(sequence) % 9
        values = [1, *([5] * (length - 1))]
        if truncated:
            values = values[: int(kwargs["max_length"])]
        return {"input_ids": values, "attention_mask": [1] * len(values)}


def test_llasmol_missing_assets_fail_before_model_loading(
    tmp_path: Path, monkeypatch
) -> None:
    from benchmarks.llasmol.environment import llasmol_asset_snapshot

    config = load_benchmark_config("configs/benchmarks/llasmol.yaml")
    monkeypatch.setattr(
        "benchmarks.llasmol.environment.repository_path", lambda path: tmp_path / str(path)
    )
    with pytest.raises(FileNotFoundError, match="local assets are incomplete"):
        llasmol_asset_snapshot(config)


def test_llasmol_whole_il_multiview_token_cache_truncation_and_conditions() -> None:
    task = _llasmol_task(slots=("cation", "anion", "solute"))
    raw = RawDataset(
        components=(("[C+]", "[Cl-]", "CCO"), ("[N+]", "[Br-]", "LONG")),
        component_count=3,
        conditions=np.asarray([[290.0], [310.0]]),
        targets=np.asarray([[1.0], [3.0]]),
        source_rows=("tiny.csv:2", "tiny.csv:3"),
        audit_rows=({}, {}),
    )
    assert llasmol_task_prefix(task.task_id) == "<LLASMOL_TINY>"
    views, names = llasmol_model_views(task, raw.components[0])
    assert views == ("[C+].[Cl-]", "CCO")
    assert names == ("ionic_liquid", "solute")
    tokenizer = FakeLlaSMolTokenizer()
    cache = {}
    stats = LlaSMolConditionStats.fit(raw.conditions)
    prepared = prepare_llasmol_split(
        raw, task, "train", tokenizer, cache, stats, max_length=512
    )
    assert prepared.model_views[0] == (
        "<LLASMOL_TINY>\n[C+].[Cl-]",
        "<LLASMOL_TINY>\nCCO",
    )
    assert prepared.audit["truncated_rows"] == ["tiny.csv:3"]
    assert len(tokenizer.calls) == 8
    prior_calls = len(tokenizer.calls)
    prepare_llasmol_split(
        raw, task, "valid", tokenizer, cache, stats, max_length=512
    )
    assert len(tokenizer.calls) == prior_calls
    collate = llasmol_collate(
        prepared, cache, TargetStats.fit(raw.targets), pad_token_id=0
    )
    input_ids, attention_mask, conditions, targets = collate([0, 1])
    assert input_ids.shape == attention_mask.shape == (4, 512)
    assert conditions[:, 0].tolist() == pytest.approx([-1.0, 1.0])
    assert targets.shape == (2, 1)
    assert torch.all(input_ids[attention_mask == 0] == 0)


class FakeLlaSMolBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        self.calls += 1
        states = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
        return SimpleNamespace(last_hidden_state=states)


class FakeLlaSMolPEFT(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.base_model = SimpleNamespace(
            model=SimpleNamespace(model=self.backbone)
        )


def test_llasmol_multiview_uses_one_backbone_forward_and_masked_mean() -> None:
    backbone = FakeLlaSMolBackbone()
    predictor = torch.nn.Linear(9, 1, bias=False)
    predictor.weight.data.fill_(1.0)
    model = SharedLlaSMolRegressor(
        FakeLlaSMolPEFT(backbone),
        predictor,
        view_count=2,
        condition_dim=1,
        hidden_dim=4,
        load_audit={"loaded": True},
    )
    input_ids = torch.asarray(
        [[0, 1, 3], [1, 3, 5], [0, 2, 4], [2, 4, 6]], dtype=torch.long
    )
    attention_mask = torch.asarray(
        [[0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.long
    )
    output = model(input_ids, attention_mask, torch.asarray([[10.0], [20.0]]))
    assert output[:, 0].tolist() == pytest.approx([30.0, 48.0])
    assert backbone.calls == 1


def test_llasmol_adapter_namespace_sampler_and_scheduler_contracts(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_benchmark_config("configs/benchmarks/llasmol.yaml")
    checkpoint = tmp_path / "adapter_model.bin"
    checkpoint.write_bytes(b"x")
    config = replace(
        config,
        model={
            **config.model,
            "adapter_snapshot": tmp_path.as_posix(),
            "adapter_model_size": 1,
            "adapter_model_sha256": "trusted",
        },
    )
    state = {
        f"base_model.model.model.layers.{layer}.{scope}.{module}.lora_{side}.weight":
        torch.zeros(1, dtype=torch.bfloat16)
        for layer in range(32)
        for scope, modules in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        )
        for module in modules
        for side in ("A", "B")
    }
    monkeypatch.setattr(
        "benchmarks.llasmol.adapter.repository_path", lambda path: tmp_path
    )
    monkeypatch.setattr("benchmarks.llasmol.adapter.sha256_file", lambda path: "trusted")
    monkeypatch.setattr("benchmarks.llasmol.adapter.torch.load", lambda *args, **kwargs: state)
    _, audit = _official_adapter_state(config)
    assert audit["state_entries"] == 448
    state.pop(next(iter(state)))
    with pytest.raises(RuntimeError, match="entry count"):
        _official_adapter_state(config)

    sampler = LlaSMolSortishBatchSampler(
        tuple(range(37)), batch_size=4, window_batches=2, seed=42
    )
    epoch0 = list(sampler)
    assert epoch0 == list(sampler)
    assert sorted(index for batch in epoch0 for index in batch) == list(range(37))
    sampler.set_epoch(1)
    assert list(sampler) != epoch0
    factors = [
        llasmol_scheduled_factor(step, total_steps=100, warmup_steps=5)
        for step in range(100)
    ]
    assert factors[4] == pytest.approx(1.0)
    assert factors[5] == pytest.approx(1.0)
    assert factors[-1] == pytest.approx(0.0)


# --- AIonopedia baseline contracts ---


def _aionopedia_task(
    *, slots: tuple[str, ...], conditions: tuple[str, ...], target: str = "secret_target",
) -> BenchmarkTask:
    return BenchmarkTask(
        benchmark="stage3",
        task_id="experiment/aionopedia_tiny",
        slots=slots,
        condition_columns=conditions,
        target_columns=(target,),
        audit_columns=(),
        train_paths=(),
        valid_paths=(),
        test_path=Path("test.csv"),
        fold=1,
        meta_group="tiny",
        registry_payload={"task_id": "experiment/aionopedia_tiny"},
    )


def _aionopedia_raw(
    components: tuple[str, ...], conditions: tuple[float, ...]
) -> RawDataset:
    return RawDataset(
        components=(components,),
        component_count=len(components),
        conditions=np.asarray([conditions], dtype=np.float64).reshape(1, len(conditions)),
        targets=np.asarray([[1.0]], dtype=np.float64),
        source_rows=("tiny.csv:2",),
        audit_rows=({},),
    )


@pytest.mark.parametrize(
    ("slots", "conditions", "components", "values", "topology", "graph_roles"),
    (
        (("cation", "anion"), (), ("[Na+]", "[Cl-]"), (), 3, ("", "[Na+]", "[Cl-]")),
        (("cation", "anion"), ("temperature_K",), ("[Na+]", "[Cl-]"), (300.0,), 2, ("", "[Na+]", "[Cl-]")),
        (("cation", "anion", "solute"), ("temperature_K",), ("[Na+]", "[Cl-]", "O"), (300.0,), 1, ("O", "[Na+]", "[Cl-]")),
        (("solute", "solvent"), ("temperature_K",), ("O", "CCO"), (300.0,), 0, ("O", "CCO", "")),
    ),
)
def test_aionopedia_registry_topologies_and_prompts_do_not_leak_target(
    slots, conditions, components, values, topology, graph_roles
) -> None:
    task = _aionopedia_task(slots=slots, conditions=conditions)
    prepared = prepare_aionopedia_split(
        task, _aionopedia_raw(components, values), None
    )
    assert prepared.topology == topology
    assert prepared.graph_roles[0] == graph_roles
    assert "secret_target" not in prepared.prompts[0]
    assert "aionopedia_tiny" not in prepared.prompts[0]


def test_aionopedia_condition_scales_and_prompt_units() -> None:
    task = _aionopedia_task(
        slots=("cation", "anion"),
        conditions=("temperature_K", "pressure_kPa", "frequency_MHz"),
    )
    pressure = AIonopediaSampleStats.fit(np.asarray([100.0, 200.0]), allow_constant=True)
    prepared = prepare_aionopedia_split(
        task,
        _aionopedia_raw(("[Na+]", "[Cl-]"), (300.0, 150.0, 18000.0)),
        pressure,
    )
    assert prepared.temperature.tolist() == pytest.approx([0.3])
    assert prepared.extra_conditions["pressure"].tolist() == pytest.approx([0.0])
    assert prepared.extra_conditions["frequency"].tolist() == pytest.approx([18.0])
    assert prepared.active_conditions == ("pressure", "frequency")
    assert "temperature 300K" in prepared.prompts[0]
    assert "pressure 150kPa" in prepared.prompts[0]
    assert "frequency 18000MHz" in prepared.prompts[0]

    wavelength_task = _aionopedia_task(
        slots=("cation", "anion"), conditions=("temperature_K", "wavelength_nm")
    )
    wavelength = prepare_aionopedia_split(
        wavelength_task,
        _aionopedia_raw(("[Na+]", "[Cl-]"), (298.15, 589.0)),
        None,
    )
    assert wavelength.extra_conditions["wavelength"].tolist() == pytest.approx([0.589])


def test_aionopedia_graph_preprocessing_matches_official_golden_contract() -> None:
    graph = aionopedia_graph("C")
    assert graph.x.shape == (1, 35)
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_attr.shape == (0, 11)
    assert hashlib.sha256(graph.x.numpy().tobytes()).hexdigest() == (
        "926e24a9ca06fd679201bf03f106f826266e2b9aa21c5d28ec5a17e782107eea"
    )


def test_aionopedia_sample_std_scheduler_and_condition_tokens() -> None:
    stats = AIonopediaSampleStats.fit(np.asarray([1.0, 3.0]), allow_constant=False)
    assert stats.mean == 2.0
    assert stats.scale == pytest.approx(2 ** 0.5)
    constant = AIonopediaSampleStats.fit(np.asarray([5.0, 5.0]), allow_constant=True)
    assert constant.constant and constant.scale == 1.0
    factors = [
        aionopedia_scheduled_factor(step, total_steps=100, warmup_steps=50)
        for step in range(101)
    ]
    assert factors[0] == 0.0
    assert factors[50] == 1.0
    assert factors[-1] == 0.0

    class FakeLLM(torch.nn.Module):
        def forward(self, input_ids, attention_mask, output_hidden_states, use_cache):
            hidden = torch.zeros((*input_ids.shape, 1024), dtype=torch.float32)
            return SimpleNamespace(hidden_states=(hidden,))

    model = AIonopediaRegressor(FakeLLM()).eval()
    cation = Batch.from_data_list([aionopedia_graph("C")])
    empty = Batch.from_data_list([
        Data(
            x=torch.empty((0, 35)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 11)),
        )
    ])
    common = {
        "solute_graph": empty,
        "cation_graph": cation,
        "anion_graph": empty,
        "temperature": torch.tensor([0.3]),
        "topology": torch.tensor([2]),
    }
    without, _ = model.encode_graphs(
        **common, extra_conditions={}, active_conditions=()
    )
    with_conditions, _ = model.encode_graphs(
        **common,
        extra_conditions={
            "pressure": torch.tensor([0.0]),
            "frequency": torch.tensor([1.0]),
            "wavelength": torch.tensor([0.5]),
        },
        active_conditions=("pressure", "frequency", "wavelength"),
    )
    assert with_conditions.shape[1] == without.shape[1] + 3


# --- ILTransR baseline contracts ---


def _iltransr_task(
    tmp_path: Path,
    *,
    slots: tuple[str, ...] = ("cation", "anion"),
    conditions: tuple[str, ...] = (),
    train_paths: tuple[Path, ...] = (),
    valid_paths: tuple[Path, ...] = (),
    test_path: Path | None = None,
) -> BenchmarkTask:
    return BenchmarkTask(
        benchmark="stage3",
        task_id="experiment/iltransr_tiny",
        slots=slots,
        condition_columns=conditions,
        target_columns=("secret_target",),
        audit_columns=(),
        train_paths=train_paths,
        valid_paths=valid_paths,
        test_path=test_path or tmp_path / "test.csv",
        fold=1,
        meta_group="tiny",
        registry_payload={"task_id": "experiment/iltransr_tiny"},
    )


def test_formal_iltransr_config_recipes_and_property_weight_guard() -> None:
    config = load_benchmark_config("configs/benchmarks/iltransr.yaml")
    tasks = configured_tasks(config, "stage3")
    official = {
        "experiment/density", "experiment/viscosity", "experiment/heat_capacity",
        "experiment/melting_point", "experiment/thermal_decomposition_temperature",
        "experiment/x_co2", "experiment/pec50",
    }
    assert set(config.training["official_recipes"]) == official
    assert sum(resolve_iltransr_recipe(config, task)["source"] == "official_notebook" for task in tasks) == 7
    assert sum(resolve_iltransr_recipe(config, task)["source"] == "registered_fallback" for task in tasks) == 13
    assert resolve_iltransr_recipe(config, "experiment/x_co2")["epochs"] == 10
    assert resolve_iltransr_recipe(config, "experiment/electrical_conductivity")["epochs"] == 10
    assert config.training["loss"] == "train_population_zscore_l1"
    changed = config.to_dict()
    changed["model"]["generic_checkpoint"] = "density_best.params"
    with pytest.raises(ValueError, match="generic-pretraining recipe"):
        benchmark_config_from_dict(changed)


def test_iltransr_character_vocab_eos_padding_clip_and_unknown(tmp_path: Path) -> None:
    mapping = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3, "A": 4}
    mapping.update({chr(0x100 + index): index + 5 for index in range(67)})
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"token_to_idx": mapping}), encoding="utf-8")
    vocabulary = ILTransRCharacterVocabulary(path)

    values, length, audit = vocabulary.encode("A" * 99)
    assert length == 100 and values[-1].item() == 3
    assert audit == {
        "characters": 99, "tokens_before_clip": 100, "clipped": 0,
        "eos_clipped": 0, "unknown_tokens": 0,
    }
    values, length, audit = vocabulary.encode("A" * 100 + "X")
    assert length == 100 and 3 not in values.tolist()
    assert audit["clipped"] == audit["eos_clipped"] == 1
    assert audit["unknown_tokens"] == 1


@pytest.mark.parametrize(
    ("slots", "components", "roles"),
    (
        (("cation", "anion"), ("[Na+]", "[Cl-]"), ("ionic_liquid",)),
        (("cation", "anion", "solute"), ("[Na+]", "[Cl-]", "F[C@H](Cl)Br"), ("ionic_liquid", "solute")),
        (("solute", "solvent"), ("F[C@H](Cl)Br", "CCO"), ("solute", "solvent")),
    ),
)
def test_iltransr_registry_views_are_non_isomeric_and_ordered(
    tmp_path: Path, slots, components, roles,
) -> None:
    views, names = iltransr_model_views(
        _iltransr_task(tmp_path, slots=slots), components
    )
    assert names == roles
    assert all("@" not in value for value in views)
    if slots[:2] == ("cation", "anion"):
        assert "." in views[0]


def test_iltransr_condition_population_includes_all_covariates_and_constants(
    tmp_path: Path,
) -> None:
    paths = tuple(tmp_path / f"fold{fold}.csv" for fold in range(1, 3))
    test_path = tmp_path / "test.csv"
    for path, temperature in zip((*paths, test_path), (280.0, 300.0, 320.0), strict=True):
        _write_csv(
            path,
            ["temperature_K", "pressure_kPa", "secret_target"],
            [{"temperature_K": temperature, "pressure_kPa": 101.325, "secret_target": "not-read"}],
        )
    task = _iltransr_task(
        tmp_path,
        conditions=("temperature_K", "pressure_kPa"),
        train_paths=(paths[0],),
        valid_paths=(paths[1],),
        test_path=test_path,
    )
    values, digest = iltransr_condition_population(task)
    stats = ILTransRConditionStats.fit(task.condition_columns, values, digest)
    assert stats.population_rows == 3
    assert stats.mean == pytest.approx((300.0, 101.325))
    assert stats.scale[0] == pytest.approx(np.std([280.0, 300.0, 320.0], ddof=0))
    assert stats.scale[1] == 1.0 and stats.constant == (False, True)
    assert stats.normalize(np.asarray([[300.0, 101.325]])).tolist() == [[0.0, 0.0]]


def test_iltransr_shared_backbone_full_fine_tuning_and_strict_state(
    tmp_path: Path,
) -> None:
    from safetensors.torch import save_file

    transformer = ILTransRTransformer(72)
    converted = tmp_path / "encoder.safetensors"
    save_file(transformer.state_dict(), str(converted))
    loaded, audit = load_converted_transformer(str(converted), vocab_size=72)
    assert audit["strict"] and audit["loaded_tensors"] == 40
    model = ILTransRRegressor(
        loaded,
        topology="smiles_only",
        view_count=2,
        condition_dim=0,
        dropout=0.1,
        load_audit=audit,
    )
    calls = []
    hook = model.transformer.register_forward_hook(lambda *_: calls.append(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    before = model.transformer.embedding.weight.detach().clone()
    token_ids = torch.randint(4, 72, (4, 10))
    predictions = model(token_ids, torch.full((4,), 10), torch.empty((2, 0)))
    torch.nn.functional.l1_loss(predictions, torch.asarray([0.0, 1.0])).backward()
    assert len(calls) == 1
    assert all(parameter.requires_grad and parameter.grad is not None for parameter in model.transformer.parameters())
    assert all(parameter.grad is not None for parameter in model.textcnn.parameters())
    optimizer.step()
    hook.remove()
    assert not torch.equal(before, model.transformer.embedding.weight)


def test_iltransr_two_bucket_sampler_is_deterministic_and_complete() -> None:
    sampler = ILTransRTwoBucketBatchSampler(range(1, 10), batch_size=3, seed=42)
    first = list(sampler)
    assert len(first) == len(sampler) == 4
    assert sorted(index for batch in first for index in batch) == list(range(9))
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)


# --- AIFC baseline contracts ---


def _aifc_task(
    tmp_path: Path,
    *,
    slots: tuple[str, ...] = ("cation", "anion"),
    conditions: tuple[str, ...] = (),
) -> BenchmarkTask:
    return BenchmarkTask(
        benchmark="stage3",
        task_id="experiment/aifc_tiny",
        slots=slots,
        condition_columns=conditions,
        target_columns=("secret_target",),
        audit_columns=(),
        train_paths=(tmp_path / "fold2.csv",),
        valid_paths=(tmp_path / "fold1.csv",),
        test_path=tmp_path / "test.csv",
        fold=1,
        meta_group="tiny",
        registry_payload={"task_id": "experiment/aifc_tiny"},
    )


def test_formal_aifc_config_assets_recipes_and_final_state_contract() -> None:
    config = load_benchmark_config("configs/benchmarks/aifc.yaml")
    assert config.seed == 1000
    assert resolve_aifc_architecture(
        config, "experiment/thermal_decomposition_temperature"
    )["hidden_dim"] == 208
    assert resolve_aifc_architecture(config, "experiment/density")["source"] == "fallback"
    assert config.model["fragment_commit"] == FRAGMENT_COMMIT
    assert config.model["fragment_blob"] == FRAGMENT_BLOB
    assert config.model["fragment_sha256"] == FRAGMENT_SHA256


def test_aifc_fragmentation_unknown_and_legacy_dgl_parity() -> None:
    fragment_path = Path("benchmarks/aifc/assets/My_fragments.csv")
    scheme = AIFCFragmentScheme.load(fragment_path)
    assert len(scheme.names) == len(set(scheme.names)) == 100
    assert set(scheme.priorities) == {1, 2, 3, 4, 5}
    unknown = smiles_to_aifc_graph("[Na+]", scheme)
    assert unknown.unknown_atoms == 1
    assert unknown.fragment_names == ("unknown",)
    assert not unknown.motif_nodes.any()
    parity = validate_legacy_parity(
        "benchmarks/aifc/assets/legacy_reference.json", fragment_path
    )
    assert max(parity["max_abs_errors"].values()) <= parity["threshold"]


@pytest.mark.parametrize(
    ("slots", "components", "roles"),
    (
        (("cation", "anion"), ("[Na+]", "[Cl-]"), ("ionic_liquid",)),
        (("cation", "anion", "solute"), ("[Na+]", "[Cl-]", "CCO"), ("cation", "anion", "solute")),
        (("solute", "solvent"), ("CCO", "O"), ("solute", "solvent")),
    ),
)
def test_aifc_registry_views_use_shared_encoder_order(
    tmp_path: Path, slots, components, roles,
) -> None:
    views, names = aifc_model_views(_aifc_task(tmp_path, slots=slots), components)
    assert names == roles
    assert len(views) == len(roles)
    if slots == ("cation", "anion"):
        assert "." in views[0]


def test_aifc_train_only_conditions_and_shared_encoder_gradient() -> None:
    stats = AIFCConditionStats.fit(
        ("temperature_K", "pressure_kPa", "frequency_MHz"),
        np.asarray([[280.0, 101.325, 10.0], [320.0, 101.325, 30.0]]),
    )
    assert stats.mean == pytest.approx((300.0, 101.325, 20.0))
    assert stats.constant == (False, True, False)
    assert stats.normalize(np.asarray([[300.0, 101.325, 20.0]])).tolist() == [[0.0, 0.0, 0.0]]

    scheme = AIFCFragmentScheme.load("benchmarks/aifc/assets/My_fragments.csv")
    graph = batch_aifc_graphs(
        [smiles_to_aifc_graph(value, scheme) for value in ("CCO", "O", "CCN", "CO")]
    )
    model = AIFCRegressor(
        fragment_dim=100, hidden_dim=16, num_heads=1, dropout=0,
        depth=2, layers=2, view_count=2, condition_dim=3,
    )
    calls = []
    predictor_inputs = []
    hook = model.encoder.register_forward_hook(lambda *_: calls.append(1))
    predictor_hook = model.predictor.register_forward_pre_hook(
        lambda _module, inputs: predictor_inputs.append(inputs[0].detach())
    )
    prediction = model(graph, torch.tensor([[-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]]))
    torch.nn.functional.mse_loss(prediction, torch.tensor([0.0, 1.0])).backward()
    hook.remove()
    predictor_hook.remove()
    assert calls == [1]
    assert predictor_inputs[0][:, -3:].tolist() == [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    assert model.encoder.fragment_node_embedding[0].weight.grad is not None
    assert model.encoder.fragment_heads[0].atom.layers[0].node_embedding[0].weight.grad is not None
    assert model.encoder.junction_heads[0].atom.layers[0].node_embedding[0].weight.grad is not None
    assert model.predictor[1].weight.grad is not None
