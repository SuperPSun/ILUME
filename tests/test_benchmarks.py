from __future__ import annotations

import csv

import json

import sys

import tempfile

import threading

import time

from pathlib import Path

import numpy as np

import pytest

import yaml

from benchmarks.common.config import benchmark_config_from_dict, load_benchmark_config

from benchmarks.common.data import configured_tasks, load_split, resolve_task

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

from benchmarks.common.features import (
    FeatureCache,
    FeaturePreprocessor,
    component_feature,
    feature_schema,
    raw_feature_matrix,
)

from benchmarks.common.metrics import mean_sample_std, regression_metrics

from common.identity import semantic_identity

from common.reporting import (
    REPORTING_SCHEMA_VERSION,
    STAGE2_CORE_EVALUATION_CONTRACT,
    STAGE2_PARTIAL_EVALUATION_CONTRACT,
    comparison_identity,
)

from stage2.atom_evaluation import PARTIAL_CHARGE_TASK, PARTIAL_CHARGE_UNIT

import scripts.benchmarks.sweep as sweep_module

from scripts.benchmarks.sweep import (
    _JobResult,
    _SweepState,
    _aggregate,
    _build_jobs,
    _parse_devices,
    _run as run_sweep_job,
    _schedule,
    _subprocess_env,
)

from benchmarks.common.summary import SUMMARY_FILES, publish_summary

from common.reporting import (
    STAGE2_BENCHMARK_SUITE_CONTRACT,
    comparison_identity,
    role_mae_diagnostics,
    stage2_full_comparison_identity,
    write_prediction_csv,
)

from stage2.evaluate import resolve_checkpoint_path

from dataclasses import replace

from types import SimpleNamespace

import torch

from benchmarks.common.config import load_benchmark_config

from benchmarks.common.data import BenchmarkTask, RawDataset

from benchmarks.common.engine import TargetStats

from benchmarks.common.environment import validate_dmpnn_environment

try:
    import chemprop  # noqa: F401
    from benchmarks.dmpnn.adapter import (
        ConditionStats,
        DMPNNTrainingBundle,
        _partial_dataset,
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
                "stage": 2,
                "task_id": "simulation/tiny",
                "task_kind": "object_property",
                "target_level": "object",
                "source_file": "simulation/tiny.csv",
                "target_columns": targets,
                "identity_columns": "cation;anion",
                "condition_columns": "temperature_K",
                "system_type": "il",
                "simulation_method": "test",
                "materialized_path": "stage2/tiny",
                "label_source": "materialized_csv",
                "resource_manifest": "",
                "strategies": "system_holdout",
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
    _write_csv(tmp_path / "stage2/tiny/train.csv", fields, values[:4])
    _write_csv(tmp_path / "stage2/tiny/valid.csv", fields, values[4:6])
    _write_csv(tmp_path / "stage2/tiny/test.csv", fields, values[6:])
    if name == "mlp":
        features = {"kind": "rdkit_2d", "radius": 2, "n_bits": 2048}
        model = {"hidden_dims": [8], "dropout": 0.0}
        training = {
            "optimizer": "adamw", "learning_rate": 0.01, "weight_decay": 0.0,
            "batch_size": 2, "max_epochs": 4, "early_stopping_patience": 2,
            "loss": "normalized_mse", "selection_metric": "raw_target_macro_mae",
            "device": "cpu", "precision": "fp32",
        }
    else:
        features = {"kind": "ecfp4", "radius": 2, "n_bits": 64}
        model = {
            "n_estimators": 8, "max_depth": 2, "learning_rate": 0.1,
            "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0,
            "objective": "reg:squarederror", "eval_metric": "mae", "tree_method": "hist",
        }
        training = {"early_stopping_rounds": 2, "n_jobs": 1, "device": "cpu", "target_space": "raw"}
    return benchmark_config_from_dict(
        {
            "name": name,
            "seed": 42,
            "data": {
                "data_root": str(tmp_path), "task_catalog": str(catalog),
                "stage3_authority_config": str(tmp_path / "unused.yaml"),
                "feature_cache": str(tmp_path / "features.sqlite3"),
            },
            "features": features,
            "model": model,
            "training": training,
            "stage3": {"enabled": False, "tasks": "all", "folds": [1, 2, 3, 4, 5]},
            "stage2_physics": {"enabled": True, "tasks": ["simulation/tiny"]},
        }
    )

def _tiny_stage3_config(tmp_path: Path):
    catalog = tmp_path / "stage3_catalog.csv"
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
                "target_columns": "value",
                "identity_columns": "cation;anion",
                "condition_columns": "temperature_K",
                "system_type": "il",
                "simulation_method": "",
                "materialized_path": "stage3/experiment/tiny",
                "label_source": "materialized_csv",
                "resource_manifest": "",
                "strategies": "il",
            }
        )
    fields = ["cation", "anion", "temperature_K", "value"]
    cations = ["[Li+]", "[Na+]", "[K+]", "[Rb+]", "C[N+](C)(C)C"]
    anions = ["[F-]", "[Cl-]", "[Br-]", "[I-]", "[Cl-]"]
    for fold in range(1, 6):
        _write_csv(
            tmp_path / f"stage3/experiment/tiny/IL/fold{fold}.csv",
            fields,
            [
                {"cation": cations[fold - 1], "anion": anions[fold - 1], "temperature_K": 290 + fold, "value": float(fold)},
                {"cation": cations[fold - 1], "anion": anions[fold - 1], "temperature_K": 300 + fold, "value": float(fold + 1)},
            ],
        )
    _write_csv(
        tmp_path / "stage3/experiment/tiny/test.csv",
        fields,
        [{"cation": "CC[N+](C)(C)C", "anion": "[Br-]", "temperature_K": 310, "value": 3.5}],
    )
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
    return benchmark_config_from_dict(
        {
            "name": "mlp",
            "seed": 42,
            "data": {
                "data_root": str(tmp_path), "task_catalog": str(catalog),
                "stage3_authority_config": str(authority),
                "feature_cache": str(tmp_path / "features.sqlite3"),
            },
            "features": {"kind": "rdkit_2d", "radius": 2, "n_bits": 2048},
            "model": {"hidden_dims": [8], "dropout": 0.0},
            "training": {
                "optimizer": "adamw", "learning_rate": 0.01, "weight_decay": 0.0,
                "batch_size": 2, "max_epochs": 3, "early_stopping_patience": 2,
                "loss": "normalized_mse", "selection_metric": "raw_target_macro_mae",
                "device": "cpu", "precision": "fp32",
            },
            "stage3": {"enabled": True, "tasks": "all", "folds": [1, 2, 3, 4, 5]},
            "stage2_physics": {"enabled": False, "tasks": []},
        }
    )

def test_formal_configs_and_registry_resolution(tmp_path: Path) -> None:
    config = load_benchmark_config("configs/benchmarks/mlp.yaml")
    assert len(configured_tasks(config, "stage3")) == 21
    assert configured_tasks(config, "stage2_physics") == (
        "simulation/heat_of_vaporization",
        "simulation/homo",
        "simulation/lumo",
    )
    solvation = resolve_task(config, "stage3", "experiment/solvation", 1)
    organic = resolve_task(config, "stage3", "experiment/transfer_organic", 1)
    orbital = resolve_task(config, "stage2_physics", "simulation/homo", None)
    assert solvation.slots == ("cation", "anion", "solute")
    assert organic.slots == ("solute", "solvent")
    assert orbital.slots == ("SMILES",) and orbital.target_columns == ("HOMO_eV",)
    assert orbital.audit_columns == (
        "ion_role", "provenance_source_file", "provenance_source_row"
    )
    missing_test = resolve_task(config, "stage3", "experiment/self_diffusion_coefficient", 1)
    empty = load_split(missing_test, "test")
    reporter = RecordingReporter()
    with FeatureCache(tmp_path / "empty-cache.sqlite3") as cache:
        matrix = raw_feature_matrix(
            empty, feature_schema(config.features), cache, reporter=reporter
        )
    assert matrix.shape == (0, 2 * 217 + 2)
    assert reporter.bars == []

def test_formal_dmpnn_config_resolves_109_training_jobs() -> None:
    config = load_benchmark_config("configs/benchmarks/dmpnn.yaml")
    stage3_tasks = configured_tasks(config, "stage3")
    stage2_tasks = configured_tasks(config, "stage2_physics")
    assert len(stage3_tasks) == 21
    assert stage2_tasks == (
        "simulation/heat_of_vaporization",
        "simulation/homo",
        "simulation/lumo",
        "simulation/partial_atomic_charge",
    )
    assert len(stage3_tasks) * len(config.stage3.folds) + len(stage2_tasks) == 109
    assert config.features is None
    assert config.data.feature_cache is None

def test_dmpnn_environment_dispatches_once_before_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_benchmark_config("configs/benchmarks/dmpnn.yaml")
    command = environment_command(
        config,
        ("scripts/benchmarks/train.py", "--config", "configs/benchmarks/dmpnn.yaml"),
        conda="/opt/conda/bin/conda",
    )
    assert command[:7] == [
        "/opt/conda/bin/conda",
        "run",
        "--no-capture-output",
        "-n",
        "ilume-dmpnn",
        "python",
        str((Path.cwd() / "scripts/benchmarks/train.py").resolve()),
    ]
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

def test_training_prepare_uses_only_training_and_validation_splits(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    test_path = tmp_path / "stage2/tiny/test.csv"
    test_path.unlink()
    reporter = RecordingReporter()
    bundle = prepare_training(
        config,
        "stage2_physics",
        "simulation/tiny",
        None,
        reporter=reporter,
    )
    assert bundle.train_features.shape[0] == 4
    assert [(bar.total, bar.n, bar.closed) for bar in reporter.bars] == [
        (4, 4, True),
        (2, 2, True),
    ]
    assert "train features" in reporter.bars[0].desc
    assert "valid features" in reporter.bars[1].desc

def test_mlp_train_checkpoint_and_test_evaluation(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path, targets="left;right")
    bundle = prepare_training(config, "stage2_physics", "simulation/tiny", None)
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
    assert 1 <= summary["best_epoch"] <= 4
    assert len(reporter.bars) == 1
    assert reporter.bars[0].n == summary["epochs_ran"]
    assert reporter.bars[0].closed
    assert set(reporter.bars[0].postfixes[-1]) == {
        "train_mse", "val_mae", "best", "patience"
    }
    evaluation_reporter = RecordingReporter()
    result = evaluate_checkpoint(
        config,
        "stage2_physics",
        "simulation/tiny",
        None,
        output,
        "test",
        reporter=evaluation_reporter,
    )
    assert result.predictions.shape == (2, 2)
    assert set(result.metrics) == {"left", "right"}
    assert [(bar.total, bar.n, bar.closed) for bar in evaluation_reporter.bars] == [
        (4, 4, True),
        (2, 2, True),
        (2, 2, True),
    ]
    assert "test features" in evaluation_reporter.bars[-1].desc
    assert "normalized_mae" in result.metrics["left"]
    assert "normalized_rmse" in result.metrics["right"]

def test_stage3_fold_training_and_normalized_evaluation(tmp_path: Path) -> None:
    config = _tiny_stage3_config(tmp_path)
    bundle = prepare_training(config, "stage3", "experiment/tiny", 1)
    assert bundle.train_features.shape[0] == 8
    assert bundle.valid_features.shape[0] == 2
    output = tmp_path / "stage3_fold1"
    train_bundle(config, bundle, output)
    result = evaluate_checkpoint(config, "stage3", "experiment/tiny", 1, output, "valid")
    assert result.predictions.shape == (2, 1)
    assert "normalized_mae" in result.metrics["value"]

def test_xgboost_uses_independent_models_and_best_iteration(tmp_path: Path) -> None:
    pytest.importorskip("xgboost")
    config = _tiny_config(tmp_path, name="ecfp_xgboost", targets="left;right")
    bundle = prepare_training(config, "stage2_physics", "simulation/tiny", None)
    output = tmp_path / "xgb_run"
    reporter = RecordingReporter()
    summary = train_bundle(config, bundle, output, reporter=reporter)
    reference_summary = train_bundle(config, bundle, tmp_path / "xgb_reference")
    assert summary == reference_summary
    assert set(summary["targets"]) == {"left", "right"}
    assert len(list(output.glob("model_*.json"))) == 2
    assert len(reporter.bars) == 2
    assert all(0 < bar.n <= bar.total and bar.closed for bar in reporter.bars)
    assert all("best" in bar.postfixes[-1] for bar in reporter.bars)
    result = evaluate_checkpoint(config, "stage2_physics", "simulation/tiny", None, output, "test")
    assert result.predictions.shape == (2, 2)

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
        stage2_tasks=("simulation/one", "simulation/two"),
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

def test_dmpnn_full_requires_complete_core_and_partial_from_same_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_benchmark_config("configs/benchmarks/dmpnn.yaml")
    monkeypatch.setattr(
        sweep_module, "repository_relative", lambda value: Path(value).as_posix()
    )
    study_id = f"dmpnn-{semantic_identity(
        'benchmark.reporting-study.v1',
        {'model': config.name, 'config': sweep_module._scientific_config(config)},
    )['hash']}"

    def completed(task: str, summary: dict[str, object]) -> None:
        run = (
            tmp_path
            / "stage2_physics"
            / task.replace("/", "__")
            / "evaluate_test"
            / "attempt-001"
        )
        run.mkdir(parents=True)
        (run / "metadata.json").write_text(
            '{"status":"completed"}\n', encoding="utf-8"
        )
        (run / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

    core_tasks = config.stage2_physics.tasks[:-1]
    for index, task in enumerate(core_tasks, start=1):
        spec = resolve_task(config, "stage2_physics", task, None)
        target = spec.target_columns[0]
        comparison = comparison_identity(
            "stage2_physics",
            split="test",
            expected=(task,),
            sources={f"{task}:test": f"hash-{index}"},
            normalization={task: {"scale": float(index)}},
        )
        completed(
            task,
            {
                "targets": {target: {"normalized_mae": index / 10}},
                "reporting": {
                    "schema_version": REPORTING_SCHEMA_VERSION,
                    "contract": STAGE2_CORE_EVALUATION_CONTRACT,
                    "study_id": study_id,
                    "comparison_identity": comparison,
                },
            },
        )
    partial_comparison = comparison_identity(
        "stage2_partial_charge",
        split="test",
        expected=(PARTIAL_CHARGE_UNIT,),
        sources={"partial:test": "partial-hash"},
        normalization={PARTIAL_CHARGE_UNIT: {"scale": 1.5}},
    )
    completed(
        PARTIAL_CHARGE_TASK,
        {
            "stage2_partial_charge_benchmark": {
                "test": {
                    "status": "complete",
                    "primary": {"molecule_macro_normalized_mae": 0.4},
                }
            },
            "reporting": {
                "schema_version": REPORTING_SCHEMA_VERSION,
                "contract": STAGE2_PARTIAL_EVALUATION_CONTRACT,
                "study_id": study_id,
                "comparison_identity": partial_comparison,
            },
        },
    )
    complete = _aggregate(tmp_path, config)
    assert complete["reporting"]["benchmarks"]["stage2_core_physics"][
        "protocol"
    ]["expected_tasks"] == list(core_tasks)
    assert complete["reporting"]["benchmarks"]["stage2_partial_charge"][
        "status"
    ] == "complete"
    assert complete["reporting"]["benchmarks"]["stage2_physics_full"][
        "status"
    ] == "complete"

    partial_run = (
        tmp_path
        / "stage2_physics"
        / PARTIAL_CHARGE_TASK.replace("/", "__")
        / "evaluate_test"
        / "attempt-001"
        / "summary.json"
    )
    payload = json.loads(partial_run.read_text(encoding="utf-8"))
    payload["stage2_partial_charge_benchmark"]["test"]["status"] = "incomplete"
    partial_run.write_text(json.dumps(payload), encoding="utf-8")
    incomplete = _aggregate(tmp_path, config)
    assert incomplete["reporting"]["benchmarks"]["stage2_partial_charge"][
        "status"
    ] == "incomplete"
    assert incomplete["reporting"]["benchmarks"]["stage2_physics_full"][
        "status"
    ] == "incomplete"

# --- Reporting contract ---

TASK_TARGETS = {
    "simulation/heat_of_vaporization": ("heat",),
    "simulation/homo": ("HOMO_eV",),
    "simulation/lumo": ("LUMO_eV",),
}

def _comparison() -> dict[str, object]:
    expected = list(TASK_TARGETS)
    return comparison_identity(
        "stage2_physics",
        split="test",
        expected=expected,
        sources={"shared:train": "a", "shared:test": "b"},
        normalization={name: {"scale": 2.0} for name in expected},
    )

def _stage2_summary(model: str, display: str, offset: float) -> dict[str, object]:
    metrics = {}
    expected = list(TASK_TARGETS)
    for task, targets in TASK_TARGETS.items():
        metrics[task] = {}
        for index, target in enumerate(targets):
            value = offset + index / 10
            metrics[task][target] = {
                "count": 4,
                "mae": value * 2,
                "rmse": value * 3,
                "r2": 0.5,
                "normalized_mae": value,
                "normalized_rmse": value * 1.5,
            }
            if task in {"simulation/homo", "simulation/lumo"}:
                metrics[task][target]["role_diagnostics"] = {
                    "cation": {"count": 2, "mae": value * 1.5},
                    "anion": {"count": 2, "mae": value * 2.5},
                }
    partial_comparison = comparison_identity(
        "stage2_partial_charge",
        split="test",
        expected=[PARTIAL_CHARGE_UNIT],
        sources={"test": "p", "manifest": "m", "mapping": "a"},
        normalization={PARTIAL_CHARGE_UNIT: {"scale": 2.0, "weighting": "molecule_equal"}},
    )
    full_comparison = stage2_full_comparison_identity(
        _comparison(), partial_comparison,
        ordered_units=(*expected, PARTIAL_CHARGE_UNIT),
    )
    subsets = {
        name: {
            "molecule_count": 4 if name == "all_mapped" else 0,
            "atom_count": 8 if name == "all_mapped" else 0,
            "molecule_macro_mae": offset * 2 if name == "all_mapped" else None,
            "molecule_macro_normalized_mae": offset if name == "all_mapped" else None,
            "atom_micro_mae": offset * 2 if name == "all_mapped" else None,
            "atom_micro_rmse": offset * 3 if name == "all_mapped" else None,
            "atom_micro_r2": 0.5 if name == "all_mapped" else None,
            "atom_micro_r2_reason": None if name == "all_mapped" else "no_samples",
        }
        for name in ("all_mapped", "unique", "ambiguous", "typed", "connectivity_only")
    }
    metrics[PARTIAL_CHARGE_TASK] = {
        "target_level": "atom", "capability": "supported", "status": "complete",
        "primary": {
            "molecule_macro_mae": offset * 2,
            "molecule_macro_normalized_mae": offset,
        },
        "atom_micro": {"count": 8, "mae": offset * 2, "rmse": offset * 3, "r2": 0.5, "r2_reason": None},
        "subsets": subsets,
        "coverage": {"test_molecule_count": 4, "mapped_molecule_count": 4, "issues": []},
    }
    return {
        "split": "test",
        "checkpoint_epoch": 5,
        "tasks": metrics,
        "reporting": {
            "schema_version": 1,
            "contract": STAGE2_BENCHMARK_SUITE_CONTRACT,
            "model_id": model,
            "model_display_name": display,
            "study_id": f"{model}-study",
            "capabilities": {
                "stage2_core_physics": "supported",
                "stage2_partial_charge": "supported",
                "stage2_physics_full": "supported",
            },
            "benchmarks": {
                "stage2_core_physics": {
                    "status": "complete", "benchmark": "stage2_physics",
                    "protocol": {"split": "test", "expected_tasks": list(TASK_TARGETS), "checkpoint_epoch": 5, "checkpoint_sha256": "checkpoint"},
                    "comparison_identity": _comparison(),
                },
                "stage2_partial_charge": {
                    "status": "complete", "benchmark": "stage2_partial_charge",
                    "protocol": {"split": "test", "expected_tasks": [PARTIAL_CHARGE_TASK], "expected_units": [PARTIAL_CHARGE_UNIT], "checkpoint_epoch": 5, "checkpoint_sha256": "checkpoint"},
                    "comparison_identity": partial_comparison,
                },
                "stage2_physics_full": {
                    "status": "complete", "benchmark": "stage2_physics_full",
                    "protocol": {"split": "test", "ordered_units": [*expected, PARTIAL_CHARGE_UNIT], "checkpoint_epoch": 5, "checkpoint_sha256": "checkpoint"},
                    "comparison_identity": full_comparison,
                },
            },
            "predictions": [],
        },
    }

def _write_run(
    root: Path, summary: dict[str, object], *, stage: str = "stage2"
) -> None:
    root.mkdir(parents=True)
    metadata = {
        "schema_version": 1,
        "stage": stage,
        "operation": "evaluate",
        "status": "completed",
        "semantic_identity": semantic_identity(
            "test.reporting-run", {"root": root.name}
        ),
        "provenance": {"reporting_schema_version": 1},
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

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

def test_stage2_suite_v1_is_health_only_after_breaking_contract(tmp_path: Path) -> None:
    inputs = tmp_path / "outputs"
    legacy = _stage2_summary("ilume", "ILUME", 0.3)
    legacy["reporting"]["contract"] = "stage2-benchmark-suite-v1"
    _write_run(inputs / "legacy", legacy)

    payload = publish_summary(inputs, tmp_path / "summary", tmp_path)
    assert payload["leaderboards"]["stage2_core_physics"] == []
    health = {row["source_run"]: row for row in payload["health"]}
    assert health["outputs/legacy"]["stage2_core_eligibility"] == "legacy"
    assert "legacy_stage2_reporting_contract" in health["outputs/legacy"]["issues"]

# --- D-MPNN runtime smoke ---

def _task(component_count: int, *, atom: bool = False) -> BenchmarkTask:
    return BenchmarkTask(
        benchmark="stage2_physics",
        task_id="simulation/partial_atomic_charge" if atom else "simulation/tiny",
        slots=tuple(f"component_{index}" for index in range(component_count)),
        condition_columns=() if atom else ("temperature_K", "pressure_kPa"),
        target_columns=("partial_charge",) if atom else ("value",),
        audit_columns=(),
        train_paths=(Path("train.csv"),),
        valid_paths=(Path("valid.csv"),),
        test_path=Path("test.csv"),
        fold=None,
        meta_group=None,
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
            "early_stopping_patience": 1,
            "warmup_epochs": 0,
        },
    )
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
        _predict(first, dataset, atom=False),
        _predict(second, dataset, atom=False),
        rtol=0,
        atol=0,
    )
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["best_valid_raw_mae"] == pytest.approx(
        checkpoint["best_valid_normalized_mae"]
        * checkpoint["target_statistics"]["scale"][0]
    )
