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
from benchmarks.common.features import (
    FeatureCache,
    FeaturePreprocessor,
    component_feature,
    feature_schema,
    raw_feature_matrix,
)
from benchmarks.common.metrics import mean_sample_std, regression_metrics
import scripts.benchmarks.sweep as sweep_module
from scripts.benchmarks.sweep import (
    _JobResult,
    _SweepState,
    _build_jobs,
    _parse_devices,
    _run as run_sweep_job,
    _schedule,
    _subprocess_env,
)


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
        "simulation/pbe_tzvp_cation_orbitals",
        "simulation/pbe_tzvp_anion_orbitals",
    )
    solvation = resolve_task(config, "stage3", "experiment/solvation", 1)
    organic = resolve_task(config, "stage3", "experiment/transfer_organic", 1)
    orbital = resolve_task(config, "stage2_physics", "simulation/pbe_tzvp_cation_orbitals", None)
    assert solvation.slots == ("cation", "anion", "solute")
    assert organic.slots == ("solute", "solvent")
    assert orbital.slots == ("cation",) and orbital.target_columns == ("HOMO_eV", "LUMO_eV")
    missing_test = resolve_task(config, "stage3", "experiment/self_diffusion_coefficient", 1)
    empty = load_split(missing_test, "test")
    reporter = RecordingReporter()
    with FeatureCache(tmp_path / "empty-cache.sqlite3") as cache:
        matrix = raw_feature_matrix(
            empty, feature_schema(config.features), cache, reporter=reporter
        )
    assert matrix.shape == (0, 2 * 217 + 2)
    assert reporter.bars == []


def test_preprocessor_uses_train_mask_median_and_population_zscore() -> None:
    train = np.asarray([[1.0, np.nan, np.nan, 4.0], [3.0, 7.0, np.inf, 4.0]])
    preprocessor = FeaturePreprocessor.fit(train)
    assert preprocessor.finite_mask == (True, True, False, True)
    transformed = preprocessor.transform(np.asarray([[np.nan, 9.0, 123.0, 4.0]]))
    assert transformed.shape == (1, 3)
    assert transformed[0].tolist() == pytest.approx([0.0, 2.0, 0.0])


def test_feature_cache_is_content_addressed(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    schema = feature_schema(config.features)
    with FeatureCache(config.data.feature_cache) as cache:
        first = component_feature("CCO", schema, cache)
        second = component_feature("CCO", schema, cache)
        assert np.array_equal(first, second, equal_nan=True)
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


def test_metrics_use_sample_std_and_constant_target_reason() -> None:
    assert mean_sample_std([1.0, 2.0, 3.0]) == pytest.approx({"mean": 2.0, "std": 1.0, "count": 3})
    metrics = regression_metrics(np.asarray([1.0, 2.0]), np.asarray([4.0, 4.0]), scale=2.0)
    assert np.isnan(metrics["r2"])
    assert metrics["r2_reason"] == "constant_target"
    assert metrics["normalized_mae"] == pytest.approx(1.25)


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


def test_sweep_training_progress_counts_success_and_completed_skip(
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(
        prefix="pytest-benchmark-progress-", dir=repository_root / "outputs"
    ) as temporary:
        working = Path(temporary)
        rows: list[dict[str, object]] = []
        status_path = working / "status.tsv"
        reporter = RecordingReporter()
        progress = reporter.bar(total=2, desc="sweep", unit="train-job")
        state = _SweepState(
            rows=rows,
            status_path=status_path,
            progress=progress,
            model_name="mlp",
        )
        common = {
            "state": state,
            "operation": "train",
            "benchmark": "stage2_physics",
            "task": "simulation/tiny",
            "fold": None,
            "required": "checkpoint.json",
            "training_job": True,
        }
        success = run_sweep_job(
            **common,
            root=working / "success",
            command=[sys.executable, "-c", "pass"],
        )
        assert success is not None
        completed = working / "completed/attempt-001"
        completed.mkdir(parents=True)
        (completed / "checkpoint.json").write_text("{}", encoding="utf-8")
        (completed / "metadata.json").write_text(
            '{"status": "completed"}\n', encoding="utf-8"
        )
        skipped = run_sweep_job(
            **common,
            root=working / "completed",
            command=[sys.executable, "-c", "raise AssertionError('must not run')"],
        )
        assert skipped == completed
        assert progress.n == 2
        assert state.progress_counts == {"done": 2, "failed": 0}
        assert progress.postfixes[-1] == state.progress_counts
        assert [row["status"] for row in rows] == ["OK", "SKIPPED_COMPLETED"]


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


def test_sweep_scheduler_preserves_serial_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs, ensembles = _build_jobs(
        root=tmp_path,
        stage3_tasks=("experiment/one", "experiment/two"),
        folds=(1, 2),
        stage2_tasks=("simulation/one",),
        devices=(),
    )
    order = []

    def execute(job, **_kwargs):
        order.append(job.key)
        return _JobResult(job, True)

    monkeypatch.setattr(sweep_module, "_execute_job", execute)
    _schedule(
        jobs=jobs,
        ensembles=ensembles,
        folds=(1, 2),
        max_workers=1,
        state=_SweepState(rows=[], status_path=tmp_path / "status.tsv"),
        config_path="unused.yaml",
        root=tmp_path,
        train_script=tmp_path / "train.py",
        evaluate_script=tmp_path / "evaluate.py",
    )
    assert order == [
        ("stage3_fold", "experiment/one", 1),
        ("stage3_fold", "experiment/one", 2),
        ("stage3_ensemble", "experiment/one", None),
        ("stage3_fold", "experiment/two", 1),
        ("stage3_fold", "experiment/two", 2),
        ("stage3_ensemble", "experiment/two", None),
        ("stage2_task", "simulation/one", None),
    ]


def test_sweep_devices_are_assigned_round_robin(tmp_path: Path) -> None:
    assert _parse_devices("cuda:0,cuda:2") == ("cuda:0", "cuda:2")
    jobs, ensembles = _build_jobs(
        root=tmp_path,
        stage3_tasks=("experiment/one",),
        folds=(1, 2),
        stage2_tasks=("simulation/one",),
        devices=("cuda:0", "cuda:1"),
    )
    assert [job.device for job in jobs] == ["cuda:0", "cuda:1", "cuda:1"]
    assert ensembles["experiment/one"].device == "cuda:0"
    assert _subprocess_env(None)["ILUME_DISABLE_PROGRESS"] == "1"
    assert _subprocess_env("cuda:3")["CUDA_VISIBLE_DEVICES"] == "3"
