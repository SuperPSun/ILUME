from __future__ import annotations

import csv
import subprocess
import sqlite3
import sys
import tempfile
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


CATALOG_FIELDS = (
    "catalog_schema_version", "stage", "task_id", "task_kind", "target_level",
    "source_file", "target_columns", "identity_columns", "condition_columns",
    "system_type", "simulation_method", "materialized_path", "label_source",
    "resource_manifest", "strategies",
)


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
    with FeatureCache(tmp_path / "empty-cache.sqlite3") as cache:
        matrix = raw_feature_matrix(empty, feature_schema(config.features), cache)
    assert matrix.shape == (0, 2 * 217 + 2)


def test_preprocessor_uses_train_mask_median_and_population_zscore() -> None:
    train = np.asarray([[1.0, np.nan, np.nan, 4.0], [3.0, 7.0, np.inf, 4.0]])
    preprocessor = FeaturePreprocessor.fit(train)
    assert preprocessor.finite_mask == (True, True, False, True)
    transformed = preprocessor.transform(np.asarray([[np.nan, 9.0, 123.0, 4.0]]))
    assert transformed.shape == (1, 3)
    assert transformed[0].tolist() == pytest.approx([0.0, 2.0, 0.0])


def test_feature_cache_is_content_addressed_and_detects_corruption(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    schema = feature_schema(config.features)
    with FeatureCache(config.data.feature_cache) as cache:
        first = component_feature("CCO", schema, cache)
        second = component_feature("CCO", schema, cache)
        assert np.array_equal(first, second, equal_nan=True)
    with sqlite3.connect(config.data.feature_cache) as connection:
        connection.execute("UPDATE features SET payload = ?", (b"broken",))
        connection.commit()
    with FeatureCache(config.data.feature_cache) as cache:
        with pytest.raises(ValueError, match="Corrupt benchmark feature cache"):
            component_feature("CCO", schema, cache)


def test_training_prepare_does_not_open_test_and_conditions_fail_strictly(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    test_path = tmp_path / "stage2/tiny/test.csv"
    test_path.unlink()
    bundle = prepare_training(config, "stage2_physics", "simulation/tiny", None)
    assert bundle.train_features.shape[0] == 4
    with pytest.raises(FileNotFoundError, match="Missing benchmark source"):
        load_split(bundle.task, "test")
    valid_path = tmp_path / "stage2/tiny/valid.csv"
    rows = list(csv.DictReader(valid_path.open(encoding="utf-8")))
    rows[0]["temperature_K"] = "nan"
    _write_csv(valid_path, list(rows[0]), rows)
    with pytest.raises(ValueError, match="Missing Stage 3 value"):
        prepare_training(config, "stage2_physics", "simulation/tiny", None)


def test_mlp_train_checkpoint_and_test_evaluation(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path, targets="left;right")
    bundle = prepare_training(config, "stage2_physics", "simulation/tiny", None)
    output = tmp_path / "mlp_run"
    summary = train_bundle(config, bundle, output)
    assert 1 <= summary["best_epoch"] <= 4
    result = evaluate_checkpoint(config, "stage2_physics", "simulation/tiny", None, output, "test")
    assert result.predictions.shape == (2, 2)
    assert set(result.metrics) == {"left", "right"}
    assert "normalized_mae" not in result.metrics["left"]


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
    summary = train_bundle(config, bundle, output)
    assert set(summary["targets"]) == {"left", "right"}
    assert len(list(output.glob("model_*.json"))) == 2
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


def test_sweep_preserves_preinitialization_failures_as_new_attempts() -> None:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="pytest-benchmark-", dir=root / "outputs") as temporary:
        working = Path(temporary)
        config = _tiny_config(working)
        train_path = working / "stage2/tiny/train.csv"
        rows = list(csv.DictReader(train_path.open(encoding="utf-8")))
        rows[0]["temperature_K"] = "nan"
        _write_csv(train_path, list(rows[0]), rows)
        payload = config.to_dict()
        relative = working.relative_to(root)
        payload["data"] = {
            "data_root": relative.as_posix(),
            "task_catalog": (relative / "task_catalog.csv").as_posix(),
            "stage3_authority_config": (relative / "unused.yaml").as_posix(),
            "feature_cache": (relative / "features.sqlite3").as_posix(),
        }
        config_path = working / "config.yaml"
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        output = relative / "sweep"
        command = [
            sys.executable, "scripts/benchmarks/sweep.py",
            "--config", config_path.relative_to(root).as_posix(),
            "--output", output.as_posix(),
        ]
        assert subprocess.run(command, cwd=root, check=False).returncode == 1
        assert subprocess.run(command, cwd=root, check=False).returncode == 1
        attempts = working / "sweep/stage2_physics/simulation__tiny/train"
        assert (attempts / "attempt-001/sweep_failure.json").is_file()
        assert (attempts / "attempt-002/sweep_failure.json").is_file()
