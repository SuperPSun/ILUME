from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.common.summary import SUMMARY_FILES, publish_summary
from common.identity import semantic_identity
from common.reporting import (
    STAGE2_BENCHMARK_SUITE_CONTRACT,
    comparison_identity,
    role_mae_diagnostics,
    stage2_full_comparison_identity,
    write_prediction_csv,
)
from stage2.atom_evaluation import PARTIAL_CHARGE_TASK, PARTIAL_CHARGE_UNIT
from stage2.evaluate import resolve_checkpoint_path


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


def test_orbital_role_diagnostics_are_sample_weighted_and_non_headline() -> None:
    diagnostics = role_mae_diagnostics(
        [0.0, 2.0, 10.0],
        [1.0, 4.0, 6.0],
        ["cation", "cation", "anion"],
    )
    assert diagnostics == {
        "cation": {"count": 2, "mae": 1.5},
        "anion": {"count": 1, "mae": 4.0},
    }


def test_stage2_checkpoint_default_selects_highest_filename(tmp_path: Path) -> None:
    for epoch in (1, 5, 3):
        (tmp_path / f"checkpoint_epoch_{epoch:05d}.pt").touch()
    assert resolve_checkpoint_path(tmp_path).name == "checkpoint_epoch_00005.pt"
    assert resolve_checkpoint_path(tmp_path, 3).name == "checkpoint_epoch_00003.pt"


def test_summarizer_ranks_runs_and_republishes_deterministically(tmp_path: Path) -> None:
    inputs = tmp_path / "outputs"
    _write_run(inputs / "ilume", _stage2_summary("ilume", "ILUME", 0.2))
    _write_run(inputs / "mlp", _stage2_summary("mlp", "MLP", 0.4))
    output = tmp_path / "summary"
    first = publish_summary(inputs, output, tmp_path)
    assert [row["model"] for row in first["leaderboards"]["stage2_core_physics"]] == [
        "ILUME", "MLP"
    ]
    assert first["leaderboards"]["stage2_core_physics"][0]["per_task_wins"] == 3
    subsets = {
        row["subset"]
        for row in first["metrics"]["stage2_core_physics"]
        if row["task"] in {"simulation/homo", "simulation/lumo"}
    }
    assert subsets == {"pooled", "cation", "anion"}
    assert [row["model"] for row in first["leaderboards"]["stage2_partial_charge"]] == ["ILUME", "MLP"]
    assert [row["model"] for row in first["leaderboards"]["stage2_physics_full"]] == ["ILUME", "MLP"]
    assert len(first["comparison_identities"]["stage2_core_physics"]) == 1
    assert tuple(sorted(path.name for path in output.iterdir())) == tuple(
        sorted(SUMMARY_FILES)
    )
    for path in output.glob("*.csv"):
        assert b"\r\n" not in path.read_bytes()
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    publish_summary(inputs, output, tmp_path)
    after = {path.name: path.read_bytes() for path in output.iterdir()}
    assert before == after


def test_stage2_unsupported_capabilities_are_not_evaluated(tmp_path: Path) -> None:
    inputs = tmp_path / "outputs"
    unsupported = _stage2_summary("mlp", "MLP", 0.3)
    unsupported["reporting"]["capabilities"]["stage2_partial_charge"] = "unsupported"
    unsupported["reporting"]["capabilities"]["stage2_physics_full"] = "unsupported"
    for name in ("stage2_partial_charge", "stage2_physics_full"):
        unsupported["reporting"]["benchmarks"][name] = {
            "status": "unsupported",
            "benchmark": name,
            "protocol": {"split": "test", "ensemble": False},
        }
    _write_run(inputs / "unsupported", unsupported)

    payload = publish_summary(inputs, tmp_path / "summary", tmp_path)
    assert [row["model"] for row in payload["leaderboards"]["stage2_core_physics"]] == ["MLP"]
    assert payload["leaderboards"]["stage2_partial_charge"] == []
    assert payload["leaderboards"]["stage2_physics_full"] == []
    health = {row["source_run"]: row for row in payload["health"]}
    assert health["outputs/unsupported"]["stage2_partial_eligibility"] == "not_evaluated"
    assert health["outputs/unsupported"]["issues"] == ""


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


def test_stage2_suite_v1_benchmark_keeps_stage3_and_is_health_only(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "outputs"
    legacy = _stage2_summary("mlp", "MLP", 0.3)
    reporting = legacy["reporting"]
    reporting["contract"] = "stage2-benchmark-suite-v1"
    reporting["source_runs"] = {}
    reporting["benchmarks"]["stage3_test"] = {
        "status": "unsupported",
        "protocol": {"expected_tasks": [], "folds": [1, 2, 3, 4, 5]},
    }
    reporting["benchmarks"]["stage3_validation"] = {
        "status": "unsupported",
        "protocol": {"expected_tasks": [], "folds": [1, 2, 3, 4, 5]},
    }
    legacy["stage3_property_benchmark"] = {
        "test_ensemble": [], "validation_five_fold": []
    }
    legacy["stage2_physics_benchmark"] = {"test": {}}
    root = inputs / "legacy"
    _write_run(root, legacy, stage="benchmark")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    metadata["operation"] = "sweep"
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    payload = publish_summary(inputs, tmp_path / "summary", tmp_path)
    assert payload["leaderboards"]["stage2_core_physics"] == []
    health = {row["source_run"]: row for row in payload["health"]}
    assert health["outputs/legacy"]["stage2_core_eligibility"] == "legacy"
