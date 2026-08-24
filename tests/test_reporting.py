from __future__ import annotations

import csv
import copy
import json
from pathlib import Path

import pytest

from benchmarks.common.summary import SUMMARY_FILES, publish_summary
from common.identity import semantic_identity
from common.reporting import (
    STAGE2_BENCHMARK_SUITE_CONTRACT,
    comparison_identity,
    stage2_full_comparison_identity,
    write_prediction_csv,
)
from stage2.atom_evaluation import PARTIAL_CHARGE_TASK, PARTIAL_CHARGE_UNIT
from stage2.evaluate import resolve_checkpoint_path


TASK_TARGETS = {
    "simulation/heat_of_vaporization": ("heat",),
    "simulation/pbe_tzvp_cation_orbitals": ("HOMO", "LUMO"),
    "simulation/pbe_tzvp_anion_orbitals": ("HOMO", "LUMO"),
}


def _comparison() -> dict[str, object]:
    expected = [
        f"{task}::{target}"
        for task, targets in TASK_TARGETS.items()
        for target in targets
    ]
    return comparison_identity(
        "stage2_physics",
        split="test",
        expected=expected,
        sources={"shared:train": "a", "shared:test": "b"},
        normalization={name: {"scale": 2.0} for name in expected},
    )


def _stage2_summary(model: str, display: str, offset: float) -> dict[str, object]:
    metrics = {}
    expected = []
    for task, targets in TASK_TARGETS.items():
        metrics[task] = {}
        for index, target in enumerate(targets):
            expected.append(f"{task}::{target}")
            value = offset + index / 10
            metrics[task][target] = {
                "count": 4,
                "mae": value * 2,
                "rmse": value * 3,
                "r2": 0.5,
                "normalized_mae": value,
                "normalized_rmse": value * 1.5,
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
                    "protocol": {"split": "test", "expected_tasks": list(TASK_TARGETS), "expected_targets": expected, "checkpoint_epoch": 5, "checkpoint_sha256": "checkpoint"},
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


def _write_run(root: Path, summary: dict[str, object]) -> None:
    root.mkdir(parents=True)
    metadata = {
        "schema_version": 1,
        "stage": "stage2",
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
    with pytest.raises(ValueError, match="Non-finite"):
        write_prediction_csv(
            path,
            [{"source_row": 2, "target": float("nan")}],
            ("source_row", "target"),
        )
    assert not path.with_suffix(".csv.tmp").exists()


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
    assert first["leaderboards"]["stage2_core_physics"][0]["per_target_wins"] == 5
    assert [row["model"] for row in first["leaderboards"]["stage2_partial_charge"]] == ["ILUME", "MLP"]
    assert [row["model"] for row in first["leaderboards"]["stage2_physics_full"]] == ["ILUME", "MLP"]
    assert len(first["comparison_identities"]["stage2_core_physics"]) == 1
    assert tuple(sorted(path.name for path in output.iterdir())) == tuple(
        sorted(SUMMARY_FILES)
    )
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    publish_summary(inputs, output, tmp_path)
    after = {path.name: path.read_bytes() for path in output.iterdir()}
    assert before == after


def test_malformed_current_summary_preserves_existing_snapshot(tmp_path: Path) -> None:
    inputs = tmp_path / "outputs"
    broken = inputs / "broken"
    broken.mkdir(parents=True)
    metadata = {
        "schema_version": 1,
        "stage": "stage2",
        "operation": "evaluate",
        "status": "completed",
        "semantic_identity": semantic_identity("test.reporting-run", {"run": 1}),
        "provenance": {"reporting_schema_version": 1},
    }
    (broken / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (broken / "summary.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "summary"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("old", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot publish summary"):
        publish_summary(inputs, output, tmp_path)
    assert marker.read_text(encoding="utf-8") == "old"


def test_incomplete_and_malformed_noncompleted_runs_remain_in_health(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "outputs"
    incomplete = _stage2_summary("ilume", "ILUME", 0.2)
    del incomplete["tasks"]["simulation/pbe_tzvp_anion_orbitals"]["LUMO"]
    _write_run(inputs / "incomplete", incomplete)

    failed = inputs / "failed"
    failed.mkdir(parents=True)
    (failed / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "stage2",
                "operation": "evaluate",
                "status": "failed",
                "semantic_identity": semantic_identity(
                    "test.reporting-run", {"run": "failed"}
                ),
                "provenance": {"reporting_schema_version": 1},
            }
        ),
        encoding="utf-8",
    )
    (failed / "summary.json").write_text("not-json", encoding="utf-8")

    payload = publish_summary(inputs, tmp_path / "summary", tmp_path)
    assert payload["leaderboards"]["stage2_core_physics"] == []
    health = {row["source_run"]: row for row in payload["health"]}
    assert health["outputs/incomplete"]["completeness"] == "incomplete"
    assert "missing_targets=" in health["outputs/incomplete"]["issues"]
    assert health["outputs/failed"]["completeness"] == "failed"
    assert "malformed_summary" in health["outputs/failed"]["issues"]


def test_stage2_capability_eligibility_and_legacy_migration(tmp_path: Path) -> None:
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

    incomplete = _stage2_summary("ilume", "ILUME", 0.2)
    incomplete["reporting"]["benchmarks"]["stage2_partial_charge"]["status"] = "incomplete"
    incomplete["reporting"]["benchmarks"]["stage2_partial_charge"]["issues"] = ["missing_predictions=1"]
    incomplete["reporting"]["benchmarks"]["stage2_physics_full"]["status"] = "incomplete"
    incomplete["reporting"]["benchmarks"]["stage2_physics_full"]["issues"] = ["partial_charge_incomplete"]
    incomplete["tasks"][PARTIAL_CHARGE_TASK]["status"] = "incomplete"
    incomplete["tasks"][PARTIAL_CHARGE_TASK]["primary"] = None
    _write_run(inputs / "incomplete_partial", incomplete)

    partial_only = _stage2_summary("partial", "PartialOnly", 0.15)
    partial_only["reporting"]["benchmarks"]["stage2_core_physics"]["status"] = "incomplete"
    partial_only["reporting"]["benchmarks"]["stage2_core_physics"]["issues"] = ["missing_targets=1"]
    partial_only["reporting"]["benchmarks"]["stage2_physics_full"]["status"] = "incomplete"
    partial_only["reporting"]["benchmarks"]["stage2_physics_full"]["issues"] = ["core_incomplete"]
    del partial_only["tasks"]["simulation/heat_of_vaporization"]["heat"]
    _write_run(inputs / "partial_only", partial_only)

    legacy = copy.deepcopy(_stage2_summary("old", "Old", 0.1))
    del legacy["reporting"]["contract"]
    _write_run(inputs / "legacy", legacy)

    payload = publish_summary(inputs, tmp_path / "summary", tmp_path)
    assert [row["model"] for row in payload["leaderboards"]["stage2_core_physics"]] == [
        "ILUME", "MLP"
    ]
    assert [row["model"] for row in payload["leaderboards"]["stage2_partial_charge"]] == [
        "PartialOnly"
    ]
    assert payload["leaderboards"]["stage2_physics_full"] == []
    health = {row["source_run"]: row for row in payload["health"]}
    assert health["outputs/unsupported"]["stage2_partial_eligibility"] == "not_evaluated"
    assert health["outputs/unsupported"]["issues"] == ""
    assert health["outputs/incomplete_partial"]["stage2_partial_eligibility"] == "not_eligible"
    assert "missing_predictions=1" in health["outputs/incomplete_partial"]["issues"]
    assert health["outputs/legacy"]["stage2_core_eligibility"] == "legacy"
    assert "legacy_stage2_reporting_contract" in health["outputs/legacy"]["issues"]
