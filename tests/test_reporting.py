from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmarks.common.summary import SUMMARY_FILES, publish_summary
from common.identity import semantic_identity
from common.reporting import (
    comparison_identity,
    reporting_block,
    write_prediction_csv,
)
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
    return {
        "split": "test",
        "checkpoint_epoch": 5,
        "tasks": metrics,
        "reporting": reporting_block(
            model_id=model,
            model_display_name=display,
            benchmark="stage2_physics",
            protocol={
                "split": "test",
                "ensemble": False,
                "expected_tasks": list(TASK_TARGETS),
                "expected_targets": expected,
                "checkpoint_epoch": 5,
            },
            comparison=_comparison(),
            study_id=f"{model}-study",
        ),
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
    assert [row["model"] for row in first["leaderboards"]["stage2_physics"]] == [
        "ILUME", "MLP"
    ]
    assert first["leaderboards"]["stage2_physics"][0]["per_target_wins"] == 5
    assert len(first["comparison_identities"]["stage2_physics"]) == 1
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
    assert payload["leaderboards"]["stage2_physics"] == []
    health = {row["source_run"]: row for row in payload["health"]}
    assert health["outputs/incomplete"]["completeness"] == "incomplete"
    assert "missing_targets=" in health["outputs/incomplete"]["issues"]
    assert health["outputs/failed"]["completeness"] == "failed"
    assert "malformed_summary" in health["outputs/failed"]["issues"]
