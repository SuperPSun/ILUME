from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from common.identity import validate_semantic_identity
from common.io import atomic_json
from common.reporting import REPORTING_SCHEMA_VERSION


SUMMARY_SCHEMA_VERSION = 1
SUMMARY_FILES = (
    "overview.md",
    "stage3_test_leaderboard.csv",
    "stage3_validation_leaderboard.csv",
    "stage2_physics_leaderboard.csv",
    "stage3_test_metrics.csv",
    "stage3_validation_metrics.csv",
    "stage2_physics_metrics.csv",
    "sweep_status.csv",
    "summary.json",
)


@dataclass(frozen=True)
class Candidate:
    root: Path
    source_run: str
    metadata: dict[str, Any]
    summary: dict[str, Any] | None
    current: bool
    issues: tuple[str, ...] = ()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _source_run(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def discover_candidates(input_root: Path, repository_root: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    malformed: list[str] = []
    for metadata_path in sorted(input_root.rglob("metadata.json")):
        try:
            metadata = _json(metadata_path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            malformed.append(f"{metadata_path}: malformed metadata: {error}")
            continue
        key = (metadata.get("stage"), metadata.get("operation"))
        if key not in {
            ("benchmark", "sweep"),
            ("stage3", "evaluate"),
            ("stage2", "evaluate"),
        }:
            continue
        root = metadata_path.parent
        current = (
            metadata.get("provenance", {}).get("reporting_schema_version")
            == REPORTING_SCHEMA_VERSION
        )
        summary_path = root / "summary.json"
        summary: dict[str, Any] | None = None
        issues: list[str] = []
        if summary_path.is_file():
            try:
                summary = _json(summary_path)
            except (OSError, json.JSONDecodeError, ValueError) as error:
                issues.append("malformed_summary")
                if current and metadata.get("status") == "completed":
                    malformed.append(f"{root}: malformed current summary: {error}")
        elif current and metadata.get("status") == "completed":
            malformed.append(f"{root}: completed current run has no summary.json")
        candidate = Candidate(
            root=root,
            source_run=_source_run(root, repository_root),
            metadata=metadata,
            summary=summary,
            current=current,
            issues=tuple(issues),
        )
        if current and metadata.get("status") == "completed" and summary is not None:
            try:
                _validate_current(candidate)
            except (KeyError, TypeError, ValueError) as error:
                malformed.append(f"{root}: {error}")
        candidates.append(candidate)
    if malformed:
        raise ValueError(
            "Cannot publish summary because formal reporting inputs are malformed:\n- "
            + "\n- ".join(malformed)
        )
    return candidates


def _validate_comparison(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} lacks comparison identity")
    validate_semantic_identity(value)
    if value.get("type") != "reporting.comparison.v1":
        raise ValueError(f"{context} has unsupported comparison identity")


def _validate_current(candidate: Candidate) -> None:
    assert candidate.summary is not None
    stage = candidate.metadata["stage"]
    summary = candidate.summary
    reporting = summary.get("reporting")
    if not isinstance(reporting, dict) or reporting.get("schema_version") != 1:
        raise ValueError("completed reporting run has no schema-v1 reporting block")
    if not reporting.get("model_id") or not reporting.get("model_display_name"):
        raise ValueError("reporting model identity is incomplete")
    if not reporting.get("study_id"):
        raise ValueError("reporting study identity is incomplete")
    if stage == "benchmark":
        benchmarks = reporting.get("benchmarks")
        if set(benchmarks or ()) != {
            "stage3_test", "stage3_validation", "stage2_physics"
        }:
            raise ValueError("benchmark sweep reporting sections are incomplete")
        for name, value in benchmarks.items():
            _validate_comparison(value.get("comparison_identity"), name)
        if not isinstance(reporting.get("source_runs"), dict):
            raise ValueError("benchmark sweep reporting lacks source runs")
    else:
        _validate_comparison(
            reporting.get("comparison_identity"), f"{stage} evaluation"
        )
        protocol = reporting.get("protocol")
        if not isinstance(protocol, dict) or protocol.get("split") not in {
            "valid", "test"
        }:
            raise ValueError(f"{stage} evaluation protocol is malformed")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _run_id(model: str, source_run: str) -> str:
    return f"{model}@{source_run}"


def _model(candidate: Candidate) -> tuple[str, str]:
    if candidate.summary:
        reporting = candidate.summary.get("reporting", {})
        if reporting.get("model_id") and reporting.get("model_display_name"):
            return str(reporting["model_id"]), str(reporting["model_display_name"])
        if candidate.metadata.get("stage") == "benchmark" and candidate.summary.get("model"):
            model = str(candidate.summary["model"])
            return model, model
    return ("ilume", "ILUME") if candidate.metadata.get("stage") != "benchmark" else ("unknown", "unknown")


def _health(candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    signatures: list[tuple[str, str, str, str, str]] = []
    for candidate in candidates:
        model, display = _model(candidate)
        provenance = candidate.metadata.get("provenance", {})
        signatures.append(
            (
                model,
                str(candidate.metadata.get("stage", "")),
                str(candidate.metadata.get("operation", "")),
                str((candidate.summary or {}).get("split", provenance.get("split", ""))),
                str((candidate.summary or {}).get("reporting", {}).get("protocol", {}).get("fold", provenance.get("fold", ""))),
            )
        )
        status = str(candidate.metadata.get("status", "unknown"))
        summary = candidate.summary or {}
        issues = list(candidate.issues)
        if not candidate.current:
            completeness = "legacy"
            issues.append("reporting_schema_missing")
        elif status != "completed":
            completeness = status
        else:
            completeness = "complete"
        expected: Any = ""
        available: Any = ""
        folds: Any = ""
        checkpoint = summary.get("checkpoint_epoch", "")
        failed_jobs = summary.get("jobs", {}).get("failed", "")
        if candidate.current and candidate.summary:
            reporting = summary["reporting"]
            if candidate.metadata["stage"] == "benchmark":
                sections = reporting["benchmarks"]
                expected = ";".join(
                    f"{name}:{len(value['protocol'].get('expected_tasks', ())) }"
                    for name, value in sorted(sections.items())
                )
                available = ";".join(
                    (
                        f"stage3_test:{len(summary['stage3_property_benchmark']['test_ensemble'])}",
                        f"stage3_validation:{len(summary['stage3_property_benchmark']['validation_five_fold'])}",
                        f"stage2_physics:{len(summary['stage2_physics_benchmark']['test'])}",
                    )
                )
                folds = ";".join(map(str, sections["stage3_validation"]["protocol"]["folds"]))
            else:
                protocol = reporting["protocol"]
                expected_tasks = protocol.get("expected_tasks", ())
                expected = len(expected_tasks)
                if candidate.metadata["stage"] == "stage3":
                    metrics = (
                        summary.get("ensemble", {}).get("tasks", {})
                        if summary.get("split") == "test"
                        else summary.get("tasks", {})
                    )
                    available = sum(
                        int(values.get("count", 0)) > 0 for values in metrics.values()
                    )
                    fold = protocol.get("fold")
                    folds = ";".join(map(str, protocol.get("folds", ()))) if fold is None else str(fold)
                else:
                    available = len(summary.get("tasks", {}))
        rows.append(
            {
                "run": _run_id(model, candidate.source_run),
                "model": display,
                "stage": candidate.metadata.get("stage", ""),
                "operation": candidate.metadata.get("operation", ""),
                "status": status,
                "completeness": completeness,
                "checkpoint_epoch": checkpoint,
                "enabled_tasks": (
                    len(summary.get("reporting", {}).get("protocol", {}).get("enabled_tasks", ()))
                    if candidate.metadata.get("stage") == "stage3" else ""
                ),
                "expected_tasks": expected,
                "available_tasks": available,
                "available_folds": folds,
                "failed_jobs": failed_jobs,
                "source_run": candidate.source_run,
                "issues": ";".join(issues),
            }
        )
    rows_by_source = {row["source_run"]: row for row in rows}

    def mark_incomplete(candidate: Candidate, issue: str) -> None:
        row = rows_by_source[candidate.source_run]
        row["completeness"] = "incomplete"
        row["issues"] = ";".join(
            item for item in (row["issues"], issue) if item
        )

    validation_groups: dict[tuple[str, str], dict[int, list[Candidate]]] = {}
    for candidate in _current_completed(candidates):
        summary = candidate.summary or {}
        reporting = summary["reporting"]
        if candidate.metadata["stage"] == "benchmark":
            sections = reporting["benchmarks"]
            test_expected = tuple(
                sections["stage3_test"]["protocol"]["expected_tasks"]
            )
            test_metrics = summary["stage3_property_benchmark"]["test_ensemble"]
            test_missing = [
                task for task in test_expected
                if task not in test_metrics
                or int(test_metrics[task].get("count", 0)) <= 0
            ]
            if test_missing:
                mark_incomplete(
                    candidate, "stage3_test_missing_tasks=" + ",".join(test_missing)
                )
            test_folds = tuple(
                sections["stage3_test"]["protocol"].get("folds", ())
            )
            if test_folds != (1, 2, 3, 4, 5):
                mark_incomplete(
                    candidate,
                    "stage3_test_missing_folds="
                    + ",".join(map(str, sorted(set(range(1, 6)) - set(test_folds)))),
                )

            valid_expected = tuple(
                sections["stage3_validation"]["protocol"]["expected_tasks"]
            )
            valid_metrics = summary["stage3_property_benchmark"][
                "validation_five_fold"
            ]
            valid_missing = [
                task for task in valid_expected
                if task not in valid_metrics
                or int(valid_metrics[task].get("normalized_mae", {}).get("count", 0))
                != 5
            ]
            if valid_missing:
                mark_incomplete(
                    candidate,
                    "stage3_validation_missing_tasks=" + ",".join(valid_missing),
                )

            stage2_expected = tuple(
                sections["stage2_physics"]["protocol"]["expected_targets"]
            )
            stage2_metrics = summary["stage2_physics_benchmark"]["test"]
            stage2_available = {
                f"{task}::{target}"
                for task, targets in stage2_metrics.items()
                for target, values in targets.items()
                if int(values.get("count", 0)) > 0
            }
            stage2_missing = sorted(set(stage2_expected) - stage2_available)
            if stage2_missing:
                mark_incomplete(
                    candidate,
                    "stage2_missing_targets=" + ",".join(stage2_missing),
                )
            try:
                if int(summary.get("jobs", {}).get("failed", 0)) > 0:
                    mark_incomplete(candidate, "failed_jobs")
            except (TypeError, ValueError):
                mark_incomplete(candidate, "failed_jobs_unknown")
        elif candidate.metadata["stage"] == "stage3":
            protocol = reporting["protocol"]
            expected_tasks = tuple(protocol["expected_tasks"])
            metrics = (
                summary.get("ensemble", {}).get("tasks", {})
                if summary.get("split") == "test"
                else summary.get("tasks", {})
            )
            missing = [
                task for task in expected_tasks
                if task not in metrics or int(metrics[task].get("count", 0)) <= 0
            ]
            if missing:
                mark_incomplete(candidate, "missing_tasks=" + ",".join(missing))
            if summary.get("split") == "test":
                folds = tuple(protocol.get("folds", ()))
                if folds != (1, 2, 3, 4, 5):
                    mark_incomplete(
                        candidate,
                        "missing_folds="
                        + ",".join(map(str, sorted(set(range(1, 6)) - set(folds)))),
                    )
            else:
                key = (str(reporting["model_id"]), str(reporting["study_id"]))
                fold = int(protocol["fold"])
                validation_groups.setdefault(key, {}).setdefault(fold, []).append(
                    candidate
                )
        elif candidate.metadata["stage"] == "stage2":
            expected_targets = tuple(reporting["protocol"]["expected_targets"])
            available_targets = {
                f"{task}::{target}"
                for task, targets in summary.get("tasks", {}).items()
                for target, values in targets.items()
                if int(values.get("count", 0)) > 0
            }
            missing = sorted(set(expected_targets) - available_targets)
            if missing:
                mark_incomplete(
                    candidate, "missing_targets=" + ",".join(missing)
                )

    for folds in validation_groups.values():
        missing = sorted(set(range(1, 6)) - set(folds))
        duplicates = sorted(fold for fold, items in folds.items() if len(items) > 1)
        unique_items = [items[0] for items in folds.values() if len(items) == 1]
        comparison_hashes = {
            candidate.summary["reporting"]["comparison_identity"]["hash"]
            for candidate in unique_items
        }
        expected_sets = {
            tuple(candidate.summary["reporting"]["protocol"]["expected_tasks"])
            for candidate in unique_items
        }
        checkpoints = {
            candidate.summary.get("checkpoint_epoch") for candidate in unique_items
        }
        for items in folds.values():
            for candidate in items:
                if missing:
                    mark_incomplete(
                        candidate, "missing_folds=" + ",".join(map(str, missing))
                    )
                if duplicates:
                    mark_incomplete(
                        candidate,
                        "duplicate_folds=" + ",".join(map(str, duplicates)),
                    )
                if len(comparison_hashes) > 1:
                    mark_incomplete(candidate, "comparison_identity_mismatch")
                if len(expected_sets) > 1:
                    mark_incomplete(candidate, "expected_tasks_mismatch")
                if len(checkpoints) > 1:
                    mark_incomplete(candidate, "checkpoint_epoch_mismatch")
    signature_counts = {
        signature: signatures.count(signature) for signature in set(signatures)
    }
    for row, signature in zip(rows, signatures, strict=True):
        if signature_counts[signature] > 1:
            row["issues"] = ";".join(
                item for item in (row["issues"], "alternative_run") if item
            )
    return sorted(rows, key=lambda row: row["run"])


def _current_completed(candidates: Sequence[Candidate]) -> list[Candidate]:
    return [
        candidate for candidate in candidates
        if candidate.current
        and candidate.metadata.get("status") == "completed"
        and candidate.summary is not None
    ]


def _stage3_test(
    candidates: Sequence[Candidate],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    leaders: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    comparisons: dict[str, list[str]] = {}
    for candidate in _current_completed(candidates):
        summary = candidate.summary or {}
        reporting = summary["reporting"]
        if candidate.metadata["stage"] == "benchmark":
            section = reporting["benchmarks"]["stage3_test"]
            metrics = summary["stage3_property_benchmark"]["test_ensemble"]
            checkpoint = ""
        elif candidate.metadata["stage"] == "stage3" and summary.get("split") == "test":
            section = reporting
            metrics = summary["ensemble"]["tasks"]
            checkpoint = summary["checkpoint_epoch"]
        else:
            continue
        expected = tuple(section["protocol"]["expected_tasks"])
        complete = bool(expected) and all(
            task in metrics
            and int(metrics[task].get("count", 0)) > 0
            and _finite(metrics[task].get("normalized_mae"))
            for task in expected
        )
        if not complete or not section["protocol"].get("ensemble") or tuple(
            section["protocol"].get("folds", ())
        ) != (1, 2, 3, 4, 5):
            continue
        model_id = reporting["model_id"]
        display = reporting["model_display_name"]
        run = _run_id(model_id, candidate.source_run)
        values = [float(metrics[task]["normalized_mae"]) for task in expected]
        comparison_hash = section["comparison_identity"]["hash"]
        comparisons.setdefault(comparison_hash, []).append(run)
        for task in expected:
            value = metrics[task]
            metrics_rows.append(
                {
                    "run": run, "model": display, "task": task,
                    "count": value["count"], "mae": value["mae"],
                    "rmse": value["rmse"], "r2": value["r2"],
                    "normalized_mae": value["normalized_mae"],
                    "normalized_rmse": value["normalized_rmse"],
                    "source_run": candidate.source_run,
                }
            )
        leaders.append(
            {
                "run": run, "model": display,
                "macro_normalized_mae": sum(values) / len(values),
                "valid_tasks": len(values), "total_tasks": len(expected),
                "per_task_wins": 0, "source_run": candidate.source_run,
                "checkpoint_epoch": checkpoint,
                "enabled_tasks": len(
                    section["protocol"].get("enabled_tasks", expected)
                ),
            }
        )
    _require_one_comparison(comparisons, "Stage 3 test")
    wins = _wins(metrics_rows, "task", "normalized_mae")
    for row in leaders:
        row["per_task_wins"] = len(wins.get(row["run"], ()))
    return _rank(leaders, "macro_normalized_mae"), sorted(
        metrics_rows, key=lambda row: (row["run"], row["task"])
    ), wins


def _stage3_validation(
    candidates: Sequence[Candidate],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    leaders: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    comparisons: dict[str, list[str]] = {}
    fold_groups: dict[tuple[str, str], dict[int, Candidate]] = {}
    ambiguous: set[tuple[str, str]] = set()
    for candidate in _current_completed(candidates):
        if candidate.metadata["stage"] != "stage3" or candidate.summary.get("split") != "valid":
            continue
        reporting = candidate.summary["reporting"]
        key = (str(reporting["model_id"]), str(reporting["study_id"]))
        fold = int(reporting["protocol"]["fold"])
        if fold in fold_groups.setdefault(key, {}):
            ambiguous.add(key)
        fold_groups[key][fold] = candidate
    for (model_id, study_id), folds in sorted(fold_groups.items()):
        if (model_id, study_id) in ambiguous or set(folds) != set(range(1, 6)):
            continue
        items = [folds[fold] for fold in range(1, 6)]
        first_reporting = items[0].summary["reporting"]
        expected = tuple(first_reporting["protocol"]["expected_tasks"])
        if any(
            tuple(item.summary["reporting"]["protocol"]["expected_tasks"]) != expected
            or item.summary["reporting"]["comparison_identity"]["hash"]
            != first_reporting["comparison_identity"]["hash"]
            for item in items[1:]
        ):
            continue
        run = f"{model_id}@study:{study_id}"
        display = first_reporting["model_display_name"]
        source = ";".join(item.source_run for item in items)
        checkpoints = {item.summary["checkpoint_epoch"] for item in items}
        if len(checkpoints) != 1:
            continue
        complete = True
        task_values: list[float] = []
        for task in expected:
            values = [item.summary.get("tasks", {}).get(task) for item in items]
            if any(
                value is None
                or int(value.get("count", 0)) <= 0
                or not _finite(value.get("normalized_mae"))
                for value in values
            ):
                complete = False
                break
            row = {"run": run, "model": display, "task": task, "source_run": source}
            for metric in ("mae", "rmse", "r2", "normalized_mae", "normalized_rmse"):
                numbers = [float(value[metric]) for value in values]
                row[f"{metric}_mean"] = statistics.mean(numbers)
                row[f"{metric}_std"] = statistics.stdev(numbers)
            metrics_rows.append(row)
            task_values.append(float(row["normalized_mae_mean"]))
        if not complete:
            metrics_rows = [row for row in metrics_rows if row["run"] != run]
            continue
        comparison_hash = first_reporting["comparison_identity"]["hash"]
        comparisons.setdefault(comparison_hash, []).append(run)
        leaders.append(
            {
                "run": run, "model": display,
                "macro_normalized_mae": sum(task_values) / len(task_values),
                "valid_tasks": len(task_values), "total_tasks": len(expected),
                "per_task_wins": 0, "source_run": source,
                "checkpoint_epoch": next(iter(checkpoints)),
            }
        )
    for candidate in _current_completed(candidates):
        if candidate.metadata["stage"] != "benchmark":
            continue
        summary = candidate.summary
        reporting = summary["reporting"]
        section = reporting["benchmarks"]["stage3_validation"]
        expected = tuple(section["protocol"]["expected_tasks"])
        metrics = summary["stage3_property_benchmark"]["validation_five_fold"]
        if not expected or any(
            task not in metrics
            or int(metrics[task]["normalized_mae"].get("count", 0)) != 5
            or int(metrics[task]["normalized_rmse"].get("count", 0)) != 5
            for task in expected
        ):
            continue
        model_id = reporting["model_id"]
        display = reporting["model_display_name"]
        run = _run_id(model_id, candidate.source_run)
        task_values = []
        for task in expected:
            value = metrics[task]
            row = {"run": run, "model": display, "task": task, "source_run": candidate.source_run}
            for metric in ("mae", "rmse", "r2", "normalized_mae", "normalized_rmse"):
                row[f"{metric}_mean"] = value[metric]["mean"]
                row[f"{metric}_std"] = value[metric]["std"]
            metrics_rows.append(row)
            task_values.append(float(value["normalized_mae"]["mean"]))
        comparisons.setdefault(section["comparison_identity"]["hash"], []).append(run)
        leaders.append(
            {
                "run": run, "model": display,
                "macro_normalized_mae": sum(task_values) / len(task_values),
                "valid_tasks": len(task_values), "total_tasks": len(expected),
                "per_task_wins": 0, "source_run": candidate.source_run,
                "checkpoint_epoch": "",
            }
        )
    _require_one_comparison(comparisons, "Stage 3 validation")
    wins = _wins(metrics_rows, "task", "normalized_mae_mean")
    for row in leaders:
        row["per_task_wins"] = len(wins.get(row["run"], ()))
    return _rank(leaders, "macro_normalized_mae"), sorted(
        metrics_rows, key=lambda row: (row["run"], row["task"])
    )


def _stage2(
    candidates: Sequence[Candidate],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    leaders: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    comparisons: dict[str, list[str]] = {}
    for candidate in _current_completed(candidates):
        summary = candidate.summary
        reporting = summary["reporting"]
        if candidate.metadata["stage"] == "benchmark":
            section = reporting["benchmarks"]["stage2_physics"]
            metrics = summary["stage2_physics_benchmark"]["test"]
            checkpoint = ""
        elif candidate.metadata["stage"] == "stage2":
            section = reporting
            metrics = summary["tasks"]
            checkpoint = summary["checkpoint_epoch"]
        else:
            continue
        expected = tuple(section["protocol"]["expected_targets"])
        flattened = {
            f"{task}::{target}": (task, target, value)
            for task, targets in metrics.items()
            for target, value in targets.items()
        }
        if not expected or any(
            scalar not in flattened
            or int(flattened[scalar][2].get("count", 0)) <= 0
            or not _finite(flattened[scalar][2].get("normalized_mae"))
            for scalar in expected
        ):
            continue
        model_id = reporting["model_id"]
        display = reporting["model_display_name"]
        run = _run_id(model_id, candidate.source_run)
        for scalar in expected:
            task, target, value = flattened[scalar]
            metrics_rows.append(
                {
                    "run": run, "model": display, "task": task, "target": target,
                    "count": value["count"], "mae": value["mae"],
                    "rmse": value["rmse"], "r2": value["r2"],
                    "normalized_mae": value["normalized_mae"],
                    "normalized_rmse": value["normalized_rmse"],
                    "source_run": candidate.source_run,
                }
            )
        values = [float(flattened[scalar][2]["normalized_mae"]) for scalar in expected]
        comparisons.setdefault(section["comparison_identity"]["hash"], []).append(run)
        leaders.append(
            {
                "run": run, "model": display,
                "macro_normalized_mae": sum(values) / len(values),
                "valid_targets": len(values), "total_targets": len(expected),
                "per_target_wins": 0, "source_run": candidate.source_run,
                "checkpoint_epoch": checkpoint,
            }
        )
    _require_one_comparison(comparisons, "Stage 2 physics")
    wins = _wins(metrics_rows, "target", "normalized_mae", secondary="task")
    for row in leaders:
        row["per_target_wins"] = len(wins.get(row["run"], ()))
    return _rank(leaders, "macro_normalized_mae"), sorted(
        metrics_rows, key=lambda row: (row["run"], row["task"], row["target"])
    ), wins


def _require_one_comparison(groups: Mapping[str, Sequence[str]], label: str) -> None:
    nonempty = {key: values for key, values in groups.items() if values}
    if len(nonempty) > 1:
        details = "; ".join(f"{key}: {','.join(values)}" for key, values in sorted(nonempty.items()))
        raise ValueError(f"{label} contains incompatible comparison identities: {details}")


def _rank(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (float(row[metric]), row["run"]))
    previous: float | None = None
    rank = 0
    for index, row in enumerate(ordered, start=1):
        value = float(row[metric])
        if previous is None or value != previous:
            rank = index
            previous = value
        row["rank"] = rank
    return ordered


def _wins(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    metric: str,
    *,
    secondary: str | None = None,
) -> dict[str, list[str]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        identity = f"{row[secondary]}::{row[key]}" if secondary else str(row[key])
        groups.setdefault(identity, []).append(row)
    result: dict[str, list[str]] = {}
    for identity, values in groups.items():
        best = min(float(row[metric]) for row in values)
        for row in values:
            if float(row[metric]) == best:
                result.setdefault(str(row["run"]), []).append(identity)
    return {run: sorted(values) for run, values in sorted(result.items())}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})


def _overview(
    stage3_test: Sequence[Mapping[str, Any]],
    stage3_validation: Sequence[Mapping[str, Any]],
    stage2: Sequence[Mapping[str, Any]],
    test_wins: Mapping[str, Sequence[str]],
    stage2_wins: Mapping[str, Sequence[str]],
    health: Sequence[Mapping[str, Any]],
) -> str:
    lines = ["# ILUME result overview", ""]
    for title, rows in (
        ("Stage 3 TEST (5-fold ensemble)", stage3_test),
        ("Stage 3 VALIDATION (5-fold mean)", stage3_validation),
        ("Stage 2 PHYSICS", stage2),
    ):
        lines.extend((f"## {title}", ""))
        if rows:
            for row in rows:
                lines.append(
                    f"{row['rank']}. {row['model']} — macro normalized MAE "
                    f"{float(row['macro_normalized_mae']):.6g}"
                )
        else:
            lines.append("No eligible run.")
        if title.startswith("Stage 3 TEST") and rows:
            lines.append(
                f"Coverage: {rows[0]['total_tasks']} test tasks / "
                f"{rows[0]['enabled_tasks']} enabled Stage 3 tasks."
            )
        lines.append("")
    lines.extend(("## Stage 3 task wins", ""))
    lines.extend(
        f"- {run}: {len(values)}" for run, values in test_wins.items()
    )
    if not test_wins:
        lines.append("- None")
    lines.extend(("", "## Stage 2 target wins", ""))
    lines.extend(
        f"- {run}: {len(values)}" for run, values in stage2_wins.items()
    )
    if not stage2_wins:
        lines.append("- None")
    lines.extend(("", "## Experiment health", ""))
    icons = {"complete": "✓", "legacy": "⚠", "running": "⚠", "failed": "✗"}
    for row in health:
        lines.append(
            f"- {icons.get(str(row['completeness']), '⚠')} {row['run']}: "
            f"{row['completeness']}"
            + (f" ({row['issues']})" if row["issues"] else "")
        )
    return "\n".join(lines) + "\n"


def build_summary(input_root: Path, repository_root: Path) -> dict[str, Any]:
    candidates = discover_candidates(input_root, repository_root)
    health = _health(candidates)
    stage3_test, stage3_test_metrics, test_wins = _stage3_test(candidates)
    stage3_validation, stage3_validation_metrics = _stage3_validation(candidates)
    stage2, stage2_metrics, stage2_wins = _stage2(candidates)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "inputs": sorted(
            {
                candidate.metadata.get("semantic_identity", {}).get("hash", "")
                for candidate in candidates
                if candidate.metadata.get("semantic_identity", {}).get("hash")
            }
        ),
        "comparison_identities": _comparison_catalog(candidates),
        "leaderboards": {
            "stage3_test": stage3_test,
            "stage3_validation": stage3_validation,
            "stage2_physics": stage2,
        },
        "metrics": {
            "stage3_test": stage3_test_metrics,
            "stage3_validation": stage3_validation_metrics,
            "stage2_physics": stage2_metrics,
        },
        "wins": {"stage3_test": test_wins, "stage2_physics": stage2_wins},
        "health": health,
    }


def _comparison_catalog(
    candidates: Sequence[Candidate],
) -> dict[str, list[dict[str, Any]]]:
    catalog: dict[str, dict[str, dict[str, Any]]] = {
        "stage3_test": {},
        "stage3_validation": {},
        "stage2_physics": {},
    }
    sources: dict[tuple[str, str], list[str]] = {}
    for candidate in _current_completed(candidates):
        reporting = candidate.summary["reporting"]
        sections: Iterable[tuple[str, Mapping[str, Any]]]
        if candidate.metadata["stage"] == "benchmark":
            sections = reporting["benchmarks"].items()
        elif candidate.metadata["stage"] == "stage3":
            name = (
                "stage3_test"
                if candidate.summary.get("split") == "test"
                else "stage3_validation"
            )
            sections = ((name, reporting),)
        else:
            sections = (("stage2_physics", reporting),)
        for name, section in sections:
            identity = section["comparison_identity"]
            identity_hash = str(identity["hash"])
            existing = catalog[name].get(identity_hash)
            if existing is not None and existing != identity:
                raise ValueError(
                    f"Comparison identity hash collision in {name}: {identity_hash}"
                )
            catalog[name][identity_hash] = dict(identity)
            sources.setdefault((name, identity_hash), []).append(candidate.source_run)
    return {
        name: [
            {
                "identity": values[identity_hash],
                "source_runs": sorted(sources[(name, identity_hash)]),
            }
            for identity_hash in sorted(values)
        ]
        for name, values in catalog.items()
    }


def write_summary_snapshot(payload: Mapping[str, Any], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    leaderboards = payload["leaderboards"]
    metrics = payload["metrics"]
    _write_csv(
        destination / "stage3_test_leaderboard.csv",
        leaderboards["stage3_test"],
        ("rank", "run", "model", "macro_normalized_mae", "valid_tasks", "total_tasks", "per_task_wins", "source_run", "checkpoint_epoch"),
    )
    _write_csv(
        destination / "stage3_validation_leaderboard.csv",
        leaderboards["stage3_validation"],
        ("rank", "run", "model", "macro_normalized_mae", "valid_tasks", "total_tasks", "per_task_wins", "source_run", "checkpoint_epoch"),
    )
    _write_csv(
        destination / "stage2_physics_leaderboard.csv",
        leaderboards["stage2_physics"],
        ("rank", "run", "model", "macro_normalized_mae", "valid_targets", "total_targets", "per_target_wins", "source_run", "checkpoint_epoch"),
    )
    _write_csv(
        destination / "stage3_test_metrics.csv",
        metrics["stage3_test"],
        ("run", "model", "task", "count", "mae", "rmse", "r2", "normalized_mae", "normalized_rmse", "source_run"),
    )
    _write_csv(
        destination / "stage3_validation_metrics.csv",
        metrics["stage3_validation"],
        ("run", "model", "task", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "r2_mean", "r2_std", "normalized_mae_mean", "normalized_mae_std", "normalized_rmse_mean", "normalized_rmse_std", "source_run"),
    )
    _write_csv(
        destination / "stage2_physics_metrics.csv",
        metrics["stage2_physics"],
        ("run", "model", "task", "target", "count", "mae", "rmse", "r2", "normalized_mae", "normalized_rmse", "source_run"),
    )
    _write_csv(
        destination / "sweep_status.csv",
        payload["health"],
        ("run", "model", "stage", "operation", "status", "completeness", "checkpoint_epoch", "enabled_tasks", "expected_tasks", "available_tasks", "available_folds", "failed_jobs", "source_run", "issues"),
    )
    (destination / "overview.md").write_text(
        _overview(
            leaderboards["stage3_test"], leaderboards["stage3_validation"],
            leaderboards["stage2_physics"], payload["wins"]["stage3_test"],
            payload["wins"]["stage2_physics"], payload["health"],
        ),
        encoding="utf-8",
    )
    atomic_json(destination / "summary.json", payload)
    actual = tuple(sorted(path.name for path in destination.iterdir()))
    if actual != tuple(sorted(SUMMARY_FILES)):
        raise AssertionError(f"Summary snapshot file set mismatch: {actual}")


def publish_summary(input_root: Path, output: Path, repository_root: Path) -> dict[str, Any]:
    payload = build_summary(input_root, repository_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent))
    shutil.rmtree(staging)
    backup: Path | None = None
    try:
        write_summary_snapshot(payload, staging)
        if output.exists():
            if not output.is_dir():
                raise FileExistsError(f"Summary output is not a directory: {output}")
            backup = Path(tempfile.mkdtemp(prefix=f".{output.name}-backup-", dir=output.parent))
            shutil.rmtree(backup)
            output.replace(backup)
        try:
            staging.replace(output)
        except BaseException:
            if backup is not None and backup.exists() and not output.exists():
                backup.replace(output)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return payload
    finally:
        if staging.exists():
            shutil.rmtree(staging)


__all__ = [
    "SUMMARY_FILES",
    "build_summary",
    "discover_candidates",
    "publish_summary",
    "write_summary_snapshot",
]
