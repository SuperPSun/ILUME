from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape

from common.identity import validate_semantic_identity
from common.identity import semantic_identity
from common.io import atomic_json
from common.reporting import REPORTING_SCHEMA_VERSION


SUMMARY_SCHEMA_VERSION = 1
SUMMARY_FILES = (
    "overview.md",
    "radar.svg",
    "stage3_test_leaderboard.csv",
    "stage3_validation_leaderboard.csv",
    "stage3_test_metrics.csv",
    "stage3_test_task_mae.csv",
    "stage3_test_task_rank.csv",
    "stage3_validation_task_mae.csv",
    "stage3_validation_task_rank.csv",
    "stage3_validation_metrics.csv",
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


def _resolved_directories(
    paths: Path | Sequence[Path], *, label: str, required: bool
) -> tuple[Path, ...]:
    values = (paths,) if isinstance(paths, Path) else tuple(paths)
    if required and not values:
        raise ValueError(f"At least one summary {label} directory is required")
    resolved = tuple(
        sorted({path.resolve() for path in values}, key=lambda path: path.as_posix())
    )
    for path in resolved:
        if not path.is_dir():
            raise FileNotFoundError(f"Summary {label} directory does not exist: {path}")
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def discover_candidates(
    input_roots: Path | Sequence[Path],
    repository_root: Path,
    *,
    include_roots: Path | Sequence[Path] = (),
) -> list[Candidate]:
    inputs = _resolved_directories(input_roots, label="input", required=True)
    includes = _resolved_directories(include_roots, label="include", required=False)
    outside = [
        include for include in includes
        if not any(_is_within(include, input_root) for input_root in inputs)
    ]
    if outside:
        raise ValueError(
            "Summary include directories must be inside an input directory:\n- "
            + "\n- ".join(str(path) for path in outside)
        )

    candidates: list[Candidate] = []
    malformed: list[str] = []
    matched_includes: set[Path] = set()
    metadata_paths = sorted(
        {
            metadata_path.resolve()
            for input_root in inputs
            for metadata_path in input_root.rglob("metadata.json")
            if not includes or any(
                _is_within(metadata_path.resolve().parent, include)
                for include in includes
            )
        },
        key=lambda path: path.as_posix(),
    )
    for metadata_path in metadata_paths:
        try:
            metadata = _json(metadata_path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            malformed.append(f"{metadata_path}: malformed metadata: {error}")
            continue
        key = (metadata.get("stage"), metadata.get("operation"))
        if key not in {
            ("benchmark", "sweep"),
            ("stage3", "evaluate"),
        }:
            continue
        root = metadata_path.parent
        matched_includes.update(
            include for include in includes if _is_within(root, include)
        )
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
    unmatched = [include for include in includes if include not in matched_includes]
    if unmatched:
        raise ValueError(
            "Summary include directories contain no reporting candidates:\n- "
            + "\n- ".join(str(path) for path in unmatched)
        )
    return candidates


def _validate_comparison(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} lacks comparison identity")
    validate_semantic_identity(value)
    if value.get("type") != "reporting.comparison.v1":
        raise ValueError(f"{context} has unsupported comparison identity")


def _stage3_comparison_key(identity: Mapping[str, Any]) -> str:
    payload = dict(identity["payload"])
    if payload.get("benchmark") != "stage3_property":
        return str(identity["hash"])
    payload.pop("normalization", None)
    return semantic_identity(
        "reporting.stage3-comparison-without-normalization.v1", payload
    )["hash"]




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
        if not isinstance(benchmarks, dict) or not {
            "stage3_test", "stage3_validation"
        }.issubset(benchmarks):
            raise ValueError(
                "benchmark sweep Stage 3 reporting sections are incomplete"
            )
        for name in ("stage3_test", "stage3_validation"):
            section = benchmarks[name]
            if section.get("status") not in {"unsupported", "incomplete"}:
                _validate_comparison(section.get("comparison_identity"), name)
        if not isinstance(reporting.get("source_runs"), dict):
            raise ValueError("benchmark sweep reporting lacks source runs")
        return
    _validate_comparison(
        reporting.get("comparison_identity"), "stage3 evaluation"
    )
    protocol = reporting.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("split") not in {
        "valid", "test"
    }:
        raise ValueError("stage3 evaluation protocol is malformed")

def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _run_id(model: str, source_run: str) -> str:
    return f"{model}@{source_run}"


def _display_name(candidate: Candidate, reporting: Mapping[str, Any]) -> str:
    """Return a human-readable model label without changing reporting identity."""
    display = str(reporting["model_display_name"])
    source = Path(candidate.source_run).parts
    marker = ("outputs", "experiments_v1", "stage3", "formal")
    for index in range(len(source) - len(marker)):
        if source[index:index + len(marker)] != marker:
            continue
        scale = source[index + len(marker)]
        scale_label = {"s": "S", "base": "Base", "l": "L", "xl": "XL"}
        if (
            candidate.metadata.get("stage") == "stage3"
            and reporting.get("model_id") == "ilume"
            and scale in scale_label
        ):
            return f"{display} Capacity v1 ({scale_label[scale]})"
    variant = _ilume_stage3_variant(candidate, reporting)
    if variant is not None:
        return f"{display} ({variant})"
    return display


def _ilume_stage3_variant(
    candidate: Candidate, reporting: Mapping[str, Any]
) -> str | None:
    if (
        candidate.metadata.get("stage") != "stage3"
        or reporting.get("model_id") != "ilume"
    ):
        return None
    source = Path(candidate.source_run).parts
    for index, part in enumerate(source[:-1]):
        marker = source[index - 2:index]
        if (
            part == "stage3"
            and index >= 2
            and marker in {("outputs", "v1"), ("outputs", "v2")}
        ):
            return source[index + 1]
    return None


def _validation_study_key(
    candidate: Candidate, reporting: Mapping[str, Any]
) -> tuple[str, str, str]:
    return (
        str(reporting["model_id"]),
        str(reporting["study_id"]),
        _ilume_stage3_variant(candidate, reporting) or "",
    )


def _model(candidate: Candidate) -> tuple[str, str]:
    if candidate.summary:
        reporting = candidate.summary.get("reporting", {})
        if reporting.get("model_id") and reporting.get("model_display_name"):
            return str(reporting["model_id"]), _display_name(candidate, reporting)
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
                display,
                str(candidate.metadata.get("stage", "")),
                str(candidate.metadata.get("operation", "")),
                str((candidate.summary or {}).get("split", provenance.get("split", ""))),
                str(
                    (candidate.summary or {})
                    .get("reporting", {})
                    .get("protocol", {})
                    .get("fold", provenance.get("fold", ""))
                ),
            )
        )
        status = str(candidate.metadata.get("status", "unknown"))
        summary = candidate.summary or {}
        issues = list(candidate.issues)
        reporting = summary.get("reporting", {})
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
            if candidate.metadata["stage"] == "benchmark":
                sections = reporting["benchmarks"]
                expected = ";".join(
                    f"{name}:{len(sections[name]['protocol'].get('expected_tasks', ()))}"
                    for name in ("stage3_test", "stage3_validation")
                )
                available = ";".join(
                    (
                        "stage3_test:"
                        + str(
                            len(
                                summary["stage3_property_benchmark"][
                                    "test_ensemble"
                                ]
                            )
                        ),
                        "stage3_validation:"
                        + str(
                            len(
                                summary["stage3_property_benchmark"][
                                    "validation_five_fold"
                                ]
                            )
                        ),
                    )
                )
                folds = ";".join(
                    map(
                        str,
                        sections["stage3_validation"]["protocol"]["folds"],
                    )
                )
            else:
                protocol = reporting["protocol"]
                expected_tasks = protocol.get("expected_tasks", ())
                expected = len(expected_tasks)
                metrics = (
                    summary.get("ensemble", {}).get("tasks", {})
                    if summary.get("split") == "test"
                    else summary.get("tasks", {})
                )
                available = sum(
                    int(values.get("count", 0)) > 0
                    for values in metrics.values()
                )
                fold = protocol.get("fold")
                folds = (
                    ";".join(map(str, protocol.get("folds", ())))
                    if fold is None
                    else str(fold)
                )
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
                    len(
                        summary.get("reporting", {})
                        .get("protocol", {})
                        .get("enabled_tasks", ())
                    )
                    if candidate.metadata.get("stage") == "stage3"
                    else ""
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

    validation_groups: dict[tuple[str, str, str], dict[int, list[Candidate]]] = {}
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
                task
                for task in test_expected
                if task not in test_metrics
                or int(test_metrics[task].get("count", 0)) <= 0
            ]
            if test_missing:
                mark_incomplete(
                    candidate,
                    "stage3_test_missing_tasks=" + ",".join(test_missing),
                )
            test_folds = tuple(
                sections["stage3_test"]["protocol"].get("folds", ())
            )
            if test_folds != (1, 2, 3, 4, 5):
                mark_incomplete(
                    candidate,
                    "stage3_test_missing_folds="
                    + ",".join(
                        map(
                            str,
                            sorted(set(range(1, 6)) - set(test_folds)),
                        )
                    ),
                )
            valid_expected = tuple(
                sections["stage3_validation"]["protocol"]["expected_tasks"]
            )
            valid_metrics = summary["stage3_property_benchmark"][
                "validation_five_fold"
            ]
            valid_missing = [
                task
                for task in valid_expected
                if task not in valid_metrics
                or int(
                    valid_metrics[task]
                    .get("normalized_mae", {})
                    .get("count", 0)
                )
                != 5
            ]
            if valid_missing:
                mark_incomplete(
                    candidate,
                    "stage3_validation_missing_tasks="
                    + ",".join(valid_missing),
                )
            try:
                if int(summary.get("jobs", {}).get("failed", 0)) > 0:
                    mark_incomplete(candidate, "failed_jobs")
            except (TypeError, ValueError):
                mark_incomplete(candidate, "failed_jobs_unknown")
        else:
            protocol = reporting["protocol"]
            expected_tasks = tuple(protocol["expected_tasks"])
            metrics = (
                summary.get("ensemble", {}).get("tasks", {})
                if summary.get("split") == "test"
                else summary.get("tasks", {})
            )
            missing = [
                task
                for task in expected_tasks
                if task not in metrics
                or int(metrics[task].get("count", 0)) <= 0
            ]
            if missing:
                mark_incomplete(
                    candidate, "missing_tasks=" + ",".join(missing)
                )
            if summary.get("split") == "test":
                folds = tuple(protocol.get("folds", ()))
                if folds != (1, 2, 3, 4, 5):
                    mark_incomplete(
                        candidate,
                        "missing_folds="
                        + ",".join(
                            map(
                                str,
                                sorted(set(range(1, 6)) - set(folds)),
                            )
                        ),
                    )
            else:
                key = _validation_study_key(candidate, reporting)
                fold = int(protocol["fold"])
                validation_groups.setdefault(key, {}).setdefault(
                    fold, []
                ).append(candidate)

    for folds in validation_groups.values():
        missing = sorted(set(range(1, 6)) - set(folds))
        duplicates = sorted(
            fold for fold, items in folds.items() if len(items) > 1
        )
        unique_items = [
            items[0] for items in folds.values() if len(items) == 1
        ]
        comparison_hashes = {
            candidate.summary["reporting"]["comparison_identity"]["hash"]
            for candidate in unique_items
        }
        expected_sets = {
            tuple(
                candidate.summary["reporting"]["protocol"][
                    "expected_tasks"
                ]
            )
            for candidate in unique_items
        }
        checkpoints = {
            candidate.summary.get("checkpoint_epoch")
            for candidate in unique_items
        }
        for items in folds.values():
            for candidate in items:
                if missing:
                    mark_incomplete(
                        candidate,
                        "missing_folds=" + ",".join(map(str, missing)),
                    )
                if duplicates:
                    mark_incomplete(
                        candidate,
                        "duplicate_folds="
                        + ",".join(map(str, duplicates)),
                    )
                if len(comparison_hashes) > 1:
                    mark_incomplete(
                        candidate, "comparison_identity_mismatch"
                    )
                if len(expected_sets) > 1:
                    mark_incomplete(candidate, "expected_tasks_mismatch")
                if len(checkpoints) > 1:
                    mark_incomplete(candidate, "checkpoint_epoch_mismatch")
    signature_counts = {
        signature: signatures.count(signature)
        for signature in set(signatures)
    }
    for row, signature in zip(rows, signatures, strict=True):
        if signature_counts[signature] > 1:
            row["issues"] = ";".join(
                item
                for item in (row["issues"], "alternative_run")
                if item
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
        display = _display_name(candidate, reporting)
        run = _run_id(model_id, candidate.source_run)
        values = [float(metrics[task]["normalized_mae"]) for task in expected]
        comparison_hash = _stage3_comparison_key(
            section["comparison_identity"]
        )
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
    fold_groups: dict[tuple[str, str, str], dict[int, Candidate]] = {}
    ambiguous: set[tuple[str, str, str]] = set()
    for candidate in _current_completed(candidates):
        if candidate.metadata["stage"] != "stage3" or candidate.summary.get("split") != "valid":
            continue
        reporting = candidate.summary["reporting"]
        key = _validation_study_key(candidate, reporting)
        fold = int(reporting["protocol"]["fold"])
        if fold in fold_groups.setdefault(key, {}):
            ambiguous.add(key)
        fold_groups[key][fold] = candidate
    for (model_id, study_id, variant), folds in sorted(fold_groups.items()):
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
        run = f"{model_id}@study:{study_id}" + (
            f":variant:{variant}" if variant else ""
        )
        display = _display_name(items[0], first_reporting)
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
        comparison_hash = _stage3_comparison_key(
            first_reporting["comparison_identity"]
        )
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
        display = _display_name(candidate, reporting)
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
        comparisons.setdefault(
            _stage3_comparison_key(section["comparison_identity"]), []
        ).append(run)
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
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})


def _stage3_task_tables(
    metrics: Sequence[Mapping[str, Any]],
    leaders: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    split: str,
) -> tuple[tuple[str, ...], list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = tuple(sorted({str(row["task"]) for row in metrics}))
    mae_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    ordered_runs = [str(row["run"]) for row in leaders]
    model_by_run = {str(row["run"]): str(row["model"]) for row in leaders}
    values: dict[tuple[str, str], float] = {}
    for row in metrics:
        run = str(row["run"])
        task = str(row["task"])
        if run not in model_by_run:
            continue
        key = (run, task)
        if key in values:
            raise ValueError(f"Duplicate Stage 3 {split} metric: {run}/{task}")
        value = float(row[metric_key])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite Stage 3 {split} MAE: {run}/{task}")
        values[key] = value

    ranks: dict[tuple[str, str], int] = {}
    for task in tasks:
        ordered = sorted(
            (values[(run, task)], run)
            for run in ordered_runs
            if (run, task) in values
        )
        previous: float | None = None
        rank = 0
        for index, (value, run) in enumerate(ordered, start=1):
            if previous is None or value != previous:
                rank = index
                previous = value
            ranks[(run, task)] = rank

    for run in ordered_runs:
        common = {"run": run, "model": model_by_run[run]}
        mae_rows.append(
            {**common, **{task: values.get((run, task), "") for task in tasks}}
        )
        rank_rows.append(
            {**common, **{task: ranks.get((run, task), "") for task in tasks}}
        )
    return tasks, mae_rows, rank_rows


def _overview(
    stage3_test: Sequence[Mapping[str, Any]],
    stage3_validation: Sequence[Mapping[str, Any]],
    health: Sequence[Mapping[str, Any]],
) -> str:
    lines = ["# ILUME result overview", ""]
    for title, rows in (
        ("Stage 3 TEST (5-fold ensemble)", stage3_test),
        ("Stage 3 VALIDATION (5-fold mean)", stage3_validation),
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
    lines.extend(("## Experiment health", ""))
    icons = {
        "complete": "✓",
        "legacy": "⚠",
        "running": "⚠",
        "failed": "✗",
    }
    for row in health:
        lines.append(
            f"- {icons.get(str(row['completeness']), '⚠')} {row['run']}: "
            f"{row['completeness']}"
            + (f" ({row['issues']})" if row["issues"] else "")
        )
    return "\n".join(lines) + "\n"

def _radar_score(value: float, best: float) -> float:
    if best == 0.0:
        return 1.0 if value == 0.0 else 0.0
    return best / value


def _radar_panel_data(
    rows: Sequence[Mapping[str, Any]],
    leaders: Sequence[Mapping[str, Any]],
    *,
    task_key: str,
    metric_key: str,
    tasks: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    model_by_run = {str(row["run"]): str(row["model"]) for row in leaders}
    ordered_runs = [str(row["run"]) for row in leaders]
    values: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.get("run") not in model_by_run or not _finite(row.get(metric_key)):
            continue
        values[str(row["run"]), str(row[task_key])] = float(row[metric_key])
    axes = tuple(tasks) if tasks is not None else tuple(sorted({
        task for run, task in values if run in model_by_run
    }))
    series: list[dict[str, Any]] = []
    for run in ordered_runs:
        scores: list[float] = []
        for task in axes:
            available = [
                value for (candidate_run, candidate_task), value in values.items()
                if candidate_task == task and candidate_run in model_by_run
            ]
            value = values.get((run, task))
            scores.append(
                _radar_score(value, min(available))
                if value is not None and available else 0.0
            )
        series.append({"run": run, "model": model_by_run[run], "scores": scores})
    return axes, series


def _svg_text(value: object) -> str:
    return escape(str(value), {'"': "&quot;"})


def _radar_point(center_x: float, center_y: float, radius: float, angle: float) -> tuple[float, float]:
    return (
        center_x + radius * math.sin(angle),
        center_y - radius * math.cos(angle),
    )


def _radar_svg(payload: Mapping[str, Any]) -> str:
    leaderboards = payload["leaderboards"]
    metrics = payload["metrics"]
    panels = (
        (
            "stage3_test",
            "Stage 3 test",
            *_radar_panel_data(
                metrics["stage3_test"],
                leaderboards["stage3_test"],
                task_key="task",
                metric_key="normalized_mae",
            ),
        ),
        (
            "stage3_validation",
            "Stage 3 validation",
            *_radar_panel_data(
                metrics["stage3_validation"],
                leaderboards["stage3_validation"],
                task_key="task",
                metric_key="normalized_mae_mean",
            ),
        ),
    )
    palette = (
        "#0072b2",
        "#d55e00",
        "#009e73",
        "#cc79a7",
        "#e69f00",
        "#56b4e9",
    )
    models = sorted(
        {
            series["model"]
            for _, _, _, items in panels
            for series in items
        }
    )
    colors = {
        model: palette[index % len(palette)]
        for index, model in enumerate(models)
    }
    centers = ((330.0, 520.0), (950.0, 520.0))
    radius = 230.0
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="820" viewBox="0 0 1280 820">',
        '<rect width="1280" height="820" fill="white"/>',
        '<text x="640" y="42" text-anchor="middle" font-family="sans-serif" font-size="26" font-weight="bold">ILUME benchmark radar scores</text>',
        '<text x="640" y="70" text-anchor="middle" font-family="sans-serif" font-size="14">Score = best normalized MAE / model normalized MAE; higher is better.</text>',
    ]
    for (panel_id, title, tasks, series), (
        center_x,
        center_y,
    ) in zip(panels, centers, strict=True):
        lines.append(
            f'<text x="{center_x:.1f}" y="112" text-anchor="middle" '
            f'font-family="sans-serif" font-size="20" font-weight="bold">'
            f'{_svg_text(title)}</text>'
        )
        if not tasks or not series:
            lines.append(
                f'<text x="{center_x:.1f}" y="{center_y:.1f}" '
                f'text-anchor="middle" font-family="sans-serif" '
                f'font-size="16">No eligible run.</text>'
            )
            continue
        angles = tuple(
            2.0 * math.pi * index / len(tasks)
            for index in range(len(tasks))
        )
        for fraction in (0.25, 0.5, 0.75, 1.0):
            points = " ".join(
                f"{x:.2f},{y:.2f}"
                for x, y in (
                    _radar_point(
                        center_x,
                        center_y,
                        radius * fraction,
                        angle,
                    )
                    for angle in angles
                )
            )
            lines.append(
                f'<polygon points="{points}" fill="none" '
                f'stroke="#c7c7c7" stroke-width="1"/>'
            )
            lines.append(
                f'<text x="{center_x + 5:.2f}" '
                f'y="{center_y - radius * fraction + 4:.2f}" '
                f'font-family="sans-serif" font-size="9" '
                f'fill="#666">{fraction:g}</text>'
            )
        lines.append(
            f'<text x="{center_x + 5:.2f}" y="{center_y + 4:.2f}" '
            f'font-family="sans-serif" font-size="9" fill="#666">0</text>'
        )
        for task, angle in zip(tasks, angles, strict=True):
            x, y = _radar_point(center_x, center_y, radius, angle)
            label_x, label_y = _radar_point(
                center_x, center_y, radius + 24.0, angle
            )
            anchor = (
                "middle"
                if abs(math.sin(angle)) < 0.2
                else "start"
                if math.sin(angle) > 0
                else "end"
            )
            label = task.rsplit("/", 1)[-1].replace("_", " ")
            lines.extend(
                (
                    f'<line x1="{center_x:.2f}" y1="{center_y:.2f}" '
                    f'x2="{x:.2f}" y2="{y:.2f}" stroke="#c7c7c7" '
                    f'stroke-width="1"/>',
                    f'<text x="{label_x:.2f}" y="{label_y:.2f}" '
                    f'text-anchor="{anchor}" font-family="sans-serif" '
                    f'font-size="10">{_svg_text(label)}</text>',
                )
            )
        for index, series_item in enumerate(series):
            points = " ".join(
                f"{x:.2f},{y:.2f}"
                for x, y in (
                    _radar_point(
                        center_x,
                        center_y,
                        radius * score,
                        angle,
                    )
                    for score, angle in zip(
                        series_item["scores"], angles, strict=True
                    )
                )
            )
            score_text = ",".join(
                f"{score:.6g}" for score in series_item["scores"]
            )
            color = colors[series_item["model"]]
            lines.append(
                f'<polygon class="series" data-panel="{panel_id}" '
                f'data-run="{_svg_text(series_item["run"])}" '
                f'data-scores="{score_text}" points="{points}" '
                f'fill="{color}" fill-opacity="0.12" stroke="{color}" '
                f'stroke-width="2"/>'
            )
            legend_y = 145 + index * 20
            lines.extend(
                (
                    f'<line x1="{center_x - radius:.1f}" y1="{legend_y}" '
                    f'x2="{center_x - radius + 18:.1f}" y2="{legend_y}" '
                    f'stroke="{color}" stroke-width="3"/>',
                    f'<text x="{center_x - radius + 24:.1f}" '
                    f'y="{legend_y + 4}" font-family="sans-serif" '
                    f'font-size="12">{_svg_text(series_item["model"])}</text>',
                )
            )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"

def build_summary(
    input_roots: Path | Sequence[Path],
    repository_root: Path,
    *,
    include_roots: Path | Sequence[Path] = (),
) -> dict[str, Any]:
    candidates = discover_candidates(
        input_roots, repository_root, include_roots=include_roots
    )
    health = _health(candidates)
    stage3_test, stage3_test_metrics, test_wins = _stage3_test(
        candidates
    )
    stage3_validation, stage3_validation_metrics = _stage3_validation(
        candidates
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "inputs": sorted(
            {
                candidate.metadata.get("semantic_identity", {}).get(
                    "hash", ""
                )
                for candidate in candidates
                if candidate.metadata.get("semantic_identity", {}).get(
                    "hash"
                )
            }
        ),
        "comparison_identities": _comparison_catalog(candidates),
        "leaderboards": {
            "stage3_test": stage3_test,
            "stage3_validation": stage3_validation,
        },
        "metrics": {
            "stage3_test": stage3_test_metrics,
            "stage3_validation": stage3_validation_metrics,
        },
        "wins": {"stage3_test": test_wins},
        "health": health,
    }

def _comparison_catalog(
    candidates: Sequence[Candidate],
) -> dict[str, list[dict[str, Any]]]:
    catalog: dict[str, dict[str, dict[str, Any]]] = {
        "stage3_test": {},
        "stage3_validation": {},
    }
    sources: dict[tuple[str, str], list[str]] = {}
    for candidate in _current_completed(candidates):
        reporting = candidate.summary["reporting"]
        if candidate.metadata["stage"] == "benchmark":
            sections = (
                (
                    "stage3_test",
                    reporting["benchmarks"]["stage3_test"],
                ),
                (
                    "stage3_validation",
                    reporting["benchmarks"]["stage3_validation"],
                ),
            )
        else:
            name = (
                "stage3_test"
                if candidate.summary.get("split") == "test"
                else "stage3_validation"
            )
            sections = ((name, reporting),)
        for name, section in sections:
            if section.get("status") in {"unsupported", "incomplete"}:
                continue
            identity = section["comparison_identity"]
            identity_hash = str(identity["hash"])
            existing = catalog[name].get(identity_hash)
            if existing is not None and existing != identity:
                raise ValueError(
                    f"Comparison identity hash collision in {name}: "
                    f"{identity_hash}"
                )
            catalog[name][identity_hash] = dict(identity)
            sources.setdefault((name, identity_hash), []).append(
                candidate.source_run
            )
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

def write_summary_snapshot(
    payload: Mapping[str, Any], destination: Path
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    leaderboards = payload["leaderboards"]
    metrics = payload["metrics"]
    test_fields, test_mae, test_rank = _stage3_task_tables(
        metrics["stage3_test"],
        leaderboards["stage3_test"],
        metric_key="mae",
        split="test",
    )
    validation_fields, validation_mae, validation_rank = _stage3_task_tables(
        metrics["stage3_validation"],
        leaderboards["stage3_validation"],
        metric_key="mae_mean",
        split="validation",
    )
    _write_csv(
        destination / "stage3_test_leaderboard.csv",
        leaderboards["stage3_test"],
        (
            "rank",
            "run",
            "model",
            "macro_normalized_mae",
            "valid_tasks",
            "total_tasks",
            "per_task_wins",
            "source_run",
            "checkpoint_epoch",
        ),
    )
    _write_csv(
        destination / "stage3_validation_leaderboard.csv",
        leaderboards["stage3_validation"],
        (
            "rank",
            "run",
            "model",
            "macro_normalized_mae",
            "valid_tasks",
            "total_tasks",
            "per_task_wins",
            "source_run",
            "checkpoint_epoch",
        ),
    )
    _write_csv(
        destination / "stage3_test_metrics.csv",
        metrics["stage3_test"],
        (
            "run",
            "model",
            "task",
            "count",
            "mae",
            "rmse",
            "r2",
            "normalized_mae",
            "normalized_rmse",
            "source_run",
        ),
    )
    _write_csv(
        destination / "stage3_test_task_mae.csv",
        test_mae,
        ("run", "model", *test_fields),
    )
    _write_csv(
        destination / "stage3_test_task_rank.csv",
        test_rank,
        ("run", "model", *test_fields),
    )
    _write_csv(
        destination / "stage3_validation_task_mae.csv",
        validation_mae,
        ("run", "model", *validation_fields),
    )
    _write_csv(
        destination / "stage3_validation_task_rank.csv",
        validation_rank,
        ("run", "model", *validation_fields),
    )
    _write_csv(
        destination / "stage3_validation_metrics.csv",
        metrics["stage3_validation"],
        (
            "run",
            "model",
            "task",
            "mae_mean",
            "mae_std",
            "rmse_mean",
            "rmse_std",
            "r2_mean",
            "r2_std",
            "normalized_mae_mean",
            "normalized_mae_std",
            "normalized_rmse_mean",
            "normalized_rmse_std",
            "source_run",
        ),
    )
    _write_csv(
        destination / "sweep_status.csv",
        payload["health"],
        (
            "run",
            "model",
            "stage",
            "operation",
            "status",
            "completeness",
            "checkpoint_epoch",
            "enabled_tasks",
            "expected_tasks",
            "available_tasks",
            "available_folds",
            "failed_jobs",
            "source_run",
            "issues",
        ),
    )
    (destination / "overview.md").write_text(
        _overview(
            leaderboards["stage3_test"],
            leaderboards["stage3_validation"],
            payload["health"],
        ),
        encoding="utf-8",
    )
    (destination / "radar.svg").write_text(
        _radar_svg(payload), encoding="utf-8"
    )
    atomic_json(destination / "summary.json", payload)
    actual = tuple(sorted(path.name for path in destination.iterdir()))
    if actual != tuple(sorted(SUMMARY_FILES)):
        raise AssertionError(
            f"Summary snapshot file set mismatch: {actual}"
        )

def publish_summary(
    input_roots: Path | Sequence[Path],
    output: Path,
    repository_root: Path,
    *,
    include_roots: Path | Sequence[Path] = (),
) -> dict[str, Any]:
    payload = build_summary(
        input_roots, repository_root, include_roots=include_roots
    )
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
