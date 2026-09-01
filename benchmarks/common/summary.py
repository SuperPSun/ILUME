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
from xml.sax.saxutils import escape

from common.identity import validate_semantic_identity
from common.identity import semantic_identity
from common.io import atomic_json
from common.reporting import (
    REPORTING_SCHEMA_VERSION,
    STAGE2_BENCHMARK_SUITE_CONTRACT,
)


SUMMARY_SCHEMA_VERSION = 1
SUMMARY_FILES = (
    "overview.md",
    "radar.svg",
    "stage3_test_leaderboard.csv",
    "stage3_validation_leaderboard.csv",
    "stage2_core_physics_leaderboard.csv",
    "stage2_partial_charge_leaderboard.csv",
    "stage2_physics_full_leaderboard.csv",
    "stage3_test_metrics.csv",
    "stage3_validation_metrics.csv",
    "stage2_core_physics_metrics.csv",
    "stage2_partial_charge_metrics.csv",
    "sweep_status.csv",
    "summary.json",
)

_PARTIAL_CHARGE_PREDICTION_FIELDS = (
    "source_row",
    "mol_id",
    "canonical_smiles",
    "role",
    "formal_charge",
    "evaluation_status",
    "exclusion_reason",
    "atom_count",
    "atom_indices",
    "elements",
    "target_charges",
    "predicted_charges",
    "absolute_errors",
    "molecule_mae",
    "mapping_status",
    "mapping_count_lower_bound",
    "selected_mapping_rank",
    "bond_match_mode",
    "unparsed_bond_types",
    "bond_fallback_reason",
)
_PARTIAL_CHARGE_RUNTIME_FIELDS = {
    "canonical_smiles",
    "predicted_charges",
    "absolute_errors",
    "molecule_mae",
}


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
            ("stage2", "evaluate"),
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


def _partial_charge_comparison_key(
    candidate: Candidate, section: Mapping[str, Any]
) -> str:
    identity = section["comparison_identity"]
    payload = identity["payload"]
    sources = dict(payload["sources"])
    task = "simulation/partial_atomic_charge"
    sources.pop(f"{task}:mapping_audit", None)
    sources.pop(f"{task}:evaluated_subset", None)
    reporting = candidate.summary["reporting"]
    prediction_root = candidate.root
    prediction = next(
        (
            item for item in reporting.get("predictions", ())
            if item.get("task") == task
        ),
        None,
    )
    if prediction is None:
        source_run = reporting.get("source_runs", {}).get("stage2_physics", {}).get(task)
        try:
            relative = Path(str(source_run)).relative_to(Path(candidate.source_run))
        except ValueError:
            relative = None
        if relative is not None:
            source_root = candidate.root / relative
            summary_path = source_root / "summary.json"
            if summary_path.is_file():
                source_summary = _json(summary_path)
                prediction = next(
                    (
                        item
                        for item in source_summary.get("reporting", {}).get("predictions", ())
                        if item.get("task") == task
                    ),
                    None,
                )
                prediction_root = source_root
    if not isinstance(prediction, Mapping) or not isinstance(prediction.get("path"), str):
        return str(identity["hash"])
    path = (prediction_root / prediction["path"]).resolve()
    if not _is_within(path, prediction_root.resolve()) or not path.is_file():
        return str(identity["hash"])
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _PARTIAL_CHARGE_PREDICTION_FIELDS:
            return str(identity["hash"])
        rows = [
            {
                name: row[name]
                for name in _PARTIAL_CHARGE_PREDICTION_FIELDS
                if name not in _PARTIAL_CHARGE_RUNTIME_FIELDS
            }
            for row in reader
        ]
    return semantic_identity(
        "reporting.partial-charge-comparison.v2",
        {
            "comparison": {
                "benchmark": payload["benchmark"],
                "split": payload["split"],
                "expected": payload["expected"],
                "sources": sources,
                "normalization": payload["normalization"],
                "folds": payload["folds"],
                "ensemble": payload["ensemble"],
            },
            "evaluation_rows": rows,
        },
    )["hash"]


def _stage2_full_comparison_key(
    candidate: Candidate,
    section: Mapping[str, Any],
    partial_section: Mapping[str, Any],
) -> str:
    identity = section["comparison_identity"]
    payload = dict(identity["payload"])
    component_hashes = dict(payload["component_hashes"])
    component_hashes["stage2_partial_charge"] = _partial_charge_comparison_key(
        candidate, partial_section
    )
    payload["component_hashes"] = component_hashes
    return semantic_identity("reporting.stage2-full-comparison.v2", payload)["hash"]


def _validate_stage2_suite(reporting: Mapping[str, Any]) -> None:
    capabilities = reporting.get("capabilities")
    sections = reporting.get("benchmarks")
    names = {
        "stage2_core_physics", "stage2_partial_charge", "stage2_physics_full"
    }
    if not isinstance(capabilities, dict) or set(capabilities) != names:
        raise ValueError("Stage 2 reporting capabilities are incomplete")
    if not isinstance(sections, dict):
        raise ValueError("Stage 2 reporting benchmarks are malformed")
    for name in names:
        capability = capabilities[name]
        status = sections[name].get("status")
        if capability not in {"supported", "unsupported"}:
            raise ValueError(f"Invalid Stage 2 capability: {name}")
        if status not in {"complete", "incomplete", "unsupported"}:
            raise ValueError(f"Invalid Stage 2 status: {name}")
        if (capability == "unsupported") != (status == "unsupported"):
            raise ValueError(f"Stage 2 capability/status mismatch: {name}")
    full = sections["stage2_physics_full"]
    if full["status"] == "complete":
        core = sections["stage2_core_physics"]
        partial = sections["stage2_partial_charge"]
        if core["status"] != "complete" or partial["status"] != "complete":
            raise ValueError("Stage 2 Full cannot outlive an incomplete component")
        payload = full["comparison_identity"]["payload"]
        if payload.get("component_hashes") != {
            "stage2_core_physics": core["comparison_identity"]["hash"],
            "stage2_partial_charge": partial["comparison_identity"]["hash"],
        }:
            raise ValueError("Stage 2 Full component identities are inconsistent")
        expected_units = [
            *core["protocol"].get("expected_tasks", ()),
            *partial["protocol"].get("expected_units", ()),
        ]
        if payload.get("ordered_units") != expected_units:
            raise ValueError("Stage 2 Full ordered units are inconsistent")


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
        current_stage2 = reporting.get("contract") == STAGE2_BENCHMARK_SUITE_CONTRACT
        current_sections = {
            "stage3_test", "stage3_validation", "stage2_core_physics",
            "stage2_partial_charge", "stage2_physics_full",
        }
        if current_stage2 and set(benchmarks or ()) != current_sections:
            raise ValueError("benchmark sweep reporting sections are incomplete")
        if not current_stage2 and not {
            "stage3_test", "stage3_validation"
        }.issubset(set(benchmarks or ())):
            raise ValueError("benchmark sweep Stage 3 reporting sections are incomplete")
        if current_stage2:
            _validate_stage2_suite(reporting)
        names_to_validate = benchmarks if current_stage2 else {
            name: benchmarks[name] for name in ("stage3_test", "stage3_validation")
        }
        for name, value in names_to_validate.items():
            if value.get("status") not in {"unsupported", "incomplete"}:
                _validate_comparison(value.get("comparison_identity"), name)
        if not isinstance(reporting.get("source_runs"), dict):
            raise ValueError("benchmark sweep reporting lacks source runs")
        if reporting.get("contract") == STAGE2_BENCHMARK_SUITE_CONTRACT:
            source_manifest = reporting.get("source_run_manifest")
            if not isinstance(source_manifest, dict):
                raise ValueError("benchmark sweep reporting lacks source-run manifest")
            validate_semantic_identity(source_manifest)
            if source_manifest.get("type") != "benchmark.source-run-manifest.v1":
                raise ValueError("benchmark sweep source-run manifest has wrong type")
            if source_manifest.get("payload", {}).get("source_runs") != reporting["source_runs"]:
                raise ValueError("benchmark sweep source-run manifest is inconsistent")
    elif stage == "stage2":
        if reporting.get("contract") != STAGE2_BENCHMARK_SUITE_CONTRACT:
            return
        benchmarks = reporting.get("benchmarks")
        if set(benchmarks or ()) != {
            "stage2_core_physics", "stage2_partial_charge", "stage2_physics_full"
        }:
            raise ValueError("Stage 2 reporting suite sections are incomplete")
        _validate_stage2_suite(reporting)
        if benchmarks["stage2_physics_full"]["status"] == "complete":
            protocols = [
                benchmarks[name]["protocol"]
                for name in (
                    "stage2_core_physics", "stage2_partial_charge",
                    "stage2_physics_full",
                )
            ]
            checkpoint_hashes = {item.get("checkpoint_sha256") for item in protocols}
            checkpoint_epochs = {item.get("checkpoint_epoch") for item in protocols}
            if (
                len(checkpoint_hashes) != 1 or None in checkpoint_hashes
                or len(checkpoint_epochs) != 1
            ):
                raise ValueError("ILUME Stage 2 suite checkpoint binding is inconsistent")
        for name, value in benchmarks.items():
            if value.get("status") not in {"unsupported", "incomplete"}:
                _validate_comparison(value.get("comparison_identity"), name)
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
    return display


def _model(candidate: Candidate) -> tuple[str, str]:
    if candidate.summary:
        reporting = candidate.summary.get("reporting", {})
        if reporting.get("model_id") and reporting.get("model_display_name"):
            return str(reporting["model_id"]), _display_name(candidate, reporting)
        if candidate.metadata.get("stage") == "benchmark" and candidate.summary.get("model"):
            model = str(candidate.summary["model"])
            return model, model
    return ("ilume", "ILUME") if candidate.metadata.get("stage") != "benchmark" else ("unknown", "unknown")


def _stage2_eligibility(reporting: Mapping[str, Any]) -> tuple[str, str, str]:
    if reporting.get("contract") != STAGE2_BENCHMARK_SUITE_CONTRACT:
        return ("legacy", "legacy", "legacy")
    sections = reporting["benchmarks"]
    values = []
    for name in (
        "stage2_core_physics", "stage2_partial_charge", "stage2_physics_full"
    ):
        status = sections[name]["status"]
        values.append(
            "eligible" if status == "complete" else
            "not_evaluated" if status == "unsupported" else
            "not_eligible"
        )
    return tuple(values)


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
                str((candidate.summary or {}).get("reporting", {}).get("protocol", {}).get("fold", provenance.get("fold", ""))),
            )
        )
        status = str(candidate.metadata.get("status", "unknown"))
        summary = candidate.summary or {}
        issues = list(candidate.issues)
        reporting = summary.get("reporting", {})
        has_stage2 = candidate.metadata.get("stage") == "stage2" or (
            candidate.metadata.get("stage") == "benchmark"
            and reporting.get("contract") == STAGE2_BENCHMARK_SUITE_CONTRACT
        )
        eligibility = (
            _stage2_eligibility(reporting) if has_stage2 and reporting else ("", "", "")
        )
        if has_stage2 and reporting and eligibility[0] == "legacy":
            issues.append("legacy_stage2_reporting_contract")
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
                    f"{name}:{len(value['protocol'].get('expected_tasks', ())) }"
                    for name, value in sorted(sections.items())
                )
                available_items = [
                    f"stage3_test:{len(summary['stage3_property_benchmark']['test_ensemble'])}",
                    f"stage3_validation:{len(summary['stage3_property_benchmark']['validation_five_fold'])}",
                ]
                if "stage2_physics_benchmark" in summary:
                    available_items.append(
                        f"stage2_physics:{len(summary['stage2_physics_benchmark']['test'])}"
                    )
                available = ";".join(available_items)
                folds = ";".join(map(str, sections["stage3_validation"]["protocol"]["folds"]))
            elif candidate.metadata["stage"] == "stage3":
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
            elif reporting.get("contract") == STAGE2_BENCHMARK_SUITE_CONTRACT:
                protocol = reporting["benchmarks"]["stage2_core_physics"]["protocol"]
                expected = len(protocol.get("expected_tasks", ()))
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
                "stage2_core_eligibility": eligibility[0],
                "stage2_partial_eligibility": eligibility[1],
                "stage2_full_eligibility": eligibility[2],
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

            if reporting.get("contract") == STAGE2_BENCHMARK_SUITE_CONTRACT:
                stage2_expected = tuple(
                    sections["stage2_core_physics"]["protocol"]["expected_tasks"]
                )
                stage2_metrics = summary["stage2_physics_benchmark"]["test"]
                stage2_available = {
                    task
                    for task, targets in stage2_metrics.items()
                    for target, values in targets.items()
                    if int(values.get("count", 0)) > 0
                }
                stage2_missing = sorted(set(stage2_expected) - stage2_available)
                if stage2_missing:
                    mark_incomplete(
                        candidate,
                        "stage2_missing_tasks=" + ",".join(stage2_missing),
                    )
                for name in (
                    "stage2_core_physics", "stage2_partial_charge",
                    "stage2_physics_full",
                ):
                    section = sections[name]
                    if section["status"] == "incomplete":
                        for issue in section.get("issues", ("incomplete",)):
                            mark_incomplete(candidate, f"{name}:{issue}")
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
            if reporting.get("contract") != STAGE2_BENCHMARK_SUITE_CONTRACT:
                continue
            sections = reporting["benchmarks"]
            expected_tasks = tuple(
                sections["stage2_core_physics"]["protocol"]["expected_tasks"]
            )
            available_tasks = {
                task
                for task, targets in summary.get("tasks", {}).items()
                if task != "simulation/partial_atomic_charge"
                for target, values in targets.items()
                if int(values.get("count", 0)) > 0
            }
            missing = sorted(set(expected_tasks) - available_tasks)
            if missing:
                mark_incomplete(
                    candidate, "missing_tasks=" + ",".join(missing)
                )
            for name in (
                "stage2_core_physics", "stage2_partial_charge",
                "stage2_physics_full",
            ):
                section = sections[name]
                if section["status"] == "incomplete":
                    for issue in section.get("issues", ("incomplete",)):
                        mark_incomplete(candidate, f"{name}:{issue}")

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


def _stage2_core(
    candidates: Sequence[Candidate],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    leaders: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    comparisons: dict[str, list[str]] = {}
    for candidate in _current_completed(candidates):
        summary = candidate.summary
        reporting = summary["reporting"]
        if reporting.get("contract") != STAGE2_BENCHMARK_SUITE_CONTRACT:
            continue
        if candidate.metadata["stage"] == "benchmark":
            section = reporting["benchmarks"]["stage2_core_physics"]
            metrics = summary["stage2_physics_benchmark"]["test"]
            checkpoint = ""
        elif candidate.metadata["stage"] == "stage2":
            section = reporting["benchmarks"]["stage2_core_physics"]
            metrics = summary["tasks"]
            checkpoint = summary["checkpoint_epoch"]
        else:
            continue
        if section.get("status") != "complete":
            continue
        expected = tuple(section["protocol"]["expected_tasks"])
        flattened: dict[str, tuple[str, Mapping[str, Any]]] = {}
        for task in expected:
            targets = metrics.get(task, {})
            if len(targets) == 1:
                target, value = next(iter(targets.items()))
                flattened[task] = (target, value)
        if not expected or any(
            task not in flattened
            or int(flattened[task][1].get("count", 0)) <= 0
            or not _finite(flattened[task][1].get("normalized_mae"))
            for task in expected
        ):
            continue
        model_id = reporting["model_id"]
        display = reporting["model_display_name"]
        run = _run_id(model_id, candidate.source_run)
        for task in expected:
            target, value = flattened[task]
            metrics_rows.append(
                {
                    "run": run, "model": display, "task": task, "target": target,
                    "subset": "pooled",
                    "count": value["count"], "mae": value["mae"],
                    "rmse": value["rmse"], "r2": value["r2"],
                    "normalized_mae": value["normalized_mae"],
                    "normalized_rmse": value["normalized_rmse"],
                    "source_run": candidate.source_run,
                }
            )
            diagnostics = value.get("role_diagnostics", {})
            for role in ("cation", "anion"):
                if role not in diagnostics:
                    continue
                role_value = diagnostics[role]
                metrics_rows.append(
                    {
                        "run": run,
                        "model": display,
                        "task": task,
                        "target": target,
                        "subset": role,
                        "count": role_value["count"],
                        "mae": role_value["mae"],
                        "source_run": candidate.source_run,
                    }
                )
        values = [
            float(flattened[task][1]["normalized_mae"])
            for task in expected
        ]
        comparisons.setdefault(section["comparison_identity"]["hash"], []).append(run)
        leaders.append(
            {
                "run": run, "model": display,
                "macro_normalized_mae": sum(values) / len(values),
                "valid_tasks": len(values), "total_tasks": len(expected),
                "per_task_wins": 0, "source_run": candidate.source_run,
                "checkpoint_epoch": checkpoint,
            }
        )
    _require_one_comparison(comparisons, "Stage 2 Core physics")
    pooled_rows = [row for row in metrics_rows if row["subset"] == "pooled"]
    wins = _wins(pooled_rows, "task", "normalized_mae")
    for row in leaders:
        row["per_task_wins"] = len(wins.get(row["run"], ()))
    return _rank(leaders, "macro_normalized_mae"), sorted(
        metrics_rows,
        key=lambda row: (
            row["run"], row["task"], row["target"], row["subset"]
        ),
    ), wins


def _partial_metrics(summary: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if "tasks" in summary:
        value = summary["tasks"].get("simulation/partial_atomic_charge")
        return value if isinstance(value, dict) else None
    value = summary.get("stage2_partial_charge_benchmark", {}).get("test")
    return value if isinstance(value, dict) else None


def _stage2_partial(
    candidates: Sequence[Candidate],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    leaders: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    comparisons: dict[str, list[str]] = {}
    for candidate in _current_completed(candidates):
        reporting = candidate.summary["reporting"]
        if reporting.get("contract") != STAGE2_BENCHMARK_SUITE_CONTRACT:
            continue
        section = reporting["benchmarks"]["stage2_partial_charge"]
        if section["status"] != "complete":
            continue
        metrics = _partial_metrics(candidate.summary)
        if metrics is None or metrics.get("status") != "complete":
            continue
        primary = metrics.get("primary") or {}
        if not _finite(primary.get("molecule_macro_normalized_mae")):
            continue
        model_id = reporting["model_id"]
        display = reporting["model_display_name"]
        run = _run_id(model_id, candidate.source_run)
        for subset in ("all_mapped", "unique", "ambiguous", "typed", "connectivity_only"):
            value = metrics.get("subsets", {}).get(subset)
            if not isinstance(value, dict):
                break
            metrics_rows.append(
                {
                    "run": run, "model": display, "subset": subset,
                    **{
                        name: value.get(name)
                        for name in (
                            "molecule_count", "atom_count", "molecule_macro_mae",
                            "molecule_macro_normalized_mae", "atom_micro_mae",
                            "atom_micro_rmse", "atom_micro_r2", "atom_micro_r2_reason",
                            "reason",
                        )
                    },
                    "source_run": candidate.source_run,
                }
            )
        else:
            coverage = metrics.get("coverage", {})
            leaders.append(
                {
                    "run": run, "model": display,
                    "molecule_macro_normalized_mae": float(
                        primary["molecule_macro_normalized_mae"]
                    ),
                    "molecule_macro_mae": primary["molecule_macro_mae"],
                    "mapped_molecules": coverage.get("mapped_molecule_count", ""),
                    "test_molecules": coverage.get("test_molecule_count", ""),
                    "source_run": candidate.source_run,
                    "checkpoint_epoch": candidate.summary.get("checkpoint_epoch", ""),
                }
            )
            comparisons.setdefault(
                _partial_charge_comparison_key(candidate, section), []
            ).append(run)
            continue
        metrics_rows = [row for row in metrics_rows if row["run"] != run]
    _require_one_comparison(comparisons, "Stage 2 Partial Charge")
    return _rank(leaders, "molecule_macro_normalized_mae"), sorted(
        metrics_rows, key=lambda row: (row["run"], row["subset"])
    )


def _stage2_full(candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
    leaders: list[dict[str, Any]] = []
    comparisons: dict[str, list[str]] = {}
    for candidate in _current_completed(candidates):
        summary = candidate.summary
        reporting = summary["reporting"]
        if reporting.get("contract") != STAGE2_BENCHMARK_SUITE_CONTRACT:
            continue
        section = reporting["benchmarks"]["stage2_physics_full"]
        if section["status"] != "complete":
            continue
        core_section = reporting["benchmarks"]["stage2_core_physics"]
        partial_section = reporting["benchmarks"]["stage2_partial_charge"]
        if core_section["status"] != "complete" or partial_section["status"] != "complete":
            continue
        expected = tuple(core_section["protocol"]["expected_tasks"])
        core_metrics = (
            summary["stage2_physics_benchmark"]["test"]
            if candidate.metadata["stage"] == "benchmark"
            else summary["tasks"]
        )
        flattened = {
            task: next(iter(targets.values()))
            for task, targets in core_metrics.items()
            if task != "simulation/partial_atomic_charge" and len(targets) == 1
        }
        partial = _partial_metrics(summary)
        primary = None if partial is None else partial.get("primary")
        if (
            not expected
            or any(
                unit not in flattened
                or not _finite(flattened[unit].get("normalized_mae"))
                for unit in expected
            )
            or not isinstance(primary, dict)
            or not _finite(primary.get("molecule_macro_normalized_mae"))
        ):
            continue
        values = [float(flattened[unit]["normalized_mae"]) for unit in expected]
        values.append(float(primary["molecule_macro_normalized_mae"]))
        model_id = reporting["model_id"]
        run = _run_id(model_id, candidate.source_run)
        leaders.append(
            {
                "run": run, "model": reporting["model_display_name"],
                "macro_normalized_mae": sum(values) / len(values),
                "valid_units": len(values), "total_units": len(values),
                "source_run": candidate.source_run,
                "checkpoint_epoch": summary.get("checkpoint_epoch", ""),
            }
        )
        comparisons.setdefault(
            _stage2_full_comparison_key(candidate, section, partial_section), []
        ).append(run)
    _require_one_comparison(comparisons, "Stage 2 Full physics")
    return _rank(leaders, "macro_normalized_mae")


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


def _overview(
    stage3_test: Sequence[Mapping[str, Any]],
    stage3_validation: Sequence[Mapping[str, Any]],
    stage2_core: Sequence[Mapping[str, Any]],
    stage2_partial: Sequence[Mapping[str, Any]],
    stage2_full: Sequence[Mapping[str, Any]],
    stage2_wins: Mapping[str, Sequence[str]],
    health: Sequence[Mapping[str, Any]],
) -> str:
    lines = ["# ILUME result overview", ""]
    for title, rows in (
        ("Stage 3 TEST (5-fold ensemble)", stage3_test),
        ("Stage 3 VALIDATION (5-fold mean)", stage3_validation),
        ("Stage 2 CORE", stage2_core),
        ("Partial Charge", stage2_partial),
        ("Stage 2 FULL", stage2_full),
    ):
        lines.extend((f"## {title}", ""))
        if rows:
            for row in rows:
                lines.append(
                    f"{row['rank']}. {row['model']} — macro normalized MAE "
                    f"{float(row.get('macro_normalized_mae', row.get('molecule_macro_normalized_mae'))):.6g}"
                )
        else:
            lines.append("No eligible run.")
        if title.startswith("Stage 3 TEST") and rows:
            lines.append(
                f"Coverage: {rows[0]['total_tasks']} test tasks / "
                f"{rows[0]['enabled_tasks']} enabled Stage 3 tasks."
            )
        if title in {"Partial Charge", "Stage 2 FULL"}:
            not_evaluated = [
                row["run"] for row in health
                if row[
                    "stage2_partial_eligibility"
                    if title == "Partial Charge" else "stage2_full_eligibility"
                ] == "not_evaluated"
            ]
            not_eligible = [
                row["run"] for row in health
                if row[
                    "stage2_partial_eligibility"
                    if title == "Partial Charge" else "stage2_full_eligibility"
                ] == "not_eligible"
            ]
            lines.append(
                "Not evaluated: " + (", ".join(not_evaluated) if not_evaluated else "None")
            )
            lines.append(
                "Not eligible: " + (", ".join(not_eligible) if not_eligible else "None")
            )
        lines.append("")
    lines.extend(("## Core task wins", ""))
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
    partial_rows: Sequence[Mapping[str, Any]] = (),
    partial_metric_key: str | None = None,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    model_by_run = {str(row["run"]): str(row["model"]) for row in leaders}
    ordered_runs = [str(row["run"]) for row in leaders]
    values: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.get("run") not in model_by_run or not _finite(row.get(metric_key)):
            continue
        values[str(row["run"]), str(row[task_key])] = float(row[metric_key])
    for row in partial_rows:
        if row.get("run") not in model_by_run or row.get("subset") != "all_mapped":
            continue
        value = row.get(partial_metric_key or metric_key)
        if _finite(value):
            values[str(row["run"]), "simulation/partial_atomic_charge"] = float(value)
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
    stage2_tasks = (
        "simulation/heat_of_vaporization",
        "simulation/homo",
        "simulation/lumo",
        "simulation/partial_atomic_charge",
    )
    panels = (
        (
            "stage2_test", "Stage 2 test", *_radar_panel_data(
                metrics["stage2_core_physics"],
                leaderboards["stage2_core_physics"],
                task_key="task",
                metric_key="normalized_mae",
                tasks=stage2_tasks,
                partial_rows=metrics["stage2_partial_charge"],
                partial_metric_key="molecule_macro_normalized_mae",
            ),
        ),
        (
            "stage3_test", "Stage 3 test", *_radar_panel_data(
                metrics["stage3_test"], leaderboards["stage3_test"],
                task_key="task", metric_key="normalized_mae",
            ),
        ),
        (
            "stage3_validation", "Stage 3 validation", *_radar_panel_data(
                metrics["stage3_validation"], leaderboards["stage3_validation"],
                task_key="task", metric_key="normalized_mae_mean",
            ),
        ),
    )
    palette = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9")
    models = sorted({series["model"] for _, _, _, items in panels for series in items})
    colors = {model: palette[index % len(palette)] for index, model in enumerate(models)}
    centers = ((330.0, 570.0), (960.0, 570.0), (1590.0, 570.0))
    radius = 240.0
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="900" viewBox="0 0 1920 900">',
        '<rect width="1920" height="900" fill="white"/>',
        '<text x="960" y="42" text-anchor="middle" font-family="sans-serif" font-size="26" font-weight="bold">ILUME benchmark radar scores</text>',
        '<text x="960" y="70" text-anchor="middle" font-family="sans-serif" font-size="14">Score = best normalized MAE / model normalized MAE; higher is better. Unsupported Stage 2 partial charge = 0.</text>',
    ]
    for (panel_id, title, tasks, series), (center_x, center_y) in zip(panels, centers, strict=True):
        lines.append(
            f'<text x="{center_x:.1f}" y="122" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">{_svg_text(title)}</text>'
        )
        if not tasks or not series:
            lines.append(
                f'<text x="{center_x:.1f}" y="{center_y:.1f}" text-anchor="middle" font-family="sans-serif" font-size="16">No eligible run.</text>'
            )
            continue
        angles = tuple(2.0 * math.pi * index / len(tasks) for index in range(len(tasks)))
        for fraction in (0.25, 0.5, 0.75, 1.0):
            points = " ".join(
                f"{x:.2f},{y:.2f}"
                for x, y in (
                    _radar_point(center_x, center_y, radius * fraction, angle)
                    for angle in angles
                )
            )
            lines.append(
                f'<polygon points="{points}" fill="none" stroke="#c7c7c7" stroke-width="1"/>'
            )
            lines.append(
                f'<text x="{center_x + 5:.2f}" y="{center_y - radius * fraction + 4:.2f}" font-family="sans-serif" font-size="9" fill="#666">{fraction:g}</text>'
            )
        lines.append(
            f'<text x="{center_x + 5:.2f}" y="{center_y + 4:.2f}" font-family="sans-serif" font-size="9" fill="#666">0</text>'
        )
        for task, angle in zip(tasks, angles, strict=True):
            x, y = _radar_point(center_x, center_y, radius, angle)
            label_x, label_y = _radar_point(center_x, center_y, radius + 24.0, angle)
            anchor = "middle" if abs(math.sin(angle)) < 0.2 else ("start" if math.sin(angle) > 0 else "end")
            label = task.rsplit("/", 1)[-1].replace("_", " ")
            lines.extend((
                f'<line x1="{center_x:.2f}" y1="{center_y:.2f}" x2="{x:.2f}" y2="{y:.2f}" stroke="#c7c7c7" stroke-width="1"/>',
                f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="{anchor}" font-family="sans-serif" font-size="10">{_svg_text(label)}</text>',
            ))
        for index, series_item in enumerate(series):
            points = " ".join(
                f"{x:.2f},{y:.2f}"
                for x, y in (
                    _radar_point(center_x, center_y, radius * score, angle)
                    for score, angle in zip(series_item["scores"], angles, strict=True)
                )
            )
            score_text = ",".join(f"{score:.6g}" for score in series_item["scores"])
            color = colors[series_item["model"]]
            lines.append(
                f'<polygon class="series" data-panel="{panel_id}" data-run="{_svg_text(series_item["run"])}" data-scores="{score_text}" points="{points}" fill="{color}" fill-opacity="0.12" stroke="{color}" stroke-width="2"/>'
            )
            legend_y = 160 + index * 20
            lines.extend((
                f'<line x1="{center_x - radius:.1f}" y1="{legend_y}" x2="{center_x - radius + 18:.1f}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{center_x - radius + 24:.1f}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{_svg_text(series_item["model"])}</text>',
            ))
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
    stage3_test, stage3_test_metrics, test_wins = _stage3_test(candidates)
    stage3_validation, stage3_validation_metrics = _stage3_validation(candidates)
    stage2_core, stage2_core_metrics, stage2_wins = _stage2_core(candidates)
    stage2_partial, stage2_partial_metrics = _stage2_partial(candidates)
    stage2_full = _stage2_full(candidates)
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
            "stage2_core_physics": stage2_core,
            "stage2_partial_charge": stage2_partial,
            "stage2_physics_full": stage2_full,
        },
        "metrics": {
            "stage3_test": stage3_test_metrics,
            "stage3_validation": stage3_validation_metrics,
            "stage2_core_physics": stage2_core_metrics,
            "stage2_partial_charge": stage2_partial_metrics,
        },
        "wins": {"stage3_test": test_wins, "stage2_core_physics": stage2_wins},
        "health": health,
    }


def _comparison_catalog(
    candidates: Sequence[Candidate],
) -> dict[str, list[dict[str, Any]]]:
    catalog: dict[str, dict[str, dict[str, Any]]] = {
        "stage3_test": {},
        "stage3_validation": {},
        "stage2_core_physics": {},
        "stage2_partial_charge": {},
        "stage2_physics_full": {},
    }
    sources: dict[tuple[str, str], list[str]] = {}
    for candidate in _current_completed(candidates):
        reporting = candidate.summary["reporting"]
        sections: Iterable[tuple[str, Mapping[str, Any]]]
        if candidate.metadata["stage"] == "benchmark":
            sections = (
                reporting["benchmarks"].items()
                if reporting.get("contract") == STAGE2_BENCHMARK_SUITE_CONTRACT
                else (
                    ("stage3_test", reporting["benchmarks"]["stage3_test"]),
                    ("stage3_validation", reporting["benchmarks"]["stage3_validation"]),
                )
            )
        elif candidate.metadata["stage"] == "stage3":
            name = (
                "stage3_test"
                if candidate.summary.get("split") == "test"
                else "stage3_validation"
            )
            sections = ((name, reporting),)
        elif reporting.get("contract") == STAGE2_BENCHMARK_SUITE_CONTRACT:
            sections = reporting["benchmarks"].items()
        else:
            sections = ()
        for name, section in sections:
            if section.get("status") in {"unsupported", "incomplete"}:
                continue
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
        destination / "stage2_core_physics_leaderboard.csv",
        leaderboards["stage2_core_physics"],
        ("rank", "run", "model", "macro_normalized_mae", "valid_tasks", "total_tasks", "per_task_wins", "source_run", "checkpoint_epoch"),
    )
    _write_csv(
        destination / "stage2_partial_charge_leaderboard.csv",
        leaderboards["stage2_partial_charge"],
        ("rank", "run", "model", "molecule_macro_normalized_mae", "molecule_macro_mae", "mapped_molecules", "test_molecules", "source_run", "checkpoint_epoch"),
    )
    _write_csv(
        destination / "stage2_physics_full_leaderboard.csv",
        leaderboards["stage2_physics_full"],
        ("rank", "run", "model", "macro_normalized_mae", "valid_units", "total_units", "source_run", "checkpoint_epoch"),
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
        destination / "stage2_core_physics_metrics.csv",
        metrics["stage2_core_physics"],
        ("run", "model", "task", "target", "subset", "count", "mae", "rmse", "r2", "normalized_mae", "normalized_rmse", "source_run"),
    )
    _write_csv(
        destination / "stage2_partial_charge_metrics.csv",
        metrics["stage2_partial_charge"],
        ("run", "model", "subset", "molecule_count", "atom_count", "molecule_macro_mae", "molecule_macro_normalized_mae", "atom_micro_mae", "atom_micro_rmse", "atom_micro_r2", "atom_micro_r2_reason", "reason", "source_run"),
    )
    _write_csv(
        destination / "sweep_status.csv",
        payload["health"],
        ("run", "model", "stage", "operation", "status", "completeness", "checkpoint_epoch", "enabled_tasks", "expected_tasks", "available_tasks", "available_folds", "failed_jobs", "stage2_core_eligibility", "stage2_partial_eligibility", "stage2_full_eligibility", "source_run", "issues"),
    )
    (destination / "overview.md").write_text(
        _overview(
            leaderboards["stage3_test"], leaderboards["stage3_validation"],
            leaderboards["stage2_core_physics"],
            leaderboards["stage2_partial_charge"],
            leaderboards["stage2_physics_full"],
            payload["wins"]["stage2_core_physics"], payload["health"],
        ),
        encoding="utf-8",
    )
    (destination / "radar.svg").write_text(_radar_svg(payload), encoding="utf-8")
    atomic_json(destination / "summary.json", payload)
    actual = tuple(sorted(path.name for path in destination.iterdir()))
    if actual != tuple(sorted(SUMMARY_FILES)):
        raise AssertionError(f"Summary snapshot file set mismatch: {actual}")


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
