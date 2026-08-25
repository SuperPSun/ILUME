from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .identity import semantic_identity
from .io import sha256_file


REPORTING_SCHEMA_VERSION = 1
STAGE2_BENCHMARK_SUITE_CONTRACT = "stage2-benchmark-suite-v2"
STAGE2_CORE_EVALUATION_CONTRACT = "stage2-core-evaluation-v2"


def role_mae_diagnostics(
    predicted: Sequence[float],
    actual: Sequence[float],
    roles: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    if not (len(predicted) == len(actual) == len(roles)) or not roles:
        raise ValueError("Role diagnostics require matching non-empty vectors")
    result: dict[str, dict[str, float | int]] = {}
    for role in ("cation", "anion"):
        errors = [
            abs(float(prediction) - float(target))
            for prediction, target, row_role in zip(
                predicted, actual, roles, strict=True
            )
            if row_role == role
        ]
        if not errors or not all(math.isfinite(value) for value in errors):
            raise ValueError(f"Role diagnostics require finite {role} rows")
        result[role] = {
            "count": len(errors),
            "mae": sum(errors) / len(errors),
        }
    if set(roles) != set(result):
        raise ValueError("Role diagnostics contain an unsupported ion_role")
    return result


def sanitize_task_id(task_id: str) -> str:
    if not task_id or any(part in {"", ".", ".."} for part in task_id.split("/")):
        raise ValueError(f"Invalid reporting task id: {task_id!r}")
    return task_id.replace("/", "__")


def comparison_identity(
    benchmark: str,
    *,
    split: str,
    expected: Sequence[str],
    sources: Mapping[str, Any],
    normalization: Mapping[str, Any],
    folds: Sequence[int] = (),
    ensemble: bool = False,
) -> dict[str, Any]:
    if benchmark not in {
        "stage2_physics", "stage2_partial_charge", "stage3_property"
    }:
        raise ValueError(f"Unsupported reporting benchmark: {benchmark}")
    if split not in {"valid", "test"} or not expected:
        raise ValueError("Reporting comparison requires a valid split and expected set")
    canonical_normalization = {
        key: {
            name: (
                float(format(float(value), ".12g"))
                if name == "scale"
                else value
            )
            for name, value in dict(stats).items()
        }
        for key, stats in normalization.items()
    }
    return semantic_identity(
        "reporting.comparison.v1",
        {
            "benchmark": benchmark,
            "split": split,
            "expected": list(expected),
            "sources": dict(sources),
            "normalization": canonical_normalization,
            "folds": list(folds),
            "ensemble": ensemble,
        },
    )


def reporting_block(
    *,
    model_id: str,
    model_display_name: str,
    benchmark: str,
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    study_id: str,
    predictions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if not model_id or not model_display_name or not study_id:
        raise ValueError("Reporting model and study identifiers must be non-empty")
    if comparison.get("type") != "reporting.comparison.v1":
        raise ValueError("Reporting comparison identity has the wrong type")
    return {
        "schema_version": REPORTING_SCHEMA_VERSION,
        "model_id": model_id,
        "model_display_name": model_display_name,
        "benchmark": benchmark,
        "protocol": dict(protocol),
        "comparison_identity": dict(comparison),
        "study_id": study_id,
        "predictions": [dict(item) for item in predictions],
    }


def stage2_full_comparison_identity(
    core: Mapping[str, Any],
    partial_charge: Mapping[str, Any],
    *,
    ordered_units: Sequence[str],
) -> dict[str, Any]:
    for name, identity in (("core", core), ("partial_charge", partial_charge)):
        if identity.get("type") != "reporting.comparison.v1":
            raise ValueError(f"Stage 2 Full {name} comparison has the wrong type")
    if not ordered_units or len(ordered_units) != len(set(ordered_units)):
        raise ValueError("Stage 2 Full units must be non-empty and unique")
    return semantic_identity(
        "reporting.comparison.v1",
        {
            "benchmark": "stage2_physics_full",
            "split": "test",
            "component_hashes": {
                "stage2_core_physics": core["hash"],
                "stage2_partial_charge": partial_charge["hash"],
            },
            "ordered_units": list(ordered_units),
            "unit_weighting": "equal",
        },
    )


def write_prediction_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> dict[str, Any]:
    destination = Path(path)
    if not fieldnames or len(fieldnames) != len(set(fieldnames)):
        raise ValueError("Prediction CSV fields must be non-empty and unique")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                if set(row) != set(fieldnames):
                    raise ValueError(
                        f"Prediction row columns differ from schema for {destination}"
                    )
                materialized = {
                    name: _csv_value(row[name], context=f"{destination}/{name}")
                    for name in fieldnames
                }
                writer.writerow(materialized)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": destination.as_posix(),
        "rows": len(rows),
        "sha256": sha256_file(destination),
    }


def _csv_value(value: Any, *, context: str) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite prediction CSV value: {context}")
        return format(value, ".17g")
    return value


__all__ = [
    "REPORTING_SCHEMA_VERSION",
    "STAGE2_BENCHMARK_SUITE_CONTRACT",
    "STAGE2_CORE_EVALUATION_CONTRACT",
    "comparison_identity",
    "reporting_block",
    "role_mae_diagnostics",
    "sanitize_task_id",
    "stage2_full_comparison_identity",
    "write_prediction_csv",
]
