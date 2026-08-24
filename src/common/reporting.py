from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .identity import semantic_identity
from .io import sha256_file


REPORTING_SCHEMA_VERSION = 1


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
    if benchmark not in {"stage2_physics", "stage3_property"}:
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
    "comparison_identity",
    "reporting_block",
    "sanitize_task_id",
    "write_prediction_csv",
]
