"""Shared scalar simulation comparison contract for ILUME and baselines."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common.identity import semantic_identity
from common.io import sha256_file
from common.reporting import comparison_identity

SCALAR_SIMULATION_TASKS = (
    "simulation/heat_of_vaporization", "simulation/thermal_expansion",
    "simulation/homo", "simulation/lumo",
)


ENTITY_SCALAR_SIMULATION_TASKS = SCALAR_SIMULATION_TASKS[:2]


def scalar_simulation_tasks(config: Any) -> tuple[str, ...]:
    return ENTITY_SCALAR_SIMULATION_TASKS if config.is_entity_home else SCALAR_SIMULATION_TASKS


def simulation_scale(train_path: Path, target_column: str) -> float:
    with train_path.open(newline="", encoding="utf-8-sig") as handle:
        values = np.asarray([float(row[target_column]) for row in csv.DictReader(handle)], dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Simulation normalization requires finite train targets")
    scale = float(values.std())
    return scale if scale > 0 else 1.0


def simulation_comparison(
    *, split: str, tasks: Sequence[str],
    sources: Mapping[str, Any], scales: Mapping[str, float],
) -> dict[str, Any]:
    return comparison_identity(
        "simulation_property", split=split, expected=tasks,
        sources=sources, normalization={task: {"scale": scales[task]} for task in tasks},
    )


def simulation_task_sources(task: str, train_path: Path, split_path: Path, rows: Sequence[str]) -> dict[str, Any]:
    with split_path.open(newline="", encoding="utf-8-sig") as handle:
        expected_rows = [f"{split_path.as_posix()}:{number}" for number, _ in enumerate(csv.DictReader(handle), start=2)]
    if list(rows) != expected_rows or not rows:
        raise ValueError(f"Simulation evaluation row coverage differs from its raw split: {task}")
    return {
        f"{task}:train": sha256_file(train_path),
        f"{task}:evaluation": sha256_file(split_path),
        f"{task}:rows": semantic_identity("simulation.source-rows.v1", {"rows": list(rows)})["hash"],
    }


def scalar_split_rows(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if set(columns) - set(reader.fieldnames or ()):
            raise ValueError(f"Simulation source lacks required columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Simulation split is empty: {path}")
    return rows


def scalar_metrics(target: np.ndarray, prediction: np.ndarray, scale: float) -> dict[str, Any]:
    actual = np.asarray(target, dtype=np.float64).reshape(-1)
    predicted = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape or not len(actual) or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Simulation metrics require matching finite predictions and targets")
    error = predicted - actual
    denominator = float(np.square(actual - actual.mean()).sum())
    mae = float(np.abs(error).mean())
    rmse = float(np.sqrt(np.square(error).mean()))
    return {"count": len(actual), "mae": mae, "rmse": rmse,
            "r2": None if denominator == 0 else 1 - float(np.square(error).sum()) / denominator,
            "normalized_mae": mae / scale, "normalized_rmse": rmse / scale}
