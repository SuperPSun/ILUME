from __future__ import annotations

import math
from typing import Any, Mapping


TASKWISE_REFINED_FORMAT_VERSION = 1


def refinement_geometry(total_epochs: int, ratio: float) -> tuple[int, int]:
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if not 0.0 < ratio < 1.0:
        raise ValueError("refinement_ratio must be in (0, 1)")
    refinement_epochs = math.ceil(total_epochs * ratio)
    boundary_epoch = total_epochs - refinement_epochs
    if boundary_epoch < 1:
        raise ValueError("refinement must leave at least one joint-training epoch")
    return boundary_epoch, refinement_epochs


def refinement_cosine_factor(
    completed_updates: int, total_updates: int, min_ratio: float = 0.0
) -> float:
    if total_updates <= 0:
        raise ValueError("total_updates must be positive")
    if not 0 <= completed_updates <= total_updates:
        raise ValueError("completed_updates must be in [0, total_updates]")
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError("min_ratio must be in [0, 1]")
    progress = completed_updates / total_updates
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def selection_record(
    *,
    metric_name: str,
    boundary_epoch: int,
    boundary_metric: float,
    selected_epoch: int,
    best_metric: float,
) -> dict[str, Any]:
    values = (boundary_metric, best_metric)
    if not metric_name or any(not math.isfinite(value) for value in values):
        raise ValueError("Task-wise selection metrics must be named and finite")
    return {
        "metric": metric_name,
        "direction": "min",
        "boundary_epoch": boundary_epoch,
        "boundary_metric": boundary_metric,
        "selected_epoch": selected_epoch,
        "best_metric": best_metric,
        "improved": best_metric < boundary_metric,
    }


def require_taskwise_refined_payload(
    payload: Mapping[str, Any], *, kind: str
) -> None:
    if payload.get("kind") != kind:
        raise ValueError(f"Unexpected task-wise refined artifact kind: {payload.get('kind')!r}")
    if payload.get("format_version") != TASKWISE_REFINED_FORMAT_VERSION:
        raise ValueError("Unsupported task-wise refined artifact format")
    required = {
        "model",
        "model_state_hash",
        "training_identity",
        "boundary_epoch",
        "shared_state_hash",
        "private_state_hashes",
        "selected_tasks",
        "validation",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("Task-wise refined artifact is incomplete: " + ", ".join(missing))


__all__ = [
    "TASKWISE_REFINED_FORMAT_VERSION",
    "refinement_cosine_factor",
    "refinement_geometry",
    "require_taskwise_refined_payload",
    "selection_record",
]
