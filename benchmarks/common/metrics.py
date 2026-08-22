from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def regression_metrics(predictions: np.ndarray, targets: np.ndarray, *, scale: float | None = None) -> dict[str, Any]:
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    actual = np.asarray(targets, dtype=np.float64).reshape(-1)
    if predicted.shape != actual.shape or not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("Benchmark predictions and targets must be matching finite vectors")
    count = len(actual)
    if count == 0:
        return {"count": 0, "reason": "no_samples"}
    delta = predicted - actual
    denominator = float(np.square(actual - actual.mean()).sum())
    result: dict[str, Any] = {
        "count": count,
        "mae": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.square(delta).mean())),
        "r2": float("nan") if denominator == 0 else 1.0 - float(np.square(delta).sum()) / denominator,
        "r2_reason": "constant_target" if denominator == 0 else None,
    }
    if scale is not None:
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Normalized benchmark metrics require a positive finite scale")
        result["normalized_mae"] = result["mae"] / scale
        result["normalized_rmse"] = result["rmse"] / scale
    return result


def target_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_columns: Sequence[str],
    scales: Sequence[float] | None,
) -> dict[str, dict[str, Any]]:
    predicted = np.asarray(predictions)
    actual = np.asarray(targets)
    if predicted.ndim == 1:
        predicted = predicted[:, None]
    if actual.ndim == 1:
        actual = actual[:, None]
    if predicted.shape != actual.shape or predicted.shape[1] != len(target_columns):
        raise ValueError("Benchmark target metric shape mismatch")
    return {
        name: regression_metrics(
            predicted[:, index], actual[:, index],
            scale=None if scales is None else float(scales[index]),
        )
        for index, name in enumerate(target_columns)
    }


def mean_sample_std(values: Sequence[float]) -> dict[str, float | int]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return {
        "mean": float(finite.mean()) if len(finite) else float("nan"),
        "std": float(finite.std(ddof=1)) if len(finite) >= 2 else float("nan"),
        "count": int(len(finite)),
    }


def macro_normalized_mae(per_task: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(metrics["normalized_mae"]) for metrics in per_task.values() if math.isfinite(float(metrics.get("normalized_mae", float("nan"))))]
    return {
        "value": sum(values) / len(values) if values else float("nan"),
        "valid_tasks": len(values),
        "total_tasks": len(per_task),
    }


__all__ = ["macro_normalized_mae", "mean_sample_std", "regression_metrics", "target_metrics"]

