from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeaturePreprocessor:
    finite_mask: tuple[bool, ...]
    median: tuple[float, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, train: np.ndarray) -> "FeaturePreprocessor":
        if train.ndim != 2 or train.shape[0] == 0:
            raise ValueError("Feature matrix must be non-empty and rank two")
        finite_mask_array = np.isfinite(train).any(axis=0)
        if not finite_mask_array.any():
            raise ValueError("Every feature column is invalid in train")
        retained = np.where(
            np.isfinite(train[:, finite_mask_array]),
            train[:, finite_mask_array],
            np.nan,
        )
        median = np.nanmedian(retained, axis=0)
        filled = np.where(np.isfinite(retained), retained, median)
        mean = filled.mean(axis=0)
        scale = filled.std(axis=0)

        min_scale = 1e-8 * np.maximum(1.0, np.abs(mean))
        scale = np.where(
            np.isfinite(scale) & (scale > min_scale),
            scale,
            1.0,
        )
        return cls(
            finite_mask=tuple(bool(value) for value in finite_mask_array),
            median=tuple(float(value) for value in median),
            mean=tuple(float(value) for value in mean),
            scale=tuple(float(value) for value in scale),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        mask = np.asarray(self.finite_mask, dtype=bool)
        if values.ndim != 2 or values.shape[1] != len(mask):
            raise ValueError("Feature width differs from preprocessing contract")
        retained = values[:, mask]
        median = np.asarray(self.median)
        filled = np.where(np.isfinite(retained), retained, median)
        transformed = (filled - np.asarray(self.mean)) / np.asarray(self.scale)
        transformed = np.clip(transformed, -10.0, 10.0)
        if not np.isfinite(transformed).all():
            raise ValueError("Non-finite feature after preprocessing")
        return transformed.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FeaturePreprocessor":
        return cls(
            **{
                name: tuple(raw[name])
                for name in ("finite_mask", "median", "mean", "scale")
            }
        )


__all__ = ["FeaturePreprocessor"]
