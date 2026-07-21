from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from rdkit.Chem import Descriptors


def rdkit_descriptor_names() -> tuple[str, ...]:
    return tuple(name for name, _ in Descriptors._descList)


def calculate_descriptors(mol, names: Sequence[str]) -> np.ndarray:
    values_by_name = Descriptors.CalcMolDescriptors(
        mol, missingVal=float("nan"), silent=True
    )
    return np.asarray([values_by_name[name] for name in names], dtype=np.float64)


@dataclass(frozen=True)
class DescriptorStandardizer:
    names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    finite_counts: np.ndarray

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        names: Sequence[str],
    ) -> "DescriptorStandardizer":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(names):
            raise ValueError("Descriptor matrix and name count do not match")
        valid = np.isfinite(values)
        finite_counts = valid.sum(axis=0).astype(np.int64)
        safe_values = np.where(valid, values, 0.0)
        sums = safe_values.sum(axis=0)
        means = np.divide(
            sums,
            finite_counts,
            out=np.zeros_like(sums),
            where=finite_counts > 0,
        )
        centered = np.where(valid, values - means, 0.0)
        variances = np.divide(
            np.square(centered).sum(axis=0),
            finite_counts,
            out=np.zeros_like(sums),
            where=finite_counts > 0,
        )
        scales = np.sqrt(variances)
        scales[(scales == 0.0) | ~np.isfinite(scales)] = 1.0
        return cls(tuple(names), means, scales, finite_counts)

    def transform(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=np.float64)
        if values.shape[-1] != len(self.names):
            raise ValueError("Descriptor input has the wrong final dimension")
        valid = np.isfinite(values) & (self.finite_counts > 0)
        standardized = np.where(valid, (values - self.means) / self.scales, 0.0)
        return standardized.astype(np.float32), valid

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.shape[-1] != len(self.names):
            raise ValueError("Descriptor input has the wrong final dimension")
        return values * self.scales + self.means

    def save(self, path: str | Path) -> None:
        payload = {
            "format_version": 1,
            "names": list(self.names),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "finite_counts": self.finite_counts.tolist(),
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        expected_names: Sequence[str] | None = None,
    ) -> "DescriptorStandardizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported descriptor scaler artifact format")
        names = tuple(payload["names"])
        if expected_names is not None and names != tuple(expected_names):
            raise ValueError("RDKit descriptor names/order do not match the artifact")
        return cls(
            names=names,
            means=np.asarray(payload["means"], dtype=np.float64),
            scales=np.asarray(payload["scales"], dtype=np.float64),
            finite_counts=np.asarray(payload["finite_counts"], dtype=np.int64),
        )
