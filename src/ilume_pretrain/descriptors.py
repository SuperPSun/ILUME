from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from rdkit.Chem import Descriptors


GROUPS_12 = (
    "mass_size",
    "lipophilicity",
    "refractivity",
    "polarity_surface",
    "charge",
    "estate",
    "vsa",
    "topology",
    "shape_kappa",
    "atom_bond_counts",
    "rings_aromaticity",
    "functional_groups",
)
GROUPS_8 = (
    "composition",
    "hydrophobic_refractive",
    "surface_polarity",
    "electronic",
    "topology",
    "shape",
    "rings_aromaticity",
    "functional_groups",
)
SEMANTIC_MAPPING_VERSION = "rdkit-217-v1"


def rdkit_descriptor_names() -> tuple[str, ...]:
    return tuple(name for name, _ in Descriptors._descList)


def calculate_descriptors(mol, names: Sequence[str]) -> np.ndarray:
    values_by_name = Descriptors.CalcMolDescriptors(
        mol, missingVal=float("nan"), silent=True
    )
    return np.asarray([values_by_name[name] for name in names], dtype=np.float64)


def _descriptor_group_12(name: str) -> str:
    lower = name.lower()
    if lower.startswith("fr_"):
        return "functional_groups"
    if "ring" in lower or "aromatic" in lower or "aliphatic" in lower:
        return "rings_aromaticity"
    if lower.startswith("num") or lower in {
        "heavyatomcount",
        "nhohcount",
        "nocount",
        "fractioncsp3",
    }:
        return "atom_bond_counts"
    if lower.startswith("peoe_vsa") or "partialcharge" in lower:
        return "charge"
    if lower.startswith("estate_vsa") or "estateindex" in lower:
        return "estate"
    if lower.startswith(("slogp_vsa", "smr_vsa")):
        return "vsa"
    if "logp" in lower:
        return "lipophilicity"
    if "molmr" in lower or "refractivity" in lower:
        return "refractivity"
    if any(key in lower for key in ("tpsa", "labuteasa", "asa")):
        return "polarity_surface"
    if lower.startswith("bcut") or lower.startswith("chi") or lower in {
        "balabanj",
        "bertzct",
        "ipc",
    }:
        return "topology"
    if lower.startswith("kappa") or lower in {"hallkieralpha", "qed"}:
        return "shape_kappa"
    if any(key in lower for key in ("molwt", "exactmolwt", "heavyatommolwt")):
        return "mass_size"
    if "charge" in lower:
        return "charge"
    return "topology"


_GROUP_12_TO_8 = {
    "mass_size": "composition",
    "atom_bond_counts": "composition",
    "lipophilicity": "hydrophobic_refractive",
    "refractivity": "hydrophobic_refractive",
    "polarity_surface": "surface_polarity",
    "vsa": "surface_polarity",
    "charge": "electronic",
    "estate": "electronic",
    "topology": "topology",
    "shape_kappa": "shape",
    "rings_aromaticity": "rings_aromaticity",
    "functional_groups": "functional_groups",
}


def _pairwise_correlations(
    values: np.ndarray,
    columns: Sequence[int] | None = None,
    chunk_size: int = 65536,
) -> np.ndarray:
    dimension = values.shape[1] if columns is None else len(columns)
    counts = np.zeros((dimension, dimension), dtype=np.float64)
    sums = np.zeros_like(counts)
    sums_squared = np.zeros_like(counts)
    cross = np.zeros_like(counts)
    for start in range(0, values.shape[0], chunk_size):
        chunk = values[start : start + chunk_size]
        if columns is not None:
            chunk = chunk[:, columns]
        valid = np.isfinite(chunk)
        filled = np.where(valid, chunk, 0.0).astype(np.float64, copy=False)
        valid_float = valid.astype(np.float64)
        counts += valid_float.T @ valid_float
        sums += filled.T @ valid_float
        sums_squared += np.square(filled).T @ valid_float
        cross += filled.T @ filled
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = cross - sums * sums.T / counts
        variance_left = sums_squared - np.square(sums) / counts
        variance_right = variance_left.T
        correlation = numerator / np.sqrt(variance_left * variance_right)
    correlation[counts < 2] = np.nan
    np.fill_diagonal(correlation, 1.0)
    return correlation


@dataclass(frozen=True)
class DescriptorSchema:
    raw_names: tuple[str, ...]
    retained_indices: tuple[int, ...]
    removal_reasons: dict[str, str]
    correlation_clusters: tuple[tuple[str, ...], ...]
    cluster_representatives: tuple[str, ...]
    semantic_mapping_version: str
    raw_semantic_groups: tuple[str, ...]
    group_names: tuple[str, ...]
    group_indices: tuple[tuple[int, ...], ...]
    mode: str
    token_count: int
    correlation_threshold: float

    @property
    def selected_names(self) -> tuple[str, ...]:
        return tuple(self.raw_names[index] for index in self.retained_indices)

    @property
    def selected_dim(self) -> int:
        return len(self.retained_indices)

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        names: Sequence[str],
        mode: str,
        token_count: int,
        correlation_threshold: float = 0.98,
    ) -> "DescriptorSchema":
        values = np.asarray(values, dtype=np.float64)
        names = tuple(names)
        if values.ndim != 2 or values.shape[1] != len(names):
            raise ValueError("Descriptor matrix and name count do not match")
        if mode not in {"full", "clean", "pruned"}:
            raise ValueError("Descriptor mode must be full, clean, or pruned")
        if token_count not in {1, 8, 12}:
            raise ValueError("Descriptor token_count must be 1, 8, or 12")

        removal: dict[str, str] = {}
        retained = list(range(len(names)))
        if mode in {"clean", "pruned"}:
            candidates: list[int] = []
            for index, name in enumerate(names):
                finite = values[np.isfinite(values[:, index]), index]
                if finite.size == 0:
                    removal[name] = "all_non_finite"
                elif np.max(finite) == np.min(finite):
                    removal[name] = "zero_variance"
                else:
                    candidates.append(index)

            digest_to_indices: dict[bytes, list[int]] = {}
            for index in candidates:
                valid = np.isfinite(values[:, index])
                digest = hashlib.sha256(
                    valid.tobytes() + values[valid, index].tobytes()
                ).digest()
                digest_to_indices.setdefault(digest, []).append(index)
            retained = []
            for indices in digest_to_indices.values():
                representative = min(indices)
                retained.append(representative)
                for duplicate in sorted(indices):
                    if duplicate != representative:
                        removal[names[duplicate]] = f"duplicate_of:{names[representative]}"
            retained.sort()

        clusters: list[tuple[str, ...]] = []
        cluster_representatives: list[str] = []
        if mode == "pruned" and retained:
            correlations = _pairwise_correlations(values, retained)
            adjacency = np.abs(correlations) > correlation_threshold
            visited: set[int] = set()
            keep_positions: set[int] = set()
            for start in range(len(retained)):
                if start in visited:
                    continue
                stack = [start]
                component: list[int] = []
                while stack:
                    position = stack.pop()
                    if position in visited:
                        continue
                    visited.add(position)
                    component.append(position)
                    neighbors = np.where(adjacency[position])[0].tolist()
                    stack.extend(neighbor for neighbor in neighbors if neighbor not in visited)
                component.sort()
                if len(component) == 1:
                    keep_positions.add(component[0])
                    continue
                finite_counts = np.asarray(
                    [
                        np.isfinite(values[:, retained[position]]).sum()
                        for position in component
                    ]
                )
                centrality = np.nansum(np.abs(correlations[np.ix_(component, component)]), axis=1)
                representative = max(
                    range(len(component)),
                    key=lambda local: (
                        int(finite_counts[local]),
                        float(centrality[local]),
                        -retained[component[local]],
                    ),
                )
                representative_position = component[representative]
                keep_positions.add(representative_position)
                cluster_names = tuple(names[retained[position]] for position in component)
                clusters.append(cluster_names)
                representative_name = names[retained[representative_position]]
                cluster_representatives.append(representative_name)
                for position in component:
                    if position != representative_position:
                        removal[names[retained[position]]] = (
                            f"correlated_with:{representative_name}"
                        )
            retained = [raw_index for position, raw_index in enumerate(retained) if position in keep_positions]

        selected_names = tuple(names[index] for index in retained)
        raw_semantic_groups = tuple(_descriptor_group_12(name) for name in names)
        if token_count == 1:
            group_names = ("all_descriptors",)
            group_indices = (tuple(range(len(retained))),)
        else:
            group_names = GROUPS_12 if token_count == 12 else GROUPS_8
            assignments: list[list[int]] = [[] for _ in group_names]
            group_to_index = {name: index for index, name in enumerate(group_names)}
            for selected_index, name in enumerate(selected_names):
                group_12 = raw_semantic_groups[retained[selected_index]]
                group = group_12 if token_count == 12 else _GROUP_12_TO_8[group_12]
                assignments[group_to_index[group]].append(selected_index)
            group_indices = tuple(tuple(indices) for indices in assignments)

        return cls(
            raw_names=names,
            retained_indices=tuple(retained),
            removal_reasons=removal,
            correlation_clusters=tuple(clusters),
            cluster_representatives=tuple(cluster_representatives),
            semantic_mapping_version=SEMANTIC_MAPPING_VERSION,
            raw_semantic_groups=raw_semantic_groups,
            group_names=tuple(group_names),
            group_indices=tuple(group_indices),
            mode=mode,
            token_count=token_count,
            correlation_threshold=correlation_threshold,
        )

    @classmethod
    def full(cls, dimension: int, token_count: int = 1) -> "DescriptorSchema":
        names = tuple(f"descriptor_{index}" for index in range(dimension))
        values = np.stack(
            [np.arange(dimension, dtype=np.float64), np.arange(dimension, dtype=np.float64) + 1]
        )
        return cls.fit(values, names, "full", token_count)

    def select(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if values.shape[-1] != len(self.raw_names):
            raise ValueError("Raw descriptor input has the wrong final dimension")
        return values[..., list(self.retained_indices)]

    def save(self, path: str | Path) -> None:
        payload = {
            "format_version": 2,
            "raw_names": list(self.raw_names),
            "retained_indices": list(self.retained_indices),
            "removal_reasons": self.removal_reasons,
            "correlation_clusters": [list(cluster) for cluster in self.correlation_clusters],
            "cluster_representatives": list(self.cluster_representatives),
            "semantic_mapping_version": self.semantic_mapping_version,
            "raw_semantic_groups": list(self.raw_semantic_groups),
            "group_names": list(self.group_names),
            "group_indices": [list(indices) for indices in self.group_indices],
            "mode": self.mode,
            "token_count": self.token_count,
            "correlation_threshold": self.correlation_threshold,
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        expected_raw_names: Sequence[str] | None = None,
    ) -> "DescriptorSchema":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 2:
            raise ValueError("Unsupported descriptor schema artifact format")
        raw_names = tuple(payload["raw_names"])
        if expected_raw_names is not None and raw_names != tuple(expected_raw_names):
            raise ValueError("RDKit descriptor names/order do not match the schema")
        return cls(
            raw_names=raw_names,
            retained_indices=tuple(payload["retained_indices"]),
            removal_reasons=dict(payload["removal_reasons"]),
            correlation_clusters=tuple(tuple(item) for item in payload["correlation_clusters"]),
            cluster_representatives=tuple(payload["cluster_representatives"]),
            semantic_mapping_version=payload["semantic_mapping_version"],
            raw_semantic_groups=tuple(payload["raw_semantic_groups"]),
            group_names=tuple(payload["group_names"]),
            group_indices=tuple(tuple(item) for item in payload["group_indices"]),
            mode=payload["mode"],
            token_count=int(payload["token_count"]),
            correlation_threshold=float(payload["correlation_threshold"]),
        )


@dataclass(frozen=True)
class DescriptorStandardizer:
    names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    finite_counts: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, names: Sequence[str]) -> "DescriptorStandardizer":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(names):
            raise ValueError("Descriptor matrix and name count do not match")
        return cls._fit_chunks(values, tuple(names))

    @classmethod
    def fit_columns(
        cls,
        values: np.ndarray,
        columns: Sequence[int],
        names: Sequence[str],
    ) -> "DescriptorStandardizer":
        if values.ndim != 2 or len(columns) != len(names):
            raise ValueError("Descriptor columns and name count do not match")
        return cls._fit_chunks(values, tuple(names), tuple(columns))

    @classmethod
    def _fit_chunks(
        cls,
        values: np.ndarray,
        names: tuple[str, ...],
        columns: tuple[int, ...] | None = None,
        chunk_size: int = 65536,
    ) -> "DescriptorStandardizer":
        dimension = values.shape[1] if columns is None else len(columns)
        finite_counts = np.zeros(dimension, dtype=np.int64)
        means = np.zeros(dimension, dtype=np.float64)
        squared_deviations = np.zeros(dimension, dtype=np.float64)
        for start in range(0, values.shape[0], chunk_size):
            chunk = values[start : start + chunk_size]
            if columns is not None:
                chunk = chunk[:, columns]
            valid = np.isfinite(chunk)
            safe_values = np.where(valid, chunk, 0.0).astype(
                np.float64, copy=False
            )
            chunk_counts = valid.sum(axis=0).astype(np.int64)
            chunk_means = np.divide(
                safe_values.sum(axis=0),
                chunk_counts,
                out=np.zeros(dimension, dtype=np.float64),
                where=chunk_counts > 0,
            )
            chunk_deviations = np.where(valid, chunk - chunk_means, 0.0)
            chunk_squared_deviations = np.square(chunk_deviations).sum(axis=0)
            combined_counts = finite_counts + chunk_counts
            delta = chunk_means - means
            means += np.divide(
                delta * chunk_counts,
                combined_counts,
                out=np.zeros_like(means),
                where=combined_counts > 0,
            )
            squared_deviations += chunk_squared_deviations + np.divide(
                np.square(delta) * finite_counts * chunk_counts,
                combined_counts,
                out=np.zeros_like(means),
                where=combined_counts > 0,
            )
            finite_counts = combined_counts
        variances = np.divide(
            squared_deviations,
            finite_counts,
            out=np.zeros_like(means),
            where=finite_counts > 0,
        )
        variances = np.maximum(variances, 0.0)
        scales = np.sqrt(variances)
        scales[(scales == 0.0) | ~np.isfinite(scales)] = 1.0
        return cls(names, means, scales, finite_counts)

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
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
