from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit import rdBase

from common.training import canonical_json_sha256
from common.progress import ProgressReporter
from stage1.descriptors import calculate_descriptors, rdkit_descriptor_names

from .config import FeatureConfig
from .data import RawDataset


FEATURE_CACHE_SCHEMA_VERSION = 1
RDKIT_DESCRIPTOR_NAMES = rdkit_descriptor_names()
SQLITE_BUSY_TIMEOUT_MS = 60_000
SQLITE_BUSY_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.0)


def _retry_sqlite_busy(operation: Callable[[], Any]) -> Any:
    for attempt in range(len(SQLITE_BUSY_RETRY_DELAYS) + 1):
        try:
            return operation()
        except sqlite3.OperationalError as error:
            message = str(error).lower()
            if not any(marker in message for marker in ("locked", "busy")):
                raise
            if attempt == len(SQLITE_BUSY_RETRY_DELAYS):
                raise
            time.sleep(SQLITE_BUSY_RETRY_DELAYS[attempt])


@dataclass(frozen=True)
class FeatureSchema:
    kind: str
    component_width: int
    radius: int | None
    n_bits: int | None
    descriptor_names: tuple[str, ...]
    rdkit_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def feature_schema(config: FeatureConfig) -> FeatureSchema:
    if config.kind == "rdkit_2d":
        return FeatureSchema(
            kind=config.kind,
            component_width=len(RDKIT_DESCRIPTOR_NAMES),
            radius=None,
            n_bits=None,
            descriptor_names=RDKIT_DESCRIPTOR_NAMES,
            rdkit_version=rdBase.rdkitVersion,
        )
    return FeatureSchema(
        kind=config.kind,
        component_width=config.n_bits,
        radius=config.radius,
        n_bits=config.n_bits,
        descriptor_names=(),
        rdkit_version=rdBase.rdkitVersion,
    )


class FeatureCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=60)
        try:
            self.connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
            _retry_sqlite_busy(
                lambda: self.connection.execute("PRAGMA journal_mode=WAL").fetchone()
            )
            _retry_sqlite_busy(
                lambda: self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS features ("
                    "cache_key TEXT PRIMARY KEY, payload BLOB NOT NULL, sha256 TEXT NOT NULL, "
                    "dtype TEXT NOT NULL, length INTEGER NOT NULL)"
                )
            )
            _retry_sqlite_busy(self.connection.commit)
        except Exception:
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "FeatureCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _key(smiles: str, schema: FeatureSchema) -> str:
        return canonical_json_sha256({"cache_schema_version": FEATURE_CACHE_SCHEMA_VERSION, "smiles": smiles, "feature": schema.to_dict()})

    def get(self, smiles: str, schema: FeatureSchema) -> np.ndarray | None:
        key = self._key(smiles, schema)
        row = self.connection.execute(
            "SELECT payload, sha256, dtype, length FROM features WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        payload, expected_hash, dtype, length = row
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(f"Corrupt benchmark feature cache entry: {key}")
        with io.BytesIO(payload) as handle:
            value = np.load(handle, allow_pickle=False)
        if str(value.dtype) != dtype or value.ndim != 1 or len(value) != length:
            raise ValueError(f"Malformed benchmark feature cache entry: {key}")
        return value

    def put(self, smiles: str, schema: FeatureSchema, value: np.ndarray) -> None:
        array = np.ascontiguousarray(value)
        with io.BytesIO() as handle:
            np.save(handle, array, allow_pickle=False)
            payload = handle.getvalue()
        key = self._key(smiles, schema)
        digest = hashlib.sha256(payload).hexdigest()
        _retry_sqlite_busy(
            lambda: self.connection.execute(
                "INSERT OR IGNORE INTO features(cache_key, payload, sha256, dtype, length) VALUES (?, ?, ?, ?, ?)",
                (key, payload, digest, str(array.dtype), len(array)),
            )
        )
        _retry_sqlite_busy(self.connection.commit)
        stored = self.get(smiles, schema)
        if stored is None or not np.array_equal(stored, array, equal_nan=True):
            raise ValueError(f"Benchmark feature cache collision: {key}")


def _calculate(smiles: str, schema: FeatureSchema) -> np.ndarray:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid canonical SMILES in feature generation: {smiles}")
    if schema.kind == "rdkit_2d":
        return calculate_descriptors(molecule, schema.descriptor_names)
    assert schema.radius is not None and schema.n_bits is not None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=schema.radius, fpSize=schema.n_bits)
    fingerprint = generator.GetFingerprint(molecule)
    result = np.zeros(schema.n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, result)
    return result


def component_feature(smiles: str, schema: FeatureSchema, cache: FeatureCache) -> np.ndarray:
    cached = cache.get(smiles, schema)
    if cached is not None:
        return cached
    value = _calculate(smiles, schema)
    cache.put(smiles, schema, value)
    return value


def raw_feature_matrix(
    dataset: RawDataset,
    schema: FeatureSchema,
    cache: FeatureCache,
    *,
    reporter: ProgressReporter | None = None,
    desc: str = "Benchmark features",
) -> np.ndarray:
    if not len(dataset):
        return np.empty(
            (0, dataset.component_count * schema.component_width + dataset.conditions.shape[1]),
            dtype=np.float64,
        )
    rows: list[np.ndarray] = []
    progress = (reporter or ProgressReporter()).bar(
        total=len(dataset), desc=desc, unit="row"
    )
    try:
        for index, components in enumerate(dataset.components):
            rows.append(
                np.concatenate(
                    [
                        *(component_feature(smiles, schema, cache) for smiles in components),
                        dataset.conditions[index],
                    ]
                )
            )
            progress.update(1)
    finally:
        progress.close()
    return np.asarray(rows, dtype=np.float64)


@dataclass(frozen=True)
class FeaturePreprocessor:
    finite_mask: tuple[bool, ...]
    median: tuple[float, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, train: np.ndarray) -> "FeaturePreprocessor":
        if train.ndim != 2 or train.shape[0] == 0:
            raise ValueError("Benchmark feature matrix must be non-empty and rank two")
        finite_mask_array = np.isfinite(train).any(axis=0)
        if not finite_mask_array.any():
            raise ValueError("Every benchmark feature column is invalid in train")
        retained = np.where(np.isfinite(train[:, finite_mask_array]), train[:, finite_mask_array], np.nan)
        median = np.nanmedian(retained, axis=0)
        filled = np.where(np.isfinite(retained), retained, median)
        mean = filled.mean(axis=0)
        scale = filled.std(axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        return cls(
            finite_mask=tuple(bool(value) for value in finite_mask_array),
            median=tuple(float(value) for value in median),
            mean=tuple(float(value) for value in mean),
            scale=tuple(float(value) for value in scale),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        mask = np.asarray(self.finite_mask, dtype=bool)
        if values.ndim != 2 or values.shape[1] != len(mask):
            raise ValueError("Benchmark feature width differs from preprocessing contract")
        retained = values[:, mask]
        median = np.asarray(self.median)
        filled = np.where(np.isfinite(retained), retained, median)
        transformed = (filled - np.asarray(self.mean)) / np.asarray(self.scale)
        if not np.isfinite(transformed).all():
            raise ValueError("Non-finite feature after benchmark preprocessing")
        return transformed.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FeaturePreprocessor":
        return cls(**{name: tuple(raw[name]) for name in ("finite_mask", "median", "mean", "scale")})


def ensure_finite_raw_features(values: np.ndarray) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError("XGBoost features contain NaN or Inf")
    return values.astype(np.float32)


__all__ = [
    "FeatureCache",
    "FeaturePreprocessor",
    "FeatureSchema",
    "ensure_finite_raw_features",
    "feature_schema",
    "raw_feature_matrix",
]
