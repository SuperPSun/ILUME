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
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit import rdBase

from common.training import canonical_json_sha256
from common.progress import ProgressReporter
from common.descriptor_preprocessing import FeaturePreprocessor
from .config import FeatureConfig
from .data import RawDataset


FEATURE_CACHE_SCHEMA_VERSION = 1
BASIC_FEATURE_SCHEMA_VERSION = "basic-molecular-statistics-v1"
BASIC_FEATURE_NAMES = (
    "molecular_weight",
    "heavy_atom_count",
    "total_atom_count_with_implicit_hydrogens",
    "C_count",
    "N_count",
    "O_count",
    "F_count",
    "P_count",
    "S_count",
    "Cl_count",
    "Br_count",
    "I_count",
    "formal_charge",
    "bond_count",
    "ring_count",
    "aromatic_atom_count",
    "aromatic_ring_count",
    "rotatable_bond_count",
    "h_bond_donor_count",
    "h_bond_acceptor_count",
    "fraction_c_sp3",
)
_ELEMENT_ATOMIC_NUMBERS = (6, 7, 8, 9, 15, 16, 17, 35, 53)
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
    feature_names: tuple[str, ...]
    schema_version: str | None
    rdkit_version: str

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "ecfp4":
            return {
                "kind": self.kind,
                "component_width": self.component_width,
                "radius": self.radius,
                "n_bits": self.n_bits,
                "descriptor_names": (),
                "rdkit_version": self.rdkit_version,
            }
        return asdict(self)


def feature_schema(config: FeatureConfig) -> FeatureSchema:
    if config.kind == "basic_molecular_statistics":
        return FeatureSchema(
            kind=config.kind,
            component_width=len(BASIC_FEATURE_NAMES),
            radius=None,
            n_bits=None,
            feature_names=BASIC_FEATURE_NAMES,
            schema_version=BASIC_FEATURE_SCHEMA_VERSION,
            rdkit_version=rdBase.rdkitVersion,
        )
    if config.radius is None or config.n_bits is None:
        raise ValueError("ECFP4 feature schema requires radius and n_bits")
    return FeatureSchema(
        kind=config.kind,
        component_width=config.n_bits,
        radius=config.radius,
        n_bits=config.n_bits,
        feature_names=(),
        schema_version=None,
        rdkit_version=rdBase.rdkitVersion,
    )


def basic_molecular_statistics(molecule: Chem.Mol) -> np.ndarray:
    element_counts = {atomic_number: 0 for atomic_number in _ELEMENT_ATOMIC_NUMBERS}
    formal_charge = 0
    aromatic_atom_count = 0
    for atom in molecule.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        if atomic_number in element_counts:
            element_counts[atomic_number] += 1
        formal_charge += atom.GetFormalCharge()
        aromatic_atom_count += int(atom.GetIsAromatic())
    values = (
        Descriptors.MolWt(molecule),
        molecule.GetNumHeavyAtoms(),
        Chem.AddHs(molecule).GetNumAtoms(),
        *(element_counts[value] for value in _ELEMENT_ATOMIC_NUMBERS),
        formal_charge,
        molecule.GetNumBonds(),
        rdMolDescriptors.CalcNumRings(molecule),
        aromatic_atom_count,
        rdMolDescriptors.CalcNumAromaticRings(molecule),
        rdMolDescriptors.CalcNumRotatableBonds(
            molecule, rdMolDescriptors.NumRotatableBondsOptions.Strict
        ),
        rdMolDescriptors.CalcNumHBD(molecule),
        rdMolDescriptors.CalcNumHBA(molecule),
        rdMolDescriptors.CalcFractionCSP3(molecule),
    )
    return np.asarray(values, dtype=np.float64)


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
    if schema.kind == "basic_molecular_statistics":
        return basic_molecular_statistics(molecule)
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


def ensure_finite_raw_features(values: np.ndarray) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError("XGBoost features contain NaN or Inf")
    return values.astype(np.float32)


__all__ = [
    "BASIC_FEATURE_NAMES",
    "BASIC_FEATURE_SCHEMA_VERSION",
    "FeatureCache",
    "FeaturePreprocessor",
    "FeatureSchema",
    "basic_molecular_statistics",
    "ensure_finite_raw_features",
    "feature_schema",
    "raw_feature_matrix",
]
