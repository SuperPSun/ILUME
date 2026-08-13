from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

from .config import FingerprintConfig


FINGERPRINT_DIMS = {"morgan": 2048, "maccs": 167}


def fingerprint_families(kind: str) -> tuple[str, ...]:
    if kind == "none":
        return ()
    if kind == "both":
        return ("morgan", "maccs")
    if kind in FINGERPRINT_DIMS:
        return (kind,)
    raise ValueError(f"Unknown fingerprint kind: {kind}")


def _to_numpy(bit_vector, dimension: int) -> np.ndarray:
    values = np.zeros((dimension,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bit_vector, values)
    return values


@lru_cache(maxsize=None)
def _morgan_generator(radius: int, bits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)


def calculate_fingerprints(mol, config: FingerprintConfig) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if "morgan" in fingerprint_families(config.kind):
        generator = _morgan_generator(config.morgan_radius, config.morgan_bits)
        result["morgan"] = _to_numpy(generator.GetFingerprint(mol), config.morgan_bits)
    if "maccs" in fingerprint_families(config.kind):
        result["maccs"] = _to_numpy(MACCSkeys.GenMACCSKeys(mol), config.maccs_bits)
    return result


@dataclass(frozen=True)
class FingerprintBatch:
    values: dict[str, torch.Tensor]
    valid: dict[str, torch.Tensor]

    def to(self, device: torch.device | str) -> "FingerprintBatch":
        return FingerprintBatch(
            values={name: value.to(device) for name, value in self.values.items()},
            valid={name: value.to(device) for name, value in self.valid.items()},
        )
