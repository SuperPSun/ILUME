from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

from common.io import sha256_file
from .config import PretrainConfig, config_from_dict
from .descriptors import DescriptorSchema, DescriptorStandardizer, rdkit_descriptor_names
from .fingerprints import calculate_fingerprints
from .graph import featurize_mol
from .tokenizer import SmilesTokenizer


ROLE_TO_ID = {"cation": 0, "anion": 1, "neutral": 2}
IPC_SQUARE_OVERFLOW_LIMIT = float(np.sqrt(np.finfo(np.float64).max))
BCUT_SUPPORTED_BOND_TYPES = {
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
}


@dataclass
class EntityQC:
    record: dict[str, Any]
    reasons: list[str]
    unsupported_bond_types: tuple[str, ...]
    ipc: float
    token_count: int | None = None


def _calculate_ipc(mol: Chem.Mol) -> float:
    return float(Descriptors.Ipc(mol))


def inspect_entity_qc(record: dict[str, Any]) -> EntityQC:
    mol = Chem.MolFromSmiles(record["canonical_smiles"])
    if mol is None:
        raise RuntimeError("Canonical SMILES unexpectedly failed RDKit parsing")
    unsupported = tuple(
        sorted(
            {
                str(bond.GetBondType())
                for bond in mol.GetBonds()
                if bond.GetBondType() not in BCUT_SUPPORTED_BOND_TYPES
            }
        )
    )
    reasons: list[str] = []
    if unsupported:
        reasons.append("unsupported_bcut_bond_type")
    try:
        ipc = _calculate_ipc(mol)
    except Exception:
        ipc = float("nan")
    if not np.isfinite(ipc):
        reasons.append("ipc_nonfinite")
    elif abs(ipc) > IPC_SQUARE_OVERFLOW_LIMIT:
        reasons.append("ipc_square_overflow")
    return EntityQC(record, reasons, unsupported, ipc)


def build_entity_sample(
    record: dict[str, Any],
    raw_descriptors: np.ndarray,
    schema: DescriptorSchema,
    standardizer: DescriptorStandardizer,
    tokenizer: SmilesTokenizer,
    config: PretrainConfig,
) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(record["canonical_smiles"])
    if mol is None:
        raise RuntimeError("Canonical SMILES unexpectedly failed RDKit parsing")
    encoded = tokenizer.encode(
        record["canonical_smiles"], config.data.max_smiles_tokens
    )
    selected = schema.select(raw_descriptors[None, :])
    standardized, descriptor_valid = standardizer.transform(selected)
    invalid = descriptor_valid & ~np.isfinite(standardized)
    if invalid.any():
        names = [
            schema.selected_names[index]
            for index in np.flatnonzero(invalid[0]).tolist()
        ]
        raise ValueError(
            f"Non-finite standardized descriptors for {record['sample_id']}: "
            + ", ".join(names)
        )
    graph = featurize_mol(mol)
    return {
        **record,
        "token_ids": torch.tensor(encoded, dtype=torch.long),
        "atom_categorical": graph.atom_categorical,
        "atom_continuous": graph.atom_continuous,
        "bond_categorical": graph.bond_categorical,
        "bond_index": graph.bond_index,
        "descriptors": torch.from_numpy(standardized[0]),
        "descriptor_valid": torch.from_numpy(descriptor_valid[0]),
        "fingerprints": {
            name: torch.from_numpy(value).float()
            for name, value in calculate_fingerprints(mol, config.fingerprint).items()
        },
    }


def load_stage1_feature_inputs(
    checkpoint_path: str | Path,
    artifact_dir: str | Path,
) -> tuple[
    PretrainConfig,
    SmilesTokenizer,
    DescriptorSchema,
    DescriptorStandardizer,
    str,
]:
    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False
    )
    if checkpoint.get("format_version") != 3:
        raise ValueError("Stage 2 requires a Stage 1 checkpoint in format v3")
    config = config_from_dict(checkpoint["config"])
    artifact_dir = Path(artifact_dir)
    artifact_hash = sha256_file(artifact_dir / "metadata.json")
    if checkpoint.get("artifact_hash") != artifact_hash:
        raise ValueError("Stage 1 checkpoint and preprocessing artifact do not match")
    del checkpoint
    schema = DescriptorSchema.load(
        artifact_dir / "descriptor_schema.json",
        expected_raw_names=rdkit_descriptor_names(),
    )
    standardizer = DescriptorStandardizer.load(
        artifact_dir / "descriptor_scaler.json",
        expected_names=schema.selected_names,
    )
    vocabulary = SmilesTokenizer.load(artifact_dir / "tokenizer.json")
    return config, vocabulary, schema, standardizer, artifact_hash


__all__ = [
    "ROLE_TO_ID",
    "EntityQC",
    "inspect_entity_qc",
    "build_entity_sample",
    "load_stage1_feature_inputs",
]
