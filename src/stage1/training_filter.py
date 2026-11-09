"""Read-only training subset over an unchanged, integrity-checked corpus."""
from __future__ import annotations

import csv
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem

from common.identity import semantic_identity
from common.io import sha256_file
from .data import PreparedCorpusDataset


def resolve_graph_training_filter(config, metadata):
    limit = config.training.max_graph_nodes
    if limit is None:
        return None
    root = config.data.artifacts_dir
    for name in ("manifest.csv", "train_index.npy"):
        if sha256_file(root / name) != metadata["artifact_hashes"][name]:
            raise ValueError(f"Training filter source hash mismatch: {name}")
    corpus = metadata["semantic"]["identities"]["corpus"]
    return _resolve_filter(
        str(root), limit, corpus["hash"], corpus["payload"]["data"]["seed"],
        metadata["artifact_hashes"]["manifest.csv"], metadata["artifact_hashes"]["train_index.npy"],
    )


@lru_cache(maxsize=2)
def _resolve_filter(root, limit, corpus_hash, seed, manifest_sha, index_sha):
    root = Path(root)
    indices = np.load(root / "train_index.npy", mmap_mode="r")
    keys = np.empty(len(indices), dtype="S32")
    excluded = []
    count = 0
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "train":
                continue
            smiles = row["canonical_smiles"]
            # The prepare contract sorts train rows by this SHA, then SMILES.
            keys[count] = hashlib.sha256(f"{seed}\0train\0{smiles}".encode()).digest()
            count += 1
            # SMILES text length bounds both atom and bond counts from above.
            if len(smiles) <= limit:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid prepared structure: {row['sample_id']}")
            atoms, bonds = mol.GetNumAtoms(), mol.GetNumBonds()
            if max(atoms, bonds) > limit:
                if row["is_augmented"] != "1":
                    raise ValueError(f"Graph training filter would exclude an original sample: {row['sample_id']}")
                excluded.append({"sample_id": row["sample_id"], "canonical_smiles": smiles,
                                 "atoms": atoms, "bonds": bonds, "role": row["role"],
                                 "is_augmented": row["is_augmented"] == "1", "sources": row["sources"]})
    if count != len(indices):
        raise ValueError("Training filter manifest/index row count mismatch")
    # Locate the few exclusions without unpacking every tensor shard. Verify
    # the derived positions against the actual, SHA-checked source records.
    dataset = PreparedCorpusDataset(root, "train", 1)
    for row in excluded:
        key = hashlib.sha256(f"{seed}\0train\0{row['canonical_smiles']}".encode()).digest()
        if np.count_nonzero(keys == key) != 1:
            raise ValueError("Ambiguous training filter corpus ordering")
        index = int(np.count_nonzero(keys < key))
        sample = dataset[index]
        if (sample["sample_id"] != row["sample_id"]
                or len(sample["atom_categorical"]) != row["atoms"]
                or sample["bond_index"].shape[1] != row["bonds"]):
            raise ValueError("Training filter manifest/shard position or structure mismatch")
        row["dataset_index"] = index
    if len(excluded) == count:
        raise ValueError("Graph training filter removed every training sample")
    return semantic_identity("stage1.graph-training-filter.v1", {
        "max_graph_nodes": limit, "split": "train", "corpus_identity": corpus_hash,
        "manifest_sha256": manifest_sha, "train_index_sha256": index_sha,
        "statistics": "reuse-prepared-tokenizer-and-scalers", "original_train_size": count,
        "retained_train_size": count - len(excluded),
        "excluded": sorted(excluded, key=lambda row: row["dataset_index"]),
    })
