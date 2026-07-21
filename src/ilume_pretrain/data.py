from __future__ import annotations

import csv
import importlib.metadata
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from rdkit import Chem, rdBase
from torch.utils.data import Dataset

from .config import DataConfig
from .descriptors import (
    DescriptorStandardizer,
    calculate_descriptors,
    rdkit_descriptor_names,
)
from .graph import GraphRecord, PackedGraph, featurize_mol
from .tokenizer import AISVocabulary


ROLE_TO_ID = {"cation": 0, "anion": 1, "neutral": 2}
ROLE_SOURCE_FILES = {
    "cation": ("cation.csv",),
    "anion": ("anion.csv",),
    "neutral": ("molecule.csv", "solute.csv", "solvent.csv"),
}


@dataclass(frozen=True)
class MaskPlan:
    smiles_labels: torch.Tensor
    atom_mask: torch.Tensor
    bond_mask: torch.Tensor
    descriptor_indicator: torch.Tensor
    descriptor_loss_mask: torch.Tensor
    modality_dropped: torch.Tensor

    def to(self, device: torch.device | str) -> "MaskPlan":
        return MaskPlan(
            smiles_labels=self.smiles_labels.to(device),
            atom_mask=self.atom_mask.to(device),
            bond_mask=self.bond_mask.to(device),
            descriptor_indicator=self.descriptor_indicator.to(device),
            descriptor_loss_mask=self.descriptor_loss_mask.to(device),
            modality_dropped=self.modality_dropped.to(device),
        )


@dataclass(frozen=True)
class MultimodalBatch:
    token_ids: torch.Tensor
    token_padding_mask: torch.Tensor
    graphs: PackedGraph
    descriptors: torch.Tensor
    descriptor_valid: torch.Tensor
    roles: torch.Tensor
    sample_ids: tuple[str, ...]
    masks: MaskPlan

    def to(self, device: torch.device | str) -> "MultimodalBatch":
        return MultimodalBatch(
            token_ids=self.token_ids.to(device),
            token_padding_mask=self.token_padding_mask.to(device),
            graphs=self.graphs.to(device),
            descriptors=self.descriptors.to(device),
            descriptor_valid=self.descriptor_valid.to(device),
            roles=self.roles.to(device),
            sample_ids=self.sample_ids,
            masks=self.masks.to(device),
        )


def _load_role_smiles(stage1_dir: Path) -> dict[str, dict[str, set[str]]]:
    by_role: dict[str, dict[str, set[str]]] = {}
    for role, filenames in ROLE_SOURCE_FILES.items():
        canonical_to_sources: dict[str, set[str]] = {}
        for filename in filenames:
            path = stage1_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing Stage 1 source: {path}")
            with path.open(newline="", encoding="utf-8") as handle:
                for row_number, row in enumerate(csv.DictReader(handle), start=2):
                    smiles = (row.get("SMILES") or "").strip()
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        raise ValueError(f"Invalid SMILES in {path}:{row_number}: {smiles}")
                    canonical = Chem.MolToSmiles(mol, canonical=True)
                    canonical_to_sources.setdefault(canonical, set()).add(path.stem)
        by_role[role] = canonical_to_sources
    return by_role


def _split_records(
    by_role: dict[str, dict[str, set[str]]],
    config: DataConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for role_index, role in enumerate(("cation", "anion", "neutral")):
        items = sorted(by_role[role].items())
        rng = random.Random(config.seed + role_index)
        rng.shuffle(items)
        if config.max_samples_per_role is not None:
            items = items[: config.max_samples_per_role]
        valid_count = max(1, round(len(items) * config.valid_fraction))
        for index, (smiles, sources) in enumerate(items):
            records.append(
                {
                    "role": role,
                    "role_id": ROLE_TO_ID[role],
                    "canonical_smiles": smiles,
                    "sources": tuple(sorted(sources)),
                    "split": "valid" if index < valid_count else "train",
                }
            )
    records.sort(key=lambda item: (item["role_id"], item["canonical_smiles"]))
    counters = {role: 0 for role in ROLE_TO_ID}
    for record in records:
        role = record["role"]
        counters[role] += 1
        record["sample_id"] = f"{role}_{counters[role]:07d}"
    return records


def prepare_corpus(config: DataConfig) -> dict[str, int]:
    descriptor_names = rdkit_descriptor_names()
    if len(descriptor_names) != config.descriptor_dim:
        raise ValueError(
            f"Configured descriptor_dim={config.descriptor_dim}, but this RDKit "
            f"provides {len(descriptor_names)} descriptors"
        )
    by_role = _load_role_smiles(config.stage1_dir)
    records = _split_records(by_role, config)
    train_smiles = [
        record["canonical_smiles"] for record in records if record["split"] == "train"
    ]
    vocabulary = AISVocabulary.fit(train_smiles)

    raw_descriptors: list[np.ndarray] = []
    graphs: list[GraphRecord] = []
    token_ids: list[list[int]] = []
    for record in records:
        mol = Chem.MolFromSmiles(record["canonical_smiles"])
        if mol is None:
            raise RuntimeError("Canonical SMILES unexpectedly failed RDKit parsing")
        token_ids.append(
            vocabulary.encode(record["canonical_smiles"], config.max_smiles_tokens)
        )
        raw_descriptors.append(calculate_descriptors(mol, descriptor_names))
        graphs.append(featurize_mol(mol))

    raw_matrix = np.stack(raw_descriptors)
    train_indices = [
        index for index, record in enumerate(records) if record["split"] == "train"
    ]
    standardizer = DescriptorStandardizer.fit(
        raw_matrix[train_indices], descriptor_names
    )
    standardized, descriptor_valid = standardizer.transform(raw_matrix)

    samples: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        graph = graphs[index]
        samples.append(
            {
                **record,
                "token_ids": torch.tensor(token_ids[index], dtype=torch.long),
                "atom_categorical": graph.atom_categorical,
                "atom_continuous": graph.atom_continuous,
                "bond_categorical": graph.bond_categorical,
                "bond_index": graph.bond_index,
                "descriptors": torch.from_numpy(standardized[index]),
                "descriptor_valid": torch.from_numpy(descriptor_valid[index]),
            }
        )

    output_dir = config.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    vocabulary.save(output_dir / "tokenizer.json")
    standardizer.save(output_dir / "descriptor_scaler.json")

    metadata = {
        "format_version": 1,
        "rdkit_version": rdBase.rdkitVersion,
        "atom_in_smiles_version": importlib.metadata.version("atomInSmiles"),
        "descriptor_names": list(descriptor_names),
        "descriptor_dim": len(descriptor_names),
        "max_smiles_tokens": config.max_smiles_tokens,
        "role_source_files": {
            role: list(files) for role, files in ROLE_SOURCE_FILES.items()
        },
        "ignored_stage1_files": ["IL.csv"],
        "seed": config.seed,
        "valid_fraction": config.valid_fraction,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save({"metadata": metadata, "samples": samples}, output_dir / "corpus.pt")

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "role",
                "canonical_smiles",
                "split",
                "sources",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "role": record["role"],
                    "canonical_smiles": record["canonical_smiles"],
                    "split": record["split"],
                    "sources": ";".join(record["sources"]),
                }
            )

    summary = {
        "total": len(records),
        "train": sum(record["split"] == "train" for record in records),
        "valid": sum(record["split"] == "valid" for record in records),
        **{
            role: sum(record["role"] == role for record in records)
            for role in ROLE_TO_ID
        },
    }
    return summary


class PreparedCorpusDataset(Dataset):
    def __init__(self, artifact_path: str | Path, split: str = "train") -> None:
        if split not in {"train", "valid", "all"}:
            raise ValueError("split must be train, valid, or all")
        payload = torch.load(Path(artifact_path), map_location="cpu", weights_only=False)
        self.metadata = payload["metadata"]
        current_names = rdkit_descriptor_names()
        if tuple(self.metadata["descriptor_names"]) != current_names:
            raise ValueError(
                "Current RDKit descriptor names/order do not match the corpus artifact"
            )
        scaler_path = Path(artifact_path).parent / "descriptor_scaler.json"
        DescriptorStandardizer.load(scaler_path, expected_names=current_names)
        self.samples = [
            sample
            for sample in payload["samples"]
            if split == "all" or sample["split"] == split
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def graph_record_from_sample(sample: dict[str, Any]) -> GraphRecord:
    return GraphRecord(
        atom_categorical=sample["atom_categorical"],
        atom_continuous=sample["atom_continuous"],
        bond_categorical=sample["bond_categorical"],
        bond_index=sample["bond_index"],
    )
