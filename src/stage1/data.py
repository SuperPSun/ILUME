from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rdkit import rdBase
from torch.utils.data import Dataset

from common.io import sha256_file
from .descriptors import DescriptorSchema, DescriptorStandardizer, rdkit_descriptor_names
from .fingerprints import FingerprintBatch
from .graph import GraphRecord, PackedGraph
from .tokenizer import tokenizer_backend_version


CORPUS_FORMAT_VERSION = 3


@dataclass(frozen=True)
class MaskPlan:
    smiles_labels: torch.Tensor
    atom_mask: torch.Tensor
    bond_mask: torch.Tensor
    descriptor_indicator: torch.Tensor
    descriptor_loss_mask: torch.Tensor
    fingerprint_indicator: dict[str, torch.Tensor]
    fingerprint_loss_mask: dict[str, torch.Tensor]
    modality_dropped: torch.Tensor

    def to(self, device: torch.device | str) -> "MaskPlan":
        return MaskPlan(
            smiles_labels=self.smiles_labels.to(device),
            atom_mask=self.atom_mask.to(device),
            bond_mask=self.bond_mask.to(device),
            descriptor_indicator=self.descriptor_indicator.to(device),
            descriptor_loss_mask=self.descriptor_loss_mask.to(device),
            fingerprint_indicator={
                name: value.to(device) for name, value in self.fingerprint_indicator.items()
            },
            fingerprint_loss_mask={
                name: value.to(device) for name, value in self.fingerprint_loss_mask.items()
            },
            modality_dropped=self.modality_dropped.to(device),
        )


@dataclass(frozen=True)
class MultimodalBatch:
    token_ids: torch.Tensor
    token_padding_mask: torch.Tensor
    graphs: PackedGraph
    descriptors: torch.Tensor
    descriptor_valid: torch.Tensor
    fingerprints: FingerprintBatch
    roles: torch.Tensor
    sample_ids: tuple[str, ...]
    masks: MaskPlan | None = None

    def to(self, device: torch.device | str) -> "MultimodalBatch":
        return MultimodalBatch(
            token_ids=self.token_ids.to(device),
            token_padding_mask=self.token_padding_mask.to(device),
            graphs=self.graphs.to(device),
            descriptors=self.descriptors.to(device),
            descriptor_valid=self.descriptor_valid.to(device),
            fingerprints=self.fingerprints.to(device),
            roles=self.roles.to(device),
            sample_ids=self.sample_ids,
            masks=None if self.masks is None else self.masks.to(device),
        )


class PreparedCorpusDataset(Dataset):
    def __init__(
        self,
        artifact_path: str | Path,
        split: str = "train",
        shard_cache_size: int = 4,
    ) -> None:
        if split not in {"train", "valid", "all"}:
            raise ValueError("split must be train, valid, or all")
        path = Path(artifact_path)
        if path.is_file():
            if path.name == "corpus.pt":
                raise ValueError(
                    "Legacy corpus.pt artifacts are unsupported; rerun scripts/stage1/prepare.py"
                )
            raise ValueError("PreparedCorpusDataset expects an artifact directory")
        self.artifact_dir = path
        self.metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata.get("format_version") != CORPUS_FORMAT_VERSION:
            raise ValueError("Unsupported corpus artifact format; rerun scripts/stage1/prepare.py")
        artifact_hashes = self.metadata.get("artifact_hashes")
        if not isinstance(artifact_hashes, dict) or not artifact_hashes:
            raise ValueError("Corpus artifact hashes are missing; rerun scripts/stage1/prepare.py")
        for filename, expected_hash in artifact_hashes.items():
            if sha256_file(path / filename) != expected_hash:
                raise ValueError(f"Artifact hash mismatch: {filename}")
        if self.metadata.get("rdkit_version") != rdBase.rdkitVersion:
            raise ValueError("RDKit version does not match the prepared corpus")
        expected_tokenizer_version = tokenizer_backend_version(
            self.metadata["tokenizer_backend"]
        )
        if self.metadata.get("tokenizer_backend_version") != expected_tokenizer_version:
            raise ValueError("Tokenizer backend version does not match the corpus")
        current_names = rdkit_descriptor_names()
        self.descriptor_schema = DescriptorSchema.load(
            path / "descriptor_schema.json", expected_raw_names=current_names
        )
        DescriptorStandardizer.load(
            path / "descriptor_scaler.json",
            expected_names=self.descriptor_schema.selected_names,
        )
        payload = json.loads((path / "corpus_index.json").read_text(encoding="utf-8"))
        if payload.get("format_version") != CORPUS_FORMAT_VERSION:
            raise ValueError("Unsupported corpus index format")
        self.entries = [
            entry for entry in payload["entries"] if split == "all" or entry["split"] == split
        ]
        self.role_ids = [int(entry["role_id"]) for entry in self.entries]
        self.shard_cache_size = shard_cache_size
        self._cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._verified_shards: set[str] = set()

    def __len__(self) -> int:
        return len(self.entries)

    def _load_shard(self, relative_path: str) -> list[dict[str, Any]]:
        if relative_path in self._cache:
            samples = self._cache.pop(relative_path)
            self._cache[relative_path] = samples
            return samples
        shard_path = self.artifact_dir / relative_path
        if relative_path not in self._verified_shards:
            expected_hash = self.metadata.get("shard_hashes", {}).get(relative_path)
            if expected_hash is None or sha256_file(shard_path) != expected_hash:
                raise ValueError(f"Shard hash mismatch: {relative_path}")
            self._verified_shards.add(relative_path)
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        if payload.get("format_version") != CORPUS_FORMAT_VERSION:
            raise ValueError(f"Unsupported shard format: {relative_path}")
        samples = payload["samples"]
        self._cache[relative_path] = samples
        while len(self._cache) > self.shard_cache_size:
            self._cache.popitem(last=False)
        return samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        return self._load_shard(entry["shard"])[int(entry["offset"])]


def graph_record_from_sample(sample: dict[str, Any]) -> GraphRecord:
    return GraphRecord(
        atom_categorical=sample["atom_categorical"],
        atom_continuous=sample["atom_continuous"],
        bond_categorical=sample["bond_categorical"],
        bond_index=sample["bond_index"],
    )
