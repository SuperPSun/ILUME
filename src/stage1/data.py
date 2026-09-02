from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from common.identity import IDENTITY_CONTRACT_VERSION
from common.io import sha256_file
from .descriptors import DescriptorSchema, DescriptorStandardizer
from .fingerprints import FingerprintBatch
from .graph import GraphRecord, PackedGraph


CORPUS_FORMAT_VERSION = 2
GLOBAL_RDKIT_CORPUS_FORMAT_VERSION = 3
CORPUS_KIND = "ilume_stage1_corpus"
CORPUS_SHARD_KIND = "ilume_stage1_corpus_shard"
INDEX_DTYPE = np.dtype(
    [("shard_id", "<u4"), ("offset", "<u4"), ("role_id", "u1")]
)


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

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "MaskPlan":
        return MaskPlan(
            smiles_labels=self.smiles_labels.to(device, non_blocking=non_blocking),
            atom_mask=self.atom_mask.to(device, non_blocking=non_blocking),
            bond_mask=self.bond_mask.to(device, non_blocking=non_blocking),
            descriptor_indicator=self.descriptor_indicator.to(
                device, non_blocking=non_blocking
            ),
            descriptor_loss_mask=self.descriptor_loss_mask.to(
                device, non_blocking=non_blocking
            ),
            fingerprint_indicator={
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self.fingerprint_indicator.items()
            },
            fingerprint_loss_mask={
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self.fingerprint_loss_mask.items()
            },
            modality_dropped=self.modality_dropped.to(
                device, non_blocking=non_blocking
            ),
        )

    def pin_memory(self) -> "MaskPlan":
        return MaskPlan(
            smiles_labels=self.smiles_labels.pin_memory(),
            atom_mask=self.atom_mask.pin_memory(),
            bond_mask=self.bond_mask.pin_memory(),
            descriptor_indicator=self.descriptor_indicator.pin_memory(),
            descriptor_loss_mask=self.descriptor_loss_mask.pin_memory(),
            fingerprint_indicator={
                name: value.pin_memory()
                for name, value in self.fingerprint_indicator.items()
            },
            fingerprint_loss_mask={
                name: value.pin_memory()
                for name, value in self.fingerprint_loss_mask.items()
            },
            modality_dropped=self.modality_dropped.pin_memory(),
        )


@dataclass(frozen=True)
class BatchFusionLayout:
    smiles_lengths: torch.Tensor
    atom_counts: torch.Tensor
    bond_counts: torch.Tensor
    atom_local_indices: torch.Tensor
    bond_local_indices: torch.Tensor
    max_core_length: int
    max_atom_count: int
    max_bond_count: int

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "BatchFusionLayout":
        return BatchFusionLayout(
            smiles_lengths=self.smiles_lengths.to(device, non_blocking=non_blocking),
            atom_counts=self.atom_counts.to(device, non_blocking=non_blocking),
            bond_counts=self.bond_counts.to(device, non_blocking=non_blocking),
            atom_local_indices=self.atom_local_indices.to(
                device, non_blocking=non_blocking
            ),
            bond_local_indices=self.bond_local_indices.to(
                device, non_blocking=non_blocking
            ),
            max_core_length=self.max_core_length,
            max_atom_count=self.max_atom_count,
            max_bond_count=self.max_bond_count,
        )

    def pin_memory(self) -> "BatchFusionLayout":
        return BatchFusionLayout(
            smiles_lengths=self.smiles_lengths.pin_memory(),
            atom_counts=self.atom_counts.pin_memory(),
            bond_counts=self.bond_counts.pin_memory(),
            atom_local_indices=self.atom_local_indices.pin_memory(),
            bond_local_indices=self.bond_local_indices.pin_memory(),
            max_core_length=self.max_core_length,
            max_atom_count=self.max_atom_count,
            max_bond_count=self.max_bond_count,
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
    fusion_layout: BatchFusionLayout
    masks: MaskPlan | None = None

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "MultimodalBatch":
        return MultimodalBatch(
            token_ids=self.token_ids.to(device, non_blocking=non_blocking),
            token_padding_mask=self.token_padding_mask.to(
                device, non_blocking=non_blocking
            ),
            graphs=self.graphs.to(device, non_blocking=non_blocking),
            descriptors=self.descriptors.to(device, non_blocking=non_blocking),
            descriptor_valid=self.descriptor_valid.to(
                device, non_blocking=non_blocking
            ),
            fingerprints=self.fingerprints.to(device, non_blocking=non_blocking),
            roles=self.roles.to(device, non_blocking=non_blocking),
            sample_ids=self.sample_ids,
            fusion_layout=self.fusion_layout.to(
                device, non_blocking=non_blocking
            ),
            masks=(
                None
                if self.masks is None
                else self.masks.to(device, non_blocking=non_blocking)
            ),
        )

    def pin_memory(self) -> "MultimodalBatch":
        return MultimodalBatch(
            token_ids=self.token_ids.pin_memory(),
            token_padding_mask=self.token_padding_mask.pin_memory(),
            graphs=self.graphs.pin_memory(),
            descriptors=self.descriptors.pin_memory(),
            descriptor_valid=self.descriptor_valid.pin_memory(),
            fingerprints=self.fingerprints.pin_memory(),
            roles=self.roles.pin_memory(),
            sample_ids=self.sample_ids,
            fusion_layout=self.fusion_layout.pin_memory(),
            masks=None if self.masks is None else self.masks.pin_memory(),
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
        if (
            self.metadata.get("kind") != CORPUS_KIND
            or self.metadata.get("format_version")
            not in {CORPUS_FORMAT_VERSION, GLOBAL_RDKIT_CORPUS_FORMAT_VERSION}
        ):
            raise ValueError(
                "Unsupported Stage 1 corpus artifact version; "
                "rerun scripts/stage1/prepare.py with a supported architecture"
            )
        self.format_version = int(self.metadata["format_version"])
        if self.metadata.get("identity_contract_version") != IDENTITY_CONTRACT_VERSION:
            raise ValueError(
                "Stage 1 corpus predates identity contract v1; regenerate the corpus"
            )
        artifact_hashes = self.metadata.get("artifact_hashes")
        if not isinstance(artifact_hashes, dict) or not artifact_hashes:
            raise ValueError("Corpus artifact hashes are missing; rerun scripts/stage1/prepare.py")
        for filename, expected_hash in artifact_hashes.items():
            if sha256_file(path / filename) != expected_hash:
                raise ValueError(f"Artifact hash mismatch: {filename}")
        self.descriptor_schema = DescriptorSchema.load(
            path / "descriptor_schema.json"
        )
        DescriptorStandardizer.load(
            path / "descriptor_scaler.json",
            expected_names=self.descriptor_schema.selected_names,
        )
        shard_manifest = json.loads(
            (path / "shard_manifest.json").read_text(encoding="utf-8")
        )
        if (
            shard_manifest.get("kind") != CORPUS_KIND
            or shard_manifest.get("format_version") != self.format_version
        ):
            raise ValueError("Unsupported Stage 1 shard manifest")
        self.shards = tuple(shard_manifest["shards"])
        shard_hashes = self.metadata.get("shard_hashes", {})
        for shard in self.shards:
            if shard.get("sha256") != shard_hashes.get(shard.get("path")):
                raise ValueError("Stage 1 shard manifest hash does not match metadata")
        selected_splits = ("train", "valid") if split == "all" else (split,)
        self._indices = tuple(
            np.load(path / f"{name}_index.npy", mmap_mode="r")
            for name in selected_splits
        )
        for index in self._indices:
            if index.dtype != INDEX_DTYPE:
                raise ValueError("Unsupported Stage 1 compact index dtype")
        self._lengths = tuple(len(index) for index in self._indices)
        self.shard_cache_size = shard_cache_size
        self._cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._verified_shards: set[str] = set()

    def __len__(self) -> int:
        return sum(self._lengths)

    @property
    def role_ids(self) -> np.ndarray:
        if len(self._indices) == 1:
            return self._indices[0]["role_id"]
        return np.concatenate([index["role_id"] for index in self._indices])

    @property
    def shard_ranges(self) -> tuple[tuple[int, int], ...]:
        ranges: list[tuple[int, int]] = []
        logical_offset = 0
        selected_splits = {
            str(self.shards[int(index[0]["shard_id"])]["split"])
            for index in self._indices
            if len(index)
        }
        for shard in self.shards:
            if shard["split"] not in selected_splits:
                continue
            count = int(shard["count"])
            ranges.append((logical_offset, count))
            logical_offset += count
        if logical_offset != len(self):
            raise ValueError("Stage 1 shard manifest does not match compact index")
        return tuple(ranges)

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
        if (
            payload.get("kind") != CORPUS_SHARD_KIND
            or payload.get("format_version") != self.format_version
        ):
            raise ValueError(f"Unsupported shard format: {relative_path}")
        samples = payload["samples"]
        self._cache[relative_path] = samples
        while len(self._cache) > self.shard_cache_size:
            self._cache.popitem(last=False)
        return samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        for compact, length in zip(self._indices, self._lengths, strict=True):
            if index < length:
                entry = compact[index]
                shard = self.shards[int(entry["shard_id"])]["path"]
                return self._load_shard(shard)[int(entry["offset"])]
            index -= length
        raise IndexError(index)

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        resolved: list[tuple[int, int]] = []
        for index in indices:
            if index < 0:
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError(index)
            for compact, length in zip(self._indices, self._lengths, strict=True):
                if index < length:
                    entry = compact[index]
                    resolved.append((int(entry["shard_id"]), int(entry["offset"])))
                    break
                index -= length
        loaded = {
            shard_id: self._load_shard(self.shards[shard_id]["path"])
            for shard_id in dict.fromkeys(shard_id for shard_id, _ in resolved)
        }
        return [loaded[shard_id][offset] for shard_id, offset in resolved]


def graph_record_from_sample(sample: dict[str, Any]) -> GraphRecord:
    return GraphRecord(
        atom_categorical=sample["atom_categorical"],
        atom_continuous=sample["atom_continuous"],
        bond_categorical=sample["bond_categorical"],
        bond_index=sample["bond_index"],
    )
