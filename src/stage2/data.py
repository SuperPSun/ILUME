from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from rdkit import rdBase

from common.io import sha256_file
from stage1.data import MultimodalBatch
from stage1.masking import MultimodalPacker
from .registry import Stage2Registry


STAGE2_ARTIFACT_VERSION = 3
STAGE2_ARTIFACT_KIND = "ilume_stage2_object_data"


def _load_metadata(artifact_dir: Path) -> dict[str, Any]:
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != STAGE2_ARTIFACT_VERSION or metadata.get("kind") != STAGE2_ARTIFACT_KIND:
        raise ValueError("Unsupported Stage 2 object data artifact; rerun prepare for Object v3")
    return metadata


def load_artifact_registry(artifact_dir: str | Path) -> Stage2Registry:
    metadata = _load_metadata(Path(artifact_dir))
    return Stage2Registry.from_snapshot(
        metadata["registry"], registry_hash=metadata["registry_hash"],
        catalog_sha256=metadata["catalog_sha256"],
    )


def _verify_artifact_file(artifact_dir: Path, metadata: dict[str, Any], relative: str) -> Path:
    path = artifact_dir / relative
    expected = metadata.get("artifact_hashes", {}).get(relative)
    if expected is None or sha256_file(path) != expected:
        raise ValueError(f"Stage 2 artifact hash mismatch: {relative}")
    return path


class Stage2EntityDataset:
    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.metadata = _load_metadata(self.artifact_dir)
        if self.metadata.get("rdkit_version") != rdBase.rdkitVersion:
            raise ValueError("RDKit version does not match the Stage 2 artifact")
        path = _verify_artifact_file(self.artifact_dir, self.metadata, "entity_index.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format_version") != STAGE2_ARTIFACT_VERSION or payload.get("kind") != STAGE2_ARTIFACT_KIND:
            raise ValueError("Unsupported Stage 2 entity index format")
        self.entries = payload["entries"]
        by_shard: dict[str, list[dict[str, Any]]] = {}
        for expected_id, entry in enumerate(self.entries):
            if int(entry.get("entity_id", -1)) != expected_id:
                raise ValueError("Stage 2 entity IDs must be contiguous")
            by_shard.setdefault(entry["shard"], []).append(entry)
        samples: list[dict[str, Any] | None] = [None] * len(self.entries)
        try:
            for relative, entries in by_shard.items():
                shard = torch.load(_verify_artifact_file(self.artifact_dir, self.metadata, relative), map_location="cpu", weights_only=False)
                if shard.get("format_version") != STAGE2_ARTIFACT_VERSION or shard.get("kind") != STAGE2_ARTIFACT_KIND:
                    raise ValueError(f"Unsupported Stage 2 entity shard: {relative}")
                for entry in entries:
                    samples[int(entry["entity_id"])] = shard["samples"][int(entry["offset"])]
        except MemoryError as error:
            raise MemoryError("Stage 2 entity preload exhausted RAM; Object v3 does not fall back") from error
        if any(sample is None for sample in samples):
            raise ValueError("Stage 2 entity preload is incomplete")
        self.samples = tuple(sample for sample in samples if sample is not None)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class Stage2TaskDataset:
    def __init__(self, artifact_dir: str | Path, task: str, split: str) -> None:
        if split not in {"train", "valid"}:
            raise ValueError("Stage 2 split must be train or valid")
        self.artifact_dir = Path(artifact_dir)
        self.task = task
        self.split = split
        metadata = _load_metadata(self.artifact_dir)
        registry = load_artifact_registry(self.artifact_dir)
        self.spec = registry.by_id(task)
        relative = f"tasks/{task}/{split}.pt"
        payload = torch.load(_verify_artifact_file(self.artifact_dir, metadata, relative), map_location="cpu", weights_only=False)
        if payload.get("format_version") != STAGE2_ARTIFACT_VERSION or payload.get("kind") != STAGE2_ARTIFACT_KIND:
            raise ValueError("Unsupported Stage 2 task artifact format")
        if payload.get("task") != task or payload.get("split") != split:
            raise ValueError("Stage 2 task artifact identity mismatch")
        self.entity_indices = payload["entity_indices"]
        self.conditions = payload["conditions"]
        self.source_rows = payload["source_rows"]
        self.condition_columns = tuple(payload["condition_columns"])
        self.target_columns = tuple(payload["target_columns"])
        self.targets = payload.get("targets")
        self.target_mask = payload.get("target_mask")
        self.raw_targets = payload.get("raw_targets")
        self.atom_target_values = payload.get("atom_target_values")
        self.atom_target_offsets = payload.get("atom_target_offsets")
        self.atom_target_mask = payload.get("atom_target_mask")
        self.raw_atom_target_values = payload.get("raw_atom_target_values")
        self.mol_ids = tuple(payload.get("mol_ids", ()))
        if self.spec.target_level == "object":
            if self.targets is None or self.target_mask is None or self.target_mask.shape != self.targets.shape:
                raise ValueError("Stage 2 object target tensor contract mismatch")
            if split == "valid" and self.raw_targets is None:
                raise ValueError("Stage 2 validation object targets require raw values")
        else:
            if self.atom_target_values is None or self.atom_target_offsets is None or self.atom_target_mask is None:
                raise ValueError("Stage 2 atom target tensor contract mismatch")
            if self.atom_target_offsets.shape != (len(self.entity_indices) + 1,):
                raise ValueError("Stage 2 atom offsets shape mismatch")
            if int(self.atom_target_offsets[-1]) != len(self.atom_target_values) or self.atom_target_mask.shape != self.atom_target_values.shape:
                raise ValueError("Stage 2 ragged atom target contract mismatch")
            if len(self.mol_ids) != len(self.entity_indices):
                raise ValueError("Stage 2 atom mol_id count mismatch")
            if split == "valid" and self.raw_atom_target_values is None:
                raise ValueError("Stage 2 validation atom targets require raw values")
        if not torch.isfinite(self.conditions).all():
            raise ValueError("Stage 2 normalized conditions must be finite")

    def __len__(self) -> int:
        return int(self.entity_indices.shape[0])


@dataclass(frozen=True)
class Stage2DeviceTaskData:
    entity_indices: torch.Tensor
    conditions: torch.Tensor
    targets: torch.Tensor | None
    target_mask: torch.Tensor | None
    raw_targets: torch.Tensor | None
    atom_target_values: torch.Tensor | None
    atom_target_offsets: torch.Tensor | None
    atom_target_mask: torch.Tensor | None
    raw_atom_target_values: torch.Tensor | None

    @classmethod
    def from_dataset(cls, dataset: Stage2TaskDataset, device: torch.device) -> "Stage2DeviceTaskData":
        move = lambda value: None if value is None else value.to(device)
        return cls(
            entity_indices=dataset.entity_indices.to(device), conditions=dataset.conditions.to(device),
            targets=move(dataset.targets), target_mask=move(dataset.target_mask), raw_targets=move(dataset.raw_targets),
            atom_target_values=move(dataset.atom_target_values), atom_target_offsets=move(dataset.atom_target_offsets),
            atom_target_mask=move(dataset.atom_target_mask), raw_atom_target_values=move(dataset.raw_atom_target_values),
        )


@dataclass(frozen=True)
class Stage2BatchDescriptor:
    task: str
    indices: torch.Tensor


@dataclass(frozen=True)
class PackedStage2Window:
    descriptors: tuple[Stage2BatchDescriptor, ...]
    entities: MultimodalBatch
    unique_entity_ids: torch.Tensor
    entity_positions: tuple[torch.Tensor, ...]

    def pin_memory(self) -> "PackedStage2Window":
        return PackedStage2Window(
            self.descriptors, self.entities.pin_memory(), self.unique_entity_ids.pin_memory(),
            tuple(value.pin_memory() for value in self.entity_positions),
        )


def pack_stage2_window(
    descriptors: Sequence[Stage2BatchDescriptor], task_datasets: dict[str, Stage2TaskDataset],
    entity_dataset: Stage2EntityDataset, packer: MultimodalPacker, *, pin_memory: bool,
) -> PackedStage2Window:
    window = tuple(descriptors)
    if not window:
        raise ValueError("Stage 2 packing window cannot be empty")
    unique_ids: list[int] = []
    local_by_global: dict[int, int] = {}
    positions: list[torch.Tensor] = []
    for descriptor in window:
        slots = task_datasets[descriptor.task].entity_indices[descriptor.indices]
        local_values: list[int] = []
        for value in slots.flatten().tolist():
            if value not in local_by_global:
                local_by_global[value] = len(unique_ids)
                unique_ids.append(value)
            local_values.append(local_by_global[value])
        positions.append(torch.tensor(local_values, dtype=torch.long).reshape_as(slots))
    try:
        entities = packer([entity_dataset[index] for index in unique_ids])
    except BaseException as error:
        raise RuntimeError(f"Stage 2 packer failed for tasks={','.join(value.task for value in window)}") from error
    result = PackedStage2Window(window, entities, torch.tensor(unique_ids, dtype=torch.long), tuple(positions))
    return result.pin_memory() if pin_memory else result


def task_batch_counts(datasets: dict[str, Stage2TaskDataset], batch_size: int) -> dict[str, int]:
    if batch_size <= 0:
        raise ValueError("Stage 2 batch size must be positive")
    return {task: math.ceil(len(dataset) / batch_size) for task, dataset in sorted(datasets.items())}


def _seed(seed: int, epoch: int, task: str, purpose: str) -> int:
    digest = hashlib.blake2b(f"{seed}\0{epoch}\0{task}\0{purpose}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def epoch_batch_schedule(
    datasets: dict[str, Stage2TaskDataset], batch_size: int, *, seed: int, epoch: int,
) -> list[Stage2BatchDescriptor]:
    if epoch <= 0:
        raise ValueError("Stage 2 epoch must be positive")
    queues: dict[str, list[Stage2BatchDescriptor]] = {}
    for task, dataset in sorted(datasets.items()):
        if len(dataset) == 0:
            raise ValueError(f"Stage 2 training dataset is empty: {task}")
        generator = torch.Generator().manual_seed(_seed(seed, epoch, task, "samples"))
        order = torch.randperm(len(dataset), generator=generator)
        queues[task] = [Stage2BatchDescriptor(task, order[start:start + batch_size]) for start in range(0, len(dataset), batch_size)]
    schedule: list[Stage2BatchDescriptor] = []
    round_index = 0
    while queues:
        active = sorted(queues)
        random.Random(_seed(seed, epoch, str(round_index), "round")).shuffle(active)
        if len(active) > 1 and schedule and active[0] == schedule[-1].task:
            active = active[1:] + active[:1]
        for task in active:
            schedule.append(queues[task].pop(0))
        queues = {task: queue for task, queue in queues.items() if queue}
        round_index += 1
    return schedule


__all__ = [
    "PackedStage2Window", "STAGE2_ARTIFACT_KIND", "STAGE2_ARTIFACT_VERSION",
    "Stage2BatchDescriptor", "Stage2DeviceTaskData", "Stage2EntityDataset",
    "Stage2TaskDataset", "epoch_batch_schedule", "load_artifact_registry",
    "pack_stage2_window", "task_batch_counts",
]
