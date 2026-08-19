from __future__ import annotations

import hashlib
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from rdkit import rdBase

from common.io import sha256_file
from stage1.data import MultimodalBatch
from stage1.masking import MultimodalPacker
from stage1.features import ROLE_TO_ID
from .registry import Stage2Registry


STAGE2_ARTIFACT_VERSION = 3
STAGE2_ARTIFACT_KIND = "ilume_stage2_object_data"
STAGE2_PREPARATION_CONTRACT_VERSION = 3
STAGE2_TENSOR_CONTRACT = {
    "conditions": "task_train_normalized_float32",
    "object_targets": "task_train_normalized_float32_masked_zero",
    "atom_targets": "ragged_molecule_equal_train_normalized_float32",
    "validation_raw_targets": "float32",
}


def _load_metadata(artifact_dir: Path) -> dict[str, Any]:
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != STAGE2_ARTIFACT_VERSION or metadata.get("kind") != STAGE2_ARTIFACT_KIND:
        raise ValueError("Unsupported Stage 2 object data artifact; rerun prepare for Object v3")
    if metadata.get("preparation_contract_version") != STAGE2_PREPARATION_CONTRACT_VERSION:
        raise ValueError("Stage 2 artifact predates the current Object v3 preparation contract; rerun prepare")
    if metadata.get("tensor_contract") != STAGE2_TENSOR_CONTRACT:
        raise ValueError("Stage 2 artifact tensor contract mismatch; rerun prepare")
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
            if self.target_mask.dtype != torch.bool or not torch.isfinite(self.targets).all():
                raise ValueError("Stage 2 normalized object targets/mask are invalid")
        else:
            if self.atom_target_values is None or self.atom_target_offsets is None or self.atom_target_mask is None:
                raise ValueError("Stage 2 atom target tensor contract mismatch")
            if self.atom_target_offsets.shape != (len(self.entity_indices) + 1,):
                raise ValueError("Stage 2 atom offsets shape mismatch")
            if int(self.atom_target_offsets[-1]) != len(self.atom_target_values) or self.atom_target_mask.shape != self.atom_target_values.shape:
                raise ValueError("Stage 2 ragged atom target contract mismatch")
            if self.atom_target_mask.dtype != torch.bool or not torch.isfinite(self.atom_target_values).all():
                raise ValueError("Stage 2 normalized atom targets/mask are invalid")
            if int(self.atom_target_offsets[0]) != 0 or not bool(
                (self.atom_target_offsets[1:] >= self.atom_target_offsets[:-1]).all()
            ):
                raise ValueError("Stage 2 atom offsets must be ordered from zero")
            for start, end in zip(
                self.atom_target_offsets[:-1].tolist(),
                self.atom_target_offsets[1:].tolist(), strict=True,
            ):
                if start == end or not bool(self.atom_target_mask[start:end].any()):
                    raise ValueError("Stage 2 atom sample requires a supervised atom")
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

    @classmethod
    def from_dataset(cls, dataset: Stage2TaskDataset, device: torch.device) -> "Stage2DeviceTaskData":
        move = lambda value: None if value is None else value.to(device)
        return cls(
            entity_indices=dataset.entity_indices.to(device), conditions=dataset.conditions.to(device),
            targets=move(dataset.targets), target_mask=move(dataset.target_mask), raw_targets=move(dataset.raw_targets),
        )


@dataclass(frozen=True)
class Stage2BatchDescriptor:
    task: str
    indices: torch.Tensor


@dataclass(frozen=True)
class PackedAtomTargets:
    values: torch.Tensor
    mask: torch.Tensor
    raw_values: torch.Tensor | None
    molecule_offsets: torch.Tensor
    atom_state_indices: torch.Tensor
    atom_sample_indices: torch.Tensor

    def pin_memory(self) -> "PackedAtomTargets":
        return PackedAtomTargets(
            self.values.pin_memory(), self.mask.pin_memory(),
            None if self.raw_values is None else self.raw_values.pin_memory(),
            self.molecule_offsets.pin_memory(), self.atom_state_indices.pin_memory(),
            self.atom_sample_indices.pin_memory(),
        )

    def to(self, device: torch.device, *, non_blocking: bool) -> "PackedAtomTargets":
        move = lambda value: None if value is None else value.to(device, non_blocking=non_blocking)
        return PackedAtomTargets(
            self.values.to(device, non_blocking=non_blocking),
            self.mask.to(device, non_blocking=non_blocking), move(self.raw_values),
            self.molecule_offsets.to(device, non_blocking=non_blocking),
            self.atom_state_indices.to(device, non_blocking=non_blocking),
            self.atom_sample_indices.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class PackedStage2Batch:
    descriptor: Stage2BatchDescriptor
    row_indices: torch.Tensor
    entities: MultimodalBatch | None
    unique_entity_ids: torch.Tensor | None
    entity_positions: torch.Tensor | None
    atom_targets: PackedAtomTargets | None

    def pin_memory(self) -> "PackedStage2Batch":
        return PackedStage2Batch(
            self.descriptor, self.row_indices.pin_memory(),
            None if self.entities is None else self.entities.pin_memory(),
            None if self.unique_entity_ids is None else self.unique_entity_ids.pin_memory(),
            None if self.entity_positions is None else self.entity_positions.pin_memory(),
            None if self.atom_targets is None else self.atom_targets.pin_memory(),
        )

    def to(self, device: torch.device, *, non_blocking: bool) -> "PackedStage2Batch":
        return PackedStage2Batch(
            self.descriptor, self.row_indices.to(device, non_blocking=non_blocking),
            None if self.entities is None else self.entities.to(device, non_blocking=non_blocking),
            None if self.unique_entity_ids is None else self.unique_entity_ids.to(device, non_blocking=non_blocking),
            None if self.entity_positions is None else self.entity_positions.to(device, non_blocking=non_blocking),
            None if self.atom_targets is None else self.atom_targets.to(device, non_blocking=non_blocking),
        )


def _pack_atom_targets(
    dataset: Stage2TaskDataset, descriptor: Stage2BatchDescriptor,
    entities: MultimodalBatch, entity_positions: torch.Tensor, *, include_raw: bool,
) -> PackedAtomTargets:
    if dataset.atom_target_values is None or dataset.atom_target_offsets is None or dataset.atom_target_mask is None:
        raise ValueError("Missing Stage 2 atom target store")
    if entity_positions.shape[1] != 1:
        raise ValueError("Atom property batch requires one entity slot")
    counts = torch.bincount(entities.graphs.atom_batch, minlength=len(entities.sample_ids))
    unique_offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    values: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    raw_values: list[torch.Tensor] = []
    state_indices: list[torch.Tensor] = []
    lengths: list[int] = []
    for sample_row, artifact_row in enumerate(descriptor.indices.tolist()):
        target_start = int(dataset.atom_target_offsets[artifact_row])
        target_end = int(dataset.atom_target_offsets[artifact_row + 1])
        local_entity = int(entity_positions[sample_row, 0])
        atom_start = int(unique_offsets[local_entity])
        atom_end = int(unique_offsets[local_entity + 1])
        if target_end - target_start != atom_end - atom_start:
            raise ValueError("Stage 2 atom target count does not match packed Stage 1 atoms")
        values.append(dataset.atom_target_values[target_start:target_end])
        masks.append(dataset.atom_target_mask[target_start:target_end])
        if include_raw:
            if dataset.raw_atom_target_values is None:
                raise ValueError("Stage 2 validation atom batch requires raw targets")
            raw_values.append(dataset.raw_atom_target_values[target_start:target_end])
        state_indices.append(torch.arange(atom_start, atom_end, dtype=torch.long))
        lengths.append(target_end - target_start)
    molecule_offsets = torch.cat((torch.zeros(1, dtype=torch.long), torch.tensor(lengths).cumsum(0)))
    atom_sample_indices = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
    return PackedAtomTargets(
        torch.cat(values), torch.cat(masks), torch.cat(raw_values) if include_raw else None,
        molecule_offsets, torch.cat(state_indices), atom_sample_indices,
    )


def pack_stage2_batch(
    descriptor: Stage2BatchDescriptor, task_datasets: dict[str, Stage2TaskDataset],
    entity_dataset: Stage2EntityDataset, packer: MultimodalPacker, *,
    needs_entities: bool, include_raw_atom_targets: bool, pin_memory: bool,
) -> PackedStage2Batch:
    if not needs_entities:
        result = PackedStage2Batch(descriptor, descriptor.indices, None, None, None, None)
        return result.pin_memory() if pin_memory else result
    dataset = task_datasets[descriptor.task]
    unique_ids: list[int] = []
    local_by_global: dict[int, int] = {}
    slots = dataset.entity_indices[descriptor.indices]
    local_values: list[int] = []
    for value in slots.flatten().tolist():
        if value not in local_by_global:
            local_by_global[value] = len(unique_ids)
            unique_ids.append(value)
        local_values.append(local_by_global[value])
    positions = torch.tensor(local_values, dtype=torch.long).reshape_as(slots)
    try:
        entities = packer([entity_dataset[index] for index in unique_ids])
    except BaseException as error:
        raise RuntimeError(f"Stage 2 packer failed for task={descriptor.task}") from error
    atom_targets = (
        _pack_atom_targets(dataset, descriptor, entities, positions, include_raw=include_raw_atom_targets)
        if dataset.spec.target_level == "atom" else None
    )
    result = PackedStage2Batch(
        descriptor, descriptor.indices, entities, torch.tensor(unique_ids, dtype=torch.long),
        positions, atom_targets,
    )
    return result.pin_memory() if pin_memory else result


def task_batch_counts(datasets: dict[str, Stage2TaskDataset], batch_size: int) -> dict[str, int]:
    if batch_size <= 0:
        raise ValueError("Stage 2 batch size must be positive")
    return {task: math.ceil(len(dataset) / batch_size) for task, dataset in sorted(datasets.items())}


def validate_runtime_task_contract(
    dataset: Stage2TaskDataset, entity_dataset: Stage2EntityDataset,
    *, loss_mode: str,
) -> None:
    expected_slots = 2 if dataset.spec.topology in {"ionic_liquid", "interaction"} else 1
    if dataset.entity_indices.ndim != 2 or dataset.entity_indices.shape[1] != expected_slots:
        raise ValueError(f"Stage 2 task topology mismatch: {dataset.task}/{dataset.split}")
    role_ids = torch.tensor(
        [int(entry["role_id"]) for entry in entity_dataset.entries], dtype=torch.long
    )[dataset.entity_indices]
    for slot, policy in enumerate(dataset.spec.role_policy):
        if policy in ROLE_TO_ID and not bool((role_ids[:, slot] == ROLE_TO_ID[policy]).all()):
            raise ValueError(f"Stage 2 task role mismatch: {dataset.task}/{dataset.split}")
    if dataset.spec.target_level == "object":
        assert dataset.target_mask is not None
        if loss_mode == "element_mean" and not bool(dataset.target_mask.all()):
            raise ValueError(f"Stage 2 element-mean task has missing targets: {dataset.task}/{dataset.split}")
        if loss_mode == "masked_target_macro" and not bool(dataset.target_mask.any(dim=1).all()):
            raise ValueError(f"Stage 2 masked task has an unsupervised row: {dataset.task}/{dataset.split}")


def _seed(seed: int, epoch: int, task: str, purpose: str) -> int:
    digest = hashlib.blake2b(f"{seed}\0{epoch}\0{task}\0{purpose}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def epoch_batch_schedule(
    datasets: dict[str, Stage2TaskDataset], batch_size: int, *, seed: int, epoch: int,
) -> list[Stage2BatchDescriptor]:
    if epoch <= 0:
        raise ValueError("Stage 2 epoch must be positive")
    queues: dict[str, deque[Stage2BatchDescriptor]] = {}
    for task, dataset in sorted(datasets.items()):
        if len(dataset) == 0:
            raise ValueError(f"Stage 2 training dataset is empty: {task}")
        generator = torch.Generator().manual_seed(_seed(seed, epoch, task, "samples"))
        order = torch.randperm(len(dataset), generator=generator)
        queues[task] = deque(
            Stage2BatchDescriptor(task, order[start:start + batch_size])
            for start in range(0, len(dataset), batch_size)
        )
    schedule: list[Stage2BatchDescriptor] = []
    round_index = 0
    while queues:
        active = sorted(queues)
        random.Random(_seed(seed, epoch, str(round_index), "round")).shuffle(active)
        if len(active) > 1 and schedule and active[0] == schedule[-1].task:
            active = active[1:] + active[:1]
        for task in active:
            schedule.append(queues[task].popleft())
        queues = {task: queue for task, queue in queues.items() if queue}
        round_index += 1
    return schedule


__all__ = [
    "PackedAtomTargets", "PackedStage2Batch", "STAGE2_ARTIFACT_KIND", "STAGE2_ARTIFACT_VERSION",
    "STAGE2_PREPARATION_CONTRACT_VERSION",
    "STAGE2_TENSOR_CONTRACT",
    "Stage2BatchDescriptor", "Stage2DeviceTaskData", "Stage2EntityDataset",
    "Stage2TaskDataset", "epoch_batch_schedule", "load_artifact_registry",
    "pack_stage2_batch", "task_batch_counts", "validate_runtime_task_contract",
]
