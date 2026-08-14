from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from rdkit import rdBase

from common.io import sha256_file
from stage1.data import MultimodalBatch
from stage1.masking import MultimodalPacker
from .config import STAGE2_TASKS


STAGE2_ARTIFACT_VERSION = 2
STAGE2_ARTIFACT_KIND = "ilume_stage2_object_data"


def _load_metadata(artifact_dir: Path) -> dict[str, Any]:
    metadata = json.loads(
        (artifact_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if (
        metadata.get("format_version") != STAGE2_ARTIFACT_VERSION
        or metadata.get("kind") != STAGE2_ARTIFACT_KIND
    ):
        raise ValueError(
            "Unsupported Stage 2 object data artifact; rerun "
            "scripts/stage2/prepare.py for Object v2"
        )
    return metadata


def _verify_artifact_file(
    artifact_dir: Path,
    metadata: dict[str, Any],
    relative: str,
) -> Path:
    path = artifact_dir / relative
    expected = metadata.get("artifact_hashes", {}).get(relative)
    if expected is None or sha256_file(path) != expected:
        raise ValueError(f"Stage 2 artifact hash mismatch: {relative}")
    return path


class Stage2EntityDataset:
    """Verified, preload-only Stage 2 entity sample store."""

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.metadata = _load_metadata(self.artifact_dir)
        if self.metadata.get("rdkit_version") != rdBase.rdkitVersion:
            raise ValueError("RDKit version does not match the Stage 2 artifact")
        index_path = _verify_artifact_file(
            self.artifact_dir, self.metadata, "entity_index.json"
        )
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if (
            payload.get("format_version") != STAGE2_ARTIFACT_VERSION
            or payload.get("kind") != STAGE2_ARTIFACT_KIND
        ):
            raise ValueError("Unsupported Stage 2 entity index format")
        self.entries = payload["entries"]
        for expected_id, entry in enumerate(self.entries):
            if int(entry.get("entity_id", -1)) != expected_id:
                raise ValueError("Stage 2 entity IDs must be contiguous and ordered")

        by_shard: dict[str, list[dict[str, Any]]] = {}
        for entry in self.entries:
            by_shard.setdefault(entry["shard"], []).append(entry)
        samples: list[dict[str, Any] | None] = [None] * len(self.entries)
        try:
            for relative, shard_entries in by_shard.items():
                path = _verify_artifact_file(
                    self.artifact_dir, self.metadata, relative
                )
                shard = torch.load(path, map_location="cpu", weights_only=False)
                if (
                    shard.get("format_version") != STAGE2_ARTIFACT_VERSION
                    or shard.get("kind") != STAGE2_ARTIFACT_KIND
                ):
                    raise ValueError(f"Unsupported Stage 2 entity shard: {relative}")
                shard_samples = shard["samples"]
                for entry in shard_entries:
                    offset = int(entry["offset"])
                    if not 0 <= offset < len(shard_samples):
                        raise ValueError(
                            f"Stage 2 entity shard offset is invalid: {relative}:{offset}"
                        )
                    samples[int(entry["entity_id"])] = shard_samples[offset]
        except MemoryError as error:
            raise MemoryError(
                "Stage 2 entity preload exhausted RAM; Object v2 does not "
                "fall back to shard loading"
            ) from error
        if any(sample is None for sample in samples):
            raise ValueError("Stage 2 entity preload is incomplete")
        self.samples = tuple(sample for sample in samples if sample is not None)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class Stage2TaskDataset:
    def __init__(self, artifact_dir: str | Path, task: str, split: str) -> None:
        if task not in STAGE2_TASKS:
            raise ValueError(f"Unknown Stage 2 task: {task}")
        if split not in {"train", "valid"}:
            raise ValueError("Stage 2 split must be train or valid")
        self.artifact_dir = Path(artifact_dir)
        self.task = task
        self.split = split
        metadata = _load_metadata(self.artifact_dir)
        relative = f"tasks/{task}_{split}.pt"
        path = _verify_artifact_file(self.artifact_dir, metadata, relative)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("format_version") != STAGE2_ARTIFACT_VERSION
            or payload.get("kind") != STAGE2_ARTIFACT_KIND
        ):
            raise ValueError("Unsupported Stage 2 task artifact format")
        if payload.get("task") != task or payload.get("split") != split:
            raise ValueError("Stage 2 task artifact identity mismatch")
        self.entity_indices = payload["entity_indices"]
        self.conditions = payload["conditions"]
        self.targets = payload["targets"]
        self.target_mask = payload["target_mask"]
        self.raw_targets = payload.get("raw_targets")
        self.source_rows = payload["source_rows"]
        self.condition_columns = tuple(payload["condition_columns"])
        self.target_columns = tuple(payload["target_columns"])
        if self.target_mask.shape != self.targets.shape:
            raise ValueError("Stage 2 target mask shape mismatch")
        if not torch.isfinite(self.conditions).all():
            raise ValueError("Stage 2 normalized conditions must be finite")
        if not torch.isfinite(self.targets[self.target_mask]).all():
            raise ValueError("Stage 2 normalized targets must be finite")
        if not torch.equal(
            self.targets.masked_select(~self.target_mask),
            torch.zeros_like(self.targets).masked_select(~self.target_mask),
        ):
            raise ValueError("Stage 2 missing normalized targets must be zero")
        if split == "valid":
            if self.raw_targets is None or self.raw_targets.shape != self.targets.shape:
                raise ValueError("Stage 2 validation artifact requires raw targets")
        elif self.raw_targets is not None:
            raise ValueError("Stage 2 train artifact must not duplicate raw targets")

    def __len__(self) -> int:
        return int(self.entity_indices.shape[0])


@dataclass(frozen=True)
class Stage2DeviceTaskData:
    entity_indices: torch.Tensor
    conditions: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    raw_targets: torch.Tensor | None

    @classmethod
    def from_dataset(
        cls, dataset: Stage2TaskDataset, device: torch.device
    ) -> "Stage2DeviceTaskData":
        return cls(
            entity_indices=dataset.entity_indices.to(device),
            conditions=dataset.conditions.to(device),
            targets=dataset.targets.to(device),
            target_mask=dataset.target_mask.to(device),
            raw_targets=(
                None
                if dataset.raw_targets is None
                else dataset.raw_targets.to(device)
            ),
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
            descriptors=self.descriptors,
            entities=self.entities.pin_memory(),
            unique_entity_ids=self.unique_entity_ids.pin_memory(),
            entity_positions=tuple(
                positions.pin_memory() for positions in self.entity_positions
            ),
        )


def pack_stage2_window(
    descriptors: Sequence[Stage2BatchDescriptor],
    task_datasets: dict[str, Stage2TaskDataset],
    entity_dataset: Stage2EntityDataset,
    packer: MultimodalPacker,
    *,
    pin_memory: bool,
) -> PackedStage2Window:
    window = tuple(descriptors)
    if not window:
        raise ValueError("Stage 2 packing window cannot be empty")
    unique_ids: list[int] = []
    local_by_global: dict[int, int] = {}
    positions_by_batch: list[torch.Tensor] = []
    for descriptor in window:
        global_slots = task_datasets[descriptor.task].entity_indices[
            descriptor.indices
        ]
        local_values: list[int] = []
        for value in global_slots.flatten().tolist():
            local = local_by_global.get(value)
            if local is None:
                local = len(unique_ids)
                local_by_global[value] = local
                unique_ids.append(value)
            local_values.append(local)
        positions_by_batch.append(
            torch.tensor(local_values, dtype=torch.long).reshape_as(global_slots)
        )
    try:
        entities = packer([entity_dataset[index] for index in unique_ids])
    except BaseException as error:
        tasks = ",".join(descriptor.task for descriptor in window)
        raise RuntimeError(
            f"Stage 2 packer failed for tasks={tasks}, entity_ids={unique_ids}"
        ) from error
    packed = PackedStage2Window(
        descriptors=window,
        entities=entities,
        unique_entity_ids=torch.tensor(unique_ids, dtype=torch.long),
        entity_positions=tuple(positions_by_batch),
    )
    return packed.pin_memory() if pin_memory else packed


def task_batch_counts(
    datasets: dict[str, Stage2TaskDataset], batch_size: int
) -> dict[str, int]:
    if batch_size <= 0:
        raise ValueError("Stage 2 batch size must be positive")
    return {
        task: math.ceil(len(datasets[task]) / batch_size)
        for task in STAGE2_TASKS
    }


def epoch_batch_schedule(
    datasets: dict[str, Stage2TaskDataset],
    batch_size: int,
    *,
    seed: int,
    epoch: int,
) -> list[Stage2BatchDescriptor]:
    if epoch <= 0:
        raise ValueError("Stage 2 epoch must be positive")
    descriptors: list[Stage2BatchDescriptor] = []
    for task_index, task in enumerate(STAGE2_TASKS):
        dataset = datasets[task]
        if len(dataset) == 0:
            raise ValueError(f"Stage 2 training dataset is empty: {task}")
        generator = torch.Generator().manual_seed(
            seed + 1_000_003 * epoch + 10_007 * (task_index + 1)
        )
        permutation = torch.randperm(len(dataset), generator=generator)
        descriptors.extend(
            Stage2BatchDescriptor(task, permutation[start : start + batch_size])
            for start in range(0, len(dataset), batch_size)
        )
    interleave = torch.Generator().manual_seed(
        seed + 2_000_003 * epoch + 500_009
    )
    order = torch.randperm(len(descriptors), generator=interleave).tolist()
    return [descriptors[index] for index in order]
