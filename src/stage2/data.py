from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from rdkit import rdBase
from torch.utils.data import Dataset

from common.io import sha256_file
from stage1.data import MultimodalBatch
from stage1.masking import MultimodalPacker
from .config import STAGE2_TASKS


STAGE2_ARTIFACT_VERSION = 1
STAGE2_ARTIFACT_KIND = "ilume_stage2_object_data"


class Stage2EntityDataset(Dataset):
    def __init__(
        self,
        artifact_dir: str | Path,
        shard_cache_size: int = 4,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.metadata = json.loads(
            (self.artifact_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if (
            self.metadata.get("format_version") != STAGE2_ARTIFACT_VERSION
            or self.metadata.get("kind") != STAGE2_ARTIFACT_KIND
        ):
            raise ValueError("Unsupported Stage 2 object data artifact")
        if self.metadata.get("rdkit_version") != rdBase.rdkitVersion:
            raise ValueError("RDKit version does not match the Stage 2 artifact")
        index_path = self.artifact_dir / "entity_index.json"
        expected_index_hash = self.metadata.get("artifact_hashes", {}).get(
            "entity_index.json"
        )
        if expected_index_hash is None or sha256_file(index_path) != expected_index_hash:
            raise ValueError("Stage 2 artifact hash mismatch: entity_index.json")
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if (
            payload.get("format_version") != STAGE2_ARTIFACT_VERSION
            or payload.get("kind") != STAGE2_ARTIFACT_KIND
        ):
            raise ValueError("Unsupported Stage 2 entity index format")
        self.entries = payload["entries"]
        self.shard_cache_size = shard_cache_size
        self._cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._verified: set[str] = set()

    def __len__(self) -> int:
        return len(self.entries)

    def _load_shard(self, relative: str) -> list[dict[str, Any]]:
        if relative in self._cache:
            samples = self._cache.pop(relative)
            self._cache[relative] = samples
            return samples
        path = self.artifact_dir / relative
        if relative not in self._verified:
            expected = self.metadata["artifact_hashes"].get(relative)
            if expected is None or sha256_file(path) != expected:
                raise ValueError(f"Stage 2 artifact hash mismatch: {relative}")
            self._verified.add(relative)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("format_version") != STAGE2_ARTIFACT_VERSION
            or payload.get("kind") != STAGE2_ARTIFACT_KIND
        ):
            raise ValueError(f"Unsupported Stage 2 entity shard: {relative}")
        samples = payload["samples"]
        self._cache[relative] = samples
        while len(self._cache) > self.shard_cache_size:
            self._cache.popitem(last=False)
        return samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        return self._load_shard(entry["shard"])[int(entry["offset"])]


class Stage2TaskDataset:
    def __init__(self, artifact_dir: str | Path, task: str, split: str) -> None:
        if task not in STAGE2_TASKS:
            raise ValueError(f"Unknown Stage 2 task: {task}")
        if split not in {"train", "valid"}:
            raise ValueError("Stage 2 split must be train or valid")
        self.artifact_dir = Path(artifact_dir)
        path = self.artifact_dir / "tasks" / f"{task}_{split}.pt"
        metadata = json.loads(
            (self.artifact_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if (
            metadata.get("format_version") != STAGE2_ARTIFACT_VERSION
            or metadata.get("kind") != STAGE2_ARTIFACT_KIND
        ):
            raise ValueError("Unsupported Stage 2 object data artifact")
        expected = metadata["artifact_hashes"].get(
            str(path.relative_to(self.artifact_dir))
        )
        if expected is None or sha256_file(path) != expected:
            raise ValueError(f"Stage 2 artifact hash mismatch: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("format_version") != STAGE2_ARTIFACT_VERSION
            or payload.get("kind") != STAGE2_ARTIFACT_KIND
        ):
            raise ValueError("Unsupported Stage 2 task artifact format")
        if payload.get("task") != task or payload.get("split") != split:
            raise ValueError("Stage 2 task artifact identity mismatch")
        self.task = task
        self.split = split
        self.entity_indices = payload["entity_indices"]
        self.conditions = payload["conditions"]
        self.targets = payload["targets"]
        self.target_mask = payload["target_mask"]
        self.source_rows = payload["source_rows"]
        self.condition_columns = tuple(payload["condition_columns"])
        self.target_columns = tuple(payload["target_columns"])
        if self.target_mask.shape != self.targets.shape:
            raise ValueError("Stage 2 target mask shape mismatch")

    def __len__(self) -> int:
        return int(self.entity_indices.shape[0])


@dataclass(frozen=True)
class Stage2Batch:
    task: str
    entities: MultimodalBatch
    entity_positions: torch.Tensor
    conditions: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    teacher_embeddings: torch.Tensor

    def to(self, device: torch.device | str) -> "Stage2Batch":
        return Stage2Batch(
            task=self.task,
            entities=self.entities.to(device),
            entity_positions=self.entity_positions.to(device),
            conditions=self.conditions.to(device),
            targets=self.targets.to(device),
            target_mask=self.target_mask.to(device),
            teacher_embeddings=self.teacher_embeddings.to(device),
        )


def build_stage2_batch(
    task_dataset: Stage2TaskDataset,
    indices: Sequence[int] | torch.Tensor,
    entity_dataset: Stage2EntityDataset,
    packer: MultimodalPacker,
    teacher_embeddings: torch.Tensor,
    scalers: dict[str, Any],
) -> Stage2Batch:
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    global_slots = task_dataset.entity_indices[index_tensor]
    unique_ids: list[int] = []
    local_by_global: dict[int, int] = {}
    local_values: list[int] = []
    for value in global_slots.flatten().tolist():
        local = local_by_global.get(value)
        if local is None:
            local = len(unique_ids)
            local_by_global[value] = local
            unique_ids.append(value)
        local_values.append(local)
    positions = torch.tensor(local_values, dtype=torch.long).reshape_as(global_slots)
    entities = packer([entity_dataset[index] for index in unique_ids])
    unique_teacher = teacher_embeddings[torch.tensor(unique_ids, dtype=torch.long)]
    conditions = task_dataset.conditions[index_tensor].clone()
    if task_dataset.condition_columns:
        temperature = scalers["temperature_K"]
        conditions = (conditions - float(temperature["mean"])) / float(
            temperature["scale"]
        )
    targets = task_dataset.targets[index_tensor].clone()
    target_mask = task_dataset.target_mask[index_tensor].clone()
    for column, name in enumerate(task_dataset.target_columns):
        stats = scalers["targets"][name]
        standardized = (targets[:, column] - float(stats["mean"])) / float(
            stats["scale"]
        )
        targets[:, column] = torch.where(
            target_mask[:, column], standardized, torch.zeros_like(standardized)
        )
    return Stage2Batch(
        task=task_dataset.task,
        entities=entities,
        entity_positions=positions,
        conditions=conditions,
        targets=targets,
        target_mask=target_mask,
        teacher_embeddings=unique_teacher,
    )


@dataclass(frozen=True)
class Stage2BatchDescriptor:
    task: str
    indices: torch.Tensor


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
