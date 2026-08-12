from __future__ import annotations

import json
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


STAGE2_ARTIFACT_VERSION = 2
IL_TASKS = ("density", "heat_capacity", "thermal_expansion")


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
        if self.metadata.get("format_version") != STAGE2_ARTIFACT_VERSION:
            raise ValueError("Unsupported Stage 2 artifact format")
        if self.metadata.get("rdkit_version") != rdBase.rdkitVersion:
            raise ValueError("RDKit version does not match the Stage 2 artifact")
        index_path = self.artifact_dir / "entity_index.json"
        expected_index_hash = self.metadata.get("artifact_hashes", {}).get(
            "entity_index.json"
        )
        if expected_index_hash is None or sha256_file(index_path) != expected_index_hash:
            raise ValueError("Stage 2 artifact hash mismatch: entity_index.json")
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("format_version") != STAGE2_ARTIFACT_VERSION:
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
        if payload.get("format_version") != STAGE2_ARTIFACT_VERSION:
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
        expected = metadata["artifact_hashes"].get(
            str(path.relative_to(self.artifact_dir))
        )
        if expected is None or sha256_file(path) != expected:
            raise ValueError(f"Stage 2 artifact hash mismatch: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format_version") != STAGE2_ARTIFACT_VERSION:
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
        self.system_offsets = payload["system_offsets"]
        self.system_rows = payload["system_rows"]
        self.condition_columns = tuple(payload["condition_columns"])
        self.target_columns = tuple(payload["target_columns"])
        if self.target_mask.shape != self.targets.shape:
            raise ValueError("Stage 2 target mask shape mismatch")
        if task in IL_TASKS:
            if (
                self.system_offsets.ndim != 1
                or self.system_offsets.numel() < 2
                or int(self.system_offsets[0]) != 0
                or int(self.system_offsets[-1]) != len(self)
                or self.system_rows.shape != (len(self),)
            ):
                raise ValueError("Invalid Stage 2 IL system index")
        elif self.system_offsets.numel() or self.system_rows.numel():
            raise ValueError("Non-IL Stage 2 tasks cannot define system indices")

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


class TaskCursor:
    def __init__(self, size: int, seed: int) -> None:
        if size <= 0:
            raise ValueError("TaskCursor requires a non-empty dataset")
        self.size = size
        self.seed = seed
        self.cycle = 0
        self.position = 0
        self._permutation = self._build_permutation()

    def _build_permutation(self) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.seed + self.cycle)
        return torch.randperm(self.size, generator=generator)

    def _next_with_cycles(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        if count <= 0:
            raise ValueError("TaskCursor count must be positive")
        chunks: list[torch.Tensor] = []
        cycle_chunks: list[torch.Tensor] = []
        remaining = count
        while remaining:
            available = self.size - self.position
            take = min(remaining, available)
            chunks.append(self._permutation[self.position : self.position + take])
            cycle_chunks.append(torch.full((take,), self.cycle, dtype=torch.long))
            self.position += take
            remaining -= take
            if self.position == self.size:
                self.cycle += 1
                self.position = 0
                self._permutation = self._build_permutation()
        return torch.cat(chunks), torch.cat(cycle_chunks)

    def next_indices(self, count: int) -> torch.Tensor:
        indices, _ = self._next_with_cycles(count)
        return indices

    def state_dict(self) -> dict[str, int]:
        return {"cycle": self.cycle, "position": self.position}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        cycle = int(state["cycle"])
        position = int(state["position"])
        if cycle < 0 or not 0 <= position < self.size:
            raise ValueError("Invalid Stage 2 task cursor state")
        self.cycle = cycle
        self.position = position
        self._permutation = self._build_permutation()


class ILSystemCursor:
    def __init__(
        self,
        system_offsets: torch.Tensor,
        system_rows: torch.Tensor,
        seed: int,
    ) -> None:
        if system_offsets.ndim != 1 or system_offsets.numel() < 2:
            raise ValueError("ILSystemCursor requires at least one system")
        if int(system_offsets[0]) != 0 or int(system_offsets[-1]) != len(system_rows):
            raise ValueError("Invalid IL system CSR index")
        self.system_offsets = system_offsets.to(dtype=torch.long, device="cpu")
        self.system_rows = system_rows.to(dtype=torch.long, device="cpu")
        self.seed = seed
        self.system_cursor = TaskCursor(len(system_offsets) - 1, seed)

    def _row_for_visit(self, system_id: int, cycle: int) -> int:
        start = int(self.system_offsets[system_id])
        end = int(self.system_offsets[system_id + 1])
        size = end - start
        if size == 1:
            return int(self.system_rows[start])
        row_cycle, offset = divmod(cycle, size)
        mixed_seed = (
            self.seed + 1_000_003 * system_id + 1_000_000_007 * row_cycle
        ) % (2**63 - 1)
        generator = torch.Generator().manual_seed(mixed_seed)
        selected = int(torch.randperm(size, generator=generator)[offset])
        return int(self.system_rows[start + selected])

    def next_indices(self, count: int) -> torch.Tensor:
        system_ids, cycles = self.system_cursor._next_with_cycles(count)
        return torch.tensor(
            [
                self._row_for_visit(int(system_id), int(cycle))
                for system_id, cycle in zip(system_ids, cycles, strict=True)
            ],
            dtype=torch.long,
        )

    def state_dict(self) -> dict[str, Any]:
        return {"kind": "il_system", "system_cursor": self.system_cursor.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("kind") != "il_system":
            raise ValueError("Invalid IL system cursor state")
        self.system_cursor.load_state_dict(state["system_cursor"])


class TaskBlockSampler:
    def __init__(
        self,
        probabilities: dict[str, float],
        block_size: int,
        seed: int,
    ) -> None:
        self.probabilities = dict(probabilities)
        self.block_size = block_size
        self.seed = seed
        tasks: list[str] = []
        for task in STAGE2_TASKS:
            quota = probabilities[task] * block_size
            if abs(quota - round(quota)) > 1.0e-8:
                raise ValueError("Task block quotas must be integers")
            tasks.extend([task] * round(quota))
        if len(tasks) != block_size:
            raise ValueError("Task block quotas do not fill the block")
        self._tasks = tuple(tasks)
        self._cache: dict[int, tuple[str, ...]] = {}

    def block(self, block_index: int) -> tuple[str, ...]:
        cached = self._cache.get(block_index)
        if cached is not None:
            return cached
        generator = torch.Generator().manual_seed(self.seed + block_index)
        order = torch.randperm(self.block_size, generator=generator).tolist()
        result = tuple(self._tasks[index] for index in order)
        self._cache = {block_index: result}
        return result

    def task_for_step(self, step: int) -> str:
        if step < 0:
            raise ValueError("Stage 2 step must be non-negative")
        block_index, offset = divmod(step, self.block_size)
        return self.block(block_index)[offset]
