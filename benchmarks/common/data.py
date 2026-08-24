from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from stage2.registry import load_stage2_registry
from stage3.config import load_stage3_config
from stage3.data import (
    canonicalize_smiles,
    finite_float,
    resolve_task_registry,
    source_path,
    test_path,
)

from .config import BenchmarkConfig, BenchmarkName


@dataclass(frozen=True)
class BenchmarkTask:
    benchmark: BenchmarkName
    task_id: str
    slots: tuple[str, ...]
    condition_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    train_paths: tuple[Path, ...]
    valid_paths: tuple[Path, ...]
    test_path: Path
    fold: int | None
    meta_group: str | None
    registry_payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name in ("train_paths", "valid_paths"):
            value[name] = [path.as_posix() for path in value[name]]
        value["test_path"] = self.test_path.as_posix()
        return value


@dataclass(frozen=True)
class RawDataset:
    components: tuple[tuple[str, ...], ...]
    component_count: int
    conditions: np.ndarray
    targets: np.ndarray
    source_rows: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.components)


def resolve_task(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
) -> BenchmarkTask:
    if benchmark == "stage3":
        if not config.stage3.enabled or fold not in config.stage3.folds:
            raise ValueError("Stage 3 benchmark requires an enabled configured --fold")
        authority = load_stage3_config(config.data.stage3_authority_config)
        if authority.data.task_catalog != config.data.task_catalog:
            raise ValueError("Benchmark and Stage 3 authority task_catalog differ")
        expected_stage3_dir = config.data.data_root / "stage3"
        if authority.data.stage3_dir != expected_stage3_dir:
            raise ValueError("Benchmark and Stage 3 authority data roots differ")
        registry = resolve_task_registry(authority)
        allowed = (
            {name for name, value in registry.items() if value.enabled}
            if config.stage3.tasks == "all"
            else set(config.stage3.tasks)
        )
        if task_id not in allowed or task_id not in registry:
            raise ValueError(f"Unknown or disabled Stage 3 benchmark task: {task_id}")
        spec = registry[task_id]
        train_folds = tuple(value for value in range(1, 6) if value != fold)
        return BenchmarkTask(
            benchmark=benchmark,
            task_id=task_id,
            slots=spec.identity_columns,
            condition_columns=spec.condition_columns,
            target_columns=(spec.target_column,),
            train_paths=tuple(source_path(authority, spec, value) for value in train_folds),
            valid_paths=(source_path(authority, spec, fold),),
            test_path=test_path(authority, spec),
            fold=fold,
            meta_group=spec.meta_group,
            registry_payload=spec.to_dict(),
        )
    if benchmark != "stage2_physics":
        raise ValueError(f"Unknown benchmark domain: {benchmark}")
    if fold is not None:
        raise ValueError("Stage 2 physics benchmark does not accept --fold")
    if not config.stage2_physics.enabled or task_id not in config.stage2_physics.tasks:
        raise ValueError(f"Unknown or disabled Stage 2 physics task: {task_id}")
    registry = load_stage2_registry(config.data.task_catalog)
    spec = registry.by_id(task_id)
    return BenchmarkTask(
        benchmark=benchmark,
        task_id=task_id,
        slots=spec.entity_columns,
        condition_columns=spec.condition_columns,
        target_columns=spec.target_columns,
        train_paths=(spec.dataset.split_path(config.data.data_root, "train"),),
        valid_paths=(spec.dataset.split_path(config.data.data_root, "valid"),),
        test_path=spec.dataset.split_path(config.data.data_root, "test"),
        fold=None,
        meta_group=None,
        registry_payload=spec.to_dict(),
    )


def configured_tasks(config: BenchmarkConfig, benchmark: BenchmarkName) -> tuple[str, ...]:
    if benchmark == "stage2_physics":
        return config.stage2_physics.tasks if config.stage2_physics.enabled else ()
    if not config.stage3.enabled:
        return ()
    authority = load_stage3_config(config.data.stage3_authority_config)
    registry = resolve_task_registry(authority)
    if config.stage3.tasks == "all":
        return tuple(task_id for task_id, spec in registry.items() if spec.enabled)
    unknown = set(config.stage3.tasks) - set(registry)
    if unknown:
        raise ValueError("Configured Stage 3 benchmark tasks are unknown: " + ", ".join(sorted(unknown)))
    return config.stage3.tasks


def _empty_dataset(task: BenchmarkTask) -> RawDataset:
    return RawDataset(
        components=(),
        component_count=len(task.slots),
        conditions=np.empty((0, len(task.condition_columns)), dtype=np.float64),
        targets=np.empty((0, len(task.target_columns)), dtype=np.float64),
        source_rows=(),
    )


def _read_paths(task: BenchmarkTask, paths: Iterable[Path], *, allow_empty: bool = False) -> RawDataset:
    components: list[tuple[str, ...]] = []
    conditions: list[list[float]] = []
    targets: list[list[float]] = []
    source_rows: list[str] = []
    required = set(task.slots) | set(task.condition_columns) | set(task.target_columns)
    for path in paths:
        if not path.is_file():
            if allow_empty and task.benchmark == "stage3":
                continue
            raise FileNotFoundError(f"Missing benchmark source: {path}")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Benchmark source missing columns in {path}: {sorted(missing)}")
            for row_number, row in enumerate(reader, start=2):
                context = f"{task.task_id}/{path.name}:{row_number}"
                components.append(tuple(canonicalize_smiles(row.get(slot, ""), f"{context}/{slot}") for slot in task.slots))
                conditions.append([finite_float(row.get(name), f"{context}/{name}") for name in task.condition_columns])
                targets.append([finite_float(row.get(name), f"{context}/{name}") for name in task.target_columns])
                source_rows.append(f"{path.as_posix()}:{row_number}")
    if not components:
        if allow_empty:
            return _empty_dataset(task)
        raise ValueError(f"Benchmark split has no rows: {task.task_id}")
    return RawDataset(
        components=tuple(components),
        component_count=len(task.slots),
        conditions=np.asarray(conditions, dtype=np.float64).reshape(len(components), len(task.condition_columns)),
        targets=np.asarray(targets, dtype=np.float64).reshape(len(components), len(task.target_columns)),
        source_rows=tuple(source_rows),
    )


def load_split(task: BenchmarkTask, split: Literal["train", "valid", "test"]) -> RawDataset:
    paths: Sequence[Path]
    if split == "train":
        paths = task.train_paths
    elif split == "valid":
        paths = task.valid_paths
    elif split == "test":
        paths = (task.test_path,)
    else:
        raise ValueError(f"Unknown benchmark split: {split}")
    return _read_paths(task, paths, allow_empty=split == "test" and task.benchmark == "stage3")


def has_test_rows(task: BenchmarkTask) -> bool:
    if not task.test_path.is_file():
        return False
    required = set(task.slots) | set(task.condition_columns) | set(task.target_columns)
    with task.test_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Benchmark test source missing columns in {task.test_path}: "
                f"{sorted(missing)}"
            )
        return next(reader, None) is not None


__all__ = [
    "BenchmarkTask",
    "RawDataset",
    "configured_tasks",
    "has_test_rows",
    "load_split",
    "resolve_task",
]
