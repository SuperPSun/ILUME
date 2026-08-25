from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

from common.io import sha256_file
from common.training import canonical_json_sha256


TaskKind = Literal["object_property", "atom_property"]
TargetLevel = Literal["object", "atom"]
Topology = Literal["single_entity", "ionic_liquid", "interaction"]

ORBITAL_TASK_TARGETS = {
    "simulation/homo": "HOMO_eV",
    "simulation/lumo": "LUMO_eV",
}
ORBITAL_AUDIT_COLUMNS = (
    "ion_role",
    "provenance_source_file",
    "provenance_source_row",
)
ORBITAL_SOURCE_FILE_BY_ROLE = {
    "cation": (
        "simulation/simulated_HOMO+LUMO_PBE_TZVP_cations_structured.csv"
    ),
    "anion": (
        "simulation/simulated_HOMO+LUMO_PBE_TZVP_anions_structured.csv"
    ),
}


def orbital_audit_columns(task_id: str) -> tuple[str, ...]:
    return ORBITAL_AUDIT_COLUMNS if task_id in ORBITAL_TASK_TARGETS else ()


def validate_orbital_audit_row(
    task_id: str,
    row: Mapping[str, str],
    *,
    inferred_role: str,
    context: str,
) -> None:
    if task_id not in ORBITAL_TASK_TARGETS:
        return
    role = row.get("ion_role", "").strip()
    if role != inferred_role or role not in ORBITAL_SOURCE_FILE_BY_ROLE:
        raise ValueError(f"Orbital ion_role/formal-charge mismatch in {context}")
    if row.get("provenance_source_file", "").strip() != ORBITAL_SOURCE_FILE_BY_ROLE[role]:
        raise ValueError(f"Orbital provenance source/role mismatch in {context}")
    try:
        source_row = int(row.get("provenance_source_row", ""))
    except ValueError as error:
        raise ValueError(f"Invalid orbital provenance_source_row in {context}") from error
    if source_row < 2:
        raise ValueError(f"Invalid orbital provenance_source_row in {context}")


def _columns(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.split(";") if part)


def _safe_relative(value: str, *, field: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Catalog {field} must be a safe relative path: {value!r}")
    return path


def _under_root(data_root: Path, relative: str, *, field: str) -> Path:
    root = data_root.resolve()
    path = data_root.joinpath(*PurePosixPath(relative).parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Catalog {field} escapes data_root: {relative!r}")
    return path


@dataclass(frozen=True)
class DatasetSpec:
    catalog_schema_version: int
    task_id: str
    task_kind: TaskKind
    target_level: TargetLevel
    source_file: str
    materialized_path: str
    identity_columns: tuple[str, ...]
    condition_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    system_type: str
    simulation_method: str | None
    label_source: str
    resource_manifest: str | None

    def split_path(self, data_root: Path, split: str) -> Path:
        if split not in {"train", "valid", "test"}:
            raise ValueError("Stage 2 split must be train, valid, or test")
        relative = str(PurePosixPath(self.materialized_path) / f"{split}.csv")
        return _under_root(data_root, relative, field="materialized_path")

    def resource_manifest_path(self, data_root: Path) -> Path | None:
        if self.resource_manifest is None:
            return None
        return _under_root(data_root, self.resource_manifest, field="resource_manifest")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_kind: TaskKind
    target_level: TargetLevel
    topology: Topology
    entity_columns: tuple[str, ...]
    role_policy: tuple[str, ...]
    condition_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    dataset: DatasetSpec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage2Registry:
    tasks: tuple[TaskSpec, ...]
    registry_hash: str
    catalog_sha256: str

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks)

    def by_id(self, task_id: str) -> TaskSpec:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(task_id)

    def snapshot(self) -> list[dict[str, Any]]:
        return [task.to_dict() for task in self.tasks]

    @classmethod
    def from_snapshot(cls, snapshot: list[dict[str, Any]], *, registry_hash: str, catalog_sha256: str) -> "Stage2Registry":
        tasks: list[TaskSpec] = []
        for raw in snapshot:
            values = dict(raw)
            dataset_values = dict(values.pop("dataset"))
            for name in ("identity_columns", "condition_columns", "target_columns"):
                dataset_values[name] = tuple(dataset_values[name])
            dataset = DatasetSpec(**dataset_values)
            for name in ("entity_columns", "role_policy", "condition_columns", "target_columns"):
                values[name] = tuple(values[name])
            tasks.append(TaskSpec(dataset=dataset, **values))
        result = cls(tuple(tasks), registry_hash, catalog_sha256)
        if canonical_json_sha256(result.snapshot()) != registry_hash:
            raise ValueError("Stage 2 registry snapshot hash mismatch")
        return result


_SYSTEM_SEMANTICS: dict[str, tuple[Topology, tuple[str, ...]]] = {
    "cation": ("single_entity", ("cation",)),
    "anion": ("single_entity", ("anion",)),
    "il": ("ionic_liquid", ("cation", "anion")),
    "solute_solvent": ("interaction", ("formal_charge", "formal_charge")),
}


def _task_from_row(row: dict[str, str]) -> TaskSpec:
    try:
        schema_version = int(row["catalog_schema_version"])
        stage = int(row["stage"])
    except (KeyError, ValueError) as error:
        raise ValueError("Invalid task catalog schema/stage") from error
    if schema_version not in {1, 2} or stage != 2:
        raise ValueError("Unsupported Stage 2 task catalog row")
    task_id = row["task_id"].strip()
    _safe_relative(task_id, field="task_id")
    if "." in task_id:
        raise ValueError(f"Stage 2 task_id cannot contain '.': {task_id}")
    task_kind = row["task_kind"].strip()
    target_level = row["target_level"].strip()
    label_source = row["label_source"].strip()
    if (task_kind, target_level, label_source) not in {
        ("object_property", "object", "materialized_csv"),
        ("atom_property", "atom", "structure_resource"),
    }:
        raise ValueError(f"Unsupported Stage 2 label contract for {task_id}")
    source_file = str(_safe_relative(row["source_file"].strip(), field="source_file"))
    materialized_path = str(_safe_relative(row["materialized_path"].strip(), field="materialized_path"))
    manifest_text = row.get("resource_manifest", "").strip()
    resource_manifest = str(_safe_relative(manifest_text, field="resource_manifest")) if manifest_text else None
    if label_source == "structure_resource" and resource_manifest is None:
        raise ValueError(f"Atom task requires resource_manifest: {task_id}")
    identity_columns = _columns(row["identity_columns"].strip())
    condition_columns = _columns(row["condition_columns"].strip())
    target_columns = _columns(row["target_columns"].strip())
    system_type = row["system_type"].strip()
    if system_type == "molecule":
        topology: Topology = "single_entity"
        role_policy = ("manifest",) if label_source == "structure_resource" else ("formal_charge",)
    else:
        try:
            topology, role_policy = _SYSTEM_SEMANTICS[system_type]
        except KeyError as error:
            raise ValueError(f"Unsupported Stage 2 system_type: {system_type}") from error
    expected_entities = 2 if topology in {"ionic_liquid", "interaction"} else 1
    if len(identity_columns) != expected_entities or len(role_policy) != expected_entities:
        raise ValueError(f"Stage 2 identity/topology mismatch for {task_id}")
    if not target_columns or (target_level == "atom" and len(target_columns) != 1):
        raise ValueError(f"Unsupported Stage 2 targets for {task_id}")
    dataset = DatasetSpec(
        catalog_schema_version=schema_version, task_id=task_id,
        task_kind=task_kind, target_level=target_level,  # type: ignore[arg-type]
        source_file=source_file, materialized_path=materialized_path,
        identity_columns=identity_columns, condition_columns=condition_columns,
        target_columns=target_columns, system_type=system_type,
        simulation_method=row.get("simulation_method", "").strip() or None,
        label_source=label_source, resource_manifest=resource_manifest,
    )
    return TaskSpec(
        task_id=task_id, task_kind=task_kind, target_level=target_level,  # type: ignore[arg-type]
        topology=topology, entity_columns=identity_columns,
        role_policy=role_policy, condition_columns=condition_columns,
        target_columns=target_columns, dataset=dataset,
    )


def load_stage2_registry(path: str | Path) -> Stage2Registry:
    catalog_path = Path(path)
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Missing task catalog: {catalog_path}")
    with catalog_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "catalog_schema_version", "stage", "task_id", "task_kind", "target_level",
            "source_file", "target_columns", "identity_columns", "condition_columns",
            "system_type", "simulation_method", "materialized_path", "label_source",
            "resource_manifest",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Task catalog is missing required Stage 2 columns")
        tasks = [_task_from_row(row) for row in reader if row.get("stage", "").strip() == "2"]
    tasks.sort(key=lambda task: task.task_id)
    task_ids = [task.task_id for task in tasks]
    if not tasks or len(task_ids) != len(set(task_ids)):
        raise ValueError("Task catalog must contain unique Stage 2 task IDs")
    snapshot = [task.to_dict() for task in tasks]
    return Stage2Registry(tuple(tasks), canonical_json_sha256(snapshot), sha256_file(catalog_path))


__all__ = [
    "DatasetSpec",
    "ORBITAL_AUDIT_COLUMNS",
    "ORBITAL_TASK_TARGETS",
    "Stage2Registry",
    "TaskSpec",
    "load_stage2_registry",
    "orbital_audit_columns",
    "validate_orbital_audit_row",
]
