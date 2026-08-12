from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from rdkit import Chem

from common.io import sha256_file
from common.training import canonical_json_sha256
from .config import STAGE3_TASKS, Stage3Config


STAGE3_ARTIFACT_VERSION = 2
LATE_SOLUTE_TASKS = ("experiment/solvation", "experiment/transfer")
CONDITION_COLUMNS = (
    "temperature_K",
    "pressure_kPa",
    "frequency_MHz",
    "wavelength_nm",
)
PHASE_TOKENS = {"<missing>": 0, "<unk>": 1, "solid": 2, "liquid": 3, "gas": 4}
MISSING_MARKERS = frozenset({"", "nan", "na", "n/a", "null", "none", "missing"})


@dataclass(frozen=True)
class Stage3TaskSpec:
    task: str
    entity_columns: tuple[str, ...]
    entity_roles: tuple[str, ...]
    target: str
    condition_columns: tuple[str, ...]
    topology: str
    meta_group: str
    fold_strategy: str

    @property
    def uses_solute(self) -> bool:
        return self.task in LATE_SOLUTE_TASKS


def _spec(
    task: str,
    target: str,
    group: str,
    *,
    conditions: tuple[str, ...] = (),
    entities: tuple[str, ...] = ("cation", "anion"),
    roles: tuple[str, ...] = ("cation", "anion"),
    topology: str = "il",
    fold: str = "IL",
) -> Stage3TaskSpec:
    return Stage3TaskSpec(task, entities, roles, target, conditions, topology, group, fold)


TASK_REGISTRY: dict[str, Stage3TaskSpec] = {
    spec.task: spec
    for spec in (
        _spec("experiment/density", "density_g/cm^3", "thermodynamic", conditions=("temperature_K", "pressure_kPa")),
        _spec("experiment/dynamic_relative_permittivity", "dynamic_relative_permittivity_unitless", "electrical", conditions=("temperature_K", "pressure_kPa", "frequency_MHz")),
        _spec("experiment/electrical_conductivity", "electrical_conductivity_S/m_log10", "transport", conditions=("temperature_K", "pressure_kPa")),
        _spec("experiment/equilibrium_pressure", "pressure_kPa_log10", "phase", conditions=("temperature_K",)),
        _spec("experiment/glass_transition_temperature", "glass_transition_temperature_K", "phase"),
        _spec("experiment/heat_capacity", "heat_capacity_J/mol/K", "thermodynamic", conditions=("temperature_K", "pressure_kPa")),
        _spec("experiment/isobaric_coefficient_of_volume_expansion", "isobaric_coefficient_of_volume_expansion_K^-1", "thermodynamic", conditions=("temperature_K", "pressure_kPa")),
        _spec("experiment/melting_point", "melting_point_K", "phase"),
        _spec("experiment/pec50", "pEC50", "biological"),
        _spec("experiment/refractive_index", "refractive_index_unitless", "optical", conditions=("temperature_K", "pressure_kPa", "wavelength_nm")),
        _spec("experiment/self_diffusion_coefficient", "self_diffusion_coefficient_10^-9*m^2/s_log10", "transport", conditions=("temperature_K", "pressure_kPa")),
        _spec("experiment/solvation", "solvation_kcal/mol", "solvation", conditions=("temperature_K",), entities=("cation", "anion", "solute"), roles=("cation", "anion", "neutral"), topology="il_solute"),
        _spec("experiment/speed_of_sound", "speed_of_sound_m/s", "transport", conditions=("temperature_K", "pressure_kPa")),
        _spec("experiment/static_relative_permittivity", "static_relative_permittivity_unitless", "electrical", conditions=("temperature_K", "pressure_kPa")),
        _spec("experiment/surface_tension", "surface_tension_mN/m", "interfacial", conditions=("temperature_K",)),
        _spec("experiment/thermal_conductivity", "thermal_conductivity_W/m/K", "transport", conditions=("temperature_K", "pressure_kPa")),
        _spec("experiment/thermal_decomposition_temperature", "thermal_decomposition_temperature_K", "phase"),
        _spec("experiment/transfer", "transfer_kcal/mol", "solvation", conditions=("temperature_K",), entities=("cation", "anion", "solute"), roles=("cation", "anion", "neutral"), topology="il_solute"),
        _spec("experiment/viscosity", "viscosity_mPa*s_log10", "transport", conditions=("temperature_K", "pressure_kPa")),
        _spec("experiment/x_co2", "x_CO2_unitless", "solubility", conditions=("temperature_K", "pressure_kPa")),
        _spec("simulation/heat_of_vaporization", "heat_of_vaporization_kJ/mol", "thermodynamic", conditions=("temperature_K",)),
        _spec("experiment/transfer_organic", "transfer_organic_kcal/mol", "solvation", conditions=("temperature_K",), entities=("solute", "solvent"), roles=("neutral", "neutral"), topology="neutral_pair", fold="solute-solvent"),
        _spec("simulation/cation_homo", "cation_HOMO_eV", "quantum", entities=("cation",), roles=("cation",), topology="single", fold="random"),
        _spec("simulation/cation_lumo", "cation_LUMO_eV", "quantum", entities=("cation",), roles=("cation",), topology="single", fold="random"),
        _spec("simulation/anion_homo", "anion_HOMO_eV", "quantum", entities=("anion",), roles=("anion",), topology="single", fold="random"),
        _spec("simulation/anion_lumo", "anion_LUMO_eV", "quantum", entities=("anion",), roles=("anion",), topology="single", fold="random"),
        _spec("simulation/charge", "charge", "quantum", entities=("SMILES",), roles=("neutral",), topology="single", fold="random"),
    )
}

if set(STAGE3_TASKS) != set(TASK_REGISTRY):
    raise RuntimeError("Stage 3 registry is incomplete")


def sanitize_task(task: str) -> str:
    return task.replace("/", "__")


def canonicalize(smiles: str, context: str) -> str:
    molecule = Chem.MolFromSmiles((smiles or "").strip())
    if molecule is None:
        raise ValueError(f"Invalid SMILES in {context}: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True)


def source_path(config: Stage3Config, task: str, fold: int) -> Path:
    if fold not in range(1, 6):
        raise ValueError("Stage 3 fold must be in 1..5")
    spec = TASK_REGISTRY[task]
    root = config.data.stage3_dir / task / spec.fold_strategy
    direct = root / f"fold{fold}.csv"
    if direct.is_file():
        return direct
    repeated = root / "cv1" / f"fold{fold}.csv"
    if repeated.is_file():
        return repeated
    return direct


def iter_source_rows(
    config: Stage3Config, task: str, folds: Sequence[int]
) -> Iterator[tuple[int, int, dict[str, str]]]:
    for fold in folds:
        path = source_path(config, task, fold)
        if not path.is_file():
            raise FileNotFoundError(f"Missing Stage 3 source: {path}")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = set(TASK_REGISTRY[task].entity_columns) | set(TASK_REGISTRY[task].condition_columns) | {TASK_REGISTRY[task].target}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Missing Stage 3 columns in {path}: {sorted(missing)}")
            for row_number, row in enumerate(reader, start=2):
                yield fold, row_number, row


def iter_test_rows(
    config: Stage3Config, task: str
) -> Iterator[tuple[int, int, dict[str, str]]]:
    path = config.data.stage3_dir / task / "test.csv"
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        spec = TASK_REGISTRY[task]
        required = set(spec.entity_columns) | set(spec.condition_columns) | {spec.target}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing Stage 3 columns in {path}: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            yield 0, row_number, row


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def to_dict(self) -> dict[str, float | int]:
        scale = math.sqrt(max(self.m2 / self.count, 0.0)) if self.count else 1.0
        if not math.isfinite(scale) or scale == 0.0:
            scale = 1.0
        return {"count": self.count, "mean": self.mean, "scale": scale}


def finite_float(raw: str, context: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Non-numeric Stage 3 value in {context}: {raw}") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite Stage 3 value in {context}: {raw}")
    return value


def _is_missing(raw: str | None) -> bool:
    return (raw or "").strip().lower() in MISSING_MARKERS


def fit_fold_scalers(
    config: Stage3Config,
    held_out_fold: int,
    tasks: Sequence[str],
) -> dict[str, Any]:
    train_folds = tuple(fold for fold in range(1, 6) if fold != held_out_fold)
    task_scalers: dict[str, Any] = {}
    for task in tasks:
        spec = TASK_REGISTRY[task]
        conditions = {name: RunningStats() for name in CONDITION_COLUMNS}
        target = RunningStats()
        for fold, row_number, row in iter_source_rows(config, task, train_folds):
            try:
                parsed_conditions = {
                    name: None
                    if _is_missing(row.get(name))
                    else finite_float(row[name], f"{task}/fold{fold}:{row_number}/{name}")
                    for name in spec.condition_columns
                }
                parsed_target = finite_float(row[spec.target], f"{task}/fold{fold}:{row_number}/{spec.target}")
            except ValueError:
                continue
            for name, value in parsed_conditions.items():
                if value is not None:
                    conditions[name].update(value)
            target.update(parsed_target)
        if target.count == 0:
            raise ValueError(f"Stage 3 task has no training target: {task}")
        task_scalers[task] = {
            "conditions": {name: stats.to_dict() for name, stats in conditions.items()},
            "target": target.to_dict(),
            "target_transform": "identity",
        }
    return task_scalers


def collect_entity_keys_with_audit(
    config: Stage3Config,
    tasks: Sequence[str],
) -> tuple[tuple[tuple[str, str], ...], tuple[dict[str, Any], ...]]:
    keys: set[tuple[str, str]] = set()
    cache: dict[str, str] = {}
    excluded: list[dict[str, Any]] = []
    for task in tasks:
        spec = TASK_REGISTRY[task]
        for fold, row_number, row in iter_source_rows(config, task, range(1, 6)):
            for column, role in zip(spec.entity_columns, spec.entity_roles, strict=True):
                raw = row[column]
                canonical = cache.get(raw)
                if canonical is None:
                    try:
                        canonical = canonicalize(raw, f"{task}/fold{fold}:{row_number}/{column}")
                    except ValueError as error:
                        excluded.append({"task": task, "fold": fold, "source_row": row_number, "column": column, "smiles": raw, "reason": str(error)})
                        continue
                    cache[raw] = canonical
                keys.add((role, canonical))
        for _, row_number, row in iter_test_rows(config, task):
            for column, role in zip(spec.entity_columns, spec.entity_roles, strict=True):
                raw = row[column]
                canonical = cache.get(raw)
                if canonical is None:
                    try:
                        canonical = canonicalize(raw, f"{task}/test:{row_number}/{column}")
                    except ValueError as error:
                        excluded.append({"task": task, "fold": 0, "source_row": row_number, "column": column, "smiles": raw, "reason": str(error)})
                        continue
                    cache[raw] = canonical
                keys.add((role, canonical))
    role_order = {"cation": 0, "anion": 1, "neutral": 2}
    return (
        tuple(sorted(keys, key=lambda item: (role_order[item[0]], item[1]))),
        tuple(excluded),
    )


def collect_entity_keys(
    config: Stage3Config, tasks: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    return collect_entity_keys_with_audit(config, tasks)[0]


def _system_key(spec: Stage3TaskSpec, entity_ids: Sequence[int]) -> tuple[int, ...]:
    if spec.topology == "il":
        return tuple(entity_ids[:2])
    return tuple(entity_ids)


def build_task_payload(
    config: Stage3Config,
    task: str,
    held_out_fold: int,
    split: str,
    entity_ids: dict[tuple[str, str], int],
    scalers: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if split not in {"train", "valid", "test"}:
        raise ValueError("Stage 3 split must be train, valid, or test")
    spec = TASK_REGISTRY[task]
    folds = tuple(fold for fold in range(1, 6) if fold != held_out_fold) if split == "train" else (held_out_fold,)
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    cache: dict[str, str] = {}
    scaler = scalers[task]
    iterator = iter_test_rows(config, task) if split == "test" else iter_source_rows(config, task, folds)
    for source_fold, row_number, row in iterator:
        ids: list[int] = []
        reason = ""
        for column, role in zip(spec.entity_columns, spec.entity_roles, strict=True):
            raw = row[column]
            canonical = cache.get(raw)
            if canonical is None:
                try:
                    canonical = canonicalize(raw, f"{task}/fold{source_fold}:{row_number}/{column}")
                except ValueError:
                    reason = f"invalid_smiles:{column}:{raw}"
                    break
                cache[raw] = canonical
            entity_id = entity_ids.get((role, canonical))
            if entity_id is None:
                reason = f"excluded_entity:{role}:{canonical}"
                break
            ids.append(entity_id)
        if reason:
            excluded.append({"task": task, "fold": source_fold, "source_row": row_number, "reason": reason})
            continue
        values: list[float] = []
        presence: list[float] = []
        try:
            for name in CONDITION_COLUMNS:
                raw = (row.get(name) or "").strip() if name in spec.condition_columns else ""
                if not _is_missing(raw):
                    numeric = finite_float(raw, f"{task}/fold{source_fold}:{row_number}/{name}")
                    stats = scaler["conditions"][name]
                    values.append((numeric - stats["mean"]) / stats["scale"])
                    presence.append(1.0)
                else:
                    values.append(0.0)
                    presence.append(0.0)
            target = finite_float(row[spec.target], f"{task}/fold{source_fold}:{row_number}/{spec.target}")
        except ValueError as error:
            excluded.append({"task": task, "fold": source_fold, "source_row": row_number, "reason": str(error)})
            continue
        phase_raw = row.get("phase")
        phase = "<missing>" if _is_missing(phase_raw) else str(phase_raw).strip().lower()
        phase_id = PHASE_TOKENS.get(phase, PHASE_TOKENS["<unk>"])
        target_stats = scaler["target"]
        rows.append(
            {
                "entity_ids": ids,
                "conditions": values + presence,
                "phase_id": phase_id,
                "target": (target - target_stats["mean"]) / target_stats["scale"],
                "raw_target": target,
                "source_fold": source_fold,
                "source_row": row_number,
                "system": _system_key(spec, ids),
            }
        )
    if not rows and split != "test":
        raise ValueError(f"No retained Stage 3 rows for {task}/{split}/fold{held_out_fold}")
    systems: dict[tuple[int, ...], list[int]] = {}
    for index, row in enumerate(rows):
        systems.setdefault(row["system"], []).append(index)
    sorted_systems = sorted(systems)
    offsets = [0]
    system_rows: list[int] = []
    for key in sorted_systems:
        system_rows.extend(systems[key])
        offsets.append(len(system_rows))
    payload = {
        "format_version": STAGE3_ARTIFACT_VERSION,
        "task": task,
        "split": split,
        "fold": held_out_fold,
        "entity_ids": torch.tensor([row["entity_ids"] for row in rows], dtype=torch.long).reshape(len(rows), len(spec.entity_columns)),
        "conditions": torch.tensor([row["conditions"] for row in rows], dtype=torch.float32).reshape(len(rows), 8),
        "phase_ids": torch.tensor([row["phase_id"] for row in rows], dtype=torch.long),
        "targets": torch.tensor([row["target"] for row in rows], dtype=torch.float32),
        "raw_targets": torch.tensor([row["raw_target"] for row in rows], dtype=torch.float32),
        "source_folds": torch.tensor([row["source_fold"] for row in rows], dtype=torch.int8),
        "source_rows": torch.tensor([row["source_row"] for row in rows], dtype=torch.long),
        "system_offsets": torch.tensor(offsets, dtype=torch.long),
        "system_rows": torch.tensor(system_rows, dtype=torch.long),
        "system_keys": [list(key) for key in sorted_systems],
    }
    return payload, excluded


class Stage3TaskDataset:
    def __init__(
        self,
        artifact_dir: str | Path,
        domain: str,
        fold: int,
        task: str,
        split: str,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.domain = domain
        self.fold = fold
        self.task = task
        self.split = split
        metadata_path = self.artifact_dir / domain / "metadata.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            self.metadata.get("format_version") != STAGE3_ARTIFACT_VERSION
            or self.metadata.get("kind") != "ilume_stage3_data"
        ):
            raise ValueError("Unsupported Stage 3 artifact format")
        if self.metadata.get("domain") != domain:
            raise ValueError("Stage 3 artifact domain mismatch")
        if task not in self.metadata.get("tasks", ()):
            raise ValueError(f"Task {task} is not registered in domain {domain}")
        self.scalers = json.loads(
            (self.artifact_dir / domain / "scalers.json").read_text(encoding="utf-8")
        )
        scaler_path = self.artifact_dir / domain / "scalers.json"
        if self.metadata.get("artifact_hashes", {}).get("scalers.json") != sha256_file(scaler_path):
            raise ValueError("Stage 3 artifact hash mismatch: scalers.json")
        relative = f"folds/fold{fold}/{sanitize_task(task)}_{split}.pt"
        path = self.artifact_dir / domain / relative
        expected = self.metadata["artifact_hashes"].get(relative)
        if expected is None or sha256_file(path) != expected:
            raise ValueError(f"Stage 3 artifact hash mismatch: {relative}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != STAGE3_ARTIFACT_VERSION:
            raise ValueError("Unsupported Stage 3 task payload")
        if (
            payload.get("task") != task
            or payload.get("fold") != fold
            or payload.get("split") != split
        ):
            raise ValueError("Stage 3 task payload identity mismatch")
        for name, value in payload.items():
            setattr(self, name, value)

    def __len__(self) -> int:
        return int(self.targets.shape[0])


class SystemCursor:
    def __init__(self, offsets: torch.Tensor, rows: torch.Tensor, *, seed: int) -> None:
        if len(offsets) < 2:
            raise ValueError("Stage 3 system cursor requires at least one system")
        self.offsets = offsets.clone()
        self.rows = rows.clone()
        self.generator = torch.Generator().manual_seed(seed)
        self.system_order = torch.randperm(len(offsets) - 1, generator=self.generator)
        self.system_position = 0
        self.row_orders: dict[int, torch.Tensor] = {}
        self.row_positions: dict[int, int] = {}

    def _next_system(self) -> int:
        if self.system_position == len(self.system_order):
            self.system_order = torch.randperm(len(self.offsets) - 1, generator=self.generator)
            self.system_position = 0
        value = int(self.system_order[self.system_position])
        self.system_position += 1
        return value

    def _next_row(self, system: int) -> int:
        start, end = int(self.offsets[system]), int(self.offsets[system + 1])
        count = end - start
        position = self.row_positions.get(system, count)
        if position == count:
            self.row_orders[system] = torch.randperm(count, generator=self.generator)
            position = 0
        result = int(self.rows[start + int(self.row_orders[system][position])])
        self.row_positions[system] = position + 1
        return result

    def next_indices(self, count: int) -> torch.Tensor:
        return torch.tensor([self._next_row(self._next_system()) for _ in range(count)], dtype=torch.long)

    def state_dict(self) -> dict[str, Any]:
        return {
            "generator": self.generator.get_state(),
            "system_order": self.system_order.clone(),
            "system_position": self.system_position,
            "row_orders": {key: value.clone() for key, value in self.row_orders.items()},
            "row_positions": dict(self.row_positions),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.generator.set_state(state["generator"])
        self.system_order = state["system_order"].clone()
        self.system_position = int(state["system_position"])
        self.row_orders = {int(key): value.clone() for key, value in state["row_orders"].items()}
        self.row_positions = {int(key): int(value) for key, value in state["row_positions"].items()}


def source_hashes(
    config: Stage3Config, tasks: Sequence[str]
) -> dict[str, str]:
    hashes = {
        str(source_path(config, task, fold).relative_to(config.data.stage3_dir)): sha256_file(source_path(config, task, fold))
        for task in tasks
        for fold in range(1, 6)
    }
    for task in tasks:
        test_path = config.data.stage3_dir / task / "test.csv"
        if test_path.is_file():
            hashes[str(test_path.relative_to(config.data.stage3_dir))] = sha256_file(test_path)
    return hashes


def signature(payload: Any) -> str:
    return canonical_json_sha256(payload)
