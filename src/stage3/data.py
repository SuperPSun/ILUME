from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from rdkit import Chem

from common.io import sha256_file
from common.training import canonical_json_sha256
from .config import Stage3Config


STAGE3_ARTIFACT_VERSION = 1
STAGE3_ARTIFACT_KIND = "ilume_stage3_sparse_data"
OBJECT_ENCODING_CONTRACT_VERSION = 1
MISSING_MARKERS = frozenset({"", "nan", "na", "n/a", "null", "none", "missing"})


def _parts(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(";") if part.strip())


@dataclass(frozen=True)
class CatalogTaskFact:
    task_id: str
    target_column: str
    identity_columns: tuple[str, ...]
    condition_columns: tuple[str, ...]
    system_type: str
    materialized_path: str
    split_strategies: tuple[str, ...]
    catalog_schema_version: int
    provenance: dict[str, str]


@dataclass(frozen=True)
class ResolvedTaskSpec:
    task_id: str
    target_column: str
    identity_columns: tuple[str, ...]
    condition_columns: tuple[str, ...]
    system_type: str
    materialized_path: str
    split_strategy: str
    cv_repeat: int
    meta_group: str
    partner_mode: str
    primary_slots: tuple[str, ...]
    partner_slots: tuple[str, ...]
    enabled: bool
    task_weight: float
    catalog_schema_version: int
    provenance: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedGroupSpec:
    group_id: str
    enabled: bool
    group_weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, order=True)
class ObjectKey:
    topology: str
    slots: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"topology": self.topology, "slots": [list(slot) for slot in self.slots]}

    @property
    def identity(self) -> str:
        return canonical_json_sha256(self.to_dict())


def sanitize_task(task_id: str) -> str:
    return task_id.replace("/", "__")


def canonicalize_smiles(raw: str, context: str) -> str:
    molecule = Chem.MolFromSmiles((raw or "").strip())
    if molecule is None:
        raise ValueError(f"Invalid SMILES in {context}: {raw}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def finite_float(raw: str | None, context: str) -> float:
    if (raw or "").strip().lower() in MISSING_MARKERS:
        raise ValueError(f"Missing Stage 3 value in {context}")
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Non-numeric Stage 3 value in {context}: {raw}") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite Stage 3 value in {context}: {raw}")
    return value


def load_task_catalog(path: str | Path) -> dict[str, CatalogTaskFact]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stage 3 task catalog: {path}")
    result: dict[str, CatalogTaskFact] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "catalog_schema_version", "stage", "task_id", "target_columns",
            "identity_columns", "condition_columns", "system_type",
            "materialized_path", "strategies",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Stage 3 catalog missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            if int(row["stage"]) != 3:
                continue
            task_id = row["task_id"].strip()
            if task_id in result:
                raise ValueError(f"Duplicate Stage 3 catalog task: {task_id}")
            targets = _parts(row["target_columns"])
            if len(targets) != 1:
                raise ValueError(f"Stage 3 task must have one scalar target: {task_id}")
            identities = _parts(row["identity_columns"])
            if not identities:
                raise ValueError(f"Stage 3 task has no identity columns: {task_id}")
            strategies = tuple(value.replace("-", "_") for value in _parts(row["strategies"]))
            if not strategies:
                raise ValueError(f"Stage 3 task has no split strategies: {task_id}")
            provenance = {
                key: row.get(key, "")
                for key in (
                    "source_file", "task_kind", "target_level", "split_unit",
                    "sample_unit", "experiment_reference", "label_source",
                    "resource_manifest",
                )
                if row.get(key, "")
            }
            try:
                schema_version = int(row["catalog_schema_version"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid catalog schema version at row {row_number}"
                ) from error
            result[task_id] = CatalogTaskFact(
                task_id=task_id,
                target_column=targets[0],
                identity_columns=identities,
                condition_columns=_parts(row["condition_columns"]),
                system_type=row["system_type"].strip(),
                materialized_path=row["materialized_path"].strip(),
                split_strategies=strategies,
                catalog_schema_version=schema_version,
                provenance=provenance,
            )
    return result


def _default_strategy(fact: CatalogTaskFact) -> str:
    if "il" in fact.split_strategies:
        return "il"
    topology = fact.system_type.replace("-", "_")
    if topology in fact.split_strategies:
        return topology
    raise ValueError(
        f"Stage 3 task has no IL or topology split strategy: {fact.task_id}"
    )


def resolve_task_registry(config: Stage3Config) -> dict[str, ResolvedTaskSpec]:
    config.validate()
    catalog = load_task_catalog(config.data.task_catalog)
    configured = set(config.tasks)
    missing = configured - set(catalog)
    if missing:
        raise ValueError("Stage 3 tasks missing from catalog: " + ", ".join(sorted(missing)))
    unknown_overrides = set(config.data.split_strategies) - configured
    unknown_repeats = set(config.data.cv_repeats) - configured
    if unknown_overrides or unknown_repeats:
        raise ValueError("Stage 3 split configuration references unknown tasks")
    resolved: dict[str, ResolvedTaskSpec] = {}
    for task_id, task in config.tasks.items():
        fact = catalog[task_id]
        strategy = config.data.split_strategies.get(task_id, _default_strategy(fact))
        strategy = strategy.replace("-", "_")
        if strategy not in fact.split_strategies:
            raise ValueError(f"Illegal split strategy for {task_id}: {strategy}")
        configured_slots = task.primary_slots + task.partner_slots
        if configured_slots != fact.identity_columns:
            raise ValueError(
                f"Stage 3 slot/catalog mismatch for {task_id}: "
                f"{configured_slots} != {fact.identity_columns}"
            )
        if fact.system_type not in {"il", "il_solute", "solute_solvent"}:
            raise ValueError(
                f"Unsupported Stage 3 topology for {task_id}: {fact.system_type}"
            )
        resolved[task_id] = ResolvedTaskSpec(
            task_id=task_id,
            target_column=fact.target_column,
            identity_columns=fact.identity_columns,
            condition_columns=fact.condition_columns,
            system_type=fact.system_type,
            materialized_path=fact.materialized_path,
            split_strategy=strategy,
            cv_repeat=config.data.cv_repeats.get(task_id, config.data.cv_repeat),
            meta_group=task.meta_group,
            partner_mode=task.partner_mode,
            primary_slots=task.primary_slots,
            partner_slots=task.partner_slots,
            enabled=task.enabled,
            task_weight=task.task_weight,
            catalog_schema_version=fact.catalog_schema_version,
            provenance=fact.provenance,
        )
    return resolved


def resolve_group_registry(config: Stage3Config) -> dict[str, ResolvedGroupSpec]:
    config.validate()
    return {
        group_id: ResolvedGroupSpec(
            group_id=group_id,
            enabled=spec.enabled,
            group_weight=spec.group_weight,
        )
        for group_id, spec in config.groups.items()
    }


_STRATEGY_DIRECTORIES = {
    "il": "IL",
    "il_solute": "IL-solute",
    "solute_solvent": "solute-solvent",
}


def task_root(config: Stage3Config, spec: ResolvedTaskSpec) -> Path:
    relative = Path(spec.materialized_path)
    if relative.parts and relative.parts[0] == "stage3":
        relative = Path(*relative.parts[1:])
    return config.data.stage3_dir / relative


def source_path(config: Stage3Config, spec: ResolvedTaskSpec, fold: int) -> Path:
    if fold not in range(1, 6):
        raise ValueError("Stage 3 fold must be in 1..5")
    directory = _STRATEGY_DIRECTORIES.get(
        spec.split_strategy, spec.split_strategy.replace("_", "-")
    )
    root = task_root(config, spec) / directory
    direct = root / f"fold{fold}.csv"
    repeated = root / f"cv{spec.cv_repeat}" / f"fold{fold}.csv"
    if spec.cv_repeat == 1 and direct.is_file():
        return direct
    if repeated.is_file():
        return repeated
    raise FileNotFoundError(
        f"Missing Stage 3 split for {spec.task_id}: {direct} or {repeated}"
    )


def test_path(config: Stage3Config, spec: ResolvedTaskSpec) -> Path:
    return task_root(config, spec) / "test.csv"


def iter_rows(
    config: Stage3Config,
    spec: ResolvedTaskSpec,
    folds: Sequence[int] | None,
) -> Iterator[tuple[int, int, dict[str, str]]]:
    paths = (
        [(0, test_path(config, spec))]
        if folds is None
        else [(fold, source_path(config, spec, fold)) for fold in folds]
    )
    required = set(spec.identity_columns) | set(spec.condition_columns) | {
        spec.target_column
    }
    for fold, path in paths:
        if not path.is_file():
            if folds is None:
                return
            raise FileNotFoundError(f"Missing Stage 3 source: {path}")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Missing Stage 3 columns in {path}: {sorted(missing)}")
            for row_number, row in enumerate(reader, start=2):
                yield fold, row_number, row


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

    def finish(self, *, target: bool, context: str) -> dict[str, float | int | bool]:
        if self.count == 0:
            raise ValueError(f"No Stage 3 training values for {context}")
        scale = math.sqrt(max(self.m2 / self.count, 0.0))
        min_scale = 1e-8 * max(1.0, abs(self.mean))
        constant = not math.isfinite(scale) or scale <= min_scale
        if target and constant:
            raise ValueError(f"Stage 3 target has zero variance: {context}")
        return {
            "count": self.count,
            "mean": self.mean,
            "scale": 1.0 if constant else scale,
            "constant": constant,
        }


def fit_normalization(
    config: Stage3Config,
    registry: Mapping[str, ResolvedTaskSpec],
    held_out_fold: int,
) -> dict[str, Any]:
    train_folds = tuple(fold for fold in range(1, 6) if fold != held_out_fold)
    result: dict[str, Any] = {}
    for task_id, spec in registry.items():
        conditions = {name: RunningStats() for name in spec.condition_columns}
        target = RunningStats()
        for fold, row_number, row in iter_rows(config, spec, train_folds):
            for name, stats in conditions.items():
                stats.update(finite_float(row.get(name), f"{task_id}/fold{fold}:{row_number}/{name}"))
            target.update(
                finite_float(
                    row.get(spec.target_column),
                    f"{task_id}/fold{fold}:{row_number}/{spec.target_column}",
                )
            )
        result[task_id] = {
            "conditions": {
                name: stats.finish(target=False, context=f"{task_id}/{name}")
                for name, stats in conditions.items()
            },
            "target": target.finish(target=True, context=f"{task_id}/target"),
        }
    return result


def _role(slot: str) -> str:
    return slot if slot in {"cation", "anion"} else "neutral"


def object_key_from_row(
    task_id: str,
    row_number: int,
    row: Mapping[str, str],
    slots: Sequence[str],
) -> ObjectKey:
    ordered = tuple(
        (
            _role(slot),
            canonicalize_smiles(row.get(slot, ""), f"{task_id}:{row_number}/{slot}"),
        )
        for slot in slots
    )
    topology = "il" if tuple(slots) == ("cation", "anion") else "molecule"
    if topology == "molecule" and len(slots) != 1:
        raise ValueError(f"Unsupported Stage 3 object slots for {task_id}: {slots}")
    return ObjectKey(topology, ordered)


def collect_object_keys(
    config: Stage3Config, registry: Mapping[str, ResolvedTaskSpec]
) -> tuple[ObjectKey, ...]:
    keys: set[ObjectKey] = set()
    for task_id, spec in registry.items():
        for _, row_number, row in iter_rows(config, spec, range(1, 6)):
            keys.add(object_key_from_row(task_id, row_number, row, spec.primary_slots))
            if spec.partner_slots:
                keys.add(object_key_from_row(task_id, row_number, row, spec.partner_slots))
        for _, row_number, row in iter_rows(config, spec, None):
            keys.add(object_key_from_row(task_id, row_number, row, spec.primary_slots))
            if spec.partner_slots:
                keys.add(object_key_from_row(task_id, row_number, row, spec.partner_slots))
    return tuple(sorted(keys))


def build_task_payload(
    config: Stage3Config,
    spec: ResolvedTaskSpec,
    held_out_fold: int,
    split: str,
    object_ids: Mapping[ObjectKey, int],
    normalization: Mapping[str, Any],
) -> dict[str, Any]:
    if split not in {"train", "valid", "test"}:
        raise ValueError("Stage 3 split must be train, valid, or test")
    folds: Sequence[int] | None
    if split == "train":
        folds = tuple(fold for fold in range(1, 6) if fold != held_out_fold)
    elif split == "valid":
        folds = (held_out_fold,)
    else:
        folds = None
    rows: list[dict[str, Any]] = []
    stats = normalization[spec.task_id]
    for source_fold, row_number, row in iter_rows(config, spec, folds):
        primary = object_key_from_row(
            spec.task_id, row_number, row, spec.primary_slots
        )
        partner = (
            object_key_from_row(spec.task_id, row_number, row, spec.partner_slots)
            if spec.partner_slots
            else None
        )
        conditions = [
            (
                finite_float(
                    row.get(name),
                    f"{spec.task_id}/fold{source_fold}:{row_number}/{name}",
                )
                - float(stats["conditions"][name]["mean"])
            )
            / float(stats["conditions"][name]["scale"])
            for name in spec.condition_columns
        ]
        raw_target = finite_float(
            row.get(spec.target_column),
            f"{spec.task_id}/fold{source_fold}:{row_number}/{spec.target_column}",
        )
        target_stats = stats["target"]
        rows.append(
            {
                "primary": object_ids[primary],
                "partner": -1 if partner is None else object_ids[partner],
                "conditions": conditions,
                "target": (raw_target - target_stats["mean"]) / target_stats["scale"],
                "raw_target": raw_target,
                "source_fold": source_fold,
                "source_row": row_number,
            }
        )
    if not rows and split != "test":
        raise ValueError(f"No Stage 3 observations for {spec.task_id}/{split}")
    return {
        "format_version": STAGE3_ARTIFACT_VERSION,
        "kind": STAGE3_ARTIFACT_KIND,
        "task_id": spec.task_id,
        "fold": held_out_fold,
        "split": split,
        "primary_object_ids": torch.tensor(
            [row["primary"] for row in rows], dtype=torch.long
        ),
        "partner_object_ids": torch.tensor(
            [row["partner"] for row in rows], dtype=torch.long
        ),
        "conditions": torch.tensor(
            [row["conditions"] for row in rows], dtype=torch.float32
        ).reshape(len(rows), len(spec.condition_columns)),
        "targets": torch.tensor([row["target"] for row in rows], dtype=torch.float32),
        "raw_targets": torch.tensor(
            [row["raw_target"] for row in rows], dtype=torch.float32
        ),
        "source_folds": torch.tensor(
            [row["source_fold"] for row in rows], dtype=torch.int8
        ),
        "source_rows": torch.tensor(
            [row["source_row"] for row in rows], dtype=torch.long
        ),
    }


class Stage3TaskDataset:
    def __init__(self, artifact_dir: str | Path, fold: int, task_id: str, split: str):
        self.artifact_dir = Path(artifact_dir)
        metadata_path = self.artifact_dir / "metadata.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        artifact_kind = self.metadata.get("kind")
        if (
            self.metadata.get("format_version") != STAGE3_ARTIFACT_VERSION
            or artifact_kind not in {
                STAGE3_ARTIFACT_KIND,
                "ilume_stage3_rdkit_sparse_data",
            }
        ):
            raise ValueError("Unsupported Stage 3 sparse artifact")
        relative = f"folds/fold{fold}/{sanitize_task(task_id)}_{split}.pt"
        path = self.artifact_dir / relative
        if self.metadata.get("artifact_hashes", {}).get(relative) != sha256_file(path):
            raise ValueError(f"Stage 3 artifact hash mismatch: {relative}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("format_version") != STAGE3_ARTIFACT_VERSION
            or payload.get("kind") != artifact_kind
            or payload.get("task_id") != task_id
            or payload.get("fold") != fold
            or payload.get("split") != split
        ):
            raise ValueError("Stage 3 task artifact identity mismatch")
        self.fold = fold
        self.task_id = task_id
        self.split = split
        for name, value in payload.items():
            setattr(self, name, value)

    def __len__(self) -> int:
        return int(self.targets.shape[0])


class Stage3RepresentationStore:
    def __init__(
        self,
        artifact_dir: str | Path,
        fold: int,
        prepared_objects: Mapping[str, Any],
        artifact_kind: str,
    ) -> None:
        self.fold = fold
        self.artifact_kind = artifact_kind
        if artifact_kind == STAGE3_ARTIFACT_KIND:
            embeddings = prepared_objects.get("embeddings")
            if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
                raise ValueError("Stage 3 Object representation matrix is malformed")
            self.output_dim = int(embeddings.shape[1])
            self.input_dims: dict[str, int] | None = None
            self._embeddings = embeddings.float()
            self._features: dict[str, torch.Tensor] = {}
            self._lookups: dict[str, torch.Tensor] = {}
            return
        if artifact_kind != "ilume_stage3_rdkit_sparse_data":
            raise ValueError(f"Unsupported Stage 3 representation kind: {artifact_kind}")
        path = Path(artifact_dir) / "folds" / f"fold{fold}" / "representation.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("kind") != artifact_kind
            or payload.get("format_version") != STAGE3_ARTIFACT_VERSION
            or payload.get("fold") != fold
        ):
            raise ValueError("RDKit Stage 3 representation identity mismatch")
        object_count = len(prepared_objects.get("objects", ()))
        self.output_dim = 512
        self.input_dims = {}
        self._embeddings = torch.empty(0)
        self._features = {}
        self._lookups = {}
        for topology, prefix in (("il", "il"), ("molecule", "single")):
            object_ids = payload[f"{prefix}_object_ids"]
            features = payload[f"{prefix}_features"]
            if (
                not isinstance(object_ids, torch.Tensor)
                or object_ids.ndim != 1
                or object_ids.dtype != torch.long
                or not isinstance(features, torch.Tensor)
                or features.ndim != 2
                or features.dtype != torch.float32
                or len(object_ids) != len(features)
                or not torch.isfinite(features).all()
            ):
                raise ValueError(f"Malformed RDKit Stage 3 {prefix} feature store")
            lookup = torch.full((object_count,), -1, dtype=torch.long)
            lookup[object_ids] = torch.arange(len(object_ids), dtype=torch.long)
            self._features[topology] = features
            self._lookups[topology] = lookup
            self.input_dims[topology] = int(features.shape[1])

    @property
    def is_rdkit(self) -> bool:
        return self.input_dims is not None

    def values(self, object_ids: torch.Tensor, topology: str) -> torch.Tensor:
        ids = object_ids.cpu().long()
        if not self.is_rdkit:
            return self._embeddings[ids]
        if topology not in self._features:
            raise ValueError(f"Unknown RDKit Stage 3 topology: {topology}")
        positions = self._lookups[topology][ids]
        if bool((positions < 0).any()):
            raise ValueError(f"RDKit Stage 3 object/topology mismatch: {topology}")
        return self._features[topology][positions]


def source_hashes(
    config: Stage3Config, registry: Mapping[str, ResolvedTaskSpec]
) -> dict[str, str]:
    result = {str(config.data.task_catalog): sha256_file(config.data.task_catalog)}
    for spec in registry.values():
        for fold in range(1, 6):
            path = source_path(config, spec, fold)
            result[str(path)] = sha256_file(path)
        path = test_path(config, spec)
        if path.is_file():
            result[str(path)] = sha256_file(path)
    return result


def stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256(str(seed).encode())
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode())
    return int.from_bytes(digest.digest()[:8], "big") % (2**63 - 1)


def resolve_batch_allocation(
    counts: Mapping[str, int], composite_batch_size: int, virtual_min_size: int
) -> dict[str, int]:
    if not counts or any(value <= 0 for value in counts.values()):
        raise ValueError("Stage 3 task counts must be positive")
    if len(counts) > composite_batch_size:
        raise ValueError("Stage 3 active tasks exceed composite batch size")
    virtual = {task: max(count, virtual_min_size) for task, count in counts.items()}
    total = sum(virtual.values())
    allocation = {
        task: max(1, math.floor(composite_batch_size * size / total))
        for task, size in virtual.items()
    }
    while sum(allocation.values()) < composite_batch_size:
        for task in sorted(counts, key=lambda name: (-counts[name], name)):
            allocation[task] += 1
            if sum(allocation.values()) == composite_batch_size:
                break
    while sum(allocation.values()) > composite_batch_size:
        changed = False
        for task in sorted(counts, key=lambda name: (counts[name], name)):
            if allocation[task] > 1:
                allocation[task] -= 1
                changed = True
                if sum(allocation.values()) == composite_batch_size:
                    break
        if not changed:
            raise ValueError("Unable to resolve Stage 3 batch allocation")
    return allocation


def composite_steps_per_epoch(
    counts: Mapping[str, int], allocation: Mapping[str, int], virtual_min_size: int
) -> int:
    if set(counts) != set(allocation):
        raise ValueError("Stage 3 count/allocation tasks differ")
    return max(
        math.ceil(max(counts[task], virtual_min_size) / allocation[task])
        for task in counts
    )


def balanced_virtual_indices(
    real_size: int,
    required_size: int,
    *,
    seed: int,
    epoch: int,
    task_id: str,
) -> torch.Tensor:
    if real_size <= 0 or required_size <= 0:
        raise ValueError("Stage 3 virtual sequence sizes must be positive")
    generator = torch.Generator().manual_seed(
        stable_seed(seed, "virtual", epoch, task_id)
    )
    chunks: list[torch.Tensor] = []
    remaining = required_size
    while remaining > 0:
        order = torch.randperm(real_size, generator=generator)
        chunks.append(order[:remaining])
        remaining -= min(real_size, remaining)
    return torch.cat(chunks)
