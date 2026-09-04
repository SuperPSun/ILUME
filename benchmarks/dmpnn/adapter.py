from __future__ import annotations

import csv
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from rdkit import Chem
import logging
import warnings

# Hide Lightning INFO messages while keeping warnings/errors.
logging.getLogger("lightning").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("lightning.fabric").setLevel(logging.WARNING)

warnings.filterwarnings(
    "ignore",
    message=r"Please use the new API settings to control TF32 behavior.*",
    category=UserWarning,
)


from common.identity import require_compatible_identity, semantic_identity, tensor_state_hash
from common.io import atomic_json, sha256_file
from common.outputs import repository_path
from stage2.atom_evaluation import (
    PARTIAL_CHARGE_TASK,
    PartialChargeBenchmark,
    build_partial_charge_benchmark,
    score_partial_charge_predictions,
)
from stage2.atom_targets import PARTIAL_CHARGE_MAPPING_CONTRACT
from stage2.config import load_stage2_config
from stage2.data import Stage2TaskDataset
from stage2.registry import ORBITAL_TASK_TARGETS
from common.reporting import role_mae_diagnostics

from benchmarks.common.config import BenchmarkConfig, BenchmarkName
from benchmarks.common.data import BenchmarkTask, RawDataset, load_split, resolve_task
from benchmarks.common.engine import EvaluationResult, TargetStats
from benchmarks.common.metrics import target_metrics

DMPNN_GRAPH_CONTRACT = {
    "implementation": "chemprop",
    "version": "2.3.1",
    "message_passing": "directed_bond",
    "extra_atom_features": False,
    "extra_bond_features": False,
    "extra_atom_descriptors": False,
    "extra_datapoint_descriptors": "registry_numeric_conditions_only",
    "pretrained": False,
}


@dataclass(frozen=True)
class ConditionStats:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, values: np.ndarray) -> "ConditionStats":
        if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
            raise ValueError("D-MPNN train conditions must be a non-empty finite matrix")
        if values.shape[1] == 0:
            return cls((), ())
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        return cls(tuple(map(float, mean)), tuple(map(float, scale)))

    def normalize(self, values: np.ndarray) -> np.ndarray:
        if values.shape[1] != len(self.mean) or not np.isfinite(values).all():
            raise ValueError("D-MPNN condition shape or values differ from training")
        if not self.mean:
            return np.empty((len(values), 0), dtype=np.float32)
        return (
            (values - np.asarray(self.mean)) / np.asarray(self.scale)
        ).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ConditionStats":
        return cls(
            tuple(float(value) for value in raw["mean"]),
            tuple(float(value) for value in raw["scale"]),
        )


@dataclass
class DMPNNTrainingBundle:
    task: BenchmarkTask
    train_dataset: Any
    valid_dataset: Any
    target_stats: TargetStats
    condition_stats: ConditionStats
    source_hashes: dict[str, Any]
    training_identity: dict[str, Any]
    target_level: str
    component_count: int


@dataclass
class PartialEvaluationResult:
    benchmark: PartialChargeBenchmark
    predictions: dict[str, np.ndarray]
    score: dict[str, Any]
    training_identity: dict[str, Any]


def _lock_sha(config: BenchmarkConfig) -> str:
    if config.environment is None:
        raise ValueError("D-MPNN environment contract is missing")
    return sha256_file(repository_path(config.environment.lock))


def _target_stats(values: Mapping[str, Any]) -> TargetStats:
    mean = float(values["mean"])
    scale = float(values["scale"])
    if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("D-MPNN target scaler must be finite and positive")
    return TargetStats((mean,), (scale,))


def _scalar_dataset(
    raw: RawDataset,
    target_stats: TargetStats,
    condition_stats: ConditionStats,
) -> Any:
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, MulticomponentDataset

    normalized_targets = target_stats.normalize(raw.targets)
    normalized_conditions = condition_stats.normalize(raw.conditions)
    datasets = []
    for component_index in range(raw.component_count):
        datapoints = []
        for row_index, components in enumerate(raw.components):
            datapoints.append(
                MoleculeDatapoint.from_smi(
                    components[component_index],
                    y=normalized_targets[row_index].copy(),
                    x_d=(
                        normalized_conditions[row_index].copy()
                        if component_index == 0 and normalized_conditions.shape[1]
                        else None
                    ),
                )
            )
        datasets.append(MoleculeDataset(datapoints))
    return datasets[0] if len(datasets) == 1 else MulticomponentDataset(datasets)


def _identity_payload(
    config: BenchmarkConfig,
    task: BenchmarkTask,
    target_stats: TargetStats,
    condition_stats: ConditionStats,
    source_hashes: Mapping[str, Any],
    *,
    target_level: str,
) -> dict[str, Any]:
    return {
        "benchmark_model": "dmpnn",
        "domain": task.benchmark,
        "task_id": task.task_id,
        "fold": task.fold,
        "registry": task.registry_payload,
        "source_hashes": dict(source_hashes),
        "graph_contract": DMPNN_GRAPH_CONTRACT,
        "component_order": list(task.slots),
        "target_level": target_level,
        "condition_statistics": condition_stats.to_dict(),
        "target_statistics": target_stats.to_dict(),
        "model": config.model,
        "training": config.training,
        "seed": config.seed,
        "environment_lock_sha256": _lock_sha(config),
    }


def _prepare_scalar(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
) -> DMPNNTrainingBundle:
    task = resolve_task(config, benchmark, task_id, fold)
    train = load_split(task, "train")
    valid = load_split(task, "valid")
    if len(task.target_columns) != 1:
        raise ValueError("D-MPNN v1 requires one scalar target per task")
    target_stats = TargetStats.fit(train.targets)
    condition_stats = ConditionStats.fit(train.conditions)
    source_hashes = {
        "train": [sha256_file(path) for path in task.train_paths],
        "valid": [sha256_file(path) for path in task.valid_paths],
    }
    identity = semantic_identity(
        "benchmark.training.v1",
        _identity_payload(
            config,
            task,
            target_stats,
            condition_stats,
            source_hashes,
            target_level="molecule",
        ),
    )
    return DMPNNTrainingBundle(
        task=task,
        train_dataset=_scalar_dataset(train, target_stats, condition_stats),
        valid_dataset=_scalar_dataset(valid, target_stats, condition_stats),
        target_stats=target_stats,
        condition_stats=condition_stats,
        source_hashes=source_hashes,
        training_identity=identity,
        target_level="molecule",
        component_count=train.component_count,
    )


def _source_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"mol_id", "SMILES", "role", "formal_charge", "source_list"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Partial-charge source missing columns: {sorted(missing)}")
        return {index: dict(row) for index, row in enumerate(reader, start=2)}


def _mapping_audit(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"mol_id", "canonical_smiles", "model_atom_count", "status"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Partial-charge mapping audit missing columns: {sorted(missing)}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            mol_id = row["mol_id"].strip()
            if not mol_id or mol_id in rows:
                raise ValueError("Partial-charge mapping audit has invalid mol_id values")
            rows[mol_id] = dict(row)
    if not rows:
        raise ValueError("Partial-charge mapping audit is empty")
    return rows


def _partial_dataset(
    dataset: Stage2TaskDataset, source_path: Path, audit_path: Path
) -> Any:
    from chemprop.data import MolAtomBondDatapoint, MolAtomBondDataset

    rows = _source_rows(source_path)
    audit = _mapping_audit(audit_path)
    datapoints = []
    for index, (source_row, mol_id) in enumerate(
        zip(dataset.source_rows.tolist(), dataset.mol_ids, strict=True)
    ):
        row = rows.get(int(source_row))
        if row is None or row["mol_id"].strip() != mol_id:
            raise ValueError("Partial-charge prepared/source row identity mismatch")
        audit_row = audit.get(mol_id)
        if audit_row is None or audit_row["status"] != "mapped":
            raise ValueError("Partial-charge retained row lacks a mapped audit record")
        canonical = audit_row["canonical_smiles"].strip()
        molecule = Chem.MolFromSmiles(canonical)
        if molecule is None:
            raise ValueError(f"Invalid prepared canonical SMILES for mol_id={mol_id}")
        start = int(dataset.atom_target_offsets[index])
        end = int(dataset.atom_target_offsets[index + 1])
        values = dataset.atom_target_values[start:end].float().numpy().copy()
        mask = dataset.atom_target_mask[start:end].numpy()
        values[~mask] = np.nan
        if (
            molecule.GetNumAtoms() != len(values)
            or int(audit_row["model_atom_count"]) != len(values)
        ):
            raise ValueError("Partial-charge prepared atom order/count mismatch")
        datapoints.append(
            MolAtomBondDatapoint.from_smi(
                canonical,
                atom_y=values.reshape(-1, 1),
                reorder_atoms=False,
                name=mol_id,
            )
        )
    return MolAtomBondDataset(datapoints)


def _prepare_partial(config: BenchmarkConfig) -> DMPNNTrainingBundle:
    if config.data.stage2_authority_config is None:
        raise ValueError("D-MPNN Partial Charge requires Stage 2 authority")
    authority_path = config.data.stage2_authority_config
    authority = load_stage2_config(authority_path)
    if authority.data.data_root != config.data.data_root:
        raise ValueError("D-MPNN and Stage 2 authority data roots differ")
    if authority.data.task_catalog_path != config.data.task_catalog:
        raise ValueError("D-MPNN and Stage 2 authority task catalogs differ")
    task = resolve_task(config, "stage2_physics", PARTIAL_CHARGE_TASK, None)
    train_artifact = Stage2TaskDataset(
        authority.data.artifacts_dir, PARTIAL_CHARGE_TASK, "train"
    )
    valid_artifact = Stage2TaskDataset(
        authority.data.artifacts_dir, PARTIAL_CHARGE_TASK, "valid"
    )
    scalers_path = authority.data.artifacts_dir / "scalers.json"
    scalers = json.loads(scalers_path.read_text(encoding="utf-8"))
    stats_payload = scalers[PARTIAL_CHARGE_TASK]["targets"][task.target_columns[0]]
    if stats_payload.get("weighting") != "molecule_equal":
        raise ValueError("Partial-charge scaler must use molecule_equal weighting")
    target_stats = _target_stats(stats_payload)
    condition_stats = ConditionStats((), ())
    train_source = task.train_paths[0]
    valid_source = task.valid_paths[0]
    metadata_path = authority.data.artifacts_dir / "metadata.json"
    audit_path = authority.data.artifacts_dir / "partial_charge_mapping_audit.csv"
    source_hashes = {
        "stage2_authority_config": sha256_file(authority_path),
        "prepared_metadata": sha256_file(metadata_path),
        "scalers": sha256_file(scalers_path),
        "prepared_train": sha256_file(
            authority.data.artifacts_dir / f"tasks/{PARTIAL_CHARGE_TASK}/train.pt"
        ),
        "prepared_valid": sha256_file(
            authority.data.artifacts_dir / f"tasks/{PARTIAL_CHARGE_TASK}/valid.pt"
        ),
        "prepared_mapping_audit": sha256_file(audit_path),
        "train_source": sha256_file(train_source),
        "valid_source": sha256_file(valid_source),
        "mapping_contract": PARTIAL_CHARGE_MAPPING_CONTRACT["hash"],
    }
    identity = semantic_identity(
        "benchmark.training.v1",
        _identity_payload(
            config,
            task,
            target_stats,
            condition_stats,
            source_hashes,
            target_level="atom",
        ),
    )
    return DMPNNTrainingBundle(
        task=task,
        train_dataset=_partial_dataset(train_artifact, train_source, audit_path),
        valid_dataset=_partial_dataset(valid_artifact, valid_source, audit_path),
        target_stats=target_stats,
        condition_stats=condition_stats,
        source_hashes=source_hashes,
        training_identity=identity,
        target_level="atom",
        component_count=1,
    )


def prepare_dmpnn_training(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
) -> DMPNNTrainingBundle:
    if config.name != "dmpnn":
        raise ValueError("D-MPNN adapter requires name=dmpnn")
    if task_id == PARTIAL_CHARGE_TASK:
        if benchmark != "stage2_physics" or fold is not None:
            raise ValueError("Partial Charge is a non-folded Stage 2 benchmark")
        return _prepare_partial(config)
    return _prepare_scalar(config, benchmark, task_id, fold)


def _output_transform(stats: TargetStats) -> Any:
    from chemprop.nn.transforms import UnscaleTransform

    return UnscaleTransform(stats.mean, stats.scale)


def build_dmpnn_model(config: BenchmarkConfig, bundle: DMPNNTrainingBundle) -> Any:
    from chemprop.models import MPNN, MolAtomBondMPNN, MulticomponentMPNN
    from chemprop.nn import (
        BondMessagePassing,
        MABBondMessagePassing,
        MulticomponentMessagePassing,
        NormAggregation,
        RegressionFFN,
    )
    from chemprop.nn.metrics import MAE

    model = config.model
    training = config.training
    message_kwargs = {
        "d_h": int(model["message_hidden_dim"]),
        "depth": int(model["depth"]),
        "dropout": float(model["dropout"]),
        "activation": str(model["activation"]),
    }
    predictor_kwargs = {
        "hidden_dim": int(model["ffn_hidden_dim"]),
        "n_layers": int(model["ffn_hidden_layers"]),
        "dropout": float(model["dropout"]),
        "activation": str(model["activation"]),
        "output_transform": _output_transform(bundle.target_stats),
    }
    schedule = {
        "warmup_epochs": int(training["warmup_epochs"]),
        "init_lr": float(training["initial_learning_rate"]),
        "max_lr": float(training["max_learning_rate"]),
        "final_lr": float(training["final_learning_rate"]),
    }
    if bundle.target_level == "atom":
        message_passing = MABBondMessagePassing(
            **message_kwargs, return_edge_embeddings=False
        )
        predictor = RegressionFFN(
            input_dim=int(message_passing.output_dims[0]), **predictor_kwargs
        )
        return MolAtomBondMPNN(
            message_passing,
            atom_predictor=predictor,
            batch_norm=bool(model["batch_norm"]),
            metrics=[MAE()],
            **schedule,
        )
    shared = bool(model["multicomponent_shared"])
    block_count = 1 if shared else bundle.component_count
    blocks = [BondMessagePassing(**message_kwargs) for _ in range(block_count)]
    message_passing = (
        blocks[0]
        if bundle.component_count == 1
        else MulticomponentMessagePassing(
            blocks,
            n_components=bundle.component_count,
            shared=shared,
        )
    )
    predictor = RegressionFFN(
        input_dim=int(message_passing.output_dim) + len(bundle.condition_stats.mean),
        **predictor_kwargs,
    )
    aggregation = NormAggregation(norm=float(model["aggregation_norm"]))
    model_class = MPNN if bundle.component_count == 1 else MulticomponentMPNN
    return model_class(
        message_passing,
        aggregation,
        predictor,
        batch_norm=bool(model["batch_norm"]),
        metrics=[MAE()],
        **schedule,
    )


class _HistoryCallback:
    def __new__(cls):
        from lightning.pytorch.callbacks import Callback

        class History(Callback):
            def __init__(self) -> None:
                self.rows: list[dict[str, Any]] = []

            def on_validation_epoch_end(self, trainer, _module) -> None:
                if trainer.sanity_checking:
                    return
                metrics = trainer.callback_metrics
                prefix = "atom_" if "atom_val/mae" in metrics else ""
                mae = metrics.get(f"{prefix}val/mae")
                train_loss = metrics.get("train_loss")
                if mae is None:
                    return
                self.rows.append(
                    {
                        "epoch": int(trainer.current_epoch) + 1,
                        "train_normalized_mse": (
                            None if train_loss is None else float(train_loss.detach().cpu())
                        ),
                        "valid_normalized_mae": float(mae.detach().cpu()),
                    }
                )

        return History()


def train_dmpnn_bundle(
    config: BenchmarkConfig,
    bundle: DMPNNTrainingBundle,
    output_dir: str | Path,
) -> dict[str, Any]:
    from chemprop.data import build_dataloader
    from chemprop.models.utils import save_model
    from lightning import pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(config.seed, workers=True)
    model = build_dmpnn_model(config, bundle)
    for dataset in (bundle.train_dataset, bundle.valid_dataset):
        for component_dataset in getattr(dataset, "datasets", (dataset,)):
            component_dataset.cache = True
    train_loader = build_dataloader(
        bundle.train_dataset,
        batch_size=int(config.training["batch_size"]),
        num_workers=8,
        seed=config.seed,
        shuffle=True,
        drop_last=False,
    )
    valid_loader = build_dataloader(
        bundle.valid_dataset,
        batch_size=int(config.training["batch_size"]),
        num_workers=8,
        shuffle=False,
        drop_last=False,
    )
    monitor = "atom_val/mae" if bundle.target_level == "atom" else "val/mae"
    history = _HistoryCallback()
    with tempfile.TemporaryDirectory(prefix="ilume-dmpnn-checkpoint-") as temporary:
        checkpoint = ModelCheckpoint(
            dirpath=temporary,
            filename="best",
            monitor=monitor,
            mode="min",
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False,
        )
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=int(config.training["early_stopping_patience"]),
            mode="min",
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            precision="32-true",
            max_epochs=int(config.training["max_epochs"]),
            callbacks=[history, checkpoint, early_stopping],
            deterministic=True,
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=(
                os.environ.get("ILUME_DISABLE_PROGRESS") != "1"
                and sys.stderr.isatty()
            ),
        )
        trainer.fit(model, train_loader, valid_loader)
        if not checkpoint.best_model_path:
            raise RuntimeError("D-MPNN training did not produce a best checkpoint")
        best_model = type(model).load_from_checkpoint(
            checkpoint.best_model_path, map_location="cpu"
        )
    model_path = root / "model.pt"
    temporary_model = root / "model.pt.tmp"
    save_model(temporary_model, best_model, output_columns=list(bundle.task.target_columns))
    temporary_model.replace(model_path)
    state_hash = tensor_state_hash("benchmark.dmpnn-state.v1", best_model.state_dict())
    best_normalized = float(checkpoint.best_model_score.detach().cpu())
    if not history.rows:
        raise RuntimeError("D-MPNN training produced no validation history")
    for row in history.rows:
        row["valid_raw_mae"] = row["valid_normalized_mae"] * bundle.target_stats.scale[0]
    selected = min(history.rows, key=lambda row: row["valid_normalized_mae"])
    best_epoch = int(selected["epoch"])
    atomic_json(root / "training_history.json", history.rows)
    manifest = {
        "format_version": 1,
        "kind": "ilume_baseline_model",
        "model_kind": (
            "dmpnn_atom"
            if bundle.target_level == "atom"
            else "dmpnn_multicomponent"
            if bundle.component_count > 1
            else "dmpnn_scalar"
        ),
        "training_identity": bundle.training_identity,
        "target_statistics": bundle.target_stats.to_dict(),
        "condition_statistics": bundle.condition_stats.to_dict(),
        "target_columns": list(bundle.task.target_columns),
        "component_count": bundle.component_count,
        "target_level": bundle.target_level,
        "best_epoch": best_epoch,
        "best_valid_normalized_mae": best_normalized,
        "best_valid_raw_mae": best_normalized * bundle.target_stats.scale[0],
        "model_state_hash": state_hash,
        "integrity": {
            "model.pt": {
                "sha256": sha256_file(model_path),
                "size": model_path.stat().st_size,
            }
        },
    }
    atomic_json(root / "checkpoint.json", manifest)
    return {
        "best_epoch": best_epoch,
        "best_valid_raw_mae": manifest["best_valid_raw_mae"],
        "epochs_ran": len(history.rows),
    }


def _manifest(root: Path) -> dict[str, Any]:
    path = root / "checkpoint.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1 or payload.get("kind") != "ilume_baseline_model":
        raise ValueError("Unsupported D-MPNN checkpoint")
    for filename, expected in payload.get("integrity", {}).items():
        artifact = root / filename
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(expected["size"])
            or sha256_file(artifact) != expected["sha256"]
        ):
            raise ValueError(f"D-MPNN checkpoint integrity mismatch: {filename}")
    return payload


def _predict(model: Any, dataset: Any, *, atom: bool) -> np.ndarray:
    from chemprop.data import build_dataloader
    from lightning import pytorch as pl

    torch.set_float32_matmul_precision("high")

    loader = build_dataloader(dataset, batch_size=64, num_workers=8, shuffle=False, drop_last=False)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    outputs = trainer.predict(model, dataloaders=loader)
    if atom:
        values = [item[1] for item in outputs]
    else:
        values = outputs
    if not values or any(item is None for item in values):
        raise RuntimeError("D-MPNN prediction produced no outputs")
    return torch.cat(values).float().cpu().numpy()


def evaluate_dmpnn_checkpoint(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
    checkpoint_dir: str | Path,
    split: str,
) -> EvaluationResult:
    from chemprop.models.utils import load_model

    if task_id == PARTIAL_CHARGE_TASK:
        raise ValueError("Use evaluate_dmpnn_partial for Partial Charge")
    if split not in {"valid", "test"}:
        raise ValueError("D-MPNN evaluation split must be valid or test")
    root = Path(checkpoint_dir)
    manifest = _manifest(root)
    bundle = prepare_dmpnn_training(config, benchmark, task_id, fold)
    require_compatible_identity(
        bundle.training_identity,
        manifest["training_identity"],
        context="D-MPNN evaluation checkpoint",
    )
    raw = load_split(bundle.task, split)
    model = load_model(
        root / "model.pt", multicomponent=bundle.component_count > 1
    )
    if not len(raw):
        predictions = np.empty((0, 1), dtype=np.float64)
    else:
        dataset = _scalar_dataset(raw, bundle.target_stats, bundle.condition_stats)
        predictions = _predict(model, dataset, atom=False)
    metrics = target_metrics(
        predictions,
        raw.targets,
        bundle.task.target_columns,
        bundle.target_stats.scale,
    )
    if task_id in ORBITAL_TASK_TARGETS:
        target = ORBITAL_TASK_TARGETS[task_id]
        metrics[target]["role_diagnostics"] = role_mae_diagnostics(
            predictions[:, 0],
            raw.targets[:, 0],
            [row["ion_role"] for row in raw.audit_rows],
        )
    return EvaluationResult(
        predictions=predictions,
        targets=raw.targets,
        source_rows=raw.source_rows,
        metrics=metrics,
        target_stats=bundle.target_stats,
        training_identity=bundle.training_identity,
        components=raw.components,
        conditions=raw.conditions,
        audit_rows=raw.audit_rows,
    )


def evaluate_dmpnn_partial(
    config: BenchmarkConfig, checkpoint_dir: str | Path
) -> PartialEvaluationResult:
    from chemprop.data import MolAtomBondDatapoint, MolAtomBondDataset
    from chemprop.models.utils import load_model

    root = Path(checkpoint_dir)
    manifest = _manifest(root)
    bundle = prepare_dmpnn_training(
        config, "stage2_physics", PARTIAL_CHARGE_TASK, None
    )
    require_compatible_identity(
        bundle.training_identity,
        manifest["training_identity"],
        context="D-MPNN Partial Charge evaluation checkpoint",
    )
    authority = load_stage2_config(config.data.stage2_authority_config)
    spec = bundle.task
    resource = spec.registry_payload.get("dataset", {}).get("resource_manifest")
    manifest_path = (
        config.data.data_root / str(resource)
        if resource
        else None
    )
    if manifest_path is None:
        from stage2.registry import load_stage2_registry

        registry_spec = load_stage2_registry(config.data.task_catalog).by_id(
            PARTIAL_CHARGE_TASK
        )
        manifest_path = registry_spec.dataset.resource_manifest_path(
            authority.data.data_root
        )
    if manifest_path is None:
        raise ValueError("Partial Charge structure manifest is missing")
    scaler_payload = json.loads(
        (authority.data.artifacts_dir / "scalers.json").read_text(encoding="utf-8")
    )[PARTIAL_CHARGE_TASK]["targets"][spec.target_columns[0]]
    benchmark = build_partial_charge_benchmark(
        spec.test_path, manifest_path, scaler_payload
    )
    datapoints = []
    atom_counts = []
    for molecule in benchmark.evaluated:
        atom_count = len(molecule.target_charges)
        atom_counts.append(atom_count)
        datapoints.append(
            MolAtomBondDatapoint.from_smi(
                molecule.canonical_smiles,
                atom_y=np.full((atom_count, 1), np.nan, dtype=np.float32),
                reorder_atoms=False,
                name=molecule.mol_id,
            )
        )
    model = load_model(root / "model.pt", mol_atom_bond=True)
    flat = _predict(model, MolAtomBondDataset(datapoints), atom=True).reshape(-1)
    predictions: dict[str, np.ndarray] = {}
    offset = 0
    for molecule, count in zip(benchmark.evaluated, atom_counts, strict=True):
        predictions[molecule.mol_id] = flat[offset : min(offset + count, len(flat))]
        offset += count
    if len(flat) > offset:
        predictions["__extra_prediction__"] = flat[offset:]
    score = score_partial_charge_predictions(benchmark, predictions)
    return PartialEvaluationResult(
        benchmark=benchmark,
        predictions=predictions,
        score=score,
        training_identity=bundle.training_identity,
    )


__all__ = [
    "ConditionStats",
    "DMPNNTrainingBundle",
    "DMPNN_GRAPH_CONTRACT",
    "PartialEvaluationResult",
    "build_dmpnn_model",
    "evaluate_dmpnn_checkpoint",
    "evaluate_dmpnn_partial",
    "prepare_dmpnn_training",
    "train_dmpnn_bundle",
]
