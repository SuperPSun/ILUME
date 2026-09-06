from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
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


def _lock_sha(config: BenchmarkConfig) -> str:
    if config.environment is None:
        raise ValueError("D-MPNN environment contract is missing")
    return sha256_file(repository_path(config.environment.lock))


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


def prepare_dmpnn_training(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
) -> DMPNNTrainingBundle:
    if config.name != "dmpnn":
        raise ValueError("D-MPNN adapter requires name=dmpnn")
    return _prepare_scalar(config, benchmark, task_id, fold)


def _output_transform(stats: TargetStats) -> Any:
    from chemprop.nn.transforms import UnscaleTransform

    return UnscaleTransform(stats.mean, stats.scale)


def build_dmpnn_model(config: BenchmarkConfig, bundle: DMPNNTrainingBundle) -> Any:
    from chemprop.models import MPNN, MulticomponentMPNN
    from chemprop.nn import (
        BondMessagePassing,
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
    monitor = "val/mae"
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
            "dmpnn_multicomponent"
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


def _predict(model: Any, dataset: Any) -> np.ndarray:
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
        predictions = _predict(model, dataset)
    metrics = target_metrics(
        predictions,
        raw.targets,
        bundle.task.target_columns,
        bundle.target_stats.scale,
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


__all__ = [
    "ConditionStats",
    "DMPNNTrainingBundle",
    "DMPNN_GRAPH_CONTRACT",
    "build_dmpnn_model",
    "evaluate_dmpnn_checkpoint",
    "prepare_dmpnn_training",
    "train_dmpnn_bundle",
]
