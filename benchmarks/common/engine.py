from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from common.identity import require_compatible_identity, semantic_identity, tensor_state_hash
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.progress import ProgressReporter
from common.reporting import role_mae_diagnostics
from stage2.registry import ORBITAL_TASK_TARGETS

from .config import BenchmarkConfig, BenchmarkName
from .data import BenchmarkTask, RawDataset, load_split, resolve_task
from .features import (
    FeatureCache,
    FeaturePreprocessor,
    FeatureSchema,
    ensure_finite_raw_features,
    feature_schema,
    raw_feature_matrix,
)
from .metrics import target_metrics


BENCHMARK_CHECKPOINT_VERSION = 1
BENCHMARK_CHECKPOINT_KIND = "ilume_baseline_model"


@dataclass(frozen=True)
class TargetStats:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, values: np.ndarray) -> "TargetStats":
        if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
            raise ValueError("Benchmark train targets must be a non-empty finite matrix")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        if not np.isfinite(scale).all() or bool((scale <= 0).any()):
            raise ValueError("Benchmark train target has zero or invalid variance")
        return cls(tuple(float(value) for value in mean), tuple(float(value) for value in scale))

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return ((values - np.asarray(self.mean)) / np.asarray(self.scale)).astype(np.float32)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return values * np.asarray(self.scale) + np.asarray(self.mean)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TargetStats":
        return cls(tuple(float(value) for value in raw["mean"]), tuple(float(value) for value in raw["scale"]))


@dataclass
class TrainingBundle:
    task: BenchmarkTask
    schema: FeatureSchema
    preprocessor: FeaturePreprocessor | None
    target_stats: TargetStats
    train_features: np.ndarray
    valid_features: np.ndarray
    train_targets: np.ndarray
    valid_targets: np.ndarray
    source_hashes: dict[str, list[str] | str]
    training_identity: dict[str, Any]


def _source_hashes(task: BenchmarkTask) -> dict[str, list[str] | str]:
    return {
        "train": [sha256_file(path) for path in task.train_paths],
        "valid": [sha256_file(path) for path in task.valid_paths],
    }


def _identity_payload(
    config: BenchmarkConfig,
    task: BenchmarkTask,
    schema: FeatureSchema,
    preprocessor: FeaturePreprocessor | None,
    target_stats: TargetStats,
    source_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "benchmark_model": config.name,
        "domain": task.benchmark,
        "task_id": task.task_id,
        "fold": task.fold,
        "registry": task.registry_payload,
        "source_hashes": dict(source_hashes),
        "feature": schema.to_dict(),
        "preprocessing": None if preprocessor is None else preprocessor.to_dict(),
        "target_statistics": target_stats.to_dict(),
        "model": config.model,
        "training": config.training,
        "seed": config.seed,
    }


def prepare_training(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
    *,
    reporter: ProgressReporter | None = None,
) -> TrainingBundle:
    if config.name == "dmpnn":
        from benchmarks.dmpnn.adapter import prepare_dmpnn_training

        return prepare_dmpnn_training(config, benchmark, task_id, fold)  # type: ignore[return-value]
    task = resolve_task(config, benchmark, task_id, fold)
    train = load_split(task, "train")
    valid = load_split(task, "valid")
    if config.features is None or config.data.feature_cache is None:
        raise ValueError("Feature baseline configuration is incomplete")
    schema = feature_schema(config.features)
    fold_suffix = f" fold{fold}" if fold is not None else ""
    with FeatureCache(config.data.feature_cache) as cache:
        train_raw = raw_feature_matrix(
            train,
            schema,
            cache,
            reporter=reporter,
            desc=f"{config.name} {task_id}{fold_suffix} train features",
        )
        valid_raw = raw_feature_matrix(
            valid,
            schema,
            cache,
            reporter=reporter,
            desc=f"{config.name} {task_id}{fold_suffix} valid features",
        )
    preprocessor: FeaturePreprocessor | None
    if config.name == "mlp":
        preprocessor = FeaturePreprocessor.fit(train_raw)
        train_features = preprocessor.transform(train_raw)
        valid_features = preprocessor.transform(valid_raw)
    else:
        preprocessor = None
        train_features = ensure_finite_raw_features(train_raw)
        valid_features = ensure_finite_raw_features(valid_raw)
    target_stats = TargetStats.fit(train.targets)
    source_hashes = _source_hashes(task)
    identity = semantic_identity(
        "benchmark.training.v1",
        _identity_payload(config, task, schema, preprocessor, target_stats, source_hashes),
    )
    return TrainingBundle(
        task=task,
        schema=schema,
        preprocessor=preprocessor,
        target_stats=target_stats,
        train_features=train_features,
        valid_features=valid_features,
        train_targets=train.targets,
        valid_targets=valid.targets,
        source_hashes=source_hashes,
        training_identity=identity,
    )


def seed_benchmark(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    #torch.use_deterministic_algorithms(True)


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    atomic_json(root / "checkpoint.json", manifest)


def train_mlp(
    config: BenchmarkConfig,
    bundle: TrainingBundle,
    output_dir: Path,
    *,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    from benchmarks.mlp.model import DescriptorMLP

    seed_benchmark(config.seed)
    device = torch.device(str(config.training["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("MLP benchmark requires CUDA; no silent CPU fallback")
    if str(config.training.get("precision", "fp32")) != "fp32":
        raise ValueError("MLP benchmark v1 supports FP32 only")
    model = DescriptorMLP(
        bundle.train_features.shape[1],
        len(bundle.task.target_columns),
        tuple(int(value) for value in config.model["hidden_dims"]),
        float(config.model["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training["learning_rate"]),
        weight_decay=float(config.training["weight_decay"]),
    )
    train_x = torch.from_numpy(bundle.train_features)
    train_y = torch.from_numpy(bundle.target_stats.normalize(bundle.train_targets))
    valid_x = torch.from_numpy(bundle.valid_features).to(device)
    batch_size = int(config.training["batch_size"])
    max_epochs = int(config.training["max_epochs"])
    patience = int(config.training["early_stopping_patience"])
    generator = torch.Generator().manual_seed(config.seed)
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    fold_suffix = f" fold{bundle.task.fold}" if bundle.task.fold is not None else ""
    progress = (reporter or ProgressReporter()).bar(
        total=max_epochs,
        desc=f"MLP {bundle.task.task_id}{fold_suffix}",
        unit="epoch",
    )
    try:
        for epoch in range(1, max_epochs + 1):
            model.train()
            order = torch.randperm(len(train_x), generator=generator)
            loss_sum = 0.0
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                features = train_x[indices].to(device)
                targets = train_y[indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(features)
                loss = torch.nn.functional.mse_loss(prediction, targets)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach().cpu()) * len(indices)
            model.eval()
            with torch.inference_mode():
                normalized = model(valid_x).float().cpu().numpy()
            raw = bundle.target_stats.denormalize(normalized)
            per_target_mae = np.abs(raw - bundle.valid_targets).mean(axis=0)
            score = float(per_target_mae.mean())
            train_mse = loss_sum / len(train_x)
            history.append(
                {
                    "epoch": epoch,
                    "train_normalized_mse": train_mse,
                    "valid_raw_macro_mae": score,
                }
            )
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            progress.set_postfix(
                {
                    "train_mse": f"{train_mse:.4f}",
                    "val_mae": f"{score:.4f}",
                    "best": f"{best_score:.4f}@{best_epoch}",
                    "patience": f"{stale}/{patience}",
                }
            )
            progress.update(1)
            if stale >= patience:
                break
    finally:
        progress.close()
    if best_state is None:
        raise RuntimeError("MLP benchmark did not produce a best model")
    state_hash = tensor_state_hash("benchmark.mlp-state.v1", best_state)
    model_path = output_dir / "model.pt"
    atomic_torch_save(model_path, {"state_dict": best_state, "state_hash": state_hash})
    atomic_json(output_dir / "training_history.json", history)
    manifest = {
        "format_version": BENCHMARK_CHECKPOINT_VERSION,
        "kind": BENCHMARK_CHECKPOINT_KIND,
        "model_kind": "mlp",
        "training_identity": bundle.training_identity,
        "feature_schema": bundle.schema.to_dict(),
        "preprocessing": bundle.preprocessor.to_dict() if bundle.preprocessor else None,
        "target_statistics": bundle.target_stats.to_dict(),
        "target_columns": list(bundle.task.target_columns),
        "input_dim": int(bundle.train_features.shape[1]),
        "hidden_dims": list(config.model["hidden_dims"]),
        "dropout": float(config.model["dropout"]),
        "best_epoch": best_epoch,
        "best_valid_raw_macro_mae": best_score,
        "model_state_hash": state_hash,
        "integrity": {"model.pt": {"sha256": sha256_file(model_path), "size": model_path.stat().st_size}},
    }
    _write_manifest(output_dir, manifest)
    return {"best_epoch": best_epoch, "best_valid_raw_macro_mae": best_score, "epochs_ran": len(history)}


def train_xgboost(
    config: BenchmarkConfig,
    bundle: TrainingBundle,
    output_dir: Path,
    *,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    try:
        import xgboost as xgb
    except ImportError as error:
        raise RuntimeError('XGBoost benchmark requires pip install -e ".[benchmarks]"') from error
    models: list[dict[str, Any]] = []
    integrity: dict[str, dict[str, Any]] = {}
    progress_reporter = reporter or ProgressReporter()
    fold_suffix = f" fold{bundle.task.fold}" if bundle.task.fold is not None else ""
    for index, target in enumerate(bundle.task.target_columns):
        progress = progress_reporter.bar(
            total=int(config.model["n_estimators"]),
            desc=f"XGBoost {bundle.task.task_id}{fold_suffix} {target}",
            unit="round",
        )

        class ProgressCallback(xgb.callback.TrainingCallback):
            def after_iteration(
                self,
                model: Any,
                epoch: int,
                evals_log: Any,
            ) -> bool:
                values = evals_log.get("validation_0", {}).get(
                    str(config.model["eval_metric"]), ()
                )
                if values:
                    value = values[-1]
                    metric = value[0] if isinstance(value, tuple) else value
                    progress.set_postfix({"val_mae": f"{float(metric):.4f}"})
                progress.update(max(0, epoch + 1 - int(progress.n)))
                return False

        callback = xgb.callback.EarlyStopping(
            rounds=int(config.training["early_stopping_rounds"]),
            metric_name=str(config.model["eval_metric"]),
            data_name="validation_0",
            maximize=False,
            save_best=True,
        )
        params = {
            **config.model,
            "random_state": config.seed,
            "n_jobs": int(config.training["n_jobs"]),
            "device": str(config.training["device"]),
            "callbacks": [callback, ProgressCallback()],
        }
        model = xgb.XGBRegressor(**params)
        try:
            model.fit(
                bundle.train_features,
                bundle.train_targets[:, index],
                eval_set=[(bundle.valid_features, bundle.valid_targets[:, index])],
                verbose=False,
            )
            evaluation_history = model.evals_result()["validation_0"][
                str(config.model["eval_metric"])
            ]
            progress.update(max(0, len(evaluation_history) - int(progress.n)))
            progress.set_postfix(
                {
                    "val_mae": f"{float(evaluation_history[-1]):.4f}",
                    "best": f"{float(model.best_score):.4f}@{int(model.best_iteration)}",
                }
            )
        finally:
            progress.close()
        filename = f"model_{index}_{target.replace('/', '_')}.json"
        temporary = output_dir / f"tmp_{filename}"
        path = output_dir / filename
        model.save_model(temporary)
        temporary.replace(path)
        best_iteration = int(model.best_iteration)
        entry = {"target": target, "file": filename, "best_iteration": best_iteration, "best_score": float(model.best_score)}
        models.append(entry)
        integrity[filename] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    manifest = {
        "format_version": BENCHMARK_CHECKPOINT_VERSION,
        "kind": BENCHMARK_CHECKPOINT_KIND,
        "model_kind": "ecfp_xgboost",
        "training_identity": bundle.training_identity,
        "feature_schema": bundle.schema.to_dict(),
        "preprocessing": None,
        "target_statistics": bundle.target_stats.to_dict(),
        "target_columns": list(bundle.task.target_columns),
        "input_dim": int(bundle.train_features.shape[1]),
        "models": models,
        "integrity": integrity,
    }
    _write_manifest(output_dir, manifest)
    return {"targets": {entry["target"]: {"best_iteration": entry["best_iteration"], "best_score": entry["best_score"]} for entry in models}}


def train_bundle(
    config: BenchmarkConfig,
    bundle: TrainingBundle,
    output_dir: str | Path,
    *,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    if config.name == "dmpnn":
        from benchmarks.dmpnn.adapter import train_dmpnn_bundle

        return train_dmpnn_bundle(config, bundle, output_dir)  # type: ignore[arg-type]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if config.name == "mlp":
        return train_mlp(config, bundle, root, reporter=reporter)
    return train_xgboost(config, bundle, root, reporter=reporter)


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "checkpoint.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing benchmark checkpoint manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != BENCHMARK_CHECKPOINT_VERSION or manifest.get("kind") != BENCHMARK_CHECKPOINT_KIND:
        raise ValueError("Unsupported benchmark checkpoint")
    for filename, expected in manifest.get("integrity", {}).items():
        artifact = root / filename
        if not artifact.is_file() or artifact.stat().st_size != int(expected["size"]) or sha256_file(artifact) != expected["sha256"]:
            raise ValueError(f"Benchmark checkpoint integrity mismatch: {filename}")
    return manifest


def _predict(config: BenchmarkConfig, manifest: Mapping[str, Any], root: Path, features: np.ndarray) -> np.ndarray:
    if manifest["model_kind"] == "mlp":
        from benchmarks.mlp.model import DescriptorMLP

        device = torch.device(str(config.training["device"]))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("MLP benchmark evaluation requires CUDA")
        model = DescriptorMLP(
            int(manifest["input_dim"]), len(manifest["target_columns"]),
            tuple(int(value) for value in manifest["hidden_dims"]), float(manifest["dropout"]),
        )
        payload = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
        if tensor_state_hash("benchmark.mlp-state.v1", payload["state_dict"]) != manifest["model_state_hash"]:
            raise ValueError("MLP checkpoint state hash mismatch")
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(device).eval()
        with torch.inference_mode():
            normalized = model(torch.from_numpy(features).to(device)).float().cpu().numpy()
        return TargetStats.from_dict(manifest["target_statistics"]).denormalize(normalized)
    try:
        import xgboost as xgb
    except ImportError as error:
        raise RuntimeError('XGBoost evaluation requires pip install -e ".[benchmarks]"') from error
    columns = []
    for entry in manifest["models"]:
        model = xgb.XGBRegressor()
        model.load_model(root / entry["file"])
        if int(model.best_iteration) != int(entry["best_iteration"]):
            raise ValueError(f"XGBoost best_iteration mismatch: {entry['target']}")
        columns.append(model.predict(features, iteration_range=(0, int(entry["best_iteration"]) + 1)))
    return np.column_stack(columns)


@dataclass
class EvaluationResult:
    predictions: np.ndarray
    targets: np.ndarray
    source_rows: tuple[str, ...]
    metrics: dict[str, dict[str, Any]]
    target_stats: TargetStats
    training_identity: dict[str, Any]
    components: tuple[tuple[str, ...], ...] = ()
    conditions: np.ndarray | None = None
    audit_rows: tuple[dict[str, str], ...] = ()


def ensemble_evaluation(
    results: Sequence[EvaluationResult], target_columns: Sequence[str]
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    if len(results) != 5:
        raise ValueError("Stage 3 test ensemble requires exactly five fold results")
    first = results[0]
    if any(
        result.source_rows != first.source_rows
        or not np.array_equal(result.targets, first.targets)
        for result in results[1:]
    ):
        raise ValueError("Stage 3 test targets or row order differ across folds")
    predictions = np.stack([result.predictions for result in results]).mean(axis=0)
    scales = np.asarray([result.target_stats.scale for result in results]).mean(axis=0)
    return predictions, target_metrics(predictions, first.targets, target_columns, scales)


def evaluate_checkpoint(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
    checkpoint_dir: str | Path,
    split: str,
    *,
    reporter: ProgressReporter | None = None,
) -> EvaluationResult:
    if config.name == "dmpnn":
        from benchmarks.dmpnn.adapter import evaluate_dmpnn_checkpoint

        return evaluate_dmpnn_checkpoint(
            config, benchmark, task_id, fold, checkpoint_dir, split
        )
    if split not in {"valid", "test"}:
        raise ValueError("Benchmark evaluation split must be valid or test")
    root = Path(checkpoint_dir)
    manifest = _load_manifest(root)
    bundle = prepare_training(
        config, benchmark, task_id, fold, reporter=reporter
    )
    require_compatible_identity(
        bundle.training_identity,
        manifest["training_identity"],
        context="Benchmark evaluation checkpoint",
    )
    dataset = load_split(bundle.task, split)  # test is opened only in evaluation
    fold_suffix = f" fold{fold}" if fold is not None else ""
    with FeatureCache(config.data.feature_cache) as cache:
        raw = raw_feature_matrix(
            dataset,
            bundle.schema,
            cache,
            reporter=reporter,
            desc=f"{config.name} {task_id}{fold_suffix} {split} features",
        )
    if manifest["model_kind"] == "mlp":
        preprocessor = FeaturePreprocessor.from_dict(manifest["preprocessing"])
        features = preprocessor.transform(raw)
    else:
        features = ensure_finite_raw_features(raw)
    predictions = (
        np.empty((0, len(bundle.task.target_columns)), dtype=np.float64)
        if not len(dataset)
        else _predict(config, manifest, root, features)
    )
    scales = bundle.target_stats.scale
    metrics = target_metrics(predictions, dataset.targets, bundle.task.target_columns, scales)
    if task_id in ORBITAL_TASK_TARGETS:
        target = ORBITAL_TASK_TARGETS[task_id]
        metrics[target]["role_diagnostics"] = role_mae_diagnostics(
            predictions[:, 0],
            dataset.targets[:, 0],
            [row["ion_role"] for row in dataset.audit_rows],
        )
    return EvaluationResult(
        predictions=predictions,
        targets=dataset.targets,
        source_rows=dataset.source_rows,
        metrics=metrics,
        target_stats=bundle.target_stats,
        training_identity=bundle.training_identity,
        components=dataset.components,
        conditions=dataset.conditions,
        audit_rows=dataset.audit_rows,
    )


def write_predictions(
    path: str | Path,
    source_rows: Sequence[str],
    target_columns: Sequence[str],
    targets: np.ndarray,
    predictions: np.ndarray,
    extra_predictions: Mapping[str, np.ndarray] | None = None,
) -> None:
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    fields = ["source_row"]
    for target in target_columns:
        fields.extend((f"target:{target}", f"prediction:{target}"))
    for name in (extra_predictions or {}):
        for target in target_columns:
            fields.append(f"prediction:{name}:{target}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, source_row in enumerate(source_rows):
            row: dict[str, Any] = {"source_row": source_row}
            for column, target in enumerate(target_columns):
                row[f"target:{target}"] = float(targets[index, column])
                row[f"prediction:{target}"] = float(predictions[index, column])
                for name, values in (extra_predictions or {}).items():
                    row[f"prediction:{name}:{target}"] = float(values[index, column])
            writer.writerow(row)
    temporary.replace(destination)


__all__ = [
    "EvaluationResult",
    "TargetStats",
    "ensemble_evaluation",
    "evaluate_checkpoint",
    "prepare_training",
    "train_bundle",
    "write_predictions",
]
