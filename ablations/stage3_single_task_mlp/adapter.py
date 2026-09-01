from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from common.identity import (
    require_compatible_identity,
    semantic_identity,
    tensor_state_hash,
)
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.progress import ProgressReporter
from stage3.config import Stage3Config, load_stage3_config
from stage3.data import ResolvedTaskSpec, Stage3TaskDataset, stable_seed
from stage3.identity import metadata_identity
from stage3.prepare import load_prepared_stage3

from benchmarks.common.config import BenchmarkConfig, BenchmarkName
from benchmarks.common.data import BenchmarkTask, RawDataset, load_split, resolve_task
from benchmarks.common.engine import EvaluationResult, TargetStats, seed_benchmark
from benchmarks.common.metrics import target_metrics
from .model import Stage3SingleTaskMLP


MODEL_KIND = "ilume_stage3_single_task_mlp"
CHECKPOINT_VERSION = 1
CHECKPOINT_KIND = "ilume_stage3_single_task_mlp_model"
STATE_NAMESPACE = "benchmark.ilume-stage3-single-task-mlp-state.v1"
INPUT_CONTRACT = {
    "embedding_source": "stage3_prepared_frozen_stage2_object_v1",
    "embedding_dim": 512,
    "embedding_transform": "none",
    "feature_order": [
        "primary_embedding",
        "partner_embedding_if_declared",
        "normalized_conditions",
    ],
    "partner_fusion": "ordered_concat",
    "condition_fusion": "ordered_concat",
}


@dataclass
class Stage3SingleTaskMLPBundle:
    task: BenchmarkTask
    spec: ResolvedTaskSpec
    train: Stage3TaskDataset
    valid: Stage3TaskDataset
    embeddings: torch.Tensor
    target_stats: TargetStats
    input_dim: int
    prepared_identity: dict[str, Any]
    stage2_encoder_identity: dict[str, Any]
    training_seed: int
    training_identity: dict[str, Any]


def _target_stats(normalization: Mapping[str, Any]) -> TargetStats:
    values = normalization["target"]
    mean = float(values["mean"])
    scale = float(values["scale"])
    if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Stage3 Single-task MLP target normalization is invalid")
    return TargetStats((mean,), (scale,))


def _load_prepared(config: BenchmarkConfig) -> tuple[Stage3Config, dict[str, Any]]:
    if config.data.stage3_prepared_artifacts is None:
        raise ValueError("Stage3 Single-task MLP prepared artifacts are not configured")
    authority = load_stage3_config(config.data.stage3_authority_config)
    if authority.data.artifacts_dir != config.data.stage3_prepared_artifacts:
        raise ValueError("Stage3 authority and ablation prepared artifact paths differ")
    prepared = load_prepared_stage3(authority)
    metadata = prepared["metadata"]
    if metadata.get("embedding_dim") != 512:
        raise ValueError("Stage3 Single-task MLP requires 512-dimensional embeddings")
    embeddings = prepared["objects"].get("embeddings")
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 2
        or embeddings.dtype != torch.float32
        or embeddings.shape[1] != 512
        or not torch.isfinite(embeddings).all()
    ):
        raise ValueError("Stage3 Single-task MLP object embeddings are invalid")
    return authority, prepared


def build_input_features(
    dataset: Stage3TaskDataset,
    embeddings: torch.Tensor,
    spec: ResolvedTaskSpec,
) -> torch.Tensor:
    if (
        dataset.conditions.ndim != 2
        or dataset.conditions.shape[1] != len(spec.condition_columns)
    ):
        raise ValueError(
            f"Stage3 Single-task MLP condition width mismatch: {spec.task_id}"
        )
    if not torch.isfinite(dataset.conditions).all():
        raise ValueError(f"Stage3 Single-task MLP conditions are non-finite: {spec.task_id}")
    if (
        dataset.primary_object_ids.ndim != 1
        or len(dataset.primary_object_ids) != len(dataset)
    ):
        raise ValueError(
            f"Stage3 Single-task MLP primary ids are malformed: {spec.task_id}"
        )
    if len(dataset) and (
        int(dataset.primary_object_ids.min()) < 0
        or int(dataset.primary_object_ids.max()) >= len(embeddings)
    ):
        raise ValueError(
            f"Stage3 Single-task MLP primary ids are out of range: {spec.task_id}"
        )
    chunks = [embeddings[dataset.primary_object_ids]]
    partner_ids = dataset.partner_object_ids
    if partner_ids.ndim != 1 or len(partner_ids) != len(dataset):
        raise ValueError(f"Stage3 Single-task MLP partner ids are malformed: {spec.task_id}")
    if spec.partner_slots:
        if len(dataset) and (
            int(partner_ids.min()) < 0 or int(partner_ids.max()) >= len(embeddings)
        ):
            raise ValueError(
                f"Stage3 Single-task MLP partner embedding is missing: {spec.task_id}"
            )
        chunks.append(embeddings[partner_ids])
    elif len(dataset) and bool((partner_ids != -1).any()):
        raise ValueError(f"Stage3 Single-task MLP received an undeclared partner: {spec.task_id}")
    chunks.append(dataset.conditions.float())
    features = torch.cat(chunks, dim=1)
    if features.dtype != torch.float32 or not torch.isfinite(features).all():
        raise ValueError(f"Stage3 Single-task MLP features are invalid: {spec.task_id}")
    expected = 512 * (2 if spec.partner_slots else 1) + len(spec.condition_columns)
    if features.shape != (len(dataset), expected):
        raise ValueError(f"Stage3 Single-task MLP input shape mismatch: {spec.task_id}")
    return features.contiguous()


def _check_raw_alignment(
    dataset: Stage3TaskDataset,
    raw: RawDataset,
    normalization: Mapping[str, Any],
    condition_columns: tuple[str, ...],
    *,
    context: str,
) -> None:
    if len(dataset) != len(raw):
        raise ValueError(f"Stage3 Single-task MLP row count mismatch: {context}")
    raw_targets = dataset.raw_targets.numpy().astype(np.float64).reshape(-1, 1)
    if not np.allclose(raw_targets, raw.targets, rtol=1e-6, atol=1e-6):
        raise ValueError(f"Stage3 Single-task MLP target order mismatch: {context}")
    source_rows = tuple(int(value.rsplit(":", 1)[1]) for value in raw.source_rows)
    if source_rows != tuple(int(value) for value in dataset.source_rows.tolist()):
        raise ValueError(f"Stage3 Single-task MLP source row mismatch: {context}")
    if raw.conditions.shape[1]:
        expected = np.column_stack(
            [
                (
                    raw.conditions[:, index]
                    - float(normalization["conditions"][name]["mean"])
                )
                / float(normalization["conditions"][name]["scale"])
                for index, name in enumerate(condition_columns)
            ]
        ).astype(np.float32)
    else:
        expected = np.empty((len(raw), 0), dtype=np.float32)
    if not np.allclose(dataset.conditions.numpy(), expected, rtol=1e-6, atol=1e-6):
        raise ValueError(f"Stage3 Single-task MLP condition normalization mismatch: {context}")


def _identity_payload(
    config: BenchmarkConfig,
    task: BenchmarkTask,
    *,
    prepared: Mapping[str, Any],
    target_stats: TargetStats,
    input_dim: int,
    training_seed: int,
) -> dict[str, Any]:
    return {
        "benchmark_model": config.name,
        "domain": task.benchmark,
        "task_id": task.task_id,
        "fold": task.fold,
        "registry": task.registry_payload,
        "prepared_identity": metadata_identity(
            prepared["metadata"], "prepared", context="Stage3 Single-task MLP artifact"
        )["hash"],
        "stage2_encoder_identity": metadata_identity(
            prepared["metadata"], "stage2_encoder", context="Stage3 Single-task MLP artifact"
        )["hash"],
        "artifact_hashes": prepared["metadata"]["artifact_hashes"],
        "input_contract": INPUT_CONTRACT,
        "input_dim": input_dim,
        "target_statistics": target_stats.to_dict(),
        "model": config.model,
        "training": config.training,
        "seed": training_seed,
    }


def prepare_stage3_single_task_mlp_training(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
) -> Stage3SingleTaskMLPBundle:
    if benchmark != "stage3" or fold not in range(1, 6):
        raise ValueError("Stage3 Single-task MLP requires one Stage 3 fold")
    task = resolve_task(config, benchmark, task_id, fold)
    authority, prepared = _load_prepared(config)
    spec = prepared["registry"][task_id]
    train = Stage3TaskDataset(authority.data.artifacts_dir, fold, task_id, "train")
    valid = Stage3TaskDataset(authority.data.artifacts_dir, fold, task_id, "valid")
    normalization = prepared["normalization"][f"fold{fold}"][task_id]
    raw_train = load_split(task, "train")
    raw_valid = load_split(task, "valid")
    _check_raw_alignment(
        train,
        raw_train,
        normalization,
        spec.condition_columns,
        context=f"{task_id}/fold{fold}/train",
    )
    _check_raw_alignment(
        valid,
        raw_valid,
        normalization,
        spec.condition_columns,
        context=f"{task_id}/fold{fold}/valid",
    )
    embeddings = prepared["objects"]["embeddings"].float()
    input_dim = int(build_input_features(train, embeddings, spec).shape[1])
    if int(build_input_features(valid, embeddings, spec).shape[1]) != input_dim:
        raise ValueError(f"Stage3 Single-task MLP train/valid input mismatch: {task_id}")
    target_stats = _target_stats(normalization)
    training_seed = stable_seed(config.seed, task_id, fold) % (2**32)
    identity = semantic_identity(
        "benchmark.training.v1",
        _identity_payload(
            config,
            task,
            prepared=prepared,
            target_stats=target_stats,
            input_dim=input_dim,
            training_seed=training_seed,
        ),
    )
    return Stage3SingleTaskMLPBundle(
        task=task,
        spec=spec,
        train=train,
        valid=valid,
        embeddings=embeddings,
        target_stats=target_stats,
        input_dim=input_dim,
        prepared_identity=dict(
            metadata_identity(
                prepared["metadata"],
                "prepared",
                context="Stage3 Single-task MLP artifact",
            )
        ),
        stage2_encoder_identity=dict(
            metadata_identity(
                prepared["metadata"],
                "stage2_encoder",
                context="Stage3 Single-task MLP artifact",
            )
        ),
        training_seed=training_seed,
        training_identity=identity,
    )


def _lr_factor(step: int, warmup: int, total: int, floor: float) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return floor + (1.0 - floor) * cosine


def _optimizer(model: torch.nn.Module, config: BenchmarkConfig) -> torch.optim.AdamW:
    decay = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(config.training["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(config.training["learning_rate"]),
        betas=tuple(float(value) for value in config.training["betas"]),
        eps=float(config.training["eps"]),
        foreach=False,
        fused=False,
    )


@torch.no_grad()
def _predict_normalized(
    model: Stage3SingleTaskMLP,
    features: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    use_bf16: bool,
) -> torch.Tensor:
    model.eval()
    values: list[torch.Tensor] = []
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            prediction = model(batch).squeeze(-1)
        if not torch.isfinite(prediction).all():
            raise RuntimeError("Stage3 Single-task MLP produced a non-finite prediction")
        values.append(prediction.float().cpu())
    return torch.cat(values) if values else torch.empty(0, dtype=torch.float32)


def _run_training_epochs(
    model: Stage3SingleTaskMLP,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    valid_features: torch.Tensor,
    valid_targets: torch.Tensor,
    config: BenchmarkConfig,
    *,
    training_seed: int,
    device: torch.device,
    use_bf16: bool,
    reporter: ProgressReporter | None,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], int, float]:
    batch_size = int(config.training["batch_size"])
    epochs = int(config.training["max_epochs"])
    steps_per_epoch = math.ceil(len(train_features) / batch_size)
    total_steps = epochs * steps_per_epoch
    warmup_steps = math.ceil(float(config.training["warmup_ratio"]) * total_steps)
    optimizer = _optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_factor(
            step,
            warmup_steps,
            total_steps,
            float(config.training["min_lr_ratio"]),
        ),
    )
    generator = torch.Generator().manual_seed(training_seed)
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    progress = (reporter or ProgressReporter()).bar(
        total=epochs,
        desc="ILUME Stage3 Single-task MLP",
        unit="epoch",
    )
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(train_features), generator=generator)
            loss_sum = 0.0
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                features = train_features[indices].to(device)
                targets = train_targets[indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bf16,
                ):
                    predictions = model(features).squeeze(-1)
                    loss = torch.nn.functional.smooth_l1_loss(
                        predictions,
                        targets,
                        beta=float(config.training["smooth_l1_beta"]),
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError("Stage3 Single-task MLP produced a non-finite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config.training["max_grad_norm"]),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                scheduler.step()
                loss_sum += float(loss.detach().float().cpu()) * len(indices)
            validation = _predict_normalized(
                model,
                valid_features,
                batch_size=batch_size,
                device=device,
                use_bf16=use_bf16,
            )
            score = float((validation - valid_targets).abs().mean())
            if not math.isfinite(score):
                raise RuntimeError("Stage3 Single-task MLP validation metric is non-finite")
            history.append(
                {
                    "epoch": epoch,
                    "train_normalized_smooth_l1": loss_sum / len(train_features),
                    "valid_normalized_mae": score,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            progress.set_postfix(
                {"val_nmae": f"{score:.4f}", "best": f"{best_score:.4f}@{best_epoch}"}
            )
            progress.update(1)
    finally:
        progress.close()
    if best_state is None or len(history) != epochs:
        raise RuntimeError("Stage3 Single-task MLP did not finish its fixed epoch budget")
    return history, best_state, best_epoch, best_score


def train_stage3_single_task_mlp_bundle(
    config: BenchmarkConfig,
    bundle: Stage3SingleTaskMLPBundle,
    output_dir: str | Path,
    *,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seed_benchmark(bundle.training_seed)
    device = torch.device(str(config.training["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage3 Single-task MLP requires CUDA; no silent CPU fallback")
    if config.training["precision"] != "bf16" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage3 Single-task MLP requires BF16-capable CUDA")
    train_features = build_input_features(bundle.train, bundle.embeddings, bundle.spec)
    valid_features = build_input_features(bundle.valid, bundle.embeddings, bundle.spec)
    model = Stage3SingleTaskMLP(
        bundle.input_dim,
        tuple(int(value) for value in config.model["hidden_dims"]),
        float(config.model["dropout"]),
    ).to(device)
    history, best_state, best_epoch, best_score = _run_training_epochs(
        model,
        train_features,
        bundle.train.targets.float(),
        valid_features,
        bundle.valid.targets.float(),
        config,
        training_seed=bundle.training_seed,
        device=device,
        use_bf16=True,
        reporter=reporter,
    )
    state_hash = tensor_state_hash(STATE_NAMESPACE, best_state)
    model_path = root / "model.pt"
    atomic_torch_save(model_path, {"state_dict": best_state, "state_hash": state_hash})
    history_path = root / "training_history.json"
    atomic_json(history_path, history)
    manifest = {
        "format_version": CHECKPOINT_VERSION,
        "kind": CHECKPOINT_KIND,
        "model_kind": MODEL_KIND,
        "training_identity": bundle.training_identity,
        "prepared_identity": bundle.prepared_identity,
        "stage2_encoder_identity": bundle.stage2_encoder_identity,
        "input_contract": INPUT_CONTRACT,
        "input_dim": bundle.input_dim,
        "hidden_dims": list(config.model["hidden_dims"]),
        "activation": config.model["activation"],
        "dropout": float(config.model["dropout"]),
        "target_statistics": {
            "mean": list(bundle.target_stats.mean),
            "scale": list(bundle.target_stats.scale),
        },
        "target_columns": list(bundle.task.target_columns),
        "best_epoch": best_epoch,
        "best_valid_normalized_mae": best_score,
        "epochs_ran": len(history),
        "model_selector": "validation_best",
        "checkpoint_epoch": None,
        "model_state_hash": state_hash,
        "integrity": {
            "model.pt": {
                "sha256": sha256_file(model_path),
                "size": model_path.stat().st_size,
            },
            "training_history.json": {
                "sha256": sha256_file(history_path),
                "size": history_path.stat().st_size,
            },
        },
    }
    atomic_json(root / "checkpoint.json", manifest)
    return {
        "best_epoch": best_epoch,
        "best_valid_normalized_mae": best_score,
        "epochs_ran": len(history),
        "model_selector": "validation_best",
        "checkpoint_epoch": None,
    }


def _manifest(root: Path) -> dict[str, Any]:
    path = root / "checkpoint.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("format_version") != CHECKPOINT_VERSION
        or payload.get("kind") != CHECKPOINT_KIND
    ):
        raise ValueError("Unsupported Stage3 Single-task MLP checkpoint")
    for filename, expected in payload.get("integrity", {}).items():
        artifact = root / filename
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(expected["size"])
            or sha256_file(artifact) != expected["sha256"]
        ):
            raise ValueError(
                f"Stage3 Single-task MLP checkpoint integrity mismatch: {filename}"
            )
    return payload


def evaluate_stage3_single_task_mlp_checkpoint(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
    checkpoint_dir: str | Path,
    split: str,
) -> EvaluationResult:
    if split not in {"valid", "test"}:
        raise ValueError("Stage3 Single-task MLP evaluation split must be valid or test")
    root = Path(checkpoint_dir)
    manifest = _manifest(root)
    bundle = prepare_stage3_single_task_mlp_training(config, benchmark, task_id, fold)
    require_compatible_identity(
        bundle.training_identity,
        manifest["training_identity"],
        context="Stage3 Single-task MLP evaluation checkpoint",
    )
    expected_manifest = {
        "model_kind": MODEL_KIND,
        "prepared_identity": bundle.prepared_identity,
        "stage2_encoder_identity": bundle.stage2_encoder_identity,
        "input_contract": INPUT_CONTRACT,
        "input_dim": bundle.input_dim,
        "hidden_dims": list(config.model["hidden_dims"]),
        "activation": "silu",
        "dropout": float(config.model["dropout"]),
        "target_statistics": {
            "mean": list(bundle.target_stats.mean),
            "scale": list(bundle.target_stats.scale),
        },
        "target_columns": list(bundle.task.target_columns),
        "epochs_ran": int(config.training["max_epochs"]),
        "model_selector": "validation_best",
        "checkpoint_epoch": None,
    }
    for name, expected in expected_manifest.items():
        if manifest.get(name) != expected:
            raise ValueError(f"Stage3 Single-task MLP checkpoint mismatch: {name}")
    authority, prepared = _load_prepared(config)
    dataset = Stage3TaskDataset(authority.data.artifacts_dir, int(fold), task_id, split)
    raw = load_split(bundle.task, split)
    normalization = prepared["normalization"][f"fold{fold}"][task_id]
    spec = prepared["registry"][task_id]
    _check_raw_alignment(
        dataset,
        raw,
        normalization,
        spec.condition_columns,
        context=f"{task_id}/fold{fold}/{split}",
    )
    features = build_input_features(dataset, bundle.embeddings, spec)
    model = Stage3SingleTaskMLP(
        int(manifest["input_dim"]),
        tuple(int(value) for value in manifest["hidden_dims"]),
        float(manifest["dropout"]),
    )
    payload = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    state_hash = tensor_state_hash(STATE_NAMESPACE, payload["state_dict"])
    if (
        state_hash != manifest["model_state_hash"]
        or payload.get("state_hash") != state_hash
    ):
        raise ValueError("Stage3 Single-task MLP checkpoint state hash mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    device = torch.device(str(config.training["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage3 Single-task MLP evaluation requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage3 Single-task MLP evaluation requires BF16-capable CUDA")
    model.to(device)
    normalized = _predict_normalized(
        model,
        features,
        batch_size=int(config.training["batch_size"]),
        device=device,
        use_bf16=True,
    ).numpy().reshape(-1, 1)
    predictions = bundle.target_stats.denormalize(normalized)
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
    "INPUT_CONTRACT",
    "Stage3SingleTaskMLPBundle",
    "build_input_features",
    "evaluate_stage3_single_task_mlp_checkpoint",
    "prepare_stage3_single_task_mlp_training",
    "train_stage3_single_task_mlp_bundle",
]
