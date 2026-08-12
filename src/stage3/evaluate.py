from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from stage2.model import sha256_file
from stage2.prepare import resolve_device
from .config import DOMAIN_TASKS, Stage3Config
from .data import Stage3TaskDataset
from .model import Stage3MultiDomainModel
from .prepare import load_frozen_embeddings
from .train import (
    STAGE3_CHECKPOINT_VERSION,
    STAGE3_DOMAIN_MODEL_KIND,
    STAGE3_MODEL_KIND,
    FrozenRepresentationStore,
)


@torch.no_grad()
def _predict(
    model: Stage3MultiDomainModel,
    domain: str,
    dataset: Stage3TaskDataset,
    store: FrozenRepresentationStore,
    config: Stage3Config,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions: list[torch.Tensor] = []
    batch_size = config.domain_training(domain).batch_size
    store.prepare_dataset(dataset)
    for start in range(0, len(dataset), batch_size):
        indices = torch.arange(
            start,
            min(len(dataset), start + batch_size),
            device=store.device,
        )
        base, conditions, phase_ids, _, solute = store.batch(
            dataset.task, dataset, indices, device
        )
        predictions.append(
            model(
                dataset.task,
                base,
                conditions,
                phase_ids,
                solute_cls=solute,
            ).predictions
        )
    normalized = (
        torch.cat(predictions).float().cpu()
        if predictions
        else torch.empty(0)
    )
    stats = dataset.scalers[f"fold{dataset.fold}"][dataset.task]["target"]
    raw = normalized * float(stats["scale"]) + float(stats["mean"])
    return raw, dataset.raw_targets.float()


def _metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> dict[str, float | int]:
    count = len(targets)
    delta = predictions - targets
    denominator = float(torch.square(targets - targets.mean()).sum())
    return {
        "count": count,
        "mae": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "r2": (
            float("nan")
            if denominator == 0.0
            else 1.0 - float(delta.square().sum()) / denominator
        ),
    }


def _validate_domain_contract(
    checkpoint_contract: dict[str, Any],
    metadata: dict[str, Any],
    artifact_root: Path,
    domain: str,
    checkpoint_path: Path,
) -> None:
    expected = {
        "data_metadata_hash": sha256_file(
            artifact_root / domain / "metadata.json"
        ),
        "source_hashes": metadata["source_hashes"],
    }
    for key, value in expected.items():
        if checkpoint_contract.get(key) != value:
            raise ValueError(
                f"Stage 3 checkpoint {domain}.{key} mismatch: "
                f"{checkpoint_path}"
            )


def _load_fold_model(
    config: Stage3Config,
    checkpoint_dir: Path,
    fold: int,
    metadata: dict[str, dict[str, Any]],
    d_model: int,
    device: torch.device,
) -> tuple[Stage3MultiDomainModel, dict[str, float]]:
    combined = len(config.active_domains) > 1
    filename = "best.pt" if combined else f"best_{config.active_domains[0]}.pt"
    checkpoint_path = checkpoint_dir / f"fold{fold}" / filename
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("format_version") != STAGE3_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported Stage 3 v2 model checkpoint: {checkpoint_path}"
        )
    if checkpoint.get("fold") != fold:
        raise ValueError(
            f"Stage 3 checkpoint fold mismatch: {checkpoint_path}"
        )
    model = Stage3MultiDomainModel(
        config, d_model, seed=config.data.seed + fold
    )
    if combined:
        if (
            checkpoint.get("kind") != STAGE3_MODEL_KIND
            or tuple(checkpoint.get("active_domains", ()))
            != config.active_domains
        ):
            raise ValueError(
                f"Expected combined Stage 3 v2 best.pt: {checkpoint_path}"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        metrics = {
            domain: float(checkpoint["domain_best_metrics"][domain])
            for domain in config.active_domains
        }
        contracts = checkpoint["domains"]
    else:
        domain = config.active_domains[0]
        if (
            checkpoint.get("kind") != STAGE3_DOMAIN_MODEL_KIND
            or checkpoint.get("domain") != domain
        ):
            raise ValueError(
                f"Expected Stage 3 v2 best_{domain}.pt: {checkpoint_path}"
            )
        model.domain_module(domain).load_state_dict(
            checkpoint["model"], strict=True
        )
        metrics = {domain: float(checkpoint["best_metric"])}
        contracts = {
            domain: {
                "data_metadata_hash": checkpoint["data_metadata_hash"],
                "source_hashes": checkpoint["source_hashes"],
            }
        }
    for domain in config.active_domains:
        _validate_domain_contract(
            contracts[domain],
            metadata[domain],
            config.data.artifacts_dir,
            domain,
            checkpoint_path,
        )
    return model.to(device).eval(), metrics


def evaluate_checkpoints(
    config: Stage3Config,
    checkpoint_dir: str | Path,
    *,
    split: str,
    ensemble_folds: bool,
) -> dict[str, Any]:
    if split not in {"valid", "test"}:
        raise ValueError("Stage 3 evaluation split must be valid or test")
    if split == "test" and not ensemble_folds:
        raise ValueError(
            "Fixed Stage 3 test evaluation requires --ensemble-folds"
        )
    device = resolve_device(config.training.device)
    loaded = {
        domain: load_frozen_embeddings(config, domain)
        for domain in config.active_domains
    }
    dimensions = {
        int(metadata["embedding_dim"])
        for _, metadata in loaded.values()
    }
    if len(dimensions) != 1:
        raise ValueError("Stage 3 domain embedding dimensions do not match")
    d_model = dimensions.pop()
    stores = {
        domain: FrozenRepresentationStore(
            payload,
            device=device,
            resident=(
                config.training.resident_data and device.type == "cuda"
            ),
        )
        for domain, (payload, _) in loaded.items()
    }
    metadata = {
        domain: domain_metadata
        for domain, (_, domain_metadata) in loaded.items()
    }
    checkpoint_dir = Path(checkpoint_dir)
    per_task_predictions: dict[str, list[torch.Tensor]] = {
        task: [] for task in config.tasks
    }
    targets: dict[str, torch.Tensor] = {}
    folds: list[dict[str, Any]] = []
    for fold in range(1, 6):
        model, best_metrics = _load_fold_model(
            config,
            checkpoint_dir,
            fold,
            metadata,
            d_model,
            device,
        )
        fold_metrics: dict[str, Any] = {
            "fold": fold,
            "domain_best_macro_normalized_mae": best_metrics,
        }
        for domain in config.active_domains:
            for task in DOMAIN_TASKS[domain]:
                dataset = Stage3TaskDataset(
                    config.data.artifacts_dir,
                    domain,
                    fold,
                    task,
                    split,
                )
                if len(dataset) == 0:
                    continue
                prediction, target = _predict(
                    model,
                    domain,
                    dataset,
                    stores[domain],
                    config,
                    device,
                )
                if split == "valid":
                    fold_metrics[task] = _metrics(prediction, target)
                else:
                    if task in targets and not torch.equal(
                        targets[task], target
                    ):
                        raise ValueError(
                            "Stage 3 fixed test rows differ across fold "
                            f"artifacts: {task}"
                        )
                    targets[task] = target
                    per_task_predictions[task].append(prediction)
        folds.append(fold_metrics)
    if split == "valid":
        domain_summary: dict[str, Any] = {}
        for domain in config.active_domains:
            values = torch.tensor(
                [
                    row["domain_best_macro_normalized_mae"][domain]
                    for row in folds
                ],
                dtype=torch.float64,
            )
            domain_summary[domain] = {
                "macro_normalized_mae_mean": float(values.mean()),
                "macro_normalized_mae_std": float(
                    values.std(unbiased=False)
                ),
            }
        task_summary: dict[str, Any] = {}
        for task in config.tasks:
            task_rows = [row[task] for row in folds if task in row]
            if not task_rows:
                continue
            task_summary[task] = {}
            for metric in ("mae", "rmse", "r2"):
                values = torch.tensor(
                    [float(row[metric]) for row in task_rows],
                    dtype=torch.float64,
                )
                task_summary[task][f"{metric}_mean"] = float(values.mean())
                task_summary[task][f"{metric}_std"] = float(
                    values.std(unbiased=False)
                )
        return {
            "split": "valid",
            "folds": folds,
            "domains": domain_summary,
            "tasks": task_summary,
        }
    tasks: dict[str, Any] = {}
    for task, predictions in per_task_predictions.items():
        if predictions:
            tasks[task] = _metrics(
                torch.stack(predictions).mean(dim=0), targets[task]
            )
    return {"split": "test", "ensemble_folds": 5, "tasks": tasks}


def write_evaluation(
    result: dict[str, Any], output: str | Path | None = None
) -> None:
    serialized = (
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True) + "\n"
    )
    print(serialized, end="")
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(path)
