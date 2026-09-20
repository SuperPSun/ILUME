from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ablations.stage3_single_task_mlp.model import Stage3SingleTaskMLP
from common.identity import semantic_identity, tensor_state_hash
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.progress import ProgressReporter
from common.training import canonical_json_sha256, resolve_device, seed_everything
from stage3.config import load_stage3_config
from stage3.data import ObjectKey, Stage3TaskDataset, stable_seed
from stage3.identity import metadata_identity
from stage3.prepare import load_prepared_stage3, materialize_object_embeddings

from .config import TransferExperimentConfig


REPRESENTATION_KIND = "ilume_stage2_stage3_transfer_representation"
REPRESENTATION_VERSION = 1
MODEL_KIND = "ilume_stage2_stage3_transfer_mlp"
MODEL_VERSION = 1


def transfer_training_seed(experiment_seed: int, task_id: str, fold: int) -> int:
    """Derive the task/fold seed within NumPy's accepted seed range."""
    return stable_seed(
        experiment_seed, "stage2-stage3-transfer", task_id, fold
    ) % (2**32)


def _object_keys(objects: Mapping[str, Any]) -> tuple[ObjectKey, ...]:
    return tuple(
        ObjectKey(
            topology=item["topology"],
            slots=tuple(tuple(slot) for slot in item["slots"]),
        )
        for item in objects["objects"]
    )


def prepare_representation_bank(
    experiment: TransferExperimentConfig,
    *,
    variant: str,
    encoder_path: str | Path,
    encoder_manifest_path: str | Path,
    destination: str | Path,
    expected_initial_shared_state_hash: str,
) -> dict[str, Any]:
    authority = load_stage3_config(experiment.stage3.authority_config)
    prepared = load_prepared_stage3(authority)
    keys = _object_keys(prepared["objects"])
    encoder_path = Path(encoder_path)
    encoder_manifest_path = Path(encoder_manifest_path)
    encoder_manifest = json.loads(encoder_manifest_path.read_text(encoding="utf-8"))
    if encoder_manifest.get("encoder_sha256") != sha256_file(encoder_path):
        raise ValueError("Transfer encoder manifest does not match its artifact")
    if encoder_manifest.get("initial_shared_state_hash") != expected_initial_shared_state_hash:
        raise ValueError("Transfer encoder did not start from the common baseline state")
    if variant == "baseline":
        if encoder_manifest.get("source_task") is not None or encoder_manifest.get("optimizer_updates") != 0:
            raise ValueError("Transfer baseline must be the zero-update encoder")
    elif encoder_manifest.get("source_task") != variant or not encoder_manifest.get("physics_only"):
        raise ValueError("Transfer source encoder manifest does not match its variant")
    destination = Path(destination)
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError(f"Transfer representation already exists: {destination}")
    variant_config = replace(
        authority,
        initialization=replace(authority.initialization, stage2_encoder=encoder_path),
        preparation=replace(
            authority.preparation,
            cache_dir=destination.parent / "object_cache" / variant,
        ),
    )
    embeddings, encoder_identity, cache = materialize_object_embeddings(
        variant_config, keys
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(keys) or embeddings.shape[1] != 1024:
        raise ValueError("Transfer representation bank must be object_count x 1024")
    object_list_hash = canonical_json_sha256([key.to_dict() for key in keys])
    embedding_hash = tensor_state_hash(
        "stage2-stage3-transfer-representation.v1", {"embeddings": embeddings}
    )
    prepared_identity = dict(
        metadata_identity(
            prepared["metadata"], "prepared", context="Stage 3 transfer authority"
        )
    )
    identity = semantic_identity(
        "stage2-stage3.transfer-representation",
        {
            "contract_version": 1,
            "variant": variant,
            "stage3_prepared_identity": prepared_identity["hash"],
            "stage2_encoder_identity": encoder_identity["hash"],
            "encoder_artifact_sha256": sha256_file(encoder_path),
            "object_list_hash": object_list_hash,
            "embedding_hash": embedding_hash,
            "shape": list(embeddings.shape),
        },
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        destination,
        {
            "kind": REPRESENTATION_KIND,
            "format_version": REPRESENTATION_VERSION,
            "identity": identity,
            "variant": variant,
            "stage3_prepared_identity": prepared_identity,
            "stage2_encoder_identity": encoder_identity,
            "objects": [key.to_dict() for key in keys],
            "object_list_hash": object_list_hash,
            "embedding_hash": embedding_hash,
            "embeddings": embeddings.float().contiguous(),
        },
    )
    manifest = {
        "kind": REPRESENTATION_KIND,
        "format_version": REPRESENTATION_VERSION,
        "artifact": destination.name,
        "artifact_sha256": sha256_file(destination),
        "identity": identity,
        "variant": variant,
        "stage3_prepared_identity": prepared_identity,
        "stage2_encoder_identity": encoder_identity,
        "object_list_hash": object_list_hash,
        "embedding_hash": embedding_hash,
        "shape": list(embeddings.shape),
        "cache": cache,
    }
    atomic_json(destination.with_suffix(".json"), manifest)
    return manifest


def load_representation_bank(
    path: str | Path, *, expected_prepared_identity: Mapping[str, Any]
) -> dict[str, Any]:
    path = Path(path)
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest.get("artifact_sha256") != sha256_file(path):
        raise ValueError("Transfer representation artifact hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != REPRESENTATION_KIND or payload.get("format_version") != REPRESENTATION_VERSION:
        raise ValueError("Unsupported transfer representation artifact")
    if payload.get("stage3_prepared_identity") != dict(expected_prepared_identity):
        raise ValueError("Transfer representation uses a different Stage 3 authority")
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2 or embeddings.shape[1] != 1024 or not torch.isfinite(embeddings).all():
        raise ValueError("Malformed transfer representation matrix")
    if payload.get("embedding_hash") != tensor_state_hash(
        "stage2-stage3-transfer-representation.v1", {"embeddings": embeddings}
    ):
        raise ValueError("Transfer representation embedding hash mismatch")
    return payload


def build_transfer_features(
    dataset: Stage3TaskDataset,
    embeddings: torch.Tensor,
    *,
    has_partner: bool,
) -> torch.Tensor:
    parts = [embeddings[dataset.primary_object_ids.long()]]
    if has_partner:
        if bool((dataset.partner_object_ids < 0).any()):
            raise ValueError("Transfer interaction row is missing a partner")
        parts.append(embeddings[dataset.partner_object_ids.long()])
    elif len(dataset) and bool((dataset.partner_object_ids != -1).any()):
        raise ValueError("Transfer non-interaction row has an unexpected partner")
    parts.append(dataset.conditions.float())
    result = torch.cat(parts, dim=1).contiguous()
    if not torch.isfinite(result).all():
        raise ValueError("Transfer MLP features are non-finite")
    return result


def _lr_factor(step: int, warmup: int, total: int, floor: float) -> float:
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return floor + (1.0 - floor) * cosine


@torch.inference_mode()
def _predict(
    model: Stage3SingleTaskMLP, features: torch.Tensor, *, batch_size: int,
    device: torch.device, amp: bool,
) -> torch.Tensor:
    model.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, len(features), batch_size):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            value = model(features[start : start + batch_size].to(device)).squeeze(-1)
        chunks.append(value.float().cpu())
    return torch.cat(chunks) if chunks else torch.empty(0)


def train_transfer_job(
    experiment: TransferExperimentConfig,
    *,
    variant: str,
    representation_path: str | Path,
    task_id: str,
    fold: int,
    output_dir: str | Path,
    device_name: str | None = None,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    if task_id not in experiment.stage3.targets or fold not in experiment.stage3.folds:
        raise ValueError("Transfer MLP task/fold is outside the configured matrix")
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"Transfer MLP output exists: {root}")
    authority = load_stage3_config(experiment.stage3.authority_config)
    prepared = load_prepared_stage3(authority)
    prepared_identity = dict(metadata_identity(prepared["metadata"], "prepared", context="Stage 3 transfer authority"))
    bank = load_representation_bank(
        representation_path, expected_prepared_identity=prepared_identity
    )
    if bank["objects"] != prepared["objects"]["objects"]:
        raise ValueError("Transfer representation object order mismatch")
    spec = prepared["registry"][task_id]
    train = Stage3TaskDataset(authority.data.artifacts_dir, fold, task_id, "train")
    valid = Stage3TaskDataset(authority.data.artifacts_dir, fold, task_id, "valid")
    train_features = build_transfer_features(
        train, bank["embeddings"], has_partner=bool(spec.partner_slots)
    )
    valid_features = build_transfer_features(
        valid, bank["embeddings"], has_partner=bool(spec.partner_slots)
    )
    recipe = experiment.stage3
    training_seed = transfer_training_seed(experiment.seed, task_id, fold)
    seed_everything(training_seed)
    device = resolve_device(device_name or "auto")
    model = Stage3SingleTaskMLP(
        int(train_features.shape[1]), recipe.hidden_dims, recipe.dropout
    ).to(device)
    initial_state_hash = tensor_state_hash(
        "stage2-stage3-transfer-mlp-initial.v1",
        {name: value.detach().cpu() for name, value in model.state_dict().items()},
    )
    decay = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": recipe.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=recipe.learning_rate, betas=recipe.betas, eps=recipe.eps,
        foreach=False, fused=False,
    )
    steps_per_epoch = math.ceil(len(train) / recipe.batch_size)
    total_steps = steps_per_epoch * recipe.epochs
    warmup = math.ceil(recipe.warmup_fraction * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_factor(step, warmup, total_steps, recipe.min_lr_ratio),
    )
    amp = recipe.amp_dtype == "bf16" and device.type == "cuda"
    generator = torch.Generator().manual_seed(training_seed)
    permutation_hashes: list[str] = []
    history: list[dict[str, Any]] = []
    progress = (reporter or ProgressReporter()).bar(
        total=recipe.epochs,
        desc=f"Transfer {variant}/{task_id}/fold{fold}",
        unit="epoch",
    )
    try:
        for epoch in range(1, recipe.epochs + 1):
            model.train()
            order = torch.randperm(len(train), generator=generator)
            permutation_hashes.append(
                tensor_state_hash(
                    "stage2-stage3-transfer-permutation.v1", {"order": order}
                )
            )
            loss_sum = 0.0
            for start in range(0, len(order), recipe.batch_size):
                indices = order[start : start + recipe.batch_size]
                inputs = train_features[indices].to(device)
                targets = train.targets[indices].float().to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=amp
                ):
                    prediction = model(inputs).squeeze(-1)
                    loss = F.smooth_l1_loss(
                        prediction, targets, beta=recipe.smooth_l1_beta
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError("Transfer MLP produced a non-finite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), recipe.max_grad_norm,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                scheduler.step()
                loss_sum += float(loss.detach().cpu()) * len(indices)
            valid_prediction = _predict(
                model, valid_features, batch_size=recipe.batch_size,
                device=device, amp=amp,
            )
            valid_nmae = float(
                (valid_prediction - valid.targets.float()).abs().mean()
            )
            train_loss = loss_sum / len(train)
            history.append({
                "epoch": epoch,
                "train_normalized_smooth_l1": train_loss,
                "valid_normalized_mae": valid_nmae,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            })
            progress.set_postfix(
                train=f"{train_loss:.4f}", valid=f"{valid_nmae:.4f}"
            )
            progress.update(1)
    finally:
        progress.close()
    final_prediction = _predict(
        model, valid_features, batch_size=recipe.batch_size, device=device, amp=amp
    )
    target_stats = prepared["normalization"][f"fold{fold}"][task_id]["target"]
    raw_prediction = (
        final_prediction * float(target_stats["scale"]) + float(target_stats["mean"])
    )
    raw_mae = float((raw_prediction - valid.raw_targets.float()).abs().mean())
    row_hash = tensor_state_hash(
        "stage2-stage3-transfer-validation-rows.v1",
        {
            "source_rows": valid.source_rows.long(),
            "targets": valid.raw_targets.float(),
        },
    )
    identity = semantic_identity(
        "stage2-stage3.transfer-mlp-training",
        {
            "contract_version": 1,
            "variant": variant,
            "representation_identity": bank["identity"]["hash"],
            "stage3_prepared_identity": prepared_identity["hash"],
            "task": task_id,
            "fold": fold,
            "seed": training_seed,
            "model": {"hidden_dims": list(recipe.hidden_dims), "dropout": recipe.dropout},
            "training": {
                "epochs": recipe.epochs, "batch_size": recipe.batch_size,
                "learning_rate": recipe.learning_rate,
                "weight_decay": recipe.weight_decay, "betas": list(recipe.betas),
                "eps": recipe.eps, "smooth_l1_beta": recipe.smooth_l1_beta,
                "warmup_fraction": recipe.warmup_fraction,
                "min_lr_ratio": recipe.min_lr_ratio,
                "max_grad_norm": recipe.max_grad_norm,
                "selection": recipe.selection,
            },
        },
    )
    root.mkdir(parents=True, exist_ok=False)
    model_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    artifact = root / "final.pt"
    atomic_torch_save(artifact, {
        "kind": MODEL_KIND, "format_version": MODEL_VERSION,
        "identity": identity, "model": model_state,
        "model_state_hash": tensor_state_hash("stage2-stage3-transfer-mlp-final.v1", model_state),
        "initial_state_hash": initial_state_hash,
        "final_epoch": recipe.epochs,
    })
    with (root / "metrics.jsonl").open("x", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (root / "validation_predictions.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_row", "target", "prediction"])
        writer.writeheader()
        for source_row, target, prediction in zip(
            valid.source_rows.tolist(), valid.raw_targets.tolist(), raw_prediction.tolist(), strict=True
        ):
            writer.writerow({"source_row": int(source_row), "target": float(target), "prediction": float(prediction)})
    manifest = {
        "kind": MODEL_KIND, "format_version": MODEL_VERSION,
        "identity": identity, "variant": variant, "task": task_id, "fold": fold,
        "representation_identity": bank["identity"],
        "initial_state_hash": initial_state_hash,
        "permutation_hashes": permutation_hashes,
        "row_target_hash": row_hash,
        "final_epoch": recipe.epochs,
        "validation_raw_mae": raw_mae,
        "validation_rows": len(valid),
        "artifact": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "predictions_sha256": sha256_file(root / "validation_predictions.csv"),
    }
    atomic_json(root / "manifest.json", manifest)
    return manifest


__all__ = [
    "MODEL_KIND", "REPRESENTATION_KIND", "build_transfer_features",
    "load_representation_bank", "prepare_representation_bank",
    "train_transfer_job", "transfer_training_seed",
]
