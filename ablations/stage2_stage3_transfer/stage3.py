from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ablations.stage3_single_task_mlp.model import Stage3SingleTaskMLP
from common.identity import semantic_identity, tensor_state_hash
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.progress import ProgressReporter
from common.training import canonical_json_sha256, resolve_device, seed_everything
from stage2.frozen import FrozenObjectSpec, load_frozen_object_encoder
from stage2.model import ObjectEncoder
from stage3.config import load_stage3_config
from stage3.data import ObjectKey, Stage3TaskDataset, stable_seed
from stage3.identity import metadata_identity
from stage3.prepare import load_prepared_stage3

from .config import TransferExperimentConfig, require_full_transfer_artifact


REPRESENTATION_KIND = "ilume_stage2_stage3_transfer_trainable_object_encoder"
REPRESENTATION_VERSION = 2
MODEL_KIND = "ilume_stage2_stage3_transfer_joint_object_encoder_mlp"
MODEL_VERSION = 2


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


def _object_encoder_contract(object_encoder: ObjectEncoder) -> dict[str, Any]:
    layers = object_encoder.encoder.layers
    if not layers:
        raise ValueError("Transfer ObjectEncoder must contain at least one layer")
    first = layers[0]
    return {
        "d_model": int(object_encoder.d_model),
        "n_heads": int(first.self_attn.num_heads),
        "layers": len(layers),
        "ffn_dim": int(first.linear1.out_features),
        "dropout": float(first.dropout.p),
    }


def _materialize_entity_slots(
    encoder: Any,
    keys: tuple[ObjectKey, ...],
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d_model = int(encoder.embedding_dim)
    entity_slots = torch.zeros((len(keys), 2, d_model), dtype=torch.float32)
    entity_roles = torch.zeros((len(keys), 2), dtype=torch.long)
    slot_counts = torch.tensor([len(key.slots) for key in keys], dtype=torch.long)
    for slot_count in (1, 2):
        indices = torch.nonzero(slot_counts == slot_count, as_tuple=False).flatten()
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            specs = [
                FrozenObjectSpec(keys[index].topology, keys[index].slots)
                for index in selected.tolist()
            ]
            slots, roles = encoder.encode_slots(specs)
            entity_slots[selected, :slot_count] = slots
            entity_roles[selected, :slot_count] = roles
    if not torch.isfinite(entity_slots).all():
        raise RuntimeError("Transfer entity-slot bank contains non-finite values")
    return entity_slots, entity_roles, slot_counts


def prepare_representation_bank(
    experiment: TransferExperimentConfig,
    *,
    variant: str,
    encoder_path: str | Path,
    encoder_manifest_path: str | Path,
    destination: str | Path,
    expected_initial_shared_state_hash: str,
) -> dict[str, Any]:
    encoder_path = Path(encoder_path)
    encoder_manifest_path = Path(encoder_manifest_path)
    encoder_manifest = json.loads(encoder_manifest_path.read_text(encoding="utf-8"))
    require_full_transfer_artifact(encoder_manifest)
    authority = load_stage3_config(experiment.stage3.authority_config)
    prepared = load_prepared_stage3(authority)
    keys = _object_keys(prepared["objects"])
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

    device = resolve_device(authority.training.device)
    encoder = load_frozen_object_encoder(encoder_path, device=device)
    entity_slots, entity_roles, slot_counts = _materialize_entity_slots(
        encoder,
        keys,
        batch_size=authority.preparation.encoding_batch_size,
    )
    object_encoder_state = {
        name: value.detach().cpu()
        for name, value in encoder.object_encoder.state_dict().items()
    }
    object_encoder_contract = _object_encoder_contract(encoder.object_encoder)
    object_encoder_state_hash = tensor_state_hash(
        "stage2-stage3-transfer-object-encoder-initial.v2",
        object_encoder_state,
    )
    slot_hash = tensor_state_hash(
        "stage2-stage3-transfer-entity-slots.v2",
        {
            "entity_slots": entity_slots,
            "entity_roles": entity_roles,
            "slot_counts": slot_counts,
        },
    )
    object_list_hash = canonical_json_sha256([key.to_dict() for key in keys])
    prepared_identity = dict(
        metadata_identity(
            prepared["metadata"], "prepared", context="Stage 3 transfer authority"
        )
    )
    identity = semantic_identity(
        "stage2-stage3.transfer-trainable-object-encoder",
        {
            "contract_version": REPRESENTATION_VERSION,
            "variant": variant,
            "stage3_prepared_identity": prepared_identity["hash"],
            "stage2_encoder_identity": encoder.encoder_identity["hash"],
            "encoder_artifact_sha256": sha256_file(encoder_path),
            "object_list_hash": object_list_hash,
            "slot_hash": slot_hash,
            "object_encoder_state_hash": object_encoder_state_hash,
            "object_encoder_contract": object_encoder_contract,
            "shape": list(entity_slots.shape),
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
            "stage2_encoder_identity": encoder.encoder_identity,
            "objects": [key.to_dict() for key in keys],
            "object_list_hash": object_list_hash,
            "slot_hash": slot_hash,
            "entity_slots": entity_slots.contiguous(),
            "entity_roles": entity_roles.contiguous(),
            "slot_counts": slot_counts.contiguous(),
            "object_encoder_contract": object_encoder_contract,
            "object_encoder_state": object_encoder_state,
            "object_encoder_state_hash": object_encoder_state_hash,
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
        "stage2_encoder_identity": encoder.encoder_identity,
        "object_list_hash": object_list_hash,
        "slot_hash": slot_hash,
        "object_encoder_state_hash": object_encoder_state_hash,
        "object_encoder_contract": object_encoder_contract,
        "shape": list(entity_slots.shape),
    }
    atomic_json(destination.with_suffix(".json"), manifest)
    return manifest


def load_representation_bank(
    path: str | Path, *, expected_prepared_identity: Mapping[str, Any]
) -> dict[str, Any]:
    path = Path(path)
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    require_full_transfer_artifact(manifest)
    if manifest.get("artifact_sha256") != sha256_file(path):
        raise ValueError("Transfer representation artifact hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require_full_transfer_artifact(payload)
    if payload.get("kind") != REPRESENTATION_KIND or payload.get("format_version") != REPRESENTATION_VERSION:
        raise ValueError("Unsupported transfer representation artifact; rerun transfer prepare")
    if payload.get("stage3_prepared_identity") != dict(expected_prepared_identity):
        raise ValueError("Transfer representation uses a different Stage 3 authority")
    slots = payload.get("entity_slots")
    roles = payload.get("entity_roles")
    counts = payload.get("slot_counts")
    if (
        not isinstance(slots, torch.Tensor)
        or slots.ndim != 3
        or slots.shape[1:] != (2, 1024)
        or not isinstance(roles, torch.Tensor)
        or roles.shape != slots.shape[:2]
        or roles.dtype != torch.long
        or not isinstance(counts, torch.Tensor)
        or counts.shape != slots.shape[:1]
        or counts.dtype != torch.long
        or not bool(torch.isin(counts, torch.tensor((1, 2))).all())
        or not torch.isfinite(slots).all()
    ):
        raise ValueError("Malformed transfer entity-slot bank")
    if payload.get("slot_hash") != tensor_state_hash(
        "stage2-stage3-transfer-entity-slots.v2",
        {"entity_slots": slots, "entity_roles": roles, "slot_counts": counts},
    ):
        raise ValueError("Transfer entity-slot hash mismatch")
    state = payload.get("object_encoder_state")
    if not isinstance(state, dict) or payload.get("object_encoder_state_hash") != tensor_state_hash(
        "stage2-stage3-transfer-object-encoder-initial.v2", state
    ):
        raise ValueError("Transfer ObjectEncoder state hash mismatch")
    return payload


def _build_object_encoder(bank: Mapping[str, Any], device: torch.device) -> ObjectEncoder:
    contract = bank["object_encoder_contract"]
    object_encoder = ObjectEncoder(
        int(contract["d_model"]),
        int(contract["n_heads"]),
        num_layers=int(contract["layers"]),
        feedforward_dim=int(contract["ffn_dim"]),
        dropout=float(contract["dropout"]),
    )
    object_encoder.load_state_dict(bank["object_encoder_state"], strict=True)
    return object_encoder.to(device)


def encode_transfer_objects(
    object_encoder: ObjectEncoder,
    bank: Mapping[str, Any],
    object_ids: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    resolved_ids = object_ids.long().cpu()
    if len(resolved_ids) == 0:
        return torch.empty((0, object_encoder.d_model), device=device)
    if bool((resolved_ids < 0).any()) or bool(
        (resolved_ids >= len(bank["slot_counts"])).any()
    ):
        raise ValueError("Transfer object ID is outside the entity-slot bank")
    encoded = torch.zeros(
        (len(resolved_ids), object_encoder.d_model), device=device
    )
    counts = bank["slot_counts"][resolved_ids]
    for slot_count in (1, 2):
        positions = torch.nonzero(counts == slot_count, as_tuple=False).flatten()
        if len(positions) == 0:
            continue
        selected = resolved_ids[positions]
        slots = bank["entity_slots"][selected, :slot_count].to(device)
        roles = bank["entity_roles"][selected, :slot_count].to(device)
        values = object_encoder(slots, roles)
        encoded = encoded.index_copy(0, positions.to(device), values)
    return encoded


def _batch_features(
    dataset: Stage3TaskDataset,
    indices: torch.Tensor,
    bank: Mapping[str, Any],
    object_encoder: ObjectEncoder,
    *,
    has_partner: bool,
    device: torch.device,
) -> torch.Tensor:
    parts = [
        encode_transfer_objects(
            object_encoder,
            bank,
            dataset.primary_object_ids[indices],
            device=device,
        )
    ]
    partner_ids = dataset.partner_object_ids[indices]
    if has_partner:
        if bool((partner_ids < 0).any()):
            raise ValueError("Transfer interaction row is missing a partner")
        parts.append(
            encode_transfer_objects(
                object_encoder, bank, partner_ids, device=device
            )
        )
    elif len(indices) and bool((partner_ids != -1).any()):
        raise ValueError("Transfer non-interaction row has an unexpected partner")
    parts.append(dataset.conditions[indices].float().to(device))
    result = torch.cat(parts, dim=1)
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
    model: Stage3SingleTaskMLP,
    object_encoder: ObjectEncoder,
    dataset: Stage3TaskDataset,
    bank: Mapping[str, Any],
    *,
    has_partner: bool,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    model.eval()
    object_encoder.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, len(dataset), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(dataset)))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            features = _batch_features(
                dataset,
                indices,
                bank,
                object_encoder,
                has_partner=has_partner,
                device=device,
            )
            value = model(features).squeeze(-1)
        chunks.append(value.float().cpu())
    return torch.cat(chunks) if chunks else torch.empty(0)


def _optimizer_groups(
    object_encoder: ObjectEncoder,
    model: Stage3SingleTaskMLP,
    *,
    object_encoder_lr: float,
    model_lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    return [
        {
            "params": [p for p in object_encoder.parameters() if p.ndim >= 2],
            "lr": object_encoder_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for p in object_encoder.parameters() if p.ndim < 2],
            "lr": object_encoder_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for p in model.parameters() if p.ndim >= 2],
            "lr": model_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for p in model.parameters() if p.ndim < 2],
            "lr": model_lr,
            "weight_decay": 0.0,
        },
    ]


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
    prepared_identity = dict(
        metadata_identity(
            prepared["metadata"], "prepared", context="Stage 3 transfer authority"
        )
    )
    bank = load_representation_bank(
        representation_path, expected_prepared_identity=prepared_identity
    )
    require_full_transfer_artifact(bank)
    if bank.get("variant") != variant:
        raise ValueError("Transfer representation variant mismatch")
    if bank["objects"] != prepared["objects"]["objects"]:
        raise ValueError("Transfer representation object order mismatch")
    spec = prepared["registry"][task_id]
    has_partner = bool(spec.partner_slots)
    train = Stage3TaskDataset(authority.data.artifacts_dir, fold, task_id, "train")
    valid = Stage3TaskDataset(authority.data.artifacts_dir, fold, task_id, "valid")
    recipe = experiment.stage3
    training_seed = transfer_training_seed(experiment.seed, task_id, fold)
    seed_everything(training_seed)
    device = resolve_device(device_name or "auto")
    input_dim = int(bank["object_encoder_contract"]["d_model"]) * (
        2 if has_partner else 1
    ) + int(train.conditions.shape[1])
    model = Stage3SingleTaskMLP(
        input_dim, recipe.hidden_dims, recipe.dropout
    ).to(device)
    initial_state_hash = tensor_state_hash(
        "stage2-stage3-transfer-mlp-initial.v2",
        {name: value.detach().cpu() for name, value in model.state_dict().items()},
    )
    object_encoder = _build_object_encoder(bank, device)
    initial_object_encoder_state_hash = tensor_state_hash(
        "stage2-stage3-transfer-object-encoder-initial.v2",
        {
            name: value.detach().cpu()
            for name, value in object_encoder.state_dict().items()
        },
    )
    if initial_object_encoder_state_hash != bank["object_encoder_state_hash"]:
        raise ValueError("Transfer ObjectEncoder initialization drifted from its bank")
    optimizer = torch.optim.AdamW(
        _optimizer_groups(
            object_encoder,
            model,
            object_encoder_lr=experiment.stage2.object_encoder_learning_rate,
            model_lr=recipe.learning_rate,
            weight_decay=recipe.weight_decay,
        ),
        betas=recipe.betas,
        eps=recipe.eps,
        foreach=False,
        fused=False,
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
    parameters = [*object_encoder.parameters(), *model.parameters()]
    try:
        for epoch in range(1, recipe.epochs + 1):
            object_encoder.train()
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
                targets = train.targets[indices].float().to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=amp
                ):
                    inputs = _batch_features(
                        train,
                        indices,
                        bank,
                        object_encoder,
                        has_partner=has_partner,
                        device=device,
                    )
                    prediction = model(inputs).squeeze(-1)
                    loss = F.smooth_l1_loss(
                        prediction, targets, beta=recipe.smooth_l1_beta
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError("Transfer joint model produced a non-finite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters, recipe.max_grad_norm, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                loss_sum += float(loss.detach().cpu()) * len(indices)
            valid_prediction = _predict(
                model,
                object_encoder,
                valid,
                bank,
                has_partner=has_partner,
                batch_size=recipe.batch_size,
                device=device,
                amp=amp,
            )
            valid_nmae = float(
                (valid_prediction - valid.targets.float()).abs().mean()
            )
            train_loss = loss_sum / len(train)
            history.append(
                {
                    "epoch": epoch,
                    "train_normalized_smooth_l1": train_loss,
                    "valid_normalized_mae": valid_nmae,
                    "object_encoder_learning_rate": float(
                        optimizer.param_groups[0]["lr"]
                    ),
                    "mlp_learning_rate": float(optimizer.param_groups[2]["lr"]),
                }
            )
            progress.set_postfix(
                train=f"{train_loss:.4f}", valid=f"{valid_nmae:.4f}"
            )
            progress.update(1)
    finally:
        progress.close()
    final_prediction = _predict(
        model,
        object_encoder,
        valid,
        bank,
        has_partner=has_partner,
        batch_size=recipe.batch_size,
        device=device,
        amp=amp,
    )
    target_stats = prepared["normalization"][f"fold{fold}"][task_id]["target"]
    raw_prediction = (
        final_prediction * float(target_stats["scale"])
        + float(target_stats["mean"])
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
        "stage2-stage3.transfer-joint-object-encoder-mlp-training",
        {
            "contract_version": MODEL_VERSION,
            "variant": variant,
            "representation_identity": bank["identity"]["hash"],
            "stage3_prepared_identity": prepared_identity["hash"],
            "task": task_id,
            "fold": fold,
            "seed": training_seed,
            "object_encoder": {
                **bank["object_encoder_contract"],
                "learning_rate": experiment.stage2.object_encoder_learning_rate,
            },
            "model": {
                "hidden_dims": list(recipe.hidden_dims),
                "dropout": recipe.dropout,
            },
            "training": {
                "epochs": recipe.epochs,
                "batch_size": recipe.batch_size,
                "learning_rate": recipe.learning_rate,
                "weight_decay": recipe.weight_decay,
                "betas": list(recipe.betas),
                "eps": recipe.eps,
                "smooth_l1_beta": recipe.smooth_l1_beta,
                "warmup_fraction": recipe.warmup_fraction,
                "min_lr_ratio": recipe.min_lr_ratio,
                "max_grad_norm": recipe.max_grad_norm,
                "selection": recipe.selection,
            },
        },
    )
    root.mkdir(parents=True, exist_ok=False)
    model_state = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    object_encoder_state = {
        name: value.detach().cpu()
        for name, value in object_encoder.state_dict().items()
    }
    artifact = root / "final.pt"
    model_state_hash = tensor_state_hash(
        "stage2-stage3-transfer-joint-final.v2",
        {
            **{f"object_encoder.{name}": value for name, value in object_encoder_state.items()},
            **{f"mlp.{name}": value for name, value in model_state.items()},
        },
    )
    atomic_torch_save(
        artifact,
        {
            "kind": MODEL_KIND,
            "format_version": MODEL_VERSION,
            "identity": identity,
            "object_encoder": object_encoder_state,
            "model": model_state,
            "model_state_hash": model_state_hash,
            "initial_state_hash": initial_state_hash,
            "initial_object_encoder_state_hash": initial_object_encoder_state_hash,
            "final_epoch": recipe.epochs,
        },
    )
    with (root / "metrics.jsonl").open("x", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (root / "validation_predictions.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["source_row", "target", "prediction"]
        )
        writer.writeheader()
        for source_row, target, prediction in zip(
            valid.source_rows.tolist(),
            valid.raw_targets.tolist(),
            raw_prediction.tolist(),
            strict=True,
        ):
            writer.writerow(
                {
                    "source_row": int(source_row),
                    "target": float(target),
                    "prediction": float(prediction),
                }
            )
    manifest = {
        "kind": MODEL_KIND,
        "format_version": MODEL_VERSION,
        "identity": identity,
        "variant": variant,
        "task": task_id,
        "fold": fold,
        "downstream_training": "joint_object_encoder_mlp",
        "representation_identity": bank["identity"],
        "initial_state_hash": initial_state_hash,
        "initial_object_encoder_state_hash": initial_object_encoder_state_hash,
        "final_model_state_hash": model_state_hash,
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
    "MODEL_KIND",
    "MODEL_VERSION",
    "REPRESENTATION_KIND",
    "REPRESENTATION_VERSION",
    "encode_transfer_objects",
    "load_representation_bank",
    "prepare_representation_bank",
    "train_transfer_job",
    "transfer_training_seed",
]
