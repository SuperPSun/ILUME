from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from common.training import resolve_device, seed_everything
from stage2 import load_frozen_object_encoder
from stage3.config import Stage3Config, effective_training_seed
from stage3.data import Stage3TaskDataset
from stage3.prepare import load_prepared_stage3
from stage3.three_phase import run_three_phase_training
from stage3.train import build_resolved_training_plan

from .representation import (
    FinetuneRecipe, FinetuneStage3Model, LiveRepresentationStore,
    STAGE1_OWNER, STAGE2_OWNER, load_features, object_keys,
)


def _encoder_owner_recipe(lr: float, epochs: int, steps: int, warmup: float, floor: float) -> dict[str, Any]:
    return {
        "nominal_lr": lr,
        "terminal_lr": lr * floor,
        "nominal_epochs": epochs,
        "effective_epochs": epochs,
        "freeze_epoch": epochs,
        "updates_per_epoch": steps,
        "actual_update_budget": epochs * steps,
        "warmup_updates": math.ceil(warmup * epochs * steps),
        "capacity": {"kind": "pretrained_representation_encoder"},
    }


def build_model_and_store(
    config: Stage3Config, prepared: Mapping[str, Any],
    features: Mapping[str, Any], *, fold: int, device: torch.device,
) -> tuple[FinetuneStage3Model, LiveRepresentationStore]:
    encoder_path = config.initialization.stage2_encoder
    assert encoder_path is not None
    encoder = load_frozen_object_encoder(encoder_path, device="cpu")
    if not hasattr(encoder, "backbone"):
        raise ValueError("Fine-tuning requires the v2 Stage 1/ObjectEncoder artifact")
    seed_everything(effective_training_seed(config) + fold)
    model = FinetuneStage3Model(
        config.model, prepared["registry"], encoder.embedding_dim,
        group_configs=config.groups,
        task_configs=config.tasks,
        task_private_recipes={
            task: config.resolved_private_recipe(task)
            for task, item in config.tasks.items() if item.enabled
        },
        backbone=encoder.backbone,
        object_encoder=encoder.object_encoder,
    ).to(device)
    model.set_trainable_owners(tuple(set(model.parameter_ownership().values())))
    store = LiveRepresentationStore(
        model, encoder.packer, object_keys(prepared), features["samples"]
    )
    return model, store


@torch.no_grad()
def validate_initial_representation(
    model: FinetuneStage3Model, store: LiveRepresentationStore,
    prepared: Mapping[str, Any],
) -> float:
    reference = prepared["objects"]["embeddings"]
    model.eval()
    largest = 0.0
    for topology in ("il", "molecule"):
        indices = [index for index, key in enumerate(store.keys) if key.topology == topology]
        if not indices:
            continue
        stride = max(1, len(indices) // 16)
        selected = indices[::stride][:16]
        observed = store.values(torch.tensor(selected), topology).cpu()
        expected = reference[selected]
        largest = max(largest, float((observed - expected).abs().max()))
        if not torch.allclose(observed, expected, rtol=1e-3, atol=1e-3):
            raise ValueError(
                "Live Stage 1/ObjectEncoder initialization differs from Base prepared embeddings"
            )
    model.set_trainable_owners(tuple(set(model.parameter_ownership().values())))
    return largest


def resolved_plan(
    config: Stage3Config, recipe: FinetuneRecipe, fold: int,
    model: FinetuneStage3Model, prepared: Mapping[str, Any],
    train_data: Mapping[str, Stage3TaskDataset],
    features: Mapping[str, Any],
) -> dict[str, Any]:
    active = tuple(task for task, spec in prepared["registry"].items() if spec.enabled)
    normalizations = prepared["normalization"][f"fold{fold}"]
    plan = build_resolved_training_plan(
        config, fold, model, train_data, active, prepared,
        {"mode": "scratch", "loaded_parameters": []}, normalizations,
    )
    phase = plan["phases"]["phase1"]
    for owner, lr in ((STAGE1_OWNER, recipe.stage1_lr), (STAGE2_OWNER, recipe.stage2_lr)):
        phase["owners"][owner.label] = _encoder_owner_recipe(
            lr, recipe.epochs, int(phase["steps_per_epoch"]),
            recipe.warmup_ratio, recipe.min_lr_ratio,
        )
    plan["encoder_finetune"] = {
        "contract_version": 1,
        "recipe": recipe.to_dict(),
        "feature_artifact_sha256": features["artifact_sha256"],
        "source_encoder_sha256": features["stage2_encoder_sha256"],
        "source_encoder_identity": features["stage2_encoder_identity"],
        "phase1_only": True,
        "encoding": "live_stage1_objectencoder_raw_v1",
    }
    plan["format_version"] = 6
    return plan


def run_finetuning(
    config: Stage3Config, recipe: FinetuneRecipe, *, fold: int,
    feature_dir: str | Path, output_dir: str | Path, resume: bool = False,
) -> list[dict[str, Any]]:
    if fold not in range(1, 6):
        raise ValueError("Fine-tuning fold must be 1..5")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Fine-tuning supports one process per fold")
    device = resolve_device(config.training.device)
    if config.training.amp_dtype == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Fine-tuning requires a BF16-capable CUDA GPU")
    prepared = load_prepared_stage3(config)
    features = load_features(config, feature_dir, prepared)
    model, store = build_model_and_store(config, prepared, features, fold=fold, device=device)
    validate_initial_representation(model, store, prepared)
    active = tuple(task for task, spec in prepared["registry"].items() if spec.enabled)
    train_data = {
        task: Stage3TaskDataset(config.data.artifacts_dir, fold, task, "train")
        for task in active
    }
    valid_data = {
        task: Stage3TaskDataset(config.data.artifacts_dir, fold, task, "valid")
        for task in active
    }
    normalizations = prepared["normalization"][f"fold{fold}"]
    plan = resolved_plan(config, recipe, fold, model, prepared, train_data, features)
    root = Path(output_dir)
    if root.exists() and not resume:
        raise FileExistsError(f"Fine-tuning output already exists: {root}")
    if resume and not root.exists():
        raise FileNotFoundError(f"Fine-tuning resume directory is missing: {root}")
    return run_three_phase_training(
        config=config, fold=fold, output_dir=root,
        resume_from=root if resume else None,
        model=model, registry=prepared["registry"], active=active,
        train_data=train_data, valid_data=valid_data,
        representations=store, normalizations=normalizations,
        plan=plan, device=device,
    )
