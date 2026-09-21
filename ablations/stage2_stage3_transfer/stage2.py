from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from common.identity import semantic_identity, tensor_state_hash
from common.io import atomic_json, sha256_file
from common.training import seed_everything
from stage1.model import load_stage1_model
from stage2.config import Stage2Config, load_stage2_config
from stage2.data import load_artifact_registry
from stage2.identity import metadata_identity
from stage2.model import Stage2ObjectModel
from stage2.train import export_stage2_encoder_artifact, run_stage2_training

from .config import TransferExperimentConfig
from .sampling import experiment_contract, resolve_balanced_rows


TRANSFER_ENCODER_MANIFEST_KIND = "ilume_stage2_transfer_encoder"
TRANSFER_ENCODER_MANIFEST_VERSION = 1


def physics_only_stage2_config(config: Stage2Config) -> Stage2Config:
    return replace(
        config,
        loss=replace(config.loss, lambda_teacher=0.0),
        training=replace(
            config.training,
            refinement_epochs=0,
            refinement_tasks=(),
        ),
    )


def _build_initialized_model(config: Stage2Config) -> tuple[Stage2ObjectModel, Any, Any]:
    seed_everything(config.data.seed)
    loaded = load_stage1_model(
        config.initialization.checkpoint,
        config.data.pretrain_artifacts_dir,
        device="cpu",
        backbone_dropout=0.0,
    )
    registry = load_artifact_registry(config.data.artifacts_dir)
    config.validate_registry(registry)
    model = Stage2ObjectModel(
        loaded.model,
        registry,
        object_layers=config.model.object_layers,
        object_ffn_dim=config.model.object_ffn_dim,
        dropout=config.model.dropout,
    )
    return model, registry, loaded


def shared_initial_state_hash(model: Stage2ObjectModel) -> str:
    state = {
        **{
            f"backbone.{name}": value.detach().cpu()
            for name, value in model.backbone.state_dict().items()
        },
        **{
            f"object_encoder.{name}": value.detach().cpu()
            for name, value in model.object_encoder.state_dict().items()
        },
    }
    return tensor_state_hash("stage2.transfer-initial-shared-state.v1", state)


def _manifest(
    *, variant: str, source_task: str | None, initial_hash: str,
    encoder_path: Path, updates: int, training_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "kind": TRANSFER_ENCODER_MANIFEST_KIND,
        "format_version": TRANSFER_ENCODER_MANIFEST_VERSION,
        "variant": variant,
        "source_task": source_task,
        "physics_only": source_task is not None,
        "initial_shared_state_hash": initial_hash,
        "optimizer_updates": updates,
        "training_identity": training_identity,
        "encoder": encoder_path.name,
        "encoder_sha256": sha256_file(encoder_path),
        "encoder_artifact": torch.load(
            encoder_path, map_location="cpu", weights_only=False
        )["semantic_identity"],
    }


def create_baseline_encoder(
    experiment: TransferExperimentConfig, output_dir: str | Path
) -> dict[str, Any]:
    config = physics_only_stage2_config(
        load_stage2_config(experiment.stage2.authority_config)
    )
    model, registry, _ = _build_initialized_model(config)
    initial_hash = shared_initial_state_hash(model)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(
        (config.data.artifacts_dir / "metadata.json").read_text(encoding="utf-8")
    )
    data_identity = dict(metadata_identity(metadata, "data", context="Stage 2 data artifact"))
    identity = semantic_identity(
        "stage2.transfer-baseline",
        {
            "contract_version": 1,
            "zero_optimizer_steps": True,
            "initial_shared_state_hash": initial_hash,
            "stage2_data_identity": data_identity["hash"],
            "seed": experiment.seed,
        },
    )
    encoder_path = root / "stage2_encoder.pt"
    export_stage2_encoder_artifact(
        encoder_path,
        model=model,
        config=config,
        registry=registry,
        data_identity=data_identity,
        provenance={
            "transfer_variant": "baseline",
            "zero_optimizer_steps": True,
            "initial_shared_state_hash": initial_hash,
            "transfer_identity": identity["hash"],
        },
    )
    manifest = _manifest(
        variant="baseline", source_task=None, initial_hash=initial_hash,
        encoder_path=encoder_path, updates=0, training_identity=identity,
    )
    manifest.update(experiment_contract(experiment))
    atomic_json(root / "manifest.json", manifest)
    return manifest


def train_source_encoder(
    experiment: TransferExperimentConfig,
    source_task: str,
    output_dir: str | Path,
    *,
    initial_shared_state_hash: str,
    resume_from: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    if source_task not in experiment.stage2.sources:
        raise ValueError(f"Source task is not configured: {source_task}")
    config = physics_only_stage2_config(
        load_stage2_config(experiment.stage2.authority_config)
    )
    if device is not None:
        config = replace(config, training=replace(config.training, device=device))
    selection = {}
    plan = None
    if experiment.stage2.sampling_mode == "balanced_rows":
        selections, plan = resolve_balanced_rows(experiment)
        selection = {"source_row_indices": selections[source_task]}
    run_stage2_training(
        config,
        output_dir=output_dir,
        resume_from=resume_from,
        active_task=source_task,
        expected_initial_shared_state_hash=initial_shared_state_hash,
        **selection,
    )
    root = Path(output_dir)
    final = json.loads((root / "final_metrics.json").read_text(encoding="utf-8"))
    if final.get("final_epoch") != experiment.stage2.final_epoch:
        raise ValueError("Stage 2 transfer source did not publish epoch 10")
    encoder_path = root / "stage2_encoder.pt"
    manifest = _manifest(
        variant="source", source_task=source_task,
        initial_hash=initial_shared_state_hash, encoder_path=encoder_path,
        updates=int(final["final_validation"]["global_optimizer_step"]),
        training_identity=final.get("training_identity"),
    )
    if plan is not None:
        if manifest["optimizer_updates"] != plan["optimizer_updates"]:
            raise ValueError("Balanced transfer optimizer budget mismatch")
        payload = (manifest.get("training_identity") or {}).get("payload", {})
        if payload.get("selection_hash") != plan["sources"][source_task]["selection_hash"]:
            raise ValueError("Balanced transfer trained row selection mismatch")
        manifest["sampling_plan"] = plan
    manifest.update(experiment_contract(experiment))
    atomic_json(root / "manifest.json", manifest)
    return manifest


__all__ = [
    "TRANSFER_ENCODER_MANIFEST_KIND", "create_baseline_encoder",
    "physics_only_stage2_config", "shared_initial_state_hash",
    "train_source_encoder",
]
