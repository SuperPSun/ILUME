from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from common.identity import require_compatible_identity, semantic_identity, tensor_state_hash
from common.io import sha256_file
from common.progress import ProgressReporter
from common.training import resolve_device
from stage3.config import Stage3Config, stage3_config_from_dict
from stage3.data import Stage3TaskDataset
from stage3.evaluate import evaluate_checkpoints
from stage3.identity import build_stage3_training_identity
from stage3.prepare import load_prepared_stage3, prepare_stage3
from stage3.three_phase import run_three_phase_training

from ablations.stage3_full_finetune.representation import (
    FinetuneRecipe, encoder_state_hashes, load_features, prepare_features,
)
from ablations.stage3_full_finetune.train import (
    build_model_and_store as build_live_model, resolved_plan as live_plan,
    validate_initial_representation,
)
from .config import Experiment, load_experiment
from .contract import load_transferable_state
from .stage3 import load_source, require_paired_data


FINAL_KIND = "ilume_stage3_home_transfer_full_finetune_three_phase_final"
PLAN_VERSION = 9


@dataclass(frozen=True)
class FinetuneExperiment:
    source: Experiment
    stage3: Stage3Config
    recipe: FinetuneRecipe
    output_root: Path

    @property
    def feature_dir(self) -> Path:
        return self.output_root / "features"

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_root / "train"


def load_config(
    path: str | Path, *, source_dir: str | Path,
    output_root: str | Path | None = None,
) -> FinetuneExperiment:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw = dict(raw)
    transfer = raw.pop("home_transfer")
    if set(transfer) != {"experiment_config"}:
        raise ValueError("home_transfer requires only experiment_config")
    default_root = raw.pop("output_root")
    root = Path(default_root if output_root is None else output_root)
    recipe_raw = raw.pop("encoder_finetune")
    if set(recipe_raw) != set(FinetuneRecipe.__dataclass_fields__):
        raise ValueError("Encoder fine-tuning recipe has missing or unknown fields")
    recipe = FinetuneRecipe(**recipe_raw)
    config = stage3_config_from_dict(raw)
    source = replace(load_experiment(transfer["experiment_config"]), output_root=Path(source_dir))
    if root.resolve() == source.output_root.resolve() or source.output_root.resolve() in root.resolve().parents:
        raise ValueError("Fine-tuning output must be isolated from the source experiment")
    expected = source.stage3.to_dict()
    expected["training"].pop("object_encoder_phase1")
    expected["training"]["microbatch_size"] = 8
    if config.to_dict() != expected:
        raise ValueError("HoME full fine-tuning must preserve the source Stage3 recipe except microbatch/encoder owners")
    recipe.validate(config)
    if recipe.to_dict() != FinetuneRecipe(5e-6, 1.5e-5, 15, 0.05, 0.1).to_dict():
        raise ValueError("HoME full fine-tuning requires the fixed Phase 1 encoder recipe")
    config = replace(
        config,
        data=replace(config.data, artifacts_dir=root / "prepare" / "artifacts"),
        preparation=replace(config.preparation, cache_dir=root / "prepare" / "object_cache"),
        initialization=replace(config.initialization, stage2_encoder=source.output_root / "stage2" / "stage2_encoder.pt"),
    )
    return FinetuneExperiment(source, config, recipe, root)


def provenance(experiment: FinetuneExperiment, source: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {
        "source_training_identity": source["training_identity"]["hash"],
        "source_artifact_sha256": sha256_file(experiment.source.output_root / "stage2" / "stage2_home_transfer.pt"),
        "source_encoder_sha256": source["stage2_encoder_sha256"],
        "transferred_state_hash": source["shared_state_hash"],
        "transferred_parameters": list(names),
    }


def prepare(experiment: FinetuneExperiment) -> dict[str, Any]:
    load_source(experiment.source)
    if experiment.output_root.exists():
        raise FileExistsError(f"Fine-tuning output already exists: {experiment.output_root}")
    prepared = prepare_stage3(experiment.stage3)
    require_paired_data(experiment.source, load_prepared_stage3(experiment.stage3), experiment.stage3)
    features = prepare_features(experiment.stage3, experiment.feature_dir)
    return {"prepared": prepared, "features": features}


def build_model_and_store(
    experiment: FinetuneExperiment, prepared: Mapping[str, Any],
    features: Mapping[str, Any], *, fold: int, device: torch.device,
) -> tuple[Any, Any, tuple[str, ...], dict[str, Any]]:
    source = load_source(experiment.source)
    model, store = build_live_model(experiment.stage3, prepared, features, fold=fold, device=device)
    names = load_transferable_state(model, source["shared_state"], source["shared_state_hash"])
    return model, store, names, source


def resolved_plan(
    experiment: FinetuneExperiment, prepared: Mapping[str, Any], features: Mapping[str, Any],
    model: Any, train_data: Mapping[str, Any], *, fold: int,
    source: Mapping[str, Any], names: tuple[str, ...],
) -> dict[str, Any]:
    plan = live_plan(experiment.stage3, experiment.recipe, fold, model, prepared, train_data, features)
    plan["plugin"] = {"mode": "stage2_home_transfer_full_finetune", "loaded_parameters": list(names)}
    plan["stage2_home_transfer"] = provenance(experiment, source, names)
    plan["format_version"] = PLAN_VERSION
    return plan


def train_fold(experiment: FinetuneExperiment, fold: int, *, resume: bool = False) -> list[dict[str, Any]]:
    if fold not in range(1, 6):
        raise ValueError("Fine-tuning fold must be 1..5")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Fine-tuning supports one process per fold")
    config = experiment.stage3
    device = resolve_device(config.training.device)
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Fine-tuning requires a BF16-capable CUDA GPU")
    prepared = load_prepared_stage3(config)
    require_paired_data(experiment.source, prepared, config)
    features = load_features(config, experiment.feature_dir, prepared)
    model, store, names, source = build_model_and_store(experiment, prepared, features, fold=fold, device=device)
    validate_initial_representation(model, store, prepared)
    tasks = tuple(task for task, spec in prepared["registry"].items() if spec.enabled)
    train_data = {task: Stage3TaskDataset(config.data.artifacts_dir, fold, task, "train") for task in tasks}
    valid_data = {task: Stage3TaskDataset(config.data.artifacts_dir, fold, task, "valid") for task in tasks}
    plan = resolved_plan(experiment, prepared, features, model, train_data, fold=fold, source=source, names=names)
    output = experiment.checkpoint_dir / f"fold{fold}"
    if output.exists() and not resume:
        raise FileExistsError(f"Fine-tuning fold output already exists: {output}")
    return run_three_phase_training(
        config=config, fold=fold, output_dir=output,
        resume_from=output if resume and output.exists() else None,
        model=model, registry=prepared["registry"], active=tasks,
        train_data=train_data, valid_data=valid_data, representations=store,
        normalizations=prepared["normalization"][f"fold{fold}"], plan=plan, device=device,
        progress=ProgressReporter(),
    )


def evaluation_identity(experiment: FinetuneExperiment, *, split: str, fold: int | None) -> dict[str, Any]:
    folds = (fold,) if split == "valid" else range(1, 6)
    anchors = []
    for current_fold in folds:
        path = experiment.checkpoint_dir / f"fold{current_fold}" / "three_phase_final.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("kind") != FINAL_KIND or manifest.get("fold") != current_fold:
            raise ValueError("HoME full fine-tuning evaluation anchor mismatch")
        anchors.append({
            "fold": current_fold, "training_identity": manifest["training_identity"]["hash"],
            "model_state_hash": manifest["model_state_hash"], "manifest_sha256": sha256_file(path),
            "artifact_sha256": manifest["artifact_sha256"],
        })
    return semantic_identity("stage3.home-transfer-full-finetune-evaluation.v1", {
        "split": split, "fold": fold, "anchors": anchors,
        "source_artifact_sha256": sha256_file(experiment.source.output_root / "stage2" / "stage2_home_transfer.pt"),
        "recipe": experiment.recipe.to_dict(),
    })


def evaluate(
    experiment: FinetuneExperiment, *, split: str, fold: int | None = None,
    predictions_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = experiment.stage3
    prepared = load_prepared_stage3(config)
    features = load_features(config, experiment.feature_dir, prepared)

    def load_model(
        requested_config: Stage3Config, requested_prepared: Mapping[str, Any], path: Path,
        current_fold: int, epoch: int, device: torch.device, *,
        taskwise_refined: bool, three_phase_final: bool,
    ) -> tuple[Any, dict[str, Any], Any]:
        if requested_config != config or taskwise_refined or not three_phase_final:
            raise ValueError("HoME full fine-tuning requires its final artifact")
        model, store, names, source = build_model_and_store(experiment, requested_prepared, features, fold=current_fold, device=device)
        tasks = tuple(task for task, spec in requested_prepared["registry"].items() if spec.enabled)
        train_data = {task: Stage3TaskDataset(config.data.artifacts_dir, current_fold, task, "train") for task in tasks}
        expected_plan = resolved_plan(experiment, requested_prepared, features, model, train_data, fold=current_fold, source=source, names=names)
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (
            artifact.get("kind") != FINAL_KIND or manifest.get("kind") != FINAL_KIND
            or artifact.get("format_version") != 1 or manifest.get("format_version") != 1
            or artifact.get("fold") != current_fold or manifest.get("fold") != current_fold
            or manifest.get("artifact_sha256") != sha256_file(path)
            or artifact.get("ownership_manifest") != model.ownership_manifest()
            or artifact.get("stage2_home_transfer") != expected_plan["stage2_home_transfer"]
            or manifest.get("stage2_home_transfer") != expected_plan["stage2_home_transfer"]
            or artifact.get("resolved_training_plan", {}).get("format_version") != PLAN_VERSION
            or artifact.get("normalization") != requested_prepared["normalization"][f"fold{current_fold}"]
            or artifact.get("normalization_hash") != expected_plan["normalization_hash"]
            or artifact.get("model_state_hash") != tensor_state_hash("stage3.three-phase-model-state", artifact["model"])
            or manifest.get("model_state_hash") != artifact.get("model_state_hash")
        ):
            raise ValueError("HoME full fine-tuning final artifact is incompatible or corrupt")
        identity = build_stage3_training_identity(expected_plan)
        require_compatible_identity(identity, artifact.get("training_identity", {}), context="HoME full fine-tuning final identity")
        require_compatible_identity(identity, manifest.get("training_identity", {}), context="HoME full fine-tuning manifest identity")
        require_compatible_identity(identity, build_stage3_training_identity(artifact["resolved_training_plan"]), context="HoME full fine-tuning resolved plan")
        model.load_state_dict(artifact["model"], strict=True)
        if artifact.get("encoder_state_hashes") != encoder_state_hashes(model) or manifest.get("encoder_state_hashes") != artifact.get("encoder_state_hashes"):
            raise ValueError("HoME full fine-tuning encoder state hash mismatch")
        store.freeze_after_phase1(model, artifact["phase1_model_state_hash"])
        return model.eval(), artifact, store

    result = evaluate_checkpoints(
        config, experiment.checkpoint_dir, split=split, fold=fold,
        ensemble_folds=split == "test", predictions_dir=predictions_dir,
        reporting_study_id="ilume-stage2-home-transfer-full-finetune-v1", model_loader=load_model,
    )
    result["ablation"] = "stage2_home_transfer_full_finetune"
    result["reporting"]["model_display_name"] = "ILUME (HoME transfer full fine-tune)"
    result["evaluation_identity"] = evaluation_identity(experiment, split=split, fold=fold)
    return result
