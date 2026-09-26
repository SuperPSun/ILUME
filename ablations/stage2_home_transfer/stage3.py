from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import torch

from common.identity import require_compatible_identity, tensor_state_hash, validate_semantic_identity
from common.io import sha256_file
from common.progress import ProgressReporter
from common.training import canonical_json_sha256, resolve_device, seed_everything
from stage3.config import Stage3Config, effective_training_seed
from stage3.data import Stage3TaskDataset
from stage3.evaluate import evaluate_checkpoints
from stage3.identity import (
    build_stage3_evaluation_identity, build_stage3_training_identity, metadata_identity,
)
from stage3.object_phase1 import build_object_phase1_model, validate_initial_object_embeddings
from stage3.prepare import load_prepared_stage3, prepare_stage3
from stage3.three_phase import run_three_phase_training
from stage3.train import build_resolved_training_plan
from stage2.train import load_stage2_encoder_artifact

from .config import Experiment
from .contract import SOURCE_GROUPS, load_transferable_state, state_hash
from .stage2 import STAGE2_HOME_ARTIFACT_KIND, training_identity


FINAL_KIND = "ilume_stage3_stage2_home_transfer_three_phase_final"


def load_source(experiment: Experiment) -> dict[str, Any]:
    root = experiment.output_root / "stage2"
    artifact_path = root / "stage2_home_transfer.pt"
    manifest_path = root / "stage2_home_transfer.json"
    if not artifact_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Stage2-HoME transfer source is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    validate_semantic_identity(payload["training_identity"])
    encoder = load_stage2_encoder_artifact(root / "stage2_encoder.pt")
    source_identity = payload["training_identity"].get("payload", {})
    stage2_metadata = json.loads(
        (experiment.stage2.data.artifacts_dir / "metadata.json").read_text(encoding="utf-8")
    )
    expected_identity = training_identity(
        experiment,
        metadata_identity(stage2_metadata, "data", context="Stage2-HoME source"),
        source_identity.get("math_contract", {}),
    )
    require_compatible_identity(
        expected_identity, payload["training_identity"],
        context="Stage2-HoME transfer source",
    )
    expected_mapping = {
        group: [task for task, mapped in SOURCE_GROUPS.items() if mapped == group]
        for group in ("thermophysical", "solvation")
    }
    if (
        payload.get("kind") != STAGE2_HOME_ARTIFACT_KIND
        or manifest.get("kind") != STAGE2_HOME_ARTIFACT_KIND
        or manifest.get("artifact_sha256") != sha256_file(artifact_path)
        or payload.get("shared_state_hash") != state_hash(payload["shared_state"])
        or manifest.get("shared_state_hash") != payload["shared_state_hash"]
        or payload.get("stage2_encoder_sha256") != sha256_file(root / "stage2_encoder.pt")
        or manifest.get("stage2_encoder_sha256") != payload["stage2_encoder_sha256"]
        or payload.get("training_identity") != manifest.get("training_identity")
        or payload.get("checkpoint_sha256") != sha256_file(root / "checkpoint_epoch_00010.pt")
        or payload.get("group_mapping") != expected_mapping
        or payload.get("encoder_state_hashes", {}).get("stage1") != encoder.get("state_hashes", {}).get("stage1_backbone")
        or payload.get("encoder_state_hashes", {}).get("object_encoder") != encoder.get("state_hashes", {}).get("object_encoder")
        or tensor_state_hash("stage2.encoder-state", payload["stage1_backbone"]) != encoder["state_hashes"]["stage1_backbone"]
        or tensor_state_hash("stage2.encoder-state", payload["object_encoder"]) != encoder["state_hashes"]["object_encoder"]
        or manifest.get("fixed_final_epoch") != 10
    ):
        raise ValueError("Stage2-HoME transfer source identity or state is corrupt")
    if (payload.get("architecture", {}).get("stage3_model") != asdict(experiment.stage3.model)
        or payload.get("architecture", {}).get("electronic_group") != expected_identity["payload"]["electronic_group"]):
        raise ValueError("Stage2-HoME transfer source model architecture mismatch")
    return payload


def stage3_config(experiment: Experiment) -> Stage3Config:
    root = experiment.output_root
    return replace(
        experiment.stage3,
        data=replace(
            experiment.stage3.data,
            artifacts_dir=root / "stage3" / "prepare" / "artifacts",
        ),
        preparation=replace(
            experiment.stage3.preparation,
            cache_dir=root / "stage3" / "prepare" / "object_cache",
        ),
        initialization=replace(
            experiment.stage3.initialization,
            stage2_encoder=root / "stage2" / "stage2_encoder.pt",
        ),
    )


def _require_paired_data(experiment: Experiment, prepared: Mapping[str, Any]) -> None:
    control = load_prepared_stage3(experiment.stage3)
    if prepared["normalization"] != control["normalization"]:
        raise ValueError("Stage2-HoME Stage3 normalization differs from Base")
    if prepared["objects"]["objects"] != control["objects"]["objects"]:
        raise ValueError("Stage2-HoME Stage3 ObjectKey order differs from Base")
    source_tasks = {task for task, spec in control["registry"].items() if spec.enabled}
    target_tasks = {task for task, spec in prepared["registry"].items() if spec.enabled}
    if source_tasks != target_tasks or len(target_tasks) != 20:
        raise ValueError("Stage2-HoME Stage3 task catalog differs from Base")
    fields = (
        "primary_object_ids", "partner_object_ids", "conditions", "targets",
        "raw_targets", "source_folds", "source_rows",
    )
    for fold in range(1, 6):
        for task in sorted(target_tasks):
            for split in ("train", "valid"):
                source = Stage3TaskDataset(experiment.stage3.data.artifacts_dir, fold, task, split)
                target = Stage3TaskDataset(stage3_config(experiment).data.artifacts_dir, fold, task, split)
                if any(not torch.equal(getattr(source, field), getattr(target, field)) for field in fields):
                    raise ValueError(f"Stage2-HoME Stage3 fold/split differs from Base: {fold}/{task}/{split}")


def prepare(experiment: Experiment) -> dict[str, Any]:
    load_source(experiment)
    config = stage3_config(experiment)
    result = prepare_stage3(config)
    _require_paired_data(experiment, load_prepared_stage3(config))
    return result


def build_model_and_store(
    experiment: Experiment, prepared: Mapping[str, Any], *, fold: int,
    device: torch.device,
) -> tuple[Any, Any, tuple[str, ...]]:
    source = load_source(experiment)
    config = stage3_config(experiment)
    seed_everything(effective_training_seed(config) + fold)
    model, store = build_object_phase1_model(config, prepared, fold=fold, device=device)
    names = load_transferable_state(model, source["shared_state"], source["shared_state_hash"])
    model.set_trainable_owners(set(model.parameter_ownership().values()))
    return model, store, names


def train_fold(experiment: Experiment, fold: int, *, resume: bool = False) -> list[dict[str, Any]]:
    if fold not in range(1, 6):
        raise ValueError("Stage2-HoME Stage3 fold must be in 1..5")
    config = stage3_config(experiment)
    source = load_source(experiment)
    prepared = load_prepared_stage3(config)
    _require_paired_data(experiment, prepared)
    device = resolve_device(config.training.device)
    model, store, loaded_names = build_model_and_store(experiment, prepared, fold=fold, device=device)
    validate_initial_object_embeddings(model, store, prepared)
    tasks = tuple(task for task, spec in prepared["registry"].items() if spec.enabled)
    train_data = {task: Stage3TaskDataset(config.data.artifacts_dir, fold, task, "train") for task in tasks}
    valid_data = {task: Stage3TaskDataset(config.data.artifacts_dir, fold, task, "valid") for task in tasks}
    normalizations = prepared["normalization"][f"fold{fold}"]
    plan = build_resolved_training_plan(
        config, fold, model, train_data, tasks, prepared,
        {"mode": "stage2_home_transfer", "loaded_parameters": list(loaded_names)},
        normalizations,
    )
    plan["stage2_home_transfer"] = {
        "source_training_identity": source["training_identity"]["hash"],
        "source_artifact_sha256": sha256_file(experiment.output_root / "stage2" / "stage2_home_transfer.pt"),
        "source_encoder_sha256": source["stage2_encoder_sha256"],
        "transferred_state_hash": source["shared_state_hash"],
        "transferred_parameters": list(loaded_names),
    }
    plan["format_version"] = 8
    output = experiment.output_root / "stage3" / "train" / f"fold{fold}"
    if output.exists() and not resume:
        raise FileExistsError(f"Stage2-HoME Stage3 fold output already exists: {output}")
    if resume and not output.exists():
        raise FileNotFoundError(f"Stage2-HoME Stage3 resume fold missing: {output}")
    return run_three_phase_training(
        config=config, fold=fold, output_dir=output,
        resume_from=output if resume else None,
        model=model, registry=prepared["registry"], active=tasks,
        train_data=train_data, valid_data=valid_data, representations=store,
        normalizations=normalizations, plan=plan, device=device,
        progress=ProgressReporter(),
    )


def evaluate(
    experiment: Experiment, *, split: str, fold: int | None = None,
    predictions_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = stage3_config(experiment)
    source = load_source(experiment)

    def load_model(
        requested_config: Stage3Config, prepared: Mapping[str, Any], path: Path,
        current_fold: int, epoch: int, device: torch.device, *,
        taskwise_refined: bool, three_phase_final: bool,
    ) -> tuple[Any, dict[str, Any], Any]:
        if requested_config != config or not three_phase_final or taskwise_refined:
            raise ValueError("Stage2-HoME evaluation requires its final three-phase artifact")
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        manifest = json.loads(path.with_name("three_phase_final.json").read_text(encoding="utf-8"))
        plan = artifact.get("resolved_training_plan")
        if (
            artifact.get("kind") != FINAL_KIND
            or manifest.get("kind") != FINAL_KIND
            or artifact.get("fold") != current_fold
            or manifest.get("fold") != current_fold
            or manifest.get("artifact_sha256") != sha256_file(path)
            or not isinstance(plan, dict)
            or plan.get("format_version") != 8
            or plan.get("stage2_home_transfer", {}).get("transferred_state_hash") != source["shared_state_hash"]
            or plan.get("stage2_home_transfer", {}).get("source_artifact_sha256") != sha256_file(experiment.output_root / "stage2" / "stage2_home_transfer.pt")
            or artifact.get("stage2_home_transfer") != plan.get("stage2_home_transfer")
            or manifest.get("stage2_home_transfer") != plan.get("stage2_home_transfer")
            or plan.get("prepared_identity") != metadata_identity(prepared["metadata"], "prepared", context="Stage2-HoME evaluation")["hash"]
            or plan.get("normalization_hash") != canonical_json_sha256(artifact.get("normalization"))
            or artifact.get("model_state_hash") != tensor_state_hash("stage3.object-phase1-model-state.v1", artifact["model"])
        ):
            raise ValueError("Stage2-HoME Stage3 final artifact is incompatible or corrupt")
        require_compatible_identity(
            build_stage3_training_identity(plan), artifact.get("training_identity", {}),
            context="Stage2-HoME Stage3 final identity",
        )
        model, store, loaded_names = build_model_and_store(
            experiment, prepared, fold=current_fold, device=device,
        )
        if artifact.get("ownership_manifest") != model.ownership_manifest() or list(loaded_names) != plan["stage2_home_transfer"]["transferred_parameters"]:
            raise ValueError("Stage2-HoME Stage3 ownership or transferred-state mismatch")
        model.load_state_dict(artifact["model"], strict=True)
        if artifact.get("object_encoder_state_hash") != tensor_state_hash("stage3.object-phase1.final-encoder.v1", model.stage2_object_encoder.state_dict()):
            raise ValueError("Stage2-HoME final ObjectEncoder state mismatch")
        store.freeze_after_phase1(model, artifact["phase1_model_state_hash"])
        if store.final_embedding_hash != artifact.get("final_embedding_hash"):
            raise ValueError("Stage2-HoME final representation hash mismatch")
        return model.eval(), artifact, store

    result = evaluate_checkpoints(
        config, experiment.output_root / "stage3" / "train",
        split=split, ensemble_folds=split == "test", fold=fold,
        predictions_dir=predictions_dir,
        reporting_study_id="ilume-stage2-home-transfer-v1", model_loader=load_model,
    )
    result["ablation"] = "stage2_home_transfer"
    result["source_stage2_home_identity"] = source["training_identity"]["hash"]
    result["reporting"]["model_display_name"] = "ILUME (Stage2-HoME transfer)"
    prepared = load_prepared_stage3(config)
    folds = range(1, 6) if split == "test" else (fold,)
    artifacts = [
        torch.load(
            experiment.output_root / "stage3" / "train" / f"fold{current_fold}" / "three_phase_final.pt",
            map_location="cpu", weights_only=False,
        )
        for current_fold in folds
    ]
    tasks = tuple(result["ensemble"]["tasks"] if split == "test" else result["tasks"])
    result["evaluation_identity"] = build_stage3_evaluation_identity(
        prepared_identity=metadata_identity(prepared["metadata"], "prepared", context="Stage2-HoME evaluation"),
        checkpoint_identities=[artifact["training_identity"] for artifact in artifacts],
        model_state_hashes=[artifact["model_state_hash"] for artifact in artifacts],
        selection_manifest_hashes=[
            sha256_file(
                experiment.output_root / "stage3" / "train" / f"fold{current_fold}" / "three_phase_final.json"
            ) for current_fold in folds
        ],
        split=split, fold=fold, checkpoint_epoch=None,
        model_selector="three_phase_final", tasks=tasks,
        ensemble_folds=split == "test",
    )
    return result
