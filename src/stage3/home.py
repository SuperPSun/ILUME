from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from common.identity import require_compatible_identity, tensor_state_hash, validate_semantic_identity
from common.io import sha256_file
from common.progress import ProgressReporter
from common.training import resolve_device, seed_everything
from stage2.home_contract import SOURCE_GROUPS, source_groups, load_transferable_state, state_hash
from stage2.home_artifact import STAGE2_HOME_FINAL_KIND, STAGE2_HOME_V4_FINAL_KIND, load_home_final
from stage2.train import load_stage2_encoder_artifact
from stage2 import load_frozen_object_encoder

from .config import Stage3Config, effective_training_seed
from .data import Stage3TaskDataset
from .identity import build_stage3_training_identity
from .object_phase1 import (
    OBJECT_ENCODER_OWNER, ObjectPhase1Model, ObjectPhase1Representations, build_object_phase1_model,
    validate_encoder_source, validate_initial_object_embeddings,
)
from .model import GLOBAL, group_owner, private_owner
from .simulation import (
    simulation_tasks, SimulationObjectPhase1Model,
    copy_simulation_owners, extend_simulation_plan,
    load_simulation_plan_data, load_simulation_training_data,
)
from .prepare import load_prepared_stage3
from .three_phase import run_three_phase_training
from .train import build_resolved_training_plan


FINAL_KIND = "ilume_stage3_home_simulation_three_phase_final_v2"


def load_source(config: Stage3Config) -> dict[str, Any] | None:
    mode = config.initialization.home_mode
    if mode == "no_stage2":
        return None
    if mode != "trained" or config.initialization.stage2_final is None:
        raise ValueError("Stage 3 HoME requires a declared source mode")
    artifact_path = config.initialization.stage2_final
    manifest_path = artifact_path.with_suffix(".json")
    encoder_path = config.initialization.stage2_encoder
    assert encoder_path is not None
    if not artifact_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Formal Stage 2 final artifact is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload, loaded_model, vocabulary = load_home_final(artifact_path)
    identity = payload.get("training_identity")
    validate_semantic_identity(identity)
    encoder = load_stage2_encoder_artifact(encoder_path)
    expected_mapping = {
        group: [task for task, mapped in source_groups(loaded_model.registry).items() if mapped == group]
        for group in ("thermophysical", "solvation")
    }
    source_recipe = identity.get("payload", {})
    if (
        payload.get("kind") != ("ilume_stage2_home_final_v5" if config.is_v5 else STAGE2_HOME_V4_FINAL_KIND if config.initialization.representation_contract == "dual_view_v4" else STAGE2_HOME_FINAL_KIND)
        or manifest.get("kind") != payload.get("kind")
        or manifest.get("artifact_sha256") != sha256_file(artifact_path)
        or manifest.get("training_identity", {}).get("hash") != identity["hash"]
        or payload.get("shared_state_hash") != state_hash(payload["shared_state"])
        or manifest.get("shared_state_hash") != payload["shared_state_hash"]
        or payload.get("stage2_encoder_sha256") != sha256_file(encoder_path)
        or manifest.get("stage2_encoder_sha256") != payload["stage2_encoder_sha256"]
        or payload.get("group_mapping") != expected_mapping
        or manifest.get("fixed_final_epoch") != payload["training_identity"]["payload"].get("epochs")
        or encoder.get("provenance", {}).get("home_training_identity") != identity["hash"]
        or encoder.get("state_hashes", {}).get("stage1_backbone") != payload.get("encoder_state_hashes", {}).get("stage1")
        or encoder.get("state_hashes", {}).get("object_encoder") != payload.get("encoder_state_hashes", {}).get("object_encoder")
        or tensor_state_hash("stage2.encoder-state", payload["stage1_backbone"]) != encoder["state_hashes"]["stage1_backbone"]
        or tensor_state_hash("stage2.encoder-state", payload["object_encoder"]) != encoder["state_hashes"]["object_encoder"]
        or payload.get("architecture", {}).get("stage3_model") != asdict(config.model)
        or source_recipe.get("stage3_model") != asdict(config.model)
        or source_recipe.get("transfer_groups") != {
            group: asdict(config.groups[group]) for group in ("thermophysical", "solvation")
        }
        or source_recipe.get("source_groups") != (source_groups(loaded_model.registry) if config.is_v5 else SOURCE_GROUPS)
        or (config.is_v5 and encoder["object_encoder_config"] != loaded_model.model_contract["object_encoder"])
    ):
        raise ValueError("Formal Stage 2 HoME source identity or state is incompatible")
    require_compatible_identity(identity, manifest["training_identity"], context="Stage 2 HoME final")
    if config.initialization.simulation_artifacts_dir is not None:
        payload["_loaded_model"] = loaded_model
        payload["_vocabulary"] = vocabulary
    return payload


def source_plan(config: Stage3Config, source: Mapping[str, Any] | None, names: tuple[str, ...]) -> dict[str, Any]:
    if source is None:
        return {"mode": "no_stage2", "transferred_parameters": []}
    artifact_path = config.initialization.stage2_final
    assert artifact_path is not None
    result = {
        "mode": "trained",
        "source_training_identity": source["training_identity"]["hash"],
        "source_artifact_sha256": sha256_file(artifact_path),
        "source_encoder_sha256": source["stage2_encoder_sha256"],
        "transferred_state_hash": source["shared_state_hash"],
        "transferred_parameters": list(names),
    }
    if config.initialization.simulation_artifacts_dir is not None:
        result["simulation_training"] = {
            "tasks": list(simulation_tasks(config)),
            "source_data_identity": source["stage2_data_identity"]["hash"],
            "source_full_model_state_hash": source["full_model_state_hash"],
        }
    return result


def build_model_and_store(
    config: Stage3Config, prepared: Mapping[str, Any], *, fold: int,
    device: torch.device, source: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, tuple[str, ...]]:
    seed_everything(effective_training_seed(config) + fold)
    simulation_enabled = config.initialization.simulation_artifacts_dir is not None
    if simulation_enabled:
        if source is None:
            source = load_source(config)
        assert source is not None
        source_model = source["_loaded_model"]
        if prepared.get("slots") is None:
            raise ValueError("Simulation Stage 3 requires prepared frozen entity slots")
        validate_encoder_source(config)
        encoder = load_frozen_object_encoder(config.initialization.stage2_encoder, device="cpu")
        specs = dict(prepared["registry"])
        from .simulation import resolve_simulation_specs

        specs.update(resolve_simulation_specs(source_model.registry, specs, config.training.simulation, simulation_tasks(config)))
        seed_everything(effective_training_seed(config) + fold)
        private_recipes = {
            task: config.resolved_private_recipe(task)
            for task, spec in config.tasks.items() if spec.enabled
        }
        experimental_model = ObjectPhase1Model(
            config.model, prepared["registry"], encoder.embedding_dim,
            group_configs=config.groups, task_configs=config.tasks,
            task_private_recipes=private_recipes,
            object_encoder=encoder.object_encoder,
            initialization_seed=effective_training_seed(config) + fold if config.is_v5 else None,
        )
        model = SimulationObjectPhase1Model(
            config.model, specs, encoder.embedding_dim,
            group_configs=config.groups, task_configs=config.tasks,
            task_private_recipes=private_recipes,
            object_encoder=encoder.object_encoder,
            initialization_seed=effective_training_seed(config) + fold if config.is_v5 else None, source_model=source_model,
        )
        # Extra source owners must not consume the experimental initialization RNG.
        state = model.state_dict()
        state.update(experimental_model.state_dict())
        model.load_state_dict(state, strict=True)
        del state, experimental_model
        model.to(device)
        model.simulation_vocabulary = source["_vocabulary"]
        store = ObjectPhase1Representations(model, prepared["slots"])
        copy_simulation_owners(model, source_model)
    else:
        model, store = build_object_phase1_model(config, prepared, fold=fold, device=device)
    model.representation_contract = config.initialization.representation_contract
    names: tuple[str, ...] = ()
    if config.initialization.home_mode == "trained":
        if source is None:
            source = load_source(config)
        assert source is not None
        names = load_transferable_state(model, source["shared_state"], source["shared_state_hash"])
    elif config.initialization.home_mode != "no_stage2":
        raise ValueError("Stage 3 HoME source mode is missing")
    if simulation_enabled:
        experimental = tuple(task for task, spec in prepared["registry"].items() if spec.enabled)
        owners = {GLOBAL, OBJECT_ENCODER_OWNER}
        owners.update(group_owner(prepared["registry"][task].meta_group) for task in experimental)
        owners.update(private_owner(task) for task in experimental)
        model.set_trainable_owners(owners)
    else:
        model.set_trainable_owners(set(model.parameter_ownership().values()))
    return model, store, names


def train_fold(
    config: Stage3Config, fold: int, *, output_dir: str | Path,
    resume_from: str | Path | None = None,
    expected_training_identity: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if fold not in range(1, 6):
        raise ValueError("Stage 3 fold must be in 1..5")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Stage 3 training supports one process per fold only")
    device = resolve_device(config.training.device)
    if config.training.amp_dtype == "bf16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("Stage 3 BF16 requires a BF16-capable CUDA GPU")
    source = load_source(config)
    prepared = load_prepared_stage3(config)
    model, store, loaded_names = build_model_and_store(
        config, prepared, fold=fold, device=device, source=source,
    )
    if source is not None:
        source.pop("_loaded_model", None)
    validate_initial_object_embeddings(model, store, prepared)
    tasks = tuple(task for task, spec in prepared["registry"].items() if spec.enabled)
    train_data = {task: Stage3TaskDataset(config.data.artifacts_dir, fold, task, "train") for task in tasks}
    valid_data = {task: Stage3TaskDataset(config.data.artifacts_dir, fold, task, "valid") for task in tasks}
    normalizations = prepared["normalization"][f"fold{fold}"]
    simulation_data = (
        load_simulation_training_data(
            config.initialization.simulation_artifacts_dir,
            source, source["_vocabulary"], device,
            amp_dtype=config.training.amp_dtype,
        )
        if config.initialization.simulation_artifacts_dir is not None else None
    )
    plan = build_resolved_training_plan(
        config, fold, model, train_data, tasks, prepared,
        {"mode": "stage2_pretraining", "loaded_parameters": list(loaded_names)},
        normalizations,
    )
    plan["stage2_pretraining"] = source_plan(config, source, loaded_names)
    if simulation_data is not None:
        extend_simulation_plan(plan, config, model, simulation_data, source)
    else:
        plan["format_version"] = 13 if config.is_v5 else 11 if config.initialization.representation_contract == "dual_view_v4" else 9
    if expected_training_identity is not None:
        require_compatible_identity(
            expected_training_identity, build_stage3_training_identity(plan),
            context="Stage 3 HoME run identity",
        )
    return run_three_phase_training(
        config=config, fold=fold, output_dir=Path(output_dir), resume_from=resume_from,
        model=model, registry=model.task_specs if simulation_data is not None else prepared["registry"], active=tasks,
        train_data=train_data, valid_data=valid_data, representations=store,
        normalizations=normalizations, plan=plan, device=device,
        progress=ProgressReporter(), simulation_data=simulation_data,
    )


def resolve_training_identity(config: Stage3Config, fold: int) -> dict[str, Any]:
    if fold not in range(1, 6):
        raise ValueError("Stage 3 fold must be in 1..5")
    source = load_source(config)
    prepared = load_prepared_stage3(config)
    model, _, loaded_names = build_model_and_store(
        config, prepared, fold=fold, device=torch.device("cpu"), source=source,
    )
    if source is not None:
        source.pop("_loaded_model", None)
    tasks = tuple(task for task, spec in prepared["registry"].items() if spec.enabled)
    datasets = {task: Stage3TaskDataset(config.data.artifacts_dir, fold, task, "train") for task in tasks}
    plan = build_resolved_training_plan(
        config, fold, model, datasets, tasks, prepared,
        {"mode": "stage2_pretraining", "loaded_parameters": list(loaded_names)},
        prepared["normalization"][f"fold{fold}"],
    )
    plan["stage2_pretraining"] = source_plan(config, source, loaded_names)
    if config.initialization.simulation_artifacts_dir is not None:
        simulation_data = load_simulation_plan_data(
            config.initialization.simulation_artifacts_dir, source,
        )
        extend_simulation_plan(plan, config, model, simulation_data, source)
    else:
        plan["format_version"] = 13 if config.is_v5 else 11 if config.initialization.representation_contract == "dual_view_v4" else 9
    return build_stage3_training_identity(plan)


def load_simulation_final(
    config: Stage3Config, checkpoint_dir: str | Path, *, fold: int,
    device: str | torch.device = "cpu",
) -> SimulationObjectPhase1Model:
    """Load a validated Stage 3 final model for its simulation task protocol."""
    if config.initialization.simulation_artifacts_dir is None or fold not in range(1, 6):
        raise ValueError("Stage 3 simulation inference requires a simulation-trained fold")
    from .evaluate import _load_model, _validate_three_phase_manifest

    root = Path(checkpoint_dir)
    nested = root / f"fold{fold}" / "three_phase_final.pt"
    artifact = nested if nested.is_file() else root / "three_phase_final.pt"
    prepared = load_prepared_stage3(config)
    model, checkpoint, _ = _load_model(
        config, prepared, artifact, fold, 0, torch.device(device),
        three_phase_final=True,
    )
    _validate_three_phase_manifest(artifact, checkpoint)
    if not isinstance(model, SimulationObjectPhase1Model):
        raise ValueError("Stage 3 final artifact lacks simulation prediction owners")
    return model
