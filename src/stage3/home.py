from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch

from common.identity import require_compatible_identity
from common.io import sha256_file
from common.progress import ProgressReporter
from common.training import resolve_device, seed_everything
from stage2.home_contract import load_transferable_state
from stage2.home_artifact import load_home_final
from stage1.identity import encoding_state_hash
from stage2.frozen import load_entity_source

from .config import Stage3Config, effective_training_seed
from .data import Stage3TaskDataset
from .identity import build_stage3_training_identity
from .entity_inputs import EntityRepresentations
from .model import Stage3SparseModel
from .model import GLOBAL, group_owner, private_owner
from .simulation import (
    simulation_tasks, SimulationEntityModel,
    copy_simulation_owners, extend_simulation_plan,
    load_simulation_plan_data, load_simulation_training_data,
)
from .prepare import load_prepared_stage3
from .three_phase import run_three_phase_training
from .train import build_resolved_training_plan


FINAL_KIND = "ilume_stage3_home_simulation_three_phase_final_v2"


def validate_transfer_architecture(source_model, target_model, source_groups, target_groups):
    """Keep shared and simulation structure fixed; owner training budgets may differ."""
    if source_model != target_model:
        raise ValueError("Stage2 source architecture mismatch: shared/simulation model")
    for name in ("thermophysical", "solvation"):
        source, target = source_groups[name], target_groups[name]
        def structure(group, model):
            return (group.enabled,
                    model.group_experts if group.experts is None else group.experts,
                    model.expert_hidden_ratio if group.expert_hidden_ratio is None else group.expert_hidden_ratio)
        if structure(source, source_model) != structure(target, target_model):
            raise ValueError(f"Stage2 source architecture mismatch: GROUP {name}")


def load_source(config: Stage3Config) -> dict[str, Any]:
    if config.runtime_backend is not None:
        return config.runtime_backend.load_source(config)
    if not config.is_entity_home or config.initialization.home_mode != "trained":
        raise ValueError("Entity HoME requires the new Stage2 full artifact")
    artifact_path = config.initialization.stage2_final
    payload, loaded_model, vocabulary = load_home_final(artifact_path)
    validate_transfer_architecture(loaded_model.home.model_config, config.model,
                                   loaded_model.home.group_configs, config.groups)
    frozen = load_entity_source(config)
    if (encoding_state_hash(loaded_model.backbone) != frozen.encoder_identity["payload"]["stage1_state_hash"]
            or payload["stage1_feature_identity"]["hash"] != frozen.encoder_identity["payload"]["feature_identity"]):
        raise ValueError("Stage2 source architecture or frozen Stage1 identity mismatch")
    payload["_loaded_model"], payload["_vocabulary"] = loaded_model, vocabulary
    return payload


def source_plan(config: Stage3Config, source: Mapping[str, Any] | None, names: tuple[str, ...]) -> dict[str, Any]:
    if config.runtime_backend is not None:
        return config.runtime_backend.source_plan(config, source, names)
    if source is None:
        return {"mode": "no_stage2", "transferred_parameters": []}
    artifact_path = config.initialization.stage2_final
    assert artifact_path is not None
    result = {
        "mode": "trained",
        "source_training_identity": source["training_identity"]["hash"],
        "source_artifact_sha256": sha256_file(artifact_path),
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
    source = source or load_source(config)
    source_model = source["_loaded_model"]
    frozen = prepared["metadata"]["stage1_encoder_identity"]["payload"]
    if (frozen["stage1_state_hash"] != encoding_state_hash(source_model.backbone)
            or frozen["feature_identity"] != source["stage1_feature_identity"]["hash"]):
        raise ValueError("Stage3 entity slots differ from Stage2 frozen Stage1")
    specs = dict(prepared["registry"])
    simulation_enabled = config.initialization.simulation_artifacts_dir is not None
    if simulation_enabled:
        from .simulation import resolve_simulation_specs
        specs.update(resolve_simulation_specs(source_model.registry, specs,
                     config.training.simulation, simulation_tasks(config)))
    kwargs = dict(group_configs=config.groups, task_configs=config.tasks,
        task_private_recipes={task: config.resolved_private_recipe(task)
            for task, spec in config.tasks.items() if spec.enabled},
        initialization_seed=effective_training_seed(config) + fold, entity_inputs=True)
    model = (SimulationEntityModel(config.model, specs, 1024, source_model=source_model, **kwargs)
        if simulation_enabled else Stage3SparseModel(config.model, specs, 1024, **kwargs))
    model.representation_contract = "entity_home_v4"
    model.to(device)
    if simulation_enabled:
        model.simulation_vocabulary = source["_vocabulary"]
        copy_simulation_owners(model, source_model)
    names = load_transferable_state(model, source["shared_state"], source["shared_state_hash"])
    if config.runtime_backend is not None:
        model, names = config.runtime_backend.transform_model(config, model, names)
    store = EntityRepresentations(prepared["slots"])
    experimental = [task for task, spec in prepared["registry"].items() if spec.enabled]
    owners = {GLOBAL, *(group_owner(specs[task].meta_group) for task in experimental),
              *(private_owner(task) for task in experimental)}
    model.set_trainable_owners(owners)
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
        plan["format_version"] = 14 if config.is_entity_home else 11 if config.initialization.representation_contract == "dual_view_v4" else 9
    if config.runtime_backend is not None:
        config.runtime_backend.finalize_plan(config, model, plan)
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
        plan["format_version"] = 14 if config.is_entity_home else 11 if config.initialization.representation_contract == "dual_view_v4" else 9
    if config.runtime_backend is not None:
        config.runtime_backend.finalize_plan(config, model, plan)
    return build_stage3_training_identity(plan)


def load_simulation_final(
    config: Stage3Config, checkpoint_dir: str | Path, *, fold: int,
    device: str | torch.device = "cpu",
) -> SimulationEntityModel:
    """Load a validated Stage 3 final model for its simulation task protocol."""
    if config.initialization.simulation_artifacts_dir is None or fold not in range(1, 6):
        raise ValueError("Stage 3 simulation inference requires a simulation-trained fold")
    from .evaluate import _load_model, _validate_three_phase_manifest

    root = Path(checkpoint_dir)
    nested = root / f"fold{fold}" / "three_phase_final.pt"
    artifact = (config.runtime_backend.artifact_path(root, fold, config) if config.runtime_backend is not None
        else nested if nested.is_file() else root / "three_phase_final.pt")
    prepared = load_prepared_stage3(config)
    model, checkpoint, _ = _load_model(
        config, prepared, artifact, fold, 0, torch.device(device),
        three_phase_final=True,
    )
    _validate_three_phase_manifest(artifact, checkpoint, config)
    if not isinstance(model, SimulationEntityModel):
        raise ValueError("Stage 3 final artifact lacks simulation prediction owners")
    return model
