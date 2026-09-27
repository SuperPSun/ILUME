from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import yaml

from common.identity import require_compatible_identity, semantic_identity, tensor_state_hash
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.progress import ProgressReporter
from common.training import resolve_device, seed_everything
from stage1.masking import MultimodalPacker
from stage2.data import Stage2EntityDataset, Stage2TaskDataset, load_artifact_registry, validate_runtime_task_contract
from stage3.config import Stage3Config, effective_training_seed, stage3_config_from_dict
from stage3.data import Stage3TaskDataset
from stage3.evaluate import evaluate_checkpoints
from stage3.identity import build_stage3_training_identity, metadata_identity
from stage3.object_phase1 import ObjectPhase1Representations, validate_initial_object_embeddings
from stage3.prepare import load_prepared_stage3, prepare_stage3
from stage3.three_phase import run_three_phase_training
from stage3.train import build_resolved_training_plan
from ablations.stage2_home_transfer.config import load_experiment
from ablations.stage2_home_transfer.stage2 import build_model, STAGE2_HOME_CHECKPOINT_KIND
from ablations.stage2_home_transfer.stage3 import load_source as load_transfer_source, require_paired_data

from .model import CrossDomainHoME
from .replay import SimulationReplay, replay_events


SOURCE_KIND = "ilume_stage2_cross_domain_home_source"
FINAL_KIND = "ilume_stage3_cross_domain_home_three_phase_final"
RECIPE = {"replay_every": 4, "replay_initial_weight": 0.1, "replay_batch_size": 256,
          "replay_microbatch_size": 128, "simulation_lr": 1e-5,
          "simulation_warmup_ratio": 0.05, "simulation_min_lr_ratio": 0.1}


@dataclass(frozen=True)
class Experiment:
    source: object
    stage3: Stage3Config
    output_root: Path
    recipe: dict

    @property
    def source_path(self):
        return self.output_root / "source" / "simulation_home.pt"

    @property
    def checkpoint_dir(self):
        return self.output_root / "train"


def load_config(path, *, source_dir, output_root=None):
    raw = yaml.safe_load(Path(path).read_text())
    root = Path(raw.pop("output_root") if output_root is None else output_root)
    raw.pop("output_root", None)
    recipe = raw.pop("cross_domain_home")
    source_config = recipe.pop("source_experiment")
    if recipe != RECIPE:
        raise ValueError("Cross-domain HoME requires the fixed replay/owner recipe")
    source = replace(load_experiment(source_config), output_root=Path(source_dir))
    config = stage3_config_from_dict(raw)
    if config.to_dict() != source.stage3.to_dict():
        raise ValueError("Cross-domain experimental recipe must equal current Base")
    if root.resolve() == source.output_root.resolve() or source.output_root.resolve() in root.resolve().parents:
        raise ValueError("Cross-domain outputs must be isolated from the Stage2 source")
    config = replace(config, data=replace(config.data, artifacts_dir=root / "prepare" / "artifacts"),
                     preparation=replace(config.preparation, cache_dir=root / "prepare" / "object_cache"),
                     initialization=replace(config.initialization, stage2_encoder=source.output_root / "stage2" / "stage2_encoder.pt"))
    return Experiment(source, config, root, dict(recipe))


def export_source(experiment):
    path = experiment.source.output_root / "stage2" / "checkpoint_epoch_00010.pt"
    if not path.is_file():
        raise ValueError("Cross-domain export requires the complete Stage2-HoME epoch10 checkpoint; a partial transfer artifact is insufficient")
    old = load_transfer_source(experiment.source)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (checkpoint.get("kind") != STAGE2_HOME_CHECKPOINT_KIND or checkpoint.get("epoch") != 10
        or checkpoint.get("training_identity") != old["training_identity"]
        or checkpoint.get("model_hash") != tensor_state_hash("stage2-home-transfer.full-model.v1", checkpoint["model"])):
        raise ValueError("Complete epoch10 simulation checkpoint is invalid")
    if experiment.source_path.exists() or experiment.source_path.with_suffix(".json").exists():
        raise FileExistsError("Cross-domain source artifact already exists")
    blocks = {block: tensor_state_hash("cross-domain.source-block", {
        name: value for name, value in checkpoint["model"].items() if name.startswith(prefix)
    }) for block, prefix in (("stage1", "backbone."), ("object_encoder", "object_encoder."),
                            ("simulation_home", "home."), ("atom_adapter", "atom_adapter."))}
    task_contracts = json.loads(json.dumps([task.to_dict() for task in load_artifact_registry(experiment.source.stage2.data.artifacts_dir).tasks]))
    identity = semantic_identity("stage2.cross-domain-home-source", {
        "contract_version": 1, "source_training_identity": old["training_identity"]["hash"],
        "checkpoint_sha256": sha256_file(path), "blocks": blocks,
        "architecture": old["architecture"], "task_contracts": task_contracts, "epoch": 10,
    })
    payload = {"kind": SOURCE_KIND, "format_version": 1, "identity": identity,
               "source_training_identity": old["training_identity"],
               "model": checkpoint["model"], "blocks": blocks,
               "model_hash": checkpoint["model_hash"], "architecture": old["architecture"],
               "task_contracts": task_contracts,
               "stage2_encoder_sha256": old["stage2_encoder_sha256"], "fixed_final_epoch": 10}
    experiment.source_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(experiment.source_path, payload)
    manifest = {key: value for key, value in payload.items() if key != "model"}
    manifest["artifact_sha256"] = sha256_file(experiment.source_path)
    atomic_json(experiment.source_path.with_suffix(".json"), manifest)
    return manifest


def load_source(experiment):
    payload = torch.load(experiment.source_path, map_location="cpu", weights_only=False)
    manifest = json.loads(experiment.source_path.with_suffix(".json").read_text())
    old = load_transfer_source(experiment.source)
    task_contracts = json.loads(json.dumps([task.to_dict() for task in load_artifact_registry(experiment.source.stage2.data.artifacts_dir).tasks]))
    if (payload.get("kind") != SOURCE_KIND or manifest.get("kind") != SOURCE_KIND
        or payload.get("format_version") != 1 or payload.get("fixed_final_epoch") != 10
        or manifest.get("artifact_sha256") != sha256_file(experiment.source_path)
        or payload.get("model_hash") != tensor_state_hash("stage2-home-transfer.full-model.v1", payload["model"])
        or payload.get("source_training_identity") != old["training_identity"]
        or payload.get("stage2_encoder_sha256") != old["stage2_encoder_sha256"]
        or payload.get("architecture") != old["architecture"]
        or payload.get("task_contracts") != task_contracts
        or any(manifest.get(key) != value for key, value in json.loads(json.dumps({
            key: value for key, value in payload.items() if key != "model"
        })).items())):
        raise ValueError("Cross-domain source artifact mismatch")
    expected = semantic_identity("stage2.cross-domain-home-source", {
        "contract_version": 1, "source_training_identity": old["training_identity"]["hash"],
        "checkpoint_sha256": sha256_file(experiment.source.output_root / "stage2" / "checkpoint_epoch_00010.pt"),
        "blocks": payload["blocks"], "architecture": old["architecture"], "task_contracts": task_contracts, "epoch": 10,
    })
    require_compatible_identity(expected, payload["identity"], context="Cross-domain simulation source")
    for block, prefix in (("stage1", "backbone."), ("object_encoder", "object_encoder."),
                          ("simulation_home", "home."), ("atom_adapter", "atom_adapter.")):
        if payload["blocks"][block] != tensor_state_hash("cross-domain.source-block", {
            name: value for name, value in payload["model"].items() if name.startswith(prefix)
        }):
            raise ValueError("Cross-domain source block hash mismatch")
    return payload


def simulation_model(experiment, payload):
    registry = load_artifact_registry(experiment.source.stage2.data.artifacts_dir)
    model, loaded = build_model(experiment.source, registry)
    model.load_state_dict(payload["model"], strict=True)
    model.backbone.requires_grad_(False).eval()
    return model, loaded, registry


def replay_datasets(experiment):
    directory = experiment.source.stage2.data.artifacts_dir
    entities = Stage2EntityDataset(directory)
    registry = load_artifact_registry(directory)
    result = {task: Stage2TaskDataset(directory, task, "train") for task in registry.task_ids}
    for task, dataset in result.items():
        validate_runtime_task_contract(dataset, entities, loss_mode=experiment.source.stage2.loss.task_loss_modes.get(task, "element_mean"))
    return result, entities


@torch.no_grad()
def prepare(experiment):
    if (experiment.output_root / "replay" / "entities.pt").exists():
        raise FileExistsError("Replay cache already exists")
    source = load_source(experiment)
    config = experiment.stage3
    prepared_result = prepare_stage3(config)
    prepared = load_prepared_stage3(config)
    require_paired_data(experiment.source, prepared, config)
    model, loaded, registry = simulation_model(experiment, source)
    device = resolve_device(config.training.device)
    model.backbone.to(device).eval()
    datasets, entities = replay_datasets(experiment)
    atom_ids = set(datasets["simulation/partial_atomic_charge"].entity_indices.flatten().tolist())
    slots = torch.empty((len(entities), model.backbone.entity_dim), dtype=torch.float32)
    roles = torch.tensor([entry["role_id"] for entry in entities.entries], dtype=torch.long)
    atoms = {}
    packer = MultimodalPacker(loaded.vocabulary)
    progress = ProgressReporter()
    bar = progress.bar(total=len(entities), desc="Frozen replay entity cache", unit="entity")
    try:
        for start in range(0, len(entities), 128):
            ids = list(range(start, min(len(entities), start + 128)))
            batch = packer([entities[index] for index in ids]).to(device)
            encoded = model.backbone.encode_entity(batch)
            slots[ids] = encoded.entity_embedding.float().cpu()
            for local_id, entity_id in enumerate(ids):
                if entity_id in atom_ids:
                    atoms[entity_id] = encoded.atom_states[encoded.atom_batch == local_id].float().cpu()
            bar.update(len(ids))
    finally:
        bar.close()
    state = {"slots": slots, "roles": roles, **{f"atoms/{index}": value for index, value in atoms.items()}}
    data_metadata = json.loads((experiment.source.stage2.data.artifacts_dir / "metadata.json").read_text())
    payload = {"kind": "ilume_cross_domain_replay_cache", "source_identity": source["identity"]["hash"],
               "stage2_data_identity": metadata_identity(data_metadata, "data", context="Replay data")["hash"],
               "slots": slots, "roles": roles, "atoms": atoms,
               "state_hash": tensor_state_hash("cross-domain.replay-cache", state)}
    path = experiment.output_root / "replay" / "entities.pt"
    if path.exists():
        raise FileExistsError("Replay cache already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(path, payload)
    manifest = {key: value for key, value in payload.items() if key not in {"slots", "roles", "atoms"}}
    manifest["artifact_sha256"] = sha256_file(path)
    atomic_json(path.with_suffix(".json"), manifest)
    authority = semantic_identity("stage3.cross-domain-home-prepared.v1", {
        "base_prepared_identity": metadata_identity(prepared["metadata"], "prepared", context="Cross-domain prepared")["hash"],
        "source_identity": source["identity"]["hash"], "replay_cache_state_hash": payload["state_hash"],
        "replay_cache_sha256": manifest["artifact_sha256"],
    })
    atomic_json(experiment.output_root / "prepare" / "experiment.json", authority)
    return {"prepared": prepared_result, "replay": manifest, "identity": authority}


def load_cache(experiment, source):
    path = experiment.output_root / "replay" / "entities.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    manifest = json.loads(path.with_suffix(".json").read_text())
    metadata = json.loads((experiment.source.stage2.data.artifacts_dir / "metadata.json").read_text())
    state = {"slots": payload["slots"], "roles": payload["roles"],
             **{f"atoms/{index}": value for index, value in payload["atoms"].items()}}
    if (payload.get("kind") != "ilume_cross_domain_replay_cache"
        or payload.get("source_identity") != source["identity"]["hash"]
        or payload.get("stage2_data_identity") != metadata_identity(metadata, "data", context="Replay data")["hash"]
        or manifest.get("artifact_sha256") != sha256_file(path)
        or manifest.get("state_hash") != payload.get("state_hash")
        or payload.get("state_hash") != tensor_state_hash("cross-domain.replay-cache", state)):
        raise ValueError("Replay cache source/data/state mismatch")
    return payload, manifest


def validate_prepared(experiment, prepared, source, cache_manifest):
    expected = semantic_identity("stage3.cross-domain-home-prepared.v1", {
        "base_prepared_identity": metadata_identity(prepared["metadata"], "prepared", context="Cross-domain prepared")["hash"],
        "source_identity": source["identity"]["hash"], "replay_cache_state_hash": cache_manifest["state_hash"],
        "replay_cache_sha256": cache_manifest["artifact_sha256"],
    })
    authority = json.loads((experiment.output_root / "prepare" / "experiment.json").read_text())
    require_compatible_identity(expected, authority, context="Cross-domain prepared authority")


def build(experiment, prepared, source, fold, device):
    simulation, _, _ = simulation_model(experiment, source)
    config = experiment.stage3
    seed_everything(effective_training_seed(config) + fold)
    model = CrossDomainHoME(config.model, prepared["registry"], simulation.backbone.entity_dim,
                            group_configs=config.groups, task_configs=config.tasks,
                            task_private_recipes={task: config.resolved_private_recipe(task) for task in config.tasks if config.tasks[task].enabled},
                            object_encoder=simulation.object_encoder,
                            simulation_home=simulation.home, atom_adapter=simulation.atom_adapter).to(device)
    model.set_trainable_owners(set(model.parameter_ownership().values()))
    return model, ObjectPhase1Representations(model, prepared["slots"])


def resolved_plan(experiment, model, prepared, train, source, cache_manifest, fold):
    config = experiment.stage3
    tasks = tuple(train)
    plan = build_resolved_training_plan(config, fold, model, train, tasks, prepared,
                                       {"mode": "cross_domain_home"}, prepared["normalization"][f"fold{fold}"])
    phase1 = plan["phases"]["phase1"]
    total = phase1["epochs"] * phase1["steps_per_epoch"]
    source_tasks = sorted(experiment.source.stage2.loss.task_weights)
    events = replay_events(total)
    for owner in set(model.parameter_ownership().values()):
        if not owner.scope.startswith("SIM_"):
            continue
        updates = ([update for index, (update, _) in enumerate(events) if source_tasks[index % 9] == owner.owner_id]
                   if owner.scope == "SIM_PRIVATE" else list(range(1, total + 1)))
        phase1["owners"][owner.label] = {
            "nominal_lr": 1e-5, "terminal_lr": 1e-6, "nominal_epochs": phase1["epochs"],
            "effective_epochs": phase1["epochs"], "freeze_epoch": phase1["epochs"],
            "updates_per_epoch": phase1["steps_per_epoch"], "actual_update_budget": len(updates),
            "warmup_updates": math.ceil(0.05 * len(updates)), "update_steps": updates,
        }
    plan["cross_domain_home"] = {
        "contract_version": 1, "source_identity": source["identity"]["hash"],
        "source_artifact_sha256": sha256_file(experiment.source_path),
        "source_blocks": source["blocks"], "recipe": experiment.recipe,
        "task_contracts": source["task_contracts"],
        "replay_cache_sha256": cache_manifest["artifact_sha256"],
        "replay_cache_state_hash": cache_manifest["state_hash"],
        "replay_stage2_data_identity": cache_manifest["stage2_data_identity"],
        "simulation_groups": list(model.sim_groups), "simulation_group_mapping": "all_visible",
        "simulation_capacity": model.simulation_home.resolved_capacity_recipe(),
        "routing": "l1_l2_domain_families_then_global_group_private_v1",
        "replay_task_weights": dict(experiment.source.stage2.loss.task_weights),
        "replay_selection": "sorted_round_robin_independent_raw_cycles_v1",
        "replay_weight": "0.1*(1-(u-1)/(15K-1))", "simulation_private_transfer": False,
    }
    plan["format_version"] = 10
    return plan


def train_fold(experiment, fold, *, resume=False):
    config = experiment.stage3
    source = load_source(experiment)
    prepared = load_prepared_stage3(config)
    require_paired_data(experiment.source, prepared, config)
    cache, cache_manifest = load_cache(experiment, source)
    validate_prepared(experiment, prepared, source, cache_manifest)
    device = resolve_device(config.training.device)
    model, store = build(experiment, prepared, source, fold, device)
    validate_initial_object_embeddings(model, store, prepared)
    tasks = tuple(task for task, spec in prepared["registry"].items() if spec.enabled)
    train = {task: Stage3TaskDataset(config.data.artifacts_dir, fold, task, "train") for task in tasks}
    valid = {task: Stage3TaskDataset(config.data.artifacts_dir, fold, task, "valid") for task in tasks}
    plan = resolved_plan(experiment, model, prepared, train, source, cache_manifest, fold)
    datasets, _ = replay_datasets(experiment)
    phase1 = plan["phases"]["phase1"]
    replay = SimulationReplay(datasets, cache, seed=effective_training_seed(config), fold=fold,
                              total_updates=phase1["epochs"] * phase1["steps_per_epoch"],
                              weights=experiment.source.stage2.loss.task_weights)
    root = experiment.output_root / "train" / f"fold{fold}"
    if root.exists() and not resume:
        raise FileExistsError("Cross-domain fold output already exists")
    return run_three_phase_training(config=config, fold=fold, output_dir=root,
                                   resume_from=root if resume and root.exists() else None,
                                   model=model, registry=prepared["registry"], active=tasks,
                                   train_data=train, valid_data=valid, representations=store,
                                   normalizations=prepared["normalization"][f"fold{fold}"],
                                   plan=plan, device=device, progress=ProgressReporter(), phase1_extension=replay)


def evaluate(experiment, *, split, fold=None, predictions_dir=None):
    config = experiment.stage3
    source = load_source(experiment)
    _, cache_manifest = load_cache(experiment, source)
    validate_prepared(experiment, load_prepared_stage3(config), source, cache_manifest)

    def loader(requested_config, prepared, path, current_fold, epoch, device, *, taskwise_refined, three_phase_final):
        if requested_config != config or not three_phase_final or taskwise_refined:
            raise ValueError("Cross-domain evaluation requires its fixed final artifact")
        model, store = build(experiment, prepared, source, current_fold, device)
        train = {task: Stage3TaskDataset(config.data.artifacts_dir, current_fold, task, "train")
                 for task, spec in prepared["registry"].items() if spec.enabled}
        plan = resolved_plan(experiment, model, prepared, train, source, cache_manifest, current_fold)
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        manifest = json.loads(path.with_suffix(".json").read_text())
        if (artifact.get("kind") != FINAL_KIND or manifest.get("kind") != FINAL_KIND
            or artifact.get("format_version") != 1 or manifest.get("format_version") != 1
            or artifact.get("fold") != current_fold or manifest.get("fold") != current_fold
            or manifest.get("artifact_sha256") != sha256_file(path)
            or artifact.get("ownership_manifest") != model.ownership_manifest()
            or artifact.get("cross_domain_home") != plan["cross_domain_home"]
            or manifest.get("cross_domain_home") != plan["cross_domain_home"]
            or artifact.get("normalization") != prepared["normalization"][f"fold{current_fold}"]
            or artifact.get("model_state_hash") != tensor_state_hash("stage3.object-phase1-model-state.v1", artifact["model"])
            or manifest.get("model_state_hash") != artifact.get("model_state_hash")):
            raise ValueError("Cross-domain final artifact/manifest mismatch")
        expected = build_stage3_training_identity(plan)
        require_compatible_identity(expected, artifact.get("training_identity", {}), context="Cross-domain final")
        require_compatible_identity(expected, manifest.get("training_identity", {}), context="Cross-domain manifest")
        require_compatible_identity(expected, build_stage3_training_identity(artifact["resolved_training_plan"]), context="Cross-domain resolved plan")
        model.load_state_dict(artifact["model"], strict=True)
        encoder_hash = tensor_state_hash("stage3.object-phase1.final-encoder.v1", model.stage2_object_encoder.state_dict())
        if artifact.get("object_encoder_state_hash") != encoder_hash or manifest.get("object_encoder_state_hash") != encoder_hash:
            raise ValueError("Cross-domain final ObjectEncoder mismatch")
        store.freeze_after_phase1(model, artifact["phase1_model_state_hash"])
        if artifact.get("final_embedding_hash") != store.final_embedding_hash or manifest.get("final_embedding_hash") != store.final_embedding_hash:
            raise ValueError("Cross-domain final representation mismatch")
        model.set_trainable_owners(())
        return model.eval(), artifact, store

    result = evaluate_checkpoints(config, experiment.output_root / "train", split=split,
                                  ensemble_folds=split == "test", fold=fold, predictions_dir=predictions_dir,
                                  reporting_study_id="ilume-cross-domain-home-v1", model_loader=loader)
    result["reporting"]["model_display_name"] = "ILUME (cross-domain HoME sharing)"
    result["ablation"] = "stage2_stage3_cross_domain_home"
    return result


def evaluation_identity(experiment, *, split, fold=None):
    anchors = []
    for current_fold in ((fold,) if split == "valid" else range(1, 6)):
        path = experiment.checkpoint_dir / f"fold{current_fold}" / "three_phase_final.json"
        manifest = json.loads(path.read_text())
        if (manifest.get("kind") != FINAL_KIND or manifest.get("fold") != current_fold
            or manifest.get("artifact_sha256") != sha256_file(path.with_suffix(".pt"))):
            raise ValueError("Cross-domain evaluation anchor mismatch")
        anchors.append({"fold": current_fold, "training_identity": manifest["training_identity"]["hash"],
                        "model_state_hash": manifest["model_state_hash"], "manifest_sha256": sha256_file(path),
                        "artifact_sha256": manifest["artifact_sha256"]})
    return semantic_identity("stage3.cross-domain-home-evaluation.v1", {
        "split": split, "fold": fold, "anchors": anchors,
        "source_artifact_sha256": sha256_file(experiment.source_path), "recipe": experiment.recipe,
    })
