from __future__ import annotations

import json
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from common.identity import require_compatible_identity, semantic_hash, semantic_identity, tensor_state_hash
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.progress import ProgressReporter
from common.training import capture_rng_state, cosine_warmup, resolve_device, restore_rng_state, seed_everything
from stage1.masking import MultimodalPacker
from stage1.config import load_config as load_stage1_config
from stage1.descriptors import DescriptorSchema, rdkit_descriptor_names
from stage1.model import LoadedStage1Model, build_stage1_model, load_stage1_model
from stage1.tokenizer import SmilesTokenizer
from stage2.data import (
    Stage2BatchDescriptor, Stage2DeviceTaskData, Stage2EntityDataset,
    Stage2TaskDataset, epoch_batch_schedule, load_artifact_registry,
    pack_stage2_batch, task_batch_counts, validate_runtime_task_contract,
)
from stage2.identity import metadata_identity
from stage2.model import molecule_equal_smooth_l1_loss
from stage2.runtime import configure_stage2_math
from stage2.train import task_compensation_scale

from .home_config import HomeRecipe
from .home_contract import SOURCE_GROUPS, source_groups, state_hash, transferable_state
from .home_model import SimulationHoME
from .home_artifact import final_kind, full_owner_manifest, full_state_hash, load_home_final


STAGE2_HOME_CHECKPOINT_KIND = "ilume_stage2_home_checkpoint_v1"


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _model_hash(state: Mapping[str, torch.Tensor], *, entity_home: bool = False) -> str:
    return tensor_state_hash("stage2.entity-home.full-model.v4", state)


def model_contract_for_recipe(experiment: HomeRecipe) -> dict[str, Any]:
    from common.entity_inputs import ENTITY_INPUT_CONTRACT
    return {**ENTITY_INPUT_CONTRACT, "source_tasks": sorted(experiment.stage2.data.tasks)}


def training_identity(
    experiment: HomeRecipe, data_identity: Mapping[str, Any],
    math_contract: Mapping[str, Any],
) -> dict[str, Any]:
    config = experiment.stage2
    return semantic_identity("stage2.entity-home-training.v4" if config.is_entity_home else "stage2.home-training.v4" if experiment.freeze_stage1 else "stage2.home-training.v1", {
        "contract_version": 4 if config.is_entity_home else 4 if experiment.freeze_stage1 else 1,
        "stage2_data_identity": data_identity["hash"],
        **({"frozen_entity_cache": json.loads((config.data.artifacts_dir / "frozen_entities.json").read_text())} if experiment.freeze_stage1 else {}),
        "stage1_source": (
            {"checkpoint_sha256": sha256_file(config.initialization.checkpoint)}
            if experiment.initialization == "pretrained"
            else {
                "config_sha256": sha256_file(config.initialization.stage1_config),
                "random_seed": experiment.random_seed,
            }
        ),
        "stage2_config": config.experiment_dict(),
        "stage3_model": asdict(experiment.stage3.model),
        "transfer_groups": {group: asdict(experiment.stage3.groups[group]) for group in ("thermophysical", "solvation")},
        **({"electronic_group": {
            "experts": experiment.stage3.groups["thermophysical"].experts,
            "expert_hidden_ratio": experiment.stage3.groups["thermophysical"].expert_hidden_ratio,
            "transferred": False,
        }} if (not config.is_entity_home or "simulation/homo" in config.data.tasks) else {}),
        "source_groups": ({task: group for task, group in SOURCE_GROUPS.items() if task in config.data.tasks} if config.is_entity_home else SOURCE_GROUPS),
        **({"entity_input_contract": model_contract_for_recipe(experiment), "initialization": "seed_owner_v1"} if config.is_entity_home else {}),
        "batch_size": config.training.batch_size,
        "microbatch_size": experiment.stage2_microbatch_size,
        "epochs": experiment.stage2_epochs,
        "loss": "physics_only_homemodel_v1",
        "backbone_frozen_epochs": "permanent" if experiment.freeze_stage1 else 0,
        "scheduler": "stage2_cosine_warmup_v1",
        "math_contract": dict(math_contract),
    })


def resolve_training_identity(recipe: HomeRecipe, math_contract: Mapping[str, Any]) -> dict[str, Any]:
    metadata = json.loads(
        (recipe.stage2.data.artifacts_dir / "metadata.json").read_text(encoding="utf-8")
    )
    data_identity = metadata_identity(metadata, "data", context="Stage 2 HoME data")
    return training_identity(recipe, data_identity, math_contract)


def load_backbone(experiment: HomeRecipe):
    config = experiment.stage2
    seed_everything(config.data.seed if experiment.random_seed is None else experiment.random_seed)
    if experiment.initialization == "pretrained":
        loaded = load_stage1_model(
            config.initialization.checkpoint, config.data.pretrain_artifacts_dir,
            device="cpu", backbone_dropout=0.0,
        )
    else:
        from dataclasses import replace
        from stage1.identity import metadata_identity as stage1_metadata_identity

        assert config.initialization.stage1_config is not None
        feature_root = config.data.pretrain_artifacts_dir
        source_config = load_stage1_config(config.initialization.stage1_config)
        source_config = replace(source_config, model=replace(source_config.model, dropout=0.0))
        vocabulary = SmilesTokenizer.load(feature_root / "tokenizer.json")
        schema = DescriptorSchema.load(
            feature_root / "descriptor_schema.json", expected_raw_names=rdkit_descriptor_names()
        )
        feature_metadata = json.loads((feature_root / "metadata.json").read_text(encoding="utf-8"))
        loaded = LoadedStage1Model(
            build_stage1_model(source_config, vocabulary, schema, encoder_only=source_config.is_dual_view), source_config,
            vocabulary, stage1_metadata_identity(
                feature_metadata, "feature", context="Stage 1 feature artifact"
            )["hash"],
        )
    return loaded


def build_model(experiment: HomeRecipe, registry: Any) -> tuple[SimulationHoME, Any]:
    loaded = load_backbone(experiment)
    if loaded.config.is_dual_view != experiment.freeze_stage1:
        raise ValueError("Stage2/Stage1 representation family mismatch: v4 requires permanent Stage1 freeze")
    model = SimulationHoME(loaded.model, registry, experiment.stage3, experiment.stage2)
    return model, loaded


def _loss_for_micro(
    task: str, predictions: torch.Tensor, packed: Any,
    task_data: Stage2DeviceTaskData, full_indices: torch.Tensor,
) -> torch.Tensor:
    if task == "simulation/partial_atomic_charge":
        atom = packed.atom_targets
        if atom is None:
            raise ValueError("Missing Stage2-HoME atom labels")
        return molecule_equal_smooth_l1_loss(
            predictions, atom.values, atom.mask, atom.atom_sample_indices,
            len(packed.row_indices),
        ) * (len(packed.row_indices) / len(full_indices))
    targets = task_data.targets[packed.row_indices]
    mask = task_data.target_mask[packed.row_indices]
    if task == "simulation/simulated_qm_elec_hf":
        full_mask = task_data.target_mask[full_indices]
        counts = full_mask.sum(dim=0)
        valid = counts > 0
        values = F.smooth_l1_loss(predictions, targets, reduction="none") * mask
        return ((values.sum(dim=0) / counts.clamp_min(1)) * valid).sum() / valid.sum()
    # Ordinary-task masks are checked once before training, on the CPU datasets.
    return F.smooth_l1_loss(predictions, targets, reduction="sum") / task_data.targets[full_indices].numel()


def simulation_loss(
    task: str, predictions: torch.Tensor, packed: Any,
    task_data: Stage2DeviceTaskData, full_indices: torch.Tensor,
) -> torch.Tensor:
    return _loss_for_micro(task, predictions, packed, task_data, full_indices)


@contextmanager
def _prefetched_batches(schedule, datasets, entities, packer, microbatch_size, *, pin_memory):
    """Pack at most two pending logical batches, in schedule order, on one CPU worker."""
    def pack(descriptor):
        started = time.perf_counter()
        batches = tuple(
            pack_stage2_batch(
                Stage2BatchDescriptor(descriptor.task, indices), datasets, entities,
                packer, needs_entities=True, include_raw_atom_targets=False,
                pin_memory=pin_memory,
            )
            for indices in descriptor.indices.split(microbatch_size)
        )
        return batches, time.perf_counter() - started

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage2-home-pack")
    pending = deque()
    descriptors = iter(schedule)

    def submit():
        descriptor = next(descriptors, None)
        if descriptor is not None:
            pending.append((descriptor, executor.submit(pack, descriptor)))

    def consume():
        while pending:
            descriptor, future = pending.popleft()
            started = time.perf_counter()
            batches, packing_seconds = future.result()
            wait_seconds = time.perf_counter() - started
            submit()
            yield descriptor, batches, packing_seconds, wait_seconds

    try:
        submit()
        submit()
        yield consume()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _train_batch(model, descriptor, packed_batches, task_data, device, optimizer,
                 scheduler, parameters, config, compensation):
    full_indices = descriptor.indices.to(device)
    optimizer.zero_grad(set_to_none=True)
    batch_loss = torch.zeros((), dtype=torch.float64, device=device)
    for cpu_batch in packed_batches:
        packed = cpu_batch.to(device, non_blocking=device.type == "cuda")
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=config.training.amp_dtype == "bf16",
        ):
            predictions = model.predict(descriptor.task, packed, task_data)
            loss = compensation * _loss_for_micro(
                descriptor.task, predictions, packed, task_data, full_indices,
            )
        loss.backward()
        batch_loss += loss.detach().double()
    # Read the device only at the logical-batch boundary, before any update.
    value = float(batch_loss)
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite Stage2-HoME loss: {descriptor.task}")
    torch.nn.utils.clip_grad_norm_(parameters, config.training.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return value


def _check_history(root: Path, completed_epoch: int) -> None:
    path = root / "metrics.jsonl"
    if not path.is_file():
        raise ValueError("Stage2-HoME resume requires continuous metrics history")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if [row.get("epoch") for row in rows] != list(range(1, completed_epoch + 1)):
        raise ValueError("Stage2-HoME history/checkpoint mismatch")


@torch.no_grad()
def _validate(
    model: SimulationHoME, datasets: Mapping[str, Stage2TaskDataset],
    task_data: Mapping[str, Stage2DeviceTaskData], entities: Stage2EntityDataset,
    packer: MultimodalPacker, device: torch.device, microbatch_size: int,
) -> dict[str, float]:
    model.eval()
    result: dict[str, float] = {}
    for task, dataset in datasets.items():
        numerator = 0.0
        denominator = 0.0
        target_sums = torch.zeros(len(dataset.spec.target_columns), dtype=torch.float64, device=device)
        target_counts = torch.zeros_like(target_sums)
        for start in range(0, len(dataset), microbatch_size):
            indices = torch.arange(start, min(len(dataset), start + microbatch_size))
            packed = pack_stage2_batch(
                Stage2BatchDescriptor(task, indices), {task: dataset}, entities,
                packer, needs_entities=True, include_raw_atom_targets=False,
                pin_memory=False,
            ).to(device, non_blocking=False)
            predictions = model.predict(task, packed, task_data[task])
            if dataset.spec.target_level == "atom":
                atom = packed.atom_targets
                assert atom is not None
                errors = (predictions - atom.values).abs() * atom.mask
                molecule_errors = torch.zeros(len(packed.row_indices), device=device).index_add_(
                    0, atom.atom_sample_indices, errors.float(),
                )
                molecule_counts = torch.zeros_like(molecule_errors).index_add_(
                    0, atom.atom_sample_indices, atom.mask.float(),
                )
                numerator += float((molecule_errors / molecule_counts.clamp_min(1)).sum())
                denominator += len(packed.row_indices)
            else:
                target = task_data[task].targets[packed.row_indices]
                mask = task_data[task].target_mask[packed.row_indices]
                errors = (predictions - target).abs() * mask
                target_sums += errors.double().sum(dim=0)
                target_counts += mask.double().sum(dim=0)
        if dataset.spec.target_level == "object":
            valid_targets = target_counts > 0
            if not bool(valid_targets.any()):
                raise ValueError(f"Stage2-HoME validation task is empty: {task}")
            result[task] = float((target_sums[valid_targets] / target_counts[valid_targets]).mean())
            continue
        if denominator <= 0:
            raise ValueError(f"Stage2-HoME validation task is empty: {task}")
        result[task] = numerator / denominator
    model.train()
    return result


def _export(
    root: Path, experiment: HomeRecipe, model: SimulationHoME,
    registry: Any, identity: Mapping[str, Any], data_identity: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_path = root / f"checkpoint_epoch_{experiment.stage2_epochs:05d}.pt"
    from stage1.identity import metadata_identity as feature_metadata_identity
    feature_root = experiment.stage2.data.pretrain_artifacts_dir
    feature_metadata = json.loads((feature_root / "metadata.json").read_text())
    features = {name: json.loads((feature_root / name).read_text()) for name in (
        "tokenizer.json", "descriptor_schema.json", "descriptor_scaler.json")}
    stage1_state = _cpu_state(model.backbone)
    stage1_hash = tensor_state_hash("stage1.encoding-state", stage1_state)
    shared = transferable_state(model.home)
    shared_hash = state_hash(shared)
    full_state = _cpu_state(model)
    owner_manifest = full_owner_manifest(model)
    data_metadata = json.loads((experiment.stage2.data.artifacts_dir / "metadata.json").read_text(encoding="utf-8"))
    payload = {
        "kind": final_kind(experiment),
        "format_version": 4,
        "training_identity": dict(identity),
        "stage2_data_identity": dict(data_identity),
        "recipe": experiment.to_dict(),
        "registry": registry.snapshot(),
        "registry_hash": registry.registry_hash,
        "catalog_sha256": registry.catalog_sha256,
        "scalers": data_metadata["scalers"],
        "scalers_hash": semantic_hash("stage2.home.scalers.v1", data_metadata["scalers"]),
        "stage1_config": model.backbone.config.to_dict(),
        "stage1_feature_identity": dict(feature_metadata_identity(feature_metadata, "feature", context="Stage1 features")),
        "stage1_encoding_contract": {**model.model_contract, "feature_generation_contract": feature_metadata["feature_generation_contract"]},
        "feature_artifacts": features,
        "feature_artifacts_hash": semantic_hash("stage2.home.feature-artifacts.v1", features),
        "full_model_state": full_state,
        "full_model_state_hash": full_state_hash(full_state, version=5 if experiment.stage2.is_entity_home else None),
        "owner_manifest": owner_manifest,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "stage1_backbone": stage1_state,
        "group_mapping": {
            "thermophysical": [task for task, group in source_groups(registry).items() if group == "thermophysical"],
            "solvation": [task for task, group in source_groups(registry).items() if group == "solvation"],
        },
        "architecture": {
            "stage3_model": asdict(experiment.stage3.model),
            "transfer_groups": ["solvation", "thermophysical"],
            **({"electronic_group": identity["payload"]["electronic_group"]} if "electronic_group" in identity["payload"] else {}),
        },
        "shared_state": shared,
        "shared_state_hash": shared_hash,
        "encoder_state_hashes": {
            "stage1": stage1_hash,
        },
    }
    artifact = root / "stage2_final.pt"
    atomic_torch_save(artifact, payload)
    manifest = {
        "kind": final_kind(experiment),
        "artifact": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "feature_artifacts_hash": payload["feature_artifacts_hash"],
        "scalers_hash": payload["scalers_hash"],
        "training_identity": dict(identity),
        "full_model_state_hash": payload["full_model_state_hash"],
        "owner_manifest": owner_manifest,
        "registry_hash": registry.registry_hash,
        "stage2_data_identity": dict(data_identity),
        "shared_state_hash": shared_hash,
        "encoder_state_hashes": payload["encoder_state_hashes"],
        "fixed_final_epoch": experiment.stage2_epochs,
    }
    atomic_json(root / "stage2_final.json", manifest)
    return manifest


def train_stage2_home(experiment: HomeRecipe, output_dir: str | Path, *, resume: bool = False) -> dict[str, Any]:
    config = experiment.stage2
    device = resolve_device(config.training.device)
    math_contract = configure_stage2_math(device)
    if config.training.amp_dtype == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Stage2-HoME BF16 requires CUDA BF16 support")
    registry = load_artifact_registry(config.data.artifacts_dir)
    config.validate_registry(registry)
    entities = Stage2EntityDataset(config.data.artifacts_dir)
    train = {task: Stage2TaskDataset(config.data.artifacts_dir, task, "train") for task in registry.task_ids}
    valid = {task: Stage2TaskDataset(config.data.artifacts_dir, task, "valid") for task in registry.task_ids}
    for task in registry.task_ids:
        mode = config.loss.task_loss_modes.get(task, "element_mean")
        validate_runtime_task_contract(train[task], entities, loss_mode=mode)
        validate_runtime_task_contract(valid[task], entities, loss_mode=mode)
        if task not in {"simulation/partial_atomic_charge", "simulation/simulated_qm_elec_hf"}:
            if not bool(train[task].target_mask.all()) or not bool(valid[task].target_mask.all()):
                raise ValueError(f"Stage2-HoME ordinary task has missing labels: {task}")
    model, loaded = build_model(experiment, registry)
    if loaded.config.is_dual_view:
        from .entity_cache import validate_frozen_entity_source
        validate_frozen_entity_source(entities, model.backbone)
    model.to(device)
    packer = MultimodalPacker(loaded.vocabulary)
    train_device = {task: Stage2DeviceTaskData.from_dataset(data, device) for task, data in train.items()}
    valid_device = {task: Stage2DeviceTaskData.from_dataset(data, device) for task, data in valid.items()}
    batches = task_batch_counts(train, config.training.batch_size)
    steps_per_epoch = sum(batches.values())
    total_steps = steps_per_epoch * experiment.stage2_epochs
    metadata = json.loads((config.data.artifacts_dir / "metadata.json").read_text(encoding="utf-8"))
    data_identity = dict(metadata_identity(metadata, "data", context="Stage2-HoME data"))
    identity = training_identity(experiment, data_identity, math_contract)
    groups = [
        {"params": model.backbone_parameters(), "lr": config.training.backbone_learning_rate},
        {"params": model.home_parameters(), "lr": config.training.task_head_learning_rate},
    ]
    if loaded.config.is_dual_view:
        groups = [group for group in groups if len(group["params"])]
    optimizer = torch.optim.AdamW(
        groups, weight_decay=config.training.weight_decay,
        fused=device.type == "cuda", foreach=False if device.type != "cuda" else None,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup(step, total_steps, config.training.warmup_fraction),
    )
    weights = config.normalized_task_weights(registry)
    output = Path(output_dir)
    if output.exists() and not resume and any(output.glob("checkpoint_epoch_*.pt")):
        raise FileExistsError(f"Stage2-HoME output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_files = sorted(output.glob("checkpoint_epoch_*.pt"))
    start_epoch = 1
    update = 0
    if resume:
        if not checkpoint_files:
            raise ValueError("Stage2-HoME resume has no epoch checkpoint")
        checkpoint = torch.load(checkpoint_files[-1], map_location="cpu", weights_only=False)
        epoch = int(checkpoint.get("epoch", -1))
        if checkpoint_files != [output / f"checkpoint_epoch_{index:05d}.pt" for index in range(1, epoch + 1)]:
            raise ValueError("Stage2-HoME checkpoint sequence is incomplete")
        _check_history(output, epoch)
        if (checkpoint.get("kind") != ("ilume_stage2_entity_home_checkpoint_v4" if config.is_entity_home else "ilume_stage2_home_checkpoint_v4" if experiment.freeze_stage1 else STAGE2_HOME_CHECKPOINT_KIND)
            or checkpoint.get("format_version") != (4 if config.is_entity_home else 4 if experiment.freeze_stage1 else 1)
            or checkpoint.get("training_identity") != identity):
            raise ValueError("Stage2-HoME resume identity mismatch")
        if config.is_entity_home and checkpoint.get("owner_manifest") != full_owner_manifest(model):
            raise ValueError("Stage2-HoME resume owner manifest mismatch")
        if checkpoint.get("model_hash") != _model_hash(checkpoint["model"], entity_home=config.is_entity_home):
            raise ValueError("Stage2-HoME resume model hash mismatch")
        history_tail = json.loads((output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        if checkpoint.get("history_tail") != history_tail:
            raise ValueError("Stage2-HoME resume history tail mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng_state(checkpoint["rng"])
        update = int(checkpoint["updates"])
        if update != epoch * steps_per_epoch or scheduler.last_epoch != update:
            raise ValueError("Stage2-HoME resume scheduler/update mismatch")
        start_epoch = epoch + 1
    if start_epoch > experiment.stage2_epochs:
        if not (output / "stage2_final.json").is_file():
            return _export(output, experiment, model, registry, identity, data_identity)
        manifest = json.loads((output / "stage2_final.json").read_text(encoding="utf-8"))
        payload, _, _ = load_home_final(output / "stage2_final.pt")
        if (
            payload["training_identity"]["hash"] != identity["hash"]
        ):
            raise ValueError("Completed Stage2-HoME artifact is corrupt")
        return manifest
    reporter = ProgressReporter()
    parameters = tuple(parameter for group in groups for parameter in group["params"])
    for epoch in range(start_epoch, experiment.stage2_epochs + 1):
        model.train()
        total_loss = 0.0
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        task_timing = {
            task: {"packing_seconds": 0.0, "packing_wait_seconds": 0.0,
                   "train_seconds": 0.0, "batches": 0, "rows": 0}
            for task in train
        }
        schedule = epoch_batch_schedule(train, config.training.batch_size, seed=config.data.seed, epoch=epoch)
        progress = reporter.bar(
            total=len(schedule), desc=f"Stage2-HoME epoch {epoch}", unit="batch"
        )
        try:
            with _prefetched_batches(
                schedule, train, entities, packer, experiment.stage2_microbatch_size,
                pin_memory=device.type == "cuda",
            ) as prepared_batches:
                for descriptor, packed_batches, packing_seconds, wait_seconds in prepared_batches:
                    task = descriptor.task
                    started = time.perf_counter()
                    compensation = task_compensation_scale(
                        weights[task], steps_per_epoch, len(descriptor.indices), len(train[task]),
                    )
                    batch_loss = _train_batch(
                        model, descriptor, packed_batches, train_device[task], device,
                        optimizer, scheduler, parameters, config, compensation,
                    )
                    update += 1
                    total_loss += batch_loss
                    timing = task_timing[task]
                    timing["packing_seconds"] += packing_seconds
                    timing["packing_wait_seconds"] += wait_seconds
                    timing["train_seconds"] += time.perf_counter() - started
                    timing["batches"] += 1
                    timing["rows"] += len(descriptor.indices)
                    progress.set_postfix_str(f"task={task.rsplit('/', 1)[-1]} loss={batch_loss:.4f}")
                    progress.update(1)
        finally:
            progress.close()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - epoch_started
        peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        validation_started = time.perf_counter()
        validation = _validate(model, valid, valid_device, entities, packer, device, config.training.batch_size)
        validation_seconds = time.perf_counter() - validation_started
        checkpoint_started = time.perf_counter()
        state = _cpu_state(model)
        row = {
            "epoch": epoch, "optimizer_updates": update,
            "train_weighted_loss": total_loss / steps_per_epoch,
            "validation_normalized_mae": validation,
        }
        checkpoint = {
            "kind": "ilume_stage2_entity_home_checkpoint_v4" if config.is_entity_home else "ilume_stage2_home_checkpoint_v4" if experiment.freeze_stage1 else STAGE2_HOME_CHECKPOINT_KIND,
            "format_version": 4 if config.is_entity_home else 4 if experiment.freeze_stage1 else 1, "epoch": epoch, "updates": update,
            "training_identity": identity, "model": state,
            **({"owner_manifest": full_owner_manifest(model)} if config.is_entity_home else {}),
            "model_hash": _model_hash(state, entity_home=config.is_entity_home),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": capture_rng_state(), "history_tail": row,
        }
        atomic_torch_save(output / f"checkpoint_epoch_{epoch:05d}.pt", checkpoint)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        performance = {
            "epoch": epoch, "microbatch_size": experiment.stage2_microbatch_size,
            "train_seconds": train_seconds, "validation_seconds": validation_seconds,
            "checkpoint_seconds": time.perf_counter() - checkpoint_started,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "train_peak_allocated_bytes": peak_memory, "tasks": task_timing,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "home_parameter_count": sum(p.numel() for p in model.home.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        with (output / "performance.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(performance, sort_keys=True) + "\n")
        print(
            f"Stage2-HoME epoch {epoch}: train={train_seconds:.1f}s "
            f"valid={validation_seconds:.1f}s checkpoint={performance['checkpoint_seconds']:.1f}s",
            flush=True,
        )
    return _export(output, experiment, model, registry, identity, data_identity)
