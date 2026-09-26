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

from common.identity import require_compatible_identity, semantic_identity, tensor_state_hash
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.progress import ProgressReporter
from common.training import capture_rng_state, cosine_warmup, resolve_device, restore_rng_state, seed_everything
from stage1.masking import MultimodalPacker
from stage1.model import load_stage1_model
from stage2.data import (
    Stage2BatchDescriptor, Stage2DeviceTaskData, Stage2EntityDataset,
    Stage2TaskDataset, epoch_batch_schedule, load_artifact_registry,
    pack_stage2_batch, task_batch_counts, validate_runtime_task_contract,
)
from stage2.identity import metadata_identity
from stage2.model import Stage2ObjectModel, molecule_equal_smooth_l1_loss
from stage2.runtime import configure_stage2_math
from stage2.train import export_stage2_encoder_artifact, task_compensation_scale

from .config import Experiment
from .contract import SOURCE_GROUPS, state_hash, transferable_state
from .model import SimulationHoME


STAGE2_HOME_CHECKPOINT_KIND = "ilume_stage2_home_transfer_checkpoint"
STAGE2_HOME_ARTIFACT_KIND = "ilume_stage2_home_transfer_shared"


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _model_hash(state: Mapping[str, torch.Tensor]) -> str:
    return tensor_state_hash("stage2-home-transfer.full-model.v1", state)


def training_identity(
    experiment: Experiment, data_identity: Mapping[str, Any],
    math_contract: Mapping[str, Any],
) -> dict[str, Any]:
    config = experiment.stage2
    return semantic_identity("stage2.home-transfer-training", {
        "contract_version": 1,
        "stage2_data_identity": data_identity["hash"],
        "stage1_checkpoint_sha256": sha256_file(config.initialization.checkpoint),
        "stage2_config": config.experiment_dict(),
        "stage3_model": asdict(experiment.stage3.model),
        "transfer_groups": {group: asdict(experiment.stage3.groups[group]) for group in ("thermophysical", "solvation")},
        "electronic_group": {
            "experts": experiment.stage3.groups["thermophysical"].experts,
            "expert_hidden_ratio": experiment.stage3.groups["thermophysical"].expert_hidden_ratio,
            "transferred": False,
        },
        "source_groups": SOURCE_GROUPS,
        "batch_size": config.training.batch_size,
        "microbatch_size": experiment.stage2_microbatch_size,
        "epochs": experiment.stage2_epochs,
        "loss": "physics_only_homemodel_v1",
        "backbone_frozen_epochs": 0,
        "scheduler": "stage2_cosine_warmup_v1",
        "math_contract": dict(math_contract),
    })


def build_model(experiment: Experiment, registry: Any) -> tuple[SimulationHoME, Any]:
    config = experiment.stage2
    seed_everything(config.data.seed)
    loaded = load_stage1_model(
        config.initialization.checkpoint, config.data.pretrain_artifacts_dir,
        device="cpu", backbone_dropout=0.0,
    )
    model = SimulationHoME(loaded.model, registry, experiment.stage3, config)
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

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="home-transfer-pack")
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
    root: Path, experiment: Experiment, model: SimulationHoME,
    registry: Any, identity: Mapping[str, Any], data_identity: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_path = root / "checkpoint_epoch_00010.pt"
    encoder_path = root / "stage2_encoder.pt"
    # The transient compatibility scaffold is never optimized and exports only encoding state.
    scaffold = Stage2ObjectModel(
        model.backbone, registry,
        object_layers=experiment.stage2.model.object_layers,
        object_ffn_dim=experiment.stage2.model.object_ffn_dim,
        dropout=experiment.stage2.model.dropout,
    )
    scaffold.object_encoder.load_state_dict(model.object_encoder.state_dict(), strict=True)
    export_stage2_encoder_artifact(
        encoder_path, model=scaffold, config=experiment.stage2, registry=registry,
        data_identity=dict(data_identity),
        provenance={
            "stage2_checkpoint_hash": sha256_file(checkpoint_path),
            "refinement_boundary_epoch": experiment.stage2_epochs,
            "home_transfer_identity": identity["hash"],
            "physics_only": True,
        },
    )
    compatible_encoder = torch.load(encoder_path, map_location="cpu", weights_only=False)
    shared = transferable_state(model.home)
    shared_hash = state_hash(shared)
    payload = {
        "kind": STAGE2_HOME_ARTIFACT_KIND,
        "format_version": 1,
        "training_identity": dict(identity),
        "stage2_encoder": encoder_path.name,
        "stage2_encoder_sha256": sha256_file(encoder_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "stage1_backbone": compatible_encoder["stage1_backbone"],
        "object_encoder": compatible_encoder["object_encoder"],
        "group_mapping": {
            "thermophysical": [task for task, group in SOURCE_GROUPS.items() if group == "thermophysical"],
            "solvation": [task for task, group in SOURCE_GROUPS.items() if group == "solvation"],
        },
        "architecture": {
            "stage3_model": asdict(experiment.stage3.model),
            "transfer_groups": ["solvation", "thermophysical"],
            "electronic_group": identity["payload"]["electronic_group"],
        },
        "shared_state": shared,
        "shared_state_hash": shared_hash,
        "encoder_state_hashes": {
            "stage1": compatible_encoder["state_hashes"]["stage1_backbone"],
            "object_encoder": compatible_encoder["state_hashes"]["object_encoder"],
        },
    }
    artifact = root / "stage2_home_transfer.pt"
    atomic_torch_save(artifact, payload)
    manifest = {
        "kind": STAGE2_HOME_ARTIFACT_KIND,
        "artifact": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "stage2_encoder_sha256": sha256_file(encoder_path),
        "training_identity": dict(identity),
        "shared_state_hash": shared_hash,
        "encoder_state_hashes": payload["encoder_state_hashes"],
        "fixed_final_epoch": 10,
    }
    atomic_json(root / "stage2_home_transfer.json", manifest)
    return manifest


def train_stage2_home(experiment: Experiment, output_dir: str | Path, *, resume: bool = False) -> dict[str, Any]:
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
        {"params": tuple(model.object_encoder.parameters()), "lr": config.training.object_encoder_learning_rate},
        {"params": model.home_parameters(), "lr": config.training.task_head_learning_rate},
    ]
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
    if output.exists() and not resume:
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
        if checkpoint.get("kind") != STAGE2_HOME_CHECKPOINT_KIND or checkpoint.get("training_identity") != identity:
            raise ValueError("Stage2-HoME resume identity mismatch")
        if checkpoint.get("model_hash") != _model_hash(checkpoint["model"]):
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
        if not (output / "stage2_home_transfer.json").is_file():
            return _export(output, experiment, model, registry, identity, data_identity)
        manifest = json.loads((output / "stage2_home_transfer.json").read_text(encoding="utf-8"))
        if (
            manifest.get("kind") != STAGE2_HOME_ARTIFACT_KIND
            or manifest.get("training_identity") != identity
            or manifest.get("artifact_sha256") != sha256_file(output / "stage2_home_transfer.pt")
            or manifest.get("stage2_encoder_sha256") != sha256_file(output / "stage2_encoder.pt")
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
            "kind": STAGE2_HOME_CHECKPOINT_KIND,
            "format_version": 1, "epoch": epoch, "updates": update,
            "training_identity": identity, "model": state,
            "model_hash": _model_hash(state),
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
        }
        with (output / "performance.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(performance, sort_keys=True) + "\n")
        print(
            f"Stage2-HoME epoch {epoch}: train={train_seconds:.1f}s "
            f"valid={validation_seconds:.1f}s checkpoint={performance['checkpoint_seconds']:.1f}s",
            flush=True,
        )
    return _export(output, experiment, model, registry, identity, data_identity)
