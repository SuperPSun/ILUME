from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from .config import PretrainConfig, _config_from_checkpoint_dict
from .data import PreparedCorpusDataset
from .masking import MultimodalMasker, MultimodalPacker
from .model import MultimodalPretrainModel
from common.io import atomic_torch_save, sha256_file
from common.progress import ProgressReporter, loss_postfix
from common.training import (
    canonical_json_sha256,
    capture_rng_state,
    cosine_warmup,
    resolve_device,
    restore_rng_state,
    seed_everything,
)
from .sampler import RoleBalancedSampler, coverage_epoch_plan
from .tokenizer import SmilesTokenizer


def _config_hash(config: PretrainConfig) -> str:
    return canonical_json_sha256(config.to_dict())


def _save_checkpoint(
    path: Path,
    *,
    model: MultimodalPretrainModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    completed_epochs: int,
    global_step: int,
    micro_step: int,
    sampler: RoleBalancedSampler,
    steps_per_epoch: int,
    draws_per_epoch: int,
    config: PretrainConfig,
    config_hash: str,
    artifact_hash: str,
    source_hashes: dict[str, str],
) -> None:
    payload = {
        "format_version": 3,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "micro_step": micro_step,
        "steps_per_epoch": steps_per_epoch,
        "draws_per_epoch": draws_per_epoch,
        "sampler": sampler.state_dict(start_offset=draws_per_epoch),
        "rng": capture_rng_state(),
        "config": config.to_dict(),
        "config_hash": config_hash,
        "artifact_hash": artifact_hash,
        "source_hashes": source_hashes,
    }
    atomic_torch_save(path, payload)


@torch.no_grad()
def _validate(
    model: MultimodalPretrainModel,
    dataset: PreparedCorpusDataset,
    vocabulary: SmilesTokenizer,
    config: PretrainConfig,
    device: torch.device,
) -> dict[str, float]:
    def evaluate(indices: list[int] | None, prefix: str) -> dict[str, float]:
        selected = dataset if indices is None else Subset(dataset, indices)
        loader = DataLoader(
            selected,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            collate_fn=MultimodalPacker(vocabulary),
        )
        masker = MultimodalMasker(
            vocabulary, config.masking, config.data.seed + 100000
        )
        totals: dict[str, float] = {}
        batches = 0
        for batch_index, packed in enumerate(loader):
            if batch_index >= config.training.validation_batches:
                break
            batch = masker.apply(packed, 0, 1, evaluation=True).to(device)
            output = model(batch)
            values = {
                "loss": output.loss,
                **{f"loss_{key}": value for key, value in output.losses.items()},
            }
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            batches += 1
        if batches == 0:
            return {}
        return {
            f"valid_{prefix}{name}": value / batches
            for name, value in totals.items()
        }

    model.eval()
    metrics = evaluate(None, "")
    role_names = ("cation", "anion", "molecule")
    for role_id, role_name in enumerate(role_names):
        indices = [
            index for index, value in enumerate(dataset.role_ids) if value == role_id
        ]
        metrics.update(evaluate(indices, f"{role_name}_"))
    model.train()
    return metrics


def run_training(
    config: PretrainConfig,
    *,
    output_dir: str | Path,
    resume_from: str | Path | None = None,
) -> list[dict[str, float | int | str]]:
    config.validate()
    if not config.sampling.require_full_coverage:
        raise ValueError(
            "Epoch training requires sampling.require_full_coverage=true"
        )
    seed_everything(config.data.seed)

    artifact_dir = config.data.artifacts_dir
    artifact_metadata = json.loads(
        (artifact_dir / "metadata.json").read_text(encoding="utf-8")
    )
    train_dataset = PreparedCorpusDataset(
        artifact_dir, "train", config.data.shard_cache_size
    )
    valid_dataset = PreparedCorpusDataset(
        artifact_dir, "valid", config.data.shard_cache_size
    )
    vocabulary = SmilesTokenizer.load(artifact_dir / "tokenizer.json")
    device = resolve_device(config.training.device)
    model = MultimodalPretrainModel(
        config, vocabulary, train_dataset.descriptor_schema
    ).to(device)

    role_counts = tuple(train_dataset.role_ids.count(role) for role in range(3))
    epoch_plan = coverage_epoch_plan(
        role_counts,
        config.training.batch_size,
        config.training.gradient_accumulation_steps,
        config.sampling.role_probabilities,
    )
    total_steps = config.training.epochs * epoch_plan.steps_per_epoch
    max_micro_steps = (
        total_steps * config.training.gradient_accumulation_steps
    )
    sampler = RoleBalancedSampler(
        train_dataset.role_ids,
        num_samples=epoch_plan.draws_per_epoch,
        role_probabilities=config.sampling.role_probabilities,
        seed=config.data.seed,
        require_full_coverage=True,
        shard_ids=[entry["shard"] for entry in train_dataset.entries],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup(
            step, total_steps, config.training.warmup_fraction
        ),
    )
    fp16 = config.training.amp_dtype == "fp16" and device.type == "cuda"
    amp_enabled = config.training.amp_dtype != "none" and device.type == "cuda"
    amp_dtype = torch.float16 if fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    current_config_hash = _config_hash(config)
    artifact_hash = sha256_file(artifact_dir / "metadata.json")
    completed_epochs = 0
    global_step = 0
    micro_step = 0
    if resume_from is not None:
        checkpoint_payload = torch.load(
            Path(resume_from),
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_version = checkpoint_payload.get("format_version")
        if checkpoint_version != 3:
            if checkpoint_version == 2:
                raise ValueError(
                    "Step-based checkpoint format v2 is incompatible with the "
                    "epoch trainer; start a new run"
                )
            raise ValueError("Unsupported checkpoint format")
        checkpoint_config = _config_from_checkpoint_dict(checkpoint_payload["config"])
        if checkpoint_config.to_dict() != config.to_dict():
            raise ValueError("Checkpoint config hash does not match the current config")
        if checkpoint_payload["artifact_hash"] != artifact_hash:
            raise ValueError("Checkpoint corpus artifact hash does not match")
        model.load_state_dict(checkpoint_payload["model"])
        optimizer.load_state_dict(checkpoint_payload["optimizer"])
        scheduler.load_state_dict(checkpoint_payload["scheduler"])
        scaler.load_state_dict(checkpoint_payload["scaler"])
        completed_epochs = int(checkpoint_payload["completed_epochs"])
        global_step = int(checkpoint_payload["global_step"])
        micro_step = int(checkpoint_payload["micro_step"])
        if int(checkpoint_payload["steps_per_epoch"]) != epoch_plan.steps_per_epoch:
            raise ValueError("Checkpoint steps_per_epoch does not match the corpus")
        if int(checkpoint_payload["draws_per_epoch"]) != epoch_plan.draws_per_epoch:
            raise ValueError("Checkpoint draws_per_epoch does not match the corpus")
        expected_global_step = completed_epochs * epoch_plan.steps_per_epoch
        if global_step != expected_global_step:
            raise ValueError("Checkpoint global_step does not match completed epochs")
        expected_micro_step = (
            global_step * config.training.gradient_accumulation_steps
        )
        if micro_step != expected_micro_step:
            raise ValueError("Checkpoint micro_step does not match global_step")
        sampler.load_state_dict(checkpoint_payload["sampler"])
        if sampler.start_offset != epoch_plan.draws_per_epoch:
            raise ValueError("Checkpoint sampler cursor is not at an epoch boundary")
        expected_sampler_epoch = completed_epochs - 1
        if sampler.epoch != expected_sampler_epoch:
            raise ValueError("Checkpoint sampler epoch does not match completed epochs")
        restore_rng_state(checkpoint_payload["rng"])

    masker = MultimodalMasker(vocabulary, config.masking, config.data.seed)
    metrics_path = output_dir / "metrics.jsonl"
    results: list[dict[str, float | int | str]] = []
    reporter = ProgressReporter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    latest_valid_loss: float | None = None
    for epoch_index in range(completed_epochs, config.training.epochs):
        sampler.set_epoch(epoch_index)
        sampler.set_start_offset(0)
        loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=sampler,
            num_workers=config.training.num_workers,
            collate_fn=MultimodalPacker(vocabulary),
            drop_last=True,
        )
        epoch_step = 0
        epoch_number = epoch_index + 1
        epoch_description = f"Epoch {epoch_number}/{config.training.epochs}"
        with reporter.bar(
            total=epoch_plan.steps_per_epoch,
            desc=epoch_description,
            unit="step",
        ) as progress:
            for packed in loader:
                batch = masker.apply(
                    packed, micro_step, max_micro_steps
                ).to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    output = model(batch)
                    scaled_loss = (
                        output.loss
                        / config.training.gradient_accumulation_steps
                    )
                if not torch.isfinite(output.loss):
                    raise RuntimeError(
                        f"Non-finite total loss at micro step {micro_step}"
                    )
                scaler.scale(scaled_loss).backward()
                micro_step += 1
                if micro_step % config.training.gradient_accumulation_steps:
                    continue

                if config.training.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.training.max_grad_norm
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                epoch_step += 1
                result: dict[str, float | int | str] = {
                    "epoch": epoch_number,
                    "epoch_step": epoch_step,
                    "global_step": global_step,
                    "micro_step": micro_step,
                    "device": str(device),
                    "loss": float(output.loss.detach().cpu()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                result.update(
                    {
                        f"loss_{name}": float(value.detach().cpu())
                        for name, value in output.losses.items()
                    }
                )
                epoch_finished = epoch_step == epoch_plan.steps_per_epoch
                should_validate = (
                    epoch_finished
                    and config.training.validation_interval_epochs > 0
                    and epoch_number
                    % config.training.validation_interval_epochs
                    == 0
                )
                if should_validate:
                    progress.set_description_str("Validating")
                    result.update(
                        _validate(
                            model,
                            valid_dataset,
                            vocabulary,
                            config,
                            device,
                        )
                    )
                    latest_valid_loss = float(result["valid_loss"])
                    progress.set_description_str(epoch_description)
                serialized = json.dumps(result, sort_keys=True)
                reporter.emit_json(result)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(serialized + "\n")
                results.append(result)
                progress.set_postfix(
                    loss_postfix(
                        result,
                        include_learning_rate=True,
                        valid_loss=latest_valid_loss,
                    ),
                    refresh=False,
                )
                progress.update(1)

            if epoch_step != epoch_plan.steps_per_epoch:
                raise RuntimeError(
                    f"Epoch {epoch_number} stopped at optimizer step "
                    f"{epoch_step}; expected {epoch_plan.steps_per_epoch}"
                )
            completed_epochs = epoch_number
            interval = config.training.save_every_n_epochs
            should_checkpoint = (
                interval not in {None, 0}
                and completed_epochs
                % interval == 0
            ) or completed_epochs == config.training.epochs
            if should_checkpoint:
                progress.set_description_str("Checkpointing")
                checkpoint_path = (
                    output_dir
                    / f"checkpoint_epoch_{completed_epochs:05d}.pt"
                )
                _save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    completed_epochs=completed_epochs,
                    global_step=global_step,
                    micro_step=micro_step,
                    sampler=sampler,
                    steps_per_epoch=epoch_plan.steps_per_epoch,
                    draws_per_epoch=epoch_plan.draws_per_epoch,
                    config=config,
                    config_hash=current_config_hash,
                    artifact_hash=artifact_hash,
                    source_hashes=artifact_metadata.get("source_hashes", {}),
                )
                _save_checkpoint(
                    output_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    completed_epochs=completed_epochs,
                    global_step=global_step,
                    micro_step=micro_step,
                    sampler=sampler,
                    steps_per_epoch=epoch_plan.steps_per_epoch,
                    draws_per_epoch=epoch_plan.draws_per_epoch,
                    config=config,
                    config_hash=current_config_hash,
                    artifact_hash=artifact_hash,
                    source_hashes=artifact_metadata.get("source_hashes", {}),
                )
                progress.set_description_str(epoch_description)

    if completed_epochs != config.training.epochs:
        raise RuntimeError(
            f"Training stopped after epoch {completed_epochs}; expected "
            f"{config.training.epochs}"
        )
    return results
