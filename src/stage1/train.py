from __future__ import annotations

import hashlib
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler, Subset

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
from .config import (
    STAGE1_CHECKPOINT_KIND,
    STAGE1_CHECKPOINT_VERSION,
    PretrainConfig,
    config_from_dict,
)
from .data import PreparedCorpusDataset
from .masking import MultimodalMasker, MultimodalPacker
from .model import LossStatistics, MultimodalPretrainModel, PretrainOutput
from .tokenizer import SmilesTokenizer


ROLE_NAMES = ("cation", "anion", "molecule")


@dataclass(frozen=True)
class _DistributedContext:
    rank: int
    world_size: int
    local_rank: int

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def _distributed_context() -> _DistributedContext:
    if not dist.is_available() or not dist.is_initialized():
        return _DistributedContext(0, 1, 0)
    return _DistributedContext(
        dist.get_rank(),
        dist.get_world_size(),
        int(os.environ.get("LOCAL_RANK", dist.get_rank())),
    )


class _EpochSampler(Sampler[int]):
    """Shard-local deterministic shuffle with rank partition and a consumed cursor."""

    def __init__(
        self,
        shard_ranges: tuple[tuple[int, int], ...],
        *,
        size: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        epoch: int = 0,
        cursor: int = 0,
    ) -> None:
        if size <= 0:
            raise ValueError("Training dataset must not be empty")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed sampler rank/world_size")
        if sum(count for _, count in shard_ranges) != size:
            raise ValueError("Shard ranges do not cover the training dataset")
        self.shard_ranges = shard_ranges
        self.size = size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = epoch
        self.cursor = cursor
        if not 0 <= cursor <= self.samples_per_rank:
            raise ValueError("Sampler cursor is outside the rank partition")

    @property
    def padding(self) -> int:
        return (-self.size) % self.world_size

    @property
    def samples_per_rank(self) -> int:
        return (self.size + self.padding) // self.world_size

    def __len__(self) -> int:
        return self.samples_per_rank - self.cursor

    def _shard_seed(self, start: int, count: int) -> int:
        digest = hashlib.sha256(
            f"{self.seed}\0{self.epoch}\0{start}\0{count}".encode()
        ).digest()
        return int.from_bytes(digest[:8], "little")

    def _global_indices(self) -> Iterator[int]:
        shard_order = list(range(len(self.shard_ranges)))
        random.Random(self.seed + self.epoch).shuffle(shard_order)
        prefix: list[int] = []
        for shard_index in shard_order:
            start, count = self.shard_ranges[shard_index]
            offsets = np.arange(count, dtype=np.int64)
            np.random.default_rng(self._shard_seed(start, count)).shuffle(offsets)
            for offset in offsets:
                index = start + int(offset)
                if len(prefix) < self.padding:
                    prefix.append(index)
                yield index
        yield from prefix

    def __iter__(self) -> Iterator[int]:
        consumed = 0
        for position, index in enumerate(self._global_indices()):
            if position % self.world_size != self.rank:
                continue
            if consumed < self.cursor:
                consumed += 1
                continue
            yield index


class _SilentProgress:
    def update(self, count: int = 1) -> None:
        del count

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def set_description_str(self, value: str) -> None:
        del value


def _config_hash(config: PretrainConfig) -> str:
    return canonical_json_sha256(config.to_dict())


def _loader_options(config: PretrainConfig, device: torch.device) -> dict[str, Any]:
    workers = config.training.num_workers
    return {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": device.type == "cuda" and workers > 0,
    }


def _loss_lambdas(config: PretrainConfig) -> dict[str, float]:
    return {
        "smiles": config.loss.lambda_smiles,
        "descriptor": config.loss.lambda_descriptor,
        "atom": config.loss.lambda_atom,
        "bond": config.loss.lambda_bond,
        "fingerprint": config.loss.lambda_fingerprint,
    }


def _global_training_losses(
    output: PretrainOutput,
    context: _DistributedContext,
    config: PretrainConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses: dict[str, torch.Tensor] = {}
    for name, statistics in output.loss_statistics.items():
        denominators = statistics.denominators.detach().clone()
        if context.enabled:
            dist.all_reduce(denominators, op=dist.ReduceOp.SUM)
        value = (
            statistics.numerators
            / torch.where(
                denominators > 0,
                denominators,
                torch.ones_like(denominators),
            )
        ).mean()
        if context.enabled:
            value = value * context.world_size
        losses[name] = value
    lambdas = _loss_lambdas(config)
    total = sum(lambdas[name] * value for name, value in losses.items())
    return total, losses


def _global_metric_losses(
    output: PretrainOutput,
    context: _DistributedContext,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, statistics in output.loss_statistics.items():
        numerators = statistics.numerators.detach().clone()
        denominators = statistics.denominators.detach().clone()
        if context.enabled:
            dist.all_reduce(numerators, op=dist.ReduceOp.SUM)
            dist.all_reduce(denominators, op=dist.ReduceOp.SUM)
        result[name] = float(
            (
                numerators
                / torch.where(
                    denominators > 0,
                    denominators,
                    torch.ones_like(denominators),
                )
            ).mean().cpu()
        )
    return result


def _empty_accumulator() -> dict[str, dict[str, torch.Tensor]]:
    return {}


def _accumulate_statistics(
    accumulator: dict[str, dict[str, torch.Tensor]],
    statistics: dict[str, LossStatistics],
) -> None:
    for name, value in statistics.items():
        if name not in accumulator:
            accumulator[name] = {
                "numerators": torch.zeros_like(value.numerators, dtype=torch.float64),
                "denominators": torch.zeros_like(value.denominators, dtype=torch.float64),
                "role_numerators": torch.zeros_like(
                    value.role_numerators, dtype=torch.float64
                ),
                "role_denominators": torch.zeros_like(
                    value.role_denominators, dtype=torch.float64
                ),
            }
        target = accumulator[name]
        target["numerators"] += value.numerators.detach().to(torch.float64)
        target["denominators"] += value.denominators.detach().to(torch.float64)
        target["role_numerators"] += value.role_numerators.detach().to(torch.float64)
        target["role_denominators"] += value.role_denominators.detach().to(torch.float64)


def _mean_components(numerators: torch.Tensor, denominators: torch.Tensor) -> float:
    safe = torch.where(
        denominators > 0, denominators, torch.ones_like(denominators)
    )
    return float((numerators / safe).mean().cpu())


def _validation_metrics(
    accumulator: dict[str, dict[str, torch.Tensor]],
    config: PretrainConfig,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    lambdas = _loss_lambdas(config)
    global_losses: dict[str, float] = {}
    for name, values in accumulator.items():
        global_losses[name] = _mean_components(
            values["numerators"], values["denominators"]
        )
        metrics[f"valid_loss_{name}"] = global_losses[name]
    metrics["valid_loss"] = sum(
        lambdas[name] * value for name, value in global_losses.items()
    )
    for role, role_name in enumerate(ROLE_NAMES):
        role_losses: dict[str, float] = {}
        for name, values in accumulator.items():
            role_losses[name] = _mean_components(
                values["role_numerators"][:, role],
                values["role_denominators"][:, role],
            )
            metrics[f"valid_{role_name}_loss_{name}"] = role_losses[name]
        metrics[f"valid_{role_name}_loss"] = sum(
            lambdas[name] * value for name, value in role_losses.items()
        )
    return metrics


def _quick_validation_indices(
    dataset: PreparedCorpusDataset, samples_per_role: int, seed: int
) -> list[int]:
    selected: list[int] = []
    role_ids = dataset.role_ids
    for role in range(3):
        indices = np.flatnonzero(role_ids == role)
        rng = np.random.default_rng(seed + role)
        rng.shuffle(indices)
        selected.extend(int(value) for value in indices[:samples_per_role])
    return selected


@torch.no_grad()
def _validate(
    model: MultimodalPretrainModel,
    dataset: PreparedCorpusDataset,
    vocabulary: SmilesTokenizer,
    config: PretrainConfig,
    device: torch.device,
    *,
    quick: bool,
    batch_size: int | None = None,
) -> dict[str, float]:
    selected: PreparedCorpusDataset | Subset
    if quick:
        selected = Subset(
            dataset,
            _quick_validation_indices(
                dataset,
                config.training.quick_validation_samples_per_role,
                config.data.seed + 200000,
            ),
        )
    else:
        selected = dataset
    loader = DataLoader(
        selected,
        batch_size=batch_size or config.training.batch_size,
        shuffle=False,
        collate_fn=MultimodalPacker(vocabulary),
        drop_last=False,
        generator=torch.Generator().manual_seed(config.data.seed + 300000),
        **_loader_options(config, device),
    )
    masker = MultimodalMasker(vocabulary, config.masking, config.data.seed + 100000)
    accumulator = _empty_accumulator()
    model.eval()
    for batch_index, packed in enumerate(loader):
        batch = masker.apply(packed, batch_index, max(1, len(loader)), evaluation=True).to(
            device
        )
        output = model(batch)
        _accumulate_statistics(accumulator, output.loss_statistics)
    model.train()
    return _validation_metrics(accumulator, config)


def _capture_all_rank_rng(
    context: _DistributedContext,
) -> list[dict[str, Any]] | None:
    local = capture_rng_state()
    if not context.enabled:
        return [local]
    gathered: list[dict[str, Any]] | None = (
        [None] * context.world_size if context.is_primary else None  # type: ignore[list-item]
    )
    dist.gather_object(local, gathered, dst=0)
    return gathered


def _checkpoint_payload(
    *,
    model: MultimodalPretrainModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch_index: int,
    epoch_cursor: int,
    completed_epochs: int,
    global_step: int,
    steps_per_epoch: int,
    train_size: int,
    world_size: int,
    rank_rng: list[dict[str, Any]],
    config: PretrainConfig,
    config_hash: str,
    artifact_hash: str,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "kind": STAGE1_CHECKPOINT_KIND,
        "format_version": STAGE1_CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch_index": epoch_index,
        "epoch_cursor": epoch_cursor,
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "micro_step": global_step,
        "steps_per_epoch": steps_per_epoch,
        "train_size": train_size,
        "world_size": world_size,
        "rank_rng": rank_rng,
        "config": config.to_dict(),
        "config_hash": config_hash,
        "artifact_hash": artifact_hash,
        "source_hashes": source_hashes,
    }


def _save_checkpoint(
    paths: tuple[Path, ...],
    *,
    context: _DistributedContext,
    model: MultimodalPretrainModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch_index: int,
    epoch_cursor: int,
    completed_epochs: int,
    global_step: int,
    steps_per_epoch: int,
    train_size: int,
    config: PretrainConfig,
    config_hash: str,
    artifact_hash: str,
    source_hashes: dict[str, str],
) -> None:
    rank_rng = _capture_all_rank_rng(context)
    if not context.is_primary:
        return
    if rank_rng is None:
        raise RuntimeError("Rank RNG collection failed")
    payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch_index=epoch_index,
        epoch_cursor=epoch_cursor,
        completed_epochs=completed_epochs,
        global_step=global_step,
        steps_per_epoch=steps_per_epoch,
        train_size=train_size,
        world_size=context.world_size,
        rank_rng=rank_rng,
        config=config,
        config_hash=config_hash,
        artifact_hash=artifact_hash,
        source_hashes=source_hashes,
    )
    for path in paths:
        atomic_torch_save(path, payload)


def _truncate_metrics(path: Path, global_step: int) -> None:
    if not path.is_file():
        return
    retained: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if int(payload["global_step"]) <= global_step:
            retained.append(json.dumps(payload, sort_keys=True))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(f"{line}\n" for line in retained), encoding="utf-8"
    )
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    *,
    context: _DistributedContext,
    config: PretrainConfig,
    config_hash: str,
    artifact_hash: str,
    train_size: int,
    steps_per_epoch: int,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("kind") != STAGE1_CHECKPOINT_KIND
        or checkpoint.get("format_version") != STAGE1_CHECKPOINT_VERSION
    ):
        raise ValueError("Unsupported Stage 1 pretraining checkpoint")
    checkpoint_config = config_from_dict(checkpoint["config"])
    if checkpoint_config.to_dict() != config.to_dict():
        raise ValueError("Checkpoint config does not match the current config")
    expected = {
        "config_hash": config_hash,
        "artifact_hash": artifact_hash,
        "world_size": context.world_size,
        "train_size": train_size,
        "steps_per_epoch": steps_per_epoch,
    }
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            raise ValueError(f"Checkpoint {name} does not match")
    if int(checkpoint["micro_step"]) != int(checkpoint["global_step"]):
        raise ValueError("Checkpoint micro_step does not match global_step")
    epoch_index = int(checkpoint["epoch_index"])
    epoch_cursor = int(checkpoint["epoch_cursor"])
    completed_epochs = int(checkpoint["completed_epochs"])
    if not 0 <= completed_epochs == epoch_index <= config.training.epochs:
        raise ValueError("Checkpoint epoch state is invalid")
    samples_per_rank = math.ceil(train_size / context.world_size)
    if not 0 <= epoch_cursor <= samples_per_rank:
        raise ValueError("Checkpoint epoch cursor is invalid")
    local_batch_size = config.training.batch_size // context.world_size
    expected_step = epoch_index * steps_per_epoch + math.ceil(
        epoch_cursor / local_batch_size
    )
    if int(checkpoint["global_step"]) != expected_step:
        raise ValueError("Checkpoint global_step does not match epoch cursor")
    if epoch_index == config.training.epochs and epoch_cursor != 0:
        raise ValueError("Completed checkpoint has a nonzero epoch cursor")
    rank_rng = checkpoint.get("rank_rng")
    if not isinstance(rank_rng, list) or len(rank_rng) != context.world_size:
        raise ValueError("Checkpoint rank RNG state does not match world_size")
    return checkpoint


def run_training(
    config: PretrainConfig,
    *,
    output_dir: str | Path,
    resume_from: str | Path | None = None,
) -> list[dict[str, float | int | str]]:
    config.validate()
    context = _distributed_context()
    seed_everything(config.data.seed + context.rank)
    if config.training.batch_size % context.world_size:
        raise ValueError("training.batch_size must be divisible by world_size")
    local_batch_size = config.training.batch_size // context.world_size

    requested_device = resolve_device(config.training.device)
    if context.enabled and requested_device.type == "cuda":
        local_rank = context.local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = requested_device
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
    raw_model = MultimodalPretrainModel(
        config, vocabulary, train_dataset.descriptor_schema
    ).to(device)
    training_model: MultimodalPretrainModel | DistributedDataParallel
    if context.enabled:
        training_model = DistributedDataParallel(
            raw_model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        training_model = raw_model

    steps_per_epoch = math.ceil(len(train_dataset) / config.training.batch_size)
    total_steps = config.training.epochs * steps_per_epoch
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
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
    if context.is_primary:
        output_dir.mkdir(parents=True, exist_ok=True)
    if context.enabled:
        dist.barrier()
    config_hash = _config_hash(config)
    artifact_hash = sha256_file(artifact_dir / "metadata.json")
    metrics_path = output_dir / "metrics.jsonl"
    epoch_index = 0
    epoch_cursor = 0
    completed_epochs = 0
    global_step = 0
    if resume_from is not None:
        checkpoint = _load_checkpoint(
            Path(resume_from),
            context=context,
            config=config,
            config_hash=config_hash,
            artifact_hash=artifact_hash,
            train_size=len(train_dataset),
            steps_per_epoch=steps_per_epoch,
        )
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        epoch_index = int(checkpoint["epoch_index"])
        epoch_cursor = int(checkpoint["epoch_cursor"])
        completed_epochs = int(checkpoint["completed_epochs"])
        global_step = int(checkpoint["global_step"])
        restore_rng_state(checkpoint["rank_rng"][context.rank])
        if context.is_primary:
            _truncate_metrics(metrics_path, global_step)
    if context.enabled:
        dist.barrier()

    masker = MultimodalMasker(
        vocabulary, config.masking, config.data.seed + context.rank * 1000003
    )
    reporter = ProgressReporter() if context.is_primary else None
    results: list[dict[str, float | int | str]] = []
    latest_valid_loss: float | None = None
    raw_model.train()
    optimizer.zero_grad(set_to_none=True)
    while epoch_index < config.training.epochs:
        sampler = _EpochSampler(
            train_dataset.shard_ranges,
            size=len(train_dataset),
            seed=config.data.seed,
            rank=context.rank,
            world_size=context.world_size,
            epoch=epoch_index,
            cursor=epoch_cursor,
        )
        loader = DataLoader(
            train_dataset,
            batch_size=local_batch_size,
            sampler=sampler,
            collate_fn=MultimodalPacker(vocabulary),
            drop_last=False,
            generator=torch.Generator().manual_seed(
                config.data.seed + context.rank * 1000003 + epoch_index
            ),
            **_loader_options(config, device),
        )
        epoch_number = epoch_index + 1
        completed_epoch_steps = math.ceil(epoch_cursor / local_batch_size)
        description = f"Epoch {epoch_number}/{config.training.epochs}"
        progress_context = (
            reporter.bar(
                total=steps_per_epoch,
                initial=completed_epoch_steps,
                desc=description,
                unit="step",
            )
            if reporter is not None
            else nullcontext(_SilentProgress())
        )
        with progress_context as progress:
            for packed in loader:
                batch = masker.apply(packed, global_step, total_steps).to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    output = training_model(batch)
                    training_loss, _ = _global_training_losses(
                        output, context, config
                    )
                if not torch.isfinite(training_loss):
                    raise RuntimeError(
                        f"Non-finite total loss at optimizer step {global_step}"
                    )
                scaler.scale(training_loss).backward()
                if config.training.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        raw_model.parameters(), config.training.max_grad_norm
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                epoch_cursor += len(packed.sample_ids)

                metric_losses = _global_metric_losses(output, context)
                result: dict[str, float | int | str] = {
                    "epoch": epoch_number,
                    "epoch_step": completed_epoch_steps + 1,
                    "global_step": global_step,
                    "micro_step": global_step,
                    "device": str(device),
                    "loss": sum(
                        _loss_lambdas(config)[name] * value
                        for name, value in metric_losses.items()
                    ),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    **{
                        f"loss_{name}": value
                        for name, value in metric_losses.items()
                    },
                }
                completed_epoch_steps += 1
                epoch_finished = completed_epoch_steps == steps_per_epoch
                should_quick_validate = (
                    not epoch_finished
                    and global_step % config.training.validation_interval_steps == 0
                )
                if epoch_finished or should_quick_validate:
                    if context.enabled:
                        dist.barrier()
                    if context.is_primary:
                        progress.set_description_str("Validating")
                        result.update(
                            _validate(
                                raw_model,
                                valid_dataset,
                                vocabulary,
                                config,
                                device,
                                quick=not epoch_finished,
                                batch_size=local_batch_size,
                            )
                        )
                        latest_valid_loss = float(result["valid_loss"])
                        progress.set_description_str(description)
                    if context.enabled:
                        dist.barrier()

                if context.is_primary:
                    serialized = json.dumps(result, sort_keys=True)
                    reporter.emit_json(result)  # type: ignore[union-attr]
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

                interval_checkpoint = (
                    not epoch_finished
                    and global_step % config.training.checkpoint_interval_steps == 0
                )
                if interval_checkpoint:
                    _save_checkpoint(
                        (output_dir / "last.pt",),
                        context=context,
                        model=raw_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch_index=epoch_index,
                        epoch_cursor=epoch_cursor,
                        completed_epochs=completed_epochs,
                        global_step=global_step,
                        steps_per_epoch=steps_per_epoch,
                        train_size=len(train_dataset),
                        config=config,
                        config_hash=config_hash,
                        artifact_hash=artifact_hash,
                        source_hashes=artifact_metadata.get("source_hashes", {}),
                    )

            if completed_epoch_steps != steps_per_epoch:
                raise RuntimeError(
                    f"Epoch {epoch_number} stopped at optimizer step "
                    f"{completed_epoch_steps}; expected {steps_per_epoch}"
                )
        completed_epochs = epoch_number
        epoch_index = completed_epochs
        epoch_cursor = 0
        _save_checkpoint(
            (
                output_dir / f"checkpoint_epoch_{completed_epochs:05d}.pt",
                output_dir / "last.pt",
            ),
            context=context,
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch_index=epoch_index,
            epoch_cursor=epoch_cursor,
            completed_epochs=completed_epochs,
            global_step=global_step,
            steps_per_epoch=steps_per_epoch,
            train_size=len(train_dataset),
            config=config,
            config_hash=config_hash,
            artifact_hash=artifact_hash,
            source_hashes=artifact_metadata.get("source_hashes", {}),
        )
    return results
