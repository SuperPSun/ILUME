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

from common.identity import IDENTITY_CONTRACT_VERSION, require_compatible_identity
from common.io import atomic_torch_save
from common.progress import ProgressReporter, loss_postfix
from common.training import (
    canonical_json_sha256,
    cosine_warmup,
    resolve_device,
    seed_everything,
)
from .config import (
    STAGE1_CHECKPOINT_KIND,
    PretrainConfig,
    config_from_dict,
)
from .data import PreparedCorpusDataset
from .masking import MultimodalMasker, MultimodalPacker
from .model import LossStatistics, MultimodalPretrainModel, PretrainOutput
from .tokenizer import SmilesTokenizer
from .identity import (
    build_stage1_training_identity,
    metadata_identity,
)


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
    """Shard-local deterministic shuffle with an equal-length rank partition."""

    def __init__(
        self,
        shard_ranges: tuple[tuple[int, int], ...],
        *,
        size: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        epoch: int = 0,
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

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @property
    def padding(self) -> int:
        return (-self.size) % self.world_size

    @property
    def samples_per_rank(self) -> int:
        return (self.size + self.padding) // self.world_size

    def __len__(self) -> int:
        return self.samples_per_rank

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
        for position, index in enumerate(self._global_indices()):
            if position % self.world_size != self.rank:
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
    return canonical_json_sha256(config.experiment_dict())


def _loader_options(config: PretrainConfig, device: torch.device) -> dict[str, Any]:
    workers = config.training.num_workers
    options: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": device.type == "cuda" and workers > 0,
    }
    if workers > 0:
        options["prefetch_factor"] = 2
    return options


def _loss_lambdas(config: PretrainConfig) -> dict[str, float]:
    values = {
        "smiles": config.loss.lambda_smiles,
        "descriptor": config.loss.lambda_descriptor,
        "atom": config.loss.lambda_atom,
        "bond": config.loss.lambda_bond,
    }
    if not config.is_global_rdkit:
        values["fingerprint"] = config.loss.lambda_fingerprint
    return values


def _global_training_losses(
    output: PretrainOutput,
    context: _DistributedContext,
    config: PretrainConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    names = tuple(output.loss_statistics)
    sizes = [output.loss_statistics[name].denominators.numel() for name in names]
    packed_denominators = torch.cat(
        [output.loss_statistics[name].denominators.detach() for name in names]
    )
    if context.enabled:
        dist.all_reduce(packed_denominators, op=dist.ReduceOp.SUM)
    losses: dict[str, torch.Tensor] = {}
    offset = 0
    for name, size in zip(names, sizes, strict=True):
        statistics = output.loss_statistics[name]
        denominators = packed_denominators[offset : offset + size]
        offset += size
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
    names = tuple(output.loss_statistics)
    sizes = [output.loss_statistics[name].numerators.numel() for name in names]
    packed = torch.cat(
        [
            tensor
            for name in names
            for tensor in (
                output.loss_statistics[name].numerators.detach(),
                output.loss_statistics[name].denominators.detach(),
            )
        ]
    )
    if context.enabled:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    result: dict[str, float] = {}
    offset = 0
    for name, size in zip(names, sizes, strict=True):
        numerators = packed[offset : offset + size]
        denominators = packed[offset + size : offset + 2 * size]
        offset += 2 * size
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


def _reduce_accumulator(
    accumulator: dict[str, dict[str, torch.Tensor]],
    context: _DistributedContext,
) -> None:
    if not context.enabled:
        return
    entries = [
        tensor
        for name in sorted(accumulator)
        for tensor in accumulator[name].values()
    ]
    sizes = [tensor.numel() for tensor in entries]
    packed = torch.cat([tensor.reshape(-1) for tensor in entries])
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    offset = 0
    for target, size in zip(entries, sizes, strict=True):
        target.copy_(packed[offset : offset + size].reshape_as(target))
        offset += size


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


@torch.inference_mode()
def _validate(
    model: MultimodalPretrainModel,
    loader: DataLoader,
    vocabulary: SmilesTokenizer,
    config: PretrainConfig,
    device: torch.device,
    context: _DistributedContext,
    *,
    quick: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    masker = MultimodalMasker(vocabulary, config.masking, config.data.seed + 100000)
    accumulator = _empty_accumulator()
    model.eval()
    for batch_index, packed in enumerate(loader):
        batch = packed.to(device, non_blocking=device.type == "cuda")
        batch = masker.apply(
            batch, batch_index, max(1, len(loader)), evaluation=True
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            output = model(batch)
        _accumulate_statistics(accumulator, output.loss_statistics)
    _reduce_accumulator(accumulator, context)
    model.train()
    return _validation_metrics(accumulator, config)


def _checkpoint_payload(
    *,
    model: MultimodalPretrainModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    completed_epoch: int,
    global_step: int,
    steps_per_epoch: int,
    train_size: int,
    world_size_at_save: int,
    attempt_id: str,
    config: PretrainConfig,
    training_identity: dict[str, Any],
    corpus_identity: dict[str, Any],
    sampler_layout_identity: dict[str, Any],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "kind": STAGE1_CHECKPOINT_KIND,
        "format_version": config.checkpoint_version,
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "completed_epoch": completed_epoch,
        "global_step": global_step,
        "steps_per_epoch": steps_per_epoch,
        "train_size": train_size,
        "world_size_at_save": world_size_at_save,
        "attempt_id": attempt_id,
        "config": config.to_dict(),
        "training_identity": training_identity,
        "corpus_identity": corpus_identity,
        "sampler_layout_identity": sampler_layout_identity,
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
    completed_epoch: int,
    global_step: int,
    steps_per_epoch: int,
    train_size: int,
    config: PretrainConfig,
    training_identity: dict[str, Any],
    corpus_identity: dict[str, Any],
    sampler_layout_identity: dict[str, Any],
    source_hashes: dict[str, str],
    attempt_id: str,
) -> None:
    if not context.is_primary:
        return
    payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        completed_epoch=completed_epoch,
        global_step=global_step,
        steps_per_epoch=steps_per_epoch,
        train_size=train_size,
        world_size_at_save=context.world_size,
        attempt_id=attempt_id,
        config=config,
        training_identity=training_identity,
        corpus_identity=corpus_identity,
        sampler_layout_identity=sampler_layout_identity,
        source_hashes=source_hashes,
    )
    for path in paths:
        atomic_torch_save(path, payload)

def _load_checkpoint(
    path: Path,
    *,
    config: PretrainConfig,
    training_identity: dict[str, Any],
    train_size: int,
    steps_per_epoch: int,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("kind") != STAGE1_CHECKPOINT_KIND
        or checkpoint.get("format_version") != config.checkpoint_version
    ):
        raise ValueError("Unsupported Stage 1 pretraining checkpoint")
    checkpoint_identity = checkpoint.get("training_identity")
    if not isinstance(checkpoint_identity, dict):
        raise ValueError(
            "Stage 1 checkpoint predates identity contract v1; retrain Stage 1"
        )
    require_compatible_identity(
        training_identity,
        checkpoint_identity,
        context="Stage 1 resume",
    )
    expected = {
        "train_size": train_size,
        "steps_per_epoch": steps_per_epoch,
    }
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            raise ValueError(f"Checkpoint {name} does not match")
    completed_epoch = int(checkpoint["completed_epoch"])
    if not 0 <= completed_epoch <= config.training.epochs:
        raise ValueError("Checkpoint epoch state is invalid")
    if int(checkpoint["global_step"]) != completed_epoch * steps_per_epoch:
        raise ValueError("Checkpoint global_step is not an epoch boundary")
    return checkpoint


def run_training(
    config: PretrainConfig,
    *,
    output_dir: str | Path,
    resume_from: str | Path | None = None,
    attempt_id: str = "direct",
) -> list[dict[str, float | int | str]]:
    config.validate()
    context = _distributed_context()
    seed_everything(config.data.seed)
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
    if device.type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    artifact_dir = config.data.artifacts_dir
    artifact_metadata = json.loads(
        (artifact_dir / "metadata.json").read_text(encoding="utf-8")
    )
    corpus_identity = dict(
        metadata_identity(artifact_metadata, "corpus", context="Stage 1 corpus")
    )
    sampler_layout_identity = dict(
        metadata_identity(
            artifact_metadata, "sampler_layout", context="Stage 1 corpus"
        )
    )
    training_identity = build_stage1_training_identity(
        config, corpus_identity, sampler_layout_identity
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
    if config.training.compile:
        raw_model.compile()
    training_model: MultimodalPretrainModel | DistributedDataParallel
    if context.enabled:
        training_model = DistributedDataParallel(
            raw_model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
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
    metrics_path = output_dir / "metrics.jsonl"
    completed_epoch = 0
    global_step = 0
    if resume_from is not None:
        checkpoint = _load_checkpoint(
            Path(resume_from),
            config=config,
            training_identity=training_identity,
            train_size=len(train_dataset),
            steps_per_epoch=steps_per_epoch,
        )
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        completed_epoch = int(checkpoint["completed_epoch"])
        global_step = int(checkpoint["global_step"])
    if context.enabled:
        dist.barrier()
    if context.is_primary and resume_from is not None:
        attempt_boundary = {
            "event": "attempt_start",
            "attempt_id": attempt_id,
            "resumed_from_attempt_id": checkpoint.get("attempt_id"),
            "completed_epoch": completed_epoch,
            "global_step": global_step,
            "world_size": context.world_size,
            "compile": config.training.compile,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(attempt_boundary, sort_keys=True) + "\n")

    masker = MultimodalMasker(
        vocabulary, config.masking, config.data.seed + context.rank * 1000003
    )
    reporter = ProgressReporter() if context.is_primary else None
    results: list[dict[str, float | int | str]] = []
    latest_valid_loss: float | None = None
    sampler = _EpochSampler(
        train_dataset.shard_ranges,
        size=len(train_dataset),
        seed=config.data.seed,
        rank=context.rank,
        world_size=context.world_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        collate_fn=MultimodalPacker(vocabulary),
        drop_last=False,
        generator=torch.Generator().manual_seed(
            config.data.seed + context.rank * 1000003
        ),
        **_loader_options(config, device),
    )
    quick_indices = _quick_validation_indices(
        valid_dataset,
        config.training.quick_validation_samples_per_role,
        config.data.seed + 200000,
    )

    def validation_loader(indices: list[int]) -> DataLoader:
        selected = Subset(valid_dataset, indices[context.rank :: context.world_size])
        return DataLoader(
            selected,
            batch_size=local_batch_size,
            shuffle=False,
            collate_fn=MultimodalPacker(vocabulary),
            drop_last=False,
            generator=torch.Generator().manual_seed(config.data.seed + 300000),
            **_loader_options(config, device),
        )

    quick_loader = validation_loader(quick_indices)
    full_loader = validation_loader(list(range(len(valid_dataset))))
    raw_model.train()
    optimizer.zero_grad(set_to_none=True)
    while completed_epoch < config.training.epochs:
        epoch_index = completed_epoch
        sampler.set_epoch(epoch_index)
        seed_everything(
            config.data.seed
            + epoch_index * 1000003
            + context.rank * 1009
            + context.world_size * 9176
        )
        epoch_number = epoch_index + 1
        completed_epoch_steps = 0
        description = f"Epoch {epoch_number}/{config.training.epochs}"
        progress_context = (
            reporter.bar(
                total=steps_per_epoch,
                initial=0,
                desc=description,
                unit="step",
            )
            if reporter is not None
            else nullcontext(_SilentProgress())
        )
        with progress_context as progress:
            for packed in train_loader:
                batch = packed.to(device, non_blocking=device.type == "cuda")
                batch = masker.apply(batch, global_step, total_steps)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    output = training_model(batch)
                    training_loss, _ = _global_training_losses(
                        output, context, config
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
                completed_epoch_steps += 1
                epoch_finished = completed_epoch_steps == steps_per_epoch
                should_quick_validate = (
                    not epoch_finished
                    and global_step % config.training.validation_interval_steps == 0
                )
                should_log = (
                    global_step % 10 == 0
                    or should_quick_validate
                    or epoch_finished
                )
                result: dict[str, float | int | str] | None = None
                if should_log:
                    metric_losses = _global_metric_losses(output, context)
                    result = {
                        "attempt_id": attempt_id,
                        "epoch": epoch_number,
                        "epoch_step": completed_epoch_steps,
                        "global_step": global_step,
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
                    if not all(
                        math.isfinite(value)
                        for key, value in result.items()
                        if key == "loss" or key.startswith("loss_")
                    ):
                        raise RuntimeError(
                            f"Non-finite loss detected at optimizer step {global_step}"
                        )
                if epoch_finished or should_quick_validate:
                    if result is None:
                        raise RuntimeError("Validation requires a metric result")
                    if context.is_primary:
                        progress.set_description_str("Validating")
                    result.update(
                        _validate(
                            raw_model,
                            full_loader if epoch_finished else quick_loader,
                            vocabulary,
                            config,
                            device,
                            context,
                            quick=not epoch_finished,
                            amp_enabled=amp_enabled,
                            amp_dtype=amp_dtype,
                        )
                    )
                    if context.is_primary:
                        latest_valid_loss = float(result["valid_loss"])
                        progress.set_description_str(description)

                if context.is_primary and result is not None:
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

            if completed_epoch_steps != steps_per_epoch:
                raise RuntimeError(
                    f"Epoch {epoch_number} stopped at optimizer step "
                    f"{completed_epoch_steps}; expected {steps_per_epoch}"
                )
        completed_epoch = epoch_number
        _save_checkpoint(
            (
                output_dir / f"checkpoint_epoch_{completed_epoch:05d}.pt",
                output_dir / "last.pt",
            ),
            context=context,
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            completed_epoch=completed_epoch,
            global_step=global_step,
            steps_per_epoch=steps_per_epoch,
            train_size=len(train_dataset),
            config=config,
            training_identity=training_identity,
            corpus_identity=corpus_identity,
            sampler_layout_identity=sampler_layout_identity,
            source_hashes=artifact_metadata.get("source_hashes", {}),
            attempt_id=attempt_id,
        )
    return results
