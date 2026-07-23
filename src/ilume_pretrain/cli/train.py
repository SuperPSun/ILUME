from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..config import PretrainConfig, load_config
from ..data import PreparedCorpusDataset
from ..masking import MultimodalMasker, MultimodalPacker
from ..model import MultimodalPretrainModel
from ..sampler import RoleBalancedSampler, minimum_samples_for_coverage
from ..tokenizer import SmilesTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run single-device ILUME multimodal pretraining."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    return parser


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable")
    return device


def _config_hash(config: PretrainConfig) -> str:
    payload = config.to_dict()
    payload["training"]["resume_from"] = None
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lr_lambda(step: int, max_steps: int, warmup_fraction: float) -> float:
    warmup_steps = max(1, round(max_steps * warmup_fraction))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(
    path: Path,
    *,
    model: MultimodalPretrainModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    optimizer_step: int,
    micro_step: int,
    sampler: RoleBalancedSampler,
    config: PretrainConfig,
    config_hash: str,
    artifact_hash: str,
    source_hashes: dict[str, str],
) -> None:
    payload = {
        "format_version": 2,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "optimizer_step": optimizer_step,
        "micro_step": micro_step,
        "sampler": sampler.state_dict(start_offset=micro_step * config.training.batch_size),
        "rng": _rng_state(),
        "config": config.to_dict(),
        "config_hash": config_hash,
        "artifact_hash": artifact_hash,
        "source_hashes": source_hashes,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


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


def run_training(config: PretrainConfig) -> list[dict[str, float | int | str]]:
    config.validate()
    random.seed(config.data.seed)
    np.random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.data.seed)

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
    device = _resolve_device(config.training.device)
    model = MultimodalPretrainModel(
        config, vocabulary, train_dataset.descriptor_schema
    ).to(device)

    total_draws = (
        config.training.max_steps
        * config.training.batch_size
        * config.training.gradient_accumulation_steps
    )
    role_counts = tuple(train_dataset.role_ids.count(role) for role in range(3))
    required_draws = minimum_samples_for_coverage(
        role_counts, config.sampling.role_probabilities
    )
    if config.sampling.require_full_coverage and total_draws < required_draws:
        draws_per_step = (
            config.training.batch_size
            * config.training.gradient_accumulation_steps
        )
        minimum_steps = math.ceil(required_draws / draws_per_step)
        raise ValueError(
            "training.max_steps is insufficient for 45/45/10 full coverage: "
            f"configured={config.training.max_steps}, minimum={minimum_steps}, "
            f"role_counts={role_counts}"
        )
    sampler = RoleBalancedSampler(
        train_dataset.role_ids,
        num_samples=total_draws,
        role_probabilities=config.sampling.role_probabilities,
        seed=config.data.seed,
        require_full_coverage=config.sampling.require_full_coverage,
        shard_ids=[entry["shard"] for entry in train_dataset.entries],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_lambda(
            step, config.training.max_steps, config.training.warmup_fraction
        ),
    )
    fp16 = config.training.amp_dtype == "fp16" and device.type == "cuda"
    amp_enabled = config.training.amp_dtype != "none" and device.type == "cuda"
    amp_dtype = torch.float16 if fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    output_dir = config.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    current_config_hash = _config_hash(config)
    artifact_hash = _file_hash(artifact_dir / "metadata.json")
    optimizer_step = 0
    micro_step = 0
    if config.training.resume_from is not None:
        checkpoint_payload = torch.load(
            config.training.resume_from,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint_payload.get("format_version") != 2:
            raise ValueError("Unsupported checkpoint format")
        if checkpoint_payload["config_hash"] != current_config_hash:
            raise ValueError("Checkpoint config hash does not match the current config")
        if checkpoint_payload["artifact_hash"] != artifact_hash:
            raise ValueError("Checkpoint corpus artifact hash does not match")
        model.load_state_dict(checkpoint_payload["model"])
        optimizer.load_state_dict(checkpoint_payload["optimizer"])
        scheduler.load_state_dict(checkpoint_payload["scheduler"])
        scaler.load_state_dict(checkpoint_payload["scaler"])
        optimizer_step = int(checkpoint_payload["optimizer_step"])
        micro_step = int(checkpoint_payload["micro_step"])
        sampler.load_state_dict(checkpoint_payload["sampler"])
        expected_offset = micro_step * config.training.batch_size
        if sampler.start_offset != expected_offset:
            raise ValueError("Checkpoint sampler cursor does not match micro_step")
        _restore_rng_state(checkpoint_payload["rng"])

    loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=sampler,
        num_workers=config.training.num_workers,
        collate_fn=MultimodalPacker(vocabulary),
        drop_last=True,
    )
    masker = MultimodalMasker(vocabulary, config.masking, config.data.seed)
    metrics_path = output_dir / "metrics.jsonl"
    results: list[dict[str, float | int | str]] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    max_micro_steps = (
        config.training.max_steps * config.training.gradient_accumulation_steps
    )
    for packed in loader:
        if optimizer_step >= config.training.max_steps:
            break
        batch = masker.apply(packed, micro_step, max_micro_steps).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            output = model(batch)
            scaled_loss = output.loss / config.training.gradient_accumulation_steps
        if not torch.isfinite(output.loss):
            raise RuntimeError(f"Non-finite total loss at micro step {micro_step}")
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
        optimizer_step += 1
        result: dict[str, float | int | str] = {
            "step": optimizer_step,
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
        if (
            config.training.validation_interval > 0
            and optimizer_step % config.training.validation_interval == 0
        ):
            result.update(
                _validate(
                    model,
                    valid_dataset,
                    vocabulary,
                    config,
                    device,
                )
            )
        serialized = json.dumps(result, sort_keys=True)
        print(serialized)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
        results.append(result)

        should_checkpoint = (
            config.training.checkpoint_interval > 0
            and optimizer_step % config.training.checkpoint_interval == 0
        ) or optimizer_step == config.training.max_steps
        if should_checkpoint:
            checkpoint_path = output_dir / f"checkpoint_step_{optimizer_step:08d}.pt"
            # Checkpoint offset is restored from micro_step, so the sampler only
            # needs its deterministic epoch state here.
            _save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                optimizer_step=optimizer_step,
                micro_step=micro_step,
                sampler=sampler,
                config=config,
                config_hash=current_config_hash,
                artifact_hash=artifact_hash,
                source_hashes=artifact_metadata.get("source_hashes", {}),
            )
            checkpoints = sorted(output_dir.glob("checkpoint_step_*.pt"))
            for stale in checkpoints[: -config.training.keep_last_checkpoints]:
                stale.unlink()

    if optimizer_step != config.training.max_steps:
        raise RuntimeError(
            f"Training stopped at step {optimizer_step}; expected {config.training.max_steps}"
        )
    return results


def main() -> None:
    args = build_parser().parse_args()
    run_training(load_config(args.config))


if __name__ == "__main__":
    main()
