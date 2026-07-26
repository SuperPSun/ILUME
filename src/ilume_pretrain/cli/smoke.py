from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import PretrainConfig, load_config
from ..data import PreparedCorpusDataset
from ..masking import MultimodalMasker, MultimodalPacker
from ..model import MultimodalPretrainModel
from ..progress import ProgressReporter, loss_postfix
from ..sampler import RoleBalancedSampler
from ..tokenizer import SmilesTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a minimal multimodal pretraining forward/backward smoke test."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    return parser


def _resolve_device(config: PretrainConfig) -> torch.device:
    requested = config.smoke.device
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable")
    return device


def build_dataloader(
    config: PretrainConfig,
    dataset: PreparedCorpusDataset,
    vocabulary: SmilesTokenizer,
) -> DataLoader:
    sampler = RoleBalancedSampler(
        role_ids=dataset.role_ids,
        num_samples=config.smoke.steps * config.smoke.batch_size,
        role_probabilities=config.sampling.role_probabilities,
        seed=config.data.seed,
        shard_ids=[entry["shard"] for entry in dataset.entries],
    )
    return DataLoader(
        dataset,
        batch_size=config.smoke.batch_size,
        sampler=sampler,
        num_workers=config.smoke.num_workers,
        collate_fn=MultimodalPacker(vocabulary),
        drop_last=True,
    )


def run_smoke(config: PretrainConfig) -> list[dict[str, float | int | str]]:
    torch.manual_seed(config.data.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.data.seed)

    artifact_dir = Path(config.data.artifacts_dir)
    dataset = PreparedCorpusDataset(
        artifact_dir,
        split="train",
        shard_cache_size=config.data.shard_cache_size,
    )
    vocabulary = SmilesTokenizer.load(artifact_dir / "tokenizer.json")
    loader = build_dataloader(config, dataset, vocabulary)
    device = _resolve_device(config)
    model = MultimodalPretrainModel(
        config, vocabulary, dataset.descriptor_schema
    ).to(device)
    masker = MultimodalMasker(vocabulary, config.masking, config.data.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.smoke.learning_rate,
        weight_decay=config.smoke.weight_decay,
    )
    amp_enabled = config.smoke.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    results: list[dict[str, float | int | str]] = []
    reporter = ProgressReporter()
    model.train()
    with reporter.bar(
        total=config.smoke.steps,
        desc="Smoke training",
        unit="step",
    ) as progress:
        for step, packed_batch in enumerate(loader, start=1):
            if step > config.smoke.steps:
                break
            batch = masker.apply(
                packed_batch,
                global_step=step - 1,
                total_steps=config.smoke.steps,
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=(
                    torch.float16
                    if device.type == "cuda"
                    else torch.bfloat16
                ),
                enabled=amp_enabled,
            ):
                output = model(batch)
            if not torch.isfinite(output.loss):
                raise RuntimeError(f"Non-finite total loss at smoke step {step}")
            scaler.scale(output.loss).backward()
            scaler.step(optimizer)
            scaler.update()

            result: dict[str, float | int | str] = {
                "step": step,
                "device": str(device),
                "loss": float(output.loss.detach().cpu()),
            }
            result.update(
                {
                    f"loss_{name}": float(value.detach().cpu())
                    for name, value in output.losses.items()
                }
            )
            reporter.emit_json(result)
            results.append(result)
            progress.set_postfix(
                loss_postfix(result, include_learning_rate=False),
                refresh=False,
            )
            progress.update(1)
    if len(results) != config.smoke.steps:
        raise RuntimeError(
            f"Smoke loader produced {len(results)} steps; expected {config.smoke.steps}"
        )
    return results


def main() -> None:
    args = build_parser().parse_args()
    run_smoke(load_config(args.config))


if __name__ == "__main__":
    main()
