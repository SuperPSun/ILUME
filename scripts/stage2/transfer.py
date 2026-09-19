from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ablations.stage2_stage3_transfer.config import load_transfer_config
from ablations.stage2_stage3_transfer.stage2 import (
    create_baseline_encoder,
    train_source_encoder,
)
from common.io import sha256_file
from stage3.data import sanitize_task


def _devices(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    values = tuple(value.strip() for value in raw.split(","))
    if not values or any(not re.fullmatch(r"cuda:\d+", value) for value in values):
        raise ValueError("--devices must be a comma-separated cuda device list")
    if len(set(values)) != len(values):
        raise ValueError("--devices contains duplicates")
    return values


def _latest_checkpoint(root: Path) -> Path | None:
    paths = sorted(root.glob("checkpoint_epoch_*.pt"))
    return paths[-1] if paths else None


def _complete(root: Path) -> bool:
    path = root / "manifest.json"
    if not path.is_file():
        return False
    import json
    payload = json.loads(path.read_text(encoding="utf-8"))
    encoder = root / str(payload.get("encoder"))
    return (
        encoder.is_file()
        and payload.get("encoder_sha256") == sha256_file(encoder)
    )


def _worker(
    config_path: str, source: str, output: str, initial_hash: str,
    resume: bool, device: str | None,
) -> None:
    root = Path(output)
    if root.exists():
        if resume and _complete(root):
            return
        if not resume:
            raise FileExistsError(f"Transfer source output exists: {root}")
    checkpoint = _latest_checkpoint(root) if resume else None
    train_source_encoder(
        load_transfer_config(config_path), source, root,
        initial_shared_state_hash=initial_hash,
        resume_from=checkpoint,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train isolated Stage 2 source variants for the transfer matrix."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--devices")
    args = parser.parse_args()
    if args.max_parallel < 1:
        parser.error("--max-parallel must be positive")
    devices = _devices(args.devices)
    if args.max_parallel > 1 and not devices:
        parser.error("--devices is required when --max-parallel exceeds 1")
    if devices and args.max_parallel > len(devices):
        parser.error("--max-parallel cannot exceed the number of devices")
    config = load_transfer_config(args.config)
    sources = tuple(args.source or config.stage2.sources)
    unknown = set(sources) - set(config.stage2.sources)
    if unknown or len(set(sources)) != len(sources):
        parser.error(f"invalid or duplicate sources: {sorted(unknown)}")
    root = Path(args.output)
    baseline_root = root / "baseline"
    if baseline_root.exists():
        if not args.resume or not _complete(baseline_root):
            raise FileExistsError(f"Transfer baseline output exists: {baseline_root}")
        import json
        baseline = json.loads((baseline_root / "manifest.json").read_text(encoding="utf-8"))
    else:
        baseline = create_baseline_encoder(config, baseline_root)
    initial_hash = str(baseline["initial_shared_state_hash"])
    jobs = [
        (
            str(args.config), source,
            str(root / "sources" / sanitize_task(source)), initial_hash,
            bool(args.resume), devices[index % len(devices)] if devices else None,
        )
        for index, source in enumerate(sources)
    ]
    if args.max_parallel == 1:
        for job in jobs:
            _worker(*job)
        return
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_parallel, mp_context=context
    ) as pool:
        futures = [pool.submit(_worker, *job) for job in jobs]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
