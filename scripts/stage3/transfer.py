from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import ExitStack
import json
import multiprocessing
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ablations.stage2_stage3_transfer.config import load_transfer_config
from ablations.stage2_stage3_transfer.sampling import require_experiment_contract
from ablations.stage2_stage3_transfer.stage3 import (
    prepare_representation_bank,
    train_transfer_job,
)
from ablations.stage2_stage3_transfer.summary import summarize_transfer_matrix
from common.io import sha256_file
from common.progress import ProgressReporter
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


def _parallel_slots(max_parallel: int, devices: tuple[str, ...]) -> int:
    if max_parallel < 1:
        raise ValueError("--max-parallel must be positive")
    if not devices:
        if max_parallel != 1:
            raise ValueError("--devices is required when --max-parallel exceeds 1")
        return 1
    if max_parallel < len(devices) or max_parallel % len(devices) != 0:
        raise ValueError(
            "--max-parallel must be a positive multiple of the number of --devices"
        )
    return max_parallel // len(devices)


def _variant_paths(stage2_root: Path, representation_root: Path, source: str | None) -> tuple[str, Path, Path, Path]:
    if source is None:
        return (
            "baseline",
            stage2_root / "baseline/stage2_encoder.pt",
            stage2_root / "baseline/manifest.json",
            representation_root / "baseline.pt",
        )
    slug = sanitize_task(source)
    return (
        source,
        stage2_root / "sources" / slug / "stage2_encoder.pt",
        stage2_root / "sources" / slug / "manifest.json",
        representation_root / "sources" / f"{slug}.pt",
    )


def _job_output(root: Path, variant: str, task: str, fold: int) -> Path:
    if variant == "baseline":
        return root / "baseline" / sanitize_task(task) / f"fold{fold}"
    return root / "sources" / sanitize_task(variant) / sanitize_task(task) / f"fold{fold}"


def _complete(root: Path) -> bool:
    path = root / "manifest.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifact = root / str(payload.get("artifact"))
    predictions = root / "validation_predictions.csv"
    return (
        artifact.is_file()
        and predictions.is_file()
        and payload.get("artifact_sha256") == sha256_file(artifact)
        and payload.get("predictions_sha256") == sha256_file(predictions)
        and payload.get("final_epoch") == 10
    )


def _train_worker(
    config_path: str, variant: str, representation: str, task: str,
    fold: int, output: str, resume: bool, show_detail_progress: bool,
    device: str | None,
) -> None:
    root = Path(output)
    config = load_transfer_config(config_path)
    if root.exists():
        if resume and _complete(root):
            manifest = json.loads((root / "manifest.json").read_text())
            require_experiment_contract(config, manifest)
            if (manifest.get("variant"), manifest.get("task"), manifest.get("fold")) != (variant, task, fold):
                raise ValueError("Transfer resumed job identity mismatch")
            bank = json.loads(Path(representation).with_suffix(".json").read_text())
            require_experiment_contract(config, bank)
            if bank.get("identity") != manifest.get("representation_identity"):
                raise ValueError("Transfer resumed representation identity mismatch")
            return
        raise FileExistsError(
            f"Incomplete/existing transfer job must use a new output root: {root}"
        )
    train_transfer_job(
        config, variant=variant,
        representation_path=representation, task_id=task, fold=fold,
        output_dir=root, device_name=device,
        reporter=(
            None
            if show_detail_progress
            else ProgressReporter(interactive=False)
        ),
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, train, and summarize the Stage2-to-Stage3 transfer matrix."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    _add_common(prepare)
    prepare.add_argument("--stage2-dir", required=True)
    prepare.add_argument("--output", required=True)
    train = commands.add_parser("train")
    _add_common(train)
    train.add_argument("--representations", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--source", nargs="+")
    train.add_argument("--target", nargs="+")
    train.add_argument("--fold", nargs="+", type=int)
    train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--max-parallel", type=int, default=1,
        help="total concurrent jobs; must be divisible by the number of devices",
    )
    train.add_argument(
        "--devices",
        help="comma-separated CUDA devices; each receives max-parallel/device-count jobs",
    )
    summarize = commands.add_parser("summarize")
    _add_common(summarize)
    summarize.add_argument("--stage3-dir", required=True)
    summarize.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_transfer_config(args.config)
    if args.command == "prepare":
        stage2_root = Path(args.stage2_dir)
        representation_root = Path(args.output)
        baseline_manifest = json.loads(
            (stage2_root / "baseline/manifest.json").read_text(encoding="utf-8")
        )
        initial_hash = str(baseline_manifest["initial_shared_state_hash"])
        variants = (None, *config.stage2.sources)
        progress = ProgressReporter().bar(
            total=len(variants), desc="Transfer representations", unit="variant"
        )
        try:
            for source in variants:
                variant, encoder, manifest, destination = _variant_paths(
                    stage2_root, representation_root, source
                )
                prepare_representation_bank(
                    config, variant=variant, encoder_path=encoder,
                    encoder_manifest_path=manifest, destination=destination,
                    expected_initial_shared_state_hash=initial_hash,
                )
                progress.update(1)
        finally:
            progress.close()
        return
    if args.command == "summarize":
        reporter = ProgressReporter()
        with reporter.status("Summarizing transfer matrix"):
            summarize_transfer_matrix(
                config, stage3_root=args.stage3_dir, output_dir=args.output
            )
        return
    try:
        devices = _devices(args.devices)
        slots_per_device = _parallel_slots(args.max_parallel, devices)
    except ValueError as error:
        parser.error(str(error))
    selected_sources = tuple(args.source or ("baseline", *config.stage2.sources))
    allowed_sources = {"baseline", *config.stage2.sources}
    if set(selected_sources) - allowed_sources or len(set(selected_sources)) != len(selected_sources):
        parser.error("--source contains an unknown or duplicate variant")
    targets = tuple(args.target or config.stage3.targets)
    if set(targets) - set(config.stage3.targets) or len(set(targets)) != len(targets):
        parser.error("--target contains an unknown or duplicate task")
    folds = tuple(args.fold or config.stage3.folds)
    if set(folds) - set(config.stage3.folds) or len(set(folds)) != len(folds):
        parser.error("--fold contains an invalid or duplicate fold")
    representations = Path(args.representations)
    output = Path(args.output)
    jobs: list[tuple[object, ...]] = []
    index = 0
    for variant in selected_sources:
        representation = (
            representations / "baseline.pt"
            if variant == "baseline"
            else representations / "sources" / f"{sanitize_task(variant)}.pt"
        )
        for task in targets:
            for fold in folds:
                jobs.append((
                    str(args.config), variant, str(representation), task, fold,
                    str(_job_output(output, variant, task, fold)), bool(args.resume),
                    args.max_parallel == 1,
                    devices[index % len(devices)] if devices else None,
                ))
                index += 1
    progress = ProgressReporter().bar(
        total=len(jobs), desc="Stage3 transfer matrix", unit="train-job"
    )
    try:
        if args.max_parallel == 1:
            for job in jobs:
                _train_worker(*job)
                progress.update(1)
            return
        context = multiprocessing.get_context("spawn")
        with ExitStack() as stack:
            pools = {
                device: stack.enter_context(
                    concurrent.futures.ProcessPoolExecutor(
                        max_workers=slots_per_device, mp_context=context
                    )
                )
                for device in devices
            }
            futures = [
                pools[str(job[-1])].submit(_train_worker, *job) for job in jobs
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
                progress.update(1)
    finally:
        progress.close()


if __name__ == "__main__":
    main()
