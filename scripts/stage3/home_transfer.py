from __future__ import annotations

import argparse
from dataclasses import replace
import json
import multiprocessing
import os
import queue
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parse_devices(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    devices = tuple(item.strip() for item in value.split(","))
    if not devices or any(not re.fullmatch(r"cuda:\d+", item) for item in devices):
        raise ValueError("--devices must be a comma-separated list such as cuda:0,cuda:1")
    if len(devices) != len(set(devices)):
        raise ValueError("--devices must not contain duplicate devices")
    return devices


def _run_fold(
    config_path: str, output_root: str, fold: int, resume: bool, device: str | None,
    show_progress: bool,
) -> str:
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
    if not show_progress:
        os.environ["ILUME_DISABLE_PROGRESS"] = "1"

    from ablations.stage2_home_transfer.config import load_experiment
    from ablations.stage2_home_transfer.stage3 import train_fold
    from stage3.config import configure_process_runtime

    experiment = load_experiment(config_path)
    experiment = replace(experiment, output_root=Path(output_root))
    configure_process_runtime(experiment.stage3)
    output = experiment.output_root / "stage3" / "train" / f"fold{fold}"
    completed = (output / "three_phase_final.pt").is_file()
    train_fold(experiment, fold, resume=resume)
    return "skipped" if completed else "completed"


def _worker_entry(
    config_path: str, output_root: str, fold: int, resume: bool, device: str | None,
    show_progress: bool, result_queue: Any,
) -> None:
    try:
        status = _run_fold(config_path, output_root, fold, resume, device, show_progress)
    except BaseException as error:
        result_queue.put(("failed", type(error).__name__, str(error)))
        traceback.print_exc()
        raise SystemExit(1) from error
    result_queue.put((status, None, None))


def _terminate_workers(running: Mapping[int, tuple[Any, int, Any]]) -> None:
    processes = [item[0] for item in running.values()]
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)
    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        process.join()
    for _, _, result_queue in running.values():
        result_queue.close()


def _run_schedule(
    *, config_path: str, output_root: str, folds: tuple[int, ...], resume: bool,
    max_parallel: int, devices: tuple[str, ...],
) -> dict[int, str]:
    from multiprocessing.connection import wait

    context = multiprocessing.get_context("spawn")
    slot_devices = tuple(
        devices[index % len(devices)] if devices else None
        for index in range(min(max_parallel, len(folds)))
    )
    pending = iter(folds)
    running: dict[int, tuple[Any, int, Any]] = {}
    results: dict[int, str] = {}

    def launch(slot: int, fold: int) -> None:
        result_queue = context.Queue()
        process = context.Process(
            target=_worker_entry,
            args=(config_path, output_root, fold, resume, slot_devices[slot], fold == folds[0], result_queue),
            name=f"stage3-home-transfer-fold{fold}",
        )
        process.start()
        running[slot] = (process, fold, result_queue)
        print(f"fold{fold} started", flush=True)

    try:
        for slot in range(len(slot_devices)):
            launch(slot, next(pending))
        while running:
            ready = wait([item[0].sentinel for item in running.values()])
            completed_slots = sorted(
                slot for slot, item in running.items() if item[0].sentinel in ready
            )
            for slot in completed_slots:
                process, fold, result_queue = running.pop(slot)
                process.join()
                try:
                    payload = result_queue.get_nowait()
                except queue.Empty:
                    payload = ("failed", "WorkerExit", f"exit code {process.exitcode}")
                finally:
                    result_queue.close()
                    result_queue.join_thread()
                status = payload[0]
                if process.exitcode != 0 or status not in {"completed", "skipped"}:
                    status = "failed"
                    print(f"fold{fold} failed: {payload[1]}: {payload[2]}", file=sys.stderr, flush=True)
                results[fold] = status
                print(f"fold{fold} {status}", flush=True)
                try:
                    next_fold = next(pending)
                except StopIteration:
                    continue
                launch(slot, next_fold)
    except BaseException:
        _terminate_workers(running)
        raise
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated Stage2-HoME to Stage3-HoME transfer.")
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--config", default="configs/ablations/stage2_home_transfer.yaml")
    parser.add_argument("--output", help="experiment root; defaults to config output_root")
    parser.add_argument("--fold", type=int, nargs="+")
    parser.add_argument("--split", choices=("valid", "test"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-parallel", type=_positive_int, default=1)
    parser.add_argument("--devices", help="comma-separated GPU list such as cuda:0,cuda:1")
    args = parser.parse_args()
    if args.command != "train" and (args.max_parallel != 1 or args.devices is not None):
        parser.error("--max-parallel and --devices are only supported by train")
    from ablations.stage2_home_transfer.config import load_experiment
    from ablations.stage2_home_transfer.stage3 import evaluate, prepare
    from common.io import atomic_json

    experiment = load_experiment(args.config)
    if args.output is not None:
        experiment = replace(experiment, output_root=Path(args.output))
    if args.command == "prepare":
        if args.fold or args.split or args.resume:
            parser.error("prepare does not accept --fold, --split, or --resume")
        print(json.dumps(prepare(experiment), sort_keys=True))
        return 0
    if args.command == "train":
        if not args.fold or args.split:
            parser.error("train requires --fold and does not accept --split")
        if len(args.fold) != len(set(args.fold)) or any(fold not in range(1, 6) for fold in args.fold):
            parser.error("train requires unique folds in 1..5")
        try:
            devices = _parse_devices(args.devices)
        except ValueError as error:
            parser.error(str(error))
        if min(args.max_parallel, len(args.fold)) > 1 and not devices:
            parser.error("--devices is required when more than one fold runs in parallel")
        if devices and experiment.stage3.training.device != "cuda":
            parser.error("--devices requires training.device: cuda")
        try:
            results = _run_schedule(
                config_path=args.config, output_root=str(experiment.output_root),
                folds=tuple(args.fold), resume=args.resume,
                max_parallel=args.max_parallel, devices=devices,
            )
        except KeyboardInterrupt:
            print("Stage3 HoME-transfer fold training interrupted", file=sys.stderr)
            return 130
        print("Stage3 HoME-transfer folds complete")
        for status in ("completed", "skipped", "failed"):
            selected = [str(fold) for fold in args.fold if results.get(fold) == status]
            print(f"{status}: {', '.join(selected) if selected else '-'}")
        return 1 if any(status == "failed" for status in results.values()) else 0
    if args.resume or args.split is None:
        parser.error("evaluate requires --split and does not accept --resume")
    if args.split == "valid":
        if not args.fold or len(args.fold) != len(set(args.fold)) or any(fold not in range(1, 6) for fold in args.fold):
            parser.error("validation requires unique --fold values in 1..5")
        for fold in args.fold:
            destination = experiment.output_root / "stage3" / "valid" / f"fold{fold}"
            if destination.exists():
                raise FileExistsError(f"Evaluation output exists: {destination}")
            destination.mkdir(parents=True)
            result = evaluate(experiment, split="valid", fold=fold, predictions_dir=destination / "predictions")
            atomic_json(destination / "summary.json", result)
            print(f"validation fold{fold} complete")
    else:
        if args.fold:
            parser.error("test ensemble forbids --fold")
        destination = experiment.output_root / "stage3" / "test"
        if destination.exists():
            raise FileExistsError(f"Evaluation output exists: {destination}")
        destination.mkdir(parents=True)
        result = evaluate(experiment, split="test", predictions_dir=destination / "predictions")
        atomic_json(destination / "summary.json", result)
        print("test ensemble complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
