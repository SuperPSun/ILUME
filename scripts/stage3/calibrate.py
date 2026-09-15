from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import queue
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
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


def _read_json(path: Path, *, context: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{context} is missing: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is unreadable: {path.name}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must contain a JSON object: {path.name}")
    return payload


def _run_fold(
    *,
    config_path: str,
    checkpoint_dir: str,
    fold: int,
    output_root: str,
    resume: bool,
    device: str | None,
    show_progress: bool,
) -> str:
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
    if not show_progress:
        os.environ["ILUME_DISABLE_PROGRESS"] = "1"

    from common.identity import require_compatible_identity
    from common.outputs import open_run_directory, repository_path, repository_relative
    from stage3.config import configure_process_runtime, load_stage3_config
    from stage3.gate_calibration import (
        resolve_gate_calibration_identity,
        run_gate_calibration,
    )

    config = load_stage3_config(config_path)
    configure_process_runtime(config)
    identity = resolve_gate_calibration_identity(config, fold, checkpoint_dir)
    fold_output = Path(output_root) / f"fold{fold}"
    fold_root = repository_path(fold_output)
    resume_from: Path | None = None
    if fold_root.exists():
        if not resume:
            raise FileExistsError(f"Output already exists: {repository_relative(fold_root)}")
        metadata = _read_json(
            fold_root / "metadata.json", context="Stage 3 calibration metadata"
        )
        require_compatible_identity(
            identity,
            metadata.get("semantic_identity", {}),
            context="Existing Stage 3 gate calibration run",
        )
        if metadata.get("stage") != "stage3" or metadata.get("operation") != "gate_calibrate":
            raise ValueError("Existing output is not a Stage 3 gate calibration run")
        if metadata.get("status") == "completed":
            summary = _read_json(
                fold_root / "summary.json", context="Stage 3 calibration summary"
            )
            manifest = run_gate_calibration(
                config,
                fold,
                checkpoint_dir=checkpoint_dir,
                output_dir=fold_root,
                resume=True,
            )
            if summary.get("gate_calibrated") != manifest:
                raise ValueError("Completed Stage 3 gate calibration summary mismatch")
            return "skipped"
        if metadata.get("status") not in {"running", "failed"}:
            raise ValueError("Existing Stage 3 calibration status is unsupported")
        resume_from = fold_root

    run = open_run_directory(
        stage="stage3",
        operation="gate_calibrate",
        config_path=config_path,
        config_payload=config.to_dict(),
        output=fold_output,
        seed=config.data.seed if config.training.seed is None else config.training.seed,
        semantic_identity=identity,
        resume=resume_from,
        details={
            "fold": fold,
            "checkpoint_dir": repository_relative(repository_path(checkpoint_dir)),
            "assigned_device": device or config.training.device,
        },
    )
    started = time.perf_counter()
    try:
        manifest = run_gate_calibration(
            config,
            fold,
            checkpoint_dir=checkpoint_dir,
            output_dir=run.root,
            resume=resume_from is not None,
        )
        run.complete(
            {
                "fold": fold,
                "wall_seconds": time.perf_counter() - started,
                "gate_calibrated": manifest,
            }
        )
    except BaseException:
        run.fail()
        raise
    return "completed"


def _worker_entry(
    config_path: str,
    checkpoint_dir: str,
    fold: int,
    output_root: str,
    resume: bool,
    device: str | None,
    show_progress: bool,
    result_queue: Any,
) -> None:
    try:
        status = _run_fold(
            config_path=config_path,
            checkpoint_dir=checkpoint_dir,
            fold=fold,
            output_root=output_root,
            resume=resume,
            device=device,
            show_progress=show_progress,
        )
    except BaseException as error:
        result_queue.put(("failed", type(error).__name__, str(error)))
        traceback.print_exc()
        raise SystemExit(1) from error
    result_queue.put((status, None, None))


def _terminate_workers(running: Mapping[int, tuple[Any, int, Any]]) -> None:
    for process, _, _ in running.values():
        if process.is_alive():
            process.terminate()
    for process, _, result_queue in running.values():
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        result_queue.close()


def _run_schedule(
    *,
    config_path: str,
    checkpoint_dir: str,
    folds: tuple[int, ...],
    output_root: str,
    resume: bool,
    max_parallel: int,
    devices: tuple[str, ...],
) -> dict[int, str]:
    from multiprocessing.connection import wait

    ctx = multiprocessing.get_context("spawn")
    effective_parallel = min(max_parallel, len(folds))
    slot_devices = tuple(
        devices[index % len(devices)] if devices else None
        for index in range(effective_parallel)
    )
    pending = iter(folds)
    running: dict[int, tuple[Any, int, Any]] = {}
    results: dict[int, str] = {}

    def launch(slot: int, fold: int) -> None:
        result_queue = ctx.Queue()
        process = ctx.Process(
            target=_worker_entry,
            args=(
                config_path,
                checkpoint_dir,
                fold,
                output_root,
                resume,
                slot_devices[slot],
                effective_parallel == 1,
                result_queue,
            ),
        )
        process.start()
        running[fold] = (process, slot, result_queue)

    try:
        for slot in range(effective_parallel):
            try:
                launch(slot, next(pending))
            except StopIteration:
                break
        while running:
            ready = wait([process.sentinel for process, _, _ in running.values()])
            for fold, (process, slot, result_queue) in list(running.items()):
                if process.sentinel not in ready:
                    continue
                process.join()
                try:
                    status, error_type, message = result_queue.get_nowait()
                except queue.Empty:
                    status, error_type, message = "failed", "WorkerExit", "no result"
                result_queue.close()
                del running[fold]
                results[fold] = status
                if status == "failed":
                    print(
                        f"fold{fold} failed: {error_type}: {message}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(f"fold{fold} {status}", flush=True)
                try:
                    launch(slot, next(pending))
                except StopIteration:
                    pass
    except BaseException:
        _terminate_workers(running)
        raise
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate Stage 3 Flat task gates from three-phase final artifacts."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--fold", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-parallel", type=_positive_int, default=1)
    parser.add_argument("--devices")
    return parser


def main() -> None:
    from stage3.config import validate_stage3_folds

    args = _build_parser().parse_args()
    folds = validate_stage3_folds(args.fold)
    devices = _parse_devices(args.devices)
    if devices and args.max_parallel > len(devices):
        raise ValueError("--max-parallel cannot exceed the number of --devices")
    results = _run_schedule(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        folds=folds,
        output_root=args.output,
        resume=args.resume,
        max_parallel=args.max_parallel,
        devices=devices,
    )
    if any(status == "failed" for status in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
