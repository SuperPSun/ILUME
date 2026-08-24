from __future__ import annotations

import argparse
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


def _history_epochs(path: Path, *, context: str) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"{context} is missing: {path.name}")
    epochs: list[int] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("epoch"), int):
                raise ValueError
            epochs.append(row["epoch"])
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{context} is unreadable: {path.name}") from error
    return epochs


def _last_history_row(path: Path) -> dict[str, Any]:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("Stage 3 metrics history is empty after training")
    payload = json.loads(rows[-1])
    if not isinstance(payload, dict):
        raise ValueError("Stage 3 final metrics row must be a JSON object")
    return payload


def _require_continuous_history(epochs: list[int], *, context: str) -> None:
    if epochs != list(range(1, len(epochs) + 1)):
        raise ValueError(f"{context} epoch history is not continuous from epoch 1")


def _checkpoint_epochs(root: Path) -> list[int]:
    epochs: list[int] = []
    for path in root.glob("checkpoint_epoch_*.pt"):
        match = re.fullmatch(r"checkpoint_epoch_(\d{5})\.pt", path.name)
        if match is not None:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def _validate_existing_identity(
    metadata: Mapping[str, Any], training_identity: Mapping[str, Any]
) -> None:
    from common.identity import require_compatible_identity

    existing = metadata.get("semantic_identity")
    if not isinstance(existing, Mapping):
        raise ValueError("Existing Stage 3 run predates identity contract v1; retrain it")
    require_compatible_identity(
        training_identity,
        existing,
        context="Existing Stage 3 train run",
    )


def _resume_action(
    root: Path,
    *,
    fold: int,
    total_epochs: int,
    training_identity: Mapping[str, Any],
) -> tuple[str, Path | None]:
    metadata = _read_json(root / "metadata.json", context="Stage 3 run metadata")
    _validate_existing_identity(metadata, training_identity)
    if metadata.get("stage") != "stage3" or metadata.get("operation") != "train":
        raise ValueError("Existing output is not a Stage 3 train run")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("fold") != fold:
        raise ValueError("Existing Stage 3 run fold does not match its directory")
    if not (root / "run_config.yaml").is_file():
        raise FileNotFoundError("Stage 3 run is missing run_config.yaml")

    metrics = _history_epochs(root / "metrics.jsonl", context="Stage 3 metrics history")
    diagnostics = _history_epochs(
        root / "diagnostics.jsonl", context="Stage 3 diagnostics history"
    )
    _require_continuous_history(metrics, context="Stage 3 metrics")
    _require_continuous_history(diagnostics, context="Stage 3 diagnostics")
    if metrics != diagnostics:
        raise ValueError("Stage 3 metrics and diagnostics histories end at different epochs")

    status = metadata.get("status")
    checkpoints = _checkpoint_epochs(root)
    if status == "completed":
        summary = _read_json(root / "summary.json", context="Stage 3 run summary")
        final = summary.get("final_epoch")
        if summary.get("fold") != fold or not isinstance(final, Mapping):
            raise ValueError("Completed Stage 3 summary does not describe the requested fold")
        if final.get("epoch") != total_epochs or metrics != list(
            range(1, total_epochs + 1)
        ):
            raise ValueError("Completed Stage 3 run does not contain the full epoch history")
        final_checkpoint = root / f"checkpoint_epoch_{total_epochs:05d}.pt"
        if not final_checkpoint.is_file():
            raise FileNotFoundError("Completed Stage 3 run is missing its final checkpoint")
        return "skipped", None

    if status not in {"failed", "running"}:
        raise ValueError(f"Stage 3 run has unsupported status: {status!r}")
    if not metrics or not checkpoints:
        raise ValueError("Stage 3 run has no legal complete checkpoint to resume")
    latest_history = metrics[-1]
    latest_checkpoint = checkpoints[-1]
    if latest_checkpoint != latest_history:
        raise ValueError(
            "Stage 3 run has no legal resume point: checkpoint and history tails differ"
        )
    return "resume", root / f"checkpoint_epoch_{latest_checkpoint:05d}.pt"


def _run_fold(
    *,
    config_path: str,
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

    from stage3.config import configure_process_runtime, load_stage3_config

    config = load_stage3_config(config_path)
    configure_process_runtime(config)

    from common.outputs import (
        open_run_directory,
        repository_path,
        repository_relative,
    )
    from stage3.train import resolve_stage3_training_identity, run_stage3_training

    training_identity = resolve_stage3_training_identity(config, fold)
    fold_output = Path(output_root) / f"fold{fold}"
    fold_root = repository_path(fold_output)
    resume_from: Path | None = None
    if fold_root.exists():
        if not resume:
            raise FileExistsError(
                f"Output already exists: {repository_relative(fold_root)}"
            )
        action, resume_from = _resume_action(
            fold_root,
            fold=fold,
            total_epochs=config.training.epochs,
            training_identity=training_identity,
        )
        if action == "skipped":
            return "skipped"

    run = open_run_directory(
        stage="stage3",
        operation="train",
        config_path=config_path,
        config_payload=config.to_dict(),
        output=fold_output,
        seed=config.data.seed,
        semantic_identity=training_identity,
        resume=resume_from,
        details={
            "fold": fold,
            "stage2_encoder": repository_relative(
                config.initialization.stage2_encoder
            ),
            "assigned_device": device or config.training.device,
        },
    )
    try:
        rows = run_stage3_training(
            config,
            fold,
            output_dir=run.root,
            resume_from=resume_from,
            expected_training_identity=training_identity,
        )
        final_epoch = rows[-1] if rows else _last_history_row(run.root / "metrics.jsonl")
        run.complete({"fold": fold, "final_epoch": final_epoch})
    except BaseException:
        run.fail()
        raise
    return "completed"


def _worker_entry(
    config_path: str,
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
    *,
    config_path: str,
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
                fold,
                output_root,
                resume,
                slot_devices[slot],
                fold == folds[0],
                result_queue,
            ),
            name=f"stage3-fold{fold}",
        )
        process.start()
        running[slot] = (process, fold, result_queue)
        print(f"fold{fold} started", flush=True)

    try:
        for slot in range(effective_parallel):
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one or more Stage 3 folds with bounded process concurrency."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", required=True, type=int, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-parallel", type=_positive_int, default=1)
    parser.add_argument(
        "--devices",
        help="optional comma-separated GPU list such as cuda:0,cuda:1",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    from stage3.config import load_stage3_config, validate_stage3_folds

    try:
        folds = validate_stage3_folds(args.fold)
        devices = _parse_devices(args.devices)
    except ValueError as error:
        parser.error(str(error))

    from common.outputs import repository_path
    config = load_stage3_config(args.config)
    repository_path(args.output)
    effective_parallel = min(args.max_parallel, len(folds))
    if effective_parallel > 1 and not devices:
        parser.error("--devices is required when more than one fold runs in parallel")
    if devices and config.training.device != "cuda":
        parser.error("--devices requires training.device: cuda")

    try:
        results = _run_schedule(
            config_path=args.config,
            folds=folds,
            output_root=args.output,
            resume=args.resume,
            max_parallel=args.max_parallel,
            devices=devices,
        )
    except KeyboardInterrupt:
        print("Stage3 fold training interrupted", file=sys.stderr)
        return 130

    print("Stage3 folds complete")
    for status in ("completed", "skipped", "failed"):
        selected = [str(fold) for fold in folds if results.get(fold) == status]
        print(f"{status}: {', '.join(selected) if selected else '-'}")
    return 1 if any(status == "failed" for status in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
