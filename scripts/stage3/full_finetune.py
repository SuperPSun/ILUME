from __future__ import annotations

import argparse
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
    config_path: str, feature_dir: str, output_root: str, fold: int,
    resume: bool, device: str | None, show_progress: bool,
) -> str:
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
    if not show_progress:
        os.environ["ILUME_DISABLE_PROGRESS"] = "1"

    from ablations.stage3_full_finetune.representation import load_config
    from ablations.stage3_full_finetune.train import run_finetuning
    from stage3.config import configure_process_runtime

    config, recipe = load_config(config_path)
    configure_process_runtime(config)
    root = Path(output_root) / f"fold{fold}"
    if root.exists() and not resume:
        raise FileExistsError(f"Fine-tuning output already exists: {root}")
    completed = (root / "three_phase_final.pt").is_file()
    run_finetuning(
        config, recipe, fold=fold, feature_dir=feature_dir,
        output_dir=root, resume=resume and root.exists(),
    )
    return "skipped" if completed else "completed"


def _worker_entry(
    config_path: str, feature_dir: str, output_root: str, fold: int,
    resume: bool, device: str | None, show_progress: bool, result_queue: Any,
) -> None:
    try:
        status = _run_fold(
            config_path, feature_dir, output_root, fold, resume, device, show_progress
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
    *, config_path: str, feature_dir: str, folds: tuple[int, ...],
    output_root: str, resume: bool, max_parallel: int,
    devices: tuple[str, ...],
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
            args=(
                config_path, feature_dir, output_root, fold, resume,
                slot_devices[slot], fold == folds[0], result_queue,
            ),
            name=f"stage3-full-finetune-fold{fold}",
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated Stage 3 Stage1/ObjectEncoder fine-tuning ablation."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Build immutable molecular input features.")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--output", required=True)
    train = commands.add_parser("train", help="Train selected folds with live Phase 1 encoders.")
    train.add_argument("--config", required=True)
    train.add_argument("--feature-dir", required=True)
    train.add_argument("--fold", type=int, nargs="+", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--max-parallel", type=_positive_int, default=1)
    train.add_argument("--devices", help="comma-separated GPU list such as cuda:0,cuda:1")
    evaluate = commands.add_parser("evaluate", help="Evaluate final fine-tuned folds.")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--feature-dir", required=True)
    evaluate.add_argument("--checkpoint-dir", required=True)
    evaluate.add_argument("--split", choices=("valid", "test"), required=True)
    evaluate.add_argument("--fold", type=int, nargs="+")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--historical-base-root", required=True)
    return parser


def _evaluation_identity(
    config: Any, recipe: Any, checkpoint_dir: Path, *, split: str,
    fold: int | None,
) -> dict[str, Any]:
    import json

    from common.identity import semantic_identity

    folds = (fold,) if fold is not None else range(1, 6)
    anchors = []
    for current_fold in folds:
        manifest = json.loads(
            (checkpoint_dir / f"fold{current_fold}" / "three_phase_final.json").read_text(
                encoding="utf-8"
            )
        )
        anchors.append({
            "fold": current_fold,
            "artifact_sha256": manifest["artifact_sha256"],
            "training_identity": manifest["training_identity"]["hash"],
        })
    return semantic_identity(
        "stage3.full-finetune-evaluation.v1",
        {
            "config": config.to_dict(), "recipe": recipe.to_dict(),
            "split": split, "fold": fold, "anchors": anchors,
        },
    )


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    from ablations.stage3_full_finetune.representation import load_config
    from stage3.config import configure_process_runtime, validate_stage3_folds

    config, recipe = load_config(args.config)
    if args.command == "prepare":
        from ablations.stage3_full_finetune.representation import prepare_features

        configure_process_runtime(config)
        prepare_features(config, args.output)
    elif args.command == "train":
        folds = validate_stage3_folds(args.fold)
        try:
            devices = _parse_devices(args.devices)
        except ValueError as error:
            parser.error(str(error))
        if min(args.max_parallel, len(folds)) > 1 and not devices:
            parser.error("--devices is required when more than one fold runs in parallel")
        if devices and config.training.device != "cuda":
            parser.error("--devices requires training.device: cuda")
        root = Path(args.output)
        if root.exists() and not args.resume:
            raise FileExistsError(f"Fine-tuning output root already exists: {root}")
        if args.resume and not root.is_dir():
            raise FileNotFoundError(f"Fine-tuning resume root is missing: {root}")
        try:
            results = _run_schedule(
                config_path=args.config, feature_dir=args.feature_dir,
                folds=folds, output_root=args.output, resume=args.resume,
                max_parallel=args.max_parallel, devices=devices,
            )
        except KeyboardInterrupt:
            print("Stage 3 full fine-tuning interrupted", file=sys.stderr)
            return 130
        for status in ("completed", "skipped", "failed"):
            selected = [str(fold) for fold in folds if results.get(fold) == status]
            print(f"{status}: {', '.join(selected) if selected else '-'}")
        return 1 if any(status == "failed" for status in results.values()) else 0
    else:
        from ablations.stage3_full_finetune.evaluate import evaluate_finetuned
        from common.outputs import open_run_directory, repository_relative
        from common.reporting import REPORTING_SCHEMA_VERSION

        configure_process_runtime(config)
        if args.split == "valid":
            folds = validate_stage3_folds(args.fold or ())
        elif args.fold is not None:
            raise ValueError("Test evaluation always uses all five folds; omit --fold")
        else:
            folds = (None,)
        root = Path(args.output)
        if root.exists():
            raise FileExistsError(f"Fine-tuning evaluation output already exists: {root}")
        root.mkdir(parents=True)
        for fold in folds:
            destination = root / (f"fold{fold}" if fold is not None else "test")
            run = open_run_directory(
                stage="stage3", operation="evaluate",
                config_path=args.config, config_payload=config.to_dict(),
                semantic_identity=_evaluation_identity(
                    config, recipe, Path(args.checkpoint_dir),
                    split=args.split, fold=fold,
                ),
                output=destination, seed=config.data.seed,
                details={
                    "reporting_schema_version": REPORTING_SCHEMA_VERSION,
                    "ablation": "stage3_full_finetune",
                    "checkpoint_dir": repository_relative(args.checkpoint_dir),
                    "split": args.split, "fold": fold,
                },
            )
            try:
                result = evaluate_finetuned(
                    config, recipe, feature_dir=args.feature_dir,
                    checkpoint_dir=args.checkpoint_dir,
                    split=args.split, fold=fold,
                    predictions_dir=run.root / "predictions",
                    historical_base_root=args.historical_base_root,
                )
                run.complete(result)
            except BaseException:
                run.fail()
                raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
