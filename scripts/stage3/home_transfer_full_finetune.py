from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def worker_entry(
    config_path: str, source_dir: str, output_root: str, fold: int,
    resume: bool, device: str | None, show_progress: bool, result_queue: Any,
) -> None:
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
    if not show_progress:
        os.environ["ILUME_DISABLE_PROGRESS"] = "1"
    try:
        from ablations.stage2_home_transfer.full_finetune import load_config, train_fold
        from stage3.config import configure_process_runtime

        experiment = load_config(config_path, source_dir=source_dir, output_root=output_root)
        configure_process_runtime(experiment.stage3)
        completed = (experiment.checkpoint_dir / f"fold{fold}" / "three_phase_final.pt").is_file()
        train_fold(experiment, fold, resume=resume)
        result_queue.put(("skipped" if completed else "completed", None, None))
    except BaseException as error:
        result_queue.put(("failed", type(error).__name__, str(error)))
        traceback.print_exc()
        raise SystemExit(1) from error


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune live Stage1/ObjectEncoder after Stage2-HoME transfer.")
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--config", default="configs/ablations/stage2_home_transfer_full_finetune.yaml")
    parser.add_argument("--source-dir", required=True, help="existing Stage2-HoME experiment root (contains stage2/)")
    parser.add_argument("--output", help="isolated experiment root; defaults to YAML output_root")
    parser.add_argument("--fold", type=int, nargs="+")
    parser.add_argument("--split", choices=("valid", "test"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--devices", help="comma-separated CUDA devices such as cuda:0,cuda:1")
    args = parser.parse_args()
    if args.max_parallel < 1:
        parser.error("--max-parallel must be at least 1")
    if args.command != "train" and (args.max_parallel != 1 or args.devices is not None):
        parser.error("--max-parallel and --devices are only supported by train")
    if args.fold and (len(args.fold) != len(set(args.fold)) or any(fold not in range(1, 6) for fold in args.fold)):
        parser.error("--fold requires unique values in 1..5")
    from ablations.stage2_home_transfer.full_finetune import (
        evaluate, evaluation_identity, load_config, prepare,
    )
    from stage3.config import configure_process_runtime

    experiment = load_config(args.config, source_dir=args.source_dir, output_root=args.output)
    configure_process_runtime(experiment.stage3)
    if args.command == "prepare":
        if args.fold or args.split or args.resume:
            parser.error("prepare does not accept --fold, --split or --resume")
        print(json.dumps(prepare(experiment), sort_keys=True))
        return 0
    if args.command == "train":
        if not args.fold or args.split:
            parser.error("train requires --fold and does not accept --split")
        devices = tuple(item.strip() for item in args.devices.split(",")) if args.devices else ()
        import re
        if devices and (len(devices) != len(set(devices)) or any(not re.fullmatch(r"cuda:\d+", item) for item in devices)):
            parser.error("--devices requires unique CUDA devices such as cuda:0,cuda:1")
        if min(args.max_parallel, len(args.fold)) > 1 and not devices:
            parser.error("--devices is required for parallel folds")
        if devices and experiment.stage3.training.device != "cuda":
            parser.error("--devices requires training.device: cuda")
        from scripts.stage3.full_finetune import run_fold_schedule

        try:
            results = run_fold_schedule(
                config_path=args.config, feature_dir=str(experiment.source.output_root),
                output_root=str(experiment.output_root), folds=tuple(args.fold),
                resume=args.resume, max_parallel=args.max_parallel, devices=devices,
                worker_entry=worker_entry,
            )
        except KeyboardInterrupt:
            print("HoME full fine-tuning interrupted", file=sys.stderr)
            return 130
        for status in ("completed", "skipped", "failed"):
            folds = [str(fold) for fold in args.fold if results.get(fold) == status]
            print(f"{status}: {', '.join(folds) if folds else '-'}")
        return 1 if any(status == "failed" for status in results.values()) else 0
    if args.resume or args.split is None:
        parser.error("evaluate requires --split and forbids --resume")
    if args.split == "test" and args.fold:
        parser.error("test ensemble forbids --fold")
    if args.split == "valid" and not args.fold:
        parser.error("validation requires --fold")

    import yaml
    from common.outputs import open_run_directory, repository_relative
    from common.reporting import REPORTING_SCHEMA_VERSION

    snapshot = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    snapshot["output_root"] = repository_relative(experiment.output_root)
    snapshot["source_dir"] = repository_relative(experiment.source.output_root)
    for fold in tuple(args.fold) if args.split == "valid" else (None,):
        destination = experiment.output_root / ("valid" if args.split == "valid" else "test")
        if fold is not None:
            destination /= f"fold{fold}"
        run = open_run_directory(
            stage="stage3", operation="evaluate", config_path=args.config,
            config_payload={"experiment": snapshot, "stage3": experiment.stage3.to_dict()},
            semantic_identity=evaluation_identity(experiment, split=args.split, fold=fold),
            output=destination, seed=experiment.stage3.data.seed,
            details={"reporting_schema_version": REPORTING_SCHEMA_VERSION,
                     "ablation": "stage2_home_transfer_full_finetune", "split": args.split, "fold": fold,
                     "checkpoint_dir": repository_relative(experiment.checkpoint_dir)},
        )
        try:
            result = evaluate(experiment, split=args.split, fold=fold, predictions_dir=run.root / "predictions")
            run.complete(result)
        except BaseException:
            run.fail()
            raise
        print(f"validation fold{fold} complete" if fold is not None else "test ensemble complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
