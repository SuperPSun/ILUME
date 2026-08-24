from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.outputs import open_run_directory, repository_path, repository_relative
from stage3.config import configure_process_runtime, load_stage3_config
from common.progress import ProgressReporter
from common.reporting import REPORTING_SCHEMA_VERSION


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--split", required=True, choices=("valid", "test"))
    parser.add_argument("--ensemble-folds", action="store_true")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--checkpoint-epoch", type=int)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--study-id")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_stage3_config(args.config)
    configure_process_runtime(config)
    from stage3.evaluate import (
        evaluate_checkpoints,
        resolve_stage3_evaluation_identity,
    )

    checkpoint_dir = repository_path(args.checkpoint_dir)
    progress = ProgressReporter()

    with progress.status("Resolving Stage 3 evaluation identity"):
        evaluation_identity = resolve_stage3_evaluation_identity(
            config,
            checkpoint_dir,
            split=args.split,
            ensemble_folds=args.ensemble_folds,
            checkpoint_epoch=args.checkpoint_epoch,
            task_subset=args.tasks,
            fold=args.fold,
        )

    run = open_run_directory(
        stage="stage3", operation="evaluate", config_path=args.config,
        config_payload=config.to_dict(), output=args.output, seed=config.data.seed,
        semantic_identity=evaluation_identity,
        details={
            "reporting_schema_version": REPORTING_SCHEMA_VERSION,
            "checkpoint_dir": repository_relative(args.checkpoint_dir),
            "split": args.split,
            "ensemble_folds": args.ensemble_folds,
            "fold": args.fold,
            "checkpoint_epoch": args.checkpoint_epoch,
            "tasks": args.tasks,
            "reporting_study_id": args.study_id,
        },
    )
    try:
        result = evaluate_checkpoints(
            config, checkpoint_dir, split=args.split,
            ensemble_folds=args.ensemble_folds,
            checkpoint_epoch=args.checkpoint_epoch,
            task_subset=args.tasks,
            fold=args.fold,
            predictions_dir=run.root / "predictions",
            reporting_study_id=args.study_id,
            expected_evaluation_identity=evaluation_identity,
        )
        run.complete(result)
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
