from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.progress import ProgressReporter
from common.outputs import open_run_directory, repository_path, repository_relative
from common.reporting import REPORTING_SCHEMA_VERSION
from common.training import resolve_device
from stage2.config import load_stage2_config
from stage2.runtime import configure_stage2_math


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 checkpoints on the physics test benchmark.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-epoch", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_stage2_config(args.config)
    device = resolve_device(config.training.device)
    math_contract = configure_stage2_math(device)
    from stage2.evaluate import (
        evaluate_stage2_checkpoints,
        resolve_checkpoint_path,
        resolve_stage2_evaluation_identity,
    )

    progress = ProgressReporter()

    checkpoint_dir = repository_path(args.checkpoint_dir)
    checkpoint_path = resolve_checkpoint_path(
        checkpoint_dir,
        args.checkpoint_epoch,
    )

    with progress.status("Resolving Stage 2 evaluation identity"):
        evaluation_identity = resolve_stage2_evaluation_identity(
            config,
            checkpoint_dir,
            checkpoint_epoch=args.checkpoint_epoch,
        )
    run = open_run_directory(
        stage="stage2",
        operation="evaluate",
        config_path=args.config,
        config_payload=config.to_dict(),
        semantic_identity=evaluation_identity,
        output=args.output,
        seed=config.data.seed,
        details={
            "reporting_schema_version": REPORTING_SCHEMA_VERSION,
            "checkpoint_dir": repository_relative(checkpoint_dir),
            "checkpoint": repository_relative(checkpoint_path),
            "checkpoint_epoch": int(checkpoint_path.stem.rsplit("_", 1)[1]),
            "split": "test",
            "math_contract": math_contract,
        },
    )
    try:
        result = evaluate_stage2_checkpoints(
            config,
            checkpoint_dir,
            checkpoint_epoch=args.checkpoint_epoch,
            predictions_dir=run.root / "predictions",
            expected_evaluation_identity=evaluation_identity,
            reporter=progress,
        )
        run.complete(result)
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
