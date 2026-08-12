from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.outputs import open_run_directory, repository_path, repository_relative
from stage3.config import load_stage3_config
from stage3.runtime import configure_stage3_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--split", required=True, choices=("valid", "test"))
    parser.add_argument("--ensemble-folds", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_stage3_config(args.config)
    configure_stage3_runtime(config)
    from stage3.evaluate import evaluate_checkpoints

    run = open_run_directory(
        stage="stage3", operation="evaluate", config_path=args.config,
        config_payload=config.to_dict(), output=args.output, seed=config.data.seed,
        details={
            "checkpoint_dir": repository_relative(args.checkpoint_dir),
            "split": args.split,
            "ensemble_folds": args.ensemble_folds,
        },
    )
    try:
        result = evaluate_checkpoints(
            config, repository_path(args.checkpoint_dir), split=args.split,
            ensemble_folds=args.ensemble_folds,
        )
        run.complete(result)
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
