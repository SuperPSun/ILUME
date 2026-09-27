from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.outputs import open_run_directory, repository_path, repository_relative
from common.reporting import REPORTING_SCHEMA_VERSION
from stage2.evaluate import evaluate_home_final, resolve_evaluation_identity
from stage2.home_config import load_home_recipe


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the formal Stage 2 HoME final model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    recipe = load_home_recipe(args.config)
    checkpoint_dir = repository_path(args.checkpoint_dir)
    identity = resolve_evaluation_identity(recipe, checkpoint_dir, split=args.split)
    run = open_run_directory(
        stage="stage2", operation="evaluate", config_path=args.config,
        config_payload=recipe.to_dict(), output=args.output, seed=recipe.stage2.data.seed,
        semantic_identity=identity,
        details={
            "reporting_schema_version": REPORTING_SCHEMA_VERSION,
            "checkpoint_dir": repository_relative(checkpoint_dir),
            "split": args.split, "model_selector": "stage2_final",
            "tasks": identity["payload"]["tasks"],
        },
    )
    try:
        summary = evaluate_home_final(
            recipe, checkpoint_dir, split=args.split,
            predictions_dir=run.root / "predictions", expected_identity=identity,
        )
        run.complete(summary)
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
