from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.outputs import open_run_directory
from stage1.regression import REGRESSION_TASKS, load_regression_config, regression_identity, run_regression


def main():
    parser = argparse.ArgumentParser(description="Train independent regression heads on frozen Stage1 representations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Complete final Stage1 checkpoint, not encoder-only export.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+", choices=REGRESSION_TASKS)
    parser.add_argument("--device", help="Execution-only override of YAML device.")
    args = parser.parse_args()
    config = load_regression_config(args.config)
    if args.device is not None:
        config = replace(config, device=args.device)
    identity = regression_identity(config, args.checkpoint, args.tasks)
    run = open_run_directory(stage="stage1", operation="regression", config_path=args.config,
                             config_payload=config.to_dict(), semantic_identity=identity,
                             output=args.output, seed=config.seed)
    try:
        run.complete(run_regression(config, args.checkpoint, run.root, args.tasks))
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
