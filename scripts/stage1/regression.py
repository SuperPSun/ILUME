from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.outputs import open_run_directory
from stage1.regression import REGRESSION_TASKS, load_regression_config, regression_identity, run_regression, prepare_regression_cache


def main():
    parser = argparse.ArgumentParser(description="Train independent regression heads on frozen Stage1 representations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Complete final Stage1 checkpoint, not encoder-only export.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+", choices=REGRESSION_TASKS)
    parser.add_argument("--prepare-cache", action="store_true", help="Prepare train/valid frozen inputs without training heads.")
    parser.add_argument("--device", help="Execution-only override of YAML device.")
    args = parser.parse_args()
    config = load_regression_config(args.config)
    if config.protocol == "hpo_v4":
        from common.hpo_protocol import validate_hpo_paths
        validate_hpo_paths(args.config, args.output)
    if args.device is not None:
        config = replace(config, device=args.device)
    identity = regression_identity(config, args.checkpoint, args.tasks)
    run = open_run_directory(stage="stage1", operation="regression", config_path=args.config,
                             config_payload=config.to_dict(), semantic_identity=identity,
                             output=args.output, seed=config.seed)
    try:
        started = time.monotonic()
        result = (prepare_regression_cache(config, args.checkpoint, run.artifacts, args.tasks) if args.prepare_cache else
                  run_regression(config, args.checkpoint, run.root, args.tasks))
        if config.protocol == "hpo_v4":
            import torch
            from common.training import resolve_device
            device = resolve_device(config.device)
            result["execution_cost"] = {"wall_seconds": time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
        run.complete(result)
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
