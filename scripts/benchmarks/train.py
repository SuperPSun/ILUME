from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.config import load_benchmark_config
from benchmarks.common.environment import (
    ensure_benchmark_environment,
    write_environment_snapshot,
)
from benchmarks.common.engine import prepare_training, train_bundle
from common.outputs import open_run_directory
from common.progress import ProgressReporter


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one ILUME baseline task/fold.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", required=True, choices=("stage3", "stage2_physics"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_benchmark_config(args.config)
    environment_snapshot = ensure_benchmark_environment(config)
    reporter = ProgressReporter()
    bundle = prepare_training(
        config, args.benchmark, args.task, args.fold, reporter=reporter
    )
    run = open_run_directory(
        stage="benchmark",
        operation="train",
        config_path=args.config,
        config_payload=config.to_dict(),
        semantic_identity=bundle.training_identity,
        output=args.output,
        seed=config.seed,
        data_metadata=["data/task_catalog.csv", "data/stage2/metadata.json"],
        details={
            "benchmark": args.benchmark,
            "task": args.task,
            "fold": args.fold,
            **(
                {}
                if environment_snapshot is None
                else {
                    "benchmark_environment": environment_snapshot["environment_name"],
                    "environment_lock_sha256": environment_snapshot["environment_lock_sha256"],
                    "chemprop_version": environment_snapshot["direct_versions"]["chemprop"],
                }
            ),
        },
    )
    try:
        if environment_snapshot is not None:
            write_environment_snapshot(
                run.root / "environment.json", environment_snapshot
            )
        result = train_bundle(config, bundle, run.root, reporter=reporter)
        run.complete({"benchmark": args.benchmark, "task": args.task, "fold": args.fold, **result})
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
