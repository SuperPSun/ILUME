from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.data_identity import write_data_identity
from common.outputs import open_run_directory, repository_relative
from common.training import resolve_device
from stage2.config import STAGE2_TASKS, load_stage2_config
from stage2.prepare import prepare_teacher_cache
from stage2.runtime import configure_stage2_math


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage 2 data and teacher cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_stage2_config(args.config)
    device = resolve_device(config.training.device)
    math_contract = configure_stage2_math(device)
    run = open_run_directory(
        stage="stage2", operation="prepare", config_path=args.config,
        config_payload=config.to_dict(), output=args.output, seed=config.data.seed,
        reusable=True,
        details={
            "checkpoint": repository_relative(config.initialization.checkpoint),
            "math_contract": math_contract,
            "teacher_dtype": "float32",
        },
        ignored_config_sections={
            "preparation",
            "initialization",
            "model",
            "loss",
            "training",
        },
        ignored_config_fields={
            "data.shard_cache_size",
            "data.teacher_batch_size",
        },
    )
    effective = replace(config, data=replace(config.data, artifacts_dir=run.artifacts))
    sources = [
        effective.data.stage2_dir / task / f"{split}.csv"
        for task in STAGE2_TASKS for split in ("train", "valid")
    ]
    try:
        write_data_identity(ROOT, "stage2", sources)
        run.complete(prepare_teacher_cache(effective))
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
