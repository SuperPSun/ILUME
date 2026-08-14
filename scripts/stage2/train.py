from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.outputs import open_run_directory, repository_path, repository_relative
from common.training import resolve_device
from stage2.config import load_stage2_config
from stage2.runtime import configure_stage2_math
from stage2.train import run_stage2_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 2.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_stage2_config(args.config)
    device = resolve_device(config.training.device)
    math_contract = configure_stage2_math(device)
    run = open_run_directory(
        stage="stage2", operation="train", config_path=args.config,
        config_payload=config.to_dict(), output=args.output, seed=config.data.seed,
        resume=args.resume,
        details={
            "checkpoint": repository_relative(config.initialization.checkpoint),
            "math_contract": math_contract,
            "optimizer_implementation": (
                "fused" if device.type == "cuda" else "single_tensor"
            ),
            "execution_contract": {
                "entity_loading": "preload",
                "pin_memory": device.type == "cuda",
                "non_blocking_h2d": device.type == "cuda",
                "teacher_dtype": "float32",
                "validation": "inference_mode",
            },
        },
        ignored_config_sections={"preparation"},
        ignored_config_fields={
            "training.packing_workers",
            "training.packing_prefetch_windows",
            "training.log_every_batches",
        },
    )
    try:
        run_stage2_training(
            config, output_dir=run.root,
            resume_from=repository_path(args.resume) if args.resume else None,
        )
        summary = json.loads((run.root / "final_metrics.json").read_text(encoding="utf-8"))
        run.complete(summary)
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
