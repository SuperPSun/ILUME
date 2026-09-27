from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.outputs import open_run_directory, repository_path, repository_relative
from common.training import resolve_device
from stage2.config import load_stage2_config
from stage2.home_config import load_home_recipe
from stage2.home_train import resolve_training_identity as resolve_home_training_identity, train_stage2_home
from stage2.runtime import configure_stage2_math
from stage2.train import resolve_stage2_training_identity, run_stage2_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 2.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", help="single GPU such as cuda:0")
    args = parser.parse_args()
    if args.device is not None:
        if not re.fullmatch(r"cuda:\d+", args.device):
            parser.error("--device must be a single GPU such as cuda:0")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]
    is_home = "home" in (yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {})
    if is_home:
        recipe = load_home_recipe(args.config)
        config = recipe.stage2
        device = resolve_device(config.training.device)
        math_contract = configure_stage2_math(device)
        identity = resolve_home_training_identity(recipe, math_contract)
        if args.resume and Path(args.resume).resolve() != Path(args.output).resolve():
            parser.error("Stage 2 HoME --resume must name the output directory")
        run = open_run_directory(
            stage="stage2", operation="train", config_path=args.config,
            config_payload=recipe.to_dict(), semantic_identity=identity,
            output=args.output, seed=config.data.seed, resume=args.resume,
            details={"math_contract": math_contract, "optimizer_implementation":
                     "fused" if device.type == "cuda" else "single_tensor"},
        )
        try:
            result = train_stage2_home(recipe, run.root, resume=bool(args.resume))
            run.complete(result)
        except BaseException:
            run.fail()
            raise
        return
    config = load_stage2_config(args.config)
    device = resolve_device(config.training.device)
    math_contract = configure_stage2_math(device)
    training_identity = resolve_stage2_training_identity(config)
    run = open_run_directory(
        stage="stage2", operation="train", config_path=args.config,
        config_payload=config.to_dict(), semantic_identity=training_identity,
        output=args.output, seed=config.data.seed,
        resume=args.resume,
        details={
            **(
                {"representation": "rdkit_2d_mlp"}
                if config.representation is not None
                else {
                    "checkpoint": repository_relative(
                        config.initialization.checkpoint
                    )
                }
            ),
            "math_contract": math_contract,
            "optimizer_implementation": (
                "fused" if device.type == "cuda" else "single_tensor"
            ),
            "execution_contract": {
                "entity_loading": "preload",
                "pin_memory": device.type == "cuda",
                "non_blocking_h2d": device.type == "cuda",
                **(
                    {}
                    if config.representation is not None
                    else {"teacher_dtype": "float32"}
                ),
                "validation": "inference_mode",
            },
        },
    )
    try:
        run_stage2_training(
            config, output_dir=run.root,
            resume_from=repository_path(args.resume) if args.resume else None,
            expected_training_identity=training_identity,
        )
        summary = json.loads((run.root / "final_metrics.json").read_text(encoding="utf-8"))
        run.complete(summary)
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
