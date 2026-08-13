from __future__ import annotations

import argparse

from ..stage3_config import load_stage3_config
from ..stage3_runtime import configure_stage3_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one fold of single-device Stage 3 multitask training.")
    parser.add_argument("--config", required=True, help="Path to a Stage 3 YAML config")
    parser.add_argument("--fold", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from ..stage3_training import run_stage3_training, with_stage3_overrides

    config = with_stage3_overrides(
        load_stage3_config(args.config),
        output_dir=args.output_dir,
        resume_from=args.resume_from,
    )
    configure_stage3_runtime(config)
    run_stage3_training(config, args.fold)


if __name__ == "__main__":
    main()
