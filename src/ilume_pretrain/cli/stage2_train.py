from __future__ import annotations

import argparse

from ..stage2_config import load_stage2_config
from ..stage2_training import run_stage2_training, with_stage2_overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run single-device Stage 2 property-alignment training."
    )
    parser.add_argument("--config", required=True, help="Path to a Stage 2 YAML config")
    parser.add_argument("--lambda-alignment", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = with_stage2_overrides(
        load_stage2_config(args.config),
        lambda_alignment=args.lambda_alignment,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
    )
    run_stage2_training(config)


if __name__ == "__main__":
    main()
