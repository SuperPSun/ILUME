from __future__ import annotations

import argparse

from ..stage3_config import load_stage3_config
from ..stage3_runtime import configure_stage3_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Stage 3 folds or evaluate a fixed-test five-fold ensemble.")
    parser.add_argument("--config", required=True, help="Path to a Stage 3 YAML config")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--split", required=True, choices=("valid", "test"))
    parser.add_argument("--ensemble-folds", action="store_true")
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_stage3_config(args.config)
    configure_stage3_runtime(config)
    from ..stage3_evaluate import evaluate_checkpoints, write_evaluation

    result = evaluate_checkpoints(
        config,
        args.checkpoint_dir,
        split=args.split,
        ensemble_folds=args.ensemble_folds,
    )
    write_evaluation(result, args.output)


if __name__ == "__main__":
    main()
