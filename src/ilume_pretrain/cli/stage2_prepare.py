from __future__ import annotations

import argparse

from ..stage2_config import load_stage2_config
from ..stage2_prepare import prepare_teacher_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Stage 2 task artifacts and frozen teacher embeddings."
    )
    parser.add_argument("--config", required=True, help="Path to a Stage 2 YAML config")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_teacher_cache(load_stage2_config(args.config))


if __name__ == "__main__":
    main()
