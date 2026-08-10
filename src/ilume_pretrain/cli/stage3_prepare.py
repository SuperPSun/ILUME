from __future__ import annotations

import argparse

from ..stage3_config import load_stage3_config
from ..stage3_runtime import configure_stage3_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare isolated Stage 3 v2 IL21 and Aux6 frozen artifacts."
        )
    )
    parser.add_argument("--config", required=True, help="Path to a Stage 3 YAML config")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_stage3_config(args.config)
    configure_stage3_runtime(config)
    from ..stage3_prepare import prepare_stage3

    prepare_stage3(config)


if __name__ == "__main__":
    main()
