from __future__ import annotations

import argparse
import json

from ..config import load_config
from ..data import prepare_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Stage 1 molecular entities for multimodal pretraining."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    summary = prepare_corpus(config)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
