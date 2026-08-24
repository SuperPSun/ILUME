from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.summary import publish_summary
from common.outputs import repository_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic ILUME and baseline result leaderboards."
    )
    parser.add_argument("--input", default="outputs")
    parser.add_argument("--output", default="summary")
    args = parser.parse_args()
    input_root = repository_path(args.input)
    output = repository_path(args.output)
    if not input_root.is_dir():
        raise FileNotFoundError(f"Summary input directory does not exist: {input_root}")
    publish_summary(input_root, output, ROOT)


if __name__ == "__main__":
    main()
