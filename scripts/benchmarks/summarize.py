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
    parser.add_argument(
        "--input", nargs="+", required=True, metavar="PATH",
        help="One or more repository-relative directories to scan recursively.",
    )
    parser.add_argument(
        "--include", nargs="+", metavar="PATH",
        help=(
            "Optional exact repository-relative directory prefixes to include; "
            "each must be inside an input directory."
        ),
    )
    parser.add_argument("--output", default="summary")
    args = parser.parse_args()
    input_roots = [repository_path(path) for path in args.input]
    include_roots = [repository_path(path) for path in (args.include or ())]
    output = repository_path(args.output)
    publish_summary(
        input_roots, output, ROOT, include_roots=include_roots
    )


if __name__ == "__main__":
    main()
