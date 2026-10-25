from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.paper import publish_paper_summary
from common.outputs import repository_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build independent paper comparison tables and figures from complete predictions.")
    parser.add_argument("--config", default="configs/v4/stage3/optimized_3407.yaml", help="Matching Stage3 YAML; only task and split definitions are used.")
    parser.add_argument("--ilume", required=True, help="Explicitly selected ILUME run root containing valid/fold1..5.")
    parser.add_argument("--baselines", required=True, help="Root containing all ten completed catalog_v4_3407 baselines.")
    parser.add_argument("--with-test", action="store_true", help="Also require and summarize ILUME test/ and baseline five-model ensemble test predictions.")
    parser.add_argument("--output", required=True, help="New repository-relative output directory; existing destinations are rejected.")
    args = parser.parse_args()
    publish_paper_summary(
        repository_path(args.config), repository_path(args.ilume),
        repository_path(args.baselines), repository_path(args.output), ROOT,
        with_test=args.with_test,
    )
    print(f"Paper report written to {args.output}")


if __name__ == "__main__":
    main()
