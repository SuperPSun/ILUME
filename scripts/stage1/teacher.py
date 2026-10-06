from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stage1.auxiliary import prepare_teacher_cache
from stage1.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Offline Uni-Mol2 84M cache for v4 Stage1; no automatic downloads.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1, help="CPU conformer/feature workers; reserve matching Slurm CPUs (default: serial)")
    parser.add_argument("--limit", type=int, help="Small audit only; requires a separate --output")
    parser.add_argument("--output", help="Optional cache root; audit roots cannot equal the configured formal cache")
    args = parser.parse_args()
    print(json.dumps(prepare_teacher_cache(load_config(args.config), device=args.device, batch_size=args.batch_size, workers=args.workers, limit=args.limit, output=args.output), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
