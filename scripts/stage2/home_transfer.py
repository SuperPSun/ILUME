from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

def main() -> None:
    parser = argparse.ArgumentParser(description="Train isolated physics-only Stage2 HoME transfer source.")
    parser.add_argument("--config", default="configs/ablations/stage2_home_transfer.yaml")
    parser.add_argument("--output", help="experiment root; defaults to config output_root")
    parser.add_argument("--device", help="single GPU such as cuda:0 or cuda:4")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.device is not None:
        if not re.fullmatch(r"cuda:\d+", args.device):
            parser.error("--device must be a single GPU such as cuda:0")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]

    from ablations.stage2_home_transfer.config import load_experiment
    from ablations.stage2_home_transfer.stage2 import train_stage2_home

    experiment = load_experiment(args.config)
    if args.output is not None:
        experiment = replace(experiment, output_root=Path(args.output))
    manifest = train_stage2_home(
        experiment, experiment.output_root / "stage2", resume=args.resume,
    )
    print(f"Stage2-HoME final: {experiment.output_root / 'stage2' / manifest['artifact']}")


if __name__ == "__main__":
    main()
