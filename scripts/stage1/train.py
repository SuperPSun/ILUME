from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.outputs import open_run_directory, repository_path
from stage1.config import load_config
from stage1.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 1.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_config(args.config)
    run = open_run_directory(
        stage="stage1", operation="train", config_path=args.config,
        config_payload=config.to_dict(), output=args.output, seed=config.data.seed,
        resume=args.resume,
    )
    try:
        rows = run_training(
            config, output_dir=run.root,
            resume_from=repository_path(args.resume) if args.resume else None,
        )
        run.complete(rows[-1] if rows else {"status": "already_complete"})
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
