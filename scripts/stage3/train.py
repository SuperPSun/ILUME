from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.outputs import open_run_directory, repository_path, repository_relative
from stage3.config import load_stage3_config
from stage3.runtime import configure_stage3_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Stage 3 fold.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_stage3_config(args.config)
    configure_stage3_runtime(config)
    from stage3.train import run_stage3_training

    run = open_run_directory(
        stage="stage3", operation="train", config_path=args.config,
        config_payload=config.to_dict(), output=args.output, seed=config.data.seed,
        resume=args.resume,
        details={
            "fold": args.fold,
            "checkpoint": repository_relative(
                config.initialization.stage2_checkpoint
            ),
        },
    )
    try:
        rows = run_stage3_training(
            config, args.fold, output_dir=run.root,
            resume_from=repository_path(args.resume) if args.resume else None,
        )
        final_by_domain = {
            domain: next(
                row for row in reversed(rows) if row["domain"] == domain
            )
            for domain in config.active_domains
            if any(row["domain"] == domain for row in rows)
        }
        run.complete(
            {"fold": args.fold, "domains": final_by_domain}
            if rows else {"status": "already_complete", "fold": args.fold}
        )
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
