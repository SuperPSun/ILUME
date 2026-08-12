from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.data_identity import write_data_identity
from common.outputs import open_run_directory
from stage1.config import load_config
from stage1.prepare import prepare_corpus, preparation_source_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage 1 data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run = open_run_directory(
        stage="stage1", operation="prepare", config_path=args.config,
        config_payload=config.to_dict(), output=args.output, seed=config.data.seed,
        reusable=True,
    )
    effective = replace(config, data=replace(config.data, artifacts_dir=run.artifacts))
    try:
        write_data_identity(ROOT, "stage1", preparation_source_paths(effective))
        run.complete(prepare_corpus(effective))
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
