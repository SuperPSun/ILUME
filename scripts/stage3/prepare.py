from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.data_identity import write_data_identity
from common.outputs import open_run_directory, repository_relative
from stage3.config import load_stage3_config
from stage3.runtime import configure_stage3_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage 3 data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_stage3_config(args.config)
    configure_stage3_runtime(config)
    from stage3.data import source_path
    from stage3.prepare import prepare_stage3

    run = open_run_directory(
        stage="stage3", operation="prepare", config_path=args.config,
        config_payload=config.to_dict(), output=args.output, seed=config.data.seed,
        reusable=True,
        details={"checkpoint": repository_relative(config.initialization.stage2_checkpoint)},
    )
    effective = replace(config, data=replace(config.data, artifacts_dir=run.artifacts))
    sources = [source_path(effective, task, fold) for task in effective.tasks for fold in range(1, 6)]
    sources.extend(
        path for task in effective.tasks
        if (path := effective.data.stage3_dir / task / "test.csv").is_file()
    )
    try:
        write_data_identity(ROOT, "stage3", sources)
        result = prepare_stage3(effective)
        run.complete({domain: value["summary"] for domain, value in result.items()})
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
