from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.data_identity import write_data_identity
from common.outputs import open_run_directory, repository_relative
from stage3.config import configure_process_runtime, load_stage3_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage 3 data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_stage3_config(args.config)
    configure_process_runtime(config)
    from stage3.data import collect_object_keys, resolve_task_registry, source_hashes
    from stage3.identity import resolve_stage3_prepared_identity
    from stage3.prepare import prepare_stage3

    registry = resolve_task_registry(config)
    objects = collect_object_keys(config, registry)
    prepared_identity = resolve_stage3_prepared_identity(config, registry, objects)
    run = open_run_directory(
        stage="stage3", operation="prepare", config_path=args.config,
        config_payload=config.to_dict(), output=args.output, seed=config.data.seed,
        semantic_identity=prepared_identity,
        reusable=False,
        details={
            "stage2_encoder": repository_relative(
                config.initialization.stage2_encoder
            )
        },
    )
    effective = replace(config, data=replace(config.data, artifacts_dir=run.artifacts))
    registry = resolve_task_registry(effective)
    sources = [Path(path) for path in source_hashes(effective, registry)]
    try:
        write_data_identity(ROOT, "stage3", sources)
        result = prepare_stage3(effective)
        run.complete(result)
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
