from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.data_identity import write_data_identity
from common.outputs import open_run_directory
from stage1.config import load_config
from stage1.prepare import prepare_corpus, preparation_source_paths
from stage1.identity import build_stage1_corpus_identity


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage 1 data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.workers is not None:
        config = replace(
            config,
            preparation=replace(config.preparation, workers=args.workers),
        )
        config.validate()
    identity_started = time.perf_counter()
    sources = preparation_source_paths(config)
    source_identity = write_data_identity(
        ROOT,
        "stage1",
        {f"source_{index:05d}": path for index, path in enumerate(sources)},
    )
    corpus_identity = build_stage1_corpus_identity(config, source_identity)
    identity_elapsed = time.perf_counter() - identity_started
    run = open_run_directory(
        stage="stage1", operation="prepare", config_path=args.config,
        config_payload=config.to_dict(), semantic_identity=corpus_identity,
        output=args.output, seed=config.data.seed,
        reusable=True,
    )
    effective = replace(config, data=replace(config.data, artifacts_dir=run.artifacts))
    try:
        run.complete(
            prepare_corpus(
                effective,
                source_identity=source_identity,
                performance_path=run.root / "performance.json",
                input_identity_elapsed_seconds=identity_elapsed,
            )
        )
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
