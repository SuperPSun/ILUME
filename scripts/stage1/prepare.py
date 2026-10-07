from __future__ import annotations

import argparse
import json
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


def prepare_charge_run(config, config_path, output):
    from stage1.partial_charge import partial_charge_recipe, prepare_partial_charge
    metadata = json.loads((config.data.artifacts_dir / "metadata.json").read_text())
    recipe = partial_charge_recipe(config, metadata)
    run = open_run_directory(stage="stage1", operation="partial_charge_prepare",
                             config_path=config_path, config_payload={"partial_charge_recipe": recipe},
                             semantic_identity=recipe, output=output, seed=config.data.seed,
                             reusable=True)
    try:
        result = prepare_partial_charge(config, run.artifacts)
        run.complete(result)
        return result
    except BaseException:
        run.fail()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage 1 data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--partial-charge-only", action="store_true",
                        help="Prepare train-only atom-label sidecar using the existing corpus.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.workers is not None:
        config = replace(
            config,
            preparation=replace(config.preparation, workers=args.workers),
        )
        config.validate()
    if args.partial_charge_only:
        prepare_charge_run(config, args.config, args.output)
        return
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
        result = prepare_corpus(
            effective,
            source_identity=source_identity,
            performance_path=run.root / "performance.json",
            input_identity_elapsed_seconds=identity_elapsed,
        )
        if effective.loss.lambda_partial_charge > 0:
            if effective.auxiliary.partial_charge_cache.name != "artifacts":
                raise ValueError("CLI partial_charge_cache must identify the sidecar run's artifacts directory")
            result["partial_charge"] = prepare_charge_run(effective, args.config, effective.auxiliary.partial_charge_cache.parent)
        run.complete(result)
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
