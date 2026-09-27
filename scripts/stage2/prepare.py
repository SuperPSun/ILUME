from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.data_identity import write_data_identity
from common.outputs import open_run_directory, repository_relative
from common.training import resolve_device
from stage2.config import load_stage2_config
from stage2.home_config import load_home_recipe
from stage2.prepare import prepare_stage2_data, prepare_teacher_cache
from stage2.registry import load_stage2_registry
from stage2.runtime import configure_stage2_math
from stage2.identity import build_stage2_data_identity
from stage1.identity import metadata_identity as stage1_metadata_identity


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage 2 representation data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    is_home = "home" in (yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {})
    recipe = load_home_recipe(args.config) if is_home else None
    config = load_stage2_config(args.config)
    device = resolve_device(config.training.device)
    math_contract = configure_stage2_math(device)
    registry = config.resolved_registry(
        load_stage2_registry(config.data.task_catalog_path)
    )
    sources = [config.data.task_catalog_path]
    for spec in registry.tasks:
        sources.extend(
            spec.dataset.split_path(config.data.data_root, split)
            for split in ("train", "valid")
        )
        manifest = spec.dataset.resource_manifest_path(config.data.data_root)
        if manifest is not None:
            sources.append(manifest)
    write_data_identity(
        ROOT,
        "stage2",
        {f"source_{index:05d}": path for index, path in enumerate(sources)},
    )
    rdkit_materialization = None
    if config.representation is not None:
        from stage2.rdkit import resolve_rdkit_stage2_materialization

        rdkit_materialization = resolve_rdkit_stage2_materialization(config)
        data_identity = rdkit_materialization["data_identity"]
    else:
        assert config.data.pretrain_artifacts_dir is not None
        stage1_metadata = json.loads(
            (config.data.pretrain_artifacts_dir / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        feature_identity = stage1_metadata_identity(
            stage1_metadata, "feature", context="Stage 1 feature artifact"
        )
        data_identity = build_stage2_data_identity(config, registry, feature_identity)
    run = open_run_directory(
        stage="stage2", operation="prepare", config_path=args.config,
        config_payload=recipe.to_dict() if recipe else config.to_dict(),
        semantic_identity=data_identity,
        output=args.output, seed=config.data.seed, reusable=True,
        details=(
            {"representation": "rdkit_2d_mlp", "math_contract": math_contract}
            if config.representation is not None
            else {
                **(
                    {"checkpoint": repository_relative(config.initialization.checkpoint)}
                    if config.initialization.checkpoint is not None
                    else {"stage1_config": repository_relative(config.initialization.stage1_config)}
                ),
                "math_contract": math_contract,
                **({} if recipe else {"teacher_dtype": "float32"}),
            }
        ),
    )
    effective = replace(config, data=replace(config.data, artifacts_dir=run.artifacts))
    try:
        result = (prepare_stage2_data(effective) if recipe else
                  prepare_teacher_cache(effective, rdkit_materialization=rdkit_materialization))
        run.complete(result)
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
