from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.outputs import open_run_directory
from common.identity import semantic_identity
from stage1.regression import load_regression_config
from stage1.regression_test import TEST_TASKS, evaluate_locked_head
from stage3.hpo.contracts import validate_lock


def main():
    parser = argparse.ArgumentParser(description="锁定后 Stage1 后训练头独立终测。")
    parser.add_argument("--lock-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", choices=TEST_TASKS, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    from common.hpo_protocol import validate_hpo_paths
    validate_hpo_paths(args.config, args.output)
    lock = validate_lock(args.lock_manifest)
    entry = lock["heads"][args.target]
    if Path(args.config) != Path(entry["config"]):
        raise ValueError("Test config is outside the locked head recipe")
    config = load_regression_config(args.config)
    if args.device:
        config = replace(config, device=args.device)
    identity = semantic_identity("stage1.hpo_v4.locked-test", {"lock": lock["hash"], "task": args.target})
    run = open_run_directory(stage="stage1", operation="regression_test", config_path=args.config,
            config_payload=config.to_dict(), semantic_identity=identity, output=args.output, seed=config.seed)
    try:
        run.complete(evaluate_locked_head(config, lock["source"]["stage1_checkpoint"], entry["head"], run.root))
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
