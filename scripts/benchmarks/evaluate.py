from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.config import load_benchmark_config
from benchmarks.common.engine import ensemble_evaluation, evaluate_checkpoint, write_predictions
from benchmarks.common.data import resolve_task
from common.identity import semantic_identity
from common.io import sha256_file
from common.outputs import open_run_directory, repository_path, repository_relative


def _latest_completed(root: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("attempt-*"):
        metadata = path / "metadata.json"
        if not metadata.is_file():
            continue
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        if payload.get("status") == "completed" and (path / "checkpoint.json").is_file():
            try:
                number = int(path.name.split("-", 1)[1])
            except ValueError:
                continue
            candidates.append((number, path))
    if not candidates:
        raise FileNotFoundError(f"No completed benchmark checkpoint under {root}")
    return max(candidates)[1]


def _checkpoint_fingerprint(path: Path) -> dict[str, Any]:
    manifest_path = path / "checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "training_identity": manifest["training_identity"],
        "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "model_integrity": manifest["integrity"],
    }


def _source_hash(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ILUME baseline checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", required=True, choices=("stage3", "stage2_physics"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", required=True, choices=("valid", "test"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--ensemble-folds", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_benchmark_config(args.config)
    if args.benchmark == "stage3" and args.split == "test":
        if not args.ensemble_folds or args.fold is not None or not args.checkpoint_dir or args.checkpoint:
            raise ValueError("Stage 3 test requires --checkpoint-dir and --ensemble-folds only")
        checkpoint_root = repository_path(args.checkpoint_dir)
        checkpoints = [_latest_completed(checkpoint_root / f"fold{fold}") for fold in config.stage3.folds]
        selector_fold = None
    else:
        if args.ensemble_folds or not args.checkpoint or args.checkpoint_dir:
            raise ValueError("Single-fold/Stage 2 evaluation requires exactly --checkpoint")
        if args.benchmark == "stage3" and args.fold not in config.stage3.folds:
            raise ValueError("Stage 3 validation requires a configured --fold")
        if args.benchmark == "stage2_physics" and args.fold is not None:
            raise ValueError("Stage 2 physics evaluation does not accept --fold")
        checkpoints = [repository_path(args.checkpoint)]
        selector_fold = args.fold
    checkpoint_fingerprints = [_checkpoint_fingerprint(path) for path in checkpoints]
    evaluation_task = resolve_task(
        config, args.benchmark, args.task,
        config.stage3.folds[0] if args.ensemble_folds else selector_fold,
    )
    evaluation_source = (
        evaluation_task.test_path
        if args.split == "test"
        else evaluation_task.valid_paths[0]
    )
    evaluation_source_hash = _source_hash(evaluation_source)
    evaluation_identity = semantic_identity(
        "benchmark.evaluation.v1",
        {
            "benchmark_model": config.name,
            "domain": args.benchmark,
            "task": args.task,
            "split": args.split,
            "fold": selector_fold,
            "ensemble_folds": args.ensemble_folds,
            "checkpoints": checkpoint_fingerprints,
            "evaluation_source_sha256": evaluation_source_hash,
        },
    )
    run = open_run_directory(
        stage="benchmark", operation="evaluate", config_path=args.config,
        config_payload=config.to_dict(), semantic_identity=evaluation_identity,
        output=args.output, seed=config.seed,
        data_metadata=["data/task_catalog.csv", "data/stage2/metadata.json"],
        details={
            "benchmark": args.benchmark, "task": args.task, "split": args.split,
            "fold": selector_fold, "ensemble_folds": args.ensemble_folds,
            "checkpoints": [repository_relative(path) for path in checkpoints],
        },
    )
    try:
        results = [
            evaluate_checkpoint(
                config, args.benchmark, args.task,
                fold if args.ensemble_folds else selector_fold,
                checkpoint, args.split,
            )
            for fold, checkpoint in zip(
                config.stage3.folds if args.ensemble_folds else (selector_fold,),
                checkpoints,
                strict=True,
            )
        ]
        if _source_hash(evaluation_source) != evaluation_source_hash:
            raise ValueError("Benchmark evaluation source changed during evaluation")
        if [_checkpoint_fingerprint(path) for path in checkpoints] != checkpoint_fingerprints:
            raise ValueError("Benchmark checkpoint changed during evaluation")
        first = results[0]
        if args.ensemble_folds:
            prediction, ensemble_metrics = ensemble_evaluation(results, tuple(first.metrics))
            summary: dict[str, Any] = {
                "benchmark": args.benchmark, "task": args.task, "split": args.split,
                "folds": {f"fold{fold}": {"targets": result.metrics} for fold, result in zip(config.stage3.folds, results, strict=True)},
                "ensemble": {"targets": ensemble_metrics},
            }
            extras = {f"fold{fold}": result.predictions for fold, result in zip(config.stage3.folds, results, strict=True)}
        else:
            prediction = first.predictions
            summary = {
                "benchmark": args.benchmark, "task": args.task, "split": args.split,
                "fold": selector_fold, "targets": first.metrics,
            }
            extras = None
        write_predictions(
            run.root / "predictions.csv", first.source_rows, tuple(first.metrics),
            first.targets, prediction, extras,
        )
        run.complete(summary)
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
