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
from benchmarks.common.environment import (
    ensure_benchmark_environment,
    environment_run_details,
    write_environment_snapshot,
)
from benchmarks.common.engine import ensemble_evaluation, evaluate_checkpoint
from benchmarks.common.data import resolve_task
from common.identity import semantic_identity
from common.io import sha256_file
from common.outputs import open_run_directory, repository_path, repository_relative
from common.progress import ProgressReporter
from common.reporting import (
    STAGE2_CORE_EVALUATION_CONTRACT,
    STAGE2_PARTIAL_EVALUATION_CONTRACT,
    REPORTING_SCHEMA_VERSION,
    comparison_identity,
    reporting_block,
    sanitize_task_id,
    write_prediction_csv,
)
from stage2.atom_evaluation import (
    PARTIAL_CHARGE_TASK,
    PARTIAL_CHARGE_UNIT,
    public_partial_charge_score,
    write_partial_charge_predictions,
)


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


def _comparison_fragment(
    task: Any,
    result: Any,
    *,
    split: str,
    fold: int | None,
    ensemble: bool,
    ensemble_results: list[Any] | None = None,
) -> dict[str, Any]:
    if task.benchmark == "stage2_physics":
        if len(task.target_columns) != 1:
            raise ValueError(f"Stage 2 Core task must be scalar: {task.task_id}")
        expected = (task.task_id,)
        sources = {
            f"{task.task_id}:train": sha256_file(task.train_paths[0]),
            f"{task.task_id}:test": sha256_file(task.test_path),
        }
        normalization = {
            task.task_id: {"scale": float(result.target_stats.scale[0])}
        }
        return comparison_identity(
            "stage2_physics",
            split="test",
            expected=expected,
            sources=sources,
            normalization=normalization,
        )
    sources = {
        f"{task.task_id}:fold{path.stem.removeprefix('fold')}": sha256_file(path)
        for path in (*task.train_paths, *task.valid_paths)
    }
    if split == "test":
        sources[f"{task.task_id}:test"] = (
            sha256_file(task.test_path) if task.test_path.is_file() else None
        )
    normalization = (
        {
            f"{task.task_id}:fold{current}": {
                "scale": float(item.target_stats.scale[0])
            }
            for current, item in zip(
                range(1, 6), ensemble_results, strict=True
            )
        }
        if ensemble_results is not None
        else {
            f"{task.task_id}:fold{fold}": {
                "scale": float(result.target_stats.scale[0])
            }
        }
    )
    return comparison_identity(
        "stage3_property",
        split=split,
        expected=(task.task_id,),
        sources=sources,
        normalization=normalization,
        folds=(int(fold),) if fold is not None else tuple(range(1, 6)),
        ensemble=ensemble,
    )


def _write_task_predictions(
    path: Path,
    task: Any,
    result: Any,
    predictions: Any,
    extra_predictions: dict[str, Any] | None,
) -> dict[str, Any]:
    multi_target = len(task.target_columns) > 1
    fields = [
        "source_row",
        *(("source_fold",) if task.benchmark == "stage3" and not extra_predictions else ()),
        *task.slots,
        *task.condition_columns,
        *task.audit_columns,
    ]
    if extra_predictions:
        if multi_target:
            raise ValueError("Ensemble prediction artifacts require a scalar task")
        fields.extend(
            ["target", *(f"prediction_{name}" for name in extra_predictions)]
        )
        fields.extend(("prediction_ensemble", "absolute_error_ensemble"))
    elif multi_target:
        for target in task.target_columns:
            fields.extend(
                (
                    f"{target}_target",
                    f"{target}_prediction",
                    f"{target}_absolute_error",
                )
            )
    else:
        fields.extend(("target", "prediction", "absolute_error"))
    rows: list[dict[str, Any]] = []
    conditions = result.conditions
    if conditions is None:
        raise ValueError("Benchmark evaluation result lacks prediction conditions")
    for index, source in enumerate(result.source_rows):
        try:
            source_row = int(source.rsplit(":", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"Malformed benchmark source row: {source}") from error
        row: dict[str, Any] = {"source_row": source_row}
        if "source_fold" in fields:
            row["source_fold"] = task.fold
        row.update(dict(zip(task.slots, result.components[index], strict=True)))
        row.update(result.audit_rows[index] if result.audit_rows else {})
        row.update(
            {
                name: float(conditions[index, column])
                for column, name in enumerate(task.condition_columns)
            }
        )
        if extra_predictions:
            actual = float(result.targets[index, 0])
            predicted = float(predictions[index, 0])
            row["target"] = actual
            for name, values in extra_predictions.items():
                row[f"prediction_{name}"] = float(values[index, 0])
            row["prediction_ensemble"] = predicted
            row["absolute_error_ensemble"] = abs(predicted - actual)
        elif multi_target:
            for column, target in enumerate(task.target_columns):
                actual = float(result.targets[index, column])
                predicted = float(predictions[index, column])
                row[f"{target}_target"] = actual
                row[f"{target}_prediction"] = predicted
                row[f"{target}_absolute_error"] = abs(predicted - actual)
        else:
            actual = float(result.targets[index, 0])
            predicted = float(predictions[index, 0])
            row["target"] = actual
            row["prediction"] = predicted
            row["absolute_error"] = abs(predicted - actual)
        rows.append(row)
    manifest = write_prediction_csv(path, rows, fields)
    manifest["path"] = f"predictions/{path.name}"
    manifest["task"] = task.task_id
    return manifest


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
    validation_best = config.name == "ilume_stage3_single_task_mlp"
    environment_snapshot = ensure_benchmark_environment(config)
    reporter = ProgressReporter()
    partial_charge = args.task == PARTIAL_CHARGE_TASK
    if partial_charge and (
        config.name != "dmpnn"
        or args.benchmark != "stage2_physics"
        or args.split != "test"
    ):
        raise ValueError("Partial Charge baseline evaluation requires D-MPNN Stage 2 test")
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
    input_audit = None
    if config.name == "molformer":
        from benchmarks.molformer.adapter import molformer_evaluation_audit

        input_audit = molformer_evaluation_audit(
            config,
            args.benchmark,
            args.task,
            config.stage3.folds[0] if args.ensemble_folds else selector_fold,
            args.split,
        )
    if config.name == "ilbert":
        from benchmarks.ilbert.adapter import ilbert_evaluation_audit

        input_audit = ilbert_evaluation_audit(
            config,
            args.benchmark,
            args.task,
            config.stage3.folds[0] if args.ensemble_folds else selector_fold,
            args.split,
        )
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
            "input_audit": input_audit,
        },
    )
    run = open_run_directory(
        stage="benchmark", operation="evaluate", config_path=args.config,
        config_payload=config.to_dict(), semantic_identity=evaluation_identity,
        output=args.output, seed=config.seed,
        data_metadata=["data/task_catalog.csv", "data/stage2/metadata.json"],
        details={
            "reporting_schema_version": REPORTING_SCHEMA_VERSION,
            **(
                {
                    "reporting_contract": (
                        STAGE2_PARTIAL_EVALUATION_CONTRACT
                        if partial_charge
                        else STAGE2_CORE_EVALUATION_CONTRACT
                    )
                }
                if args.benchmark == "stage2_physics" else {}
            ),
            "benchmark": args.benchmark, "task": args.task, "split": args.split,
            "fold": selector_fold, "ensemble_folds": args.ensemble_folds,
            "checkpoints": [repository_relative(path) for path in checkpoints],
            **(
                {"model_selector": "validation_best", "checkpoint_epoch": None}
                if validation_best
                else {}
            ),
            **environment_run_details(environment_snapshot),
        },
    )
    try:
        if environment_snapshot is not None:
            write_environment_snapshot(
                run.root / "environment.json", environment_snapshot
            )
        if partial_charge:
            from benchmarks.dmpnn.adapter import evaluate_dmpnn_partial

            partial = evaluate_dmpnn_partial(config, checkpoints[0])
            if _source_hash(evaluation_source) != evaluation_source_hash:
                raise ValueError("Benchmark evaluation source changed during evaluation")
            if [_checkpoint_fingerprint(path) for path in checkpoints] != checkpoint_fingerprints:
                raise ValueError("Benchmark checkpoint changed during evaluation")
            prediction_manifest = write_partial_charge_predictions(
                run.root / "predictions" / f"{sanitize_task_id(args.task)}.csv",
                partial.benchmark,
                partial.score,
            )
            prediction_manifest["path"] = (
                f"predictions/{sanitize_task_id(args.task)}.csv"
            )
            study = semantic_identity(
                "benchmark.reporting-study.v1",
                {
                    "model": config.name,
                    "config": {
                        key: value
                        for key, value in config.to_dict().items()
                        if key not in {"display_name", "runtime"}
                    },
                },
            )["hash"]
            summary = {
                "benchmark": args.benchmark,
                "task": args.task,
                "split": args.split,
                "stage2_partial_charge_benchmark": {
                    "test": public_partial_charge_score(partial.score)
                },
                "reporting": reporting_block(
                    model_id=config.name,
                    model_display_name=config.display_name,
                    benchmark="stage2_partial_charge",
                    protocol={
                        "split": "test",
                        "folds": [],
                        "ensemble": False,
                        "expected_units": [PARTIAL_CHARGE_UNIT],
                    },
                    comparison=partial.benchmark.comparison_identity,
                    study_id=f"{config.name}-{study}",
                    predictions=[prediction_manifest],
                ),
            }
            summary["reporting"]["contract"] = STAGE2_PARTIAL_EVALUATION_CONTRACT
            run.complete(summary)
            return
        results = [
            evaluate_checkpoint(
                config, args.benchmark, args.task,
                fold if args.ensemble_folds else selector_fold,
                checkpoint, args.split,
                reporter=reporter,
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
        if input_audit is not None and any(
            result.input_audit != input_audit for result in results
        ):
            raise ValueError("Token-baseline evaluation input audit changed during evaluation")
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
        if validation_best:
            summary.update(
                {"model_selector": "validation_best", "checkpoint_epoch": None}
            )
        if input_audit is not None:
            summary["input_audit"] = input_audit
        prediction_manifest = _write_task_predictions(
            run.root / "predictions" / f"{sanitize_task_id(args.task)}.csv",
            evaluation_task,
            first,
            prediction,
            extras,
        )
        comparison = _comparison_fragment(
            evaluation_task,
            first,
            split=args.split,
            fold=selector_fold,
            ensemble=args.ensemble_folds,
            ensemble_results=results if args.ensemble_folds else None,
        )
        study = semantic_identity(
            "benchmark.reporting-study.v1",
            {
                "model": config.name,
                "config": {
                    key: value
                    for key, value in config.to_dict().items()
                    if key not in {"display_name", "runtime"}
                },
            },
        )["hash"]
        summary["reporting"] = reporting_block(
            model_id=config.name,
            model_display_name=config.display_name,
            benchmark=(
                "stage3_property"
                if args.benchmark == "stage3"
                else "stage2_physics"
            ),
            protocol={
                "split": args.split,
                "fold": selector_fold,
                "folds": list(config.stage3.folds) if args.benchmark == "stage3" else [],
                "ensemble": args.ensemble_folds,
                "expected_tasks": [args.task],
                **(
                    {"model_selector": "validation_best", "checkpoint_epoch": None}
                    if validation_best
                    else {}
                ),
            },
            comparison=comparison,
            study_id=f"{config.name}-{study}",
            predictions=[prediction_manifest],
        )
        if args.benchmark == "stage2_physics":
            summary["reporting"]["contract"] = STAGE2_CORE_EVALUATION_CONTRACT
        run.complete(summary)
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
