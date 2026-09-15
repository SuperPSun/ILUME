from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.outputs import open_run_directory, repository_path, repository_relative
from common.progress import ProgressReporter
from common.reporting import REPORTING_SCHEMA_VERSION
from stage3.config import (
    Stage3Config,
    configure_process_runtime,
    load_stage3_config,
    validate_stage3_folds,
)
from stage3.model import ROUTING_MODES


ResolveIdentity = Callable[..., dict[str, Any]]
EvaluateCheckpoints = Callable[..., dict[str, Any]]


def _model_selector(
    config: Stage3Config,
    checkpoint_epoch: int | None,
    gate_calibrated: bool = False,
) -> str:
    if gate_calibrated:
        return "gate_calibrated"
    if checkpoint_epoch is not None:
        return "epoch_checkpoint"
    return (
        "three_phase_final"
        if config.training.schedule_mode == "three_phase"
        else "taskwise_refined"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--gate-calibration-dir",
        help=(
            "Load gate_calibrated.pt from this root while --checkpoint-dir "
            "continues to identify the Flat three-phase anchor."
        ),
    )
    parser.add_argument("--split", required=True, choices=("valid", "test"))
    parser.add_argument("--ensemble-folds", action="store_true")
    parser.add_argument("--fold", type=int, nargs="+")
    parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        help=(
            "Evaluate legacy ordinary fold epoch checkpoints; omitted loads the "
            "config's final artifact. Three-phase configs reject this option."
        ),
    )
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument(
        "--routing-mode",
        choices=ROUTING_MODES,
        default="learned_gate",
        help="Apply an evaluation-only intervention to the Stage 3 task gate.",
    )
    parser.add_argument("--study-id")
    parser.add_argument("--output", required=True)
    return parser


def _validate_request(args: argparse.Namespace) -> tuple[int, ...] | None:
    if args.split == "valid":
        if args.fold is None:
            raise ValueError("Stage 3 validation requires --fold")
        if args.ensemble_folds:
            raise ValueError("Stage 3 validation forbids --ensemble-folds")
        return validate_stage3_folds(args.fold)
    if args.fold is not None:
        raise ValueError("Stage 3 test evaluation forbids --fold")
    if not args.ensemble_folds:
        raise ValueError("Stage 3 test evaluation requires --ensemble-folds")
    return None


def _run_fold(
    *,
    config: Stage3Config,
    config_path: str,
    checkpoint_dir: Path,
    output_root: str,
    fold: int,
    checkpoint_epoch: int | None,
    tasks: list[str] | None,
    routing_mode: str,
    study_id: str,
    progress: ProgressReporter,
    resolve_identity: ResolveIdentity,
    evaluate_checkpoints: EvaluateCheckpoints,
    gate_calibration_dir: Path | None = None,
) -> None:
    model_selector = _model_selector(
        config, checkpoint_epoch, gate_calibration_dir is not None
    )
    with progress.status(f"Resolving Stage 3 fold{fold} evaluation identity"):
        identity_kwargs = dict(
            split="valid",
            ensemble_folds=False,
            checkpoint_epoch=checkpoint_epoch,
            task_subset=tasks,
            fold=fold,
            routing_mode=routing_mode,
        )
        if gate_calibration_dir is not None:
            identity_kwargs["gate_calibration_dir"] = gate_calibration_dir
        evaluation_identity = resolve_identity(
            config,
            checkpoint_dir,
            **identity_kwargs,
        )
    output = Path(output_root) / f"fold{fold}"
    details = {
        "reporting_schema_version": REPORTING_SCHEMA_VERSION,
        "checkpoint_dir": repository_relative(checkpoint_dir),
        "split": "valid",
        "ensemble_folds": False,
        "fold": fold,
        "checkpoint_epoch": checkpoint_epoch,
        "model_selector": model_selector,
        "tasks": tasks,
        "reporting_study_id": study_id,
    }
    if routing_mode != "learned_gate":
        details["routing_mode"] = routing_mode
    if gate_calibration_dir is not None:
        details["gate_calibration_dir"] = repository_relative(gate_calibration_dir)
    run = open_run_directory(
        stage="stage3",
        operation="evaluate",
        config_path=config_path,
        config_payload=config.to_dict(),
        output=output,
        seed=config.data.seed,
        semantic_identity=evaluation_identity,
        details=details,
    )
    try:
        evaluation_kwargs = dict(
            split="valid",
            ensemble_folds=False,
            checkpoint_epoch=checkpoint_epoch,
            task_subset=tasks,
            fold=fold,
            predictions_dir=run.root / "predictions",
            reporting_study_id=study_id,
            expected_evaluation_identity=evaluation_identity,
            routing_mode=routing_mode,
        )
        if gate_calibration_dir is not None:
            evaluation_kwargs["gate_calibration_dir"] = gate_calibration_dir
        result = evaluate_checkpoints(config, checkpoint_dir, **evaluation_kwargs)
        run.complete(result)
    except BaseException:
        run.fail()
        raise


def _run_validation_schedule(
    *,
    config: Stage3Config,
    config_path: str,
    checkpoint_dir: Path,
    output_root: str,
    folds: tuple[int, ...],
    checkpoint_epoch: int | None,
    tasks: list[str] | None,
    routing_mode: str,
    study_id: str,
    progress: ProgressReporter,
    resolve_identity: ResolveIdentity,
    evaluate_checkpoints: EvaluateCheckpoints,
    gate_calibration_dir: Path | None = None,
) -> dict[int, str]:
    results: dict[int, str] = {}
    for fold in folds:
        print(f"fold{fold} started", flush=True)
        try:
            _run_fold(
                config=config,
                config_path=config_path,
                checkpoint_dir=checkpoint_dir,
                output_root=output_root,
                fold=fold,
                checkpoint_epoch=checkpoint_epoch,
                tasks=tasks,
                routing_mode=routing_mode,
                study_id=study_id,
                progress=progress,
                resolve_identity=resolve_identity,
                evaluate_checkpoints=evaluate_checkpoints,
                gate_calibration_dir=gate_calibration_dir,
            )
        except Exception as error:
            results[fold] = "failed"
            print(
                f"fold{fold} failed: {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
        else:
            results[fold] = "completed"
            print(f"fold{fold} completed", flush=True)
    return results


def _run_test(
    *,
    args: argparse.Namespace,
    config: Stage3Config,
    checkpoint_dir: Path,
    progress: ProgressReporter,
    resolve_identity: ResolveIdentity,
    evaluate_checkpoints: EvaluateCheckpoints,
) -> None:
    requested_calibration_dir = getattr(args, "gate_calibration_dir", None)
    gate_calibration_dir = (
        repository_path(requested_calibration_dir)
        if requested_calibration_dir is not None
        else None
    )
    model_selector = _model_selector(
        config, args.checkpoint_epoch, gate_calibration_dir is not None
    )
    with progress.status("Resolving Stage 3 evaluation identity"):
        identity_kwargs = dict(
            split="test",
            ensemble_folds=True,
            checkpoint_epoch=args.checkpoint_epoch,
            task_subset=args.tasks,
            fold=None,
            routing_mode=args.routing_mode,
        )
        if gate_calibration_dir is not None:
            identity_kwargs["gate_calibration_dir"] = gate_calibration_dir
        evaluation_identity = resolve_identity(
            config,
            checkpoint_dir,
            **identity_kwargs,
        )
    details = {
        "reporting_schema_version": REPORTING_SCHEMA_VERSION,
        "checkpoint_dir": repository_relative(checkpoint_dir),
        "split": "test",
        "ensemble_folds": True,
        "fold": None,
        "checkpoint_epoch": args.checkpoint_epoch,
        "model_selector": model_selector,
        "tasks": args.tasks,
        "reporting_study_id": args.study_id,
    }
    if args.routing_mode != "learned_gate":
        details["routing_mode"] = args.routing_mode
    if gate_calibration_dir is not None:
        details["gate_calibration_dir"] = repository_relative(gate_calibration_dir)
    run = open_run_directory(
        stage="stage3",
        operation="evaluate",
        config_path=args.config,
        config_payload=config.to_dict(),
        output=args.output,
        seed=config.data.seed,
        semantic_identity=evaluation_identity,
        details=details,
    )
    try:
        evaluation_kwargs = dict(
            split="test",
            ensemble_folds=True,
            checkpoint_epoch=args.checkpoint_epoch,
            task_subset=args.tasks,
            fold=None,
            predictions_dir=run.root / "predictions",
            reporting_study_id=args.study_id,
            expected_evaluation_identity=evaluation_identity,
            routing_mode=args.routing_mode,
        )
        if gate_calibration_dir is not None:
            evaluation_kwargs["gate_calibration_dir"] = gate_calibration_dir
        result = evaluate_checkpoints(config, checkpoint_dir, **evaluation_kwargs)
        run.complete(result)
    except BaseException:
        run.fail()
        raise


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        folds = _validate_request(args)
    except ValueError as error:
        parser.error(str(error))

    config = load_stage3_config(args.config)
    if args.gate_calibration_dir is not None and args.checkpoint_epoch is not None:
        parser.error("--gate-calibration-dir forbids --checkpoint-epoch")
    if args.gate_calibration_dir is not None and args.routing_mode != "learned_gate":
        parser.error("--gate-calibration-dir forbids forced --routing-mode")
    configure_process_runtime(config)
    from stage3.evaluate import (
        evaluate_checkpoints,
        resolve_stage3_evaluation_identity,
        resolve_stage3_reporting_study_id,
    )

    checkpoint_dir = repository_path(args.checkpoint_dir)
    repository_path(args.output)
    progress = ProgressReporter()
    if args.split == "test":
        _run_test(
            args=args,
            config=config,
            checkpoint_dir=checkpoint_dir,
            progress=progress,
            resolve_identity=resolve_stage3_evaluation_identity,
            evaluate_checkpoints=evaluate_checkpoints,
        )
        return 0

    assert folds is not None
    study_id = args.study_id
    if study_id is None:
        with progress.status("Resolving Stage 3 validation study identity"):
            study_id = resolve_stage3_reporting_study_id(
                config, checkpoint_epoch=args.checkpoint_epoch,
                routing_mode=args.routing_mode,
                gate_calibrated=args.gate_calibration_dir is not None,
            )
    try:
        results = _run_validation_schedule(
            config=config,
            config_path=args.config,
            checkpoint_dir=checkpoint_dir,
            output_root=args.output,
            folds=folds,
            checkpoint_epoch=args.checkpoint_epoch,
            tasks=args.tasks,
            routing_mode=args.routing_mode,
            study_id=study_id,
            progress=progress,
            resolve_identity=resolve_stage3_evaluation_identity,
            evaluate_checkpoints=evaluate_checkpoints,
            gate_calibration_dir=(
                repository_path(args.gate_calibration_dir)
                if args.gate_calibration_dir is not None
                else None
            ),
        )
    except KeyboardInterrupt:
        print("Stage3 validation evaluation interrupted", file=sys.stderr)
        return 130

    print("Stage3 validation evaluation complete")
    for status in ("completed", "failed"):
        selected = [str(fold) for fold in folds if results.get(fold) == status]
        print(f"{status}: {', '.join(selected) if selected else '-'}")
    return 1 if any(status == "failed" for status in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
