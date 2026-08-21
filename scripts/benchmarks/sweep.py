from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.config import load_benchmark_config
from benchmarks.common.data import configured_tasks
from benchmarks.common.metrics import macro_normalized_mae, mean_sample_std
from common.identity import semantic_identity
from common.io import atomic_json
from common.outputs import open_run_directory, repository_path, repository_relative


FIELDS = ("operation", "benchmark", "task", "fold", "attempt", "status", "exit_code", "output")


def _sanitize(task: str) -> str:
    return task.replace("/", "__")


def _metadata(path: Path) -> dict[str, Any] | None:
    metadata = path / "metadata.json"
    return json.loads(metadata.read_text(encoding="utf-8")) if metadata.is_file() else None


def _latest_completed(root: Path, required: str) -> Path | None:
    candidates = []
    for path in root.glob("attempt-*"):
        payload = _metadata(path)
        if payload and payload.get("status") == "completed" and (path / required).is_file():
            try:
                candidates.append((int(path.name.split("-", 1)[1]), path))
            except ValueError:
                pass
    return max(candidates)[1] if candidates else None


def _next_attempt(root: Path) -> tuple[int, Path]:
    numbers = []
    for path in root.glob("attempt-*"):
        try:
            numbers.append(int(path.name.split("-", 1)[1]))
        except ValueError:
            pass
    number = max(numbers, default=0) + 1
    return number, root / f"attempt-{number:03d}"


def _write_status(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tsv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _run(rows: list[dict[str, Any]], status_path: Path, *, operation: str, benchmark: str, task: str, fold: int | None, root: Path, required: str, command: list[str]) -> Path | None:
    completed = _latest_completed(root, required)
    if completed is not None:
        rows.append({"operation": operation, "benchmark": benchmark, "task": task, "fold": "" if fold is None else fold, "attempt": completed.name, "status": "SKIPPED_COMPLETED", "exit_code": 0, "output": repository_relative(completed)})
        _write_status(status_path, rows)
        return completed
    number, output = _next_attempt(root)
    result = subprocess.run([*command, "--output", repository_relative(output)], cwd=ROOT, check=False)
    status = "OK" if result.returncode == 0 else "FAILED"
    if result.returncode != 0 and not output.exists():
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(
            output / "sweep_failure.json",
            {"status": "failed_before_run_initialization", "exit_code": result.returncode},
        )
    rows.append({"operation": operation, "benchmark": benchmark, "task": task, "fold": "" if fold is None else fold, "attempt": f"attempt-{number:03d}", "status": status, "exit_code": result.returncode, "output": repository_relative(output)})
    _write_status(status_path, rows)
    return output if result.returncode == 0 else None


def _aggregate(root: Path, config: Any) -> dict[str, Any]:
    stage3_valid: dict[str, Any] = {}
    stage3_test: dict[str, Any] = {}
    stage2_test: dict[str, Any] = {}
    for task in configured_tasks(config, "stage3"):
        task_root = root / "stage3" / _sanitize(task)
        fold_values = []
        for fold in config.stage3.folds:
            run = _latest_completed(task_root / f"evaluate_valid_fold{fold}", "summary.json")
            if run:
                fold_values.append(json.loads((run / "summary.json").read_text(encoding="utf-8")))
        if len(fold_values) == len(config.stage3.folds):
            target = next(iter(fold_values[0]["targets"]))
            stage3_valid[task] = {
                metric: mean_sample_std([float(row["targets"][target][metric]) for row in fold_values])
                for metric in ("mae", "rmse", "r2", "normalized_mae")
            }
        run = _latest_completed(task_root / "evaluate_test", "summary.json")
        if run:
            payload = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            stage3_test[task] = next(iter(payload["ensemble"]["targets"].values()))
    for task in configured_tasks(config, "stage2_physics"):
        run = _latest_completed(root / "stage2_physics" / _sanitize(task) / "evaluate_test", "summary.json")
        if run:
            stage2_test[task] = json.loads((run / "summary.json").read_text(encoding="utf-8"))["targets"]
    return {
        "model": config.name,
        "stage3_property_benchmark": {
            "validation_five_fold": stage3_valid,
            "test_ensemble": stage3_test,
            "macro_normalized_mae": macro_normalized_mae(stage3_test),
        },
        "stage2_physics_benchmark": {"test": stage2_test, "aggregate": None},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all configured ILUME baseline jobs sequentially.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_benchmark_config(args.config)
    root = repository_path(args.output)
    identity = semantic_identity("benchmark.sweep.v1", {"config": config.to_dict()})
    run = open_run_directory(
        stage="benchmark", operation="sweep", config_path=args.config,
        config_payload=config.to_dict(), semantic_identity=identity,
        output=args.output, seed=config.seed, reusable=True,
        data_metadata=["data/task_catalog.csv", "data/stage2/metadata.json"],
    )
    rows: list[dict[str, Any]] = []
    status_path = root / "status.tsv"
    train_script = ROOT / "scripts/benchmarks/train.py"
    evaluate_script = ROOT / "scripts/benchmarks/evaluate.py"
    failures = 0
    for task in configured_tasks(config, "stage3"):
        task_root = root / "stage3" / _sanitize(task)
        checkpoints: dict[int, Path] = {}
        for fold in config.stage3.folds:
            checkpoint = _run(
                rows, status_path, operation="train", benchmark="stage3", task=task, fold=fold,
                root=task_root / f"fold{fold}", required="checkpoint.json",
                command=[sys.executable, str(train_script), "--config", args.config, "--benchmark", "stage3", "--task", task, "--fold", str(fold)],
            )
            if checkpoint is None:
                failures += 1
                continue
            checkpoints[fold] = checkpoint
            if _run(
                rows, status_path, operation="evaluate_valid", benchmark="stage3", task=task, fold=fold,
                root=task_root / f"evaluate_valid_fold{fold}", required="summary.json",
                command=[sys.executable, str(evaluate_script), "--config", args.config, "--benchmark", "stage3", "--task", task, "--split", "valid", "--fold", str(fold), "--checkpoint", repository_relative(checkpoint)],
            ) is None:
                failures += 1
        if len(checkpoints) == len(config.stage3.folds):
            if _run(
                rows, status_path, operation="evaluate_test", benchmark="stage3", task=task, fold=None,
                root=task_root / "evaluate_test", required="summary.json",
                command=[sys.executable, str(evaluate_script), "--config", args.config, "--benchmark", "stage3", "--task", task, "--split", "test", "--ensemble-folds", "--checkpoint-dir", repository_relative(task_root)],
            ) is None:
                failures += 1
    for task in configured_tasks(config, "stage2_physics"):
        task_root = root / "stage2_physics" / _sanitize(task)
        checkpoint = _run(
            rows, status_path, operation="train", benchmark="stage2_physics", task=task, fold=None,
            root=task_root / "train", required="checkpoint.json",
            command=[sys.executable, str(train_script), "--config", args.config, "--benchmark", "stage2_physics", "--task", task],
        )
        if checkpoint is None:
            failures += 1
            continue
        if _run(
            rows, status_path, operation="evaluate_test", benchmark="stage2_physics", task=task, fold=None,
            root=task_root / "evaluate_test", required="summary.json",
            command=[sys.executable, str(evaluate_script), "--config", args.config, "--benchmark", "stage2_physics", "--task", task, "--split", "test", "--checkpoint", repository_relative(checkpoint)],
        ) is None:
            failures += 1
    summary = _aggregate(root, config)
    summary["jobs"] = {"total": len(rows), "failed": failures}
    if failures:
        run.fail()
        raise SystemExit(1)
    run.complete(summary)


if __name__ == "__main__":
    main()
