from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.config import load_benchmark_config
from benchmarks.common.data import configured_tasks, has_test_rows, resolve_task
from benchmarks.common.environment import (
    ensure_benchmark_environment,
    environment_run_details,
    write_environment_snapshot,
)
from benchmarks.common.metrics import macro_normalized_mae, mean_sample_std
from common.identity import semantic_identity
from common.io import atomic_json
from common.outputs import open_run_directory, repository_path, repository_relative
from common.progress import ProgressReporter
from common.reporting import (
    REPORTING_SCHEMA_VERSION,
    STAGE2_BENCHMARK_SUITE_CONTRACT,
    STAGE2_CORE_EVALUATION_CONTRACT,
    STAGE2_PARTIAL_EVALUATION_CONTRACT,
    comparison_identity,
    stage2_full_comparison_identity,
)
from stage2.atom_evaluation import PARTIAL_CHARGE_TASK, PARTIAL_CHARGE_UNIT

FIELDS = ("operation", "benchmark", "task", "fold", "attempt", "status", "exit_code", "output")


def _sanitize(task: str) -> str:
    return task.replace("/", "__")


def _metadata(path: Path) -> dict[str, Any] | None:
    metadata = path / "metadata.json"
    return json.loads(metadata.read_text(encoding="utf-8")) if metadata.is_file() else None


def _latest_completed(
    root: Path, required: str, *, reporting_contract: str | None = None
) -> Path | None:
    candidates = []
    for path in root.glob("attempt-*"):
        payload = _metadata(path)
        required_path = path / required
        reporting_current = True
        if required == "summary.json" and required_path.is_file():
            try:
                summary = json.loads(required_path.read_text(encoding="utf-8"))
                reporting_current = (
                    summary.get("reporting", {}).get("schema_version")
                    == REPORTING_SCHEMA_VERSION
                )
                if reporting_contract is not None:
                    reporting_current = reporting_current and (
                        summary.get("reporting", {}).get("contract")
                        == reporting_contract
                    )
            except (json.JSONDecodeError, OSError):
                reporting_current = False
        if (
            payload
            and payload.get("status") == "completed"
            and required_path.is_file()
            and reporting_current
        ):
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


@dataclass
class _SweepState:
    rows: list[dict[str, Any]]
    status_path: Path
    progress: Any | None = None
    model_name: str | None = None
    progress_counts: dict[str, int] = field(default_factory=lambda: {"done": 0, "failed": 0})
    lock: threading.Lock = field(default_factory=threading.Lock)
    reserved_roots: set[Path] = field(default_factory=set)

    def reserve(
        self, *, root: Path, required: str, operation: str, benchmark: str,
        task: str, fold: int | None, training_job: bool,
    ) -> tuple[int, Path] | Path:
        with self.lock:
            completed = _latest_completed(
                root,
                required,
                reporting_contract=(
                    (
                        STAGE2_PARTIAL_EVALUATION_CONTRACT
                        if task == PARTIAL_CHARGE_TASK
                        else STAGE2_CORE_EVALUATION_CONTRACT
                    )
                    if operation.startswith("evaluate") and benchmark == "stage2_physics"
                    else None
                ),
            )
            if completed is not None:
                self._record_locked(
                    operation=operation, benchmark=benchmark, task=task, fold=fold,
                    attempt=completed.name, status="SKIPPED_COMPLETED", exit_code=0,
                    output=completed, training_job=training_job,
                )
                return completed
            if root in self.reserved_roots:
                raise RuntimeError(f"Duplicate benchmark job submission: {root}")
            number, output = _next_attempt(root)
            self.reserved_roots.add(root)
            return number, output

    def finish(
        self, *, root: Path, operation: str, benchmark: str, task: str,
        fold: int | None, attempt: str, status: str, exit_code: int,
        output: Path, training_job: bool,
    ) -> None:
        with self.lock:
            self.reserved_roots.remove(root)
            self._record_locked(
                operation=operation, benchmark=benchmark, task=task, fold=fold,
                attempt=attempt, status=status, exit_code=exit_code, output=output,
                training_job=training_job,
            )

    def scheduler_failure(self, job: "_Job", error: Exception) -> None:
        with self.lock:
            self._record_locked(
                operation="scheduler", benchmark=job.benchmark, task=job.task,
                fold=job.fold, attempt="", status="FAILED", exit_code=-1,
                output=None, training_job=False,
            )
            print(
                f"Sweep worker failed for {job.key}: {type(error).__name__}: {error}",
                file=sys.stderr,
            )

    def _record_locked(
        self, *, operation: str, benchmark: str, task: str, fold: int | None,
        attempt: str, status: str, exit_code: int, output: Path | None,
        training_job: bool,
    ) -> None:
        self.rows.append(
            {
                "operation": operation, "benchmark": benchmark, "task": task,
                "fold": "" if fold is None else fold, "attempt": attempt,
                "status": status, "exit_code": exit_code,
                "output": "" if output is None else repository_relative(output),
            }
        )
        _write_status(self.status_path, self.rows)
        if training_job and self.progress is not None:
            if status in {"OK", "SKIPPED_COMPLETED"}:
                self.progress_counts["done"] += 1
            else:
                self.progress_counts["failed"] += 1
            self.progress.set_postfix(self.progress_counts)
            self.progress.update(1)

    def set_progress_description(
        self,
        *,
        benchmark: str,
        task: str,
        fold: int | None,
    ) -> None:
        return


def _run(
    state: _SweepState, *, operation: str, benchmark: str, task: str,
    fold: int | None, root: Path, required: str, command: list[str],
    training_job: bool = False, env: dict[str, str] | None = None,
) -> Path | None:
    if training_job:
        state.set_progress_description(benchmark=benchmark, task=task, fold=fold)
    reservation = state.reserve(
        root=root, required=required, operation=operation, benchmark=benchmark,
        task=task, fold=fold, training_job=training_job,
    )
    if isinstance(reservation, Path):
        return reservation
    number, output = reservation
    try:
        result = subprocess.run(
            [*command, "--output", repository_relative(output)],
            cwd=ROOT, check=False, env=env,
        )
        exit_code = result.returncode
    except OSError as error:
        exit_code = -1
        if not output.exists():
            output.mkdir(parents=True, exist_ok=False)
            atomic_json(
                output / "sweep_failure.json",
                {"status": "failed_to_launch_subprocess", "error_type": type(error).__name__, "error": str(error)},
            )
    status = "OK" if exit_code == 0 else "FAILED"
    if exit_code != 0 and not output.exists():
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(
            output / "sweep_failure.json",
            {"status": "failed_before_run_initialization", "exit_code": exit_code},
        )
    state.finish(
        root=root, operation=operation, benchmark=benchmark, task=task, fold=fold,
        attempt=f"attempt-{number:03d}", status=status, exit_code=exit_code,
        output=output, training_job=training_job,
    )
    return output if exit_code == 0 else None


JobKind = Literal["stage3_fold", "stage3_ensemble", "stage2_task"]


@dataclass(frozen=True)
class _Job:
    priority: tuple[int, int, int]
    kind: JobKind
    benchmark: str
    task: str
    fold: int | None
    device: str | None

    @property
    def key(self) -> tuple[str, str, int | None]:
        return self.kind, self.task, self.fold


@dataclass(frozen=True)
class _JobResult:
    job: _Job
    train_succeeded: bool


def _job_roots(root: Path, job: _Job) -> tuple[Path, ...]:
    if job.kind == "stage3_fold":
        assert job.fold is not None
        task_root = root / "stage3" / _sanitize(job.task)
        return task_root / f"fold{job.fold}", task_root / f"evaluate_valid_fold{job.fold}"
    if job.kind == "stage3_ensemble":
        return (root / "stage3" / _sanitize(job.task) / "evaluate_test",)
    task_root = root / "stage2_physics" / _sanitize(job.task)
    return task_root / "train", task_root / "evaluate_test"


def _build_jobs(
    *, root: Path, stage3_tasks: tuple[str, ...], folds: tuple[int, ...],
    stage2_tasks: tuple[str, ...], devices: tuple[str, ...],
) -> tuple[list[_Job], dict[str, _Job]]:
    jobs: list[_Job] = []
    ensembles: dict[str, _Job] = {}
    device_index = 0

    def next_device() -> str | None:
        nonlocal device_index
        if not devices:
            return None
        device = devices[device_index % len(devices)]
        device_index += 1
        return device

    for task_index, task in enumerate(stage3_tasks):
        for fold_index, fold in enumerate(folds):
            jobs.append(
                _Job((0, task_index, fold_index), "stage3_fold", "stage3", task, fold, next_device())
            )
        ensembles[task] = _Job(
            (0, task_index, len(folds)), "stage3_ensemble", "stage3", task, None, next_device()
        )
    for task_index, task in enumerate(stage2_tasks):
        jobs.append(
            _Job((1, task_index, 0), "stage2_task", "stage2_physics", task, None, next_device())
        )

    all_jobs = [*jobs, *ensembles.values()]
    keys = [job.key for job in all_jobs]
    roots = [job_root for job in all_jobs for job_root in _job_roots(root, job)]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate logical benchmark jobs in sweep configuration")
    if len(roots) != len(set(roots)):
        raise ValueError("Duplicate benchmark output roots in sweep configuration")
    return jobs, ensembles


def _subprocess_env(device: str | None) -> dict[str, str]:
    environment = os.environ.copy()

    # Child train/evaluate processes should stay quiet.
    # The sweep process owns the global progress bar.
    environment["ILUME_DISABLE_PROGRESS"] = "1"

    if device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]

    return environment


def _execute_job(
    job: _Job, *, state: _SweepState, config_path: str, root: Path,
    train_script: Path, evaluate_script: Path,
) -> _JobResult:
    env = _subprocess_env(job.device)
    if job.kind == "stage3_fold":
        assert job.fold is not None
        task_root = root / "stage3" / _sanitize(job.task)
        checkpoint = _run(
            state, operation="train", benchmark="stage3", task=job.task, fold=job.fold,
            root=task_root / f"fold{job.fold}", required="checkpoint.json",
            command=[sys.executable, str(train_script), "--config", config_path, "--benchmark", "stage3", "--task", job.task, "--fold", str(job.fold)],
            training_job=True, env=env,
        )
        if checkpoint is None:
            return _JobResult(job, False)
        _run(
            state, operation="evaluate_valid", benchmark="stage3", task=job.task, fold=job.fold,
            root=task_root / f"evaluate_valid_fold{job.fold}", required="summary.json",
            command=[sys.executable, str(evaluate_script), "--config", config_path, "--benchmark", "stage3", "--task", job.task, "--split", "valid", "--fold", str(job.fold), "--checkpoint", repository_relative(checkpoint)],
            env=env,
        )
        return _JobResult(job, True)

    if job.kind == "stage3_ensemble":
        task_root = root / "stage3" / _sanitize(job.task)
        _run(
            state, operation="evaluate_test", benchmark="stage3", task=job.task, fold=None,
            root=task_root / "evaluate_test", required="summary.json",
            command=[sys.executable, str(evaluate_script), "--config", config_path, "--benchmark", "stage3", "--task", job.task, "--split", "test", "--ensemble-folds", "--checkpoint-dir", repository_relative(task_root)],
            env=env,
        )
        return _JobResult(job, False)

    task_root = root / "stage2_physics" / _sanitize(job.task)
    checkpoint = _run(
        state, operation="train", benchmark="stage2_physics", task=job.task, fold=None,
        root=task_root / "train", required="checkpoint.json",
        command=[sys.executable, str(train_script), "--config", config_path, "--benchmark", "stage2_physics", "--task", job.task],
        training_job=True, env=env,
    )
    if checkpoint is None:
        return _JobResult(job, False)
    _run(
        state, operation="evaluate_test", benchmark="stage2_physics", task=job.task, fold=None,
        root=task_root / "evaluate_test", required="summary.json",
        command=[sys.executable, str(evaluate_script), "--config", config_path, "--benchmark", "stage2_physics", "--task", job.task, "--split", "test", "--checkpoint", repository_relative(checkpoint)],
        env=env,
    )
    return _JobResult(job, True)


def _schedule(
    *, jobs: list[_Job], ensembles: dict[str, _Job], folds: tuple[int, ...],
    max_workers: int, state: _SweepState, config_path: str, root: Path,
    train_script: Path, evaluate_script: Path,
) -> None:
    ready: list[tuple[tuple[int, int, int], int, _Job]] = []
    sequence = 0
    for job in jobs:
        heapq.heappush(ready, (job.priority, sequence, job))
        sequence += 1
    fold_results: dict[str, dict[int, bool]] = {task: {} for task in ensembles}
    submitted_keys: set[tuple[str, str, int | None]] = set()
    running: dict[Future[_JobResult], _Job] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while ready or running:
            while ready and len(running) < max_workers:
                _, _, job = heapq.heappop(ready)
                if job.key in submitted_keys:
                    raise RuntimeError(f"Duplicate benchmark job submission: {job.key}")
                submitted_keys.add(job.key)
                future = executor.submit(
                    _execute_job, job, state=state, config_path=config_path,
                    root=root, train_script=train_script, evaluate_script=evaluate_script,
                )
                running[future] = job
            completed, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=lambda item: running[item].priority):
                job = running.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    state.scheduler_failure(job, error)
                    result = _JobResult(job, False)
                if job.kind != "stage3_fold":
                    continue
                assert job.fold is not None
                task_results = fold_results[job.task]
                task_results[job.fold] = result.train_succeeded
                if len(task_results) == len(folds) and all(task_results.values()):
                    ensemble = ensembles[job.task]
                    heapq.heappush(ready, (ensemble.priority, sequence, ensemble))
                    sequence += 1


def _aggregate(root: Path, config: Any) -> dict[str, Any]:
    stage3_valid: dict[str, Any] = {}
    stage3_test: dict[str, Any] = {}
    stage2_test: dict[str, Any] = {}
    stage2_partial: dict[str, Any] | None = None
    stage3_valid_reporting: list[dict[str, Any]] = []
    stage3_test_reporting: list[dict[str, Any]] = []
    stage2_reporting: list[dict[str, Any]] = []
    stage2_partial_reporting: list[dict[str, Any]] = []
    source_runs: dict[str, Any] = {
        "stage3_validation": {}, "stage3_test": {}, "stage2_physics": {}
    }
    for task in configured_tasks(config, "stage3"):
        task_root = root / "stage3" / _sanitize(task)
        fold_values = []
        for fold in config.stage3.folds:
            run = _latest_completed(task_root / f"evaluate_valid_fold{fold}", "summary.json")
            if run:
                payload = json.loads((run / "summary.json").read_text(encoding="utf-8"))
                fold_values.append(payload)
                stage3_valid_reporting.append(payload["reporting"])
                source_runs["stage3_validation"].setdefault(task, {})[
                    f"fold{fold}"
                ] = repository_relative(run)
        if len(fold_values) == len(config.stage3.folds):
            target = next(iter(fold_values[0]["targets"]))
            stage3_valid[task] = {
                metric: mean_sample_std([float(row["targets"][target][metric]) for row in fold_values])
                for metric in (
                    "mae", "rmse", "r2", "normalized_mae", "normalized_rmse"
                )
            }
        run = _latest_completed(task_root / "evaluate_test", "summary.json")
        if run:
            payload = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            stage3_test[task] = next(iter(payload["ensemble"]["targets"].values()))
            stage3_test_reporting.append(payload["reporting"])
            source_runs["stage3_test"][task] = repository_relative(run)
    for task in configured_tasks(config, "stage2_physics"):
        contract = (
            STAGE2_PARTIAL_EVALUATION_CONTRACT
            if task == PARTIAL_CHARGE_TASK
            else STAGE2_CORE_EVALUATION_CONTRACT
        )
        run = _latest_completed(
            root / "stage2_physics" / _sanitize(task) / "evaluate_test",
            "summary.json",
            reporting_contract=contract,
        )
        if run:
            payload = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            if task == PARTIAL_CHARGE_TASK:
                stage2_partial = payload["stage2_partial_charge_benchmark"]["test"]
                stage2_partial_reporting.append(payload["reporting"])
            else:
                stage2_test[task] = payload["targets"]
                stage2_reporting.append(payload["reporting"])
            source_runs["stage2_physics"][task] = repository_relative(run)

    def merged_comparison(
        fragments: list[dict[str, Any]],
        *,
        benchmark: str,
        split: str,
        expected: list[str],
        ensemble: bool,
    ) -> dict[str, Any]:
        sources: dict[str, Any] = {}
        normalization: dict[str, Any] = {}
        for fragment in fragments:
            payload = fragment["comparison_identity"]["payload"]
            for destination, values in (
                (sources, payload["sources"]),
                (normalization, payload["normalization"]),
            ):
                for key, value in values.items():
                    if key in destination and destination[key] != value:
                        raise ValueError(f"Conflicting reporting comparison item: {key}")
                    destination[key] = value
        return comparison_identity(
            benchmark,
            split=split,
            expected=expected,
            sources=sources,
            normalization=normalization,
            folds=config.stage3.folds if benchmark == "stage3_property" else (),
            ensemble=ensemble,
        )

    valid_expected = list(configured_tasks(config, "stage3"))
    test_expected = [
        task
        for task in valid_expected
        if has_test_rows(
            resolve_task(config, "stage3", task, config.stage3.folds[0])
        )
    ]
    configured_stage2 = list(configured_tasks(config, "stage2_physics"))
    stage2_expected = [
        task for task in configured_stage2 if task != PARTIAL_CHARGE_TASK
    ]
    stage2_values = []
    for task in stage2_expected:
        spec = resolve_task(config, "stage2_physics", task, None)
        if len(spec.target_columns) != 1:
            raise ValueError(f"Stage 2 Core task must be scalar: {task}")
        target = spec.target_columns[0]
        metrics = stage2_test.get(task, {}).get(target, {})
        if "normalized_mae" in metrics:
            stage2_values.append(float(metrics["normalized_mae"]))
    stage2_valid_count = sum(math.isfinite(value) for value in stage2_values)
    stage2_complete = stage2_valid_count == len(stage2_expected)
    study_ids = {
        item["study_id"]
        for item in [
            *stage3_valid_reporting,
            *stage3_test_reporting,
            *stage2_reporting,
            *stage2_partial_reporting,
        ]
    }
    study_id = f"{config.name}-" + semantic_identity(
        "benchmark.reporting-study.v1",
        {"model": config.name, "config": _scientific_config(config)},
    )["hash"]
    if study_ids - {study_id}:
        raise ValueError("Benchmark sweep contains incompatible reporting study identities")
    partial_supported = PARTIAL_CHARGE_TASK in configured_stage2
    partial_complete = bool(
        partial_supported
        and stage2_partial is not None
        and stage2_partial.get("status") == "complete"
        and stage2_partial_reporting
    )
    core_comparison = merged_comparison(
        stage2_reporting,
        benchmark="stage2_physics",
        split="test",
        expected=stage2_expected,
        ensemble=False,
    )
    partial_comparison = (
        stage2_partial_reporting[0]["comparison_identity"]
        if stage2_partial_reporting
        else None
    )
    full_complete = stage2_complete and partial_complete
    full_comparison = (
        stage2_full_comparison_identity(
            core_comparison,
            partial_comparison,
            ordered_units=(*stage2_expected, PARTIAL_CHARGE_UNIT),
        )
        if full_complete and partial_comparison is not None
        else None
    )
    result = {
        "model": config.name,
        "stage3_property_benchmark": {
            "validation_five_fold": stage3_valid,
            "test_ensemble": stage3_test,
            "macro_normalized_mae": macro_normalized_mae(stage3_test),
        },
        "stage2_physics_benchmark": {
            "test": stage2_test,
            "aggregate": {
                "macro_normalized_mae": (
                    sum(stage2_values) / len(stage2_values)
                    if stage2_values else float("nan")
                ),
                "valid_tasks": len(stage2_values),
                "total_tasks": len(stage2_expected),
            },
        },
        "stage2_partial_charge_benchmark": {
            "test": stage2_partial
        },
        "reporting": {
            "schema_version": REPORTING_SCHEMA_VERSION,
            "contract": STAGE2_BENCHMARK_SUITE_CONTRACT,
            "model_id": config.name,
            "model_display_name": config.display_name,
            "study_id": study_id,
            "benchmarks": {
                "stage3_test": {
                    "benchmark": "stage3_property",
                    "protocol": {
                        "split": "test", "folds": list(config.stage3.folds),
                        "ensemble": True, "expected_tasks": test_expected,
                        "enabled_tasks": valid_expected,
                    },
                    "comparison_identity": merged_comparison(
                        [
                            item for item in stage3_test_reporting
                            if item["protocol"]["expected_tasks"][0] in test_expected
                        ],
                        benchmark="stage3_property", split="test",
                        expected=test_expected, ensemble=True,
                    ),
                },
                "stage3_validation": {
                    "benchmark": "stage3_property",
                    "protocol": {
                        "split": "valid", "folds": list(config.stage3.folds),
                        "ensemble": False, "expected_tasks": valid_expected,
                    },
                    "comparison_identity": merged_comparison(
                        stage3_valid_reporting,
                        benchmark="stage3_property", split="valid",
                        expected=valid_expected, ensemble=False,
                    ),
                },
                "stage2_core_physics": {
                    "status": "complete" if stage2_complete else "incomplete",
                    "benchmark": "stage2_physics",
                    "protocol": {
                        "split": "test", "folds": [], "ensemble": False,
                        "expected_tasks": stage2_expected,
                    },
                    "comparison_identity": merged_comparison(
                        stage2_reporting,
                        benchmark="stage2_physics", split="test",
                        expected=stage2_expected, ensemble=False,
                    ),
                    "issues": (
                        [] if stage2_complete
                        else [f"missing_or_invalid_tasks={len(stage2_expected) - stage2_valid_count}"]
                    ),
                },
                "stage2_partial_charge": {
                    "status": (
                        "complete"
                        if partial_complete
                        else "incomplete"
                        if partial_supported
                        else "unsupported"
                    ),
                    "benchmark": "stage2_partial_charge",
                    "protocol": {
                        "split": "test",
                        "folds": [],
                        "ensemble": False,
                        "expected_units": [PARTIAL_CHARGE_UNIT],
                    },
                    **(
                        {}
                        if partial_comparison is None
                        else {"comparison_identity": partial_comparison}
                    ),
                    "issues": (
                        []
                        if partial_complete or not partial_supported
                        else ["missing_or_incomplete_partial_charge"]
                    ),
                },
                "stage2_physics_full": {
                    "status": (
                        "complete"
                        if full_complete
                        else "incomplete"
                        if partial_supported
                        else "unsupported"
                    ),
                    "benchmark": "stage2_physics_full",
                    "protocol": {
                        "split": "test",
                        "folds": [],
                        "ensemble": False,
                        "expected_units": [*stage2_expected, PARTIAL_CHARGE_UNIT],
                    },
                    **(
                        {}
                        if full_comparison is None
                        else {"comparison_identity": full_comparison}
                    ),
                    "issues": (
                        []
                        if full_complete or not partial_supported
                        else ["core_or_partial_incomplete"]
                    ),
                },
            },
            "capabilities": {
                "stage2_core_physics": "supported",
                "stage2_partial_charge": (
                    "supported" if partial_supported else "unsupported"
                ),
                "stage2_physics_full": (
                    "supported" if partial_supported else "unsupported"
                ),
            },
            "source_runs": source_runs,
            "source_run_manifest": semantic_identity(
                "benchmark.source-run-manifest.v1", {"source_runs": source_runs}
            ),
        },
    }
    return result


def _scientific_config(config: Any) -> dict[str, Any]:
    payload = config.to_dict()
    payload.pop("display_name", None)
    payload.pop("runtime", None)
    return payload


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parse_devices(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    devices = tuple(item.strip() for item in value.split(","))
    if not devices or any(not re.fullmatch(r"cuda:\d+", item) for item in devices):
        raise ValueError("--devices must be a comma-separated list such as cuda:0,cuda:1")
    if len(devices) != len(set(devices)):
        raise ValueError("--devices must not contain duplicate devices")
    return devices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run configured ILUME baseline jobs with bounded subprocess concurrency."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max-workers", type=_positive_int, default=1,
        help=("maximum concurrent train/evaluate subprocesses (default: 1); for "
              "XGBoost, size max-workers * training.n_jobs for the available CPUs"),
    )
    parser.add_argument(
        "--devices",
        help=("optional MLP/D-MPNN/MoLFormer GPU list such as cuda:0,cuda:1; logical job chains are "
              "assigned round-robin via CUDA_VISIBLE_DEVICES"),
    )
    args = parser.parse_args()
    config = load_benchmark_config(args.config)
    environment_snapshot = ensure_benchmark_environment(config)
    try:
        devices = _parse_devices(args.devices)
    except ValueError as error:
        parser.error(str(error))
    if devices and config.name not in {"mlp", "dmpnn", "molformer", "ilbert"}:
        parser.error("--devices is only supported for GPU neural-network benchmarks")
    if devices and config.training.get("device") != "cuda":
        parser.error("--devices requires training.device: cuda")

    root = repository_path(args.output)
    identity = semantic_identity("benchmark.sweep.v1", {"config": _scientific_config(config)})
    run = open_run_directory(
        stage="benchmark", operation="sweep", config_path=args.config,
        config_payload=config.to_dict(), semantic_identity=identity,
        output=args.output, seed=config.seed, reusable=True,
        data_metadata=["data/task_catalog.csv", "data/stage2/metadata.json"],
        details={
            "reporting_schema_version": REPORTING_SCHEMA_VERSION,
            "reporting_contract": STAGE2_BENCHMARK_SUITE_CONTRACT,
            **environment_run_details(environment_snapshot),
        },
    )
    if environment_snapshot is not None:
        write_environment_snapshot(run.root / "environment.json", environment_snapshot)
    rows: list[dict[str, Any]] = []
    train_script = ROOT / "scripts/benchmarks/train.py"
    evaluate_script = ROOT / "scripts/benchmarks/evaluate.py"
    stage3_tasks = configured_tasks(config, "stage3")
    stage2_tasks = configured_tasks(config, "stage2_physics")
    training_job_total = len(stage3_tasks) * len(config.stage3.folds) + len(stage2_tasks)
    sweep_progress = ProgressReporter().bar(
        total=training_job_total, desc=f"{config.name} sweep", unit="train-job"
    )
    state = _SweepState(
        rows=rows, status_path=root / "status.tsv", progress=sweep_progress,
        model_name=config.name,
    )
    jobs, ensembles = _build_jobs(
        root=root, stage3_tasks=stage3_tasks, folds=config.stage3.folds,
        stage2_tasks=stage2_tasks, devices=devices,
    )
    try:
        _schedule(
            jobs=jobs, ensembles=ensembles, folds=config.stage3.folds,
            max_workers=args.max_workers, state=state, config_path=args.config,
            root=root, train_script=train_script, evaluate_script=evaluate_script,
        )
    except BaseException:
        run.fail()
        raise
    finally:
        sweep_progress.close()

    failures = sum(row["status"] == "FAILED" for row in rows)
    summary = _aggregate(root, config)
    summary["jobs"] = {"total": len(rows), "failed": failures}
    if failures:
        run.fail()
        raise SystemExit(1)
    run.complete(summary)


if __name__ == "__main__":
    main()
