from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import queue
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import hashlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


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


def _read_json(path: Path, *, context: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{context} is missing: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is unreadable: {path.name}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must contain a JSON object: {path.name}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _history_epochs(path: Path, *, context: str) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"{context} is missing: {path.name}")
    epochs: list[int] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("epoch"), int):
                raise ValueError
            epochs.append(row["epoch"])
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{context} is unreadable: {path.name}") from error
    return epochs


def _last_history_row(path: Path) -> dict[str, Any]:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("Stage 3 metrics history is empty after training")
    payload = json.loads(rows[-1])
    if not isinstance(payload, dict):
        raise ValueError("Stage 3 final metrics row must be a JSON object")
    return payload


def _require_continuous_history(epochs: list[int], *, context: str) -> None:
    if epochs != list(range(1, len(epochs) + 1)):
        raise ValueError(f"{context} epoch history is not continuous from epoch 1")


def _checkpoint_epochs(root: Path) -> list[int]:
    epochs: list[int] = []
    for path in root.glob("checkpoint_epoch_*.pt"):
        match = re.fullmatch(r"checkpoint_epoch_(\d{5})\.pt", path.name)
        if match is not None:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def _validate_existing_identity(
    metadata: Mapping[str, Any], training_identity: Mapping[str, Any]
) -> None:
    from common.identity import require_compatible_identity

    existing = metadata.get("semantic_identity")
    if not isinstance(existing, Mapping):
        raise ValueError("Existing Stage 3 run predates identity contract v1; retrain it")
    require_compatible_identity(
        training_identity,
        existing,
        context="Existing Stage 3 train run",
    )


def _resume_action(
    root: Path,
    *,
    fold: int,
    total_epochs: int,
    training_identity: Mapping[str, Any],
) -> tuple[str, Path | None]:
    metadata = _read_json(root / "metadata.json", context="Stage 3 run metadata")
    _validate_existing_identity(metadata, training_identity)
    if metadata.get("stage") != "stage3" or metadata.get("operation") != "train":
        raise ValueError("Existing output is not a Stage 3 train run")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("fold") != fold:
        raise ValueError("Existing Stage 3 run fold does not match its directory")
    if not (root / "run_config.yaml").is_file():
        raise FileNotFoundError("Stage 3 run is missing run_config.yaml")

    metrics = _history_epochs(root / "metrics.jsonl", context="Stage 3 metrics history")
    diagnostics = _history_epochs(
        root / "diagnostics.jsonl", context="Stage 3 diagnostics history"
    )
    _require_continuous_history(metrics, context="Stage 3 metrics")
    _require_continuous_history(diagnostics, context="Stage 3 diagnostics")
    if metrics != diagnostics:
        raise ValueError("Stage 3 metrics and diagnostics histories end at different epochs")

    status = metadata.get("status")
    checkpoints = _checkpoint_epochs(root)
    if status == "completed":
        summary = _read_json(root / "summary.json", context="Stage 3 run summary")
        final = summary.get("final_epoch")
        if summary.get("fold") != fold or not isinstance(final, Mapping):
            raise ValueError("Completed Stage 3 summary does not describe the requested fold")
        if final.get("epoch") != total_epochs or metrics != list(
            range(1, total_epochs + 1)
        ):
            raise ValueError("Completed Stage 3 run does not contain the full epoch history")
        final_checkpoint = root / f"checkpoint_epoch_{total_epochs:05d}.pt"
        if not final_checkpoint.is_file():
            raise FileNotFoundError("Completed Stage 3 run is missing its final checkpoint")
        refinement = _read_json(
            root / "taskwise_refinement.json",
            context="Stage 3 task-wise refinement manifest",
        )
        artifact = root / "taskwise_refined.pt"
        if not artifact.is_file() or refinement.get("artifact_sha256") != _sha256(artifact):
            raise ValueError("Completed Stage 3 refined artifact is missing or corrupt")
        if summary.get("taskwise_refinement") != refinement:
            raise ValueError("Completed Stage 3 summary/refinement manifest mismatch")
        return "skipped", None

    if status not in {"failed", "running"}:
        raise ValueError(f"Stage 3 run has unsupported status: {status!r}")
    if not metrics or not checkpoints:
        raise ValueError("Stage 3 run has no legal complete checkpoint to resume")
    latest_history = metrics[-1]
    latest_checkpoint = checkpoints[-1]
    if latest_checkpoint != latest_history:
        raise ValueError(
            "Stage 3 run has no legal resume point: checkpoint and history tails differ"
        )
    return "resume", root / f"checkpoint_epoch_{latest_checkpoint:05d}.pt"


def _run_fold(
    *,
    config_path: str,
    fold: int,
    output_root: str,
    resume: bool,
    device: str | None,
    show_progress: bool,
) -> str:
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
    if not show_progress:
        os.environ["ILUME_DISABLE_PROGRESS"] = "1"

    from stage3.config import (
        configure_process_runtime,
        effective_training_seed,
        load_stage3_config,
    )

    config = load_stage3_config(config_path)
    configure_process_runtime(config)

    from common.outputs import (
        open_run_directory,
        repository_path,
        repository_relative,
    )
    from stage3.train import resolve_stage3_training_identity, run_stage3_training

    training_identity = resolve_stage3_training_identity(config, fold)
    fold_output = Path(output_root) / f"fold{fold}"
    fold_root = repository_path(fold_output)
    resume_from: Path | None = None
    if fold_root.exists():
        if not resume:
            raise FileExistsError(
                f"Output already exists: {repository_relative(fold_root)}"
            )
        action, resume_from = _resume_action(
            fold_root,
            fold=fold,
            total_epochs=config.training.epochs,
            training_identity=training_identity,
        )
        if action == "skipped":
            return "skipped"

    run = open_run_directory(
        stage="stage3",
        operation="train",
        config_path=config_path,
        config_payload=config.to_dict(),
        output=fold_output,
        seed=effective_training_seed(config),
        semantic_identity=training_identity,
        resume=resume_from,
        details={
            "fold": fold,
            "stage2_encoder": repository_relative(
                config.initialization.stage2_encoder
            ),
            "assigned_device": device or config.training.device,
        },
    )
    try:
        rows = run_stage3_training(
            config,
            fold,
            output_dir=run.root,
            resume_from=resume_from,
            expected_training_identity=training_identity,
        )
        final_epoch = rows[-1] if rows else _last_history_row(run.root / "metrics.jsonl")
        refinement = _read_json(
            run.root / "taskwise_refinement.json",
            context="Stage 3 task-wise refinement manifest",
        )
        run.complete({
            "fold": fold,
            "final_epoch": final_epoch,
            "taskwise_refinement": refinement,
        })
    except BaseException:
        run.fail()
        raise
    return "completed"


def _worker_entry(
    config_path: str,
    fold: int,
    output_root: str,
    resume: bool,
    device: str | None,
    show_progress: bool,
    result_queue: Any,
) -> None:
    try:
        status = _run_fold(
            config_path=config_path,
            fold=fold,
            output_root=output_root,
            resume=resume,
            device=device,
            show_progress=show_progress,
        )
    except BaseException as error:
        result_queue.put(("failed", type(error).__name__, str(error)))
        traceback.print_exc()
        raise SystemExit(1) from error
    result_queue.put((status, None, None))


def _terminate_workers(running: Mapping[int, tuple[Any, int, Any]]) -> None:
    processes = [item[0] for item in running.values()]
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)
    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        process.join()
    for _, _, result_queue in running.values():
        result_queue.close()


def _run_schedule(
    *,
    config_path: str,
    folds: tuple[int, ...],
    output_root: str,
    resume: bool,
    max_parallel: int,
    devices: tuple[str, ...],
) -> dict[int, str]:
    from multiprocessing.connection import wait

    ctx = multiprocessing.get_context("spawn")
    effective_parallel = min(max_parallel, len(folds))
    slot_devices = tuple(
        devices[index % len(devices)] if devices else None
        for index in range(effective_parallel)
    )
    pending = iter(folds)
    running: dict[int, tuple[Any, int, Any]] = {}
    results: dict[int, str] = {}

    def launch(slot: int, fold: int) -> None:
        result_queue = ctx.Queue()
        process = ctx.Process(
            target=_worker_entry,
            args=(
                config_path,
                fold,
                output_root,
                resume,
                slot_devices[slot],
                fold == folds[0],
                result_queue,
            ),
            name=f"stage3-fold{fold}",
        )
        process.start()
        running[slot] = (process, fold, result_queue)
        print(f"fold{fold} started", flush=True)

    try:
        for slot in range(effective_parallel):
            launch(slot, next(pending))
        while running:
            ready = wait([item[0].sentinel for item in running.values()])
            completed_slots = sorted(
                slot for slot, item in running.items() if item[0].sentinel in ready
            )
            for slot in completed_slots:
                process, fold, result_queue = running.pop(slot)
                process.join()
                try:
                    payload = result_queue.get_nowait()
                except queue.Empty:
                    payload = ("failed", "WorkerExit", f"exit code {process.exitcode}")
                finally:
                    result_queue.close()
                    result_queue.join_thread()
                status = payload[0]
                if process.exitcode != 0 or status not in {"completed", "skipped"}:
                    status = "failed"
                results[fold] = status
                print(f"fold{fold} {status}", flush=True)
                try:
                    next_fold = next(pending)
                except StopIteration:
                    continue
                launch(slot, next_fold)
    except BaseException:
        _terminate_workers(running)
        raise
    return results


@dataclass(frozen=True)
class _CapacityFoldJob:
    trial_number: int
    config_path: str
    fold: int
    output_root: str
    resume: bool
    device: str
    show_progress: bool = False


def _run_assigned_capacity_jobs(
    jobs: Sequence[_CapacityFoldJob],
) -> dict[tuple[int, int], str]:
    from multiprocessing.connection import wait

    ctx = multiprocessing.get_context("spawn")
    running: dict[tuple[int, int], tuple[Any, Any]] = {}
    results: dict[tuple[int, int], str] = {}
    try:
        for job in jobs:
            key = (job.trial_number, job.fold)
            result_queue = ctx.Queue()
            process = ctx.Process(
                target=_worker_entry,
                args=(
                    job.config_path,
                    job.fold,
                    job.output_root,
                    job.resume,
                    job.device,
                    job.show_progress,
                    result_queue,
                ),
                name=f"capacity-trial{job.trial_number}-fold{job.fold}",
            )
            process.start()
            running[key] = (process, result_queue)
            print(
                f"trial{job.trial_number} fold{job.fold} started on {job.device}",
                flush=True,
            )
        while running:
            ready = wait([item[0].sentinel for item in running.values()])
            completed = sorted(
                key for key, item in running.items() if item[0].sentinel in ready
            )
            for key in completed:
                process, result_queue = running.pop(key)
                process.join()
                try:
                    payload = result_queue.get_nowait()
                except queue.Empty:
                    payload = ("failed", "WorkerExit", f"exit code {process.exitcode}")
                finally:
                    result_queue.close()
                    result_queue.join_thread()
                status = payload[0]
                if process.exitcode != 0 or status not in {"completed", "skipped"}:
                    status = "failed"
                results[key] = status
                print(f"trial{key[0]} fold{key[1]} {status}", flush=True)
    except BaseException:
        _terminate_workers(
            {
                index: (process, key[1], result_queue)
                for index, (key, (process, result_queue)) in enumerate(running.items())
            }
        )
        raise
    return results


def _attempt_fold_root(
    trial_root: Path, phase: str, attempt: int, fold: int
) -> Path:
    return trial_root / phase / f"attempt{attempt}" / f"fold{fold}"


def _completed_attempt(
    trial_root: Path, phase: str, fold: int, max_retries: int
) -> Path | None:
    for attempt in range(max_retries + 1):
        root = _attempt_fold_root(trial_root, phase, attempt, fold)
        metadata = root / "metadata.json"
        if not metadata.is_file():
            continue
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "completed":
            return root
    return None


def _next_capacity_attempt(
    trial_root: Path, phase: str, fold: int, max_retries: int
) -> tuple[int, bool] | None:
    for attempt in range(max_retries + 1):
        root = _attempt_fold_root(trial_root, phase, attempt, fold)
        metadata = root / "metadata.json"
        if not root.exists():
            return attempt, False
        if not metadata.is_file():
            continue
        try:
            status = json.loads(metadata.read_text(encoding="utf-8")).get("status")
        except (OSError, json.JSONDecodeError):
            status = "failed"
        if status == "completed":
            return None
        if status == "running":
            return attempt, True
    return None


def _run_capacity_wave(
    trials: Sequence[tuple[int, Path, Path]],
    *,
    phase: str,
    folds: tuple[int, ...],
    devices: tuple[str, ...],
    max_retries: int,
) -> dict[int, dict[int, Path]]:
    if len(trials) * len(folds) > len(devices):
        raise ValueError("Capacity wave has more fold jobs than device slots")
    roots: dict[int, dict[int, Path]] = {number: {} for number, _, _ in trials}
    pending: list[tuple[int, Path, Path, int, int, bool, str]] = []
    slot = 0
    for number, config_path, trial_root in trials:
        for fold in folds:
            completed = _completed_attempt(trial_root, phase, fold, max_retries)
            if completed is not None:
                roots[number][fold] = completed
                slot += 1
                continue
            action = _next_capacity_attempt(trial_root, phase, fold, max_retries)
            if action is not None:
                attempt, resume = action
                pending.append(
                    (
                        number,
                        config_path,
                        trial_root,
                        fold,
                        attempt,
                        resume,
                        devices[slot],
                    )
                )
            slot += 1

    def execute(
        values: Sequence[tuple[int, Path, Path, int, int, bool, str]]
    ) -> dict[tuple[int, int], str]:
        from common.outputs import repository_relative

        jobs = [
            _CapacityFoldJob(
                trial_number=number,
                config_path=repository_relative(config_path),
                fold=fold,
                output_root=repository_relative(
                    trial_root / phase / f"attempt{attempt}"
                ),
                resume=resume,
                device=device,
                show_progress=index == 0,
            )
            for index, (
                number,
                config_path,
                trial_root,
                fold,
                attempt,
                resume,
                device,
            ) in enumerate(values)
        ]
        return _run_assigned_capacity_jobs(jobs)

    first_results = execute(pending) if pending else {}
    retry: list[tuple[int, Path, Path, int, int, bool, str]] = []
    for value in pending:
        number, config_path, trial_root, fold, attempt, _, device = value
        if first_results.get((number, fold)) in {"completed", "skipped"}:
            roots[number][fold] = _attempt_fold_root(
                trial_root, phase, attempt, fold
            )
        elif attempt < max_retries:
            retry.append(
                (number, config_path, trial_root, fold, attempt + 1, False, device)
            )
    retry_results = execute(retry) if retry else {}
    for value in retry:
        number, _, trial_root, fold, attempt, _, _ = value
        if retry_results.get((number, fold)) in {"completed", "skipped"}:
            roots[number][fold] = _attempt_fold_root(
                trial_root, phase, attempt, fold
            )
    return roots


CapacityWaveRunner = Callable[
    [Sequence[tuple[int, Path, Path]]], dict[int, dict[int, Path]]
]


def _freeze_capacity_file(path: Path, payload: Mapping[str, Any]) -> None:
    from common.io import atomic_yaml

    if path.is_file():
        import yaml

        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        if existing != dict(payload):
            raise ValueError(f"Capacity study snapshot changed: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_yaml(path, dict(payload))


def _write_capacity_json(path: Path, payload: Any) -> None:
    from common.io import atomic_json

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, payload)


def _trial_config_path(output: Path, number: int) -> Path:
    return output / "trials" / f"trial_{number:03d}" / "config.yaml"


def _public_capacity_path(path: Path) -> str:
    from common.outputs import repository_relative

    try:
        return repository_relative(path)
    except ValueError:
        return str(path)


def _run_capacity_study(
    *,
    base_config: Any,
    study_config: Any,
    output: Path,
    resume: bool,
    devices: tuple[str, ...],
    run_wave: Callable[..., dict[int, dict[int, Path]]],
    managed_output: bool = False,
) -> dict[str, Any]:
    import math
    import optuna
    from optuna.trial import TrialState

    from stage3.capacity import (
        PRIMARY_METRIC_PATH,
        aggregate_fold_summaries,
        config_for_trial,
        confirmation_trial_numbers,
        suggest_trial_parameters,
        refined_validation_summary,
    )

    if base_config.training.epochs != 20 or base_config.training.seed != 42:
        raise ValueError("Capacity HPO base config requires epochs: 20 and training.seed: 42")
    if base_config.training.device != "cuda":
        raise ValueError("Capacity HPO requires training.device: cuda")
    required_devices = study_config.trials_per_wave * len(study_config.folds)
    if len(devices) != required_devices:
        raise ValueError(
            f"Capacity HPO requires exactly {required_devices} distinct devices"
        )
    if output.exists() and not resume and not managed_output:
        raise FileExistsError(f"Capacity study output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2,
        "base_config": base_config.to_dict(),
        "study": study_config.to_dict(),
    }
    _freeze_capacity_file(output / "study_manifest.yaml", manifest)
    finished_report = output / "confirmation_report.json"
    if resume and finished_report.is_file():
        return json.loads(finished_report.read_text(encoding="utf-8"))

    sampler = optuna.samplers.TPESampler(
        seed=study_config.sampler_seed,
        n_startup_trials=study_config.startup_trials,
    )
    database = output / "study.sqlite3"
    study = optuna.create_study(
        study_name=study_config.study_name,
        storage=f"sqlite:///{database.resolve()}",
        sampler=sampler,
        direction="minimize",
        load_if_exists=resume,
    )
    if not study.trials:
        study.enqueue_trial(
            study_config.baseline,
            user_attrs={"capacity_role": "baseline"},
        )

    while True:
        trials = study.get_trials(deepcopy=False)
        finished = [
            trial
            for trial in trials
            if trial.state in {TrialState.COMPLETE, TrialState.FAIL}
        ]
        if len(finished) >= study_config.attempted_trials:
            break
        running = sorted(
            (trial for trial in trials if trial.state == TrialState.RUNNING),
            key=lambda trial: trial.number,
        )
        wave_numbers: list[int] = [
            trial.number for trial in running[: study_config.trials_per_wave]
        ]
        live_trials: dict[int, Any] = {}
        while (
            len(wave_numbers) < study_config.trials_per_wave
            and len(finished) + len(wave_numbers) < study_config.attempted_trials
        ):
            trial = study.ask()
            suggest_trial_parameters(trial, study_config)
            wave_numbers.append(trial.number)
            live_trials[trial.number] = trial

        trial_specs: list[tuple[int, Path, Path]] = []
        for number in wave_numbers:
            frozen = study.trials[number]
            parameters = frozen.params
            if number in live_trials:
                parameters = live_trials[number].params
            trial_config = config_for_trial(base_config, parameters)
            config_path = _trial_config_path(output, number)
            _freeze_capacity_file(config_path, trial_config.to_dict())
            trial_specs.append((number, config_path, config_path.parent))

        roots = run_wave(
            trial_specs,
            phase="search",
            folds=study_config.folds,
            devices=devices,
            max_retries=study_config.max_retries,
        )
        for number, _, trial_root in sorted(trial_specs):
            try:
                if set(roots.get(number, {})) != set(study_config.folds):
                    raise RuntimeError("one or more search folds failed twice")
                fold_summaries = {
                    fold: refined_validation_summary(
                        roots[number][fold],
                        expected_epochs=base_config.training.epochs,
                    )
                    for fold in study_config.folds
                }
                summary = aggregate_fold_summaries(fold_summaries)
                summary.update(
                    {
                        "trial_number": number,
                        "parameters": dict(study.trials[number].params),
                        "fold_runs": {
                            str(fold): _public_capacity_path(path)
                            for fold, path in roots[number].items()
                        },
                    }
                )
                _write_capacity_json(trial_root / "search_result.json", summary)
                study.tell(number, float(summary["score"]))
            except (FileNotFoundError, ValueError, RuntimeError) as error:
                _write_capacity_json(
                    trial_root / "search_failure.json",
                    {
                        "trial_number": number,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "completed_fold_runs": {
                            str(fold): _public_capacity_path(path)
                            for fold, path in roots.get(number, {}).items()
                        },
                    },
                )
                study.tell(number, state=TrialState.FAIL)
        progress = {
            "attempted": len(
                [
                    trial
                    for trial in study.trials
                    if trial.state in {TrialState.COMPLETE, TrialState.FAIL}
                ]
            ),
            "complete": sum(
                trial.state == TrialState.COMPLETE for trial in study.trials
            ),
            "failed": sum(trial.state == TrialState.FAIL for trial in study.trials),
        }
        _write_capacity_json(output / "search_progress.json", progress)

    complete = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(trial.value)
    ]
    if not any(trial.number == 0 for trial in complete):
        raise RuntimeError("Capacity HPO baseline trial did not complete")
    if len(complete) < study_config.top_k:
        raise RuntimeError("Capacity HPO has too few completed trials for Top-K")
    shortlist = confirmation_trial_numbers(
        [
            {"number": trial.number, "score": trial.value}
            for trial in complete
        ],
        baseline_trial=0,
        top_k=study_config.top_k,
    )
    _write_capacity_json(output / "confirmation_shortlist.json", list(shortlist))

    confirmed: list[dict[str, Any]] = []
    confirmation_failed = False
    for number in shortlist:
        config_path = _trial_config_path(output, number)
        trial_root = config_path.parent
        roots = run_wave(
            [(number, config_path, trial_root)],
            phase="confirmation",
            folds=study_config.confirmation_folds,
            devices=devices,
            max_retries=study_config.max_retries,
        )
        search_result = json.loads(
            (trial_root / "search_result.json").read_text(encoding="utf-8")
        )
        all_roots = {
            int(fold): Path(path)
            for fold, path in search_result["fold_runs"].items()
        }
        all_roots.update(roots.get(number, {}))
        if set(all_roots) != set(range(1, 6)):
            confirmation_failed = True
            confirmed.append(
                {
                    "trial_number": number,
                    "status": "failed",
                    "parameters": dict(study.trials[number].params),
                    "completed_folds": sorted(all_roots),
                    "failed_folds": sorted(set(range(1, 6)) - set(all_roots)),
                }
            )
            continue
        fold_summaries = {
            fold: refined_validation_summary(
                root,
                expected_epochs=base_config.training.epochs,
            )
            for fold, root in sorted(all_roots.items())
        }
        summary = aggregate_fold_summaries(fold_summaries)
        summary.update(
            {
                "trial_number": number,
                "status": "completed",
                "parameters": dict(study.trials[number].params),
                "fold_runs": {
                    str(fold): _public_capacity_path(path)
                    for fold, path in sorted(all_roots.items())
                },
            }
        )
        confirmed.append(summary)
    completed_confirmation = sorted(
        (row for row in confirmed if row["status"] == "completed"),
        key=lambda row: (float(row["score"]), int(row["trial_number"])),
    )
    report = {
        "schema_version": 2,
        "study_name": study_config.study_name,
        "primary_metric": PRIMARY_METRIC_PATH,
        "model_selector": "taskwise_refined",
        "shortlist": list(shortlist),
        "confirmation_failed": confirmation_failed,
        "ranking": completed_confirmation,
        "failed": [row for row in confirmed if row["status"] == "failed"],
    }
    _write_capacity_json(finished_report, report)
    return json.loads(finished_report.read_text(encoding="utf-8"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one or more Stage 3 folds with bounded process concurrency."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, nargs="+")
    parser.add_argument(
        "--study-config",
        help="capacity-v1 HPO study YAML; uses folds and waves frozen by the study",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-parallel", type=_positive_int, default=1)
    parser.add_argument(
        "--devices",
        help="optional comma-separated GPU list such as cuda:0,cuda:1",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    from stage3.config import load_stage3_config, validate_stage3_folds

    try:
        devices = _parse_devices(args.devices)
    except ValueError as error:
        parser.error(str(error))

    from common.outputs import repository_path
    config = load_stage3_config(args.config)
    output = repository_path(args.output)
    if args.study_config is not None:
        if args.fold is not None:
            parser.error("--study-config forbids --fold")
        if args.max_parallel != 1:
            parser.error("--study-config manages concurrency and forbids --max-parallel")
        from stage3.capacity import (
            load_capacity_study_config,
            validate_anchor_decision,
        )

        study_config = load_capacity_study_config(args.study_config)
        anchor_decision = validate_anchor_decision(study_config, args.config)
        from common.identity import semantic_identity
        from common.outputs import open_run_directory
        from stage3.train import resolve_stage3_training_identity

        base_fold_identities = {
            f"fold{fold}": resolve_stage3_training_identity(config, fold)["hash"]
            for fold in study_config.folds
        }
        study_payload = study_config.to_dict()
        study_payload.pop("anchor_decision")
        study_identity = semantic_identity(
            "stage3.capacity-study",
            {
                "contract_version": 1,
                "base_fold_training_identities": base_fold_identities,
                "study": study_payload,
                "anchor": {
                    "selected_candidate": anchor_decision["selected_candidate"],
                    "selected_config": anchor_decision["selected_config"],
                },
            },
        )
        resume_locator = (
            Path(args.output) / "study.sqlite3" if args.resume else None
        )
        run = open_run_directory(
            stage="stage3",
            operation="capacity_hpo",
            config_path=args.study_config,
            config_payload={
                "stage3": config.to_dict(),
                "capacity_study": study_config.to_dict(),
                "anchor_decision": anchor_decision,
            },
            semantic_identity=study_identity,
            output=args.output,
            seed=study_config.sampler_seed,
            resume=resume_locator,
            details={
                "devices": list(devices),
                "base_config": args.config,
                "anchor_candidate": anchor_decision["selected_candidate"],
            },
        )
        try:
            report = _run_capacity_study(
                base_config=config,
                study_config=study_config,
                output=run.root,
                resume=args.resume,
                devices=devices,
                run_wave=_run_capacity_wave,
                managed_output=True,
            )
            run.complete(report)
        except KeyboardInterrupt:
            run.fail()
            print("Stage3 capacity HPO interrupted", file=sys.stderr)
            return 130
        except BaseException:
            run.fail()
            raise
        print(f"Stage3 capacity HPO complete: {args.output}")
        return 1 if report["confirmation_failed"] else 0

    if args.fold is None:
        parser.error("--fold is required unless --study-config is used")
    try:
        folds = validate_stage3_folds(args.fold)
    except ValueError as error:
        parser.error(str(error))
    effective_parallel = min(args.max_parallel, len(folds))
    if effective_parallel > 1 and not devices:
        parser.error("--devices is required when more than one fold runs in parallel")
    if devices and config.training.device != "cuda":
        parser.error("--devices requires training.device: cuda")

    try:
        results = _run_schedule(
            config_path=args.config,
            folds=folds,
            output_root=args.output,
            resume=args.resume,
            max_parallel=args.max_parallel,
            devices=devices,
        )
    except KeyboardInterrupt:
        print("Stage3 fold training interrupted", file=sys.stderr)
        return 130

    print("Stage3 folds complete")
    for status in ("completed", "skipped", "failed"):
        selected = [str(fold) for fold in folds if results.get(fold) == status]
        print(f"{status}: {', '.join(selected) if selected else '-'}")
    return 1 if any(status == "failed" for status in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
