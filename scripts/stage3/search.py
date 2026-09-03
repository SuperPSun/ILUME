from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in value.split(","))
    if not devices or any(not re.fullmatch(r"cuda:\d+", item) for item in devices):
        raise ValueError("--devices must be a comma-separated list such as cuda:0,cuda:1")
    return devices


def _phase_name(phase: str) -> str:
    return {"a": "search_a", "b": "search_b", "c": "search_c"}[phase]


def _read_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("ranking"), list):
        raise ValueError(f"Malformed prerequisite Stage 3 search result: {path}")
    return payload


def _candidate_catalog(
    phase: str,
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    from common.io import sha256_file
    from stage3.search import expert_candidates, grouping_candidates

    prerequisites: dict[str, str] = {}
    if phase == "a":
        return [item.to_dict() for item in grouping_candidates()], prerequisites

    a_path = output_root / "search_a" / "result.json"
    a_result = _read_result(a_path)
    prerequisites["search_a"] = sha256_file(a_path)
    grouping_fields = (
        "candidate_id", "source", "group_count", "reason", "assignments"
    )
    top_groupings = [
        {name: row["candidate"][name] for name in grouping_fields}
        for row in a_result["ranking"][:3]
    ]
    if len(top_groupings) != 3:
        raise ValueError("Search A did not publish Top-3 grouping candidates")
    if phase == "b":
        catalog = []
        for source_index, source in enumerate(("local", "ablation", "higher_capacity")):
            values = [item for item in expert_candidates() if item.source == source]
            if len(values) != 10:
                raise AssertionError("Search B source allocation changed")
            for index, experts in enumerate(values):
                grouping = top_groupings[(index + source_index) % 3]
                catalog.append(
                    {
                        "candidate_id": f"b-{len(catalog):02d}",
                        "source": source,
                        "grouping": grouping,
                        "experts": experts.to_dict(),
                    }
                )
        counts = {
            grouping["candidate_id"]: sum(
                row["grouping"]["candidate_id"] == grouping["candidate_id"]
                for row in catalog
            )
            for grouping in top_groupings
        }
        if set(counts.values()) != {10}:
            raise AssertionError("Search B must assign ten trials to each grouping")
        return catalog, prerequisites

    b_path = output_root / "search_b" / "result.json"
    b_result = _read_result(b_path)
    prerequisites["search_b"] = sha256_file(b_path)
    top_experts: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for row in b_result["ranking"]:
        experts = row["candidate"]["experts"]
        key = (
            int(experts["global_experts"]),
            int(experts["group_experts"]),
            int(experts["private_experts"]),
        )
        if key not in seen:
            seen.add(key)
            top_experts.append(experts)
        if len(top_experts) == 3:
            break
    if len(top_experts) != 3:
        raise ValueError("Search B did not publish three distinct expert structures")
    catalog = [
        {
            "candidate_id": f"c-pair-{index:02d}",
            "grouping": grouping,
            "experts": experts,
        }
        for index, (grouping, experts) in enumerate(
            itertools.product(top_groupings, top_experts),
            start=1,
        )
    ]
    if len(catalog) != 9:
        raise AssertionError("Search C requires a 3x3 candidate cross product")
    return catalog, prerequisites


def _suggest_parameters(trial: Any, phase: str, catalog: Sequence[Mapping[str, Any]], spec: Any) -> dict[str, Any]:
    ids = [str(item["candidate_id"]) for item in catalog]
    candidate_id = trial.suggest_categorical("candidate_id", ids)
    result: dict[str, Any] = {"candidate_id": candidate_id}
    if phase == "c":
        result.update(
            {
                "tier1_weight": trial.suggest_float("tier1_weight", *spec.tier_weight),
                "tier2_weight": trial.suggest_float("tier2_weight", *spec.tier_weight),
                "tier3_weight": trial.suggest_float("tier3_weight", *spec.tier_weight),
                "learning_rate": trial.suggest_float(
                    "learning_rate", *spec.learning_rate, log=True
                ),
                "dropout": trial.suggest_float("dropout", *spec.dropout),
                "weight_decay": trial.suggest_float(
                    "weight_decay", *spec.weight_decay, log=True
                ),
            }
        )
    return result


def _trial_config(base: Any, phase: str, candidate: Mapping[str, Any], parameters: Mapping[str, Any]) -> Any:
    from stage3.search import (
        ExpertCandidate,
        GroupingCandidate,
        config_for_experts,
        config_for_grouping,
        config_for_search_c,
    )

    def grouping(raw: Mapping[str, Any]) -> GroupingCandidate:
        return GroupingCandidate(**raw)

    def experts(raw: Mapping[str, Any]) -> ExpertCandidate:
        return ExpertCandidate(**raw)

    if phase == "a":
        return config_for_grouping(base, grouping(candidate))
    if phase == "b":
        return config_for_experts(
            config_for_grouping(base, grouping(candidate["grouping"])),
            experts(candidate["experts"]),
        )
    return config_for_search_c(
        base,
        grouping(candidate["grouping"]),
        experts(candidate["experts"]),
        parameters,
    )


def _write_trial_files(
    root: Path,
    config: Any,
    candidate: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> Path:
    from common.io import atomic_json, atomic_yaml

    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.yaml"
    payload = config.to_dict()
    if config_path.is_file():
        existing = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"Stage 3 search trial config changed: {root.name}")
    else:
        atomic_yaml(config_path, payload)
    manifest = {"candidate": dict(candidate), "parameters": dict(parameters)}
    manifest_path = root / "candidate.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"Stage 3 search candidate changed: {root.name}")
    else:
        atomic_json(manifest_path, manifest)
    return config_path


def _run_study(
    *,
    phase: str,
    base: Any,
    spec: Any,
    catalog: Sequence[Mapping[str, Any]],
    phase_root: Path,
    resume: bool,
    devices: tuple[str, ...],
    max_parallel: int,
) -> dict[str, Any]:
    import optuna
    from optuna.trial import TrialState
    from common.io import atomic_json
    from scripts.stage3.train import _run_capacity_wave
    from stage3.search import aggregate_search_trial, fold_search_summary, rank_trials

    target = {"a": 50, "b": 30, "c": 20}[phase]
    if len(catalog) != ({"a": 50, "b": 30, "c": 9}[phase]):
        raise ValueError(f"Stage 3 Search {phase.upper()} candidate catalog changed")
    phase_root.mkdir(parents=True, exist_ok=True)
    storage = phase_root / "study.sqlite3"
    sampler = optuna.samplers.TPESampler(seed=spec.sampler_seed, n_startup_trials=9)
    study = optuna.create_study(
        study_name=f"{spec.study_name}-{phase}",
        storage=f"sqlite:///{storage.resolve()}",
        sampler=sampler,
        direction="minimize",
        load_if_exists=resume,
    )
    if not study.trials:
        if phase in {"a", "b"}:
            for item in catalog:
                study.enqueue_trial({"candidate_id": item["candidate_id"]})
        else:
            baseline = {
                "tier1_weight": 1.0,
                "tier2_weight": 1.0,
                "tier3_weight": 1.0,
                "learning_rate": 3.0e-4,
                "dropout": 0.10,
                "weight_decay": 1.0e-2,
            }
            for item in catalog:
                study.enqueue_trial({"candidate_id": item["candidate_id"], **baseline})

    by_id = {str(item["candidate_id"]): item for item in catalog}
    wave_trials = max_parallel // len(spec.folds)
    while True:
        finished = [
            trial
            for trial in study.trials
            if trial.state in {TrialState.COMPLETE, TrialState.FAIL}
        ]
        if len(finished) >= target:
            break
        running = sorted(
            (trial for trial in study.trials if trial.state == TrialState.RUNNING),
            key=lambda item: item.number,
        )
        numbers = [trial.number for trial in running[:wave_trials]]
        live: dict[int, Any] = {}
        while len(numbers) < wave_trials and len(finished) + len(numbers) < target:
            trial = study.ask()
            _suggest_parameters(trial, phase, catalog, spec)
            numbers.append(trial.number)
            live[trial.number] = trial

        trial_specs = []
        trial_details: dict[int, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
        for number in numbers:
            parameters = dict(
                live[number].params if number in live else study.trials[number].params
            )
            candidate = by_id[str(parameters["candidate_id"])]
            config = _trial_config(base, phase, candidate, parameters)
            trial_root = phase_root / "trials" / f"trial_{number:03d}"
            config_path = _write_trial_files(
                trial_root, config, candidate, parameters
            )
            trial_specs.append((number, config_path, trial_root))
            trial_details[number] = (candidate, parameters)

        roots = _run_capacity_wave(
            trial_specs,
            phase="screen",
            folds=spec.folds,
            devices=devices[: len(trial_specs) * len(spec.folds)],
            max_retries=spec.max_retries,
        )
        for number, _, trial_root in trial_specs:
            try:
                if set(roots.get(number, {})) != set(spec.folds):
                    raise RuntimeError("one or more Stage 3 search folds failed twice")
                fold_summaries = {
                    fold: fold_search_summary(root, expected_epochs=spec.epochs)
                    for fold, root in roots[number].items()
                }
                candidate, parameters = trial_details[number]
                row = aggregate_search_trial(
                    fold_summaries,
                    trial_number=number,
                    candidate={**dict(candidate), "parameters": dict(parameters)},
                )
                row["fold_runs"] = {
                    str(fold): root.relative_to(ROOT).as_posix()
                    for fold, root in roots[number].items()
                }
                atomic_json(trial_root / "search_result.json", row)
                study.tell(number, float(row["score"]))
            except (FileNotFoundError, ValueError, RuntimeError) as error:
                atomic_json(
                    trial_root / "search_failure.json",
                    {
                        "trial_number": number,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                study.tell(number, state=TrialState.FAIL)
        atomic_json(
            phase_root / "progress.json",
            {
                "attempted": sum(
                    trial.state in {TrialState.COMPLETE, TrialState.FAIL}
                    for trial in study.trials
                ),
                "complete": sum(trial.state == TrialState.COMPLETE for trial in study.trials),
                "failed": sum(trial.state == TrialState.FAIL for trial in study.trials),
                "budget": target,
            },
        )

    rows = []
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE:
            continue
        path = phase_root / "trials" / f"trial_{trial.number:03d}" / "search_result.json"
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if len(rows) < spec.top_k:
        raise RuntimeError(f"Search {phase.upper()} has fewer than three completed trials")
    ranking = rank_trials(rows)
    result = {
        "schema_version": 1,
        "phase": phase.upper(),
        "attempted_trials": target,
        "completed_trials": len(ranking),
        "failed_trials": target - len(ranking),
        "primary_metric": "weak_property_weighted_validation_normalized_mae",
        "ranking": ranking,
        "top3": ranking[:3],
    }
    if phase == "a":
        result["top3_groupings"] = [
            row["candidate"] for row in ranking[:3]
        ]
    elif phase == "b":
        result["top3_expert_structures"] = [
            row["candidate"]["experts"] for row in ranking[:3]
        ]
    else:
        result["top3_recipes"] = ranking[:3]
        result["winner"] = ranking[0]
        winner_config = phase_root / "trials" / f"trial_{ranking[0]['trial_number']:03d}" / "config.yaml"
        winner_payload = __import__("yaml").safe_load(winner_config.read_text(encoding="utf-8"))
        from common.io import atomic_yaml

        atomic_yaml(phase_root / "winner_base.yaml", winner_payload)
    atomic_json(phase_root / "result.json", result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed 100-trial ILUME v2 Stage 3 A/B/C search."
    )
    parser.add_argument("--study-config", required=True)
    parser.add_argument("--phase", required=True, choices=("a", "b", "c"))
    parser.add_argument("--output", required=True, help="shared v2 search output root")
    parser.add_argument("--devices", required=True)
    parser.add_argument("--max-parallel", type=_positive_int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        devices = _parse_devices(args.devices)
    except ValueError as error:
        parser.error(str(error))
    if args.max_parallel < 2 or args.max_parallel % 2:
        parser.error("--max-parallel must be an even number of fold workers")
    if len(devices) != args.max_parallel:
        parser.error("--devices must contain exactly --max-parallel slots")

    from common.identity import semantic_identity
    from common.io import atomic_json
    from common.outputs import open_run_directory, repository_path
    from stage3.config import load_stage3_config
    from stage3.search import load_search_config, search_base_config

    spec = load_search_config(repository_path(args.study_config))
    base = search_base_config(load_stage3_config(repository_path(spec.base_config)), spec)
    output_root = repository_path(args.output)
    catalog, prerequisites = _candidate_catalog(args.phase, output_root)
    phase_root = output_root / _phase_name(args.phase)
    manifest = {
        "schema_version": 1,
        "phase": args.phase.upper(),
        "search": spec.to_dict(),
        "base_config": base.to_dict(),
        "prerequisites": prerequisites,
        "candidates": catalog,
    }
    identity = semantic_identity(
        "stage3.v2-search",
        {"contract_version": 1, "manifest": manifest},
    )
    resume_locator = phase_root / "study.sqlite3" if args.resume else None
    run = open_run_directory(
        stage="stage3",
        operation=f"v2_search_{args.phase}",
        config_path=args.study_config,
        config_payload=manifest,
        semantic_identity=identity,
        output=phase_root.relative_to(ROOT),
        seed=spec.sampler_seed,
        resume=resume_locator,
        details={
            "devices": list(devices),
            "max_parallel": args.max_parallel,
            "base_config": spec.base_config,
        },
    )
    atomic_json(phase_root / "candidate_manifest.json", manifest)
    try:
        result = _run_study(
            phase=args.phase,
            base=base,
            spec=spec,
            catalog=catalog,
            phase_root=phase_root,
            resume=args.resume,
            devices=devices,
            max_parallel=args.max_parallel,
        )
        run.complete(result)
    except KeyboardInterrupt:
        run.fail()
        return 130
    except BaseException:
        run.fail()
        raise
    print(f"Stage 3 Search {args.phase.upper()} complete: {phase_root.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
