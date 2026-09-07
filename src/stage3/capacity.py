from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from common.io import sha256_file

RECIPE_TIE_PRIORITY = {
    "r4": 0,
    "r3": 1,
    "r5": 2,
    "r2": 3,
    "r6": 4,
    "r1": 5,
    "r7": 6,
    "r8": 7,
}


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _nested_metric(row: Mapping[str, Any], keys: Sequence[str], context: str) -> float:
    value: Any = row
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"{context} is missing {'.'.join(keys)}")
        value = value[key]
    return _finite_float(value, context)


def refined_validation_summary(
    run_root: str | Path, *, expected_epochs: int
) -> dict[str, Any]:
    path = Path(run_root) / "taskwise_refinement.json"
    if not path.is_file():
        raise FileNotFoundError(f"Stage 3 refinement manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Stage 3 refinement manifest is unreadable: {path}") from error
    if not isinstance(payload, dict) or payload.get("kind") != "ilume_stage3_taskwise_refined":
        raise ValueError("Stage 3 refinement manifest has the wrong contract")
    artifact = Path(run_root) / str(payload.get("artifact", ""))
    if (
        artifact.name != "taskwise_refined.pt"
        or not artifact.is_file()
        or payload.get("artifact_sha256") != sha256_file(artifact)
    ):
        raise ValueError("Stage 3 refined artifact is missing or corrupt")
    if payload.get("ordinary_final_epoch") != expected_epochs:
        raise ValueError("Stage 3 refinement manifest has the wrong epoch budget")
    validation = payload.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("Stage 3 refinement manifest lacks stitched validation")
    score = _nested_metric(
        validation,
        ("macro_task_equal", "normalized_mae", "value"),
        "stitched validation primary metric",
    )
    tasks = validation.get("tasks")
    groups = validation.get("groups")
    if not isinstance(tasks, Mapping) or not isinstance(groups, Mapping):
        raise ValueError("Stage 3 stitched validation lacks task/group metrics")
    return {
        "run_root": str(run_root),
        "expected_epochs": expected_epochs,
        "model_selector": "taskwise_refined",
        "score": score,
        "group_equal_score": _nested_metric(
            validation,
            ("macro_group_equal", "normalized_mae", "value"),
            "stitched validation group-equal metric",
        ),
        "task_scores": {
            task: _finite_float(values["normalized_mae"], f"{task} normalized MAE")
            for task, values in tasks.items()
        },
        "group_scores": {
            group: _finite_float(values["normalized_mae"], f"{group} normalized MAE")
            for group, values in groups.items()
        },
    }


def aggregate_fold_summaries(
    summaries: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    if not summaries:
        raise ValueError("At least one fold summary is required")
    fold_scores = {
        str(fold): _finite_float(summary.get("score"), f"fold{fold} score")
        for fold, summary in sorted(summaries.items())
    }
    values = list(fold_scores.values())
    task_ids = sorted(
        set.intersection(
            *(set(summary.get("task_scores", {})) for summary in summaries.values())
        )
    )
    group_ids = sorted(
        set.intersection(
            *(set(summary.get("group_scores", {})) for summary in summaries.values())
        )
    )
    return {
        "fold_scores": fold_scores,
        "score": statistics.fmean(values),
        "fold_sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "group_equal_score": statistics.fmean(
            _finite_float(
                summary.get("group_equal_score"), f"fold{fold} group-equal score"
            )
            for fold, summary in summaries.items()
        ),
        "task_scores": {
            task: statistics.fmean(
                _finite_float(
                    summary["task_scores"][task], f"fold{fold} {task} score"
                )
                for fold, summary in summaries.items()
            )
            for task in task_ids
        },
        "group_scores": {
            group: statistics.fmean(
                _finite_float(
                    summary["group_scores"][group], f"fold{fold} {group} score"
                )
                for fold, summary in summaries.items()
            )
            for group in group_ids
        },
    }


def select_probe_winners(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_scale: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in candidates:
        scale = str(candidate["scale"])
        recipe = str(candidate["recipe"]).lower()
        if recipe not in RECIPE_TIE_PRIORITY:
            raise ValueError(f"Unknown Stage 2 Base selection recipe: {recipe}")
        _finite_float(candidate.get("score"), f"{scale}/{recipe} score")
        by_scale.setdefault(scale, []).append(candidate)
    winners = []
    for scale, values in by_scale.items():
        selected = min(
            values,
            key=lambda row: (
                float(row["score"]),
                RECIPE_TIE_PRIORITY[str(row["recipe"]).lower()],
            ),
        )
        winners.append(dict(selected))
    return sorted(winners, key=lambda row: str(row["scale"]))


def summarize_capacity_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        raise ValueError("capacity report manifest requires schema_version: 2")
    kind = raw.get("kind")
    expected_epochs = int(raw.get("expected_epochs", 0))
    if expected_epochs <= 0:
        raise ValueError("capacity report epoch count must be positive")
    if kind in {"probe", "comparison"}:
        candidates = raw.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("capacity candidate manifest cannot be empty")
        summaries: list[dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict) or not isinstance(
                candidate.get("folds"), dict
            ):
                raise ValueError("capacity candidate requires fold paths")
            fold_summaries = {
                int(fold): refined_validation_summary(
                    run_root,
                    expected_epochs=expected_epochs,
                )
                for fold, run_root in candidate["folds"].items()
            }
            summary = aggregate_fold_summaries(fold_summaries)
            summary.update(
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "folds"
                }
            )
            summary["fold_runs"] = {
                str(fold): str(run_root)
                for fold, run_root in candidate["folds"].items()
            }
            summaries.append(summary)
        ranking = sorted(
            summaries,
            key=lambda row: (float(row["score"]), str(row.get("id", ""))),
        )
        result: dict[str, Any] = {
            "schema_version": 2,
            "kind": kind,
            "expected_epochs": expected_epochs,
            "model_selector": "taskwise_refined",
            "ranking": ranking,
        }
        if kind == "probe":
            result["scale_winners"] = select_probe_winners(summaries)
        return result
    if kind == "robustness":
        runs = raw.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError("capacity robustness manifest cannot be empty")
        by_seed: dict[int, dict[int, dict[str, Any]]] = {}
        run_paths: dict[str, str] = {}
        for run in runs:
            if not isinstance(run, dict):
                raise ValueError("capacity robustness run must be a mapping")
            seed, fold = int(run["seed"]), int(run["fold"])
            if fold in by_seed.setdefault(seed, {}):
                raise ValueError(f"duplicate robustness seed/fold: {seed}/{fold}")
            by_seed[seed][fold] = refined_validation_summary(
                run["path"],
                expected_epochs=expected_epochs,
            )
            run_paths[f"seed{seed}/fold{fold}"] = str(run["path"])
        seed_summaries = {
            seed: aggregate_fold_summaries(folds)
            for seed, folds in sorted(by_seed.items())
        }
        seed_scores = [summary["score"] for summary in seed_summaries.values()]
        task_ids = sorted(
            set.intersection(
                *(set(summary["task_scores"]) for summary in seed_summaries.values())
            )
        )
        return {
            "schema_version": 2,
            "kind": kind,
            "expected_epochs": expected_epochs,
            "model_selector": "taskwise_refined",
            "run_paths": run_paths,
            "seeds": {str(seed): value for seed, value in seed_summaries.items()},
            "seed_score_mean": statistics.fmean(seed_scores),
            "seed_score_sample_sd": (
                statistics.stdev(seed_scores) if len(seed_scores) > 1 else 0.0
            ),
            "seed_score_range": max(seed_scores) - min(seed_scores),
            "worst_seed": max(
                seed_summaries, key=lambda seed: seed_summaries[seed]["score"]
            ),
            "task_seed_variation": {
                task: {
                    "mean": statistics.fmean(
                        summary["task_scores"][task]
                        for summary in seed_summaries.values()
                    ),
                    "sample_sd": (
                        statistics.stdev(
                            summary["task_scores"][task]
                            for summary in seed_summaries.values()
                        )
                        if len(seed_summaries) > 1
                        else 0.0
                    ),
                    "range": max(
                        summary["task_scores"][task]
                        for summary in seed_summaries.values()
                    )
                    - min(
                        summary["task_scores"][task]
                        for summary in seed_summaries.values()
                    ),
                }
                for task in task_ids
            },
        }
    raise ValueError("capacity report kind must be probe, robustness, or comparison")


__all__ = [
    "aggregate_fold_summaries",
    "refined_validation_summary",
    "select_probe_winners",
    "summarize_capacity_manifest",
]
