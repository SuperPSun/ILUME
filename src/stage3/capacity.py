from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from common.io import sha256_file

from .config import Stage3Config


PRIMARY_METRIC_PATH = "taskwise_refinement.validation.macro_task_equal.normalized_mae.value"
RECIPE_TIE_PRIORITY = {"default": 0, "conservative": 1, "aggressive": 2}


@dataclass(frozen=True)
class CapacityStudyConfig:
    study_name: str
    anchor_decision: str
    attempted_trials: int
    startup_trials: int
    trials_per_wave: int
    folds: tuple[int, ...]
    confirmation_folds: tuple[int, ...]
    top_k: int
    max_retries: int
    sampler_seed: int
    global_experts: tuple[int, ...]
    group_experts: tuple[int, ...]
    private_experts: tuple[int, ...]
    expert_hidden_ratio: tuple[float, ...]
    dropout: tuple[float, float]
    learning_rate: tuple[float, float]
    weight_decay: tuple[float, float]
    baseline: dict[str, int | float]

    def validate(self) -> None:
        if not self.study_name:
            raise ValueError("capacity study_name cannot be empty")
        if not self.anchor_decision:
            raise ValueError("capacity anchor_decision cannot be empty")
        if self.attempted_trials <= 0 or self.startup_trials <= 0:
            raise ValueError("capacity trial counts must be positive")
        if self.startup_trials > self.attempted_trials:
            raise ValueError("capacity startup_trials exceeds attempted_trials")
        if self.trials_per_wave != 2:
            raise ValueError("capacity v1 requires trials_per_wave == 2")
        if self.folds != (1, 2) or self.confirmation_folds != (3, 4, 5):
            raise ValueError("capacity v1 fixes search folds 1/2 and confirmation folds 3/4/5")
        if self.top_k <= 0:
            raise ValueError("capacity top_k must be positive")
        if self.max_retries != 1:
            raise ValueError("capacity v1 permits exactly one identical retry")
        for name in (
            "global_experts",
            "group_experts",
            "private_experts",
            "expert_hidden_ratio",
        ):
            if not getattr(self, name):
                raise ValueError(f"capacity {name} cannot be empty")
        for name in ("dropout", "learning_rate", "weight_decay"):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"capacity {name} bounds are reversed")
        required = {
            "global_experts",
            "group_experts",
            "private_experts",
            "expert_hidden_ratio",
            "dropout",
            "learning_rate",
            "weight_decay",
        }
        if set(self.baseline) != required:
            raise ValueError("capacity baseline must define all seven search variables")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "folds",
            "confirmation_folds",
            "global_experts",
            "group_experts",
            "private_experts",
            "expert_hidden_ratio",
            "dropout",
            "learning_rate",
            "weight_decay",
        ):
            payload[name] = list(payload[name])
        return payload


def load_capacity_study_config(path: str | Path) -> CapacityStudyConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict) or raw.pop("schema_version", None) != 2:
        raise ValueError("capacity study requires schema_version: 2")
    allowed = set(CapacityStudyConfig.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            "Unknown capacity study fields: " + ", ".join(sorted(unknown))
        )
    tuple_fields = (
        "folds",
        "confirmation_folds",
        "global_experts",
        "group_experts",
        "private_experts",
        "expert_hidden_ratio",
        "dropout",
        "learning_rate",
        "weight_decay",
    )
    for name in tuple_fields:
        if name in raw:
            raw[name] = tuple(raw[name])
    config = CapacityStudyConfig(**raw)
    config.validate()
    return config


def validate_anchor_decision(
    config: CapacityStudyConfig, base_config_path: str | Path
) -> dict[str, Any]:
    from common.outputs import repository_path, repository_relative

    path = repository_path(config.anchor_decision)
    if not path.is_file():
        raise FileNotFoundError(f"Capacity anchor decision is missing: {config.anchor_decision}")
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("capacity anchor decision requires schema_version: 1")
    if raw.get("kind") != "anchor" or not str(raw.get("reason", "")).strip():
        raise ValueError("capacity anchor decision requires kind: anchor and a reason")
    selected_config = raw.get("selected_config")
    if selected_config != repository_relative(base_config_path):
        raise ValueError("capacity anchor decision does not select --config")
    probe_report = repository_path(str(raw.get("probe_report", "")))
    if not probe_report.is_file():
        raise FileNotFoundError("capacity anchor decision probe report is missing")
    report = json.loads(probe_report.read_text(encoding="utf-8"))
    winners = report.get("scale_winners")
    if not isinstance(winners, list) or raw.get("selected_candidate") not in {
        winner.get("id") for winner in winners if isinstance(winner, dict)
    }:
        raise ValueError("capacity anchor decision is not one of the probe scale winners")
    return raw


def suggest_trial_parameters(trial: Any, config: CapacityStudyConfig) -> dict[str, Any]:
    return {
        "global_experts": trial.suggest_categorical(
            "global_experts", list(config.global_experts)
        ),
        "group_experts": trial.suggest_categorical(
            "group_experts", list(config.group_experts)
        ),
        "private_experts": trial.suggest_categorical(
            "private_experts", list(config.private_experts)
        ),
        "expert_hidden_ratio": trial.suggest_categorical(
            "expert_hidden_ratio", list(config.expert_hidden_ratio)
        ),
        "dropout": trial.suggest_float("dropout", *config.dropout),
        "learning_rate": trial.suggest_float(
            "learning_rate", *config.learning_rate, log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", *config.weight_decay, log=True
        ),
    }


def config_for_trial(
    base: Stage3Config, parameters: Mapping[str, int | float]
) -> Stage3Config:
    return replace(
        base,
        model=replace(
            base.model,
            global_experts=int(parameters["global_experts"]),
            group_experts=int(parameters["group_experts"]),
            private_experts=int(parameters["private_experts"]),
            expert_hidden_ratio=float(parameters["expert_hidden_ratio"]),
            dropout=float(parameters["dropout"]),
        ),
        training=replace(
            base.training,
            learning_rate=float(parameters["learning_rate"]),
            weight_decay=float(parameters["weight_decay"]),
        ),
    )


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
        int(fold): _finite_float(summary.get("score"), f"fold{fold} score")
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
            raise ValueError(f"Unknown Stage 2 capacity recipe: {recipe}")
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


def confirmation_trial_numbers(
    completed_trials: Sequence[Mapping[str, Any]],
    *,
    baseline_trial: int,
    top_k: int,
) -> tuple[int, ...]:
    ranked = sorted(
        (
            (int(row["number"]), _finite_float(row.get("score"), "trial score"))
            for row in completed_trials
        ),
        key=lambda item: (item[1], item[0]),
    )
    selected = [number for number, _ in ranked[:top_k]]
    if baseline_trial not in selected:
        selected.append(baseline_trial)
    return tuple(selected)


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


def materialize_final_recipe_configs(
    decision_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    from common.io import atomic_yaml
    from common.outputs import repository_path
    from .config import load_stage3_config

    decision_path = Path(decision_path)
    with decision_path.open(encoding="utf-8") as handle:
        decision = yaml.safe_load(handle) or {}
    if not isinstance(decision, dict) or decision.get("schema_version") != 1:
        raise ValueError("final recipe decision requires schema_version: 1")
    if decision.get("kind") != "final_recipe" or not str(
        decision.get("reason", "")
    ).strip():
        raise ValueError("final recipe decision requires kind and a reason")
    trial_number = int(decision.get("trial_number", -1))
    hpo_value = str(decision.get("hpo_output", ""))
    hpo_output = Path(hpo_value)
    if not hpo_output.is_absolute():
        hpo_output = repository_path(hpo_value)
    confirmation = json.loads(
        (hpo_output / "confirmation_report.json").read_text(encoding="utf-8")
    )
    completed = {
        int(row["trial_number"])
        for row in confirmation.get("ranking", [])
        if isinstance(row, dict)
    }
    if trial_number not in completed:
        raise ValueError("selected final recipe is not a completed confirmed trial")
    selected_path = hpo_output / "trials" / f"trial_{trial_number:03d}" / "config.yaml"
    selected = load_stage3_config(selected_path)
    search = json.loads(
        (selected_path.parent / "search_result.json").read_text(encoding="utf-8")
    )
    scale_configs = decision.get("scale_configs")
    if not isinstance(scale_configs, dict) or set(scale_configs) != {
        "s",
        "base",
        "l",
        "xl",
    }:
        raise ValueError("final recipe decision requires s/base/l/xl scale_configs")
    seed_output_root = str(decision.get("seed_output_root", ""))
    formal_output_root = str(decision.get("formal_output_root", ""))
    if not seed_output_root or not formal_output_root:
        raise ValueError("final recipe decision requires seed and formal output roots")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def freeze(path: Path, payload: Mapping[str, Any]) -> None:
        if path.is_file():
            existing = yaml.safe_load(path.read_text(encoding="utf-8"))
            if existing != dict(payload):
                raise ValueError(f"materialized capacity config changed: {path.name}")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_yaml(path, dict(payload))

    freeze(output_dir / "decision.yaml", decision)
    seeds = (42, 10042, 20042, 30042, 40042)
    generated: dict[str, str] = {}
    for seed in seeds:
        config = replace(
            selected,
            training=replace(selected.training, seed=seed, epochs=20),
        )
        config.validate()
        path = output_dir / "seed" / f"seed{seed}.yaml"
        freeze(path, config.to_dict())
        generated[f"seed{seed}"] = str(path)

    for scale, config_value in sorted(scale_configs.items()):
        config_path = Path(str(config_value))
        if not config_path.is_absolute():
            config_path = repository_path(config_path)
        probe = load_stage3_config(config_path)
        config = replace(
            probe,
            model=selected.model,
            training=replace(selected.training, seed=42, epochs=50),
        )
        config.validate()
        path = output_dir / "formal" / f"{scale}.yaml"
        freeze(path, config.to_dict())
        generated[f"formal-{scale}"] = str(path)

    robustness_runs = []
    search_roots = {int(fold): path for fold, path in search["fold_runs"].items()}
    for seed in seeds:
        for fold in (1, 2):
            path = (
                search_roots[fold]
                if seed == 42
                else f"{seed_output_root}/seed{seed}/fold{fold}"
            )
            robustness_runs.append({"seed": seed, "fold": fold, "path": path})
    freeze(
        output_dir / "robustness-report.yaml",
        {
            "schema_version": 2,
            "kind": "robustness",
            "expected_epochs": 20,
            "model_selector": "taskwise_refined",
            "runs": robustness_runs,
        },
    )
    freeze(
        output_dir / "formal-report.yaml",
        {
            "schema_version": 2,
            "kind": "comparison",
            "expected_epochs": 50,
            "model_selector": "taskwise_refined",
            "candidates": [
                {
                    "id": scale,
                    "scale": scale.upper() if scale != "base" else "Base",
                    "folds": {
                        fold: f"{formal_output_root}/{scale}/fold{fold}"
                        for fold in range(1, 6)
                    },
                }
                for scale in ("s", "base", "l", "xl")
            ],
        },
    )
    return {
        "trial_number": trial_number,
        "generated_configs": generated,
        "robustness_manifest": str(output_dir / "robustness-report.yaml"),
        "formal_manifest": str(output_dir / "formal-report.yaml"),
    }


__all__ = [
    "CapacityStudyConfig",
    "PRIMARY_METRIC_PATH",
    "aggregate_fold_summaries",
    "config_for_trial",
    "confirmation_trial_numbers",
    "load_capacity_study_config",
    "materialize_final_recipe_configs",
    "select_probe_winners",
    "suggest_trial_parameters",
    "summarize_capacity_manifest",
    "refined_validation_summary",
    "validate_anchor_decision",
]
