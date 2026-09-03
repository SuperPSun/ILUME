from __future__ import annotations

import itertools
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from common.io import sha256_file
from .capacity import refined_validation_summary
from .config import (
    BASE_GROUP_TASKS,
    Stage3Config,
    Stage3GroupConfig,
    Stage3TaskConfig,
)


TIER_1 = (
    "experiment/static_relative_permittivity",
    "experiment/dynamic_relative_permittivity",
    "experiment/pec50",
    "experiment/thermal_conductivity",
)
TIER_2 = (
    "experiment/melting_point",
    "experiment/thermal_decomposition_temperature",
)
TIER_3 = (
    "experiment/heat_capacity",
    "experiment/equilibrium_pressure",
    "experiment/speed_of_sound",
    "experiment/glass_transition_temperature",
)
WEAK_TASKS = TIER_1 + TIER_2 + TIER_3
ALL_TASKS = tuple(
    task for tasks in BASE_GROUP_TASKS.values() for task in tasks
)
EVALUATION_WEIGHTS = {
    task: (
        3.0
        if task in TIER_1
        else 2.0
        if task in TIER_2
        else 1.5
        if task in TIER_3
        else 1.0
    )
    for task in ALL_TASKS
}
EVALUATION_WEIGHT_SUM = sum(EVALUATION_WEIGHTS.values())


@dataclass(frozen=True)
class Stage3SearchConfig:
    study_name: str
    base_config: str
    folds: tuple[int, ...]
    epochs: int
    refinement_ratio: float
    max_retries: int
    sampler_seed: int
    top_k: int
    learning_rate: tuple[float, float]
    dropout: tuple[float, float]
    weight_decay: tuple[float, float]
    tier_weight: tuple[float, float]

    def validate(self) -> None:
        if not self.study_name:
            raise ValueError("Stage 3 search study_name cannot be empty")
        if self.base_config != "configs/v2/stage3/base.yaml":
            raise ValueError("Stage 3 search fixes configs/v2/stage3/base.yaml")
        if self.folds != (1, 2):
            raise ValueError("Stage 3 search fixes folds 1/2")
        if self.epochs != 20 or self.refinement_ratio != 0.20:
            raise ValueError("Stage 3 search fixes 20 epochs with 20% refinement")
        if self.max_retries != 1 or self.top_k != 3:
            raise ValueError("Stage 3 search fixes one retry and Top-3 selection")
        for name in ("learning_rate", "dropout", "weight_decay", "tier_weight"):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"Stage 3 search {name} bounds are reversed")
        if self.learning_rate != (1.0e-4, 5.0e-4):
            raise ValueError("Stage 3 search learning-rate range changed")
        if self.dropout != (0.05, 0.20):
            raise ValueError("Stage 3 search dropout range changed")
        if self.weight_decay != (1.0e-3, 3.0e-2):
            raise ValueError("Stage 3 search weight-decay range changed")
        if self.tier_weight != (1.0, 5.0):
            raise ValueError("Stage 3 search tier-weight range changed")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "folds", "learning_rate", "dropout", "weight_decay", "tier_weight"
        ):
            payload[name] = list(payload[name])
        return payload


@dataclass(frozen=True)
class GroupingCandidate:
    candidate_id: str
    source: str
    group_count: int
    reason: str
    assignments: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExpertCandidate:
    candidate_id: str
    source: str
    global_experts: int
    group_experts: int
    private_experts: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_search_config(path: str | Path) -> Stage3SearchConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or raw.pop("schema_version", None) != 1:
        raise ValueError("Stage 3 search requires schema_version: 1")
    allowed = set(Stage3SearchConfig.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError("Unknown Stage 3 search fields: " + ", ".join(sorted(unknown)))
    for name in (
        "folds", "learning_rate", "dropout", "weight_decay", "tier_weight"
    ):
        if name in raw:
            raw[name] = tuple(raw[name])
    config = Stage3SearchConfig(**raw)
    config.validate()
    return config


def _canonical_partition(groups: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    normalized = tuple(sorted(tuple(sorted(group)) for group in groups if group))
    if sorted(task for group in normalized for task in group) != sorted(ALL_TASKS):
        raise ValueError("Stage 3 grouping must assign every task exactly once")
    return normalized


def _candidate(
    candidate_id: str,
    source: str,
    reason: str,
    groups: Sequence[Sequence[str]],
) -> GroupingCandidate:
    partition = _canonical_partition(groups)
    assignments = {
        task: f"group_{index:02d}"
        for index, group in enumerate(partition, start=1)
        for task in group
    }
    return GroupingCandidate(
        candidate_id=candidate_id,
        source=source,
        group_count=len(partition),
        reason=reason,
        assignments=assignments,
    )


def _balanced_partition(order: Sequence[str], count: int) -> list[list[str]]:
    quotient, remainder = divmod(len(order), count)
    result: list[list[str]] = []
    offset = 0
    for index in range(count):
        size = quotient + (1 if index < remainder else 0)
        result.append(list(order[offset : offset + size]))
        offset += size
    return result


def _anchor_groups(count: int) -> list[list[str]]:
    base = {name: list(tasks) for name, tasks in BASE_GROUP_TASKS.items()}
    if count == 2:
        interaction = {
            "experiment/solvation",
            "experiment/transfer",
            "experiment/transfer_organic",
        }
        return [
            [task for task in ALL_TASKS if task not in interaction],
            [task for task in ALL_TASKS if task in interaction],
        ]
    if count == 3:
        return [
            base["transport"] + base["thermophysical"],
            base["phase_stability"]
            + base["dielectric_optical"]
            + base["biological"],
            base["solvation"],
        ]
    if count == 6:
        return list(base.values())
    groups = list(base.values())
    isolated = [
        "experiment/static_relative_permittivity",
        "experiment/dynamic_relative_permittivity",
        "experiment/thermal_conductivity",
    ]
    if count == 12:
        isolated += list(TIER_2) + ["experiment/heat_capacity"]
    for task in isolated:
        for group in groups:
            if task in group:
                group.remove(task)
                groups.append([task])
                break
    return [group for group in groups if group]


def grouping_candidates(seed: int = 42) -> tuple[GroupingCandidate, ...]:
    counts = (2, 3, 6, 9, 12)
    candidates: list[GroupingCandidate] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()

    def add(candidate: GroupingCandidate) -> None:
        partition = _canonical_partition(
            [
                [task for task, group in candidate.assignments.items() if group == group_id]
                for group_id in sorted(set(candidate.assignments.values()))
            ]
        )
        if partition in seen:
            raise ValueError(f"Duplicate Stage 3 grouping: {candidate.candidate_id}")
        seen.add(partition)
        candidates.append(candidate)

    for count in counts:
        add(
            _candidate(
                f"anchor-g{count}",
                "anchor",
                f"G{count} baseline hierarchy anchor",
                _anchor_groups(count),
            )
        )

    base_order = list(ALL_TASKS)
    topology_order = sorted(
        ALL_TASKS,
        key=lambda task: (
            task not in {
                "experiment/solvation",
                "experiment/transfer",
                "experiment/transfer_organic",
            },
            task,
        ),
    )
    weak_order = list(WEAK_TASKS) + [task for task in ALL_TASKS if task not in WEAK_TASKS]
    cross_domain_order = [
        task
        for row in itertools.zip_longest(*BASE_GROUP_TASKS.values())
        for task in row
        if task is not None
    ]
    manual_orders = (
        ("mechanism", base_order, "mechanism-ordered hierarchy"),
        ("topology", topology_order, "input-topology and partner hierarchy"),
        ("weak", weak_order, "weak-property isolation hierarchy"),
        ("cross-domain", cross_domain_order, "cross-domain coupling hierarchy"),
    )
    for design, order, reason in manual_orders:
        for count in counts:
            add(
                _candidate(
                    f"manual-{design}-g{count}",
                    "manual",
                    reason,
                    _balanced_partition(order, count),
                )
            )

    rng = random.Random(seed)
    for count in counts:
        generated = 0
        while generated < 5:
            order = list(ALL_TASKS)
            rng.shuffle(order)
            groups = [[task] for task in order[:count]]
            for task in order[count:]:
                groups[rng.randrange(count)].append(task)
            partition = _canonical_partition(groups)
            if partition in seen:
                continue
            generated += 1
            add(
                _candidate(
                    f"combination-g{count}-{generated:02d}",
                    "combination",
                    f"seed-{seed} constrained G{count} combination",
                    partition,
                )
            )
    if len(candidates) != 50:
        raise AssertionError("Stage 3 Search A must contain 50 candidates")
    return tuple(candidates)


def expert_candidates() -> tuple[ExpertCandidate, ...]:
    local = (
        (2, 2, 1), (1, 2, 1), (2, 1, 1), (2, 3, 1), (2, 2, 0),
        (1, 1, 1), (1, 3, 1), (2, 1, 0), (2, 3, 0), (1, 2, 0),
    )
    ablation = (
        (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1), (0, 3, 0),
        (0, 3, 1), (0, 4, 0), (0, 6, 0), (1, 1, 0), (1, 3, 0),
    )
    capacity = tuple(
        (global_count, group_count, private_count)
        for global_count in (1, 2)
        for group_count in (4, 6)
        for private_count in (0, 1)
    ) + ((0, 4, 1), (0, 6, 1))
    result = []
    for source, values in (
        ("local", local), ("ablation", ablation), ("higher_capacity", capacity)
    ):
        for index, (global_count, group_count, private_count) in enumerate(
            values, start=1
        ):
            result.append(
                ExpertCandidate(
                    candidate_id=f"{source}-{index:02d}",
                    source=source,
                    global_experts=global_count,
                    group_experts=group_count,
                    private_experts=private_count,
                )
            )
    tuples = {
        (item.global_experts, item.group_experts, item.private_experts)
        for item in result
    }
    expected = set(itertools.product((0, 1, 2), (1, 2, 3, 4, 6), (0, 1)))
    if len(result) != 30 or tuples != expected:
        raise AssertionError("Stage 3 Search B must cover all 30 expert tuples")
    return tuple(result)


def config_for_grouping(base: Stage3Config, candidate: GroupingCandidate) -> Stage3Config:
    groups = {
        group: Stage3GroupConfig(enabled=True, group_weight=1.0)
        for group in sorted(set(candidate.assignments.values()))
    }
    tasks = {
        task: replace(spec, meta_group=candidate.assignments[task], task_weight=1.0)
        for task, spec in base.tasks.items()
    }
    config = replace(base, groups=groups, tasks=tasks)
    config.validate()
    return config


def config_for_experts(base: Stage3Config, candidate: ExpertCandidate) -> Stage3Config:
    config = replace(
        base,
        model=replace(
            base.model,
            global_experts=candidate.global_experts,
            group_experts=candidate.group_experts,
            private_experts=candidate.private_experts,
        ),
    )
    config.validate()
    return config


def config_for_search_c(
    base: Stage3Config,
    grouping: GroupingCandidate,
    experts: ExpertCandidate,
    parameters: Mapping[str, float],
) -> Stage3Config:
    config = config_for_experts(config_for_grouping(base, grouping), experts)
    tier_weights = {
        task: float(parameters[f"tier{tier}_weight"])
        for tier, tasks in enumerate((TIER_1, TIER_2, TIER_3), start=1)
        for task in tasks
    }
    tasks: dict[str, Stage3TaskConfig] = {
        task: replace(spec, task_weight=tier_weights.get(task, 1.0))
        for task, spec in config.tasks.items()
    }
    config = replace(
        config,
        tasks=tasks,
        model=replace(config.model, dropout=float(parameters["dropout"])),
        training=replace(
            config.training,
            learning_rate=float(parameters["learning_rate"]),
            weight_decay=float(parameters["weight_decay"]),
        ),
    )
    config.validate()
    return config


def search_base_config(base: Stage3Config, spec: Stage3SearchConfig) -> Stage3Config:
    config = replace(
        base,
        training=replace(
            base.training,
            seed=spec.sampler_seed,
            epochs=spec.epochs,
            refinement_ratio=spec.refinement_ratio,
        ),
    )
    config.validate()
    return config


def fold_search_summary(run_root: str | Path, *, expected_epochs: int) -> dict[str, Any]:
    refined = refined_validation_summary(run_root, expected_epochs=expected_epochs)
    task_scores = refined["task_scores"]
    if set(task_scores) != set(ALL_TASKS):
        raise ValueError("Stage 3 search fold does not contain all 21 tasks")
    weighted = sum(
        task_scores[task] * EVALUATION_WEIGHTS[task] for task in ALL_TASKS
    ) / EVALUATION_WEIGHT_SUM
    summary_path = Path(run_root) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cost = summary.get("training_cost")
    if not isinstance(cost, dict):
        raise ValueError("Stage 3 search fold lacks training cost")
    return {
        **refined,
        "weighted_normalized_mae": weighted,
        "original_macro_task_score": refined["score"],
        "weak_task_scores": {task: task_scores[task] for task in WEAK_TASKS},
        "training_cost": cost,
    }


def aggregate_search_trial(
    folds: Mapping[int, Mapping[str, Any]],
    *,
    trial_number: int,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    if set(folds) != {1, 2}:
        raise ValueError("Stage 3 search trial requires folds 1/2")
    fold_scores = [float(folds[fold]["weighted_normalized_mae"]) for fold in (1, 2)]
    macros = [float(folds[fold]["original_macro_task_score"]) for fold in (1, 2)]
    costs = [folds[fold]["training_cost"] for fold in (1, 2)]
    return {
        "trial_number": trial_number,
        "candidate": dict(candidate),
        "score": statistics.fmean(fold_scores),
        "fold_sample_sd": statistics.stdev(fold_scores),
        "original_macro_task_score": statistics.fmean(macros),
        "weak_task_scores": {
            task: statistics.fmean(
                float(folds[fold]["weak_task_scores"][task]) for fold in (1, 2)
            )
            for task in WEAK_TASKS
        },
        "training_cost": {
            "wall_seconds": sum(float(value["wall_seconds"]) for value in costs),
            "gpu_seconds": sum(float(value["gpu_seconds"]) for value in costs),
            "peak_allocated_bytes": max(
                int(value["peak_allocated_bytes"]) for value in costs
            ),
            "total_parameters": int(costs[0]["total_parameters"]),
            "trainable_parameters": int(costs[0]["trainable_parameters"]),
        },
        "folds": {str(fold): dict(folds[fold]) for fold in (1, 2)},
    }


def rank_trials(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    valid = []
    for row in rows:
        for name in ("score", "original_macro_task_score"):
            if not math.isfinite(float(row[name])):
                raise ValueError(f"Stage 3 search trial has non-finite {name}")
        valid.append(dict(row))
    return sorted(
        valid,
        key=lambda row: (
            float(row["score"]),
            float(row["original_macro_task_score"]),
            float(row["training_cost"]["gpu_seconds"]),
            int(row["trial_number"]),
        ),
    )


def result_sha256(path: str | Path) -> str:
    return sha256_file(path)


__all__ = [
    "ALL_TASKS",
    "EVALUATION_WEIGHTS",
    "EVALUATION_WEIGHT_SUM",
    "ExpertCandidate",
    "GroupingCandidate",
    "Stage3SearchConfig",
    "TIER_1",
    "TIER_2",
    "TIER_3",
    "WEAK_TASKS",
    "aggregate_search_trial",
    "config_for_experts",
    "config_for_grouping",
    "config_for_search_c",
    "expert_candidates",
    "fold_search_summary",
    "grouping_candidates",
    "load_search_config",
    "rank_trials",
    "result_sha256",
    "search_base_config",
]
