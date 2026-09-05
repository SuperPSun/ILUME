from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape

from common.io import sha256_file
from .search import (
    ALL_TASKS,
    EVALUATION_WEIGHTS,
    EVALUATION_WEIGHT_SUM,
    TIER_1,
    TIER_2,
    TIER_3,
    rank_trials,
)


SEARCH_REPORT_SCHEMA_VERSION = 1
PHASES = ("A", "B", "C")
PHASE_BUDGETS = {"A": 50, "B": 30, "C": 20}
PHASE_DIRECTORIES = {phase: f"search_{phase.lower()}" for phase in PHASES}
TASK_ORDER = TIER_1 + TIER_2 + TIER_3 + tuple(
    task for task in ALL_TASKS if task not in TIER_1 + TIER_2 + TIER_3
)
REPORT_ARTIFACTS = (
    "overview.md",
    "trial_rankings.csv",
    "task_metrics.csv",
    "search_a_property_matrix.svg",
    "search_b_property_matrix.svg",
    "search_c_property_matrix.svg",
    "property_radar.svg",
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Stage 3 search report has invalid {label}") from error
    if not math.isfinite(number):
        raise ValueError(f"Stage 3 search report has non-finite {label}")
    return number


def _close(actual: Any, expected: float, label: str) -> None:
    if not math.isclose(_finite(actual, label), expected, rel_tol=1.0e-10, abs_tol=1.0e-12):
        raise ValueError(f"Stage 3 search report has inconsistent {label}")


def _candidate_id(row: Mapping[str, Any]) -> str:
    candidate = row.get("candidate")
    if not isinstance(candidate, dict) or not isinstance(candidate.get("candidate_id"), str):
        raise ValueError("Stage 3 search result has no candidate_id")
    return candidate["candidate_id"]


def _validate_candidate(
    row: Mapping[str, Any], candidates: Mapping[str, Mapping[str, Any]], phase: str
) -> None:
    candidate = row["candidate"]
    candidate_id = _candidate_id(row)
    if candidate_id not in candidates:
        raise ValueError(f"Search {phase} result contains unknown candidate {candidate_id}")
    stored = {key: value for key, value in candidate.items() if key != "parameters"}
    if stored != candidates[candidate_id]:
        raise ValueError(f"Search {phase} candidate {candidate_id} changed")
    parameters = candidate.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("candidate_id") != candidate_id:
        raise ValueError(f"Search {phase} candidate {candidate_id} lacks trial parameters")


def _task_values(row: Mapping[str, Any], phase: str) -> dict[str, dict[str, float]]:
    folds = row.get("folds")
    if not isinstance(folds, dict) or set(folds) != {"1", "2"}:
        raise ValueError(f"Search {phase} trial {_candidate_id(row)} requires folds 1/2")
    fold_scores: dict[str, dict[str, float]] = {}
    weighted_scores: list[float] = []
    macro_scores: list[float] = []
    for fold in ("1", "2"):
        summary = folds[fold]
        task_scores = summary.get("task_scores") if isinstance(summary, dict) else None
        if not isinstance(task_scores, dict) or set(task_scores) != set(ALL_TASKS):
            raise ValueError(
                f"Search {phase} trial {_candidate_id(row)} fold{fold} does not cover all 21 tasks"
            )
        values = {
            task: _finite(task_scores[task], f"Search {phase} fold{fold} {task}")
            for task in TASK_ORDER
        }
        if any(value < 0.0 for value in values.values()):
            raise ValueError(f"Search {phase} trial {_candidate_id(row)} has negative normalized MAE")
        weighted = sum(values[task] * EVALUATION_WEIGHTS[task] for task in ALL_TASKS) / EVALUATION_WEIGHT_SUM
        _close(summary.get("weighted_normalized_mae"), weighted, f"Search {phase} fold{fold} weighted score")
        weighted_scores.append(weighted)
        macro_scores.append(_finite(summary.get("original_macro_task_score"), f"Search {phase} fold{fold} macro score"))
        fold_scores[fold] = values
    _close(row.get("score"), statistics.fmean(weighted_scores), f"Search {phase} trial score")
    _close(row.get("fold_sample_sd"), statistics.stdev(weighted_scores), f"Search {phase} fold sample SD")
    _close(row.get("original_macro_task_score"), statistics.fmean(macro_scores), f"Search {phase} macro score")
    return {
        task: {
            "fold1": fold_scores["1"][task],
            "fold2": fold_scores["2"][task],
            "mean": statistics.fmean((fold_scores["1"][task], fold_scores["2"][task])),
            "sample_sd": statistics.stdev((fold_scores["1"][task], fold_scores["2"][task])),
        }
        for task in TASK_ORDER
    }


def _validate_phase(
    phase: str,
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expected_prerequisites: Mapping[str, str],
) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != 1 or manifest.get("phase") != phase:
        raise ValueError(f"Search {phase} candidate manifest is incompatible")
    if manifest.get("prerequisites") != dict(expected_prerequisites):
        raise ValueError(f"Search {phase} prerequisite hashes do not match current results")
    raw_candidates = manifest.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError(f"Search {phase} candidate manifest has no candidates")
    expected_candidates = {"A": 50, "B": 30, "C": 9}[phase]
    candidates = {
        str(candidate.get("candidate_id")): candidate
        for candidate in raw_candidates
        if isinstance(candidate, dict) and isinstance(candidate.get("candidate_id"), str)
    }
    if len(raw_candidates) != expected_candidates or len(candidates) != expected_candidates:
        raise ValueError(f"Search {phase} candidate count changed")
    budget = PHASE_BUDGETS[phase]
    if result.get("schema_version") != 1 or result.get("phase") != phase:
        raise ValueError(f"Search {phase} result is incompatible")
    if result.get("primary_metric") != "weak_property_weighted_validation_normalized_mae":
        raise ValueError(f"Search {phase} primary metric changed")
    if result.get("attempted_trials") != budget:
        raise ValueError(f"Search {phase} attempted-trial budget changed")
    completed = result.get("completed_trials")
    failed = result.get("failed_trials")
    if not isinstance(completed, int) or not isinstance(failed, int) or completed + failed != budget:
        raise ValueError(f"Search {phase} completion counts are inconsistent")
    ranking = result.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != completed or completed < 3:
        raise ValueError(f"Search {phase} ranking is incomplete")
    if result.get("top3") != ranking[:3]:
        raise ValueError(f"Search {phase} Top-3 does not match its ranking")
    if rank_trials(ranking) != ranking:
        raise ValueError(f"Search {phase} ranking order is inconsistent")
    if phase == "A" and result.get("top3_groupings") != [
        row["candidate"] for row in ranking[:3]
    ]:
        raise ValueError("Search A grouping Top-3 does not match its ranking")
    if phase == "B" and result.get("top3_expert_structures") != [
        row["candidate"]["experts"] for row in ranking[:3]
    ]:
        raise ValueError("Search B expert Top-3 does not match its ranking")
    if phase == "C" and (result.get("winner") != ranking[0] or result.get("top3_recipes") != ranking[:3]):
        raise ValueError("Search C winner or Top-3 recipes do not match its ranking")
    trial_numbers: set[int] = set()
    rows: list[dict[str, Any]] = []
    for rank, source in enumerate(ranking, start=1):
        if not isinstance(source, dict) or not isinstance(source.get("trial_number"), int):
            raise ValueError(f"Search {phase} ranking contains a malformed trial")
        trial_number = source["trial_number"]
        if trial_number in trial_numbers:
            raise ValueError(f"Search {phase} ranking repeats trial {trial_number}")
        trial_numbers.add(trial_number)
        _validate_candidate(source, candidates, phase)
        task_metrics = _task_values(source, phase)
        costs = source.get("training_cost")
        required_costs = (
            "wall_seconds", "gpu_seconds", "peak_allocated_bytes",
            "total_parameters", "trainable_parameters",
        )
        if not isinstance(costs, dict) or any(name not in costs for name in required_costs):
            raise ValueError(f"Search {phase} trial {trial_number} lacks training cost")
        for name in required_costs:
            value = _finite(costs[name], f"Search {phase} trial {trial_number} {name}")
            if value < 0.0:
                raise ValueError(f"Search {phase} trial {trial_number} has negative {name}")
        rows.append({**source, "phase": phase, "rank": rank, "task_metrics": task_metrics})
    return rows


def build_search_report(search_root: str | Path) -> dict[str, Any]:
    root = Path(search_root)
    result_paths = {
        phase: root / PHASE_DIRECTORIES[phase] / "result.json" for phase in PHASES
    }
    manifest_paths = {
        phase: root / PHASE_DIRECTORIES[phase] / "candidate_manifest.json" for phase in PHASES
    }
    for path in (*result_paths.values(), *manifest_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(f"Stage 3 search report input is missing: {path}")
    results = {phase: _json(path) for phase, path in result_paths.items()}
    result_hashes = {phase: sha256_file(path) for phase, path in result_paths.items()}
    manifests = {phase: _json(path) for phase, path in manifest_paths.items()}
    rows_by_phase: dict[str, list[dict[str, Any]]] = {}
    for phase in PHASES:
        prerequisites = {}
        if phase in {"B", "C"}:
            prerequisites["search_a"] = result_hashes["A"]
        if phase == "C":
            prerequisites["search_b"] = result_hashes["B"]
        rows_by_phase[phase] = _validate_phase(
            phase, results[phase], manifests[phase], prerequisites
        )
    all_rows = [row for phase in PHASES for row in rows_by_phase[phase]]
    global_best = {
        task: min(row["task_metrics"][task]["mean"] for row in all_rows)
        for task in TASK_ORDER
    }
    base = next(
        (row for row in rows_by_phase["A"] if _candidate_id(row) == "anchor-g6"),
        None,
    )
    if base is None:
        raise ValueError("Search A ranking does not contain the anchor-g6 Base reference")
    radar_rows = (
        ("Base anchor-g6", base),
        ("Search A winner", rows_by_phase["A"][0]),
        ("Search B winner", rows_by_phase["B"][0]),
        ("Search C winner", rows_by_phase["C"][0]),
    )
    radar = [
        {
            "label": label,
            "phase": row["phase"],
            "trial_number": row["trial_number"],
            "candidate_id": _candidate_id(row),
            "scores": {
                task: _relative_score(row["task_metrics"][task]["mean"], global_best[task])
                for task in TASK_ORDER
            },
        }
        for label, row in radar_rows
    ]
    return {
        "schema_version": SEARCH_REPORT_SCHEMA_VERSION,
        "metric": {
            "property": "fold1/2 mean taskwise-refined validation normalized_mae",
            "primary": "weak_property_weighted_validation_normalized_mae",
            "relative_score": "search-wide property best / trial property normalized_mae",
        },
        "source_sha256": {
            PHASE_DIRECTORIES[phase]: {
                "result.json": result_hashes[phase],
                "candidate_manifest.json": sha256_file(manifest_paths[phase]),
            }
            for phase in PHASES
        },
        "task_order": list(TASK_ORDER),
        "global_property_best": global_best,
        "phases": {
            phase: {
                "attempted_trials": results[phase]["attempted_trials"],
                "completed_trials": results[phase]["completed_trials"],
                "failed_trials": results[phase]["failed_trials"],
                "top3": rows_by_phase[phase][:3],
                "winner": rows_by_phase[phase][0],
            }
            for phase in PHASES
        },
        "trials": all_rows,
        "radar": radar,
    }


def _relative_score(value: float, best: float) -> float:
    if best == 0.0:
        return 1.0 if value == 0.0 else 0.0
    return best / value


def _grouping(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    grouping = candidate.get("grouping")
    return grouping if isinstance(grouping, dict) else candidate


def _experts(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    experts = candidate.get("experts")
    return experts if isinstance(experts, dict) else {}


def _parameters(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    parameters = candidate.get("parameters")
    return parameters if isinstance(parameters, dict) else {}


def _recipe_label(row: Mapping[str, Any]) -> str:
    candidate = row["candidate"]
    grouping = _grouping(candidate)
    experts = _experts(candidate)
    parameters = _parameters(candidate)
    parts = [f"#{row['rank']} t{row['trial_number']:03d}", str(candidate["candidate_id"])]
    grouping_id = grouping.get("candidate_id")
    if grouping_id and grouping_id != candidate["candidate_id"]:
        parts.append(f"grp={grouping_id}")
    if experts:
        parts.append(
            "E=" + "/".join(
                str(experts[name])
                for name in ("global_experts", "group_experts", "private_experts")
            )
        )
    if row["phase"] == "C":
        parts.extend((
            f"w={parameters['tier1_weight']:.3g}/{parameters['tier2_weight']:.3g}/{parameters['tier3_weight']:.3g}",
            f"lr={parameters['learning_rate']:.2g}",
            f"do={parameters['dropout']:.3g}",
            f"wd={parameters['weight_decay']:.2g}",
        ))
    elif grouping.get("group_count") is not None:
        parts.append(f"G={grouping['group_count']}")
    return " | ".join(parts)


def _task_label(task: str) -> str:
    return task.rsplit("/", 1)[-1].replace("_", " ")


def _heat_color(score: float) -> str:
    score = min(1.0, max(0.0, score))
    start = (245, 248, 250)
    end = (0, 114, 178)
    return "#" + "".join(
        f"{round(left + (right - left) * score):02x}"
        for left, right in zip(start, end, strict=True)
    )


def _matrix_svg(phase: str, rows: Sequence[Mapping[str, Any]], best: Mapping[str, float]) -> str:
    label_width = 610
    cell_width = 82
    row_height = 29
    header_height = 300
    width = label_width + cell_width * len(TASK_ORDER) + 30
    height = header_height + row_height * len(rows) + 70
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-family="sans-serif" font-size="24" font-weight="bold">Search {phase} property matrix</text>',
        f'<text x="{width / 2:.1f}" y="60" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#444">Cell text: fold1/2 mean normalized MAE (lower is better). Blue intensity: search-wide property best / value (higher is better).</text>',
        '<text x="20" y="92" font-family="sans-serif" font-size="12" fill="#555">Rows follow the official phase ranking; complete grouping assignments are retained in CSV/JSON.</text>',
    ]
    tier_ranges = (
        ("Tier 1", 0, len(TIER_1)),
        ("Tier 2", len(TIER_1), len(TIER_1) + len(TIER_2)),
        ("Tier 3", len(TIER_1) + len(TIER_2), len(TIER_1) + len(TIER_2) + len(TIER_3)),
        ("Other", len(TIER_1) + len(TIER_2) + len(TIER_3), len(TASK_ORDER)),
    )
    for label, start, stop in tier_ranges:
        x1 = label_width + start * cell_width
        x2 = label_width + stop * cell_width
        lines.extend((
            f'<line x1="{x1}" y1="112" x2="{x2}" y2="112" stroke="#555" stroke-width="2"/>',
            f'<text x="{(x1 + x2) / 2:.1f}" y="106" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="bold">{label}</text>',
        ))
    for index, task in enumerate(TASK_ORDER):
        x = label_width + index * cell_width + cell_width / 2
        lines.append(
            f'<text x="{x:.1f}" y="282" transform="rotate(-58 {x:.1f} 282)" text-anchor="start" font-family="sans-serif" font-size="10">{escape(_task_label(task))}</text>'
        )
    for row_index, row in enumerate(rows):
        y = header_height + row_index * row_height
        fill = "#f7f7f7" if row_index % 2 else "#ffffff"
        lines.append(f'<rect x="0" y="{y}" width="{width}" height="{row_height}" fill="{fill}"/>')
        label = _recipe_label(row)
        lines.append(
            f'<text x="{label_width - 10}" y="{y + 19}" text-anchor="end" font-family="monospace" font-size="10">{escape(label)}</text>'
        )
        for task_index, task in enumerate(TASK_ORDER):
            value = row["task_metrics"][task]["mean"]
            score = _relative_score(value, best[task])
            x = label_width + task_index * cell_width
            text_color = "white" if score >= 0.72 else "#222"
            tooltip = f"{label}; {task}; normalized MAE={value:.8g}; relative score={score:.8g}"
            lines.extend((
                f'<rect class="metric" data-task="{escape(task)}" data-normalized-mae="{value:.12g}" data-relative-score="{score:.12g}" x="{x}" y="{y}" width="{cell_width}" height="{row_height}" fill="{_heat_color(score)}" stroke="#d9d9d9" stroke-width="0.5"><title>{escape(tooltip)}</title></rect>',
                f'<text x="{x + cell_width / 2:.1f}" y="{y + 19}" text-anchor="middle" font-family="monospace" font-size="10" fill="{text_color}" pointer-events="none">{value:.4f}</text>',
            ))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _radar_point(cx: float, cy: float, radius: float, angle: float) -> tuple[float, float]:
    return cx + radius * math.sin(angle), cy - radius * math.cos(angle)


def _radar_svg(payload: Mapping[str, Any]) -> str:
    width, height = 1800, 1500
    cx, cy, radius = 900.0, 790.0, 520.0
    angles = tuple(2.0 * math.pi * index / len(TASK_ORDER) for index in range(len(TASK_ORDER)))
    palette = ("#0072b2", "#d55e00", "#009e73", "#cc79a7")
    dashes = ("", "10 4", "3 3", "12 4 3 4")
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<text x="900" y="38" text-anchor="middle" font-family="sans-serif" font-size="26" font-weight="bold">Stage 3 search property comparison</text>',
        '<text x="900" y="68" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#444">Score = search-wide best normalized MAE / recipe normalized MAE; higher is better. Validation folds 1/2 only.</text>',
    ]
    for index, series in enumerate(payload["radar"]):
        y = 104 + index * 23
        color = palette[index]
        dash = f' stroke-dasharray="{dashes[index]}"' if dashes[index] else ""
        lines.extend((
            f'<line x1="610" y1="{y}" x2="650" y2="{y}" stroke="{color}" stroke-width="3"{dash}/>',
            f'<text x="662" y="{y + 4}" font-family="sans-serif" font-size="13">{escape(series["label"])} (trial {series["trial_number"]}, {escape(series["candidate_id"])})</text>',
        ))
    for fraction in (0.25, 0.5, 0.75, 1.0):
        points = " ".join(
            f"{x:.2f},{y:.2f}"
            for x, y in (_radar_point(cx, cy, radius * fraction, angle) for angle in angles)
        )
        lines.extend((
            f'<polygon points="{points}" fill="none" stroke="#c7c7c7" stroke-width="1"/>',
            f'<text x="{cx + 5:.1f}" y="{cy - radius * fraction + 4:.1f}" font-family="monospace" font-size="10" fill="#666">{fraction:g}</text>',
        ))
    for task, angle in zip(TASK_ORDER, angles, strict=True):
        x, y = _radar_point(cx, cy, radius, angle)
        lx, ly = _radar_point(cx, cy, radius + 58.0, angle)
        anchor = "middle" if abs(math.sin(angle)) < 0.15 else ("start" if math.sin(angle) > 0 else "end")
        lines.extend((
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#c7c7c7" stroke-width="1"/>',
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-family="sans-serif" font-size="11">{escape(_task_label(task))}</text>',
        ))
    for index, series in enumerate(payload["radar"]):
        scores = [series["scores"][task] for task in TASK_ORDER]
        points = " ".join(
            f"{x:.2f},{y:.2f}"
            for x, y in (
                _radar_point(cx, cy, radius * score, angle)
                for score, angle in zip(scores, angles, strict=True)
            )
        )
        score_text = ",".join(f"{score:.8g}" for score in scores)
        dash = f' stroke-dasharray="{dashes[index]}"' if dashes[index] else ""
        lines.append(
            f'<polygon class="series" data-label="{escape(series["label"])}" data-scores="{score_text}" points="{points}" fill="{palette[index]}" fill-opacity="0.07" stroke="{palette[index]}" stroke-width="3"{dash}/>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _trial_record(row: Mapping[str, Any]) -> dict[str, Any]:
    candidate = row["candidate"]
    grouping = _grouping(candidate)
    experts = _experts(candidate)
    parameters = _parameters(candidate)
    cost = row["training_cost"]
    return {
        "phase": row["phase"],
        "rank": row["rank"],
        "trial_number": row["trial_number"],
        "candidate_id": candidate["candidate_id"],
        "source": candidate.get("source", ""),
        "recipe": _recipe_label(row),
        "grouping_id": grouping.get("candidate_id", ""),
        "group_count": grouping.get("group_count", ""),
        "grouping_assignments_json": json.dumps(grouping.get("assignments", {}), sort_keys=True),
        "global_experts": experts.get("global_experts", ""),
        "group_experts": experts.get("group_experts", ""),
        "private_experts": experts.get("private_experts", ""),
        "tier1_weight": parameters.get("tier1_weight", ""),
        "tier2_weight": parameters.get("tier2_weight", ""),
        "tier3_weight": parameters.get("tier3_weight", ""),
        "learning_rate": parameters.get("learning_rate", ""),
        "dropout": parameters.get("dropout", ""),
        "weight_decay": parameters.get("weight_decay", ""),
        "weighted_normalized_mae": row["score"],
        "fold_sample_sd": row["fold_sample_sd"],
        "macro_task_normalized_mae": row["original_macro_task_score"],
        "wall_seconds": cost["wall_seconds"],
        "gpu_seconds": cost["gpu_seconds"],
        "peak_allocated_bytes": cost["peak_allocated_bytes"],
        "total_parameters": cost["total_parameters"],
        "trainable_parameters": cost["trainable_parameters"],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty report table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _overview(payload: Mapping[str, Any]) -> str:
    c_winner = payload["phases"]["C"]["winner"]
    base = next(series for series in payload["radar"] if series["label"] == "Base anchor-g6")
    improved = sum(
        c_winner["task_metrics"][task]["mean"]
        < next(
            row["task_metrics"][task]["mean"]
            for row in payload["trials"]
            if row["phase"] == "A" and row["trial_number"] == base["trial_number"]
        )
        for task in TASK_ORDER
    )
    lines = [
        "# ILUME v2 Stage3 search report",
        "",
        "## Technical summary",
        "",
        f"Search C selected `{_candidate_id(c_winner)}` (trial {c_winner['trial_number']}) with weighted validation normalized MAE `{c_winner['score']:.6g}`. It improves {improved}/21 per-property normalized MAE values relative to the Search A `anchor-g6` Base reference.",
        "",
        "This is a descriptive two-fold, 20-epoch search report. It does not constitute five-fold confirmation or test evaluation.",
        "",
        "## Stage winners and search health",
        "",
        "| Phase | Completed | Failed | Winner | Primary score | Macro score | Fold SD | GPU seconds |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for phase in PHASES:
        stage = payload["phases"][phase]
        winner = stage["winner"]
        lines.append(
            f"| {phase} | {stage['completed_trials']} | {stage['failed_trials']} | trial {winner['trial_number']} / `{_candidate_id(winner)}` | {winner['score']:.6g} | {winner['original_macro_task_score']:.6g} | {winner['fold_sample_sd']:.6g} | {winner['training_cost']['gpu_seconds']:.6g} |"
        )
    lines.extend((
        "",
        "Each stage retained its fixed attempted-trial budget. Failed trials are reported above but are not assigned fabricated property metrics.",
        "",
        "## Property-level evidence",
        "",
        "The three matrices show every successful trial in official rank order. Cell text is the fold1/2 mean normalized MAE; blue intensity is the search-wide per-property best divided by that value.",
        "",
        "- [Search A property matrix](search_a_property_matrix.svg)",
        "- [Search B property matrix](search_b_property_matrix.svg)",
        "- [Search C property matrix](search_c_property_matrix.svg)",
        "- [Base and stage-winner radar](property_radar.svg)",
        "",
        "Exact lookup and audit data are available in [trial rankings](trial_rankings.csv) and [task metrics](task_metrics.csv).",
        "",
        "## Metric and method definitions",
        "",
        "- Property metric: arithmetic mean of fold1/2 taskwise-refined stitched-validation normalized MAE.",
        "- Primary ranking metric: fixed evaluation-tier weighted normalized MAE; evaluation weights sum to 33 and are separate from training task weights.",
        "- Fold SD: sample standard deviation across fold1/2.",
        "- Radar and matrix color score: search-wide per-property best normalized MAE divided by the displayed recipe value; higher is better.",
        "",
        "## Limitations and robustness boundary",
        "",
        "Only folds 1/2 and the fixed 20-epoch screening schedule are represented. The visual ratios are descriptive and do not estimate statistical significance. Per-property winners may differ from the recipe selected by the fixed aggregate ranking metric.",
        "",
        "## Recommended next step",
        "",
        "Freeze the Search C winner as the Base recipe, then run the separately authorized multi-fold confirmation or cross-scale study without modifying this 100-trial record.",
        "",
        "## Further question",
        "",
        "The main unresolved question is whether the Search C winner's property-level gains persist on folds 3/4/5 and the untouched test split.",
        "",
    ))
    return "\n".join(lines)


def write_search_report(payload: Mapping[str, Any], output: str | Path) -> None:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    trials = [_trial_record(row) for row in payload["trials"]]
    task_rows = []
    for row in payload["trials"]:
        for task in TASK_ORDER:
            metric = row["task_metrics"][task]
            tier = 1 if task in TIER_1 else 2 if task in TIER_2 else 3 if task in TIER_3 else "other"
            task_rows.append({
                "phase": row["phase"],
                "rank": row["rank"],
                "trial_number": row["trial_number"],
                "candidate_id": _candidate_id(row),
                "task": task,
                "tier": tier,
                "evaluation_weight": EVALUATION_WEIGHTS[task],
                "fold1_normalized_mae": metric["fold1"],
                "fold2_normalized_mae": metric["fold2"],
                "mean_normalized_mae": metric["mean"],
                "fold_sample_sd": metric["sample_sd"],
            })
    _write_csv(destination / "trial_rankings.csv", trials)
    _write_csv(destination / "task_metrics.csv", task_rows)
    best = payload["global_property_best"]
    for phase in PHASES:
        rows = [row for row in payload["trials"] if row["phase"] == phase]
        (destination / f"search_{phase.lower()}_property_matrix.svg").write_text(
            _matrix_svg(phase, rows, best), encoding="utf-8"
        )
    (destination / "property_radar.svg").write_text(_radar_svg(payload), encoding="utf-8")
    (destination / "overview.md").write_text(_overview(payload), encoding="utf-8")


__all__ = [
    "REPORT_ARTIFACTS",
    "SEARCH_REPORT_SCHEMA_VERSION",
    "TASK_ORDER",
    "build_search_report",
    "write_search_report",
]
