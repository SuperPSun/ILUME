from __future__ import annotations

import csv
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any

from common.io import atomic_json, sha256_file
from stage3.data import sanitize_task

from .config import TransferExperimentConfig


def transfer_gain(baseline_mae: float, transfer_mae: float) -> float:
    if not math.isfinite(baseline_mae) or baseline_mae <= 0:
        raise ValueError("Baseline validation MAE must be finite and positive")
    if not math.isfinite(transfer_mae) or transfer_mae < 0:
        raise ValueError("Transfer validation MAE must be finite and non-negative")
    return (baseline_mae - transfer_mae) / baseline_mae


def _manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing transfer job manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifact = path.parent / str(payload.get("artifact"))
    predictions = path.parent / "validation_predictions.csv"
    if payload.get("artifact_sha256") != sha256_file(artifact):
        raise ValueError(f"Transfer model artifact hash mismatch: {path.parent}")
    if payload.get("predictions_sha256") != sha256_file(predictions):
        raise ValueError(f"Transfer prediction hash mismatch: {path.parent}")
    if payload.get("final_epoch") != 10:
        raise ValueError(f"Transfer job did not publish epoch 10: {path.parent}")
    return payload


def collect_transfer_pairs(
    experiment: TransferExperimentConfig, stage3_root: str | Path
) -> list[dict[str, Any]]:
    root = Path(stage3_root)
    sources = tuple(sorted(experiment.stage2.sources))
    targets = tuple(sorted(experiment.stage3.targets))
    baseline: dict[tuple[str, int], dict[str, Any]] = {}
    for target in targets:
        for fold in experiment.stage3.folds:
            item = _manifest(
                root / "baseline" / sanitize_task(target) / f"fold{fold}" / "manifest.json"
            )
            if item.get("variant") != "baseline" or item.get("task") != target or item.get("fold") != fold:
                raise ValueError("Transfer baseline job identity mismatch")
            baseline[target, fold] = item
    rows: list[dict[str, Any]] = []
    for source in sources:
        for target in targets:
            baseline_values: list[float] = []
            transfer_values: list[float] = []
            gains: list[float] = []
            row: dict[str, Any] = {"source": source, "target": target}
            for fold in experiment.stage3.folds:
                base = baseline[target, fold]
                transfer = _manifest(
                    root / "sources" / sanitize_task(source) / sanitize_task(target) / f"fold{fold}" / "manifest.json"
                )
                if transfer.get("variant") != source or transfer.get("task") != target or transfer.get("fold") != fold:
                    raise ValueError("Transfer source job identity mismatch")
                for key in ("row_target_hash", "initial_state_hash", "permutation_hashes"):
                    if transfer.get(key) != base.get(key):
                        raise ValueError(
                            f"Baseline/transfer {key} mismatch for {source}->{target}/fold{fold}"
                        )
                baseline_mae = float(base["validation_raw_mae"])
                transfer_mae = float(transfer["validation_raw_mae"])
                gain = transfer_gain(baseline_mae, transfer_mae)
                baseline_values.append(baseline_mae)
                transfer_values.append(transfer_mae)
                gains.append(gain)
                row[f"baseline_mae_fold{fold}"] = baseline_mae
                row[f"transfer_mae_fold{fold}"] = transfer_mae
                row[f"TG_fold{fold}"] = gain
            row.update({
                "baseline_mae_mean": statistics.fmean(baseline_values),
                "transfer_mae_mean": statistics.fmean(transfer_values),
                "TG_mean": statistics.fmean(gains),
                "TG_median": statistics.median(gains),
                "positive_folds": sum(value > 0 for value in gains),
            })
            rows.append(row)
    if len(rows) != len(sources) * len(targets):
        raise RuntimeError("Transfer pair table is incomplete")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _heatmap_svg(
    path: Path, sources: tuple[str, ...], targets: tuple[str, ...],
    matrix: dict[tuple[str, str], float],
) -> None:
    cell_w, cell_h, left, top = 92, 28, 260, 260
    width = left + cell_w * len(targets) + 20
    height = top + cell_h * len(sources) + 40
    limit = max((abs(value) for value in matrix.values()), default=1.0) or 1.0

    def color(value: float) -> str:
        strength = min(abs(value) / limit, 1.0)
        channel = round(255 - 150 * strength)
        if value >= 0:
            return f"rgb({channel},255,{channel})"
        return f"rgb(255,{channel},{channel})"

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:10px}.label{font-size:11px}</style>',
    ]
    for column, target in enumerate(targets):
        x = left + column * cell_w + cell_w / 2
        chunks.append(
            f'<text class="label" x="{x}" y="{top - 8}" transform="rotate(-60 {x} {top - 8})" text-anchor="start">{html.escape(target)}</text>'
        )
    for row_index, source in enumerate(sources):
        y = top + row_index * cell_h
        chunks.append(
            f'<text class="label" x="{left - 8}" y="{y + 18}" text-anchor="end">{html.escape(source)}</text>'
        )
        for column, target in enumerate(targets):
            value = matrix[source, target]
            x = left + column * cell_w
            chunks.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color(value)}" stroke="#ddd"/>')
            chunks.append(f'<text x="{x + cell_w / 2}" y="{y + 18}" text-anchor="middle">{value * 100:.1f}%</text>')
    chunks.append("</svg>")
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def summarize_transfer_matrix(
    experiment: TransferExperimentConfig,
    *,
    stage3_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows = collect_transfer_pairs(experiment, stage3_root)
    folds = experiment.stage3.folds
    pair_fields = ["source", "target"]
    for fold in folds:
        pair_fields.extend((f"baseline_mae_fold{fold}", f"transfer_mae_fold{fold}"))
    pair_fields.extend(f"TG_fold{fold}" for fold in folds)
    pair_fields.extend(("baseline_mae_mean", "transfer_mae_mean", "TG_mean", "TG_median", "positive_folds"))
    pairs_path = output / "transfer_gain_pairs.csv"
    _write_csv(pairs_path, rows, pair_fields)
    sources = tuple(sorted(experiment.stage2.sources))
    targets = tuple(sorted(experiment.stage3.targets))
    matrix = {(row["source"], row["target"]): float(row["TG_mean"]) for row in rows}
    matrix_rows = [
        {"source": source, **{target: matrix[source, target] for target in targets}}
        for source in sources
    ]
    matrix_path = output / "transfer_gain_matrix.csv"
    _write_csv(matrix_path, matrix_rows, ["source", *targets])
    heatmap_path = output / "transfer_gain_heatmap.svg"
    _heatmap_svg(heatmap_path, sources, targets, matrix)
    summary = {
        "kind": "ilume_stage2_stage3_transfer_summary",
        "format_version": 1,
        "metric": "validation_raw_mae",
        "tg_definition": "mean((baseline_mae-transfer_mae)/baseline_mae over folds)",
        "sources": list(sources),
        "targets": list(targets),
        "folds": list(folds),
        "pair_count": len(rows),
        "artifacts": {
            path.name: sha256_file(path)
            for path in (pairs_path, matrix_path, heatmap_path)
        },
    }
    atomic_json(output / "summary.json", summary)
    return summary


__all__ = [
    "collect_transfer_pairs", "summarize_transfer_matrix", "transfer_gain",
]
