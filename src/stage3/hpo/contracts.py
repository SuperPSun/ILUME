"""HPO 排名、锁定清单和只读来源检查。"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean

from common.io import sha256_file
from common.training import canonical_json_sha256


def joint_score(evaluations, tasks, folds):
    tasks, folds = tuple(tasks), tuple(folds)
    if len(tasks) != 24 or len(set(tasks)) != 24 or set(evaluations) != set(folds):
        raise ValueError("HPO requires exactly 24 tasks and the requested fold set")
    values = {task: [] for task in tasks}
    for fold in folds:
        root = Path(evaluations[fold])
        summary = json.loads((root / "summary.json").read_text())
        protocol = summary["reporting"]["protocol"]
        if summary.get("split") != "valid" or protocol.get("fold") != fold or set(summary["tasks"]) != set(tasks):
            raise ValueError("HPO scoring accepts complete validation results only")
        manifests = summary["reporting"]["predictions"]
        if len(manifests) != 24 or {row["task"] for row in manifests} != set(tasks):
            raise ValueError("HPO prediction task roster is incomplete")
        for row in manifests:
            file = root / row["path"]
            if not file.resolve().is_relative_to(root.resolve()) or sha256_file(file) != row["sha256"] or row["rows"] <= 0:
                raise ValueError("HPO prediction integrity mismatch")
            with file.open(newline="") as handle:
                predictions = list(csv.DictReader(handle))
            if (len(predictions) != row["rows"] or summary["tasks"][row["task"]]["count"] != row["rows"]
                    or len({item["source_row"] for item in predictions}) != len(predictions)):
                raise ValueError("HPO prediction row coverage mismatch")
            if any(not math.isfinite(float(item[field])) for item in predictions for field in ("target", "prediction")):
                raise ValueError("Non-finite HPO predictions")
        for task in tasks:
            metric = summary["tasks"][task]["normalized_mae"]
            if not math.isfinite(metric) or metric < 0:
                raise ValueError("Non-finite HPO main objective")
            values[task].append(metric)
    per_task = {task: mean(scores) for task, scores in values.items()}
    return {"score": mean(per_task.values()), "tasks": per_task, "folds": list(folds)}


def rank(rows, folds):
    candidates = [row for row in rows if row.get("status") == "completed" and row.get("folds") == list(folds)]
    ordered = sorted(candidates, key=lambda row: (row["score"], row["parameters_count"], row["gpu_seconds"], row["ordinal"]))
    unique = {}
    for row in ordered:
        unique.setdefault(row.get("model_id", row.get("id", row["ordinal"])), row)
    return list(unique.values())


def validate_lock(path):
    lock = json.loads(Path(path).read_text())
    payload = {key: value for key, value in lock.items() if key != "hash"}
    if lock.get("protocol") != "hpo_v4" or canonical_json_sha256(payload) != lock.get("hash"):
        raise ValueError("HPO lock identity mismatch")
    from stage1.regression import REGRESSION_TASKS
    if set(lock["heads"]) != set(REGRESSION_TASKS) or set(lock["joint"]["folds"]) != {str(fold) for fold in range(1, 6)} or len(lock["heads"]) != 14:
        raise ValueError("HPO lock requires five models and fourteen heads")
    for file, digest in lock["files"].items():
        if Path(file).is_absolute() or ".." in Path(file).parts or sha256_file(file) != digest:
            raise ValueError(f"HPO locked file changed: {file}")
    return lock
