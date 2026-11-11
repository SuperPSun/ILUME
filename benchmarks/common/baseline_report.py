"""Assemble audited baseline sources into a complete, independent 22-task report."""
from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
import tempfile
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from common.identity import semantic_identity, validate_semantic_identity
from common.io import sha256_file
from common.reporting import comparison_identity, sanitize_task_id
from stage3.config import load_stage3_config
from stage3.data import canonicalize_smiles, resolve_task_registry
from .config import (CATALOG_V4_SIMULATION_TASKS, FLFT_22_COMPLETION_PROTOCOL,
                     FLFT_22_COMPLETION_TASKS, FLFT_22_PROTOCOL, FLFT_22_RETRAIN_TASKS,
                     XGBOOST_LIGHTWEIGHT_PROTOCOL, load_benchmark_config)
from .data import load_split, resolve_task
from .features import feature_schema
from .paper import BASELINE_MODELS, _metrics


PROTOCOL = "flft_v4_3407_22_report"
METRICS = ("mae", "rmse", "r2", "normalized_mae", "normalized_rmse")


@lru_cache(maxsize=65536)
def _canonical(smiles: str) -> str:
    return canonicalize_smiles(smiles, "Baseline prediction")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _equal(actual: Any, expected: Any, label: str) -> None:
    if json.dumps(actual, sort_keys=True, default=str) != json.dumps(expected, sort_keys=True, default=str):
        raise ValueError("Baseline report mismatch: " + label)


class SourceAudit:
    def __init__(self, repository: Path, baseline: Path):
        self.repository, self.baseline = repository.resolve(), baseline.resolve()
        if not self.baseline.is_relative_to(self.repository) or not self.baseline.is_dir():
            raise ValueError("Baseline root must be an existing repository directory")
        self.files: dict[str, dict[str, Any]] = {}
        self.moves: dict[str, dict[str, Any]] = {}

    def path(self, name: str) -> Path:
        p = (self.repository / name).resolve()
        if Path(name).is_absolute() or not p.is_relative_to(self.repository):
            raise ValueError("Source path escapes the repository: " + name)
        return p

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.repository).as_posix()

    def fingerprint(self, path: Path) -> dict[str, Any]:
        name = self.relative(path)
        if name not in self.files:
            self.files[name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
        return self.files[name]

    def check(self, path: Path, expected: dict[str, Any]) -> None:
        _equal(self.fingerprint(path), {k: expected[k] for k in ("size", "sha256")}, self.relative(path))

    def migrated(self, name: str) -> Path:
        self.path(name)
        if name in self.moves:
            entry = self.moves[name]
            p = self.path(entry["destination"])
            self.check(p, entry)
            return p
        old = "outputs/benchmarks/v4_3407/"
        return self.baseline / name[len(old):] if name.startswith(old) else self.path(name)


def _receipts(settings: dict[str, Any], audit: SourceAudit) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    records: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for entry in settings["source_audits"]:
        path = audit.path(entry["path"])
        _equal(audit.fingerprint(path)["sha256"], entry["sha256"], "source audit SHA")
        payload = _json(path)
        for record in payload if isinstance(payload, list) else payload["records"]:
            if record.get("errors") or record.get("status", "passed") != "passed":
                continue
            key = (record["model"], record["domain"], record["task"])
            records.setdefault(key, []).append(record)
    for entry in settings["path_maps"]:
        path = audit.path(entry["path"])
        _equal(audit.fingerprint(path)["sha256"], entry["sha256"], "path map SHA")
        for move in _json(path)["moves"]:
            audit.path(move["source"])
            dest = audit.path(move["destination"])
            if not dest.is_relative_to(audit.baseline) or move["source"] in audit.moves:
                raise ValueError("Invalid or ambiguous artifact relocation")
            audit.moves[move["source"]] = move
    return records


def _task_config(repository: Path, model: str, domain: str, task: str, authority: Path):
    completed = model in {"mlp", "ecfp_xgboost", "ilbert"} and task not in FLFT_22_RETRAIN_TASKS
    changed = domain == "stage3" and task in FLFT_22_RETRAIN_TASKS
    if model == "ecfp_xgboost":
        path = Path("configs/benchmarks/FLftv4_3407/lightweight/ecfp_xgboost_500_d4.yaml")
    elif completed:
        path = Path(f"configs/benchmarks/FLftv4_3407/completion/{model}.yaml")
    elif changed:
        name = "ilbert_condition_zscore" if model == "ilbert" else model
        path = Path(f"configs/benchmarks/FLftv4_3407/{name}.yaml")
    else:
        path = Path(f"configs/benchmarks/catalog_v4_3407/{model}.yaml")
    config = load_benchmark_config(repository / path)
    config = replace(config, data=replace(config.data, stage3_authority_config=authority))
    if domain == "stage3":
        config = replace(config, stage3=replace(config.stage3, tasks=(task,)))
    protocol = (XGBOOST_LIGHTWEIGHT_PROTOCOL if model == "ecfp_xgboost" else
                FLFT_22_COMPLETION_PROTOCOL if completed else FLFT_22_PROTOCOL if changed else "catalog_v4_3407")
    return config, path, protocol


def _training(config, task, run: Path, expected_protocol: str, receipt: dict[str, Any], audit: SourceAudit):
    manifest = _json(run / "checkpoint.json")
    identity = manifest["training_identity"]
    validate_semantic_identity(identity)
    _equal(identity["hash"], receipt.get("training_identity_hash", receipt.get("identity_hash")), "audited training identity")
    metadata = _json(run / "metadata.json")
    _equal(metadata["status"], "completed", "training completion")
    _equal(metadata["semantic_identity"], identity, "training metadata identity")
    payload = identity["payload"]
    ai = config.name == "aionopedia"
    _equal(identity["type"], "benchmark.aionopedia-training.v1" if ai else "benchmark.training.v1", "training identity type")
    selector = payload["task"] if ai else payload
    registry = dict(selector["registry_payload"] if ai else payload["registry"])
    _equal(registry.pop("baseline_protocol", None), expected_protocol, "training protocol")
    if "condition_width" in registry:
        _equal(registry.pop("condition_width"), len(task.condition_columns), "condition width")
    wanted = dict(task.registry_payload)
    wanted.pop("baseline_protocol", None)
    wanted.pop("condition_width", None)
    _equal(registry, wanted, "task registry")
    for key, value in (("benchmark_model", config.name), ("model", config.model), ("training", config.training)):
        _equal(payload[key], value, key)
    _equal(selector["task_id"], task.task_id, "training task")
    _equal(selector["fold"], task.fold, "training fold")
    _equal(selector["benchmark"] if ai else payload["domain"], task.benchmark, "training domain")
    _equal(payload["effective_seed"] if ai else payload["seed"], config.seed + int(task.fold or 1) - 1 if ai else config.seed, "seed")
    frozen = yaml.safe_load((run / "run_config.yaml").read_text())
    for key in ("model", "training", "seed", "features", "environment", "runtime"):
        _equal(frozen.get(key), config.to_dict().get(key), "frozen configuration/" + key)
    for split, paths in (("train", task.train_paths), ("valid", task.valid_paths)):
        _equal(payload["source_hashes"][split], [audit.fingerprint(p)["sha256"] for p in paths], "training sources/" + split)
    if config.name in {"mlp", "ecfp_xgboost"}:
        _equal(payload["feature"], feature_schema(config.features).to_dict(), "feature schema")
    if config.name == "ilbert":
        _equal(config.training.get("condition_transform"), "train_only_zscore", "ILBERT condition standardization")
    if not manifest.get("integrity"):
        raise ValueError("Training checkpoint lacks an integrity manifest")
    for name, entry in manifest["integrity"].items():
        path = audit.path(audit.relative(run) + "/" + name)
        if not path.is_relative_to(run.resolve()):
            raise ValueError("Checkpoint artifact escapes its run")
        audit.check(path, entry)
    for name in ("checkpoint.json", "metadata.json", "run_config.yaml", "environment.json"):
        if (run / name).is_file():
            audit.fingerprint(run / name)
    return manifest


def _completed(parent: Path) -> Path:
    runs = [p for p in parent.glob("attempt-*") if (p / "summary.json").is_file()
            and (p / "metadata.json").is_file() and _json(p / "metadata.json").get("status") == "completed"]
    if not runs:
        raise ValueError("Completed evaluation is missing: " + str(parent))
    return max(runs, key=lambda p: int(p.name.removeprefix("attempt-")))


def _evaluation(run: Path, task, split: str, manifests: list[tuple[Path, dict[str, Any]]], audit: SourceAudit):
    meta, summary = _json(run / "metadata.json"), _json(run / "summary.json")
    identity = meta["semantic_identity"]
    validate_semantic_identity(identity)
    _equal((identity["type"], meta["stage"], meta["operation"]), ("benchmark.evaluation.v1", "benchmark", "evaluate"), "evaluation kind")
    data, report = identity["payload"], summary["reporting"]
    model = manifests[0][1]["training_identity"]["payload"]["benchmark_model"]
    for key, value in (("benchmark_model", model), ("domain", task.benchmark), ("task", task.task_id),
                       ("split", split), ("fold", task.fold)):
        _equal(data[key], value, "evaluation selector/" + key)
    ensemble = task.benchmark == "stage3" and split == "test"
    _equal(data["ensemble_folds"], ensemble, "ensemble selector")
    _equal((summary["benchmark"], summary["task"], summary["split"], summary.get("fold")),
           (task.benchmark, task.task_id, split, task.fold), "summary selector")
    _equal((summary["model_selector"], summary.get("checkpoint_epoch")), ("final_training_state", None), "model selection")
    _equal((report["schema_version"], report["model_id"], report["benchmark"]),
           (1, model, "stage3_property" if task.benchmark == "stage3" else "simulation_property"), "report schema")
    _equal((report["protocol"]["ensemble"], report["protocol"]["fold"], report["protocol"]["expected_tasks"]),
           (ensemble, task.fold, [task.task_id]), "report protocol")
    originals = meta["provenance"]["checkpoints"]
    _equal(len(originals), len(manifests), "checkpoint count")
    _equal(len(data["checkpoints"]), len(manifests), "checkpoint fingerprint count")
    for original, fingerprint, (checkpoint, manifest) in zip(originals, data["checkpoints"], manifests, strict=True):
        _equal(audit.migrated(original + "/checkpoint.json"), checkpoint / "checkpoint.json", "checkpoint relocation")
        _equal(fingerprint["checkpoint_manifest_sha256"], audit.fingerprint(checkpoint / "checkpoint.json")["sha256"], "checkpoint SHA")
        _equal(fingerprint["training_identity"], manifest["training_identity"], "evaluation training identity")
        expected = {k: v for k, v in manifest["integrity"].items() if k in fingerprint["model_integrity"]}
        if not expected:
            raise ValueError("Evaluation has no model integrity evidence")
        _equal(fingerprint["model_integrity"], expected, "evaluation model integrity")
    source = task.test_path if split == "test" else task.valid_paths[0]
    source_sha = audit.fingerprint(source)["sha256"] if source.is_file() else None
    _equal(data["evaluation_source_sha256"], source_sha, "evaluation source SHA")
    comparison = report["comparison_identity"]
    validate_semantic_identity(comparison)
    cp = comparison["payload"]
    _equal((comparison["type"], cp["benchmark"], cp["split"], cp["expected"], cp["ensemble"], cp["folds"]),
           ("reporting.comparison.v1", report["benchmark"], split, [task.task_id], ensemble,
            [1, 2, 3, 4, 5] if ensemble else [task.fold] if task.fold else []), "comparison protocol")
    sources = cp["sources"]
    if task.benchmark == "stage3":
        wanted = {f"{task.task_id}:fold{p.stem.removeprefix('fold')}": audit.fingerprint(p)["sha256"]
                  for p in (*task.train_paths, *task.valid_paths)}
        if split == "test":
            wanted[task.task_id + ":test"] = source_sha
        _equal(sources, wanted, "comparison data sources")
    else:
        _equal(sources[task.task_id + ":train"], audit.fingerprint(task.train_paths[0])["sha256"], "simulation training source")
        _equal(sources[task.task_id + ":evaluation"], source_sha, "simulation evaluation source")
    audit.fingerprint(run / "metadata.json")
    audit.fingerprint(run / "summary.json")
    return summary


def _predictions(run: Path, summary: dict[str, Any], task, split: str, raw, audit: SourceAudit):
    entries = summary["reporting"]["predictions"]
    if not len(raw) and not entries:
        return np.empty(0), np.empty(0)
    if len(entries) != 1:
        raise ValueError("Prediction manifest must be unique")
    entry = entries[0]
    _equal((entry["path"], entry["task"], entry["rows"]),
           (f"predictions/{sanitize_task_id(task.task_id)}.csv", task.task_id, len(raw)), "prediction manifest")
    path = run / entry["path"]
    _equal(audit.fingerprint(path)["sha256"], entry["sha256"], "prediction SHA")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    _equal(len(rows), len(raw), "complete evaluation row set")
    ensemble = task.benchmark == "stage3" and split == "test"
    values = []
    for index, row in enumerate(rows):
        _equal(int(row["source_row"]), int(raw.source_rows[index].rsplit(":", 1)[1]), "prediction row order")
        if task.fold:
            _equal(int(row["source_fold"]), task.fold, "prediction source fold")
        _equal(tuple(_canonical(row[k]) for k in task.slots), raw.components[index], "molecular identity")
        _equal([float(row[k]) for k in task.condition_columns], raw.conditions[index].tolist(), "prediction conditions")
        _equal(float(row["target"]), float(raw.targets[index, 0]), "prediction targets")
        for key in task.audit_columns:
            _equal(row.get(key, ""), raw.audit_rows[index].get(key, ""), "prediction audit column")
        value = float(row["prediction_ensemble" if ensemble else "prediction"])
        if not math.isfinite(value):
            raise ValueError("Predictions must be finite")
        if ensemble:
            members = [float(row[f"prediction_fold{f}"]) for f in range(1, 6)]
            if not all(map(math.isfinite, members)) or not any(
                    math.isclose(value, mean, rel_tol=1e-7, abs_tol=1e-9)
                    for mean in (statistics.mean(members), float(np.asarray(members, dtype=np.float32).mean()))):
                raise ValueError("Test ensemble is not the original-unit five-model mean")
        values.append(value)
    return raw.targets[:, 0], np.asarray(values)


def _tables(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = list(dict.fromkeys(row["task"] for row in rows))
    task_rows = []
    for task in tasks:
        current = []
        for model in BASELINE_MODELS:
            values = [r for r in rows if r["task"] == task and r["model_id"] == model]
            result = {k: values[0][k] for k in ("task", "model_id", "model", "group")}
            result["count"] = sum(r["count"] for r in values)
            for metric in METRICS:
                numbers = [r[metric] for r in values]
                result[metric] = statistics.mean(numbers) if None not in numbers else None
                result[metric + "_fold_std"] = statistics.stdev(numbers) if len(numbers) == 5 and None not in numbers else None
            current.append(result)
        for row in current:
            row["rank"] = 1 + sum(other["mae"] < row["mae"] for other in current)
        task_rows.extend(current)
    overall = []
    for model in BASELINE_MODELS:
        values = [r for r in task_rows if r["model_id"] == model]
        overall.append({"model_id": model, "model": values[0]["model"], "tasks": len(values),
                        "macro_normalized_mae": statistics.mean(r["normalized_mae"] for r in values),
                        "macro_normalized_rmse": statistics.mean(r["normalized_rmse"] for r in values),
                        "mean_rank": statistics.mean(r["rank"] for r in values), "wins": sum(r["rank"] == 1 for r in values)})
    return {"tasks": task_rows, "folds": rows, "overall": sorted(overall, key=lambda r: (r["macro_normalized_mae"], r["model_id"]))}


def build_baseline_report(settings: dict[str, Any], repository: Path) -> dict[str, Any]:
    _equal(settings["protocol"], PROTOCOL, "report configuration protocol")
    baseline = repository / settings["baseline_root"]
    audit = SourceAudit(repository, baseline)
    authority_path = audit.path(settings["authority_config"])
    authority = load_stage3_config(authority_path)
    registry = {k: v for k, v in resolve_task_registry(authority).items() if v.enabled}
    _equal(set(registry) == set(FLFT_22_RETRAIN_TASKS + FLFT_22_COMPLETION_TASKS), True, "exact 22-task registry")
    _equal((authority.data.stage3_dir, authority.data.task_catalog),
           (Path("data/3407/stage3"), Path("data/3407/task_catalog.csv")), "3407 data roots")
    receipts = _receipts(settings, audit)
    groups, rows, comparisons = [], {k: [] for k in ("stage3_validation", "stage3_test", "simulation_validation", "simulation_test")}, {}
    datasets, training_targets = {}, {}
    no_test = set()
    for model in BASELINE_MODELS:
        for domain, tasks in (("stage3", tuple(registry)), ("simulation", CATALOG_V4_SIMULATION_TASKS)):
            for task_id in tasks:
                config, config_path, protocol = _task_config(repository, model, domain, task_id, Path(settings["authority_config"]))
                parent = baseline / ("retrain_22" if protocol == FLFT_22_PROTOCOL and model != "ilbert" else "") / model / domain / sanitize_task_id(task_id)
                if protocol == XGBOOST_LIGHTWEIGHT_PROTOCOL:
                    parent = baseline / "lightweight" / "ecfp_xgboost_500_d4" / domain / sanitize_task_id(task_id)
                records = receipts.get((model, domain, task_id), [])
                accepted = []
                for receipt in records:
                    training = receipt.get("training", receipt.get("folds", []))
                    if training and all(audit.migrated(r["run"] + "/checkpoint.json").parent.parent.parent == parent.resolve() for r in training):
                        accepted.append(receipt)
                if len(accepted) != 1:
                    raise ValueError(f"Exactly one approved source group is required: {model}/{task_id}")
                receipt = accepted[0]
                training = receipt.get("training", receipt.get("folds", []))
                folds = list(range(1, 6)) if domain == "stage3" else [None]
                _equal([r["fold"] for r in training], folds, "complete audited training folds")
                group = {"model_id": model, "domain": domain, "task": task_id, "protocol": protocol,
                         "source_root": audit.relative(parent), "config": str(config_path), "training": [], "evaluations": []}
                manifests, scales, task_refs = [], [], {}
                audit.fingerprint(repository / config_path)
                for record in training:
                    fold = record["fold"]
                    task = resolve_task(config, domain, task_id, fold)
                    run = audit.migrated(record["run"] + "/checkpoint.json").parent
                    manifest = _training(config, task, run, protocol, record, audit)
                    manifests.append((run, manifest))
                    group["training"].append({"run": audit.relative(run), "original_run": record["run"],
                                               "identity": manifest["training_identity"], "manifest_sha256": audit.fingerprint(run / "checkpoint.json")["sha256"]})
                    key = (task_id, fold)
                    if key not in datasets:
                        for path in task.train_paths:
                            if path not in training_targets:
                                with path.open(newline="", encoding="utf-8-sig") as handle:
                                    training_targets[path] = np.asarray([float(row[task.target_columns[0]]) for row in csv.DictReader(handle)])
                        targets = np.concatenate([training_targets[path] for path in task.train_paths])
                        scale = float(np.std(targets, ddof=0))
                        if not math.isfinite(scale):
                            raise ValueError("Training targets must be finite")
                        datasets[key] = (scale if scale > 0 else 1.0, load_split(task, "valid"))
                    scale, valid = datasets[key]
                    scales.append(scale)
                    task_refs[fold] = (task, valid)
                for split in ("valid", "test"):
                    selected = list(task_refs) if split == "valid" else [None]
                    for fold in selected:
                        if split == "valid":
                            task, raw = task_refs[fold]
                        else:
                            task = replace(next(iter(task_refs.values()))[0], fold=None)
                            key = (task_id, "test")
                            if key not in datasets:
                                datasets[key] = load_split(task, "test")
                            raw = datasets[key]
                        ev = _completed(parent / (f"evaluate_valid_fold{fold}" if fold else f"evaluate_{split}"))
                        chosen = [manifests[fold - 1]] if fold else manifests
                        summary = _evaluation(ev, task, split, chosen, audit)
                        approved = [e for e in receipt["evaluations"] if e["split"] == split and e.get("fold") == fold]
                        if len(approved) != 1:
                            raise ValueError("Exactly one approved evaluation is required")
                        _equal(approved[0]["rows"], len(raw), "audited evaluation row coverage")
                        if approved[0].get("run"):
                            _equal(audit.migrated(approved[0]["run"] + "/metadata.json"), ev / "metadata.json", "audited evaluation path")
                        if len(raw):
                            _equal([e["sha256"] for e in summary["reporting"]["predictions"]],
                                   [approved[0]["prediction_sha256"]], "audited prediction SHA")
                        y, predicted = _predictions(ev, summary, task, split, raw, audit)
                        group["evaluations"].append({"run": audit.relative(ev), "identity": _json(ev / "metadata.json")["semantic_identity"], "rows": len(y)})
                        if not len(y):
                            if domain != "stage3" or split != "test":
                                raise ValueError("Only experimental test splits may be empty")
                            no_test.add(task_id)
                            continue
                        scale = scales[fold - 1] if fold else statistics.mean(scales)
                        label = ("stage3" if domain == "stage3" else "simulation") + ("_validation" if split == "valid" else "_test")
                        rows[label].append({"model_id": model, "model": config.display_name, "task": task_id,
                                            "group": task.meta_group, "fold": fold, "source_run": audit.relative(ev),
                                            "common_scale": scale, **_metrics(y, predicted, scale)})
                        refs = comparisons.setdefault(label, {"sources": {}, "normalization": {}})
                        for path in (*task.train_paths, *task.valid_paths, *((task.test_path,) if split == "test" else ())):
                            if path.is_file():
                                refs["sources"][path.as_posix()] = audit.fingerprint(path)["sha256"]
                        refs["normalization"][task_id + (f":fold{fold}" if fold else "")] = {"scale": scale}
                groups.append(group)
        print(f"Verified baseline sources: {model}", flush=True)
    sections = {label: _tables(values) for label, values in rows.items()}
    for label, refs in comparisons.items():
        simulation = label.startswith("simulation")
        comparisons[label] = comparison_identity("simulation_property" if simulation else "stage3_property",
            split="valid" if label.endswith("validation") else "test", expected=list(dict.fromkeys(r["task"] for r in rows[label])),
            sources=refs["sources"], normalization=refs["normalization"], folds=() if simulation else (1, 2, 3, 4, 5),
            ensemble=label == "stage3_test")
    source_identity = semantic_identity("benchmark.baseline-22-sources.v1", {"protocol": PROTOCOL, "groups": groups, "files": audit.files})
    identity = semantic_identity("reporting.baseline-22-comparison.v1", {"protocol": PROTOCOL, "comparisons": comparisons,
        "registry": {k: v.to_dict() for k, v in registry.items()}, "source_identity": source_identity["hash"],
        "normalization": "valid: four-fold train population std; test: mean of five train std; simulation: train population std; constant scale 1"})
    return {"schema_version": 1, "protocol": PROTOCOL, "comparison_identity": identity,
            "source_manifest": source_identity, "sections": sections,
            "coverage": {"models": list(BASELINE_MODELS), "experimental_tasks": list(registry),
                         "experimental_no_test": sorted(no_test), "simulation_tasks": list(CATALOG_V4_SIMULATION_TASKS),
                         "training_jobs": sum(len(g["training"]) for g in groups),
                         "evaluation_runs": sum(len(g["evaluations"]) for g in groups)},
            "scope": {"ilume_included": False, "historical_reports_changed": False, "simulations_in_experiment_macro": False,
                      "fold_std": "折间样本标准差，不表示独立种子稳定性或统计显著性",
                      "source_audits": "复核原审计批准的身份、当前数据SHA、配方、权重及完整有序预测；不重新执行原生模型预处理或模型推理"}}


def publish_baseline_report(config_path: Path, output: Path, repository: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Baseline reports must not overwrite existing output")
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    baseline = (repository / settings["baseline_root"]).resolve()
    if output.resolve().is_relative_to(baseline) or baseline.is_relative_to(output.resolve()):
        raise ValueError("Baseline report input and output must not contain each other")
    payload = build_baseline_report(settings, repository)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent))
    try:
        source = payload.pop("source_manifest")
        (staging / "source_manifest.json").write_text(json.dumps(source, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        payload["source_manifest"] = {"path": "source_manifest.json", "sha256": sha256_file(staging / "source_manifest.json"), "identity_hash": source["hash"]}
        lines = ["# 完整22任务 baseline 报告", "", "仅包含十个baseline，不包含ILUME；实验与模拟分别计算任务等权汇总。", "",
                 "实验验证采用五折指标均值及折间样本标准差；测试采用五模型原单位预测均值。共同归一化尺度为每折完整训练行目标总体标准差，测试取五折尺度均值；模拟取原训练集总体标准差，零方差取1。", "",
                 "无测试集的实验任务：" + "、".join(payload["coverage"]["experimental_no_test"]), ""]
        for label, section in payload["sections"].items():
            dest = staging / label
            dest.mkdir()
            for name, values in section.items():
                with (dest / (name + ".csv")).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                    writer.writeheader()
                    writer.writerows(values)
            lines += ["## " + label, "", "| 模型 | 任务数 | 宏平均NMAE | 平均排名 | 第一名任务数 |", "|---|---:|---:|---:|---:|"]
            lines += [f"| {r['model']} | {r['tasks']} | {r['macro_normalized_mae']:.6g} | {r['mean_rank']:.3f} | {r['wins']} |" for r in section["overall"]]
            lines += [""]
        (staging / "overview.md").write_text("\n".join(lines), encoding="utf-8")
        (staging / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        (staging / "metadata.json").write_text(json.dumps({"stage": "benchmark", "operation": "baseline-22-summary",
            "status": "completed", "semantic_identity": payload["comparison_identity"],
            "provenance": {"source_manifest_sha256": payload["source_manifest"]["sha256"]}}, indent=2) + "\n")
        shutil.copyfile(config_path, staging / "run_config.yaml")
        files = {p.relative_to(staging).as_posix(): {"size": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(staging.rglob("*")) if p.is_file()}
        (staging / "manifest.json").write_text(json.dumps({"comparison_identity": payload["comparison_identity"], "files": files}, indent=2) + "\n")
        if output.exists():
            raise FileExistsError("Baseline report destination already exists")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return payload
