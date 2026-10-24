"""独立论文报告：完整评估预测、共同训练集尺度、明确选择的 ILUME。"""
from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from common.identity import semantic_identity, validate_semantic_identity
from common.io import sha256_file
from common.reporting import comparison_identity, sanitize_task_id
from stage3.config import load_stage3_config
from stage3.data import canonicalize_smiles, iter_rows, resolve_task_registry, source_path, test_path
from .summary import discover_candidates


BASELINE_MODELS = (
    "aifc", "aionopedia", "dmpnn", "ecfp_xgboost", "ilbert", "iltransr",
    "llasmol", "mlp", "molformer", "spmm",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _within(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Report source path escapes its root: " + name)
    return path


@lru_cache(maxsize=65536)
def _canonical(value: str) -> str:
    return canonicalize_smiles(value, "Paper prediction molecular identity")


def _validate_comparison(identity: dict[str, Any], expected: dict[str, Any], *, fragment: bool = False) -> None:
    validate_semantic_identity(identity)
    actual = dict(identity["payload"])
    wanted = dict(expected["payload"])
    actual.pop("normalization", None)
    wanted.pop("normalization", None)
    if fragment:
        tasks = actual["expected"]
        if len(tasks) != 1 or tasks[0] not in wanted["expected"]:
            raise ValueError("Evaluation fragment task mismatch")
        wanted["expected"] = tasks
        wanted["sources"] = {k: v for k, v in wanted["sources"].items() if k.startswith(tasks[0] + ":")}
        if wanted["split"] == "valid" and (len(actual["folds"]) != 1 or actual["folds"][0] not in range(1, 6)):
            raise ValueError("Validation fragment fold mismatch")
        if wanted["split"] == "test" and actual["folds"] != [1, 2, 3, 4, 5]:
            raise ValueError("Test fragment must include all five folds")
        wanted["folds"] = actual["folds"]
    if identity["type"] != "reporting.comparison.v1" or actual != wanted:
        raise ValueError("Paper comparison identity mismatch: sources, tasks, split, or ensemble protocol differ")


def _run(root: Path, repository_root: Path, audit: dict[str, Any]) -> dict[str, Any]:
    metadata = _json(root / "metadata.json")
    summary = _json(root / "summary.json")
    identity = metadata["semantic_identity"]
    validate_semantic_identity(identity)
    if metadata.get("status") != "completed" or metadata.get("provenance", {}).get("reporting_schema_version") != 1:
        raise ValueError("Paper inputs must be completed schema-v1 evaluations: " + _relative(root, repository_root))
    reporting = summary.get("reporting", {})
    if reporting.get("schema_version") != 1:
        raise ValueError("Evaluation lacks reporting schema-v1")
    if metadata.get("stage") == "stage3":
        selector = identity["payload"].get("selector", {})
        protocol = reporting["protocol"]
        expected_selector = {"split": summary.get("split"), "fold": protocol.get("fold"),
                             "checkpoint_epoch": None, "model_selector": "three_phase_final",
                             "tasks": protocol["expected_tasks"], "ensemble_folds": bool(protocol.get("ensemble"))}
        if identity["type"] != "stage3.evaluation" or selector != expected_selector:
            raise ValueError("ILUME evaluation selector does not match its original identity")
    relative = _relative(root, repository_root)
    checkpoints = identity["payload"].get("checkpoints", [])
    originals = metadata.get("provenance", {}).get("checkpoints", [])
    if metadata.get("stage") == "benchmark" and metadata.get("operation") == "evaluate":
        data = identity["payload"]
        if identity["type"] != "benchmark.evaluation.v1" or not checkpoints or len(originals) != len(checkpoints):
            raise ValueError("Baseline evaluation lacks complete checkpoint provenance")
        if (data.get("domain"), data.get("benchmark_model"), data.get("task"), data.get("split"), data.get("fold")) != (
                "stage3", reporting["model_id"], summary.get("task"), summary.get("split"), summary.get("fold")):
            raise ValueError("Baseline evaluation identity does not match its report")
        if data.get("ensemble_folds") != (summary["split"] == "test"):
            raise ValueError("Baseline ensemble selection does not match its original identity")
        training_folds = []
        for fingerprint in checkpoints:
            training = fingerprint["training_identity"]
            validate_semantic_identity(training)
            payload = training["payload"]
            if reporting["model_id"] == "aionopedia":
                expected_type = "benchmark.aionopedia-training.v1"
                task_identity = payload.get("task", {})
                registry = task_identity.get("registry_payload", {})
            else:
                expected_type = "benchmark.training.v1"
                task_identity = payload
                registry = payload.get("registry", {})
            expected = (expected_type, reporting["model_id"], summary.get("task"), "catalog_v4_3407")
            actual = (training["type"], payload.get("benchmark_model"), task_identity.get("task_id"), registry.get("baseline_protocol"))
            if actual != expected:
                raise ValueError(f"Baseline training identity mismatch at {relative}: expected {expected}, got {actual}")
            training_folds.append(task_identity.get("fold"))
        expected_folds = [summary["fold"]] if summary["split"] == "valid" else [1, 2, 3, 4, 5]
        if training_folds != expected_folds:
            raise ValueError(f"Training checkpoint fold mismatch at {relative}: expected {expected_folds}, got {training_folds}")
    if originals:
        for name, fingerprint in zip(originals, checkpoints, strict=True):
            manifest = _within(repository_root, name) / "checkpoint.json"
            if sha256_file(manifest) != fingerprint["checkpoint_manifest_sha256"]:
                raise ValueError("Evaluation checkpoint manifest SHA256 mismatch")
            if _json(manifest).get("training_identity") != fingerprint["training_identity"]:
                raise ValueError("Evaluation checkpoint training identity mismatch")
    audit[relative] = {
        "semantic_identity": identity,
        "summary_sha256": sha256_file(root / "summary.json"),
        "metadata_sha256": sha256_file(root / "metadata.json"),
        "comparison_identity": reporting.get("comparison_identity"),
        "predictions": reporting.get("predictions", []),
    }
    return summary


def _reference(config: Any, registry: dict[str, Any], split: str) -> tuple[dict[str, Any], dict[str, Any]]:
    references = {}
    sources = {}
    scales = {}
    for task, spec in registry.items():
        folds = {f: list(iter_rows(config, spec, (f,))) for f in range(1, 6)}
        targets = {f: [float(row[spec.target_column]) for _, _, row in rows] for f, rows in folds.items()}
        for fold in range(1, 6):
            if not targets[fold]:
                raise ValueError("Validation folds must not be empty: " + task)
            sources[f"{task}:fold{fold}"] = sha256_file(source_path(config, spec, fold))
            train = [y for f in range(1, 6) if f != fold for y in targets[f]]
            scale = float(np.std(train, ddof=0))
            if not math.isfinite(scale):
                raise ValueError("Training targets contain non-finite values: " + task)
            scales[f"{task}:fold{fold}"] = {"scale": scale if scale > 0 else 1.0}
        rows = list(iter_rows(config, spec, None)) if split == "test" else []
        if split == "test" and not rows:
            for fold in range(1, 6):
                sources.pop(f"{task}:fold{fold}")
                scales.pop(f"{task}:fold{fold}")
            continue
        if split == "test":
            sources[f"{task}:test"] = sha256_file(test_path(config, spec))
        selected = {0: rows} if split == "test" else folds
        references[task] = {
            "rows": {f: rows for f, rows in selected.items()},
            "train_counts": {f: sum(len(v) for k, v in folds.items() if k != f) for f in range(1, 6)},
            "train_systems": {f: len({tuple(_canonical(row[name]) for name in spec.identity_columns)
                                      for k, rows in folds.items() if k != f for _, _, row in rows}) for f in range(1, 6)},
        }
    if not references:
        raise ValueError("The selected split has no comparable tasks")
    return references, comparison_identity(
        "stage3_property", split=split, expected=tuple(references), sources=sources,
        normalization=scales, folds=(1, 2, 3, 4, 5), ensemble=split == "test",
    )


def _predictions(run: Path, summary: dict[str, Any], spec: Any, rows: list[Any], fold: int,
                 repository_root: Path, audit: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    manifests = [m for m in summary["reporting"].get("predictions", []) if m.get("task") == spec.task_id]
    if len(manifests) != 1:
        raise ValueError("Prediction manifests must be complete and unique: " + spec.task_id)
    manifest = manifests[0]
    path = _within(run, manifest["path"])
    if manifest["path"] != f"predictions/{sanitize_task_id(spec.task_id)}.csv" or sha256_file(path) != manifest["sha256"]:
        raise ValueError("Prediction path or SHA256 mismatch: " + spec.task_id)
    field = "prediction" if fold else "prediction_ensemble"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"source_row", "target", field, *spec.identity_columns, *spec.condition_columns}
        if fold:
            required.add("source_fold")
        else:
            required.update(f"prediction_fold{f}" for f in range(1, 6))
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Prediction file lacks required columns: " + spec.task_id)
        predicted_rows = list(reader)
    if len(predicted_rows) != len(rows) or len(rows) != manifest["rows"]:
        raise ValueError("Predictions do not cover the complete evaluation row set: " + spec.task_id)
    y, prediction = [], []
    for predicted, (source_fold, row_number, original) in zip(predicted_rows, rows, strict=True):
        if int(predicted["source_row"]) != row_number or (fold and int(predicted["source_fold"]) != source_fold):
            raise ValueError("Prediction source row or order mismatch: " + spec.task_id)
        if any(_canonical(predicted[name]) != _canonical(original[name]) for name in spec.identity_columns):
            raise ValueError("Prediction molecular identity mismatch: " + spec.task_id)
        for name in (*spec.condition_columns, "target"):
            a = float(predicted[name])
            b = float(original[spec.target_column if name == "target" else name])
            if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Prediction condition or target mismatch: " + spec.task_id)
        value = float(predicted[field])
        if not math.isfinite(value):
            raise ValueError("Predictions contain non-finite values: " + spec.task_id)
        if not fold:
            members = [float(predicted[f"prediction_fold{f}"]) for f in range(1, 6)]
            averages = (statistics.mean(members), float(np.asarray(members, dtype=np.float32).mean()))
            if not all(map(math.isfinite, members)) or not any(math.isclose(value, mean, rel_tol=1e-7, abs_tol=1e-9) for mean in averages):
                raise ValueError("Test prediction does not match the five-model mean in original units: " + spec.task_id)
        y.append(float(predicted["target"]))
        prediction.append(value)
    audit[_relative(run, repository_root)]["verified_predictions"] = True
    return np.asarray(y), np.asarray(prediction)


def _metrics(y: np.ndarray, prediction: np.ndarray, scale: float) -> dict[str, Any]:
    error = prediction - y
    denominator = float(np.square(y - y.mean()).sum())
    mae = float(np.abs(error).mean())
    rmse = float(np.sqrt(np.square(error).mean()))
    return {"count": len(y), "mae": mae, "rmse": rmse,
            "r2": 1.0 - float(np.square(error).sum()) / denominator if denominator > 0 else None,
            "normalized_mae": mae / scale, "normalized_rmse": rmse / scale}


def _tables(folds: list[dict[str, Any]], specs: dict[str, Any], split: str) -> dict[str, Any]:
    tasks = tuple(dict.fromkeys(row["task"] for row in folds))
    models = ("ilume", *BASELINE_MODELS)
    task_rows = []
    for task in tasks:
        for model in models:
            values = [r for r in folds if r["task"] == task and r["model_id"] == model]
            row = {"task": task, "group": specs[task].meta_group, "model_id": model,
                   "model": values[0]["model"], "count": sum(r["count"] for r in values)}
            for metric in ("mae", "rmse", "r2", "normalized_mae", "normalized_rmse"):
                numbers = [r[metric] for r in values if r[metric] is not None]
                row[metric] = statistics.mean(numbers) if len(numbers) == len(values) else None
                row[metric + "_fold_std"] = statistics.stdev(numbers) if len(numbers) == 5 else None
            task_rows.append(row)
        current = task_rows[-len(models):]
        for row in current:
            # 并列采用 competition rank；首位并列全部计入 wins。
            row["rank"] = 1 + sum(r["mae"] < row["mae"] for r in current)
            best = min(r["mae"] for r in current)
            row["ratio_to_best"] = row["mae"] / best if best > 0 else (1.0 if row["mae"] == 0 else None)
    overall = []
    for model in models:
        values = [r for r in task_rows if r["model_id"] == model]
        overall.append({"model_id": model, "model": values[0]["model"], "tasks": len(tasks),
                        "macro_normalized_mae": statistics.mean(r["normalized_mae"] for r in values),
                        "mean_rank": statistics.mean(r["rank"] for r in values),
                        "wins": sum(r["rank"] == 1 for r in values),
                        "top3": sum(r["rank"] <= 3 for r in values)})
    overall.sort(key=lambda r: (r["macro_normalized_mae"], r["model_id"]))
    gains = []
    for task in tasks:
        values = [r for r in task_rows if r["task"] == task]
        ilume = next(r for r in values if r["model_id"] == "ilume")
        best = min((r for r in values if r["model_id"] != "ilume"), key=lambda r: (r["mae"], r["model_id"]))
        gains.append({"task": task, "group": specs[task].meta_group, "ilume_mae": ilume["mae"],
                      "best_baseline": best["model"], "best_baseline_mae": best["mae"],
                      "mae_reduction_percent": 100 * (1 - ilume["mae"] / best["mae"]) if best["mae"] > 0 else None,
                      "ilume_rank": ilume["rank"]})
    thresholds = sorted({1.0, *(r["ratio_to_best"] for r in task_rows if r["ratio_to_best"] is not None)})
    profile = [{"model_id": model, "model": next(r["model"] for r in task_rows if r["model_id"] == model),
                "relative_mae_threshold": threshold,
                "task_fraction": sum(r["ratio_to_best"] is not None and r["ratio_to_best"] <= threshold
                                     for r in task_rows if r["model_id"] == model) / len(tasks)}
               for model in models for threshold in thresholds]
    return {"split": split, "overall": overall, "tasks": task_rows, "folds": folds,
            "gains": gains, "performance_profile": profile}


def build_paper_summary(config_path: Path, ilume_root: Path, baseline_root: Path,
                        repository_root: Path, *, with_test: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_stage3_config(config_path)
    if config.data.seed != 3407 or config.data.stage3_dir != Path("data/3407/stage3"):
        raise ValueError("Paper reporting requires a matching catalog_v4_3407 data configuration")
    registry = {k: v for k, v in resolve_task_registry(config).items() if v.enabled}
    candidates = discover_candidates(baseline_root, repository_root)
    baseline = {c.summary["reporting"]["model_id"]: c for c in candidates
                if c.current and c.summary and c.metadata.get("status") == "completed"}
    if len(candidates) != len(BASELINE_MODELS) or set(baseline) != set(BASELINE_MODELS):
        raise ValueError("Paper reporting requires ten unique, completed formal baselines")
    for candidate in baseline.values():
        if candidate.summary["reporting"].get("data_protocol") != "catalog_v4_3407" or candidate.summary.get("jobs", {}).get("failed") != 0:
            raise ValueError("Baseline protocol mismatch or failed jobs remain")
    simulation = ilume_root / "valid" / "simulation"
    selected = discover_candidates(
        ilume_root / "valid", repository_root,
        exclude_roots=[simulation] if simulation.is_dir() else (),
    )
    validation = {c.summary["reporting"]["protocol"].get("fold"): c for c in selected
                  if c.current and c.summary and c.metadata.get("status") == "completed"}
    if len(selected) != 5 or set(validation) != set(range(1, 6)):
        raise ValueError("ILUME requires exactly five complete validation folds from one selected run")
    if len({c.summary["reporting"]["study_id"] for c in validation.values()}) != 1:
        raise ValueError("ILUME validation folds belong to different studies")
    prepared = {c.metadata["semantic_identity"]["payload"].get("prepared_identity") for c in validation.values()}
    training = [validation[f].metadata["semantic_identity"]["payload"].get("checkpoint_training_identities", []) for f in range(1, 6)]
    if len(prepared) != 1 or None in prepared or any(len(ids) != 1 for ids in training) or len({ids[0] for ids in training}) != 5:
        raise ValueError("ILUME folds have inconsistent preparation identities or training provenance")
    test = None
    if with_test:
        simulation = ilume_root / "test" / "simulation"
        selected_test = discover_candidates(
            ilume_root / "test", repository_root,
            exclude_roots=[simulation] if simulation.is_dir() else (),
        )
        if (len(selected_test) != 1 or not selected_test[0].current
                or selected_test[0].metadata.get("status") != "completed" or selected_test[0].summary is None):
            raise ValueError("ILUME requires exactly one completed test ensemble")
        test = selected_test[0]
        if test.summary["reporting"]["study_id"] != validation[1].summary["reporting"]["study_id"]:
            raise ValueError("Validation and test must belong to the same ILUME study")
        data = test.metadata["semantic_identity"]["payload"]
        if data.get("prepared_identity") not in prepared or data.get("checkpoint_training_identities") != [ids[0] for ids in training]:
            raise ValueError("Test models must match the training identities of all five validation folds")
    audit: dict[str, Any] = {}
    for candidate in baseline.values():
        _run(candidate.root, repository_root, audit)
    sections = {}
    comparisons = {}
    scatters = {}
    for split in (("valid", "test") if with_test else ("valid",)):
        references, expected = _reference(config, registry, split)
        comparisons[split] = expected
        label = "stage3_validation" if split == "valid" else "stage3_test"
        rows = []
        for model in ("ilume", *BASELINE_MODELS):
            if model != "ilume":
                report = baseline[model].summary["reporting"]
                section = report["benchmarks"][label]
                if section.get("status") not in {None, "complete"}:
                    raise ValueError("Baseline evaluation is incomplete: " + model)
                _validate_comparison(section["comparison_identity"], expected)
                mapping = report["source_runs"][label]
                covered = set(mapping)
                if (split == "valid" and covered != set(references)) or not set(references) <= covered or not covered <= set(registry):
                    raise ValueError("Baseline task coverage is incomplete: " + model)
                if split == "valid" and any(set(v) != {f"fold{f}" for f in range(1, 6)} for v in mapping.values()):
                    raise ValueError("Baseline validation sources must cover exactly five folds")
            for task, reference in references.items():
                points = []
                for fold, source_rows in reference["rows"].items():
                    if model == "ilume":
                        candidate = validation[fold] if fold else test
                        run = candidate.root
                    else:
                        relative = mapping[task][f"fold{fold}"] if fold else mapping[task]
                        run = _within(repository_root, relative)
                        if not run.is_relative_to(baseline[model].root.resolve()):
                            raise ValueError("Baseline evaluation source is outside the selected model root")
                    summary = _run(run, repository_root, audit)
                    reporting = summary["reporting"]
                    if model != "ilume":
                        source_key = f"{task}:fold{fold}" if fold else f"{task}:test"
                        identity = audit[_relative(run, repository_root)]["semantic_identity"]
                        if identity["payload"].get("evaluation_source_sha256") != expected["payload"]["sources"][source_key]:
                            raise ValueError("Evaluation source SHA256 does not match its original identity")
                    if reporting["model_id"] != model or summary.get("split") != split:
                        raise ValueError("Evaluation model or split mismatch")
                    if model == "ilume" and summary.get("model_selector") != "three_phase_final":
                        raise ValueError("Paper reporting only accepts ILUME three_phase_final evaluations")
                    protocol = reporting["protocol"]
                    if bool(protocol.get("ensemble")) != (split == "test") or protocol.get("fold") != (fold or None):
                        raise ValueError("Evaluation fold or ensemble protocol mismatch")
                    if model != "ilume" and reporting["comparison_identity"]["payload"]["folds"] != ([fold] if fold else [1, 2, 3, 4, 5]):
                        raise ValueError("Baseline comparison fragment fold does not match the selected fold")
                    _validate_comparison(reporting["comparison_identity"], expected, fragment=model != "ilume")
                    y, prediction = _predictions(run, summary, registry[task], source_rows, fold, repository_root, audit)
                    scale = (expected["payload"]["normalization"][f"{task}:fold{fold}"]["scale"] if fold else
                             statistics.mean(expected["payload"]["normalization"][f"{task}:fold{f}"]["scale"] for f in range(1, 6)))
                    values = _metrics(y, prediction, scale)
                    rows.append({"model_id": model, "model": reporting["model_display_name"], "task": task,
                                 "fold": fold or None, "group": registry[task].meta_group,
                                 "split_strategy": registry[task].split_strategy, "common_scale": scale,
                                 "reference_train_rows": reference["train_counts"].get(fold),
                                 "reference_train_systems": reference["train_systems"].get(fold),
                                 "source_run": _relative(run, repository_root), **values})
                    if model == "ilume":
                        points.extend(zip(y.tolist(), prediction.tolist()))
                if model == "ilume":
                    scatters[(split, task)] = points
        sections[split] = _tables(rows, registry, split)
    normalization = {"valid": "每折其余四折完整训练集目标 population std；零方差取1；五折再按任务等权",
                     "test": "先对五模型原单位预测逐行平均；归一化尺度为五折共同训练集 std 的算术平均"}
    comparison = semantic_identity("reporting.paper-comparison.v1", {
        "data_protocol": "catalog_v4_3407", "comparisons": comparisons,
        "config_sha256": sha256_file(config_path),
        "normalization": normalization, "task_rank_metric": "five_fold_mean_raw_mae",
        "ilume": _relative(ilume_root, repository_root),
        "input_fingerprints": {k: {name: v[name] for name in ("metadata_sha256", "summary_sha256")}
                               for k, v in sorted(audit.items())},
    })
    payload = {"schema_version": 1, "comparison_identity": comparison,
               "config": {"path": _relative(config_path, repository_root), "sha256": sha256_file(config_path)},
               "normalization": normalization, "sections": sections, "sources": audit,
               "scope": {"experimental_only": True, "selected_split": 3407,
                         "split_selection": "3407由五套划分的验证结果选出；不代表预先固定或独立随机种子重复",
                         "uncertainty": "fold_std仅表示折间变化，不是显著性结论",
                         "training_information": "共同评估尺度不表示预训练数据或HPO预算相同；需披露各模型训练信息与搜索预算",
                         "simulations": "实验报告不混入模拟任务；三模拟及共同热膨胀使用现有独立汇总",
                         "unmeasured": "不推断消融贡献、条件趋势或计算成本；需独立配套证据"}}
    return payload, scatters


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _overview(payload: dict[str, Any]) -> str:
    lines = ["# ILUME 论文结果报告", "", "## 口径", "",
             "总体指标使用共同完整训练集尺度；原始训练、评估身份和既有榜单不改写。",
             "逐任务排名、胜出数、前三数、相对最强基线变化及性能覆盖曲线均按原单位MAE。",
             "最强基线是逐任务十模型的最优结果集合；负增益完整保留。并列采用competition rank。",
             "验证散点合并五个留出折；散点的样本合并误差与五折等权指标不同。",
             "", "## 解释边界", ""]
    lines.extend("- " + value for key, value in payload["scope"].items() if isinstance(value, str))
    for split, section in payload["sections"].items():
        lines.extend(["", "## " + ("五折验证" if split == "valid" else "五模型测试集成"), "",
                      "| 模型 | 任务数 | 共同尺度 nMAE ↓ | 平均名次 ↓ | 第一 | 前三 |",
                      "|---|---:|---:|---:|---:|---:|"])
        for row in section["overall"]:
            lines.append(f"| {row['model']} | {row['tasks']} | {row['macro_normalized_mae']:.6f} | {row['mean_rank']:.3f} | {row['wins']} | {row['top3']} |")
        lines.extend(["", "| 任务 | 最强基线 | ILUME MAE | 基线 MAE | MAE降低率 |", "|---|---|---:|---:|---:|"])
        for row in section["gains"]:
            gain = "未定义：基线MAE为0" if row["mae_reduction_percent"] is None else f"{row['mae_reduction_percent']:.2f}%"
            lines.append(f"| {row['task']} | {row['best_baseline']} | {row['ilume_mae']:.6g} | {row['best_baseline_mae']:.6g} | {gain} |")
    return "\n".join(lines) + "\n"


def publish_paper_summary(config_path: Path, ilume_root: Path, baseline_root: Path,
                          output: Path, repository_root: Path, *, with_test: bool = False) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Paper reports must not overwrite existing output: " + str(output))
    for source in (ilume_root, baseline_root):
        if output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
            raise ValueError("Paper input and output directories must not contain each other")
    from .paper_plots import write_paper_figures

    payload, scatters = build_paper_summary(config_path, ilume_root, baseline_root, repository_root, with_test=with_test)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent))
    try:
        for split, section in payload["sections"].items():
            destination = staging / split
            destination.mkdir()
            for name in ("overall", "tasks", "folds", "gains", "performance_profile"):
                _write_csv(destination / f"{name}.csv", section[name])
        (staging / "overview.md").write_text(_overview(payload), encoding="utf-8")
        (staging / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        write_paper_figures(payload, scatters, staging)
        files = {p.relative_to(staging).as_posix(): {"size": p.stat().st_size, "sha256": sha256_file(p)}
                 for p in sorted(staging.rglob("*")) if p.is_file()}
        (staging / "manifest.json").write_text(json.dumps({"comparison_identity": payload["comparison_identity"], "files": files}, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError("Paper report destination already exists")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return payload


__all__ = ["build_paper_summary", "publish_paper_summary"]
