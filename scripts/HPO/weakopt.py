"""Frozen Stage3 parameter trials with validation-only selection on GPU2/3."""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import contextmanager
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

from _common import ROOT, Job, evaluate_fold, native, run_queue
from _configs import fingerprint, read_yaml, write_yaml
from common.artifact_cache import artifact_lock
from common.identity import validate_semantic_identity
from common.io import atomic_json, sha256_file
from stage3.config import load_stage3_config, stage3_config_from_dict

CONFIGS = ROOT / "configs/HPO/WeakOpt"
OUTPUT = ROOT / "outputs/HPO/WeakOpt"
BASE = ROOT / "configs/v4/stage3/optimized_3407.yaml"
BASE_RUN = ROOT / "outputs/v4/stage3/optimized_3407"
BASELINES = ROOT / "outputs/benchmarks/v4_3407"
LIMIT = 80
FIELDS = {"phase1_private_lr", "phase1_private_epochs", "phase2_private_epochs", "phase3_private_epochs"}


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes its source directory")
    return path


def verify_files(files, root=None):
    root = ROOT if root is None else root
    for name, digest in files.items():
        path = checked_path(root, name)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Frozen artifact SHA256 mismatch: {name}")


def validate_recipe(base, raw):
    """Permit task-local learning rates, budgets and dropout, with fixed topology."""
    remainder = deepcopy(raw)
    for task, detail in base["tasks"].items():
        if task not in remainder.get("tasks", {}):
            raise ValueError("Task coverage differs from the frozen reference")
        for field in FIELDS:
            if field in detail:
                remainder["tasks"][task][field] = detail[field]
            else:
                remainder["tasks"][task].pop(field, None)
        overrides = remainder["tasks"][task].setdefault("model_overrides", {})
        original = detail.get("model_overrides", {})
        if "private_dropout" in original:
            overrides["private_dropout"] = original["private_dropout"]
        else:
            overrides.pop("private_dropout", None)
        if not overrides and "model_overrides" not in detail:
            remainder["tasks"][task].pop("model_overrides")
    if remainder != base:
        raise ValueError("Trial changes a field outside the WeakOpt parameter whitelist")
    stage3_config_from_dict(raw)


def first_batch(base):
    tasks = ["viscosity", "density", "refractive_index", "surface_tension", "glass_transition_temperature", "melting_point"]
    proposals = []
    for field in ("dropout", "lr", "phase3"):
        for suffix in tasks:
            task = "experiment/" + suffix
            change = ({"model_overrides": {"private_dropout": 0.05}} if field == "dropout" else
                      {"phase1_private_lr": base["tasks"][task]["phase1_private_lr"] * 0.7} if field == "lr" else
                      {"phase3_private_epochs": base["tasks"][task]["phase3_private_epochs"] + 2})
            proposals.append({"description": f"{suffix}: {field}", "overrides": {task: change}})
    for value in (0.0, 0.05):
        proposals.append({"description": f"water activity dropout {value}", "overrides": {
            "experiment/water_activity_coefficient": {"model_overrides": {"private_dropout": value}}}})
    for value in (1, 2):
        proposals.append({"description": f"static permittivity Phase3 {value}", "overrides": {
            "experiment/static_relative_permittivity": {"phase3_private_epochs": value}}})
    proposals.extend([
        {"description": "pEC50 learning rate x0.7", "overrides": {"experiment/pec50": {
            "phase1_private_lr": base["tasks"]["experiment/pec50"]["phase1_private_lr"] * 0.7}}},
        {"description": "pEC50 dropout 0.05", "overrides": {"experiment/pec50": {"model_overrides": {"private_dropout": 0.05}}}},
    ])
    return proposals


def apply_overrides(base, proposal):
    raw = deepcopy(base)
    for task, changes in proposal["overrides"].items():
        if task not in raw["tasks"] or not task.startswith("experiment/"):
            raise ValueError("Only existing experimental task parameters may be changed")
        for key, value in changes.items():
            if key == "model_overrides":
                if set(value) != {"private_dropout"}:
                    raise ValueError("Only PRIVATE dropout is configurable inside model_overrides")
                dropout = value["private_dropout"]
                if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout):
                    raise ValueError("PRIVATE dropout must be a finite number")
                raw["tasks"][task].setdefault(key, {})["private_dropout"] = float(dropout)
            elif key in FIELDS:
                if key.endswith("epochs"):
                    if type(value) is not int:
                        raise ValueError("PRIVATE epoch budgets must be integers")
                elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("PRIVATE learning rate must be a finite number")
                raw["tasks"][task][key] = float(value) if key.endswith("lr") else value
            else:
                raise ValueError(f"Parameter is outside the WeakOpt whitelist: {key}")
    validate_recipe(base, raw)
    return raw


def baseline_snapshot():
    """Reuse the public audited paper reader, requesting validation only."""
    from benchmarks.common.paper import build_paper_summary

    payload, _ = build_paper_summary(BASE, BASE_RUN, BASELINES, ROOT, with_test=False)
    files = {}
    for name, source in payload["sources"].items():
        for filename, field in (("metadata.json", "metadata_sha256"), ("summary.json", "summary_sha256")):
            files[name + "/" + filename] = source[field]
        for prediction in source["predictions"]:
            files[name + "/" + prediction["path"]] = prediction["sha256"]
    return {"folds": payload["sections"]["valid"]["folds"],
            "comparison_identity": payload["comparison_identity"], "sources": files}


def validate_data_sources(snapshot, base):
    from stage3.data import resolve_task_registry, source_path

    config = stage3_config_from_dict(base)
    registry = resolve_task_registry(config)
    reference = next(r for r in snapshot["folds"] if r["model_id"] == "ilume")
    summary = read_json(checked_path(ROOT, reference["source_run"]) / "summary.json")
    sources = summary["reporting"]["comparison_identity"]["payload"]["sources"]
    for task, spec in registry.items():
        for fold in range(1, 6):
            if sha256_file(source_path(config, spec, fold)) != sources[f"{task}:fold{fold}"]:
                raise ValueError("Current training/validation data differ from the frozen baseline sources")


def load_campaign(*, verify_sources=True):
    manifest = read_json(CONFIGS / "campaign.json")
    verify_files(manifest["files"])
    base = read_yaml(CONFIGS / "reference.yaml")
    seen = {fingerprint(base)}
    if len(manifest["trials"]) > LIMIT:
        raise ValueError("WeakOpt exceeds its 80-candidate budget")
    for index, trial in enumerate(manifest["trials"], 1):
        if trial["id"] != f"{index:03}":
            raise ValueError("Candidate IDs must be contiguous and immutable")
        path = CONFIGS / "trials" / (trial["id"] + ".yaml")
        if sha256_file(path) != trial["sha256"]:
            raise ValueError("Frozen candidate configuration was modified")
        raw = read_yaml(path)
        validate_recipe(base, raw)
        identity = fingerprint(raw)
        if identity != trial["recipe_hash"] or identity in seen:
            raise ValueError("Duplicate or altered candidate identity")
        seen.add(identity)
    if {p.stem for p in (CONFIGS / "trials").glob("*.yaml")} != {t["id"] for t in manifest["trials"]}:
        raise ValueError("Unregistered candidate configurations must not be ignored")
    if verify_sources:
        verify_files(manifest["sources"])
        snapshot = read_json(OUTPUT / "reference/baselines.json")
        verify_files(snapshot["sources"])
        validate_data_sources(snapshot, base)
    return manifest, base


def generate(proposal_path=None):
    CONFIGS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with artifact_lock(OUTPUT / "locks/campaign.lock"):
        campaign = CONFIGS / "campaign.json"
        if not campaign.exists():
            if (CONFIGS / "reference.yaml").exists():
                raise FileExistsError("Incomplete campaign retained; inspect it before continuing")
            print("Auditing validation baseline sources; test predictions are excluded.", flush=True)
            snapshot = baseline_snapshot()
            base = read_yaml(BASE)
            reference = OUTPUT / "reference"
            reference.mkdir(exist_ok=False)
            atomic_json(reference / "baselines.json", snapshot)
            write_yaml(CONFIGS / "reference.yaml", base)
            sources = {relative(BASE): sha256_file(BASE)}
            for key in ("stage1_encoder", "stage2_final"):
                path = ROOT / base["initialization"][key]
                sources[relative(path)] = sha256_file(path)
            for path in (ROOT / base["data"]["artifacts_dir"] / "metadata.json",
                         ROOT / base["preparation"]["data_metadata_path"]):
                sources[relative(path)] = sha256_file(path)
            manifest = {"version": 1, "limit": LIMIT, "trials": [], "sources": sources,
                        "files": {relative(CONFIGS / "reference.yaml"): sha256_file(CONFIGS / "reference.yaml"),
                                  relative(reference / "baselines.json"): sha256_file(reference / "baselines.json")}}
            atomic_json(campaign, manifest)
        manifest, base = load_campaign()
        if proposal_path:
            proposals = read_yaml(proposal_path)
            if not isinstance(proposals, list) or not proposals:
                raise ValueError("Proposal YAML must contain a nonempty list")
        elif manifest["trials"]:
            print(f"Existing campaign verified: {len(manifest['trials'])} candidates; no new trial generated.")
            return
        else:
            proposals = first_batch(base)
        if len(manifest["trials"]) + len(proposals) > LIMIT:
            raise ValueError("Proposal exceeds the total budget of 80 new configurations")
        seen = {fingerprint(base), *(t["recipe_hash"] for t in manifest["trials"])}
        pending = []
        for proposal in proposals:
            raw = apply_overrides(base, proposal)
            digest = fingerprint(raw)
            if digest in seen:
                raise ValueError("Duplicate candidate or unchanged reference recipe")
            seen.add(digest)
            pending.append((proposal, raw, digest))
        for proposal, raw, digest in pending:
            trial = f"{len(manifest['trials']) + 1:03}"
            path = CONFIGS / "trials" / (trial + ".yaml")
            path.parent.mkdir(exist_ok=True)
            if path.exists():
                raise FileExistsError("Unregistered candidate retained; refusing to overwrite it")
            write_yaml(path, raw)
            manifest["trials"].append({"id": trial, "description": proposal["description"],
                "recipe_hash": digest, "sha256": sha256_file(path), "overrides": proposal["overrides"]})
            atomic_json(campaign, manifest)
        print(f"Frozen candidates: {len(manifest['trials'])}/{LIMIT}")


def devices_and_folds(devices, maximum, folds):
    slots = tuple(devices.split(","))
    if not slots or len(set(slots)) != len(slots) or not set(slots) <= {"cuda:2", "cuda:3"}:
        raise ValueError("WeakOpt only permits physical cuda:2 and cuda:3, without duplicates")
    if maximum < 1 or maximum > 3 * len(slots):
        raise ValueError("Use at most three concurrent jobs per selected GPU")
    if not folds or len(set(folds)) != len(folds) or not set(folds) <= set(range(1, 6)):
        raise ValueError("Specify unique folds between 1 and 5")
    return slots


def selected_trials(manifest, ids):
    requested = set(ids or [t["id"] for t in manifest["trials"]])
    if requested - {t["id"] for t in manifest["trials"]}:
        raise ValueError("Unknown WeakOpt candidate ID")
    return [t for t in manifest["trials"] if t["id"] in requested]


@contextmanager
def gpu_slot(device):
    """Reserve one of three slots, retaining both previous GPU lock names."""
    directory = OUTPUT / "locks"
    directory.mkdir(parents=True, exist_ok=True)
    name = device.replace(":", "-")
    while True:
        for filename in (f"{name}.lock", f"{name}-slot2.lock", f"{name}-slot3.lock"):
            handle = (directory / filename).open("a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()
            return
        time.sleep(0.1)


def lock_busy(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def fill_slots(children, failed):
    """Inspect scheduling locks and completion markers without reading metrics."""
    for tid, (process, device) in list(children.items()):
        code = process.poll()
        if code is not None:
            if code:
                failed.add(tid)
            del children[tid]
    manifest, _ = load_campaign(verify_sources=False)
    launched = []
    occupied = {}
    for device in ("cuda:2", "cuda:3"):
        name = device.replace(":", "-")
        used = sum(lock_busy(OUTPUT / "locks" / filename) for filename in
                   (f"{name}.lock", f"{name}-slot2.lock", f"{name}-slot3.lock"))
        starting = sum(assigned == device and not lock_busy(OUTPUT / f"locks/trial-{tid}.lock")
                       for tid, (_, assigned) in children.items())
        occupied[device] = used
        for _ in range(max(0, 3 - used - starting)):
            for trial in manifest["trials"]:
                tid = trial["id"]
                root = OUTPUT / "trials" / tid
                if tid in children or tid in failed or lock_busy(OUTPUT / f"locks/trial-{tid}.lock"):
                    continue
                if (root / "status.json").exists() and read_json(root / "status.json").get("status") == "failed":
                    continue
                if all(any(root.glob(f"valid*/fold{fold}/hpo_evaluation.json")) for fold in (1, 3)):
                    continue
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
                log = OUTPUT / "watch" / f"{stamp}-{tid}.log"
                log.parent.mkdir(exist_ok=True)
                env = dict(os.environ)
                env.pop("CUDA_VISIBLE_DEVICES", None)
                with log.open("x") as handle:
                    process = subprocess.Popen([sys.executable, "-u", "scripts/HPO/weakopt.py", "run",
                        "--ids", tid, "--fold", "1", "3", "--devices", device, "--max-parallel", "1"],
                        cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=handle,
                        stderr=subprocess.STDOUT, start_new_session=True)
                children[tid] = process, device
                launched.append({"id": tid, "device": device, "pid": process.pid})
                break
            else:
                break
    return {"time_utc": datetime.now(timezone.utc).isoformat(), "occupied_slots": occupied,
            "launched": launched, "failed_ids": sorted(failed)}


def watch():
    children, failed = {}, set()
    with artifact_lock(OUTPUT / "locks/watch.lock"):
        while not (OUTPUT / "selection.json").exists():
            record = fill_slots(children, failed)
            with (OUTPUT / "watch.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            time.sleep(15 * 60)


def run_trial(trial, folds, device, dry_run=False):
    config = CONFIGS / "trials" / (trial["id"] + ".yaml")
    root = OUTPUT / "trials" / trial["id"]
    if dry_run:
        print(f"DRY RUN {trial['id']}: folds={folds}, physical device={device}, output={relative(root)}")
        return
    with artifact_lock(OUTPUT / f"locks/trial-{trial['id']}.lock"), gpu_slot(device):
        root.mkdir(parents=True, exist_ok=True)
        identity_path = root / "trial.json"
        identity = {"recipe_hash": trial["recipe_hash"], "config_sha256": trial["sha256"]}
        if identity_path.exists() and read_json(identity_path) != identity:
            raise ValueError("Existing trial belongs to another frozen configuration")
        atomic_json(identity_path, identity)
        state = {"status": "running", "folds_requested": list(folds), "device": device}
        atomic_json(root / "status.json", state)
        try:
            arguments = ["--config", relative(config), "--fold", *folds, "--output", relative(root / "train"),
                         "--devices", device, "--max-parallel", 1, "--resume"]
            native("stage3/train", arguments, root / "train.log", device)
            job = Job(trial["id"], "WeakOpt", root, config3=config, folds=tuple(folds))
            for fold in folds:
                evaluate_fold(job, fold, device, resume=True)
            state["status"] = "completed"
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            atomic_json(root / "status.json", state)


def validation_rows(trial, folds, baseline, *, configs=None, output=None):
    """Require original evaluation identities and every declared file checksum."""
    from stage3.evaluate import resolve_stage3_evaluation_identity

    root = (OUTPUT if output is None else output) / "trials" / trial["id"]
    config = load_stage3_config((CONFIGS if configs is None else configs) / "trials" / (trial["id"] + ".yaml"))
    rows = []
    scales = {(r["task"], int(r["fold"])): r for r in baseline["folds"] if r["model_id"] == "ilume"}
    for fold in folds:
        markers = sorted(root.glob(f"valid*/fold{fold}/hpo_evaluation.json"))
        if not markers:
            return None
        expected = resolve_stage3_evaluation_identity(config, root / "train", split="valid", ensemble_folds=False, fold=fold)
        chosen = None
        for marker in markers:
            saved = read_json(marker)
            if saved["identity"] != expected:
                raise ValueError("Candidate evaluation identity differs from its frozen checkpoint")
            if not {"summary.json", "metadata.json"} <= set(saved.get("files", {})):
                raise ValueError("Validation marker lacks required metadata and summary checksums")
            verify_files(saved["files"], marker.parent)
            chosen = marker.parent
        summary = read_json(chosen / "summary.json")
        source = scales[next(iter(config.tasks)), fold]["source_run"]
        original = read_json(checked_path(ROOT, source) / "summary.json")
        comparison = summary["reporting"]["comparison_identity"]
        validate_semantic_identity(comparison)
        if comparison != original["reporting"]["comparison_identity"]:
            raise ValueError("Candidate and baseline comparison identities differ")
        for task, values in summary["tasks"].items():
            reference = scales[task, fold]
            if values["count"] != reference["count"] or not math.isfinite(values["mae"]):
                raise ValueError("Candidate validation coverage or metrics differ from the frozen source")
            predictions = [p for p in summary["reporting"]["predictions"] if p["task"] == task]
            if len(predictions) != 1:
                raise ValueError("Expected one validation prediction file per task")
            path = checked_path(chosen, predictions[0]["path"])
            if sha256_file(path) != predictions[0]["sha256"]:
                raise ValueError("Validation prediction SHA256 mismatch")
            with path.open(newline="") as handle:
                points = list(csv.DictReader(handle))
            errors = [abs(float(p["target"]) - float(p["prediction"])) for p in points]
            if len(errors) != reference["count"] or not all(math.isfinite(e) for e in errors):
                raise ValueError("Prediction coverage or values differ from the frozen reference")
            mae = statistics.mean(errors)
            rows.append({"task": task, "fold": fold, "mae": mae,
                         "normalized_mae": mae / reference["common_scale"]})
    if len(rows) != len(config.tasks) * len(folds):
        raise ValueError("Candidate evaluation must cover every experimental task")
    return rows


def score_rows(rows, baseline, folds):
    selected = [r for r in baseline["folds"] if int(r["fold"]) in folds]
    tasks = sorted({r["task"] for r in selected})
    details = []
    for task in tasks:
        own = [r for r in rows if r["task"] == task]
        if len(own) != len(folds):
            raise ValueError("Ranking requires complete requested fold coverage")
        competitors = {}
        for row in selected:
            if row["task"] == task and row["model_id"] != "ilume":
                competitors.setdefault(row["model_id"], []).append(row["mae"])
        if len(competitors) != 10 or any(len(v) != len(folds) for v in competitors.values()):
            raise ValueError("Ranking requires ten baselines on the same requested folds")
        mae = statistics.mean(r["mae"] for r in own)
        best = min(statistics.mean(v) for v in competitors.values())
        old = statistics.mean(r["mae"] for r in selected if r["task"] == task and r["model_id"] == "ilume")
        details.append({"task": task, "mae": mae, "baseline_best_mae": best, "reference_mae": old,
                        "rank": 1 + sum(statistics.mean(v) < mae for v in competitors.values()),
                        "normalized_mae": statistics.mean(r["normalized_mae"] for r in own),
                        "mae_change_percent": 100 * (mae / old - 1) if old else None})
    return {"wins": sum(r["rank"] == 1 for r in details),
            "macro_normalized_mae": statistics.mean(r["normalized_mae"] for r in details),
            "tasks": details}


def report(folds, ids=None):
    manifest, _ = load_campaign()
    baseline = read_json(OUTPUT / "reference/baselines.json")
    reference = [{k: r[k] for k in ("task", "fold", "mae", "normalized_mae")}
                 for r in baseline["folds"] if r["model_id"] == "ilume" and int(r["fold"]) in folds]
    result = {"folds": list(folds), "reference": score_rows(reference, baseline, folds), "ranked": [], "incomplete": []}
    for trial in selected_trials(manifest, ids):
        rows = validation_rows(trial, folds, baseline)
        if rows is None:
            result["incomplete"].append(trial["id"])
        else:
            result["ranked"].append({"id": trial["id"], **score_rows(rows, baseline, folds)})
    result["ranked"].sort(key=lambda r: (-r["wins"], r["macro_normalized_mae"], r["id"]))
    result["validation_target_met"] = set(folds) == set(range(1, 6)) and any(r["wins"] >= 15 for r in result["ranked"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    root = OUTPUT / "reports" / stamp
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "ranking.json", result)
    with (root / "ranking.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "wins", "macro_normalized_mae"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in result["ranked"])
    print(f"Validation report: {relative(root)}; complete={len(result['ranked'])}, incomplete={len(result['incomplete'])}")
    for row in result["ranked"]:
        print(f"{row['id']}: wins={row['wins']}/24, nMAE={row['macro_normalized_mae']:.6f}")
    return result, root


def finalize(trial_id):
    with artifact_lock(OUTPUT / "locks/finalize.lock"):
        if (OUTPUT / "selection.json").exists():
            raise FileExistsError("Final selection is already locked")
        result, root = report((1, 2, 3, 4, 5))
        if not result["ranked"] or result["ranked"][0]["id"] != trial_id:
            raise ValueError("Final selection must be the highest ranked complete five-fold candidate")
        atomic_json(OUTPUT / "selection.json", {"id": trial_id, "validation_report": relative(root),
                    "report_sha256": sha256_file(root / "ranking.json"), "validation": result["ranked"][0],
                    "test_policy": "One locked five-model ensemble; test metrics never select trials"})
        print(f"Locked trial {trial_id}; no test evaluation was launched.")


def main():
    parser = argparse.ArgumentParser(description="WeakOpt: immutable Stage3 parameter trials; validation-only selection.")
    sub = parser.add_subparsers(dest="command", required=True)
    generate_parser = sub.add_parser("generate", help="Freeze the reference and append static candidate configurations.")
    generate_parser.add_argument("--proposal", type=Path, help="A YAML list of described task-parameter overrides.")
    run = sub.add_parser("run", help="Train and validate requested folds on physical GPU2/3 or a subset.")
    run.add_argument("--ids", nargs="+")
    run.add_argument("--fold", type=int, nargs="+", default=[1, 3])
    run.add_argument("--devices", default="cuda:2,cuda:3")
    run.add_argument("--max-parallel", type=int, default=6)
    run.add_argument("--resume", action="store_true", help="Accepted for explicitness; native strict resume is always used.")
    run.add_argument("--dry-run", action="store_true")
    summary = sub.add_parser("report", help="Rank complete validation trials; never read test metrics.")
    summary.add_argument("--fold", type=int, nargs="+", default=[1, 3])
    summary.add_argument("--ids", nargs="+")
    final = sub.add_parser("finalize", help="Lock an explicitly selected five-fold validation winner.")
    final.add_argument("--id", required=True)
    sub.add_parser("watch", help="Check slots every 15 minutes and fill pending frozen two-fold trials.")
    args = parser.parse_args()
    os.chdir(ROOT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT / "tmp"
    temporary.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(temporary)
    import tempfile
    import torch
    tempfile.tempdir = str(temporary)
    torch.set_num_threads(1)
    if args.command == "generate":
        generate(args.proposal)
    elif args.command == "run":
        devices = devices_and_folds(args.devices, args.max_parallel, args.fold)
        manifest, _ = load_campaign()
        trials = {trial["id"]: trial for trial in selected_trials(manifest, args.ids)}
        jobs = [Job(tid, "WeakOpt", OUTPUT / "trials" / tid) for tid in trials]
        failures = run_queue(jobs, devices, args.max_parallel,
            lambda job, device: run_trial(trials[job.trial], args.fold, device, args.dry_run), record_failures=False)
        return int(bool(failures))
    elif args.command == "report":
        devices_and_folds("cuda:2,cuda:3", 2, args.fold)
        report(args.fold, args.ids)
    elif args.command == "watch":
        watch()
    else:
        finalize(args.id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
