"""Immutable Stage2/Stage3 parameter pairs with validation-only selection."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

import weakopt
from _common import Job, evaluate_fold, native, run_queue, stop_children, train_stage2
from _configs import fingerprint, read_yaml, validate_pair, write_yaml
from common.artifact_cache import artifact_lock
from common.io import atomic_json, sha256_file
from stage2.home_config import home_recipe_from_dict

ROOT = weakopt.ROOT
CONFIGS = ROOT / "configs/HPO/WeakOpt/Joint"
OUTPUT = ROOT / "outputs/HPO/WeakOpt/Joint"
BASE2 = ROOT / "configs/v4/stage2/optimized_3407.yaml"
FIELDS2 = {"epochs", "task_head_learning_rate", "weight_decay", "warmup_fraction"}
LIMIT = 80


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def read(path):
    return json.loads(Path(path).read_text())


def stage2_recipe(base, changes):
    if not isinstance(changes, dict) or set(changes) - FIELDS2:
        raise ValueError("Only Stage2 epochs, HoME learning rate, weight decay and warmup may change")
    raw = deepcopy(base)
    raw["training"].update(changes)
    home_recipe_from_dict(raw)
    return raw


def normalized_stage3(raw, base):
    value = deepcopy(raw)
    value["initialization"]["stage2_final"] = base["initialization"]["stage2_final"]
    return value


def pair_hash(s2, s3, base3):
    return fingerprint([s2, normalized_stage3(s3, base3)])


def load(*, verify_sources=True):
    manifest = read(CONFIGS / "campaign.json")
    weakopt.verify_files(manifest["files"])
    base2, base3 = read_yaml(CONFIGS / "reference-stage2.yaml"), read_yaml(CONFIGS / "reference-stage3.yaml")
    if manifest["limit"] != LIMIT or len(manifest["trials"]) > LIMIT:
        raise ValueError("Joint WeakOpt exceeds its 80-pair budget")
    stages = manifest["stage2"]
    seen2 = set()
    for index, item in enumerate(stages, 1):
        if item["id"] != f"S{index:03}":
            raise ValueError("Stage2 IDs must be contiguous")
        path = CONFIGS / "stage2" / (item["id"] + ".yaml")
        if sha256_file(path) != item["sha256"]:
            raise ValueError("Frozen Stage2 configuration was modified")
        raw = read_yaml(path)
        changes = {k: raw["training"][k] for k in FIELDS2}
        if stage2_recipe(base2, changes) != raw or fingerprint(raw) != item["recipe_hash"]:
            raise ValueError("Stage2 configuration changes protected parameters")
        if item["recipe_hash"] in seen2:
            raise ValueError("Duplicate Stage2 recipes must share one canonical source")
        seen2.add(item["recipe_hash"])
    sources = {s["id"]: s for s in stages}
    seen = {pair_hash(base2, base3, base3)}
    for index, trial in enumerate(manifest["trials"], 1):
        if trial["id"] != f"{index:03}" or trial["stage2_id"] not in sources:
            raise ValueError("Candidate ID or Stage2 dependency is invalid")
        path = CONFIGS / "trials" / (trial["id"] + ".yaml")
        if sha256_file(path) != trial["sha256"]:
            raise ValueError("Frozen Stage3 configuration was modified")
        raw = read_yaml(path)
        expected = relative(OUTPUT / "stage2" / trial["stage2_id"] / "train/stage2_final.pt")
        if raw["initialization"]["stage2_final"] != expected:
            raise ValueError("Stage3 source differs from its registered Stage2 dependency")
        weakopt.validate_recipe(base3, normalized_stage3(raw, base3))
        s2 = read_yaml(CONFIGS / "stage2" / (trial["stage2_id"] + ".yaml"))
        validate_pair(s2, raw)
        digest = pair_hash(s2, raw, base3)
        if digest != trial["recipe_hash"] or digest in seen:
            raise ValueError("Duplicate or altered joint recipe")
        seen.add(digest)
    for folder, items in (("stage2", stages), ("trials", manifest["trials"])):
        if {p.stem for p in (CONFIGS / folder).glob("*.yaml")} != {x["id"] for x in items}:
            raise ValueError("Unregistered configurations must not be ignored")
    if verify_sources:
        weakopt.verify_files(manifest["sources"])
        snapshot = read(OUTPUT / "reference/baselines.json")
        weakopt.verify_files(snapshot["sources"])
        weakopt.validate_data_sources(snapshot, base3)
    return manifest, base2, base3


def first_batch():
    prior = read(weakopt.CONFIGS / "campaign.json")
    selected = next(t["overrides"] for t in prior["trials"] if t["id"] == "038")
    return [{"description": f"Stage2 epochs={epochs}, lr={lr}; Stage3 {label}",
             "stage2": {"epochs": epochs, "task_head_learning_rate": lr}, "stage3": overrides}
            for epochs in (5, 15, 20) for lr in (6e-5, 1e-4, 2e-4, 3e-4)
            for label, overrides in (("reference", {}), ("previous validation winner 038", selected))]


def generate(proposal=None):
    CONFIGS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with artifact_lock(OUTPUT / "locks/campaign.lock"):
        path = CONFIGS / "campaign.json"
        if not path.exists():
            previous, base3 = weakopt.load_campaign()
            base2 = read_yaml(BASE2)
            (OUTPUT / "reference").mkdir(exist_ok=False)
            atomic_json(OUTPUT / "reference/baselines.json", read(weakopt.OUTPUT / "reference/baselines.json"))
            write_yaml(CONFIGS / "reference-stage2.yaml", base2)
            write_yaml(CONFIGS / "reference-stage3.yaml", base3)
            sources = dict(previous["sources"])
            for p in (BASE2, weakopt.CONFIGS / "campaign.json",
                      ROOT / base2["data"]["artifacts_dir"] / "metadata.json",
                      ROOT / base2["preparation"]["data_metadata_path"]):
                sources[relative(p)] = sha256_file(p)
            files = {relative(p): sha256_file(p) for p in (
                CONFIGS / "reference-stage2.yaml", CONFIGS / "reference-stage3.yaml",
                OUTPUT / "reference/baselines.json")}
            atomic_json(path, {"version": 1, "limit": LIMIT, "trials": [], "stage2": [],
                               "files": files, "sources": sources})
        manifest, base2, base3 = load()
        if proposal is None and manifest["trials"]:
            print(f"Verified {len(manifest['trials'])} frozen joint candidates")
            return
        proposals = read_yaml(proposal) if proposal else first_batch()
        if not isinstance(proposals, list) or not proposals:
            raise ValueError("Proposals must be a nonempty YAML list")
        if len(manifest["trials"]) + len(proposals) > LIMIT:
            raise ValueError("Proposal exceeds the 80-pair budget")
        seen = {pair_hash(base2, base3, base3), *(t["recipe_hash"] for t in manifest["trials"])}
        pending = []
        for p in proposals:
            if set(p) != {"description", "stage2", "stage3"}:
                raise ValueError("Each proposal requires description, stage2 and stage3 fields")
            s2 = stage2_recipe(base2, p["stage2"])
            s3 = weakopt.apply_overrides(base3, {"overrides": p["stage3"]})
            validate_pair(s2, s3)
            digest = pair_hash(s2, s3, base3)
            if digest in seen:
                raise ValueError("Duplicate joint candidate or unchanged reference pair")
            seen.add(digest)
            pending.append((p, s2, s3, digest))
        for p, s2, s3, digest in pending:
            source = next((s for s in manifest["stage2"] if s["recipe_hash"] == fingerprint(s2)), None)
            if source is None:
                sid = f"S{len(manifest['stage2']) + 1:03}"
                config = CONFIGS / "stage2" / (sid + ".yaml")
                if config.exists():
                    raise FileExistsError("Unregistered Stage2 configuration retained")
                write_yaml(config, s2)
                source = {"id": sid, "sha256": sha256_file(config), "recipe_hash": fingerprint(s2)}
                manifest["stage2"].append(source)
            tid = f"{len(manifest['trials']) + 1:03}"
            s3["initialization"]["stage2_final"] = relative(OUTPUT / "stage2" / source["id"] / "train/stage2_final.pt")
            config = CONFIGS / "trials" / (tid + ".yaml")
            if config.exists():
                raise FileExistsError("Unregistered Stage3 configuration retained")
            write_yaml(config, s3)
            manifest["trials"].append({"id": tid, "stage2_id": source["id"], "recipe_hash": digest,
                "sha256": sha256_file(config), "description": p["description"], "overrides": p})
            atomic_json(path, manifest)
        print(f"Frozen {len(manifest['trials'])}/80 pairs; {len(manifest['stage2'])} unique Stage2 sources")


def run_stage2(source, device, dry_run=False):
    sid = source["id"]
    if dry_run:
        print(f"DRY RUN Stage2 {sid}: physical {device}")
        return
    with weakopt.gpu_slot(device):
        path = OUTPUT / "stage2" / sid
        path.mkdir(parents=True, exist_ok=True)
        job = Job(sid, "WeakOptJoint", path, config2=CONFIGS / "stage2" / (sid + ".yaml"),
                  stage2_output=path / "train")
        train_stage2(job, device, resume=True)


def run_trial(trial, folds, device, dry_run=False):
    tid = trial["id"]
    if dry_run:
        print(f"DRY RUN pair {tid}: Stage2 {trial['stage2_id']}, folds={folds}, physical {device}")
        return
    with artifact_lock(OUTPUT / f"locks/trial-{tid}.lock"), weakopt.gpu_slot(device):
        path = OUTPUT / "trials" / tid
        path.mkdir(parents=True, exist_ok=True)
        source = OUTPUT / "stage2" / trial["stage2_id"] / "train"
        binding = {"recipe_hash": trial["recipe_hash"], "config_sha256": trial["sha256"],
                   "stage2_final_sha256": sha256_file(source / "stage2_final.pt"),
                   "stage2_manifest_sha256": sha256_file(source / "stage2_final.json")}
        marker = path / "trial.json"
        if marker.exists() and read(marker) != binding:
            raise ValueError("Existing Stage3 trial has another configuration or Stage2 source")
        atomic_json(marker, binding)
        state = {"status": "running", "folds_requested": list(folds), "device": device}
        atomic_json(path / "status.json", state)
        try:
            config = CONFIGS / "trials" / (tid + ".yaml")
            native("stage3/train", ["--config", relative(config), "--fold", *folds,
                "--output", relative(path / "train"), "--devices", device, "--max-parallel", 1, "--resume"],
                path / "train.log", device)
            job = Job(tid, "WeakOptJoint", path, config3=config, folds=tuple(folds))
            for fold in folds:
                evaluate_fold(job, fold, device, resume=True)
            state["status"] = "completed"
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            atomic_json(path / "status.json", state)


def report(folds):
    manifest, _, _ = load()
    baseline = read(OUTPUT / "reference/baselines.json")
    ref = [{k: r[k] for k in ("task", "fold", "mae", "normalized_mae")}
           for r in baseline["folds"] if r["model_id"] == "ilume" and int(r["fold"]) in folds]
    result = {"folds": list(folds), "reference": weakopt.score_rows(ref, baseline, folds),
              "ranked": [], "incomplete": []}
    for trial in manifest["trials"]:
        rows = weakopt.validation_rows(trial, folds, baseline, configs=CONFIGS, output=OUTPUT)
        if rows is None:
            result["incomplete"].append(trial["id"])
        else:
            result["ranked"].append({"id": trial["id"], "stage2_id": trial["stage2_id"],
                                     **weakopt.score_rows(rows, baseline, folds)})
    result["ranked"].sort(key=lambda r: (-r["wins"], r["macro_normalized_mae"], r["id"]))
    result["validation_target_met"] = set(folds) == set(range(1, 6)) and any(r["wins"] >= 15 for r in result["ranked"])
    path = OUTPUT / "reports" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    path.mkdir(parents=True, exist_ok=False)
    atomic_json(path / "ranking.json", result)
    print(f"Validation report: {relative(path)}; complete={len(result['ranked'])}")
    for row in result["ranked"]:
        print(f"{row['id']}: wins={row['wins']}/24; nMAE={row['macro_normalized_mae']:.6f}")
    return result, path


def finalize(tid):
    with artifact_lock(OUTPUT / "locks/finalize.lock"):
        if (OUTPUT / "selection.json").exists():
            raise FileExistsError("Final joint selection is already locked")
        result, path = report((1, 2, 3, 4, 5))
        if not result["ranked"] or result["ranked"][0]["id"] != tid:
            raise ValueError("Select the highest ranked complete five-fold joint candidate")
        atomic_json(OUTPUT / "selection.json", {"id": tid, "validation_report": relative(path),
            "report_sha256": sha256_file(path / "ranking.json"), "validation": result["ranked"][0],
            "test_policy": "One locked five-model ensemble; no test-based selection"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate", help="Freeze or append immutable Stage2/Stage3 pairs")
    p.add_argument("--proposal", type=Path)
    p = sub.add_parser("run", help="Train unique Stage2 sources, then requested Stage3 folds")
    p.add_argument("--ids", nargs="+")
    p.add_argument("--fold", type=int, nargs="+", default=[1, 3])
    p.add_argument("--devices", default="cuda:2,cuda:3")
    p.add_argument("--max-parallel", type=int, default=6)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("report", help="Audit and rank validation only")
    p.add_argument("--fold", type=int, nargs="+", default=[1, 3])
    p = sub.add_parser("finalize", help="Lock a complete five-fold validation winner")
    p.add_argument("--id", required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    (OUTPUT / "tmp").mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(OUTPUT / "tmp")
    tempfile.tempdir = str(OUTPUT / "tmp")
    import torch
    torch.set_num_threads(1)
    if args.command == "generate":
        generate(args.proposal)
    elif args.command == "run":
        devices = weakopt.devices_and_folds(args.devices, args.max_parallel, args.fold)
        manifest, _, _ = load()
        trials = weakopt.selected_trials(manifest, args.ids)
        required = {t["stage2_id"] for t in trials}
        sources = {s["id"]: s for s in manifest["stage2"] if s["id"] in required}
        try:
            jobs = [Job(sid, "WeakOptJoint", OUTPUT / "stage2" / sid) for sid in sources]
            failures = run_queue(jobs, devices, args.max_parallel,
                lambda job, device: run_stage2(sources[job.trial], device, args.dry_run), record_failures=False)
            if failures:
                raise RuntimeError(f"Stage2 dependencies failed: {failures}")
            by_id = {t["id"]: t for t in trials}
            jobs = [Job(tid, "WeakOptJoint", OUTPUT / "trials" / tid) for tid in by_id]
            failures = run_queue(jobs, devices, args.max_parallel,
                lambda job, device: run_trial(by_id[job.trial], args.fold, device, args.dry_run), record_failures=False)
            if failures:
                raise RuntimeError(f"Joint trials failed: {failures}")
        except KeyboardInterrupt:
            stop_children()
            raise
    elif args.command == "report":
        weakopt.devices_and_folds("cuda:2,cuda:3", 6, args.fold)
        report(tuple(args.fold))
    else:
        finalize(args.id)


if __name__ == "__main__":
    main()
