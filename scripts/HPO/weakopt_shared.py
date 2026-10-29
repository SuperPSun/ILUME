"""Bounded Stage3 shared-owner trials with validation-only selection."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import tempfile

import weakopt
import weakopt_joint as joint
from _common import Job, run_queue, stop_children
from _configs import fingerprint, read_yaml, write_yaml
from common.artifact_cache import artifact_lock
from common.io import atomic_json, sha256_file
from stage3.config import stage3_config_from_dict

ROOT = weakopt.ROOT
CONFIGS = ROOT / "configs/HPO/WeakOpt/Shared"
OUTPUT = ROOT / "outputs/HPO/WeakOpt/Shared"
BASE = joint.CONFIGS / "trials/068.yaml"
LIMIT = 48
PROMOTIONS = 8


def validate_recipe(base, raw):
    """Protect topology and algorithms while allowing explicit owner budgets."""
    remainder = deepcopy(raw)
    owners = [(raw["training"]["three_phase"]["global"],
               remainder["training"]["three_phase"]["global"],
               base["training"]["three_phase"]["global"])]
    active = {task["meta_group"] for task in base["tasks"].values()}
    for group in active:
        for phase in ("phase1", "phase2"):
            owners.append((raw["groups"][group][phase], remainder["groups"][group][phase],
                           base["groups"][group][phase]))
    for changed, restored, original in owners:
        for key in ("lr", "epochs"):
            value = changed[key]
            if key == "epochs":
                valid = type(value) is int and 1 <= value <= 30
            else:
                valid = type(value) in (int, float) and math.isfinite(value) and value > 0
            if not valid:
                raise ValueError("Owner learning rates must be finite and epochs must be integers in [1, 30]")
            restored[key] = original[key]
    weakopt.validate_recipe(base, remainder)
    for task in raw["tasks"].values():
        for key in weakopt.FIELDS - {"phase1_private_lr"}:
            if task.get(key, 0) > 30:
                raise ValueError("PRIVATE budgets cannot exceed 30 epochs")
    stage3_config_from_dict(raw)


def apply(base, proposal):
    if set(proposal) != {"description", "private", "global", "groups"}:
        raise ValueError("Each proposal requires description, private, global and groups")
    raw = weakopt.apply_overrides(base, {"overrides": proposal["private"]})
    if set(proposal["global"]) - {"lr", "epochs"}:
        raise ValueError("Only GLOBAL lr and epochs may change")
    raw["training"]["three_phase"]["global"].update(proposal["global"])
    active = {task["meta_group"] for task in base["tasks"].values()}
    for group, phases in proposal["groups"].items():
        if group not in active or set(phases) - {"phase1", "phase2"}:
            raise ValueError("Only active GROUP phase1/phase2 budgets may change")
        for phase, changes in phases.items():
            if set(changes) - {"lr", "epochs"}:
                raise ValueError("Only GROUP lr and epochs may change")
            raw["groups"][group][phase].update(changes)
    validate_recipe(base, raw)
    return raw


def first_batch(base):
    rollback = {"experiment/" + t: {"phase3_private_epochs": n} for t, n in {
        "anodic_potential_limit": 2, "cathodic_potential_limit": 2,
        "surface_tension": 4, "melting_point": 4, "refractive_index": 2,
        "glass_transition_temperature": 2}.items()}
    def proposal(description, private=None, global_scope=None, groups=None):
        return {"description": description, "private": deepcopy(rollback if private is None else private),
                "global": global_scope or {}, "groups": groups or {}}
    batch = [proposal("Restore short PRIVATE Phase3 except density and viscosity"),
             proposal("Restore only potential Phase3", {t: v for t, v in rollback.items() if "potential" in t}),
             proposal("Short PRIVATE tails", {**rollback, "experiment/melting_point": {"phase3_private_epochs": 1},
                                              "experiment/refractive_index": {"phase3_private_epochs": 1}})]
    for factor in (.9, 1.2):
        batch.append(proposal(f"GLOBAL LR x{factor}", global_scope={"lr": base["training"]["three_phase"]["global"]["lr"] * factor}))
    batch.append(proposal("GLOBAL lifetime 20", global_scope={"epochs": 20}))
    for group, factors, epochs in [("phase_stability", (.85, 1.15), (12, 28)),
                                   ("thermophysical", (.75, 1.25), (8,)),
                                   ("dielectric_optical", (.75, 1.25), (6,))]:
        for factor in factors:
            changes = {phase: {"lr": base["groups"][group][phase]["lr"] * factor}
                       for phase in ("phase1", "phase2")}
            batch.append(proposal(f"{group} LR x{factor}", groups={group: changes}))
        for n in epochs:
            batch.append(proposal(f"{group} Phase2 lifetime {n}", groups={group: {"phase2": {"epochs": n}}}))
    return batch


def load(verify_sources=True):
    manifest = joint.read(CONFIGS / "campaign.json")
    if manifest["limit"] != LIMIT or manifest["promotions"] != PROMOTIONS or len(manifest["trials"]) > LIMIT:
        raise ValueError("Shared-owner campaign budget differs from its registered limit")
    weakopt.verify_files(manifest["files"])
    base = read_yaml(CONFIGS / "reference.yaml")
    seen = {fingerprint(base)}
    for index, trial in enumerate(manifest["trials"], 1):
        path = CONFIGS / "trials" / (trial["id"] + ".yaml")
        if trial["id"] != f"{index:03}" or sha256_file(path) != trial["sha256"]:
            raise ValueError("Frozen candidate configuration or ID was modified")
        raw = read_yaml(path)
        validate_recipe(base, raw)
        digest = fingerprint(raw)
        if digest != trial["recipe_hash"] or digest in seen:
            raise ValueError("Duplicate or altered shared-owner recipe")
        seen.add(digest)
    if {p.stem for p in (CONFIGS / "trials").glob("*.yaml")} != {t["id"] for t in manifest["trials"]}:
        raise ValueError("Unregistered configurations must not be ignored")
    if verify_sources:
        weakopt.verify_files(manifest["sources"])
        baseline = joint.read(OUTPUT / "reference/baselines.json")
        weakopt.verify_files(baseline["sources"])
        weakopt.validate_data_sources(baseline, base)
    return manifest, base


def generate(proposal=None):
    CONFIGS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with artifact_lock(OUTPUT / "locks/campaign.lock"):
        if (OUTPUT / 'selection.json').exists():
            raise ValueError("Selection is locked; additional candidates are forbidden")
        path = CONFIGS / "campaign.json"
        if not path.exists():
            previous, _, _ = joint.load()
            base = read_yaml(BASE)
            (OUTPUT / "reference").mkdir(exist_ok=False)
            atomic_json(OUTPUT / "reference/baselines.json", joint.read(joint.OUTPUT / "reference/baselines.json"))
            write_yaml(CONFIGS / "reference.yaml", base)
            source = ROOT / base["initialization"]["stage2_final"]
            sources = dict(previous["sources"])
            for p in (BASE, source, source.with_suffix('.json'), joint.CONFIGS / 'stage2/S030.yaml'):
                sources[joint.relative(p)] = sha256_file(p)
            files = {joint.relative(p): sha256_file(p) for p in (CONFIGS / "reference.yaml", OUTPUT / "reference/baselines.json")}
            atomic_json(path, {"version": 1, "limit": LIMIT, "promotions": PROMOTIONS,
                               "trials": [], "files": files, "sources": sources})
        manifest, base = load()
        if proposal is None and manifest["trials"]:
            print(f"Verified {len(manifest['trials'])} frozen candidates")
            return
        proposals = read_yaml(proposal) if proposal else first_batch(base)
        if not isinstance(proposals, list) or not proposals or len(manifest["trials"]) + len(proposals) > LIMIT:
            raise ValueError("Proposals must fit the 48-candidate budget")
        seen = {fingerprint(base), *(t["recipe_hash"] for t in manifest["trials"])}
        pending = []
        for p in proposals:
            raw = apply(base, p)
            digest = fingerprint(raw)
            if digest in seen:
                raise ValueError("Duplicate candidate or unchanged reference")
            seen.add(digest)
            pending.append((p, raw, digest))
        for p, raw, digest in pending:
            tid = f"{len(manifest['trials']) + 1:03}"
            config = CONFIGS / 'trials' / (tid + '.yaml')
            if config.exists():
                raise FileExistsError("Unregistered candidate retained")
            write_yaml(config, raw)
            manifest['trials'].append({'id': tid, 'stage2_id': 'S030', 'description': p['description'],
                'recipe_hash': digest, 'sha256': sha256_file(config), 'overrides': p})
            atomic_json(path, manifest)
        print(f"Frozen {len(manifest['trials'])}/48 candidates; existing Stage2 S030 remains fixed")


def reserve(trials, folds):
    """Bound unique fold requests across concurrent screening and promotion queues."""
    if set(folds) not in ({1, 3}, {2, 4, 5}, {1, 2, 3, 4, 5}):
        raise ValueError("Request screening folds 1/3 or promotion folds 2/4/5")
    with artifact_lock(OUTPUT / 'locks/budget.lock'):
        path = OUTPUT / 'budget.json'
        budget = joint.read(path) if path.exists() else {}
        if (OUTPUT / 'selection.json').exists():
            raise ValueError("Selection is locked; additional training is forbidden")
        updated = deepcopy(budget)
        for t in trials:
            tid = t['id']
            if set(folds) & {2, 4, 5}:
                if not all((OUTPUT / f'trials/{tid}/valid/fold{f}/hpo_evaluation.json').is_file() for f in (1, 3)):
                    raise ValueError("Promotion requires completed screening evaluations")
            updated[tid] = sorted(set(updated.get(tid, [])) | set(folds))
        if sum(bool(set(fs) & {2, 4, 5}) for fs in updated.values()) > PROMOTIONS or sum(map(len, updated.values())) > 120:
            raise ValueError("Promotion exceeds the eight-candidate or 120-fold budget")
        atomic_json(path, updated)


def report(folds):
    manifest, _ = load()
    baseline = joint.read(OUTPUT / 'reference/baselines.json')
    ref = [{k: r[k] for k in ('task', 'fold', 'mae', 'normalized_mae')}
           for r in baseline['folds'] if r['model_id'] == 'ilume' and int(r['fold']) in folds]
    result = {'folds': list(folds), 'reference': weakopt.score_rows(ref, baseline, folds), 'ranked': [], 'incomplete': []}
    for t in manifest['trials']:
        rows = weakopt.validation_rows(t, folds, baseline, configs=CONFIGS, output=OUTPUT)
        if rows is None:
            result['incomplete'].append(t['id'])
        else:
            result['ranked'].append({'id': t['id'], **weakopt.score_rows(rows, baseline, folds)})
    result['ranked'].sort(key=lambda r: (-r['wins'], r['macro_normalized_mae'], r['id']))
    result['validation_target_met'] = set(folds) == set(range(1, 6)) and any(r['wins'] >= 15 for r in result['ranked'])
    path = OUTPUT / 'reports' / datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    path.mkdir(parents=True, exist_ok=False)
    atomic_json(path / 'ranking.json', result)
    print(f"Validation report: {joint.relative(path)}; complete={len(result['ranked'])}", flush=True)
    for row in result['ranked']:
        print(f"{row['id']}: wins={row['wins']}/24; nMAE={row['macro_normalized_mae']:.6f}", flush=True)
    return result, path


def finalize(tid):
    with artifact_lock(OUTPUT / 'locks/finalize.lock'):
        if (OUTPUT / 'selection.json').exists():
            raise FileExistsError("Final shared-owner selection is already locked")
        result, path = report((1, 2, 3, 4, 5))
        if not result['ranked'] or result['ranked'][0]['id'] != tid:
            raise ValueError("Select the highest ranked complete five-fold candidate")
        atomic_json(OUTPUT / 'selection.json', {'id': tid, 'validation_report': joint.relative(path),
            'report_sha256': sha256_file(path / 'ranking.json'), 'validation': result['ranked'][0],
            'test_policy': 'One locked five-model ensemble; no test-based selection'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('generate', help='Freeze bounded owner parameter proposals')
    p.add_argument('--proposal', type=Path)
    p = commands.add_parser('run', help='Train and validate existing candidates')
    p.add_argument('--ids', nargs='+')
    p.add_argument('--fold', type=int, nargs='+', default=[1, 3])
    p.add_argument('--devices', default='cuda:2,cuda:3')
    p.add_argument('--max-parallel', type=int, default=6)
    p.add_argument('--dry-run', action='store_true')
    p = commands.add_parser('report', help='Audit and rank validation only')
    p.add_argument('--fold', type=int, nargs='+', default=[1, 3])
    p = commands.add_parser('finalize', help='Lock the complete five-fold validation winner')
    p.add_argument('--id', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    (OUTPUT / 'tmp').mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(OUTPUT / 'tmp')
    tempfile.tempdir = str(OUTPUT / 'tmp')
    import torch
    torch.set_num_threads(1)
    if args.command == 'generate':
        generate(args.proposal)
    elif args.command == 'run':
        devices = weakopt.devices_and_folds(args.devices, args.max_parallel, args.fold)
        manifest, _ = load()
        trials = weakopt.selected_trials(manifest, args.ids)
        if not args.dry_run:
            reserve(trials, args.fold)
        jobs = [Job(t['id'], 'WeakOptShared', OUTPUT / 'trials' / t['id']) for t in trials]
        by_id = {t['id']: t for t in trials}
        try:
            failures = run_queue(jobs, devices, args.max_parallel,
                lambda job, device: joint.run_trial(by_id[job.trial], args.fold, device, args.dry_run,
                                                   configs=CONFIGS, output=OUTPUT), record_failures=False)
            if failures:
                raise RuntimeError(f"Shared-owner candidates failed: {failures}")
        except KeyboardInterrupt:
            stop_children()
            raise
    elif args.command == 'report':
        weakopt.devices_and_folds('cuda:2,cuda:3', 6, args.fold)
        report(tuple(args.fold))
    else:
        finalize(args.id)


if __name__ == '__main__':
    main()
