"""人工对照、ask/tell 搜索、复核与终测的单写入者控制器。"""
from __future__ import annotations

import csv
import json
import os
import pickle
import re
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev

import yaml

from common.io import atomic_json, sha256_file
from common.training import canonical_json_sha256
from stage1.regression import REGRESSION_TASKS
from .contracts import joint_score, rank, validate_lock
from .scheduler import Job, Scheduler, BudgetExhausted, completed_output
from .spaces import compile_joint, compile_posttraining, joint_controls, posttraining_controls, write_yaml


def read_json(file):
    return json.loads(Path(file).read_text())


def save_json(file, data):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(file, data)


def suggest(trial, space):
    result = {}
    for name, spec in space.items():
        if spec["kind"] == "categorical":
            result[name] = trial.suggest_categorical(name, spec["choices"])
        elif spec["kind"] == "int":
            result[name] = trial.suggest_int(name, spec["low"], spec["high"])
        else:
            result[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    return result


class Workflow:
    def __init__(self, study_file, args):
        self.study_file = Path(study_file)
        self.study = yaml.safe_load(self.study_file.read_text())
        self.args = args
        study_id = args.study_id or self.study["study_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", study_id):
            raise ValueError("study_id must be a portable directory name")
        self.root = Path("outputs/HPO_v4") / study_id
        self.config_root = Path("configs/HPO_v4")
        self.study_id = study_id
        from common.hpo_protocol import validate_hpo_paths
        validate_hpo_paths(self.study_file, self.root)
        if self.study["protocol"] != "hpo_v4" or self.study["optuna_version"] != "4.7.0":
            raise ValueError("HPO protocol/version mismatch")
        if self.study["split_seeds"] != [42, 123, 1234, 3407, 90756] or self.study["confirm_folds"] != [1, 2, 3, 4, 5]:
            raise ValueError("HPO requires the five existing splits and five confirmation folds")
        budgets = self.study["budgets"]
        if (budgets["joint_controls"] > 48 or budgets["joint_structures"] > 2 or budgets["joint_tpe_per_structure"] > 80
                or budgets["joint_local"] > 32 or budgets["joint_candidates"] > 240
                or budgets["joint_controls"] + budgets["joint_structures"] * budgets["joint_tpe_per_structure"] + budgets["joint_local"] > budgets["joint_candidates"]):
            raise ValueError("HPO joint candidate budget exceeds the approved 48+160+32 contract")
        self.bases = {key: yaml.safe_load(Path(file).read_text()) for key, file in self.study["bases"].items()}
        self.spaces = {key: yaml.safe_load(Path(file).read_text()) for key, file in self.study["spaces"].items()}
        self.tasks = tuple(self.bases["stage3"]["tasks"])
        if len(self.tasks) != 24 or len(REGRESSION_TASKS) != 14:
            raise ValueError("Unexpected HPO task roster")
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.root / "studies" / "candidates.json"
        self.rows = read_json(self.registry_file) if self.registry_file.exists() else []
        self.cache_file = self.root / "studies" / "caches.json"
        self.caches = read_json(self.cache_file) if self.cache_file.exists() else {}
        settings = self.study["execution"]
        devices = args.devices.split(",") if args.devices else settings["devices"]
        quota = settings["jobs_per_gpu"] if args.jobs_per_gpu is None else args.jobs_per_gpu
        self.scheduler = Scheduler(self.root, devices=devices,
            jobs_per_gpu=quota, max_parallel=min(settings["max_parallel"], len(devices) * quota) if args.max_parallel is None else args.max_parallel,
            reserve_gb=args.gpu_memory_reserve_gb if args.gpu_memory_reserve_gb is not None else settings["gpu_memory_reserve_gb"],
            max_gpu_hours=settings["max_gpu_hours"] if args.max_gpu_hours is None else args.max_gpu_hours, resume=args.resume)
        snapshot = {"study": self.study, "base_files": {file: sha256_file(file) for file in self.study["bases"].values()},
                    "spaces": self.spaces}
        snapshot_file = self.root / "studies" / "study.json"
        if snapshot_file.exists() and not args.resume and args.action != "report" and not args.dry_run:
            raise FileExistsError("Use --resume to continue an existing HPO study")
        if snapshot_file.exists() and read_json(snapshot_file) != snapshot:
            raise ValueError("Study configuration changed; create another study_id")
        save_json(snapshot_file, snapshot)

    def persist(self):
        save_json(self.registry_file, self.rows)

    def command(self, script, config, output, *extra):
        return (sys.executable, f"scripts/{script}.py", "--config", str(config), "--output", str(output), *map(str, extra))

    def run_jobs(self, jobs):
        if self.args.dry_run:
            file = self.root / "reports" / f"{self.args.action}_jobs.json"
            previous = read_json(file) if file.exists() else []
            rows = previous + [{"key": job.key, "argv": [Path(job.argv[0]).name, *job.argv[1:]], "output": str(job.output), "completion": job.completion} for job in jobs]
            save_json(file, list({row["key"]: row for row in rows}.values()))
            return {job.key: job.output for job in jobs}
        resources = self.root / "audit" / "resources.json"
        if resources.exists():
            measured = read_json(resources)["measurements"]
            peak3 = max((cost["peak_allocated_bytes"] for row in measured for cost in row["cost"]), default=0) / 2**30
            peak2 = max((row["stage2_cost"].get("peak_allocated_bytes", 0) for row in measured), default=0) / 2**30
            jobs = [replace(job, memory_gb=peak3 if "scripts/stage3/train.py" in job.argv else peak2 if "scripts/stage2/train.py" in job.argv else 0) for job in jobs]
        try:
            return self.scheduler.run(jobs)
        finally:
            self.scheduler.resume = True

    def candidate(self, branch, phase, params, *, target=None, split=42, seed=42):
        identity = {"branch": branch, "phase": phase, "parameters": params, "target": target, "split": split, "seed": seed}
        key = canonical_json_sha256(identity)[:16]
        found = next((row for row in self.rows if row["id"] == key), None)
        if found:
            return found
        row = {"id": key, **identity, "ordinal": len(self.rows), "status": "pending", "folds": [],
               "parameters_count": 0, "gpu_seconds": 0}
        self.rows.append(row)
        self.persist()
        return row

    def audit(self):
        from stage1.regression import load_regression_config, regression_identity
        from stage1.partial_charge import PartialChargeCache
        from stage2 import load_frozen_stage1_entities
        source = self.study["source"]
        head_config = load_regression_config(self.study["bases"]["posttraining"])
        identity = regression_identity(head_config, source["stage1_checkpoint"])
        frozen = load_frozen_stage1_entities(source["stage1_encoder"], source["stage1_artifacts"])
        import torch
        from stage1.identity import encoding_state_hash
        from stage1.model import load_stage1_model
        complete = load_stage1_model(source["stage1_checkpoint"], source["stage1_artifacts"], device="cpu")
        if encoding_state_hash(complete.model) != frozen.encoder_identity["payload"]["stage1_state_hash"]:
            raise ValueError("Stage1 full final and encoder-only export are not paired")
        charge_recipe = replace(complete.config, auxiliary=replace(complete.config.auxiliary,
            simulation_dir=head_config.simulation_dir, partial_charge_manifest=head_config.partial_charge_manifest,
            partial_charge_cache=head_config.partial_charge_cache))
        charge = PartialChargeCache(charge_recipe, read_json(Path(source["stage1_artifacts"]) / "metadata.json"))
        checkpoint = torch.load(source["stage1_checkpoint"], map_location="cpu", weights_only=False)
        if charge.manifest["identity"]["hash"] != checkpoint["training_identity"]["payload"]["training"].get("partial_charge_identity"):
            raise ValueError("Stage1 partial-charge sidecar differs from the complete final")
        del checkpoint
        files = [Path(source["stage1_checkpoint"]), Path(source["stage1_encoder"])]
        files += [head_config.partial_charge_cache / name for name in ("metadata.json", "targets.pt", "mapping_audit.json")]
        files += sorted(Path(source["stage1_artifacts"]).glob("*.json"))
        files += sorted(Path(self.bases["posttraining"]["simulation_dir"]).rglob("*.csv"))
        for seed in self.study["split_seeds"]:
            directory = Path("data") / str(seed)
            if not (directory / "stage2").is_dir() or not (directory / "stage3").is_dir():
                raise FileNotFoundError(f"Missing paired split: {seed}")
            files += [directory / "task_catalog.csv"]
            files += sorted((directory / "stage2").rglob("*.csv"))
            files += sorted((directory / "stage3").rglob("*.csv"))
        result = {"protocol": "hpo_v4", "source_identity": identity, "encoder_identity": frozen.encoder_identity,
                  "partial_charge_identity": charge.manifest["identity"],
                  "files": {str(file): sha256_file(file) for file in files}, "test_metrics_computed": False}
        save_json(self.root / "audit" / "sources.json", result)
        if self.args.measure:
            self.prepare()
            measurements = []
            quota, limit = self.scheduler.quota, self.scheduler.limit
            for concurrency in self.study["calibration"]["jobs_per_gpu"]:
                self.scheduler.quota = concurrency
                self.scheduler.limit = concurrency * len(self.scheduler.devices)
                if self.args.max_parallel is not None:
                    self.scheduler.limit = min(self.scheduler.limit, self.args.max_parallel)
                controls = []
                slots = self.scheduler.limit
                for index, params in enumerate(self.study["calibration"]["candidate_parameters"]):
                    controls += [self.candidate("joint", f"calibration_{concurrency}_{index}_{replica}", params)
                                 for replica in range(slots)]
                started = time.monotonic()
                self.joint(controls, (1,))
                for row in controls:
                    measurements.append({"jobs_per_gpu": concurrency, "candidate": row["id"],
                                         "stage2_cost": self.scheduler.state["jobs"].get(row.get("stage2_job"), {}),
                                         "status": row["status"], "cost": row.get("cost", []),
                                         "wall_seconds": time.monotonic() - started, "host_load": os.getloadavg(),
                                         "max_parallel": self.scheduler.limit,
                                         "effective_jobs_per_gpu": self.scheduler.quota})
            self.scheduler.quota, self.scheduler.limit = quota, limit
            save_json(self.root / "audit" / "resources.json", {"measurements": measurements, "status": "completed"})
            self.report()
        return result

    def verify_audit(self):
        audit = read_json(self.root / "audit" / "sources.json")
        for file, digest in audit["files"].items():
            if sha256_file(file) != digest:
                raise ValueError(f"Audited source changed: {file}")
        return audit

    def prepare(self):
        if not self.args.dry_run:
            self.verify_audit()
        config = self.config_root / "trials" / self.study_id / "cache" / "posttraining.yaml"
        write_yaml(config, self.bases["posttraining"])
        output = self.root / "cache" / "posttraining"
        jobs = [Job("cache_posttraining", self.command("stage1/regression", config, output,
                    "--checkpoint", self.study["source"]["stage1_checkpoint"], "--prepare-cache"), output)]
        for seed in self.study["split_seeds"]:
            two, _ = compile_joint(self.bases["stage2"], self.bases["stage3"], {}, root=str(self.root), split_seed=seed)
            file = self.config_root / "trials" / self.study_id / "cache" / f"split{seed}_stage2.yaml"
            write_yaml(file, two)
            output = self.root / "cache" / f"split{seed}" / "stage2"
            jobs.append(Job(f"cache_stage2_{seed}", self.command("stage2/prepare", file, output), output))
        paths = self.run_jobs(jobs)
        if not self.args.dry_run and any(path is None for path in paths.values()):
            raise RuntimeError("HPO cache preparation failed")
        if not self.args.dry_run:
            self.caches["posttraining"] = str(paths["cache_posttraining"])
            for seed in self.study["split_seeds"]:
                self.caches[f"stage2_{seed}"] = str(paths[f"cache_stage2_{seed}"])
            save_json(self.cache_file, self.caches)

    def joint(self, candidates, folds):
        folds = tuple(folds)
        generated, stage2_jobs = {}, {}
        for row in candidates:
            params = row["parameters"]
            two, three = compile_joint(self.bases["stage2"], self.bases["stage3"], params,
                          root=str(self.root), split_seed=row["split"], training_seed=row["seed"])
            cache2 = self.caches.get(f"stage2_{row['split']}")
            if cache2:
                two["data"]["artifacts_dir"] = f"{cache2}/artifacts"
                three["initialization"]["simulation_artifacts_dir"] = two["data"]["artifacts_dir"]
            key = canonical_json_sha256({"recipe": two, "calibration_replica": row["phase"] if row["phase"].startswith("calibration") else None})[:16]
            file2 = self.config_root / "trials" / self.study_id / f"stage2_{key}.yaml"
            output2 = self.root / "joint" / f"stage2_{key}" / "stage2"
            write_yaml(file2, two)
            if row["phase"] == "control":
                write_yaml(self.config_root / "controls" / self.study_id / row["id"] / "stage2.yaml", two)
            stage2_jobs[key] = Job(f"stage2_{key}", self.command("stage2/train", file2, output2), output2)
            generated[row["id"]] = (key, three)
        accepted = self.run_jobs(list(stage2_jobs.values()))
        preparations, train_jobs, valid_jobs = {}, [], []
        for row in candidates:
            key, three = generated[row["id"]]
            source = accepted[f"stage2_{key}"]
            if source is None:
                row["status"] = "failed"
                continue
            row["stage2"] = str(source)
            row["stage2_job"] = f"stage2_{key}"
            three["initialization"]["stage2_final"] = str(source / "stage2_final.pt")
            cached3 = self.caches.get(f"stage3_{row['split']}")
            if cached3:
                three["data"]["artifacts_dir"] = f"{cached3}/artifacts"
            model_key = canonical_json_sha256(three)[:16]
            row["model_id"] = model_key
            file3 = self.config_root / "trials" / self.study_id / model_key / "stage3.yaml"
            write_yaml(file3, three)
            if row["phase"] == "control":
                write_yaml(self.config_root / "controls" / self.study_id / row["id"] / "stage3.yaml", three)
            row["config"] = str(file3)
            cache = self.root / "cache" / f"split{row['split']}" / "stage3"
            if row["split"] not in preparations and not cached3:
                preparations[row["split"]] = Job(f"cache_stage3_{row['split']}", self.command("stage3/prepare", file3, cache), cache)
            output = self.root / "joint" / model_key
            row["output"] = str(output)
            for fold in folds:
                target = output / "fold_runs" / f"fold{fold}"
                train_jobs.append(Job(f"{row['model_id']}_train_{fold}", self.command("stage3/train", file3, target,
                                      "--fold", fold, "--max-parallel", 1), target, f"fold{fold}"))
        if preparations:
            cached = self.run_jobs(list(preparations.values()))
            if not self.args.dry_run:
                if any(path is None for path in cached.values()):
                    raise RuntimeError("Stage3 cache preparation failed")
                for seed in preparations:
                    self.caches[f"stage3_{seed}"] = str(cached[f"cache_stage3_{seed}"])
                save_json(self.cache_file, self.caches)
                # Recompile downstream configs after accepting cache locators, without changing the old snapshots.
                if any(cached[job.key] != job.output for job in preparations.values()):
                    return self.joint(candidates, folds)
        trained = self.run_jobs(list({job.key: job for job in train_jobs}.values()))
        for row in candidates:
            if row["status"] == "failed":
                continue
            output = Path(row["output"])
            checkpoint = output / "train"
            checkpoint.mkdir(parents=True, exist_ok=True)
            for fold in folds:
                actual = trained[f"{row['model_id']}_train_{fold}"]
                if actual is None:
                    row["status"] = "failed"
                    break
                target = checkpoint / f"fold{fold}"
                if not self.args.dry_run:
                    wanted = actual / f"fold{fold}"
                    if target.is_symlink() and target.resolve() != wanted.resolve():
                        raise ValueError("Accepted fold locator changed")
                    if not target.exists():
                        target.symlink_to(os.path.relpath(wanted, checkpoint), target_is_directory=True)
                validation = output / "validation_runs" / f"fold{fold}"
                valid_jobs.append(Job(f"{row['model_id']}_valid_{fold}", self.command("stage3/evaluate", row["config"], validation,
                                      "--checkpoint-dir", checkpoint, "--split", "valid", "--domain", "experimental", "--fold", fold,
                                      "--study-id", self.study_id), validation, f"fold{fold}"))
        evaluated = self.run_jobs(list({job.key: job for job in valid_jobs}.values()))
        if self.args.dry_run:
            self.persist()
            return candidates
        for row in candidates:
            if row["status"] == "failed":
                continue
            evaluations = {fold: evaluated[f"{row['model_id']}_valid_{fold}"] for fold in folds}
            if any(file is None for file in evaluations.values()):
                row["status"] = "failed"
                continue
            evaluations = {fold: file / f"fold{fold}" for fold, file in evaluations.items()}
            try:
                scored = joint_score(evaluations, self.tasks, folds)
            except ValueError as error:
                row.update(status="failed", failure=f"invalid_validation: {error}")
                continue
            row.update(scored, status="completed")
            row["evaluations"] = {str(fold): str(file) for fold, file in evaluations.items()}
            costs = [read_json(Path(row["output"]) / "train" / f"fold{fold}" / "summary.json")["training_cost"] for fold in folds]
            row["cost"] = costs
            row["parameters_count"] = max(item["total_parameters"] for item in costs)
            row["gpu_seconds"] = sum(item["gpu_seconds"] for item in costs) + self.scheduler.state["jobs"][row["stage2_job"]]["wall_seconds"]
            self.persist()
        if folds == tuple(self.study["confirm_folds"]):
            simulation_jobs = []
            for row in candidates:
                if row["status"] == "completed":
                    output = Path(row["output"]) / "simulation_validation"
                    simulation_jobs.append(Job(f"{row['model_id']}_simulation_valid", self.command("stage3/evaluate", row["config"], output,
                        "--checkpoint-dir", Path(row["output"]) / "train", "--split", "valid", "--domain", "simulation",
                        "--ensemble-folds", "--study-id", self.study_id), output))
                    row["simulation_validation"] = str(output)
            results = self.run_jobs(list({job.key: job for job in simulation_jobs}.values()))
            for row in candidates:
                if row["status"] == "completed" and results[f"{row['model_id']}_simulation_valid"] is None:
                    row["status"] = "failed"
        self.persist()
        return candidates

    def posttraining(self, candidates):
        jobs = []
        for row in candidates:
            raw = compile_posttraining(self.bases["posttraining"], {**row["parameters"], "seed": row["seed"]},
                                      row["target"], Path(self.caches.get("posttraining", str(self.root / "cache" / "posttraining"))) / "artifacts")
            model_key = canonical_json_sha256({"config": raw, "target": row["target"]})[:16]
            row["model_id"] = model_key
            config = self.config_root / "trials" / self.study_id / model_key / "posttraining.yaml"
            write_yaml(config, raw)
            if row["phase"] == "control":
                write_yaml(self.config_root / "controls" / self.study_id / row["target"] / row["id"] / "posttraining.yaml", raw)
            row["config"] = str(config)
            output = self.root / "posttraining" / row["target"] / model_key
            jobs.append(Job(row["model_id"], self.command("stage1/regression", config, output,
                       "--checkpoint", self.study["source"]["stage1_checkpoint"], "--tasks", row["target"]), output))
        results = self.run_jobs(list({job.key: job for job in jobs}.values()))
        if self.args.dry_run:
            return candidates
        import math
        for row in candidates:
            output = results[row["model_id"]]
            if output is None:
                row["status"] = "failed"
                continue
            head = read_json(output / "tasks" / row["target"] / "regression_head.json")
            score = head["validation"]["mae"]
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                row.update(status="failed", failure="nonfinite_validation")
                continue
            row.update(status="completed", score=score, output=str(output), parameters_count=head["parameter_count"],
                       gpu_seconds=self.scheduler.state["jobs"][row["model_id"]]["wall_seconds"], actual_seed=head["seed"])
        self.persist()
        return candidates

    def sensitivity(self):
        if not self.args.dry_run:
            self.verify_audit()
        if self.args.branch in {"joint", "all"}:
            controls = joint_controls(self.bases["stage2"], self.bases["stage3"], self.spaces["joint"]["parameters"],
                                      root=str(self.root), limit=self.study["budgets"]["joint_controls"])
            rows = [self.candidate("joint", "control", params) for _, params in controls]
            for row, (name, _) in zip(rows, controls, strict=True):
                row["control"] = name
            self.joint(rows, self.study["screen_folds"])
        if self.args.branch in {"posttraining", "all"}:
            for target in REGRESSION_TASKS:
                rows = [self.candidate("posttraining", "control", params, target=target) for _, params in
                        posttraining_controls(self.bases["posttraining"], self.study["budgets"]["post_controls"])]
                self.posttraining(rows)
        self.report()

    def sensitivities(self):
        controls = [row for row in self.rows if row["branch"] == "joint" and row["phase"] == "control" and row["status"] == "completed"]
        baseline = next(row for row in controls if not row["parameters"])
        scores = {}
        for key in self.spaces["joint"]["parameters"]:
            paired = [abs(row["score"] - baseline["score"]) for row in controls if list(row["parameters"]) == [key]]
            scores[key] = max(paired, default=0)
        priority = {key: index for index, key in enumerate(scores)}
        return sorted(scores, key=lambda key: (-scores[key], priority[key])), scores

    def search(self):
        if self.args.dry_run:
            raise ValueError("Use sensitivity --dry-run to inspect concrete control YAMLs; search requires measured results")
        self.verify_audit()
        if self.study["calibration"]["mandatory_before_search"]:
            resources = read_json(self.root / "audit" / "resources.json")
            if resources["status"] != "completed" or any(row["status"] != "completed" for row in resources["measurements"]):
                raise ValueError("Finish resource calibration before TPE search")
        if self.args.branch in {"posttraining", "all"}:
            for target in REGRESSION_TASKS:
                controls = self.post_rank(target, phases={"control"})
                structures = []
                for row in controls:
                    raw = compile_posttraining(self.bases["posttraining"], row["parameters"], target, "unused")
                    structure = {key: value for key, value in raw["predictor"].items() if key != "dropout"}
                    if structure not in [item[1] for item in structures]:
                        structures.append((row["parameters"], structure))
                    if len(structures) == self.study["budgets"]["post_structures"]:
                        break
                if len(structures) != self.study["budgets"]["post_structures"]:
                    raise ValueError("Not enough completed post-training structures")
                for index, (params, structure) in enumerate(structures):
                    base = deepcopy(params)
                    base["type"] = structure["type"]
                    space = deepcopy(self.spaces["posttraining"]["parameters"])
                    if base["type"] == "linear":
                        space.pop("dropout")
                    self.tpe(f"post_{target}_{index}", space, self.study["budgets"]["post_tpe_per_structure"],
                             base, branch="posttraining", target=target)
        if self.args.branch in {"joint", "all"}:
            ordering, _ = self.sensitivities()
            if self.args.search_phase == "local":
                winner = self.joint_rank(self.study["screen_folds"])[0]
                space = deepcopy(self.spaces["joint"]["secondary"])
                space = dict(list(space.items())[:self.study["budgets"]["joint_local_dimensions"]])
                first_budget = self.study["budgets"]["joint_local"] // 2
                saved = self.root / "studies" / "joint_local_decay.space.json"
                base = read_json(saved)["base"] if saved.exists() else winner["parameters"]
                self.tpe("joint_local_decay", space, first_budget, base, branch="joint", phase="local")
                winner = self.joint_rank(self.study["screen_folds"])[0]
                group_scores, class_scores = {}, {}
                for task, recipe in self.bases["stage3"]["tasks"].items():
                    group_scores.setdefault(recipe["meta_group"], []).append(winner["tasks"][task])
                    class_scores.setdefault(recipe["size_class"], []).append(winner["tasks"][task])
                groups = sorted(group_scores, key=lambda key: (-mean(group_scores[key]), key))[:2]
                classes = sorted(class_scores, key=lambda key: (-mean(class_scores[key]), key))[:2]
                space = {f"group_width:{key}": {"kind": "float", "low": .8, "high": 1.2} for key in groups}
                space.update({f"private_width:{key}": {"kind": "float", "low": .8, "high": 1.2} for key in classes})
                space.update({key: {"kind": "float", "low": winner["parameters"].get(key, 1) * .8,
                                   "high": winner["parameters"].get(key, 1) * 1.2} for key in ("s3_group_lr", "s3_private_lr")})
                saved = self.root / "studies" / "joint_local_capacity.space.json"
                base = winner["parameters"]
                if saved.exists():
                    contract = read_json(saved)
                    base, space = contract["base"], contract["space"]
                self.tpe("joint_local_capacity", space, self.study["budgets"]["joint_local"] - first_budget,
                         base, branch="joint", phase="local")
            else:
                structural = ("global_count", "group_count", "private_count", "global_width", "group_width", "private_width")
                structures = []
                for row in self.joint_rank(self.study["screen_folds"], phases={"control"}):
                    shape = {key: row["parameters"].get(key, self.spaces["joint"]["parameters"][key]["base"]) for key in structural}
                    if shape not in structures:
                        structures.append(shape)
                    if len(structures) == self.study["budgets"]["joint_structures"]:
                        break
                if len(structures) != self.study["budgets"]["joint_structures"]:
                    raise ValueError("Not enough complete joint control structures")
                numeric = [key for key in ordering if key not in structural and self.spaces["joint"]["parameters"][key]["kind"] != "categorical"]
                space = {key: self.spaces["joint"]["parameters"][key] for key in numeric[:self.study["budgets"]["joint_max_dimensions"]]}
                for index, structure in enumerate(structures):
                    self.tpe(f"joint_{index}", space, self.study["budgets"]["joint_tpe_per_structure"], structure, branch="joint")
        self.report()

    def tpe(self, name, space, budget, base, *, branch, target=None, phase="tpe"):
        import optuna
        if optuna.__version__ != "4.7.0":
            raise RuntimeError("HPO requires optuna==4.7.0")
        sampler_file = self.root / "studies" / f"{name}.sampler.pkl"
        sampler = pickle.loads(sampler_file.read_bytes()) if sampler_file.exists() else optuna.samplers.TPESampler(
            seed=self.study["sampler_seed"], multivariate=True, constant_liar=True, n_startup_trials=self.study["budgets"]["joint_startup"] if branch == "joint" and phase == "tpe" else min(16, max(4, budget // 4)))
        storage = f"sqlite:///{self.root}/studies/optuna.sqlite3"
        study = optuna.create_study(study_name=name, storage=storage, sampler=sampler, direction="minimize", load_if_exists=True)
        recipe_file = self.root / "studies" / f"{name}.space.json"
        recipe = {"space": space, "base": base, "branch": branch, "target": target, "folds": self.study["screen_folds"] if branch == "joint" else []}
        if recipe_file.exists() and read_json(recipe_file) != recipe:
            raise ValueError("TPE search contract changed")
        save_json(recipe_file, recipe)
        def save_sampler():
            temp = sampler_file.with_suffix(".tmp")
            temp.write_bytes(pickle.dumps(study.sampler))
            temp.replace(sampler_file)
        while True:
            frozen = study.get_trials(deepcopy=False)
            running = [item for item in frozen if item.state == optuna.trial.TrialState.RUNNING]
            generated = sum("candidate" in item.user_attrs for item in frozen)
            if not running and generated >= budget:
                break
            if len(frozen) - generated > max(100, budget * 100):
                raise ValueError("Search space produced too few legal configurations; no workers launched for rejected proposals")
            batch = []
            if running:
                # User attrs were committed before dispatch, so a crashed worker replays the same proposal.
                for item in running:
                    if "candidate" not in item.user_attrs:
                        study.tell(item.number, state=optuna.trial.TrialState.FAIL)
                        continue
                    row = next(row for row in self.rows if row["id"] == item.user_attrs["candidate"])
                    batch.append((item.number, row))
            else:
                for _ in range(min(self.scheduler.limit, budget - generated)):
                    trial = study.ask()
                    params = {**deepcopy(base), **suggest(trial, space)}
                    try:
                        if branch == "joint":
                            compile_joint(self.bases["stage2"], self.bases["stage3"], params, root=str(self.root))
                    except ValueError as error:
                        trial.set_user_attr("generation_failure", str(error))
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                        save_sampler()
                        continue
                    row = self.candidate(branch, phase, params, target=target)
                    trial.set_user_attr("candidate", row["id"])
                    save_sampler()
                    batch.append((trial.number, row))
            if not batch:
                continue
            rows = [row for _, row in batch]
            self.joint(rows, self.study["screen_folds"]) if branch == "joint" else self.posttraining(rows)
            for number, row in batch:
                study.tell(number, row["score"]) if row["status"] == "completed" else study.tell(number, state=optuna.trial.TrialState.FAIL)
                with (self.root / "studies" / "events.jsonl").open("a") as handle:
                    handle.write(json.dumps({"study": name, "trial": number, "candidate": row["id"], "status": row["status"]}) + "\n")
                save_sampler()

    def joint_rank(self, folds, phases=None):
        return rank([row for row in self.rows if row["branch"] == "joint" and row["split"] == 42 and row["seed"] == 42
                     and (phases is None or row["phase"] in phases) and not row["phase"].startswith("calibration")], tuple(folds))

    def post_rank(self, target, phases=None):
        rows = [row for row in self.rows if row["branch"] == "posttraining" and row["target"] == target
                and (phases is None or row["phase"] in phases)]
        return rank(rows, ())

    def confirm(self):
        if self.args.dry_run:
            raise ValueError("Confirmation requires complete candidate results")
        self.verify_audit()
        if self.args.branch in {"joint", "all"}:
            # Scores for screening and all-fold confirmation remain separate records.
            if self.args.confirm_phase == "folds":
                top = self.joint_rank(self.study["screen_folds"])[:self.study["budgets"]["confirm_top"]]
                baseline = next(row for row in self.rows if row["branch"] == "joint" and row["phase"] == "control" and not row["parameters"])
                selected = {row["id"]: row for row in [*top, baseline]}
                rows = [self.candidate("joint", "confirm", row["parameters"]) for row in selected.values()]
                for row in rows:
                    screen = next(source for source in selected.values() if source["parameters"] == row["parameters"])
                    row["reuse"] = screen["id"]
                self.joint(rows, self.study["confirm_folds"])
            else:
                top = self.joint_rank(self.study["confirm_folds"], phases={"confirm"})[:self.study["budgets"]["seed_top"]]
                if len(top) < self.study["budgets"]["seed_top"]:
                    raise ValueError("Complete five-fold recipe confirmation first")
                rows = [self.candidate("joint", "seed", row["parameters"], seed=seed) for row in top for seed in self.study["training_seeds"]]
                self.joint(rows, self.study["confirm_folds"])
                best = rank(rows, tuple(self.study["confirm_folds"]))[0]
                save_json(self.root / "studies" / "joint_winner.json", {"candidate": best["id"]})
        if self.args.branch in {"posttraining", "all"}:
            winners = {}
            for target in REGRESSION_TASKS:
                top = self.post_rank(target, phases={"control", "tpe"})[:self.study["budgets"]["post_top_recipes"]]
                if len(top) < self.study["budgets"]["post_top_recipes"]:
                    raise ValueError("Finish per-target post-training search first")
                rows = [self.candidate("posttraining", "seed", row["parameters"], target=target, seed=seed)
                        for row in top for seed in self.study["training_seeds"]]
                self.posttraining(rows)
                winners[target] = self.post_rank(target, phases={"seed"})[0]["id"]
            save_json(self.root / "studies" / "head_winners.json", winners)
        self.report()

    def winner(self, file):
        key = read_json(self.root / "studies" / file)["candidate"]
        return next(row for row in self.rows if row["id"] == key)

    def splits(self):
        winner = self.winner("joint_winner.json")
        rows = [self.candidate("joint", "split", winner["parameters"], split=seed, seed=winner["seed"])
                for seed in self.study["split_seeds"]]
        self.joint(rows, self.study["confirm_folds"])
        ordered = rank(rows, tuple(self.study["confirm_folds"]))
        if len(ordered) != 5:
            raise ValueError("All five split comparisons must finish")
        save_json(self.root / "studies" / "split_winner.json", {"candidate": ordered[0]["id"], "label": "验证选出的最佳划分"})
        save_json(self.root / "split_comparison" / "comparison.json", ordered)
        with (self.root / "split_comparison" / "comparison.csv").open("w", newline="") as handle:
            fields = ("id", "split", "seed", "score", "parameters_count", "gpu_seconds")
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in ordered)
        self.report()

    def lock(self):
        self.verify_audit()
        winner = self.winner("split_winner.json")
        heads = read_json(self.root / "studies" / "head_winners.json")
        files = dict(read_json(self.root / "audit" / "sources.json")["files"])
        source_files = [*Path("src").rglob("*.py"), *Path("scripts").rglob("*.py"), self.study_file,
                        *(Path(file) for file in self.study["bases"].values()), *(Path(file) for file in self.study["spaces"].values())]
        for file in source_files:
            files[str(file)] = sha256_file(file)
        locked_heads = {}
        locked_config = self.config_root / "locked" / self.study_id
        final_models = self.root / "final" / "models"
        def alias(destination, source):
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink() and destination.resolve() != source.resolve():
                raise ValueError("Locked model locator changed")
            if not destination.exists():
                destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=True)
            return destination
        for target in REGRESSION_TASKS:
            row = next(row for row in self.rows if row["id"] == heads[target])
            folder = alias(final_models / "stage1" / target, Path(row["output"]) / "tasks" / target)
            config = locked_config / "heads" / f"{target}.yaml"
            write_yaml(config, yaml.safe_load(Path(row["config"]).read_text()))
            files[str(config)] = sha256_file(config)
            locked_heads[target] = {"candidate": row["id"], "config": str(config), "head": str(folder), "seed": row["seed"], "actual_seed": row["actual_seed"]}
            for file in (folder / "regression_head.pt", folder / "regression_head.json", Path(row["config"])):
                files[str(file)] = sha256_file(file)
        source_checkpoint = Path(winner["output"]) / "train"
        checkpoint = final_models / "stage3"
        joint_config = locked_config / "joint.yaml"
        write_yaml(joint_config, yaml.safe_load(Path(winner["config"]).read_text()))
        files[str(joint_config)] = sha256_file(joint_config)
        folds = {}
        for fold in range(1, 6):
            folder = alias(checkpoint / f"fold{fold}", source_checkpoint / f"fold{fold}")
            if not completed_output(folder):
                raise ValueError("Lock requires all five completed folds")
            folds[str(fold)] = str(folder)
            for file in (folder / "three_phase_final.pt", folder / "three_phase_final.json", folder / "resolved_training_plan.json"):
                files[str(file)] = sha256_file(file)
        for file in (Path(winner["stage2"]) / "stage2_final.pt", Path(winner["stage2"]) / "stage2_final.json", Path(winner["config"])):
            files[str(file)] = sha256_file(file)
        lock = {"protocol": "hpo_v4", "study_id": self.study_id, "heads": locked_heads, "source": self.study["source"],
                "joint": {"candidate": winner["id"], "config": str(joint_config), "checkpoint_dir": str(checkpoint),
                          "folds": folds, "split_seed": winner["split"], "training_seed": winner["seed"], "parameters": winner["parameters"]},
                "files": files, "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "test_batch": "locked-best-seed-five-fold-ensemble", "selection_bias": "跨划分验证选择；终测不作为完全独立泛化估计"}
        lock["hash"] = canonical_json_sha256(lock)
        path = self.config_root / "locked" / self.study_id / "manifest.json"
        if path.exists() and read_json(path) != lock:
            raise ValueError("Lock already exists and is immutable")
        save_json(path, lock)
        validate_lock(path)
        save_json(self.root / "final" / "lock.json", lock)
        return lock

    def repeat(self):
        lock = validate_lock(self.config_root / "locked" / self.study_id / "manifest.json")
        seeds = [lock["joint"]["training_seed"], *self.study["repeat_seeds"]]
        rows = [self.candidate("joint", "repeat", lock["joint"]["parameters"], split=lock["joint"]["split_seed"], seed=seed) for seed in seeds]
        self.joint(rows, self.study["confirm_folds"])
        if any(row["status"] != "completed" for row in rows):
            raise ValueError("All repeat seeds must finish; no seed reselection")
        scores = [row["score"] for row in rows]
        tasks = {task: {"mean": mean(row["tasks"][task] for row in rows), "std": stdev(row["tasks"][task] for row in rows),
                        "min": min(row["tasks"][task] for row in rows), "max": max(row["tasks"][task] for row in rows)} for task in self.tasks}
        save_json(self.root / "final" / "repeat.json", {"seeds": seeds, "mean": mean(scores), "std": stdev(scores),
                  "min": min(scores), "max": max(scores), "tasks": tasks, "selection_changed": False})
        self.report()

    def test(self):
        path = self.config_root / "locked" / self.study_id / "manifest.json"
        lock = validate_lock(path)
        if not (self.root / "final" / "repeat.json").is_file():
            raise ValueError("Finish locked repeated validation before the final test batch")
        job = Job("test_stage3", self.command("stage3/evaluate", lock["joint"]["config"], self.root / "final" / "test" / "stage3",
                  "--checkpoint-dir", lock["joint"]["checkpoint_dir"], "--split", "test", "--domain", "all", "--ensemble-folds",
                  "--lock-manifest", path, "--study-id", self.study_id), self.root / "final" / "test" / "stage3")
        jobs = [job]
        from stage1.regression_test import TEST_TASKS
        for target in TEST_TASKS:
            output = self.root / "final" / "test" / "stage1" / target
            head = lock["heads"][target]
            jobs.append(Job(f"test_head_{target}", self.command("stage1/regression_test", head["config"], output,
                "--lock-manifest", path, "--target", target), output))
        results = self.run_jobs(jobs)
        if any(result is None for result in results.values()):
            raise RuntimeError("Locked test batch has unfinished technical failures")
        save_json(self.root / "final" / "test_batch.json", {"lock_hash": lock["hash"], "outputs": {key: str(value) for key, value in results.items()},
                  "unavailable_stage1_tests": [task for task in REGRESSION_TASKS if task not in TEST_TASKS], "selection_changed": False})
        self.report()

    def report(self):
        directory = self.root / "reports"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "candidates.csv").open("w", newline="") as handle:
            fields = ["id", "branch", "phase", "target", "split", "seed", "status", "folds", "score", "parameters_count", "gpu_seconds"]
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({key: row.get(key) for key in fields})
        report = {"protocol": "hpo_v4", "study_id": self.study_id, "completed": sum(row["status"] == "completed" for row in self.rows),
                  "failed": sum(row["status"] == "failed" for row in self.rows), "gpu_hours_used": self.scheduler.state["gpu_hours"],
                  "budgets": self.study["budgets"], "test_selection": False, "candidates": self.rows}
        report["posttraining"] = {}
        for target in REGRESSION_TASKS:
            controls = [row for row in self.rows if row["branch"] == "posttraining" and row["target"] == target and row["phase"] == "control" and row["status"] == "completed"]
            baseline = next((row for row in controls if not row["parameters"]), None)
            report["posttraining"][target] = {"ranking": [row["id"] for row in self.post_rank(target)],
                "paired_controls": [{"id": row["id"], "mae": row["score"], "delta_mae": row["score"] - baseline["score"] if baseline else None} for row in controls]}
        report["joint_rankings"] = {"two_fold": [row["id"] for row in self.joint_rank(self.study["screen_folds"])],
                                    "five_fold": [row["id"] for row in self.joint_rank(self.study["confirm_folds"])]}
        if any(row["phase"] == "control" and row["branch"] == "joint" and not row["parameters"] and row["status"] == "completed" for row in self.rows):
            ordering, sensitivity = self.sensitivities()
            report["sensitivity"] = {key: sensitivity[key] for key in ordering}
        resources = self.root / "audit" / "resources.json"
        if resources.exists():
            measurements = read_json(resources)["measurements"]
            costs = [cost for row in measurements for cost in row["cost"]]
            if costs:
                fold_seconds = mean(cost["gpu_seconds"] for cost in costs)
                b = self.study["budgets"]
                folds = b["joint_candidates"] * 2 + (b["confirm_top"] + 1) * 3 + b["seed_top"] * 5 * 5 + 5 * 5 + 4 * 5
                stage2_costs = [row["stage2_cost"].get("wall_seconds", 0) for row in measurements]
                stage2_count = b["joint_candidates"] + b["seed_top"] * 5 + 5 + 4
                post_count = len(REGRESSION_TASKS) * (b["post_controls"] + b["post_structures"] * b["post_tpe_per_structure"] + b["post_top_recipes"] * 5)
                post_costs = [row["gpu_seconds"] for row in self.rows if row["branch"] == "posttraining" and row["status"] == "completed"]
                storage_files = [file for file in self.root.rglob("*") if file.is_file() and not file.is_symlink()]
                storage_bytes = sum(file.stat().st_size for file in storage_files)
                checkpoint_bytes = [file.stat().st_size for file in storage_files if file.name == "three_phase_final.pt"]
                fold_storage = [sum(file.stat().st_size for file in (Path(row["output"]) / "fold_runs").rglob("*") if file.is_file()) / max(1, len(row["folds"]))
                                for row in self.rows if row["phase"].startswith("calibration") and row["status"] == "completed"]
                source_storage = [sum(file.stat().st_size for file in Path(row["stage2"]).rglob("*") if file.is_file())
                                  for row in self.rows if row["phase"].startswith("calibration") and row["status"] == "completed"]
                report["cost_estimate"] = {"stage3_folds_upper_bound": folds, "stage3_gpu_hours_at_measured_recipe": fold_seconds * folds / 3600,
                    "stage2_candidates_upper_bound": stage2_count, "stage2_gpu_hours_at_measured_recipe": mean(stage2_costs) * stage2_count / 3600,
                    "posttraining_trials_upper_bound": post_count, "posttraining_gpu_hours_at_measured_recipe": mean(post_costs) * post_count / 3600 if post_costs else None,
                    "cache_and_checkpoint_bytes_current": storage_bytes,
                    "stage3_checkpoints_bytes_at_measured_recipe": mean(fold_storage) * folds if fold_storage else None,
                    "stage2_checkpoints_bytes_at_measured_recipe": mean(source_storage) * stage2_count if source_storage else None,
                    "final_artifact_bytes_projection": mean(checkpoint_bytes) * folds if checkpoint_bytes else None,
                    "cache_seconds_measured": sum(row.get("wall_seconds", 0) for key, row in self.scheduler.state["jobs"].items() if key.startswith("cache_")),
                    "note": "测量配方外推，非硬件承诺；轮数/容量变化影响成本，未测量的后训练为 null；周期检查点按测量目录计入，缓存/模型复用不重复计费"}
        for name in ("joint_winner", "split_winner", "head_winners"):
            file = self.root / "studies" / f"{name}.json"
            if file.exists():
                report[name] = read_json(file)
        repeated = self.root / "final" / "repeat.json"
        if repeated.exists():
            report["repeat"] = read_json(repeated)
        tested = self.root / "final" / "test_batch.json"
        if tested.exists():
            batch = read_json(tested)
            report["final_test"] = {"batch": batch, "results": {key: read_json(Path(output) / "summary.json") for key, output in batch["outputs"].items()}}
            simulation = Path(batch["outputs"]["test_stage3"]) / "simulation" / "summary.json"
            if simulation.exists():
                report["final_test"]["simulation"] = read_json(simulation)
        save_json(directory / "report.json", report)
        lines = [f"# HPO v4：{self.study_id}", "", f"完成 {report['completed']} 个候选，失败 {report['failed']} 个；已用 GPU 小时 {report['gpu_hours_used']:.4f}。", "",
                 "选模只使用末轮验证：24 实验任务等权，两模拟独立报告。两折与五折成绩分开保存。", "",
                 "锁定后重复验证不重新选 seed；终测使用最佳共同训练 seed 的五折原单位集成。", "",
                 "五套划分存在开发/测试体系交叉，胜者称为验证选出的最佳划分，终测不能解释为完全独立泛化估计。", "",
                 "实际阶段结果见 report.json、candidates.csv；尚未完成的阶段保持待执行，不生成虚构成绩。"]
        if "repeat" in report:
            repeated = report["repeat"]
            lines += ["", f"重复五折：Q 均值 {repeated['mean']:.6g}，标准差 {repeated['std']:.6g}，范围 [{repeated['min']:.6g}, {repeated['max']:.6g}]。"]
        if "final_test" in report:
            lines += ["", "锁定终测批次已完成；实验、模拟及三个独立头的结果见 report.json 的 final_test，不回流选择。"]
        (directory / "report.md").write_text("\n".join(lines) + "\n")
        return report
