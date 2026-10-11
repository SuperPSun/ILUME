from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from common.io import sha256_file
from stage3.config import load_stage3_config, stage3_config_from_dict
from stage2.home_config import load_home_recipe
from stage3.hpo.contracts import joint_score, rank
from stage3.hpo.spaces import compile_joint, compile_posttraining, joint_controls
from stage3.hpo.scheduler import Job, Scheduler


def test_hpo_expansion_separates_split_seed_and_owner_capacity():
    two = yaml.safe_load(Path("configs/v4/stage2/base.yaml").read_text())
    three = yaml.safe_load(Path("configs/v4/stage3/base.yaml").read_text())
    parameters = {"global_count": 4, "group_count": 2, "private_count": 2, "global_width": .5,
                  "private_width": 2, "group_width": .5, "global_dropout": .01, "group_dropout": .2,
                  "private_dropout": .3, "s2_global_epochs": 5, "s2_group_epochs": 20, "s2_private_epochs": 7,
                  "s3_global_epochs": 30, "s3_private_epochs": 2, "s3_group_epochs": 2}
    compiled2, compiled3 = compile_joint(two, three, parameters, root="outputs/HPO_v4/tiny", split_seed=1234, training_seed=23)
    assert compiled2["data"]["seed"] == compiled3["data"]["seed"] == 1234
    assert compiled2["training"]["seed"] == compiled3["training"]["seed"] == 23
    assert compiled2["training"]["epochs"] == 20
    assert compiled2["home"]["model"] == compiled3["model"]
    assert compiled3["model"]["expert_hidden_ratio"] == 1
    assert compiled3["model"]["private_hidden_ratio"] == 4
    config = stage3_config_from_dict(compiled3)
    overridden = config.resolved_private_recipe("experiment/electrical_conductivity")
    assert overridden.private_hidden_ratio == 1.5
    assert overridden.private_dropout == .3 and overridden.phase3_epochs == 8
    assert config.resolved_private_recipe("experiment/speed_of_sound").phase3_epochs == 0
    assert config.groups["solvation"].experts == 6
    assert compiled2["home"]["groups"]["solvation"]["phase1"] == two["home"]["groups"]["solvation"]["phase1"]
    assert load_stage3_config("configs/v4/stage3/base.yaml").to_dict()["model"] == three["model"]
    assert load_home_recipe("configs/v4/stage2/base.yaml").to_dict()["home"]["model"] == {**two["home"]["model"], "group_experts": 2}
    with pytest.raises(ValueError):
        compile_joint(two, three, {"s3_private_lr": 2}, root="outputs/HPO_v4/tiny")
    space = yaml.safe_load(Path("configs/HPO_v4/spaces/joint.yaml").read_text())["parameters"]
    controls = joint_controls(two, three, space, root="outputs/HPO_v4/tiny", limit=48)
    assert len(controls) <= 48
    for _, parameters in controls:
        compile_joint(two, three, parameters, root="outputs/HPO_v4/tiny")


def test_posttraining_linear_has_no_dropout_and_atom_width_is_independent():
    base = yaml.safe_load(Path("configs/HPO_v4/controls/posttraining.yaml").read_text())
    linear = compile_posttraining(base, {"type": "linear", "dropout": .3}, "HOMO_eV", "cache")
    assert linear["predictor"] == {"type": "linear"}
    atom = compile_posttraining(base, {"type": "residual_mlp", "width": 2, "depth": 3}, "partial_atomic_charge", "cache")
    assert atom["predictor"]["hidden_dims"] == [512, 512, 256]
    assert atom["protocol"] == "hpo_v4"


def evaluation(tmp_path, fold, tasks):
    root = tmp_path / f"fold{fold}"
    (root / "predictions").mkdir(parents=True)
    manifests = []
    for index, task in enumerate(tasks):
        file = root / "predictions" / f"{index}.csv"
        file.write_text("source_row,target,prediction\n1,0,1\n")
        manifests.append({"task": task, "path": f"predictions/{index}.csv", "sha256": sha256_file(file), "rows": 1})
    summary = {"split": "valid", "tasks": {task: {"normalized_mae": index + fold, "count": 1} for index, task in enumerate(tasks)},
               "reporting": {"protocol": {"fold": fold}, "predictions": manifests}}
    (root / "summary.json").write_text(json.dumps(summary))
    return root


def test_objective_is_equal_task_fold_mean_and_rejects_incomplete_predictions(tmp_path):
    tasks = [f"experiment/task{index}" for index in range(24)]
    roots = {fold: evaluation(tmp_path, fold, tasks) for fold in (1, 3)}
    result = joint_score(roots, tasks, (1, 3))
    assert result["score"] == pytest.approx(13.5)
    with pytest.raises(ValueError, match="fold set"):
        joint_score(roots, tasks, range(1, 6))
    summary = json.loads((roots[1] / "summary.json").read_text())
    summary["tasks"][tasks[0]]["normalized_mae"] = float("nan")
    (roots[1] / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Non-finite"):
        joint_score(roots, tasks, (1, 3))
    summary["tasks"][tasks[0]]["normalized_mae"] = 1
    summary["reporting"]["predictions"].pop()
    (roots[1] / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="incomplete"):
        joint_score(roots, tasks, (1, 3))
    rows = [{"status": "completed", "folds": folds, "score": score, "parameters_count": count,
             "gpu_seconds": 10, "ordinal": index} for index, (folds, score, count) in
             enumerate([([1, 3], 1, 20), ([1, 3], 1, 10), ([1, 2, 3, 4, 5], .1, 10)])]
    assert [row["ordinal"] for row in rank(rows, (1, 3))] == [1, 0]


def test_scheduler_oom_preserves_attempt_and_binds_resume_outputs(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text('''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True)
if not out.name.endswith('attempt2'):
    print('CUDA out of memory');raise SystemExit(1)
(out/'metadata.json').write_text(json.dumps({'status':'completed'}))
(out/'summary.json').write_text('{}')
''')
    root = tmp_path / "study"
    output = root / "training"
    job = Job("train", (sys.executable, str(script), "--output", str(output)), output)
    runner = Scheduler(root, devices=["cpu"], jobs_per_gpu=2, max_parallel=2, reserve_gb=0, poll_seconds=.01)
    accepted = runner.run([job])["train"]
    assert output.is_dir() and accepted.name == "training_attempt2"
    assert runner.quota == 1
    assert runner.state["jobs"]["train"]["previous_attempts"][0]["failure"] == "oom"
    resumed = Scheduler(root, devices=["cpu"], resume=True, reserve_gb=0, poll_seconds=.01)
    assert resumed.run([job])["train"] == accepted
    (accepted / "summary.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="damaged"):
        resumed.run([job])


def test_controller_promotes_reuses_and_locks_before_test(tmp_path, monkeypatch):
    """Exercise orchestration using tiny worker artifacts; numerical workers have separate integration tests."""
    from types import SimpleNamespace
    from stage1.regression import REGRESSION_TASKS
    from stage3.hpo.workflow import Workflow, save_json
    root = Path.cwd()
    files = [*Path("configs/HPO_v4").rglob("*.yaml")]
    for file in files:
        destination = tmp_path / file
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / file).read_bytes())
    monkeypatch.chdir(tmp_path)
    study = yaml.safe_load(Path("configs/HPO_v4/study.yaml").read_text())
    study["budgets"].update(post_controls=6, post_tpe_per_structure=1, post_top_recipes=1,
                           joint_controls=8, joint_tpe_per_structure=2, joint_local=2, confirm_top=3, seed_top=1)
    study["calibration"]["mandatory_before_search"] = False
    Path("configs/HPO_v4/study.yaml").write_text(yaml.safe_dump(study, sort_keys=False))
    args = SimpleNamespace(study_id="tiny", devices="cpu", jobs_per_gpu=1, max_parallel=1,
            gpu_memory_reserve_gb=0, max_gpu_hours=None, resume=False, dry_run=False,
            action="sensitivity", branch="all", measure=False, search_phase="broad", confirm_phase="folds")
    workflow = Workflow("configs/HPO_v4/study.yaml", args)
    monkeypatch.setattr(workflow, "verify_audit", lambda: {})
    invocations, requested_splits = {}, []
    def workers(jobs):
        accepted = {}
        for job in jobs:
            accepted[job.key] = job.output
            if job.key in invocations:
                assert invocations[job.key] == job.argv
                continue
            invocations[job.key] = job.argv
            config = yaml.safe_load(Path(job.argv[job.argv.index("--config") + 1]).read_text())
            output = job.output / job.completion
            output.mkdir(parents=True, exist_ok=True)
            (output / "metadata.json").write_text('{"status":"completed"}')
            summary = {}
            script = job.argv[1]
            if script == "scripts/stage2/train.py":
                (output / "stage2_final.pt").write_bytes(job.key.encode())
                (output / "stage2_final.json").write_text('{}')
            elif script == "scripts/stage3/train.py":
                for name in ("three_phase_final.pt", "three_phase_final.json", "resolved_training_plan.json"):
                    (output / name).write_bytes(job.key.encode())
                summary["training_cost"] = {"total_parameters": config["model"]["global_experts"] * 100,
                                            "gpu_seconds": 1, "peak_allocated_bytes": 1}
            elif script == "scripts/stage3/evaluate.py":
                split = job.argv[job.argv.index("--split") + 1]
                requested_splits.append(split)
                if "--fold" in job.argv:
                    fold = int(job.argv[job.argv.index("--fold") + 1])
                    temporary = tmp_path / ("pred_" + job.key)
                    produced = evaluation(temporary, fold, workflow.tasks)
                    summary = json.loads((produced / "summary.json").read_text())
                    for file in (produced / "predictions").iterdir():
                        destination = output / "predictions" / file.name
                        destination.parent.mkdir(exist_ok=True)
                        destination.write_bytes(file.read_bytes())
                    score = 1 / config["model"]["global_experts"] + config["training"]["seed"] / 100000 + config["data"]["seed"] % 5 / 100
                    for metrics in summary["tasks"].values():
                        metrics["normalized_mae"] = score
                else:
                    summary = {"split": split}
            elif script == "scripts/stage1/regression.py":
                target = job.argv[job.argv.index("--tasks") + 1]
                head = output / "tasks" / target
                head.mkdir(parents=True)
                manifest = {"validation": {"mae": 1 / config["epochs"]}, "parameter_count": 10,
                            "seed": config["seed"] + REGRESSION_TASKS.index(target)}
                (head / "regression_head.json").write_text(json.dumps(manifest))
                (head / "regression_head.pt").write_bytes(job.key.encode())
            elif script == "scripts/stage1/regression_test.py":
                assert "--lock-manifest" in job.argv
            else:
                assert script == "scripts/stage3/prepare.py"
                (output / "artifacts").mkdir()
            (output / "summary.json").write_text(json.dumps(summary))
            workflow.scheduler.state["jobs"][job.key] = {"wall_seconds": 1}
        return accepted
    monkeypatch.setattr(workflow, "run_jobs", workers)
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: "test-revision\n")
    workflow.sensitivity()
    invalid = workflow.candidate("joint", "invalid_objective", {"global_width": 1.1})
    with monkeypatch.context() as scoring:
        def reject(*args):
            raise ValueError("Non-finite HPO main objective")
        scoring.setattr("stage3.hpo.workflow.joint_score", reject)
        workflow.joint([invalid], study["screen_folds"])
    assert invalid["status"] == "failed" and "invalid_validation" in invalid["failure"]
    assert invalid not in workflow.joint_rank(study["screen_folds"])
    workflow.search()
    args.branch, args.search_phase = "joint", "local"
    workflow.search()
    args.branch = "all"
    workflow.confirm()
    args.confirm_phase = "seeds"
    workflow.confirm()
    workflow.splits()
    assert set(requested_splits) == {"valid"}
    save_json(workflow.root / "audit" / "sources.json", {"files": {}})
    lock = workflow.lock()
    assert len(lock["heads"]) == 14 and len(lock["joint"]["folds"]) == 5
    with pytest.raises(ValueError, match="repeated validation"):
        workflow.test()
    train_count = sum("scripts/stage3/train.py" == argv[1] for argv in invocations.values())
    workflow.repeat()
    # The best seed's completed folds were reused, only four new seeds trained.
    assert sum("scripts/stage3/train.py" == argv[1] for argv in invocations.values()) == train_count + 20
    workflow.test()
    assert "test" in requested_splits
    locked = json.loads((workflow.root / "final" / "test_batch.json").read_text())
    assert locked["lock_hash"] == lock["hash"] and len(locked["unavailable_stage1_tests"]) == 11


def test_gpu_queue_respects_per_device_and_global_slots(tmp_path, monkeypatch):
    import stage3.hpo.scheduler as scheduler
    worker = tmp_path / "worker.py"
    worker.write_text('''import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output');args=p.parse_args()
time.sleep(.15)
out=Path(args.output);out.mkdir(parents=True)
(out/'metadata.json').write_text(json.dumps({'status':'completed'}))
(out/'summary.json').write_text(json.dumps({'device':os.environ['CUDA_VISIBLE_DEVICES']}))
''')
    monkeypatch.setattr(scheduler, "free_memory_gb", lambda devices: {device: 100 for device in devices})
    save = scheduler.atomic_json
    peaks = []
    def observed(file, state):
        active = [row for row in state["jobs"].values() if row["status"] == "running"]
        peaks.append(len(active))
        assert len(active) <= 3
        assert all(sum(row["device"] == device for row in active) <= 2 for device in ("cuda:0", "cuda:1"))
        save(file, state)
    monkeypatch.setattr(scheduler, "atomic_json", observed)
    runner = Scheduler(tmp_path / "study", devices=["cuda:0", "cuda:1"], jobs_per_gpu=2, max_parallel=3, poll_seconds=.01)
    jobs = [Job(f"job{index}", (sys.executable, str(worker), "--output", str(tmp_path / f"out{index}")), tmp_path / f"out{index}") for index in range(5)]
    accepted = runner.run(jobs)
    assert max(peaks) == 3 and len(accepted) == 5
    assert all(json.loads((path / "summary.json").read_text())["device"] in {"0", "1"} for path in accepted.values())
