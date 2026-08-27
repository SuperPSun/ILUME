from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from common.identity import semantic_identity
from stage3.capacity import (
    CapacityStudyConfig,
    aggregate_fold_summaries,
    config_for_trial,
    confirmation_trial_numbers,
    load_capacity_study_config,
    materialize_final_recipe_configs,
    select_probe_winners,
    summarize_capacity_manifest,
    refined_validation_summary,
    validate_anchor_decision,
)
from stage3.config import load_stage3_config
import scripts.stage3.train as launcher
from stage1.config import load_config
from stage1.identity import build_stage1_corpus_identity
from stage2.config import load_stage2_config
from common.io import sha256_file


def _write_metrics(path: Path, values: list[float]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for epoch, value in enumerate(values, start=1):
        rows.append(
            {
                "epoch": epoch,
                "validation": {
                    "tasks": {
                        "experiment/a": {"normalized_mae": value + 0.1},
                        "experiment/b": {"normalized_mae": value + 0.2},
                    },
                    "groups": {
                        "group-a": {"normalized_mae": value + 0.15},
                    },
                    "macro_task_equal": {
                        "normalized_mae": {"value": value}
                    },
                    "macro_group_equal": {
                        "normalized_mae": {"value": value + 0.05}
                    },
                },
            }
        )
    (path / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    artifact = path / "taskwise_refined.pt"
    artifact.write_bytes(b"fake-refined-artifact")
    (path / "taskwise_refinement.json").write_text(
        json.dumps({
            "kind": "ilume_stage3_taskwise_refined",
            "format_version": 1,
            "artifact": artifact.name,
            "artifact_sha256": sha256_file(artifact),
            "ordinary_final_epoch": len(values),
            "validation": rows[-1]["validation"],
        }),
        encoding="utf-8",
    )


def test_refined_score_uses_stitched_validation(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_metrics(root, [1.0] * 19 + [0.25])
    summary = refined_validation_summary(root, expected_epochs=20)
    assert summary["score"] == pytest.approx(0.25)
    assert summary["model_selector"] == "taskwise_refined"


def test_refined_score_rejects_corrupt_artifact(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_metrics(root, [0.25] * 20)
    (root / "taskwise_refined.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="missing or corrupt"):
        refined_validation_summary(root, expected_epochs=20)


def test_fold_aggregation_and_probe_tie_priority() -> None:
    aggregate = aggregate_fold_summaries(
        {
            1: {
                "score": 0.2,
                "group_equal_score": 0.25,
                "task_scores": {"a": 0.1},
                "group_scores": {"g": 0.15},
            },
            2: {
                "score": 0.4,
                "group_equal_score": 0.45,
                "task_scores": {"a": 0.3},
                "group_scores": {"g": 0.35},
            },
        }
    )
    assert aggregate["score"] == pytest.approx(0.3)
    assert aggregate["fold_sample_sd"] == pytest.approx(2**0.5 / 10)
    assert aggregate["group_scores"]["g"] == pytest.approx(0.25)
    winners = select_probe_winners(
        [
            {"scale": "S", "recipe": "r8", "score": 0.2},
            {"scale": "S", "recipe": "r4", "score": 0.2},
            {"scale": "Base", "recipe": "r2", "score": 0.1},
        ]
    )
    assert [(row["scale"], row["recipe"]) for row in winners] == [
        ("Base", "r2"),
        ("S", "r4"),
    ]


def test_confirmation_shortlist_deduplicates_baseline() -> None:
    trials = [
        {"number": number, "score": score}
        for number, score in enumerate((0.3, 0.1, 0.2, 0.4, 0.5, 0.6))
    ]
    assert confirmation_trial_numbers(
        trials, baseline_trial=0, top_k=2
    ) == (1, 2, 0)
    assert confirmation_trial_numbers(
        trials, baseline_trial=0, top_k=4
    ) == (1, 2, 0, 3)


def test_capacity_study_and_trial_config_are_strict(tmp_path: Path) -> None:
    study = tmp_path / "study.yaml"
    study.write_text(
        """
schema_version: 2
study_name: test
anchor_decision: outputs/test/anchor.yaml
attempted_trials: 40
startup_trials: 10
trials_per_wave: 2
folds: [1, 2]
confirmation_folds: [3, 4, 5]
top_k: 5
max_retries: 1
sampler_seed: 42
global_experts: [1, 2, 3, 4]
group_experts: [1, 2, 3, 4]
private_experts: [1, 2]
expert_hidden_ratio: [1.0, 1.5, 2.0, 3.0, 4.0]
dropout: [0.0, 0.3]
learning_rate: [0.0001, 0.001]
weight_decay: [0.0001, 0.1]
baseline:
  global_experts: 2
  group_experts: 2
  private_experts: 1
  expert_hidden_ratio: 2.0
  dropout: 0.1
  learning_rate: 0.0003
  weight_decay: 0.01
""".lstrip(),
        encoding="utf-8",
    )
    spec = load_capacity_study_config(study)
    base = load_stage3_config("configs/v1/stage3/base.yaml")
    base = replace(base, training=replace(base.training, epochs=20, seed=42))
    trial = config_for_trial(base, spec.baseline)
    assert trial.model.global_experts == 2
    assert trial.model.expert_hidden_ratio == 2.0
    assert trial.training.learning_rate == pytest.approx(3.0e-4)
    assert trial.training.seed == 42


def test_capacity_study_runs_synchronous_waves_and_resumes(tmp_path: Path) -> None:
    baseline = {
        "global_experts": 2,
        "group_experts": 2,
        "private_experts": 1,
        "expert_hidden_ratio": 2.0,
        "dropout": 0.1,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-2,
    }
    spec = CapacityStudyConfig(
        study_name="capacity-test",
        anchor_decision="outputs/test/anchor.yaml",
        attempted_trials=4,
        startup_trials=2,
        trials_per_wave=2,
        folds=(1, 2),
        confirmation_folds=(3, 4, 5),
        top_k=2,
        max_retries=1,
        sampler_seed=42,
        global_experts=(1, 2),
        group_experts=(1, 2),
        private_experts=(1, 2),
        expert_hidden_ratio=(1.0, 2.0),
        dropout=(0.0, 0.3),
        learning_rate=(1.0e-4, 1.0e-3),
        weight_decay=(1.0e-4, 1.0e-1),
        baseline=baseline,
    )
    spec.validate()
    base = load_stage3_config("configs/v1/stage3/base.yaml")
    base = replace(base, training=replace(base.training, epochs=20, seed=42))
    calls: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def run_wave(trials, *, phase, folds, devices, max_retries):
        assert max_retries == 1
        calls.append(
            (
                phase,
                tuple(number for number, _, _ in trials),
                tuple(folds),
            )
        )
        result = {}
        for number, _, trial_root in trials:
            result[number] = {}
            for fold in folds:
                root = trial_root / phase / "attempt0" / f"fold{fold}"
                _write_metrics(root, [0.1 + number / 100] * 20)
                result[number][fold] = root
        return result

    output = tmp_path / "study"
    report = launcher._run_capacity_study(
        base_config=base,
        study_config=spec,
        output=output,
        resume=False,
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        run_wave=run_wave,
    )
    assert calls[:2] == [
        ("search", (0, 1), (1, 2)),
        ("search", (2, 3), (1, 2)),
    ]
    assert report["shortlist"] == [0, 1]
    assert [row["trial_number"] for row in report["ranking"]] == [0, 1]
    calls.clear()
    resumed = launcher._run_capacity_study(
        base_config=base,
        study_config=spec,
        output=output,
        resume=True,
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        run_wave=run_wave,
    )
    assert resumed == report
    assert calls == []

    decision = tmp_path / "final-recipe.yaml"
    probe_config = "configs/experiments_v1/stage3/probe/base-r4.yaml"
    decision.write_text(
        __import__("yaml").safe_dump(
            {
                    "schema_version": 1,
                "kind": "final_recipe",
                "hpo_output": str(output),
                "trial_number": 0,
                "reason": "test confirmed baseline",
                "scale_configs": {
                    scale: probe_config for scale in ("s", "base", "l", "xl")
                },
                "seed_output_root": "outputs/test/seeds",
                "formal_output_root": "outputs/test/formal",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    materialized = materialize_final_recipe_configs(
        decision, tmp_path / "materialized"
    )
    assert materialized["trial_number"] == 0
    seed_config = load_stage3_config(
        tmp_path / "materialized/seed/seed10042.yaml"
    )
    formal_config = load_stage3_config(tmp_path / "materialized/formal/xl.yaml")
    assert (seed_config.training.seed, seed_config.training.epochs) == (10042, 20)
    assert (formal_config.training.seed, formal_config.training.epochs) == (42, 50)


def test_capacity_wave_retries_failed_fold_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import common.outputs as outputs_module

    monkeypatch.setattr(
        outputs_module, "repository_relative", lambda path: str(path)
    )
    calls = []

    def run_jobs(jobs):
        calls.append(tuple(jobs))
        return {
            (job.trial_number, job.fold): (
                "failed"
                if "attempt0" in job.output_root and job.fold == 1
                else "completed"
            )
            for job in jobs
        }

    monkeypatch.setattr(launcher, "_run_assigned_capacity_jobs", run_jobs)
    config_path = tmp_path / "config.yaml"
    trial_root = tmp_path / "trial_000"
    roots = launcher._run_capacity_wave(
        [(0, config_path, trial_root)],
        phase="search",
        folds=(1, 2),
        devices=("cuda:0", "cuda:1"),
        max_retries=1,
    )

    assert len(calls) == 2
    assert [(job.fold, Path(job.output_root).name) for job in calls[0]] == [
        (1, "attempt0"),
        (2, "attempt0"),
    ]
    assert [(job.fold, Path(job.output_root).name) for job in calls[1]] == [
        (1, "attempt1"),
    ]
    assert roots[0][1] == trial_root / "search/attempt1/fold1"
    assert roots[0][2] == trial_root / "search/attempt0/fold2"


def test_capacity_v1_static_configs_cover_base_r1_to_r8_selection() -> None:
    root = Path("configs/experiments_v1")
    stage1_paths = sorted((root / "stage1").glob("*.yaml"))
    stage2_paths = sorted((root / "stage2").glob("*.yaml"))
    probe_paths = sorted((root / "stage3/probe").glob("*.yaml"))
    assert [path.stem for path in stage1_paths] == ["base", "l", "s", "xl"]
    assert [path.stem for path in stage2_paths] == [
        "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"
    ]
    assert [path.stem for path in probe_paths] == [
        "base-r1", "base-r2", "base-r3", "base-r4", "base-r5", "base-r6", "base-r7", "base-r8"
    ]

    expected_widths = {"s": 384, "base": 512, "l": 640, "xl": 768}
    shared_corpus = Path(
        "outputs/experiments_v1/stage1/base/prepare/artifacts"
    )
    source = semantic_identity(
        "test.stage1-source", {"sources": {"source": {"sha256": "test"}}}
    )
    source_audit = {
        "semantic": {"identities": {"source": source}},
        "locator": {"files": {"source": "data/stage1/source.csv"}},
        "integrity": {"files": {"source": {"sha256": "test"}}},
    }
    corpus_identities = []
    for path in stage1_paths:
        config = load_config(path)
        assert config.model.d_model == expected_widths[path.stem]
        assert config.model.d_model // config.model.n_heads == 64
        assert config.model.feedforward_dim == 4 * config.model.d_model
        assert config.model.descriptor_hidden_dim == 2 * config.model.d_model
        assert config.model.graph_depth == 6
        assert config.training.epochs == 15
        assert config.data.artifacts_dir == shared_corpus
        corpus_identities.append(build_stage1_corpus_identity(config, source_audit))
    assert all(identity == corpus_identities[0] for identity in corpus_identities)

    expected_recipes = {
        "r1": (4, 3.0e-06, 1.0e-05, 3.0e-05, 0.30),
        "r2": (3, 5.0e-06, 1.5e-05, 5.0e-05, 0.20),
        "r3": (2, 7.0e-06, 2.0e-05, 7.0e-05, 0.15),
        "r4": (1, 1.0e-05, 3.0e-05, 1.0e-04, 0.10),
        "r5": (0, 1.0e-05, 3.0e-05, 1.0e-04, 0.10),
        "r6": (0, 1.5e-05, 4.5e-05, 1.5e-04, 0.075),
        "r7": (0, 2.0e-05, 6.0e-05, 2.0e-04, 0.05),
        "r8": (0, 3.0e-05, 9.0e-05, 3.0e-04, 0.03),
    }
    for path in stage2_paths:
        recipe = path.stem
        config = load_stage2_config(path)
        assert config.model.object_ffn_dim == 1024
        assert config.model.object_layers == 2
        assert config.training.epochs == 10
        assert recipe in expected_recipes
        assert config.data.pretrain_artifacts_dir == shared_corpus
        assert config.initialization.checkpoint == Path(
            "outputs/experiments_v1/stage1/base/train/checkpoint_epoch_00015.pt"
        )
        assert (
            config.training.backbone_frozen_epochs,
            config.training.backbone_learning_rate,
            config.training.object_encoder_learning_rate,
            config.training.task_head_learning_rate,
            config.loss.lambda_teacher,
        ) == pytest.approx(expected_recipes[recipe])

    for path in probe_paths:
        config = load_stage3_config(path)
        recipe = path.stem.removeprefix("base-")
        assert config.training.seed == 42
        assert config.training.epochs == 30
        assert config.training.checkpoint_interval_epochs == 10
        assert str(config.data.artifacts_dir).startswith(
            "outputs/experiments_v1/"
        )
        assert config.initialization.stage2_encoder == Path(
            f"outputs/experiments_v1/stage2/base/{recipe}/train/stage2_encoder.pt"
        )

    study = load_capacity_study_config(
        root / "stage3/hpo.yaml"
    )
    assert study.attempted_trials == 40
    assert study.startup_trials == 10
    assert study.baseline["learning_rate"] == pytest.approx(3.0e-4)

    all_paths = sorted(root.rglob("*.yaml"))
    assert len(all_paths) == 22

    def strings(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, str):
            yield value

    output_references = [
        value
        for path in all_paths
        for value in strings(yaml.safe_load(path.read_text(encoding="utf-8")))
        if value.startswith("outputs/")
    ]
    assert output_references
    assert all(
        value.startswith("outputs/experiments_v1/")
        for value in output_references
    )


def test_robustness_manifest_reports_seed_and_task_variation(tmp_path: Path) -> None:
    runs = []
    for seed_index, seed in enumerate((42, 10042)):
        for fold in (1, 2):
            root = tmp_path / f"seed{seed}" / f"fold{fold}"
            _write_metrics(root, [0.2 + seed_index * 0.1 + fold * 0.01] * 20)
            runs.append({"seed": seed, "fold": fold, "path": str(root)})
    manifest = tmp_path / "robustness.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "kind": "robustness",
                "expected_epochs": 20,
                "runs": runs,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report = summarize_capacity_manifest(manifest)
    assert report["worst_seed"] == 10042
    assert report["seed_score_mean"] == pytest.approx(0.265)
    assert report["seed_score_range"] == pytest.approx(0.1)
    assert report["task_seed_variation"]["experiment/a"]["range"] == pytest.approx(
        0.1
    )


def test_anchor_decision_requires_selected_probe_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import common.outputs as outputs_module

    monkeypatch.setattr(outputs_module, "REPOSITORY_ROOT", tmp_path)
    config_path = tmp_path / "configs/anchor.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("training: {}\n", encoding="utf-8")
    report_path = tmp_path / "outputs/probe/summary.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps({"scale_winners": [{"id": "l-default"}]}), encoding="utf-8"
    )
    decision_path = tmp_path / "outputs/anchor.yaml"
    decision_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "kind": "anchor",
                "selected_candidate": "l-default",
                "selected_config": "configs/anchor.yaml",
                "probe_report": "outputs/probe/summary.json",
                "reason": "Pareto evidence",
            }
        ),
        encoding="utf-8",
    )
    base_spec = load_capacity_study_config(
        "configs/experiments_v1/stage3/hpo.yaml"
    )
    spec = replace(base_spec, anchor_decision="outputs/anchor.yaml")
    decision = validate_anchor_decision(spec, config_path)
    assert decision["selected_candidate"] == "l-default"
