from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.stage3.evaluate as launcher
from common.identity import semantic_identity
from stage3.config import Stage3Config
from stage3.evaluate import resolve_stage3_reporting_study_id


EVALUATION_IDENTITY = semantic_identity("stage3.evaluation", {"contract_version": 1})


class _Progress:
    class _Status:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    def status(self, message: str) -> _Status:
        del message
        return self._Status()


class _Run:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.completed: dict[str, Any] | None = None
        self.failed = False

    def complete(self, result: dict[str, Any]) -> None:
        self.completed = result

    def fail(self) -> None:
        self.failed = True


def _args(*values: str) -> argparse.Namespace:
    return launcher._build_parser().parse_args(values)


def test_validation_fold_cli_is_multi_value_and_ordered() -> None:
    parsed = _args(
        "--config", "base.yaml", "--checkpoint-dir", "train", "--split", "valid",
        "--fold", "3", "1", "5", "--output", "evaluate",
    )
    assert launcher._validate_request(parsed) == (3, 1, 5)
    assert parsed.checkpoint_epoch is None


def test_removed_taskwise_refined_flag_is_rejected() -> None:
    with pytest.raises(SystemExit):
        _args(
            "--config", "base.yaml", "--checkpoint-dir", "train",
            "--split", "valid", "--fold", "1", "--taskwise-refined",
            "--output", "evaluate",
        )


def test_validation_schedule_is_sequential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int | None, list[str] | None, str]] = []

    def run_fold(**kwargs: Any) -> None:
        calls.append(
            (
                kwargs["fold"],
                kwargs["checkpoint_epoch"],
                kwargs["tasks"],
                kwargs["study_id"],
            )
        )

    monkeypatch.setattr(launcher, "_run_fold", run_fold)
    results = launcher._run_validation_schedule(
        config=Stage3Config(),
        config_path="base.yaml",
        checkpoint_dir=Path("train"),
        output_root="evaluate",
        folds=(3, 2, 5),
        checkpoint_epoch=10,
        tasks=["task/a"],
        study_id="study-a",
        progress=_Progress(),
        resolve_identity=lambda *args, **kwargs: EVALUATION_IDENTITY,
        evaluate_checkpoints=lambda *args, **kwargs: {},
    )
    assert calls == [
        (3, 10, ["task/a"], "study-a"),
        (2, 10, ["task/a"], "study-a"),
        (5, 10, ["task/a"], "study-a"),
    ]
    assert results == {3: "completed", 2: "completed", 5: "completed"}


def test_validation_main_reports_requested_order(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        launcher.sys,
        "argv",
        [
            "evaluate.py",
            "--config", "base.yaml",
            "--checkpoint-dir", "train",
            "--split", "valid",
            "--fold", "3", "1", "5",
            "--study-id", "study-a",
            "--output", "evaluate",
        ],
    )
    monkeypatch.setattr(launcher, "load_stage3_config", lambda path: Stage3Config())
    monkeypatch.setattr(launcher, "configure_process_runtime", lambda config: None)
    monkeypatch.setattr(launcher, "repository_path", lambda path: Path(path))
    monkeypatch.setattr(
        launcher,
        "_run_validation_schedule",
        lambda **kwargs: {3: "completed", 1: "completed", 5: "completed"},
    )
    assert launcher.main() == 0
    assert capsys.readouterr().out.splitlines() == [
        "Stage3 validation evaluation complete",
        "completed: 3, 1, 5",
        "failed: -",
    ]


def test_single_validation_fold_uses_fold_directory_and_run_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs: list[_Run] = []
    open_calls: list[dict[str, Any]] = []
    evaluate_calls: list[dict[str, Any]] = []

    def open_run(**kwargs: Any) -> _Run:
        open_calls.append(kwargs)
        run = _Run(Path(f"/repo/{kwargs['output']}"))
        runs.append(run)
        return run

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        evaluate_calls.append(kwargs)
        return {"split": "valid"}

    monkeypatch.setattr(launcher, "open_run_directory", open_run)
    launcher._run_fold(
        config=Stage3Config(),
        config_path="base.yaml",
        checkpoint_dir=launcher.ROOT / "train",
        output_root="evaluate",
        fold=3,
        checkpoint_epoch=10,
        tasks=["task/a"],
        study_id="study-a",
        progress=_Progress(),
        resolve_identity=lambda *args, **kwargs: EVALUATION_IDENTITY,
        evaluate_checkpoints=evaluate,
    )
    assert open_calls[0]["output"] == Path("evaluate/fold3")
    assert open_calls[0]["details"]["reporting_study_id"] == "study-a"
    assert evaluate_calls[0]["fold"] == 3
    assert evaluate_calls[0]["predictions_dir"] == Path(
        "/repo/evaluate/fold3/predictions"
    )
    assert runs[0].completed == {"split": "valid"}
    assert runs[0].failed is False


def test_test_path_remains_one_root_ensemble_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    open_calls: list[dict[str, Any]] = []
    evaluate_calls: list[dict[str, Any]] = []
    run = _Run(Path("/repo/evaluate_test"))

    def open_run(**kwargs: Any) -> _Run:
        open_calls.append(kwargs)
        return run

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        evaluate_calls.append(kwargs)
        return {"split": "test"}

    monkeypatch.setattr(launcher, "open_run_directory", open_run)
    args = SimpleNamespace(
        config="base.yaml",
        output="evaluate_test",
        checkpoint_epoch=100,
        tasks=None,
        study_id=None,
    )
    launcher._run_test(
        args=args,
        config=Stage3Config(),
        checkpoint_dir=launcher.ROOT / "train",
        progress=_Progress(),
        resolve_identity=lambda *args, **kwargs: EVALUATION_IDENTITY,
        evaluate_checkpoints=evaluate,
    )
    assert open_calls[0]["output"] == "evaluate_test"
    assert evaluate_calls == [
        {
            "split": "test",
            "ensemble_folds": True,
            "checkpoint_epoch": 100,
            "task_subset": None,
            "fold": None,
            "predictions_dir": Path("/repo/evaluate_test/predictions"),
            "reporting_study_id": None,
            "expected_evaluation_identity": EVALUATION_IDENTITY,
        }
    ]
    assert run.completed == {"split": "test"}


def test_default_validation_study_id_keeps_existing_fold_independent_rule(
    tmp_path: Path,
) -> None:
    prepared_identity = semantic_identity("stage3.prepared-data", {"version": 1})
    metadata = {
        "semantic": {"identities": {"prepared": prepared_identity}}
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    config = Stage3Config(
        data=replace(Stage3Config().data, artifacts_dir=tmp_path)
    )
    assert resolve_stage3_reporting_study_id(
        config, checkpoint_epoch=10
    ) == f"ilume-stage3-{prepared_identity['hash']}-epoch10"
    assert resolve_stage3_reporting_study_id(
        config
    ) == f"ilume-stage3-{prepared_identity['hash']}-taskwise-refined"
