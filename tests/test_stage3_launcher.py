from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from common.identity import semantic_identity
from stage3.config import validate_stage3_folds
import scripts.stage3.train as launcher


TRAINING_IDENTITY = semantic_identity(
    "stage3.training", {"contract_version": 1, "microbatch_size": 1024}
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_history(path: Path, epochs: list[int]) -> None:
    path.write_text(
        "".join(json.dumps({"epoch": epoch}) + "\n" for epoch in epochs),
        encoding="utf-8",
    )


def _write_run(
    root: Path,
    *,
    status: str,
    epochs: list[int],
    checkpoints: list[int],
    identity: dict[str, Any] = TRAINING_IDENTITY,
) -> None:
    root.mkdir()
    (root / "run_config.yaml").write_text("training: {}\n", encoding="utf-8")
    _write_json(
        root / "metadata.json",
        {
            "stage": "stage3",
            "operation": "train",
            "status": status,
            "provenance": {"fold": 2},
            "semantic_identity": identity,
        },
    )
    _write_history(root / "metrics.jsonl", epochs)
    _write_history(root / "diagnostics.jsonl", epochs)
    for epoch in checkpoints:
        (root / f"checkpoint_epoch_{epoch:05d}.pt").write_bytes(b"checkpoint")


def test_fold_and_device_cli_values_are_strict_and_ordered() -> None:
    parsed = launcher._build_parser().parse_args(
        ["--config", "base.yaml", "--fold", "3", "--output", "outputs/test"]
    )
    assert parsed.fold == [3]
    assert parsed.max_parallel == 1
    assert parsed.resume is False
    assert validate_stage3_folds([3, 1, 5]) == (3, 1, 5)
    with pytest.raises(ValueError, match="1..5"):
        validate_stage3_folds([0])
    with pytest.raises(ValueError, match="duplicate"):
        validate_stage3_folds([1, 1])
    assert launcher._parse_devices("cuda:0,cuda:3") == ("cuda:0", "cuda:3")
    with pytest.raises(ValueError, match="comma-separated"):
        launcher._parse_devices("0,1")
    with pytest.raises(ValueError, match="duplicate"):
        launcher._parse_devices("cuda:0,cuda:0")


def test_completed_run_is_skipped_only_after_full_identity_and_history_checks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fold2"
    _write_run(root, status="completed", epochs=[1, 2], checkpoints=[2])
    _write_json(root / "summary.json", {"fold": 2, "final_epoch": {"epoch": 2}})
    assert launcher._resume_action(
        root, fold=2, total_epochs=2, training_identity=TRAINING_IDENTITY
    ) == ("skipped", None)

    (root / "checkpoint_epoch_00002.pt").unlink()
    with pytest.raises(FileNotFoundError, match="final checkpoint"):
        launcher._resume_action(
            root, fold=2, total_epochs=2, training_identity=TRAINING_IDENTITY
        )
    (root / "checkpoint_epoch_00002.pt").write_bytes(b"checkpoint")

    old_identity = semantic_identity(
        "stage3.training", {"contract_version": 1, "microbatch_size": 128}
    )
    _write_json(
        root / "metadata.json",
        {
            "stage": "stage3",
            "operation": "train",
            "status": "completed",
            "provenance": {"fold": 2},
            "semantic_identity": old_identity,
        },
    )
    with pytest.raises(ValueError, match="semantic identity mismatch"):
        launcher._resume_action(
            root, fold=2, total_epochs=2, training_identity=TRAINING_IDENTITY
        )


def test_resume_requires_checkpoint_and_both_history_tails_to_match(
    tmp_path: Path,
) -> None:
    legal = tmp_path / "legal"
    _write_run(legal, status="failed", epochs=[1, 2], checkpoints=[1, 2])
    assert launcher._resume_action(
        legal, fold=2, total_epochs=4, training_identity=TRAINING_IDENTITY
    ) == ("resume", legal / "checkpoint_epoch_00002.pt")

    history_ahead = tmp_path / "history-ahead"
    _write_run(
        history_ahead, status="failed", epochs=[1, 2, 3], checkpoints=[1, 2]
    )
    with pytest.raises(ValueError, match="tails differ"):
        launcher._resume_action(
            history_ahead,
            fold=2,
            total_epochs=4,
            training_identity=TRAINING_IDENTITY,
        )

    no_checkpoint = tmp_path / "no-checkpoint"
    _write_run(no_checkpoint, status="running", epochs=[], checkpoints=[])
    with pytest.raises(ValueError, match="no legal complete checkpoint"):
        launcher._resume_action(
            no_checkpoint,
            fold=2,
            total_epochs=4,
            training_identity=TRAINING_IDENTITY,
        )


class _FakeQueue:
    def __init__(self) -> None:
        self.items: list[tuple[str, str | None, str | None]] = []
        self.closed = False

    def put(self, value: tuple[str, str | None, str | None]) -> None:
        self.items.append(value)

    def get_nowait(self) -> tuple[str, str | None, str | None]:
        if not self.items:
            raise launcher.queue.Empty
        return self.items.pop(0)

    def close(self) -> None:
        self.closed = True

    def join_thread(self) -> None:
        return


class _FakeProcess:
    created: list["_FakeProcess"] = []

    def __init__(self, *, target, args, name: str) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.sentinel = object()
        self.exitcode: int | None = None
        self.terminated = False
        self.killed = False
        self.created.append(self)

    def start(self) -> None:
        try:
            self.target(*self.args)
        except SystemExit as error:
            self.exitcode = int(error.code)
        else:
            self.exitcode = 0

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def is_alive(self) -> bool:
        return self.exitcode is None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.exitcode = -9


class _FakeContext:
    Process = _FakeProcess

    @staticmethod
    def Queue() -> _FakeQueue:
        return _FakeQueue()


def test_scheduler_binds_slots_and_continues_after_fold_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeProcess.created = []
    calls: list[tuple[int, str | None, bool]] = []

    def worker(config, fold, output, resume, device, progress, result_queue):
        del config, output, resume
        calls.append((fold, device, progress))
        status = "failed" if fold == 2 else "completed"
        result_queue.put((status, None, None))
        if status == "failed":
            raise SystemExit(1)

    monkeypatch.setattr(launcher.multiprocessing, "get_context", lambda mode: _FakeContext())
    monkeypatch.setattr("multiprocessing.connection.wait", lambda sentinels: sentinels)
    monkeypatch.setattr(launcher, "_worker_entry", worker)
    results = launcher._run_schedule(
        config_path="config.yaml",
        folds=(1, 2, 3),
        output_root="outputs/test",
        resume=False,
        max_parallel=2,
        devices=("cuda:0", "cuda:1"),
    )
    assert results == {1: "completed", 2: "failed", 3: "completed"}
    assert calls == [
        (1, "cuda:0", True),
        (2, "cuda:1", False),
        (3, "cuda:0", False),
    ]


def test_interrupt_terminates_then_kills_workers_that_remain_alive() -> None:
    process = _FakeProcess(target=lambda: None, args=(), name="worker")
    result_queue = _FakeQueue()
    launcher._terminate_workers({0: (process, 1, result_queue)})
    assert process.terminated
    assert process.killed
    assert process.exitcode == -9
    assert result_queue.closed
