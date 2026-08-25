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


def test_fold_and_device_cli_values_are_ordered() -> None:
    parsed = launcher._build_parser().parse_args(
        ["--config", "base.yaml", "--fold", "3", "--output", "outputs/test"]
    )
    assert parsed.fold == [3]
    assert parsed.max_parallel == 1
    assert parsed.resume is False
    assert validate_stage3_folds([3, 1, 5]) == (3, 1, 5)
    assert launcher._parse_devices("cuda:0,cuda:3") == ("cuda:0", "cuda:3")


def test_completed_run_is_skipped_after_identity_and_history_checks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fold2"
    _write_run(root, status="completed", epochs=[1, 2], checkpoints=[2])
    _write_json(root / "summary.json", {"fold": 2, "final_epoch": {"epoch": 2}})
    assert launcher._resume_action(
        root, fold=2, total_epochs=2, training_identity=TRAINING_IDENTITY
    ) == ("skipped", None)


def test_resume_uses_latest_complete_checkpoint_and_matching_history(
    tmp_path: Path,
) -> None:
    legal = tmp_path / "legal"
    _write_run(legal, status="failed", epochs=[1, 2], checkpoints=[1, 2])
    assert launcher._resume_action(
        legal, fold=2, total_epochs=4, training_identity=TRAINING_IDENTITY
    ) == ("resume", legal / "checkpoint_epoch_00002.pt")


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

class _FakeContext:
    Process = _FakeProcess

    @staticmethod
    def Queue() -> _FakeQueue:
        return _FakeQueue()


def test_scheduler_binds_slots_for_successful_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeProcess.created = []
    calls: list[tuple[int, str | None, bool]] = []

    def worker(config, fold, output, resume, device, progress, result_queue):
        del config, output, resume
        calls.append((fold, device, progress))
        result_queue.put(("completed", None, None))

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
    assert results == {1: "completed", 2: "completed", 3: "completed"}
    assert calls == [
        (1, "cuda:0", True),
        (2, "cuda:1", False),
        (3, "cuda:0", False),
    ]
