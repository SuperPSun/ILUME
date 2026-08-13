from __future__ import annotations

import json

from ilume_pretrain.progress import ProgressReporter, loss_postfix


def test_noninteractive_progress_preserves_json_stdout(capsys):
    reporter = ProgressReporter(interactive=False)
    reporter.emit_json({"step": 3, "loss": 1.25})

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"step": 3, "loss": 1.25}
    assert captured.err == ""


def test_interactive_progress_uses_stderr_and_suppresses_step_json(capsys):
    reporter = ProgressReporter(interactive=True)
    progress = reporter.bar(total=1, desc="Epoch 1/5", unit="step")
    reporter.emit_json({"step": 1, "loss": 1.25})
    progress.set_postfix({"loss": "1.2500"})
    progress.update(1)
    progress.close()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Epoch 1/5" in captured.err
    assert "100%" in captured.err


def test_loss_postfix_contains_all_modalities_learning_rate_and_validation():
    postfix = loss_postfix(
        {
            "loss": 6.0,
            "loss_smiles": 1.0,
            "loss_atom": 1.1,
            "loss_bond": 1.2,
            "loss_descriptor": 1.3,
            "loss_fingerprint": 1.4,
            "learning_rate": 2.0e-4,
        },
        include_learning_rate=True,
        valid_loss=5.5,
    )

    assert tuple(postfix) == (
        "loss",
        "smiles",
        "atom",
        "bond",
        "desc",
        "fp",
        "lr",
        "val",
    )
    assert postfix["lr"] == "2.00e-04"
    assert postfix["val"] == "5.5000"
