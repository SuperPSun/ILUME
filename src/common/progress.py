from __future__ import annotations

import json
import sys
import os
from contextlib import contextmanager
from typing import Any, Iterator

from tqdm import tqdm


LOSS_POSTFIX_FIELDS = (
    ("loss", "loss"),
    ("smiles", "loss_smiles"),
    ("atom", "loss_atom"),
    ("bond", "loss_bond"),
    ("desc", "loss_descriptor"),
    ("fp", "loss_fingerprint"),
)


def loss_postfix(
    metrics: dict[str, float | int | str],
    *,
    include_learning_rate: bool,
    valid_loss: float | None = None,
) -> dict[str, str]:
    postfix = {
        label: f"{float(metrics[key]):.4f}"
        for label, key in LOSS_POSTFIX_FIELDS
        if key in metrics
    }
    if include_learning_rate and "learning_rate" in metrics:
        postfix["lr"] = f"{float(metrics['learning_rate']):.2e}"
    if valid_loss is not None:
        postfix["val"] = f"{valid_loss:.4f}"
    return postfix


class ProgressReporter:
    """TTY-aware progress bars with JSON fallback for redirected output."""

    def __init__(self, interactive: bool | None = None) -> None:
        disabled = os.environ.get("ILUME_DISABLE_PROGRESS") == "1"

        self.interactive = (
            sys.stderr.isatty() if interactive is None else interactive
        ) and not disabled

    def bar(
        self,
        *,
        total: int,
        desc: str,
        unit: str,
        initial: int = 0,
    ) -> Any:
        return tqdm(
            total=total,
            desc=desc,
            unit=unit,
            initial=initial,
            disable=not self.interactive,
            dynamic_ncols=True,
            mininterval=0.5,
            leave=True,
        )

    @contextmanager
    def status(self, desc: str) -> Iterator[None]:
        progress = tqdm(
            total=None,
            desc=desc,
            unit="stage",
            disable=not self.interactive,
            dynamic_ncols=True,
            mininterval=0.5,
            leave=True,
            bar_format="{desc}: {elapsed}",
        )
        try:
            yield
        finally:
            progress.close()

    def emit_json(self, payload: dict[str, Any]) -> None:
        if not self.interactive:
            print(json.dumps(payload, sort_keys=True))
