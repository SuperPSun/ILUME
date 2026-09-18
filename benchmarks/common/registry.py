"""Lazy entrypoints for model-specific benchmark adapters and environments."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable


@dataclass(frozen=True)
class Adapter:
    package: str
    name: str
    isolated_environment: bool = True
    evaluation_audit: bool = True
    reports_progress: bool = True

    def load(self, operation: str) -> Callable[..., Any]:
        if operation == "environment":
            return getattr(import_module(f"{self.package}.environment"), f"validate_{self.name}_environment")
        names = {
            "prepare": f"prepare_{self.name}_training",
            "train": f"train_{self.name}_bundle",
            "evaluate": f"evaluate_{self.name}_checkpoint",
            "audit": f"{self.name}_evaluation_audit",
        }
        return getattr(import_module(f"{self.package}.adapter"), names[operation])


ADAPTERS = {
    "dmpnn": Adapter("benchmarks.dmpnn", "dmpnn", evaluation_audit=False, reports_progress=False),
    "molformer": Adapter("benchmarks.molformer", "molformer"),
    "ilbert": Adapter("benchmarks.ilbert", "ilbert"),
    "spmm": Adapter("benchmarks.spmm", "spmm"),
    "llasmol": Adapter("benchmarks.llasmol", "llasmol"),
    "aionopedia": Adapter("benchmarks.aionopedia", "aionopedia"),
    "iltransr": Adapter("benchmarks.iltransr", "iltransr"),
    "aifc": Adapter("benchmarks.aifc", "aifc"),
    "ilume_stage3_single_task_mlp": Adapter(
        "ablations.stage3_single_task_mlp", "stage3_single_task_mlp",
        isolated_environment=False, evaluation_audit=False,
    ),
}
