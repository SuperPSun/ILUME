from __future__ import annotations

import ast
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from common.training import (
    canonical_json_sha256,
    capture_rng_state,
    cosine_warmup,
    restore_rng_state,
    seed_everything,
)


ROOT = Path(__file__).resolve().parents[1]
STAGES = {"stage1", "stage2", "stage3"}


def _cross_stage_private_imports(
    source: str,
    current_stage: str,
) -> list[tuple[int, str]]:
    violations: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            target_stage = node.module.split(".", 1)[0]
            if target_stage in STAGES and target_stage != current_stage:
                for alias in node.names:
                    if alias.name == "*" or alias.name.startswith("_"):
                        violations.append((node.lineno, f"{node.module}.{alias.name}"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if (
                    parts[0] in STAGES
                    and parts[0] != current_stage
                    and any(part.startswith("_") for part in parts[1:])
                ):
                    violations.append((node.lineno, alias.name))
    return violations


def test_stage_packages_only_use_cross_stage_public_contracts() -> None:
    violations: list[str] = []
    for stage in sorted(STAGES):
        for path in sorted((ROOT / "src" / stage).rglob("*.py")):
            for line, imported in _cross_stage_private_imports(
                path.read_text(encoding="utf-8"), stage
            ):
                violations.append(f"{path.relative_to(ROOT)}:{line}: {imported}")
    assert violations == []


def test_cross_stage_import_rule_accepts_public_and_rejects_private() -> None:
    allowed = "from stage1.features import build_entity_sample\n"
    rejected = (
        "from stage1.features import _calculate_ipc\n"
        "from stage1.features import *\n"
        "import stage1._internal\n"
    )
    assert _cross_stage_private_imports(allowed, "stage2") == []
    assert [item for _, item in _cross_stage_private_imports(rejected, "stage2")] == [
        "stage1.features._calculate_ipc",
        "stage1.features.*",
        "stage1._internal",
    ]


def test_common_training_primitives_preserve_exact_contracts() -> None:
    payload = {"z": [1, 2], "a": {"value": 3.5}}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert canonical_json_sha256(payload) == hashlib.sha256(encoded).hexdigest()
    assert [cosine_warmup(step, 10, 0.2) for step in range(4)] == [
        0.5,
        1.0,
        1.0,
        0.9619397662556434,
    ]

    seed_everything(17)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual == expected
