from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = {"stage1", "stage2", "stage3"}
EXTERNAL_PACKAGES = {"ablations", "benchmarks"}


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


def test_stage_packages_do_not_import_benchmarks_or_ablations() -> None:
    violations = []
    for stage in sorted(STAGES):
        for path in sorted((ROOT / "src" / stage).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            for node in ast.walk(ast.parse(source)):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.split(".", 1)[0] in EXTERNAL_PACKAGES
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
                if isinstance(node, ast.Import) and any(
                    alias.name.split(".", 1)[0] in EXTERNAL_PACKAGES
                    for alias in node.names
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert violations == []
