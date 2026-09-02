from __future__ import annotations

import ast
from pathlib import Path

from stage1.config import load_config as load_stage1_config
from stage2.config import load_stage2_config
from stage3.config import load_stage3_config

ROOT = Path(__file__).resolve().parents[1]
STAGES = {"stage1", "stage2", "stage3"}
EXTERNAL_PACKAGES = {"ablations", "benchmarks"}


def test_global_rdkit_v2_base_configs_are_isolated() -> None:
    stage1 = load_stage1_config(ROOT / "configs/v2/stage1/base.yaml")
    stage2 = load_stage2_config(ROOT / "configs/v2/stage2/base.yaml")
    stage3 = load_stage3_config(ROOT / "configs/v2/stage3/base.yaml")
    legacy_stage3 = load_stage3_config(ROOT / "configs/v1/stage3/base.yaml")

    assert stage1.architecture.kind == "global_rdkit_v2"
    assert stage1.data.descriptor_dim == 217
    assert stage1.descriptor.mode == "full"
    assert stage1.descriptor.token_count == 1
    assert stage1.model.d_model == 512
    assert "fingerprint" not in stage1.to_dict()
    assert stage2.model.object_layers == 2
    assert stage2.model.object_ffn_dim == 2048
    assert stage2.loss.lambda_teacher == 0.10
    assert "outputs/v2" in str(stage2.initialization.checkpoint)
    assert stage3.model == legacy_stage3.model
    assert stage3.groups == legacy_stage3.groups
    assert stage3.tasks == legacy_stage3.tasks
    assert stage3.training == legacy_stage3.training
    assert "outputs/v2" in str(stage3.initialization.stage2_encoder)


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
