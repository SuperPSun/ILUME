from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from stage1.config import load_config as load_stage1_config
from stage2.config import load_stage2_config
from stage3.config import load_stage3_config


ROOT = Path(__file__).resolve().parents[1]
STAGES = {"stage1", "stage2", "stage3"}
EXTERNAL_PACKAGES = {"ablations", "benchmarks"}


def test_global_rdkit_v2_base_configs_are_isolated() -> None:
    stage1 = load_stage1_config(ROOT / "configs/v2/stage1/base.yaml")
    stage2 = load_stage2_config(ROOT / "configs/v3/stage2/base.yaml")
    stage3 = load_stage3_config(ROOT / "configs/v3/stage3/base.yaml")
    legacy_stage3 = load_stage3_config(ROOT / "configs/v1/stage3/base.yaml")

    assert stage1.architecture.kind == "global_rdkit_v2"
    assert stage2.initialization.checkpoint == Path("outputs/v3/stage1/base/train/checkpoint_epoch_00005.pt")
    assert stage3.training.schedule_mode != legacy_stage3.training.schedule_mode
    assert "outputs/v3" in str(stage3.initialization.stage2_encoder)


def test_home_capacity_and_budget_candidates_are_isolated() -> None:
    from dataclasses import asdict
    from stage2.home_config import home_recipe_from_dict, load_home_recipe

    base2 = yaml.safe_load((ROOT / "configs/v3/stage2/base.yaml").read_text())
    base3 = yaml.safe_load((ROOT / "configs/v3/stage3/base.yaml").read_text())
    changes = {
        1: ("home.model.global_experts", "model.global_experts"),
        2: ("home.groups.thermophysical.experts", "home.groups.solvation.experts",
            "groups.thermophysical.experts", "groups.solvation.experts", "groups.electronic_structure.experts"),
        3: ("home.model.expert_hidden_ratio", "model.expert_hidden_ratio"),
        4: ("home.groups.thermophysical.expert_hidden_ratio", "home.groups.solvation.expert_hidden_ratio",
            "groups.thermophysical.expert_hidden_ratio", "groups.solvation.expert_hidden_ratio",
            "groups.electronic_structure.expert_hidden_ratio"),
        5: ("training.epochs",), 6: ("training.epochs",),
        7: ("training.object_encoder_phase1.epochs", "training.three_phase.global.epochs"),
        8: ("groups.phase_stability.phase2.epochs",),
        9: ("training.three_phase.private_classes.large.phase3_epochs",),
        10: ("training.three_phase.global.lr",),
    }
    values = {
        1: (3, 3), 2: (3, 4, 3, 4, 3), 3: (2.5, 2.5),
        4: (2.0, 2.0, 2.0, 2.0, 2.0), 5: (8,), 6: (12,),
        7: (18, 18), 8: (24,), 9: (10,), 10: (3.0e-4,),
    }
    for number in range(1, 11):
        name = f"base1-{number}"
        stage3_path = ROOT / f"configs/v3/stage3/candidates/{name}.yaml"
        raw3 = yaml.safe_load(stage3_path.read_text())
        stage3 = load_stage3_config(stage3_path)
        expected3 = deepcopy(base3)
        if number <= 6:
            for field in ("artifacts_dir",):
                expected3["data"][field] = raw3["data"][field]
            expected3["preparation"]["cache_dir"] = raw3["preparation"]["cache_dir"]
            for field in ("stage2_encoder", "stage2_final"):
                expected3["initialization"][field] = raw3["initialization"][field]
            assert name in raw3["initialization"]["stage2_final"]
            assert name in raw3["data"]["artifacts_dir"]
            stage2_path = ROOT / f"configs/v3/stage2/candidates/{name}.yaml"
            raw2 = yaml.safe_load(stage2_path.read_text())
            stage2 = load_home_recipe(stage2_path)
            assert raw2["data"]["artifacts_dir"] == base2["data"]["artifacts_dir"]
            assert asdict(stage2.stage3.model) == asdict(stage3.model)
            for group in ("thermophysical", "solvation"):
                assert asdict(stage2.stage3.groups[group]) == asdict(stage3.groups[group])
            assert stage3.groups["electronic_structure"].experts == stage3.groups["thermophysical"].experts
            assert stage3.groups["electronic_structure"].expert_hidden_ratio == stage3.groups["thermophysical"].expert_hidden_ratio
            expected2 = deepcopy(base2)
        else:
            assert raw3["initialization"] == base3["initialization"]
            assert raw3["data"] == base3["data"]
            assert not (ROOT / f"configs/v3/stage2/candidates/{name}.yaml").exists()
        for key, value in zip(changes[number], values[number], strict=True):
            parent = expected2 if number <= 6 and key.startswith(("home.", "training.epochs")) else expected3
            candidate = raw2 if parent is expected2 else raw3
            pieces = key.split(".")
            for part in pieces[:-1]:
                parent, candidate = parent[part], candidate[part]
            assert parent[pieces[-1]] != candidate[pieces[-1]]
            assert candidate[pieces[-1]] == value
            parent[pieces[-1]] = candidate[pieces[-1]]
        assert raw3 == expected3
        if number <= 6:
            assert raw2 == expected2

    invalid = deepcopy(base2)
    invalid["training"]["epochs"] = 9
    with pytest.raises(ValueError, match="fixed nine-task"):
        home_recipe_from_dict(invalid)


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
