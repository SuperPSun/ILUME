from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from stage3.config import Stage3GroupConfig, Stage3ModelConfig, Stage3OwnerBudgetConfig

from .config import DEFAULT_TASK_WEIGHTS, Stage2Config, stage2_config_from_dict
from .home_contract import SOURCE_GROUPS, PHYSICS_TASKS


@dataclass(frozen=True)
class HomeArchitecture:
    model: Stage3ModelConfig
    groups: dict[str, Stage3GroupConfig]


@dataclass(frozen=True)
class HomeRecipe:
    stage2: Stage2Config
    stage3: HomeArchitecture
    stage2_microbatch_size: int
    stage2_epochs: int
    initialization: str
    random_seed: int | None
    freeze_stage1: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.stage2.to_dict(),
            "home": {
                "microbatch_size": self.stage2_microbatch_size,
                "initialization": self.initialization,
                "random_seed": self.random_seed,
                "model": asdict(self.stage3.model),
                "groups": {name: asdict(group) for name, group in self.stage3.groups.items()},
                **({"freeze_stage1": True} if self.freeze_stage1 else {}),
            },
        }


def load_home_recipe(path: str | Path) -> HomeRecipe:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return home_recipe_from_dict(raw)


def home_recipe_from_dict(raw: Any) -> HomeRecipe:
    if not isinstance(raw, dict) or not isinstance(raw.get("home"), dict):
        raise ValueError("Stage 2 HoME config requires a home section")
    source = dict(raw)
    home = dict(source.pop("home"))
    freeze_stage1 = home.pop("freeze_stage1", False)
    if type(freeze_stage1) is not bool:
        raise ValueError("home.freeze_stage1 must be a boolean")
    if set(home) != {"microbatch_size", "initialization", "random_seed", "model", "groups"}:
        raise ValueError("Stage 2 HoME config has missing or unknown home fields")
    config = stage2_config_from_dict(source)
    if not isinstance(home["model"], dict) or set(home["model"]) - set(Stage3ModelConfig.__dataclass_fields__):
        raise ValueError("Invalid Stage 2 HoME model recipe")
    model = Stage3ModelConfig(**home["model"])
    if not isinstance(home["groups"], dict) or set(home["groups"]) != {"thermophysical", "solvation"}:
        raise ValueError("Stage 2 HoME requires thermophysical and solvation groups")
    groups = {}
    for name, value in home["groups"].items():
        if not isinstance(value, dict) or set(value) - set(Stage3GroupConfig.__dataclass_fields__):
            raise ValueError(f"Invalid Stage 2 HoME group: {name}")
        details = dict(value)
        for phase in ("phase1", "phase2"):
            if details.get(phase) is not None:
                details[phase] = Stage3OwnerBudgetConfig(**details[phase])
        groups[name] = Stage3GroupConfig(**details)
    microbatch = home["microbatch_size"]
    initialization = home["initialization"]
    random_seed = home["random_seed"]
    if (
        type(microbatch) is not int or not 1 <= microbatch <= config.training.batch_size
        or initialization not in {"pretrained", "random_stage1"}
        or (initialization == "random_stage1") != (type(random_seed) is int and random_seed >= 0)
        or (initialization == "random_stage1") != (config.initialization.stage1_config is not None)
        or (initialization == "random_stage1" and config.initialization.random_seed != random_seed)
        or config.representation is not None
        or config.loss.lambda_teacher != 0.0
        or config.training.backbone_frozen_epochs != 0
        or config.training.batch_size != 256
        or config.training.epochs not in {8, 10, 12}
        or config.training.refinement_epochs != 0
        or (set(config.loss.task_weights) not in (set(SOURCE_GROUPS), set(PHYSICS_TASKS)) if config.is_entity_home else set(config.loss.task_weights) != set(SOURCE_GROUPS))
        or (config.is_entity_home and (not freeze_stage1 or microbatch != 256 or config.training.epochs != 10
            or config.loss.task_weights != {task: DEFAULT_TASK_WEIGHTS[task] for task in config.data.tasks}))
    ):
        raise ValueError("Stage 2 HoME requires the fixed physics-only task and training recipe" if config.is_entity_home
                         else "Stage 2 HoME requires the fixed nine-task physics-only recipe")
    return HomeRecipe(config, HomeArchitecture(model, groups), microbatch, config.training.epochs, initialization, random_seed, freeze_stage1)
