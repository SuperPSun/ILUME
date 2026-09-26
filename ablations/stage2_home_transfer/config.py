from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from stage2.config import Stage2Config, load_stage2_config
from stage3.config import Stage3Config, load_stage3_config

from .contract import SOURCE_GROUPS


@dataclass(frozen=True)
class Experiment:
    stage2: Stage2Config
    stage3: Stage3Config
    stage2_microbatch_size: int
    stage2_epochs: int
    output_root: Path


def load_experiment(path: str | Path) -> Experiment:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {
        "kind", "format_version", "stage2_config", "stage3_base_config",
        "stage2_microbatch_size", "stage2_epochs", "output_root",
    } or raw["kind"] != "ilume_stage2_home_transfer_experiment" or raw["format_version"] != 1:
        raise ValueError("Invalid Stage2-HoME transfer experiment config")
    authority = load_stage2_config(raw["stage2_config"])
    stage2 = replace(
        authority,
        loss=replace(authority.loss, lambda_teacher=0.0),
        training=replace(authority.training, backbone_frozen_epochs=0),
    )
    stage3 = load_stage3_config(raw["stage3_base_config"])
    microbatch = raw["stage2_microbatch_size"]
    if type(microbatch) is not int or not 1 <= microbatch <= stage2.training.batch_size:
        raise ValueError("stage2_microbatch_size must be an integer in 1..256")
    if (
        int(raw["stage2_epochs"]) != 10
        or stage2.training.batch_size != 256
        or stage3.training.schedule_mode != "three_phase"
        or stage3.training.object_encoder_phase1 is None
        or stage2.loss.task_weights.keys() != SOURCE_GROUPS.keys()
        or len([task for task in stage3.tasks.values() if task.enabled]) != 20
    ):
        raise ValueError("Stage2-HoME transfer requires the frozen 9-to-20 Base recipe")
    return Experiment(stage2, stage3, microbatch, 10, Path(raw["output_root"]))
