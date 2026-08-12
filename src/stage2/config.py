from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


STAGE2_TASKS = (
    "simulated_qm_elec_hf",
    "density",
    "heat_capacity",
    "thermal_expansion",
    "transfer_organic",
)


@dataclass(frozen=True)
class Stage2DataConfig:
    stage2_dir: Path = Path("data/stage2")
    pretrain_artifacts_dir: Path = Path(
        "outputs/v1/stage1/base/prepare/artifacts"
    )
    artifacts_dir: Path = Path("outputs/v1/stage2/base/prepare/artifacts")
    entity_shard_size: int = 4096
    shard_cache_size: int = 10
    teacher_batch_size: int = 256
    transfer_validation_limit: int = 10000
    seed: int = 42


@dataclass(frozen=True)
class Stage2InitializationConfig:
    checkpoint: Path = Path(
        "outputs/v1/stage1/base/train/checkpoint_epoch_00005.pt"
    )


@dataclass(frozen=True)
class Stage2SamplingConfig:
    probabilities: dict[str, float] = field(
        default_factory=lambda: {
            "simulated_qm_elec_hf": 0.35,
            "density": 0.20,
            "heat_capacity": 0.15,
            "thermal_expansion": 0.15,
            "transfer_organic": 0.15,
        }
    )
    block_size: int = 20


@dataclass(frozen=True)
class Stage2ModelConfig:
    head_dropout: float = 0.10


@dataclass(frozen=True)
class Stage2LossConfig:
    lambda_alignment: float = 0.10


@dataclass(frozen=True)
class Stage2TrainingConfig:
    batch_size: int = 256
    gradient_accumulation_steps: int = 1
    max_steps: int = 23440
    backbone_learning_rate: float = 1.0e-5
    head_learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    backbone_freeze_fraction: float = 0.10
    max_grad_norm: float = 1.0
    device: str = "auto"
    amp_dtype: str = "bf16"
    validation_interval_steps: int = 1000
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1.0e-4
    save_every_n_steps: int | None = 1000


@dataclass(frozen=True)
class Stage2Config:
    data: Stage2DataConfig = field(default_factory=Stage2DataConfig)
    initialization: Stage2InitializationConfig = field(
        default_factory=Stage2InitializationConfig
    )
    sampling: Stage2SamplingConfig = field(default_factory=Stage2SamplingConfig)
    model: Stage2ModelConfig = field(default_factory=Stage2ModelConfig)
    loss: Stage2LossConfig = field(default_factory=Stage2LossConfig)
    training: Stage2TrainingConfig = field(default_factory=Stage2TrainingConfig)

    def validate(self) -> None:
        if self.data.entity_shard_size <= 0 or self.data.shard_cache_size <= 0:
            raise ValueError("Stage 2 shard sizes must be positive")
        if self.data.teacher_batch_size <= 0:
            raise ValueError("data.teacher_batch_size must be positive")
        if self.data.transfer_validation_limit <= 0:
            raise ValueError("data.transfer_validation_limit must be positive")
        probabilities = self.sampling.probabilities
        if set(probabilities) != set(STAGE2_TASKS):
            raise ValueError(
                "sampling.probabilities must define exactly: "
                + ", ".join(STAGE2_TASKS)
            )
        if any(value <= 0.0 for value in probabilities.values()):
            raise ValueError("Stage 2 task probabilities must be positive")
        if abs(sum(probabilities.values()) - 1.0) > 1.0e-8:
            raise ValueError("Stage 2 task probabilities must sum to 1")
        if self.sampling.block_size <= 0:
            raise ValueError("sampling.block_size must be positive")
        quotas = {
            task: probability * self.sampling.block_size
            for task, probability in probabilities.items()
        }
        if any(abs(value - round(value)) > 1.0e-8 for value in quotas.values()):
            raise ValueError(
                "sampling.block_size must produce integer task quotas"
            )
        if not 0.0 <= self.model.head_dropout <= 1.0:
            raise ValueError("model.head_dropout must be between 0 and 1")
        if self.loss.lambda_alignment < 0.0:
            raise ValueError("loss.lambda_alignment must be non-negative")
        training = self.training
        positive = {
            "training.batch_size": training.batch_size,
            "training.gradient_accumulation_steps": (
                training.gradient_accumulation_steps
            ),
            "training.max_steps": training.max_steps,
            "training.validation_interval_steps": (
                training.validation_interval_steps
            ),
            "training.early_stopping_patience": (
                training.early_stopping_patience
            ),
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if training.max_steps % self.sampling.block_size:
            raise ValueError(
                "training.max_steps must be divisible by sampling.block_size"
            )
        if training.backbone_learning_rate <= 0.0 or training.head_learning_rate <= 0.0:
            raise ValueError("Stage 2 learning rates must be positive")
        if training.weight_decay < 0.0:
            raise ValueError("training.weight_decay must be non-negative")
        if not 0.0 <= training.warmup_fraction < 1.0:
            raise ValueError("training.warmup_fraction must be in [0, 1)")
        if not 0.0 <= training.backbone_freeze_fraction < 1.0:
            raise ValueError(
                "training.backbone_freeze_fraction must be in [0, 1)"
            )
        if training.max_grad_norm < 0.0:
            raise ValueError("training.max_grad_norm must be non-negative")
        if training.amp_dtype not in {"bf16", "fp16", "none"}:
            raise ValueError("training.amp_dtype must be bf16, fp16, or none")
        if training.early_stopping_min_delta < 0.0:
            raise ValueError(
                "training.early_stopping_min_delta must be non-negative"
            )
        if (
            training.save_every_n_steps is not None
            and training.save_every_n_steps < 0
        ):
            raise ValueError("training.save_every_n_steps cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))


_SECTIONS: dict[str, type] = {
    "data": Stage2DataConfig,
    "initialization": Stage2InitializationConfig,
    "sampling": Stage2SamplingConfig,
    "model": Stage2ModelConfig,
    "loss": Stage2LossConfig,
    "training": Stage2TrainingConfig,
}


def _construct(
    section_type: type, values: dict[str, Any] | None, *, legacy: bool = False
) -> Any:
    values = dict(values or {})
    if section_type is Stage2TrainingConfig and legacy:
        for key in ("keep_last_checkpoints", "output_dir", "resume_from"):
            values.pop(key, None)
    known = set(section_type.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"Unknown {section_type.__name__} fields: "
            + ", ".join(sorted(unknown))
        )
    if section_type is Stage2DataConfig:
        for key in ("stage2_dir", "pretrain_artifacts_dir", "artifacts_dir"):
            if key in values:
                values[key] = Path(values[key])
    elif section_type is Stage2InitializationConfig:
        if "checkpoint" in values:
            values["checkpoint"] = Path(values["checkpoint"])
    return section_type(**values)


def _parse_stage2_config(raw: dict[str, Any], *, legacy: bool) -> Stage2Config:
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(
            "Unknown Stage 2 config sections: " + ", ".join(sorted(unknown))
        )
    parts = {
        name: _construct(section, raw.get(name), legacy=legacy)
        for name, section in _SECTIONS.items()
    }
    config = Stage2Config(**parts)
    config.validate()
    return config


def stage2_config_from_dict(raw: dict[str, Any]) -> Stage2Config:
    return _parse_stage2_config(raw, legacy=False)


def stage2_config_from_checkpoint_dict(raw: dict[str, Any]) -> Stage2Config:
    return _parse_stage2_config(raw, legacy=True)


def load_stage2_config(path: str | Path) -> Stage2Config:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Stage 2 configuration root must be a mapping")
    return stage2_config_from_dict(raw)


def backbone_unfreeze_step(config: Stage2Config) -> int:
    """Round the frozen phase to the nearest complete task block."""
    fraction = config.training.backbone_freeze_fraction
    if fraction == 0.0:
        return 0
    block_size = config.sampling.block_size
    block_count = config.training.max_steps // block_size
    frozen_blocks = math.floor(
        config.training.max_steps * fraction / block_size + 0.5
    )
    if block_count <= 1:
        return 0
    frozen_blocks = min(max(frozen_blocks, 1), block_count - 1)
    return frozen_blocks * block_size
