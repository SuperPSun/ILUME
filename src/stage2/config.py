from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .registry import Stage2Registry


DEFAULT_TASK_WEIGHTS = {
    "simulation/density": 1.0,
    "simulation/heat_capacity": 0.8,
    "simulation/heat_of_vaporization": 1.0,
    "simulation/partial_atomic_charge": 1.0,
    "simulation/pbe_tzvp_anion_orbitals": 1.0,
    "simulation/pbe_tzvp_cation_orbitals": 1.0,
    "simulation/simulated_qm_elec_hf": 1.0,
    "simulation/thermal_expansion": 0.8,
    "simulation/transfer_organic": 0.5,
}

STAGE2_CHECKPOINT_VERSION = 3
STAGE2_CHECKPOINT_KIND = "ilume_stage2_object"


@dataclass(frozen=True)
class Stage2DataConfig:
    data_root: Path = Path("data")
    task_catalog_path: Path = Path("data/task_catalog.csv")
    pretrain_artifacts_dir: Path = Path("outputs/v1/stage1/base/prepare/artifacts")
    artifacts_dir: Path = Path("outputs/v1/stage2/base/prepare/artifacts")
    entity_shard_size: int = 4096
    seed: int = 42


@dataclass(frozen=True)
class Stage2PreparationConfig:
    workers: int = 8
    teacher_batch_size: int = 512


@dataclass(frozen=True)
class Stage2InitializationConfig:
    checkpoint: Path = Path("outputs/v1/stage1/base/train/checkpoint_epoch_00005.pt")


@dataclass(frozen=True)
class Stage2ModelConfig:
    object_layers: int = 2
    object_ffn_dim: int = 1024
    dropout: float = 0.10


@dataclass(frozen=True)
class Stage2LossConfig:
    lambda_teacher: float = 0.10
    task_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TASK_WEIGHTS))
    task_loss_modes: dict[str, str] = field(
        default_factory=lambda: {"simulation/simulated_qm_elec_hf": "masked_target_macro"}
    )


@dataclass(frozen=True)
class Stage2TrainingConfig:
    batch_size: int = 256
    gradient_accumulation_steps: int = 1
    epochs: int = 5
    backbone_frozen_epochs: int = 1
    backbone_learning_rate: float = 1.0e-5
    object_encoder_learning_rate: float = 3.0e-5
    task_head_learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    max_grad_norm: float = 1.0
    packing_workers: int = 4
    packing_prefetch_batches: int = 4
    cuda_prefetch_batches: int = 1
    log_every_batches: int = 50
    device: str = "auto"
    amp_dtype: str = "bf16"


@dataclass(frozen=True)
class Stage2Config:
    data: Stage2DataConfig = field(default_factory=Stage2DataConfig)
    preparation: Stage2PreparationConfig = field(default_factory=Stage2PreparationConfig)
    initialization: Stage2InitializationConfig = field(default_factory=Stage2InitializationConfig)
    model: Stage2ModelConfig = field(default_factory=Stage2ModelConfig)
    loss: Stage2LossConfig = field(default_factory=Stage2LossConfig)
    training: Stage2TrainingConfig = field(default_factory=Stage2TrainingConfig)

    def validate(self) -> None:
        if not self.data.task_catalog_path.resolve().is_relative_to(self.data.data_root.resolve()):
            raise ValueError("data.task_catalog_path must be contained by data.data_root")
        if self.data.entity_shard_size <= 0:
            raise ValueError("data.entity_shard_size must be positive")
        if self.preparation.workers <= 0 or self.preparation.teacher_batch_size <= 0:
            raise ValueError("Stage 2 preparation sizes must be positive")
        if self.model.object_layers <= 0 or self.model.object_ffn_dim <= 0:
            raise ValueError("Stage 2 ObjectEncoder dimensions must be positive")
        if not 0.0 <= self.model.dropout <= 1.0:
            raise ValueError("model.dropout must be between 0 and 1")
        if self.loss.lambda_teacher < 0.0:
            raise ValueError("loss.lambda_teacher must be non-negative")
        if not self.loss.task_weights or any(value <= 0 for value in self.loss.task_weights.values()):
            raise ValueError("Stage 2 task weights must be positive")
        if any(mode not in {"element_mean", "masked_target_macro"} for mode in self.loss.task_loss_modes.values()):
            raise ValueError("Unsupported Stage 2 task loss mode")
        training = self.training
        if training.batch_size <= 0 or training.epochs <= 0:
            raise ValueError("Stage 2 batch size and epochs must be positive")
        if training.gradient_accumulation_steps != 1:
            raise ValueError("Stage 2 Object v3 requires gradient_accumulation_steps == 1")
        if not 0 <= training.backbone_frozen_epochs < training.epochs:
            raise ValueError("training.backbone_frozen_epochs must be in [0, epochs)")
        if any(value <= 0 for value in (training.backbone_learning_rate, training.object_encoder_learning_rate, training.task_head_learning_rate)):
            raise ValueError("Stage 2 learning rates must be positive")
        if training.weight_decay < 0 or not 0 <= training.warmup_fraction < 1:
            raise ValueError("Invalid Stage 2 optimizer schedule")
        if training.max_grad_norm < 0:
            raise ValueError("training.max_grad_norm must be non-negative")
        if training.packing_workers <= 0 or training.packing_prefetch_batches <= 0 or training.log_every_batches <= 0:
            raise ValueError("Stage 2 execution sizes must be positive")
        if training.cuda_prefetch_batches != 1:
            raise ValueError("Stage 2 Object v3 requires cuda_prefetch_batches == 1")
        if training.amp_dtype not in {"bf16", "fp16", "none"}:
            raise ValueError("training.amp_dtype must be bf16, fp16, or none")

    def validate_registry(self, registry: Stage2Registry) -> None:
        expected = set(registry.task_ids)
        if set(self.loss.task_weights) != expected:
            raise ValueError("loss.task_weights must exactly match the Stage 2 registry")
        if not set(self.loss.task_loss_modes).issubset(expected):
            raise ValueError("loss.task_loss_modes contains an unknown Stage 2 task")
        for task in registry.tasks:
            mode = self.loss.task_loss_modes.get(task.task_id, "element_mean")
            if mode == "masked_target_macro" and task.target_level != "object":
                raise ValueError("masked_target_macro requires an object target")

    def normalized_task_weights(self, registry: Stage2Registry) -> dict[str, float]:
        self.validate_registry(registry)
        total = sum(self.loss.task_weights.values())
        return {task_id: self.loss.task_weights[task_id] / total for task_id in registry.task_ids}

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

    def experiment_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("preparation")
        for name in (
            "packing_workers", "packing_prefetch_batches",
            "cuda_prefetch_batches", "log_every_batches",
        ):
            payload["training"].pop(name)
        return payload


_SECTIONS = {
    "data": Stage2DataConfig,
    "preparation": Stage2PreparationConfig,
    "initialization": Stage2InitializationConfig,
    "model": Stage2ModelConfig,
    "loss": Stage2LossConfig,
    "training": Stage2TrainingConfig,
}


def _construct(section_type: type, values: dict[str, Any] | None) -> Any:
    values = dict(values or {})
    unknown = set(values) - set(section_type.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown {section_type.__name__} fields: " + ", ".join(sorted(unknown)))
    if section_type is Stage2DataConfig:
        for key in ("data_root", "task_catalog_path", "pretrain_artifacts_dir", "artifacts_dir"):
            if key in values:
                values[key] = Path(values[key])
    elif section_type is Stage2InitializationConfig and "checkpoint" in values:
        values["checkpoint"] = Path(values["checkpoint"])
    return section_type(**values)


def stage2_config_from_dict(raw: dict[str, Any]) -> Stage2Config:
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError("Unknown Stage 2 config sections: " + ", ".join(sorted(unknown)))
    config = Stage2Config(**{name: _construct(section, raw.get(name)) for name, section in _SECTIONS.items()})
    config.validate()
    return config


def stage2_config_from_checkpoint_dict(raw: dict[str, Any]) -> Stage2Config:
    return stage2_config_from_dict(raw)


def load_stage2_config(path: str | Path) -> Stage2Config:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Stage 2 configuration root must be a mapping")
    return stage2_config_from_dict(raw)
