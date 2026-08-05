from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


IL21_TASKS = (
    "experiment/density",
    "experiment/dynamic_relative_permittivity",
    "experiment/electrical_conductivity",
    "experiment/equilibrium_pressure",
    "experiment/glass_transition_temperature",
    "experiment/heat_capacity",
    "experiment/isobaric_coefficient_of_volume_expansion",
    "experiment/melting_point",
    "experiment/pec50",
    "experiment/refractive_index",
    "experiment/self_diffusion_coefficient",
    "experiment/speed_of_sound",
    "experiment/static_relative_permittivity",
    "experiment/surface_tension",
    "experiment/thermal_conductivity",
    "experiment/thermal_decomposition_temperature",
    "experiment/viscosity",
    "experiment/x_co2",
    "simulation/heat_of_vaporization",
    "experiment/solvation",
    "experiment/transfer",
)
AUX6_TASKS = (
    "experiment/transfer_organic",
    "simulation/cation_homo",
    "simulation/cation_lumo",
    "simulation/anion_homo",
    "simulation/anion_lumo",
    "simulation/charge",
)
STAGE3_TASKS = IL21_TASKS + AUX6_TASKS
DOMAIN_TASKS = {"il21": IL21_TASKS, "aux6": AUX6_TASKS}
DOMAIN_NAMES = tuple(DOMAIN_TASKS)


@dataclass(frozen=True)
class Stage3DataConfig:
    stage3_dir: Path = Path("data/stage3")
    artifacts_dir: Path = Path("artifacts/stage3_v2/data")
    entity_batch_size: int = 256
    seed: int = 42


@dataclass(frozen=True)
class Stage3InitializationConfig:
    stage2_checkpoint: Path = Path(
        "artifacts/stage_v2/training/comparisons/base_reference/best.pt"
    )


@dataclass(frozen=True)
class Stage3ModelConfig:
    architecture: str = "home"
    global_experts: int = 2
    group_experts: int = 2
    private_experts: int = 1
    dropout: float = 0.10
    feature_gate: bool = True
    self_gate: bool = True
    solute_injection: str = "late"


@dataclass(frozen=True)
class Stage3DomainTrainingConfig:
    batch_size: int = 64
    max_blocks: int = 10000
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    warmup_fraction: float = 0.10
    max_grad_norm: float = 1.0
    amp_dtype: str = "bf16"
    validation_interval_blocks: int = 100
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1.0e-4
    backward_mode: str = "per_task"


@dataclass(frozen=True)
class Stage3TrainingConfig:
    il21: Stage3DomainTrainingConfig = field(
        default_factory=Stage3DomainTrainingConfig
    )
    aux6: Stage3DomainTrainingConfig = field(
        default_factory=Stage3DomainTrainingConfig
    )
    device: str = "auto"
    cpu_threads: int = 4
    cpu_interop_threads: int = 1
    resident_data: bool = True
    keep_last_checkpoints: int = 3
    output_dir: Path = Path("artifacts/stage3_v2/training/home/fold1")
    resume_from: Path | None = None


@dataclass(frozen=True)
class Stage3Config:
    active_domains: tuple[str, ...] = DOMAIN_NAMES
    data: Stage3DataConfig = field(default_factory=Stage3DataConfig)
    initialization: Stage3InitializationConfig = field(
        default_factory=Stage3InitializationConfig
    )
    model: Stage3ModelConfig = field(default_factory=Stage3ModelConfig)
    training: Stage3TrainingConfig = field(default_factory=Stage3TrainingConfig)

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(
            task
            for domain in self.active_domains
            for task in DOMAIN_TASKS[domain]
        )

    def tasks_for_domain(self, domain: str) -> tuple[str, ...]:
        if domain not in self.active_domains:
            raise ValueError(f"Inactive Stage 3 domain: {domain}")
        return DOMAIN_TASKS[domain]

    def domain_training(self, domain: str) -> Stage3DomainTrainingConfig:
        if domain not in self.active_domains:
            raise ValueError(f"Inactive Stage 3 domain: {domain}")
        return getattr(self.training, domain)

    def validate(self) -> None:
        if not self.active_domains:
            raise ValueError("Stage 3 requires at least one active domain")
        if len(set(self.active_domains)) != len(self.active_domains):
            raise ValueError("Stage 3 active_domains cannot contain duplicates")
        if any(domain not in DOMAIN_NAMES for domain in self.active_domains):
            raise ValueError("Stage 3 active_domains must contain il21 and/or aux6")
        if tuple(
            domain for domain in DOMAIN_NAMES if domain in self.active_domains
        ) != self.active_domains:
            raise ValueError("Stage 3 active_domains must use canonical order")
        if self.data.entity_batch_size <= 0:
            raise ValueError("data.entity_batch_size must be positive")
        if self.model.architecture not in {"home", "shared_bottom", "mmoe"}:
            raise ValueError(
                "model.architecture must be home, shared_bottom, or mmoe"
            )
        if self.model.solute_injection not in {"late", "early"}:
            raise ValueError("model.solute_injection must be late or early")
        for name in ("global_experts", "group_experts", "private_experts"):
            if getattr(self.model, name) <= 0:
                raise ValueError(f"model.{name} must be positive")
        if not 0.0 <= self.model.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        for domain in self.active_domains:
            training = self.domain_training(domain)
            for name in (
                "batch_size",
                "max_blocks",
                "validation_interval_blocks",
                "early_stopping_patience",
            ):
                if getattr(training, name) <= 0:
                    raise ValueError(f"training.{domain}.{name} must be positive")
            if training.batch_size < 2:
                raise ValueError(
                    f"training.{domain}.batch_size must be at least 2 for BatchNorm"
                )
            if training.learning_rate <= 0.0 or training.weight_decay < 0.0:
                raise ValueError(
                    f"training.{domain} learning rate must be positive and "
                    "weight decay non-negative"
                )
            if not 0.0 <= training.warmup_fraction < 1.0:
                raise ValueError(
                    f"training.{domain}.warmup_fraction must be in [0, 1)"
                )
            if training.max_grad_norm < 0.0:
                raise ValueError(
                    f"training.{domain}.max_grad_norm must be non-negative"
                )
            if training.amp_dtype not in {"bf16", "fp16", "none"}:
                raise ValueError(
                    f"training.{domain}.amp_dtype must be bf16, fp16, or none"
                )
            if training.backward_mode not in {"per_task", "domain"}:
                raise ValueError(
                    f"training.{domain}.backward_mode must be per_task or domain"
                )
            if training.early_stopping_min_delta < 0.0:
                raise ValueError(
                    f"training.{domain}.early_stopping_min_delta must be non-negative"
                )
        if self.training.keep_last_checkpoints <= 0:
            raise ValueError("training.keep_last_checkpoints must be positive")
        if self.training.cpu_threads <= 0:
            raise ValueError("training.cpu_threads must be positive")
        if self.training.cpu_interop_threads <= 0:
            raise ValueError("training.cpu_interop_threads must be positive")
        if not isinstance(self.training.resident_data, bool):
            raise ValueError("training.resident_data must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))


def _construct(section_type: type, values: dict[str, Any] | None) -> Any:
    values = dict(values or {})
    unknown = set(values) - set(section_type.__dataclass_fields__)
    if unknown:
        raise ValueError(
            f"Unknown {section_type.__name__} fields: "
            + ", ".join(sorted(unknown))
        )
    if section_type is Stage3DataConfig:
        for key in ("stage3_dir", "artifacts_dir"):
            if key in values:
                values[key] = Path(values[key])
    elif section_type is Stage3InitializationConfig:
        if "stage2_checkpoint" in values:
            values["stage2_checkpoint"] = Path(values["stage2_checkpoint"])
    elif section_type is Stage3TrainingConfig:
        for domain in DOMAIN_NAMES:
            if domain in values:
                values[domain] = _construct(
                    Stage3DomainTrainingConfig, values[domain]
                )
        for key in ("output_dir", "resume_from"):
            if key in values and values[key] is not None:
                values[key] = Path(values[key])
    return section_type(**values)


def stage3_config_from_dict(raw: dict[str, Any]) -> Stage3Config:
    sections = {
        "data": Stage3DataConfig,
        "initialization": Stage3InitializationConfig,
        "model": Stage3ModelConfig,
        "training": Stage3TrainingConfig,
    }
    unknown = set(raw) - ({"active_domains"} | set(sections))
    if unknown:
        raise ValueError(
            "Unknown Stage 3 config sections: " + ", ".join(sorted(unknown))
        )
    active_domains = tuple(raw.get("active_domains", DOMAIN_NAMES))
    config = Stage3Config(
        active_domains=active_domains,
        **{
            name: _construct(section, raw.get(name))
            for name, section in sections.items()
        },
    )
    config.validate()
    return config


def load_stage3_config(path: str | Path) -> Stage3Config:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Stage 3 configuration root must be a mapping")
    return stage3_config_from_dict(raw)
