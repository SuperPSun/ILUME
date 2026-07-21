from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    stage1_dir: Path = Path("data/stage1")
    artifacts_dir: Path = Path("artifacts/smoke")
    valid_fraction: float = 0.05
    seed: int = 42
    max_smiles_tokens: int = 384
    descriptor_dim: int = 217
    max_samples_per_role: int | None = None


@dataclass(frozen=True)
class MaskingConfig:
    smiles_ratio: float = 0.15
    atom_ratio: float = 0.15
    bond_ratio: float = 0.15
    descriptor_ratio: float = 0.15
    smiles_dropout: float = 0.10
    graph_dropout: float = 0.10
    descriptor_dropout: float = 0.10
    asymmetric_enabled: bool = False
    asymmetric_probability: float = 0.25
    asymmetric_ratio: float = 0.50


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    smiles_layers: int = 2
    graph_depth: int = 3
    descriptor_hidden_dim: int = 256
    descriptor_blocks: int = 2
    fusion_layers: int = 2
    feedforward_dim: int = 512
    dropout: float = 0.10


@dataclass(frozen=True)
class LossConfig:
    lambda_smiles: float = 1.0
    lambda_descriptor: float = 1.0
    lambda_atom: float = 1.0
    lambda_bond: float = 1.0


@dataclass(frozen=True)
class SmokeConfig:
    batch_size: int = 10
    steps: int = 2
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True


@dataclass(frozen=True)
class PretrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    smoke: SmokeConfig = field(default_factory=SmokeConfig)

    def validate(self) -> None:
        if not 0.0 < self.data.valid_fraction < 1.0:
            raise ValueError("data.valid_fraction must be between 0 and 1")
        if self.data.max_smiles_tokens < 3:
            raise ValueError("data.max_smiles_tokens must be at least 3")
        if self.data.descriptor_dim <= 0:
            raise ValueError("data.descriptor_dim must be positive")
        if (
            self.data.max_samples_per_role is not None
            and self.data.max_samples_per_role < 2
        ):
            raise ValueError("data.max_samples_per_role must be at least 2 or null")
        if self.model.d_model % self.model.n_heads:
            raise ValueError("model.d_model must be divisible by model.n_heads")
        if self.smoke.batch_size <= 0 or self.smoke.steps <= 0:
            raise ValueError("smoke.batch_size and smoke.steps must be positive")
        probability_fields = {
            "masking.smiles_ratio": self.masking.smiles_ratio,
            "masking.atom_ratio": self.masking.atom_ratio,
            "masking.bond_ratio": self.masking.bond_ratio,
            "masking.descriptor_ratio": self.masking.descriptor_ratio,
            "masking.smiles_dropout": self.masking.smiles_dropout,
            "masking.graph_dropout": self.masking.graph_dropout,
            "masking.descriptor_dropout": self.masking.descriptor_dropout,
            "masking.asymmetric_probability": self.masking.asymmetric_probability,
            "masking.asymmetric_ratio": self.masking.asymmetric_ratio,
        }
        for name, value in probability_fields.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


def _construct(section_type: type, values: dict[str, Any] | None) -> Any:
    values = values or {}
    known = {item.name for item in section_type.__dataclass_fields__.values()}
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"Unknown {section_type.__name__} fields: {', '.join(sorted(unknown))}"
        )
    if section_type is DataConfig:
        values = values.copy()
        for key in ("stage1_dir", "artifacts_dir"):
            if key in values:
                values[key] = Path(values[key])
    return section_type(**values)


def load_config(path: str | Path) -> PretrainConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    known_sections = {"data", "masking", "model", "loss", "smoke"}
    unknown = set(raw) - known_sections
    if unknown:
        raise ValueError(f"Unknown config sections: {', '.join(sorted(unknown))}")
    config = PretrainConfig(
        data=_construct(DataConfig, raw.get("data")),
        masking=_construct(MaskingConfig, raw.get("masking")),
        model=_construct(ModelConfig, raw.get("model")),
        loss=_construct(LossConfig, raw.get("loss")),
        smoke=_construct(SmokeConfig, raw.get("smoke")),
    )
    config.validate()
    return config
