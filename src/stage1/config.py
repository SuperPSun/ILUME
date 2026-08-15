from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


STAGE1_CHECKPOINT_VERSION = 2
STAGE1_CHECKPOINT_KIND = "ilume_stage1_pretraining"


@dataclass(frozen=True)
class DataConfig:
    stage1_dir: Path = Path("data/stage1")
    artifacts_dir: Path = Path("outputs/v1/stage1/base/prepare/artifacts")
    valid_fraction: float = 0.05
    seed: int = 42
    max_smiles_tokens: int = 256
    descriptor_dim: int = 217
    max_samples_per_role: int | None = None
    shard_size: int = 8192
    shard_cache_size: int = 4
    include_augmentation: bool = False


@dataclass(frozen=True)
class TokenizerConfig:
    backend: str = "ais"
    vocab_size: int = 2048
    min_frequency: int = 1


@dataclass(frozen=True)
class DescriptorConfig:
    mode: str = "full"
    token_count: int = 1
    correlation_threshold: float = 0.98


@dataclass(frozen=True)
class FingerprintConfig:
    kind: str = "none"
    morgan_radius: int = 2
    morgan_bits: int = 2048
    maccs_bits: int = 167
    chunk_size: int = 128


@dataclass(frozen=True)
class PreparationConfig:
    workers: int = 1
    catalog_batch_size: int = 10000
    qc_batch_size: int = 2048
    tokenizer_batch_size: int = 2048
    descriptor_batch_size: int = 512


@dataclass(frozen=True)
class MaskingConfig:
    smiles_ratio: float = 0.15
    atom_ratio: float = 0.15
    bond_ratio: float = 0.15
    descriptor_ratio: float = 0.15
    fingerprint_ratio: float = 0.15
    smiles_dropout: float = 0.10
    graph_dropout: float = 0.10
    descriptor_dropout: float = 0.10
    fingerprint_dropout: float = 0.10
    dropout_schedule: str = "static"
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
    role_embedding: bool = True
    graph_head: str = "mlp"
    gradient_checkpointing: bool = False


@dataclass(frozen=True)
class LossConfig:
    lambda_smiles: float = 1.0
    lambda_descriptor: float = 1.0
    lambda_atom: float = 1.0
    lambda_bond: float = 1.0
    lambda_fingerprint: float = 1.0
    role_weights: tuple[float, float, float] = (2.0, 2.0, 1.0)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    max_grad_norm: float = 1.0
    num_workers: int = 8
    device: str = "auto"
    amp_dtype: str = "bf16"
    compile: bool = False
    validation_interval_steps: int = 5000
    quick_validation_samples_per_role: int = 256


@dataclass(frozen=True)
class PretrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    descriptor: DescriptorConfig = field(default_factory=DescriptorConfig)
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)
    preparation: PreparationConfig = field(default_factory=PreparationConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if not 0.0 < self.data.valid_fraction < 1.0:
            raise ValueError("data.valid_fraction must be between 0 and 1")
        if self.data.max_smiles_tokens < 3:
            raise ValueError("data.max_smiles_tokens must be at least 3")
        if self.data.descriptor_dim != 217:
            raise ValueError("data.descriptor_dim must remain the raw RDKit size 217")
        if self.data.shard_size <= 0 or self.data.shard_cache_size <= 0:
            raise ValueError("data shard sizes must be positive")
        if self.data.max_samples_per_role is not None and self.data.max_samples_per_role < 2:
            raise ValueError("data.max_samples_per_role must be at least 2 or null")
        if not isinstance(self.data.include_augmentation, bool):
            raise ValueError("data.include_augmentation must be true or false")
        if self.tokenizer.backend not in {"ais", "ape", "bpe", "spe"}:
            raise ValueError("tokenizer.backend must be ais, ape, bpe, or spe")
        if self.tokenizer.vocab_size <= len(("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")):
            raise ValueError("tokenizer.vocab_size is too small")
        if self.tokenizer.min_frequency < 1:
            raise ValueError("tokenizer.min_frequency must be positive")
        if self.descriptor.mode not in {"full", "clean", "pruned"}:
            raise ValueError("descriptor.mode must be full, clean, or pruned")
        if self.descriptor.token_count not in {1, 8, 12}:
            raise ValueError("descriptor.token_count must be 1, 8, or 12")
        if not 0.0 < self.descriptor.correlation_threshold < 1.0:
            raise ValueError("descriptor.correlation_threshold must be between 0 and 1")
        if self.fingerprint.kind not in {"none", "morgan", "maccs", "both"}:
            raise ValueError("fingerprint.kind must be none, morgan, maccs, or both")
        if (
            self.fingerprint.morgan_radius != 2
            or self.fingerprint.morgan_bits != 2048
            or self.fingerprint.maccs_bits != 167
            or self.fingerprint.chunk_size != 128
        ):
            raise ValueError(
                "fingerprint layout is fixed to Morgan radius=2/2048 bits, "
                "MACCS 167 bits, and 128-bit chunks"
            )
        preparation_values = {
            "preparation.workers": self.preparation.workers,
            "preparation.catalog_batch_size": self.preparation.catalog_batch_size,
            "preparation.qc_batch_size": self.preparation.qc_batch_size,
            "preparation.tokenizer_batch_size": self.preparation.tokenizer_batch_size,
            "preparation.descriptor_batch_size": self.preparation.descriptor_batch_size,
        }
        for name, value in preparation_values.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if len(self.loss.role_weights) != 3 or any(
            value <= 0 for value in self.loss.role_weights
        ):
            raise ValueError("loss.role_weights must contain three positive values")
        if self.model.d_model % self.model.n_heads:
            raise ValueError("model.d_model must be divisible by model.n_heads")
        if self.model.graph_head not in {"linear", "mlp"}:
            raise ValueError("model.graph_head must be linear or mlp")
        if self.masking.dropout_schedule not in {"off", "static", "curriculum"}:
            raise ValueError("masking.dropout_schedule must be off, static, or curriculum")
        probability_fields = {
            "masking.smiles_ratio": self.masking.smiles_ratio,
            "masking.atom_ratio": self.masking.atom_ratio,
            "masking.bond_ratio": self.masking.bond_ratio,
            "masking.descriptor_ratio": self.masking.descriptor_ratio,
            "masking.fingerprint_ratio": self.masking.fingerprint_ratio,
            "masking.smiles_dropout": self.masking.smiles_dropout,
            "masking.graph_dropout": self.masking.graph_dropout,
            "masking.descriptor_dropout": self.masking.descriptor_dropout,
            "masking.fingerprint_dropout": self.masking.fingerprint_dropout,
            "masking.asymmetric_probability": self.masking.asymmetric_probability,
            "masking.asymmetric_ratio": self.masking.asymmetric_ratio,
        }
        for name, value in probability_fields.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.training.batch_size <= 0 or self.training.epochs <= 0:
            raise ValueError("training.batch_size and training.epochs must be positive")
        if self.training.amp_dtype not in {"bf16", "fp16", "none"}:
            raise ValueError("training.amp_dtype must be bf16, fp16, or none")
        if not isinstance(self.training.compile, bool):
            raise ValueError("training.compile must be true or false")
        if not 0.0 <= self.training.warmup_fraction < 1.0:
            raise ValueError("training.warmup_fraction must be in [0, 1)")
        if self.training.validation_interval_steps <= 0:
            raise ValueError("training.validation_interval_steps must be positive")
        if self.training.quick_validation_samples_per_role <= 0:
            raise ValueError(
                "training.quick_validation_samples_per_role must be positive"
            )
        if self.training.num_workers < 0:
            raise ValueError("training.num_workers cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))

    def experiment_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("preparation")
        payload["training"].pop("compile")
        return payload


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "tokenizer": TokenizerConfig,
    "descriptor": DescriptorConfig,
    "fingerprint": FingerprintConfig,
    "preparation": PreparationConfig,
    "masking": MaskingConfig,
    "model": ModelConfig,
    "loss": LossConfig,
    "training": TrainingConfig,
}


def _construct(section_type: type, values: dict[str, Any] | None) -> Any:
    values = dict(values or {})
    known = {item.name for item in section_type.__dataclass_fields__.values()}
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"Unknown {section_type.__name__} fields: {', '.join(sorted(unknown))}"
        )
    if section_type is DataConfig:
        for key in ("stage1_dir", "artifacts_dir"):
            if key in values:
                values[key] = Path(values[key])
    elif section_type is LossConfig and "role_weights" in values:
        values["role_weights"] = tuple(values["role_weights"])
    return section_type(**values)


def config_from_dict(raw: dict[str, Any]) -> PretrainConfig:
    raw = dict(raw)
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown config sections: {', '.join(sorted(unknown))}")
    parts = {
        name: _construct(section, raw.get(name))
        for name, section in _SECTIONS.items()
    }
    config = PretrainConfig(**parts)
    config.validate()
    return config


def load_config(path: str | Path) -> PretrainConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    return config_from_dict(raw)
