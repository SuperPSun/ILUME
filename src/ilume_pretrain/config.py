from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


AugmentationLimit = float | str


@dataclass(frozen=True)
class DataConfig:
    stage1_dir: Path = Path("data/stage1")
    artifacts_dir: Path = Path("artifacts/smoke")
    valid_fraction: float = 0.05
    seed: int = 42
    max_smiles_tokens: int = 256
    descriptor_dim: int = 217
    max_samples_per_role: int | None = None
    shard_size: int = 8192
    shard_cache_size: int = 4
    augmentation: dict[str, AugmentationLimit] = field(
        default_factory=lambda: {"cation": 0.0, "anion": 0.0, "neutral": 0.0}
    )


@dataclass(frozen=True)
class TokenizerConfig:
    backend: str = "ais"
    vocab_size: int = 2048
    min_frequency: int = 2


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
class SamplingConfig:
    role_probabilities: tuple[float, float, float] = (0.45, 0.45, 0.10)
    require_full_coverage: bool = True


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
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 5
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    max_grad_norm: float = 1.0
    num_workers: int = 0
    device: str = "auto"
    amp_dtype: str = "bf16"
    validation_interval_epochs: int = 1
    validation_batches: int = 20
    checkpoint_interval_epochs: int = 1
    keep_last_checkpoints: int = 3
    output_dir: Path = Path("artifacts/training")
    resume_from: Path | None = None


@dataclass(frozen=True)
class PretrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    descriptor: DescriptorConfig = field(default_factory=DescriptorConfig)
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    smoke: SmokeConfig = field(default_factory=SmokeConfig)
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
        for role in ("cation", "anion", "neutral"):
            value = self.data.augmentation.get(role, 0.0)
            if value != "all" and (not isinstance(value, (int, float)) or value < 0):
                raise ValueError(f"data.augmentation.{role} must be non-negative or 'all'")
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
        probabilities = self.sampling.role_probabilities
        if len(probabilities) != 3 or any(value <= 0 for value in probabilities):
            raise ValueError("sampling.role_probabilities must contain three positive values")
        if abs(sum(probabilities) - 1.0) > 1e-8:
            raise ValueError("sampling.role_probabilities must sum to 1")
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
        if self.smoke.batch_size <= 0 or self.smoke.steps <= 0:
            raise ValueError("smoke.batch_size and smoke.steps must be positive")
        if self.training.batch_size <= 0 or self.training.epochs <= 0:
            raise ValueError("training.batch_size and training.epochs must be positive")
        if self.training.gradient_accumulation_steps <= 0:
            raise ValueError("training.gradient_accumulation_steps must be positive")
        if self.training.amp_dtype not in {"bf16", "fp16", "none"}:
            raise ValueError("training.amp_dtype must be bf16, fp16, or none")
        if not 0.0 <= self.training.warmup_fraction < 1.0:
            raise ValueError("training.warmup_fraction must be in [0, 1)")
        if self.training.validation_batches <= 0:
            raise ValueError("training.validation_batches must be positive")
        if self.training.keep_last_checkpoints <= 0:
            raise ValueError("training.keep_last_checkpoints must be positive")
        if (
            self.training.validation_interval_epochs < 0
            or self.training.checkpoint_interval_epochs < 0
        ):
            raise ValueError("training epoch intervals cannot be negative")
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


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "tokenizer": TokenizerConfig,
    "descriptor": DescriptorConfig,
    "fingerprint": FingerprintConfig,
    "sampling": SamplingConfig,
    "masking": MaskingConfig,
    "model": ModelConfig,
    "loss": LossConfig,
    "smoke": SmokeConfig,
    "training": TrainingConfig,
}


def _construct(section_type: type, values: dict[str, Any] | None) -> Any:
    values = dict(values or {})
    known = {item.name for item in section_type.__dataclass_fields__.values()}
    unknown = set(values) - known
    if unknown:
        legacy_training_fields = {
            "max_steps",
            "validation_interval",
            "checkpoint_interval",
        }
        if section_type is TrainingConfig and unknown & legacy_training_fields:
            fields = ", ".join(sorted(unknown & legacy_training_fields))
            raise ValueError(
                "Step-based training fields are incompatible with the epoch "
                f"trainer: {fields}"
            )
        raise ValueError(
            f"Unknown {section_type.__name__} fields: {', '.join(sorted(unknown))}"
        )
    if section_type is DataConfig:
        for key in ("stage1_dir", "artifacts_dir"):
            if key in values:
                values[key] = Path(values[key])
        if "augmentation" in values:
            raw = values["augmentation"] or {}
            values["augmentation"] = {
                "cation": raw.get("cation", 0.0),
                "anion": raw.get("anion", 0.0),
                "neutral": raw.get("neutral", raw.get("molecule", 0.0)),
            }
    elif section_type is SamplingConfig and "role_probabilities" in values:
        values["role_probabilities"] = tuple(values["role_probabilities"])
    elif section_type is TrainingConfig:
        for key in ("output_dir", "resume_from"):
            if key in values and values[key] is not None:
                values[key] = Path(values[key])
    return section_type(**values)


def load_config(path: str | Path) -> PretrainConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown config sections: {', '.join(sorted(unknown))}")
    parts = {name: _construct(section, raw.get(name)) for name, section in _SECTIONS.items()}
    config = PretrainConfig(**parts)
    config.validate()
    return config
