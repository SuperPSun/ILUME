from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from stage2.config import load_stage2_config
from stage3.config import load_stage3_config


@dataclass(frozen=True)
class Stage2TransferConfig:
    authority_config: Path
    stage1_checkpoint: Path
    prepared_artifacts: Path
    sources: tuple[str, ...]
    object_layers: int
    object_ffn_dim: int
    dropout: float
    batch_size: int
    epochs: int
    backbone_frozen_epochs: int
    backbone_learning_rate: float
    object_encoder_learning_rate: float
    task_head_learning_rate: float
    weight_decay: float
    warmup_fraction: float
    max_grad_norm: float
    amp_dtype: str
    optimizer: str
    physics_only: bool
    final_epoch: int
    sampling_mode: str = "full"


@dataclass(frozen=True)
class Stage3TransferConfig:
    authority_config: Path
    prepared_artifacts: Path
    targets: tuple[str, ...]
    folds: tuple[int, ...]
    hidden_dims: tuple[int, int]
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    smooth_l1_beta: float
    warmup_fraction: float
    min_lr_ratio: float
    max_grad_norm: float
    amp_dtype: str
    selection: str
    metric: str


@dataclass(frozen=True)
class TransferExperimentConfig:
    name: str
    seed: int
    stage2: Stage2TransferConfig
    stage3: Stage3TransferConfig

    def validate(self) -> None:
        if self.stage2.sampling_mode not in {"full", "balanced_rows"}:
            raise ValueError("Transfer sampling_mode must be full or balanced_rows")
        stage2 = load_stage2_config(self.stage2.authority_config)
        stage3 = load_stage3_config(self.stage3.authority_config)
        if self.seed != stage2.data.seed or self.seed != stage3.data.seed:
            raise ValueError("Transfer seed must match both Stage 2 and Stage 3 authorities")
        if len(self.stage2.sources) != 9 or len(set(self.stage2.sources)) != 9:
            raise ValueError("Transfer experiment requires exactly 9 unique Stage 2 sources")
        if tuple(sorted(self.stage2.sources)) != tuple(sorted(stage2.loss.task_weights)):
            raise ValueError("Transfer sources must exactly match the Stage 2 registry")
        if not self.stage2.physics_only or self.stage2.epochs != 10 or self.stage2.final_epoch != 10:
            raise ValueError("Stage 2 transfer variants require physics-only epoch-10 final state")
        if stage2.training.epochs != self.stage2.epochs:
            raise ValueError("Transfer Stage 2 epochs must match the v2 authority")
        expected_stage2 = {
            "stage1_checkpoint": stage2.initialization.checkpoint,
            "prepared_artifacts": stage2.data.artifacts_dir,
            "object_layers": stage2.model.object_layers,
            "object_ffn_dim": stage2.model.object_ffn_dim,
            "dropout": stage2.model.dropout,
            "batch_size": stage2.training.batch_size,
            "epochs": stage2.training.epochs,
            "backbone_frozen_epochs": stage2.training.backbone_frozen_epochs,
            "backbone_learning_rate": stage2.training.backbone_learning_rate,
            "object_encoder_learning_rate": stage2.training.object_encoder_learning_rate,
            "task_head_learning_rate": stage2.training.task_head_learning_rate,
            "weight_decay": stage2.training.weight_decay,
            "warmup_fraction": stage2.training.warmup_fraction,
            "max_grad_norm": stage2.training.max_grad_norm,
            "amp_dtype": stage2.training.amp_dtype,
        }
        for name, expected in expected_stage2.items():
            if getattr(self.stage2, name) != expected:
                raise ValueError(f"Transfer Stage 2 {name} differs from the v2 authority")
        if self.stage2.optimizer != "AdamW":
            raise ValueError("Transfer Stage 2 optimizer must be AdamW")
        if stage2.training.backbone_frozen_epochs != 1:
            raise ValueError("Transfer Stage 2 requires the v2 first-epoch freeze")
        if len(self.stage3.targets) != 21 or len(set(self.stage3.targets)) != 21:
            raise ValueError("Transfer experiment requires exactly 21 unique Stage 3 targets")
        if tuple(sorted(self.stage3.targets)) != tuple(sorted(stage3.enabled_task_ids)):
            raise ValueError("Transfer targets must exactly match enabled Stage 3 tasks")
        if self.stage3.prepared_artifacts != stage3.data.artifacts_dir:
            raise ValueError("Transfer Stage 3 prepared artifact differs from its authority")
        if self.stage3.folds != (1, 2, 3, 4, 5):
            raise ValueError("Transfer experiment requires folds 1 through 5")
        if self.stage3.hidden_dims != (512, 256) or self.stage3.dropout != 0.1:
            raise ValueError("Transfer MLP must use 512/256 hidden widths and dropout 0.1")
        if self.stage3.batch_size != 128 or self.stage3.epochs != 10:
            raise ValueError("Transfer MLP requires batch 128 and 10 epochs")
        if any(value <= 0 for value in (
            self.stage3.learning_rate, self.stage3.eps,
            self.stage3.smooth_l1_beta, self.stage3.max_grad_norm,
        )) or self.stage3.weight_decay < 0:
            raise ValueError("Transfer MLP optimizer values are invalid")
        if len(self.stage3.betas) != 2 or any(
            not 0 <= value < 1 for value in self.stage3.betas
        ):
            raise ValueError("Transfer MLP AdamW betas are invalid")
        if self.stage3.selection != "final" or self.stage3.metric != "validation_raw_mae":
            raise ValueError("Transfer MLP must publish final state and raw validation MAE")
        if not 0 <= self.stage3.warmup_fraction < 1 or not 0 < self.stage3.min_lr_ratio <= 1:
            raise ValueError("Invalid transfer MLP scheduler")
        if self.stage3.amp_dtype not in {"bf16", "none"}:
            raise ValueError("Transfer MLP amp_dtype must be bf16 or none")

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        result = convert(asdict(self))
        if self.stage2.sampling_mode == "full":
            result["stage2"].pop("sampling_mode")
        return result


def _strict(raw: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown {context} fields: {', '.join(sorted(unknown))}")


def transfer_config_from_dict(raw: dict[str, Any]) -> TransferExperimentConfig:
    _strict(raw, {"name", "seed", "stage2", "stage3"}, "transfer config")
    stage2_raw = dict(raw.get("stage2") or {})
    _strict(stage2_raw, {
        "authority_config", "stage1_checkpoint", "prepared_artifacts", "sources",
        "object_layers", "object_ffn_dim", "dropout", "batch_size", "epochs",
        "backbone_frozen_epochs", "backbone_learning_rate",
        "object_encoder_learning_rate", "task_head_learning_rate", "weight_decay",
        "warmup_fraction", "max_grad_norm", "amp_dtype", "optimizer",
        "physics_only", "final_epoch", "sampling_mode",
    }, "transfer stage2")
    stage3_raw = dict(raw.get("stage3") or {})
    _strict(stage3_raw, {
        "authority_config", "prepared_artifacts", "targets", "folds", "hidden_dims", "dropout",
        "batch_size", "epochs", "learning_rate", "weight_decay", "betas",
        "eps", "smooth_l1_beta", "warmup_fraction", "min_lr_ratio",
        "max_grad_norm", "amp_dtype", "selection", "metric",
    }, "transfer stage3")
    config = TransferExperimentConfig(
        name=str(raw["name"]),
        seed=int(raw["seed"]),
        stage2=Stage2TransferConfig(
            authority_config=Path(stage2_raw["authority_config"]),
            stage1_checkpoint=Path(stage2_raw["stage1_checkpoint"]),
            prepared_artifacts=Path(stage2_raw["prepared_artifacts"]),
            sources=tuple(stage2_raw["sources"]),
            object_layers=int(stage2_raw["object_layers"]),
            object_ffn_dim=int(stage2_raw["object_ffn_dim"]),
            dropout=float(stage2_raw["dropout"]),
            batch_size=int(stage2_raw["batch_size"]),
            epochs=int(stage2_raw["epochs"]),
            backbone_frozen_epochs=int(stage2_raw["backbone_frozen_epochs"]),
            backbone_learning_rate=float(stage2_raw["backbone_learning_rate"]),
            object_encoder_learning_rate=float(stage2_raw["object_encoder_learning_rate"]),
            task_head_learning_rate=float(stage2_raw["task_head_learning_rate"]),
            weight_decay=float(stage2_raw["weight_decay"]),
            warmup_fraction=float(stage2_raw["warmup_fraction"]),
            max_grad_norm=float(stage2_raw["max_grad_norm"]),
            amp_dtype=str(stage2_raw["amp_dtype"]),
            optimizer=str(stage2_raw["optimizer"]),
            physics_only=bool(stage2_raw["physics_only"]),
            final_epoch=int(stage2_raw["final_epoch"]),
            sampling_mode=str(stage2_raw.get("sampling_mode", "full")),
        ),
        stage3=Stage3TransferConfig(
            authority_config=Path(stage3_raw["authority_config"]),
            prepared_artifacts=Path(stage3_raw["prepared_artifacts"]),
            targets=tuple(stage3_raw["targets"]),
            folds=tuple(int(value) for value in stage3_raw["folds"]),
            hidden_dims=tuple(int(value) for value in stage3_raw["hidden_dims"]),  # type: ignore[arg-type]
            dropout=float(stage3_raw["dropout"]),
            batch_size=int(stage3_raw["batch_size"]),
            epochs=int(stage3_raw["epochs"]),
            learning_rate=float(stage3_raw["learning_rate"]),
            weight_decay=float(stage3_raw["weight_decay"]),
            betas=tuple(float(value) for value in stage3_raw["betas"]),  # type: ignore[arg-type]
            eps=float(stage3_raw["eps"]),
            smooth_l1_beta=float(stage3_raw["smooth_l1_beta"]),
            warmup_fraction=float(stage3_raw["warmup_fraction"]),
            min_lr_ratio=float(stage3_raw["min_lr_ratio"]),
            max_grad_norm=float(stage3_raw["max_grad_norm"]),
            amp_dtype=str(stage3_raw["amp_dtype"]),
            selection=str(stage3_raw["selection"]),
            metric=str(stage3_raw["metric"]),
        ),
    )
    config.validate()
    return config


def load_transfer_config(path: str | Path) -> TransferExperimentConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Transfer configuration root must be a mapping")
    return transfer_config_from_dict(raw)


__all__ = [
    "Stage2TransferConfig", "Stage3TransferConfig", "TransferExperimentConfig",
    "load_transfer_config", "transfer_config_from_dict",
]
