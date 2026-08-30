from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


BenchmarkName = Literal["stage3", "stage2_physics"]


@dataclass(frozen=True)
class DataConfig:
    data_root: Path
    task_catalog: Path
    stage3_authority_config: Path
    feature_cache: Path | None
    stage2_authority_config: Path | None = None


@dataclass(frozen=True)
class FeatureConfig:
    kind: Literal["rdkit_2d", "ecfp4"]
    radius: int = 2
    n_bits: int = 2048


@dataclass(frozen=True)
class EnvironmentConfig:
    name: str
    definition: Path
    lock: Path


@dataclass(frozen=True)
class Stage3BenchmarkConfig:
    enabled: bool
    tasks: Literal["all"] | tuple[str, ...]
    folds: tuple[int, ...]


@dataclass(frozen=True)
class Stage2PhysicsConfig:
    enabled: bool
    tasks: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkConfig:
    name: Literal["mlp", "ecfp_xgboost", "dmpnn", "molformer"]
    data: DataConfig
    features: FeatureConfig | None
    environment: EnvironmentConfig | None
    model: dict[str, Any]
    training: dict[str, Any]
    runtime: dict[str, Any]
    stage3: Stage3BenchmarkConfig
    stage2_physics: Stage2PhysicsConfig
    seed: int
    display_name: str = ""

    def validate(self) -> None:
        if self.name not in {"mlp", "ecfp_xgboost", "dmpnn", "molformer"}:
            raise ValueError(f"Unknown benchmark model: {self.name}")
        if not self.display_name:
            raise ValueError("Benchmark display_name must be non-empty")
        if self.name == "mlp" and (
            self.features is None or self.features.kind != "rdkit_2d"
        ):
            raise ValueError("MLP benchmark requires RDKit 2D descriptors")
        if self.name == "ecfp_xgboost" and (
            self.features is None or self.features.kind != "ecfp4"
        ):
            raise ValueError("XGBoost benchmark requires ECFP4 features")
        if self.features is not None and (
            self.features.radius <= 0 or self.features.n_bits <= 0
        ):
            raise ValueError("Fingerprint radius and n_bits must be positive")
        advanced = self.name in {"dmpnn", "molformer"}
        if not advanced and self.data.feature_cache is None:
            raise ValueError("Feature baselines require data.feature_cache")
        if not advanced and self.environment is not None:
            raise ValueError("Only advanced baselines use a dedicated environment")
        if self.name == "dmpnn":
            self._validate_dmpnn()
        if self.name == "molformer":
            self._validate_molformer()
        elif self.runtime:
            raise ValueError("Only MoLFormer currently declares benchmark runtime settings")
        if not self.model or not self.training or self.seed < 0:
            raise ValueError("Benchmark model/training contract is incomplete")
        if not self.stage3.folds or any(fold not in range(1, 6) for fold in self.stage3.folds):
            raise ValueError("Benchmark Stage 3 folds must be in 1..5")
        if len(self.stage3.folds) != len(set(self.stage3.folds)):
            raise ValueError("Benchmark Stage 3 folds must be unique")
        if self.stage3.enabled and self.stage3.tasks != "all" and not self.stage3.tasks:
            raise ValueError("Enabled Stage 3 benchmark has no tasks")
        if self.stage2_physics.enabled and not self.stage2_physics.tasks:
            raise ValueError("Enabled Stage 2 physics benchmark has no tasks")

    def _validate_dmpnn(self) -> None:
        if self.features is not None:
            raise ValueError("D-MPNN uses Chemprop graphs and does not accept features")
        if self.environment is None or not all(
            (self.environment.name, str(self.environment.definition), str(self.environment.lock))
        ):
            raise ValueError("D-MPNN requires a dedicated environment definition and lock")
        if self.environment.name != "ilume-dmpnn":
            raise ValueError("D-MPNN environment name must be ilume-dmpnn")
        if self.data.stage2_authority_config is None:
            raise ValueError("D-MPNN requires data.stage2_authority_config")
        expected_model = {
            "message_hidden_dim": 300,
            "depth": 3,
            "dropout": 0.0,
            "activation": "relu",
            "aggregation": "norm",
            "aggregation_norm": 100.0,
            "ffn_hidden_dim": 300,
            "ffn_hidden_layers": 1,
            "batch_norm": False,
            "multicomponent_shared": False,
        }
        if self.model != expected_model:
            raise ValueError("D-MPNN model must match the registered Chemprop recipe")
        expected_training = {
            "optimizer": "adam",
            "scheduler": "noam",
            "warmup_epochs": 2,
            "initial_learning_rate": 1.0e-4,
            "max_learning_rate": 1.0e-3,
            "final_learning_rate": 1.0e-4,
            "batch_size": 64,
            "max_epochs": 50,
            "early_stopping_patience": 10,
            "loss": "mse",
            "selection_metric": "validation_mae",
            "device": "cuda",
            "precision": "fp32",
        }
        if self.training != expected_training:
            raise ValueError("D-MPNN training must match the registered Chemprop recipe")

    def _validate_molformer(self) -> None:
        if self.features is not None:
            raise ValueError("MoLFormer tokenizes SMILES and does not accept features")
        if self.environment is None or not all(
            (self.environment.name, str(self.environment.definition), str(self.environment.lock))
        ):
            raise ValueError("MoLFormer requires a dedicated environment definition and lock")
        if self.environment.name != "ilume-molformer":
            raise ValueError("MoLFormer environment name must be ilume-molformer")
        if self.data.stage2_authority_config is not None:
            raise ValueError("MoLFormer does not use a Stage 2 prepare authority")
        expected_model = {
            "repository": "ibm-research/MoLFormer-XL-both-10pct",
            "revision": "361063d0ad524ef77cf39b08469f6be770dc550f",
            "trust_remote_code": True,
            "pretrained": True,
            "full_fine_tuning": True,
            "deterministic_eval": True,
            "remove_stereochemistry": True,
            "max_input_tokens": 202,
            "hidden_dim": 768,
            "shared_backbone": True,
            "fusion": "ordered_concat_linear",
            "regression_head": "official_sequence_classification",
            "input_cache": "unique_smiles_memory_token_cache",
            "component_forward": "merged_component_backbone_forward",
        }
        if self.model != expected_model:
            raise ValueError("MoLFormer model must match the registered pretrained recipe")
        expected_training = {
            "optimizer": "adamw",
            "encoder_learning_rate": 5.0e-6,
            "new_parameter_learning_rate": 5.0e-5,
            "weight_decay": 1.0e-2,
            "scheduler": "cosine",
            "warmup_fraction": 0.05,
            "batch_size": 128,
            "gradient_accumulation_steps": 1,
            "max_epochs": 50,
            "early_stopping_patience": 8,
            "tf32": True,
            "length_bucketing": "sortish_length_bucketing_v1",
            "bucket_window_batches": 20,
            "loss": "mse",
            "selection_metric": "validation_mae",
            "device": "cuda",
            "precision": "fp32",
        }
        if self.training != expected_training:
            raise ValueError("MoLFormer training must match the registered fine-tuning recipe")
        expected_runtime = {
            "num_workers": 4,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "pin_memory": True,
            "non_blocking_transfer": True,
        }
        if self.runtime != expected_runtime:
            raise ValueError("MoLFormer runtime must match the registered throughput recipe")

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return value.as_posix()
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        payload = convert(asdict(self))
        if not payload["runtime"]:
            payload.pop("runtime")
        return payload


def _mapping(raw: Any, context: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a mapping")
    return dict(raw)


def _only(values: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {context} fields: " + ", ".join(sorted(unknown)))


def benchmark_config_from_dict(raw: dict[str, Any]) -> BenchmarkConfig:
    _only(
        raw,
        {"name", "display_name", "seed", "data", "features", "environment", "model", "training", "runtime", "stage3", "stage2_physics"},
        "benchmark config",
    )
    data = _mapping(raw.get("data"), "data")
    _only(
        data,
        {"data_root", "task_catalog", "stage3_authority_config", "feature_cache", "stage2_authority_config"},
        "data",
    )
    features_raw = raw.get("features")
    features = None if features_raw is None else _mapping(features_raw, "features")
    if features is not None:
        _only(features, {"kind", "radius", "n_bits"}, "features")
    environment_raw = raw.get("environment")
    environment = (
        None
        if environment_raw is None
        else _mapping(environment_raw, "environment")
    )
    if environment is not None:
        _only(environment, {"name", "definition", "lock"}, "environment")
    stage3 = _mapping(raw.get("stage3"), "stage3")
    _only(stage3, {"enabled", "tasks", "folds"}, "stage3")
    stage2 = _mapping(raw.get("stage2_physics"), "stage2_physics")
    _only(stage2, {"enabled", "tasks"}, "stage2_physics")
    tasks: Literal["all"] | tuple[str, ...]
    if stage3.get("tasks") == "all":
        tasks = "all"
    elif isinstance(stage3.get("tasks"), list):
        tasks = tuple(str(value) for value in stage3["tasks"])
    else:
        raise ValueError("stage3.tasks must be 'all' or a list")
    config = BenchmarkConfig(
        name=str(raw.get("name")),  # type: ignore[arg-type]
        display_name=str(raw.get("display_name", str(raw.get("name", "")).upper())),
        seed=int(raw.get("seed", 42)),
        data=DataConfig(
            data_root=Path(data["data_root"]),
            task_catalog=Path(data["task_catalog"]),
            stage3_authority_config=Path(data["stage3_authority_config"]),
            feature_cache=(
                None if data.get("feature_cache") is None else Path(data["feature_cache"])
            ),
            stage2_authority_config=(
                None
                if data.get("stage2_authority_config") is None
                else Path(data["stage2_authority_config"])
            ),
        ),
        features=(
            None
            if features is None
            else FeatureConfig(
                kind=str(features["kind"]),  # type: ignore[arg-type]
                radius=int(features.get("radius", 2)),
                n_bits=int(features.get("n_bits", 2048)),
            )
        ),
        environment=(
            None
            if environment is None
            else EnvironmentConfig(
                name=str(environment["name"]),
                definition=Path(environment["definition"]),
                lock=Path(environment["lock"]),
            )
        ),
        model=_mapping(raw.get("model"), "model"),
        training=_mapping(raw.get("training"), "training"),
        runtime=_mapping(raw.get("runtime", {}), "runtime"),
        stage3=Stage3BenchmarkConfig(
            enabled=bool(stage3.get("enabled", False)),
            tasks=tasks,
            folds=tuple(int(value) for value in stage3.get("folds", ())),
        ),
        stage2_physics=Stage2PhysicsConfig(
            enabled=bool(stage2.get("enabled", False)),
            tasks=tuple(str(value) for value in stage2.get("tasks", ())),
        ),
    )
    config.validate()
    return config


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Benchmark configuration root must be a mapping")
    return benchmark_config_from_dict(raw)


__all__ = [
    "BenchmarkConfig",
    "BenchmarkName",
    "EnvironmentConfig",
    "FeatureConfig",
    "benchmark_config_from_dict",
    "load_benchmark_config",
]
