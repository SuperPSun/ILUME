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
    feature_cache: Path


@dataclass(frozen=True)
class FeatureConfig:
    kind: Literal["rdkit_2d", "ecfp4"]
    radius: int = 2
    n_bits: int = 2048


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
    name: Literal["mlp", "ecfp_xgboost"]
    data: DataConfig
    features: FeatureConfig
    model: dict[str, Any]
    training: dict[str, Any]
    stage3: Stage3BenchmarkConfig
    stage2_physics: Stage2PhysicsConfig
    seed: int

    def validate(self) -> None:
        if self.name not in {"mlp", "ecfp_xgboost"}:
            raise ValueError(f"Unknown benchmark model: {self.name}")
        if self.name == "mlp" and self.features.kind != "rdkit_2d":
            raise ValueError("MLP benchmark requires RDKit 2D descriptors")
        if self.name == "ecfp_xgboost" and self.features.kind != "ecfp4":
            raise ValueError("XGBoost benchmark requires ECFP4 features")
        if self.features.radius <= 0 or self.features.n_bits <= 0:
            raise ValueError("Fingerprint radius and n_bits must be positive")
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

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return value.as_posix()
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        return convert(asdict(self))


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
        {"name", "seed", "data", "features", "model", "training", "stage3", "stage2_physics"},
        "benchmark config",
    )
    data = _mapping(raw.get("data"), "data")
    _only(data, {"data_root", "task_catalog", "stage3_authority_config", "feature_cache"}, "data")
    features = _mapping(raw.get("features"), "features")
    _only(features, {"kind", "radius", "n_bits"}, "features")
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
        seed=int(raw.get("seed", 42)),
        data=DataConfig(
            data_root=Path(data["data_root"]),
            task_catalog=Path(data["task_catalog"]),
            stage3_authority_config=Path(data["stage3_authority_config"]),
            feature_cache=Path(data["feature_cache"]),
        ),
        features=FeatureConfig(
            kind=str(features["kind"]),  # type: ignore[arg-type]
            radius=int(features.get("radius", 2)),
            n_bits=int(features.get("n_bits", 2048)),
        ),
        model=_mapping(raw.get("model"), "model"),
        training=_mapping(raw.get("training"), "training"),
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
    "FeatureConfig",
    "benchmark_config_from_dict",
    "load_benchmark_config",
]
