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
    stage3_prepared_artifacts: Path | None = None


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
    name: Literal[
        "mlp", "ecfp_xgboost", "dmpnn", "molformer", "ilbert", "spmm",
        "ilume_stage3_single_task_mlp",
    ]
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
        if self.name not in {
            "mlp", "ecfp_xgboost", "dmpnn", "molformer", "ilbert", "spmm",
            "ilume_stage3_single_task_mlp",
        }:
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
        if self.name == "ilume_stage3_single_task_mlp":
            self._validate_ilume_stage3_single_task_mlp()
        advanced = self.name in {
            "dmpnn", "molformer", "ilbert", "spmm",
            "ilume_stage3_single_task_mlp",
        }
        if not advanced and self.data.feature_cache is None:
            raise ValueError("Feature baselines require data.feature_cache")
        if not advanced and self.environment is not None:
            raise ValueError("Only advanced baselines use a dedicated environment")
        if self.name == "dmpnn":
            self._validate_dmpnn()
        if self.name == "molformer":
            self._validate_molformer()
        if self.name == "ilbert":
            self._validate_ilbert()
        if self.name == "spmm":
            self._validate_spmm()
        elif self.name not in {"molformer", "ilbert", "spmm"} and self.runtime:
            raise ValueError("Only token baselines currently declare benchmark runtime settings")
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

    def _validate_ilume_stage3_single_task_mlp(self) -> None:
        if self.features is not None or self.environment is not None:
            raise ValueError(
                "ILUME Stage3 Single-task MLP uses prepared embeddings in the main environment"
            )
        if self.data.feature_cache is not None:
            raise ValueError(
                "ILUME Stage3 Single-task MLP does not use a feature cache"
            )
        if self.data.stage3_prepared_artifacts is None:
            raise ValueError(
                "ILUME Stage3 Single-task MLP requires prepared Stage 3 artifacts"
            )
        if not self.stage3.enabled or self.stage3.tasks != "all":
            raise ValueError(
                "ILUME Stage3 Single-task MLP requires all Stage 3 tasks"
            )
        if self.stage3.folds != (1, 2, 3, 4, 5):
            raise ValueError(
                "ILUME Stage3 Single-task MLP requires folds [1, 2, 3, 4, 5]"
            )
        if self.stage2_physics.enabled or self.stage2_physics.tasks:
            raise ValueError(
                "ILUME Stage3 Single-task MLP is a Stage 3-only ablation"
            )
        expected_model = {
            "hidden_dims": [512, 256],
            "activation": "silu",
            "dropout": 0.1,
        }
        if self.model != expected_model:
            raise ValueError(
                "ILUME Stage3 Single-task MLP model must match the registered recipe"
            )
        expected_training = {
            "optimizer": "adamw",
            "learning_rate": 3.0e-4,
            "weight_decay": 1.0e-2,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "scheduler": "linear_warmup_cosine",
            "warmup_ratio": 0.05,
            "min_lr_ratio": 0.05,
            "batch_size": 128,
            "max_epochs": 100,
            "loss": "normalized_smooth_l1",
            "smooth_l1_beta": 1.0,
            "max_grad_norm": 1.0,
            "selection_metric": "validation_normalized_mae",
            "device": "cuda",
            "precision": "bf16",
        }
        if self.training != expected_training:
            raise ValueError(
                "ILUME Stage3 Single-task MLP training must match the registered recipe"
            )

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

    def _validate_ilbert(self) -> None:
        if self.features is not None:
            raise ValueError("ILBERT tokenizes SMILES and does not accept features")
        if self.environment is None or not all(
            (self.environment.name, str(self.environment.definition), str(self.environment.lock))
        ):
            raise ValueError("ILBERT requires a dedicated environment definition and lock")
        if self.environment.name != "ilume-ilbert":
            raise ValueError("ILBERT environment name must be ilume-ilbert")
        if self.data.stage2_authority_config is not None:
            raise ValueError("ILBERT does not use a Stage 2 prepare authority")
        expected_model = {
            "repository": "Yu-Xin-Qiu/ILBERT",
            "revision": "f9dc6f1b23a40b6988480735f3724a6332f68c12",
            "checkout": "artifacts/benchmarks/ilbert/upstream",
            "model_source_sha256": "ed4c1441bc479eb3e999f77a52b4ae70eec4069da9f15bb87603afa02d8ccbc8",
            "tokenizer_source_sha256": "f435f4fed84d740061f3d85982441a47ea0aae596d2618cb902947311911d84d",
            "vocab_sha256": "87dd3943b31b42d179912220497516f6422483e021c45a994302e634311cd44f",
            "pretrained_checkpoint": "artifacts/benchmarks/ilbert/pretrained_model.pth",
            "pretrained_sha256": "556d0f63d206de786a6589e8ad4e3f05fc31ed744cb3234fd13e4eff7c397f7c",
            "pretrained": True,
            "full_fine_tuning": True,
            "tokenizer": "official_ais_atomwise",
            "vocab_size": 2000,
            "hidden_dim": 512,
            "layers": 6,
            "heads": 4,
            "ffn_hidden_dim": 1024,
            "dropout": 0.0,
            "textcnn_kernel_sizes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
            "textcnn_filters": [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160],
            "max_length": 100,
            "truncation": True,
            "padding": "max_length",
            "shared_backbone": True,
            "component_forward": "merged_sequence_view_forward",
            "fusion": "ordered_concat_native_predictor",
        }
        if self.model != expected_model:
            raise ValueError("ILBERT model must match the registered upstream recipe")
        expected_training = {
            "optimizer": "adam",
            "learning_rate": 1.0e-4,
            "weight_decay": 0.0,
            "scheduler": "reduce_on_plateau",
            "scheduler_metric": "validation_raw_rmse",
            "scheduler_patience": 7,
            "scheduler_factor": 0.3,
            "minimum_learning_rate": 3.0e-5,
            "batch_size": 16,
            "gradient_accumulation_steps": 1,
            "max_epochs": 100,
            "early_stopping_patience": 15,
            "tf32": True,
            "loss": "mse",
            "selection_metric": "validation_raw_mae",
            "condition_transform": "raw_physical_units",
            "device": "cuda",
            "precision": "fp32",
        }
        if self.training != expected_training:
            raise ValueError("ILBERT training must match the registered fine-tuning recipe")
        expected_runtime = {
            "num_workers": 4,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "pin_memory": True,
            "non_blocking_transfer": True,
        }
        if self.runtime != expected_runtime:
            raise ValueError("ILBERT runtime must match the registered loader recipe")

    def _validate_spmm(self) -> None:
        if self.features is not None:
            raise ValueError("SPMM tokenizes SMILES and does not accept features")
        if self.environment is None or not all(
            (self.environment.name, str(self.environment.definition), str(self.environment.lock))
        ):
            raise ValueError("SPMM requires a dedicated environment definition and lock")
        if self.environment.name != "ilume-spmm":
            raise ValueError("SPMM environment name must be ilume-spmm")
        if self.data.stage2_authority_config is not None:
            raise ValueError("SPMM does not use a Stage 2 prepare authority")
        expected_model = {
            "repository": "jinhojsk515/SPMM",
            "revision": "046976484f31b3cbc862b8f2094e38df72fcfce7",
            "checkout": "artifacts/benchmarks/spmm/upstream",
            "spmm_source_sha256": "6adddd7db6287151fd594fdf01325d614dcedebd2309839bad6193d1c53e1ff2",
            "xbert_source_sha256": "13d205f5f50699d9e58b165eda932150c371c9bcaf3d35740b7f8df16335c04a",
            "regression_source_sha256": "c18c74660ca8c201ad1b06a39379c4522c7eef4f380bd06aecb9e5ef2b25b384",
            "vocab_sha256": "760a96b6855fcdc10c384d520c8be6e66140e2db98dde3cb930467c51b0f102a",
            "bert_config_sha256": "85a090cea9435faac75c08eae698754198123d528a78cdad1819066e5c9a7376",
            "pretrained_checkpoint": "artifacts/benchmarks/spmm/checkpoint_SPMM.ckpt",
            "pretrained_sha256": "6b8eafd693eba42680e20ea06e3bf4efde640ac54ee1068ab34e6934bd0aca01",
            "pretrained_size": 2358591924,
            "pretrained": True,
            "full_fine_tuning": True,
            "modality": "smiles_text_only",
            "remove_stereochemistry": True,
            "tokenizer": "official_bert_wordpiece",
            "wordpiece_max_input_chars_per_word": 350,
            "vocab_size": 300,
            "hidden_dim": 768,
            "text_layers": 6,
            "heads": 12,
            "ffn_hidden_dim": 3072,
            "dropout": 0.1,
            "max_length": 100,
            "encoder_max_length": 99,
            "truncation": True,
            "padding": "longest",
            "shared_backbone": True,
            "component_forward": "merged_component_backbone_forward",
            "fusion": "ordered_concat_native_regression_head",
            "input_cache": "unique_smiles_memory_token_cache",
        }
        if self.model != expected_model:
            raise ValueError("SPMM model must match the registered upstream recipe")
        expected_training = {
            "optimizer": "adamw",
            "learning_rate": 5.0e-5,
            "weight_decay": 2.0e-2,
            "scheduler": "linear_warmup_cosine",
            "warmup_epochs": 1,
            "warmup_learning_rate": 5.0e-6,
            "minimum_learning_rate": 3.0e-6,
            "batch_size": 128,
            "gradient_accumulation_steps": 1,
            "max_epochs": 50,
            "early_stopping_patience": 10,
            "length_bucketing": "sortish_length_bucketing_v1",
            "bucket_window_batches": 20,
            "loss": "mse",
            "selection_metric": "validation_raw_mae",
            "condition_transform": "train_only_zscore",
            "device": "cuda",
            "precision": "fp32",
            "cuda_matmul_tf32": True,
            "cudnn_tf32": True,
            "cudnn_benchmark": True,
        }
        if self.training != expected_training:
            raise ValueError("SPMM training must match the registered fine-tuning recipe")
        expected_runtime = {
            "num_workers": 4,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "pin_memory": True,
            "non_blocking_transfer": True,
        }
        if self.runtime != expected_runtime:
            raise ValueError("SPMM runtime must match the registered loader recipe")

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
        if payload["data"].get("stage3_prepared_artifacts") is None:
            payload["data"].pop("stage3_prepared_artifacts")
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
        {
            "data_root", "task_catalog", "stage3_authority_config",
            "feature_cache", "stage2_authority_config",
            "stage3_prepared_artifacts",
        },
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
            stage3_prepared_artifacts=(
                None
                if data.get("stage3_prepared_artifacts") is None
                else Path(data["stage3_prepared_artifacts"])
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
