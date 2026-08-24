from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml


BASE_GROUP_TASKS: dict[str, tuple[str, ...]] = {
    "transport": (
        "experiment/electrical_conductivity",
        "experiment/self_diffusion_coefficient",
        "experiment/viscosity",
        "experiment/thermal_conductivity",
    ),
    "thermophysical": (
        "experiment/density",
        "experiment/heat_capacity",
        "experiment/isobaric_coefficient_of_volume_expansion",
        "experiment/speed_of_sound",
        "experiment/surface_tension",
    ),
    "phase_stability": (
        "experiment/equilibrium_pressure",
        "experiment/glass_transition_temperature",
        "experiment/melting_point",
        "experiment/thermal_decomposition_temperature",
    ),
    "dielectric_optical": (
        "experiment/dynamic_relative_permittivity",
        "experiment/static_relative_permittivity",
        "experiment/refractive_index",
    ),
    "solvation": (
        "experiment/solvation",
        "experiment/transfer",
        "experiment/transfer_organic",
        "experiment/x_co2",
    ),
    "biological": ("experiment/pec50",),
}


def validate_stage3_folds(folds: Sequence[int]) -> tuple[int, ...]:
    values = tuple(folds)
    if not values:
        raise ValueError("--fold requires at least one value")
    if any(fold not in range(1, 6) for fold in values):
        raise ValueError("--fold values must be in 1..5")
    if len(values) != len(set(values)):
        raise ValueError("--fold must not contain duplicate folds")
    return values


@dataclass(frozen=True)
class Stage3TaskConfig:
    meta_group: str
    partner_mode: str = "none"
    primary_slots: tuple[str, ...] = ("cation", "anion")
    partner_slots: tuple[str, ...] = ()
    enabled: bool = True
    task_weight: float = 1.0
    model_overrides: dict[str, Any] = field(default_factory=dict)


def _base_task_registry() -> dict[str, Stage3TaskConfig]:
    result: dict[str, Stage3TaskConfig] = {}
    for group, tasks in BASE_GROUP_TASKS.items():
        for task_id in tasks:
            if task_id in {"experiment/solvation", "experiment/transfer"}:
                primary = ("cation", "anion")
                partner = ("solute",)
                partner_mode = "interaction"
            elif task_id == "experiment/transfer_organic":
                primary = ("solute",)
                partner = ("solvent",)
                partner_mode = "interaction"
            else:
                primary = ("cation", "anion")
                partner = ()
                partner_mode = "none"
            result[task_id] = Stage3TaskConfig(
                meta_group=group,
                partner_mode=partner_mode,
                primary_slots=primary,
                partner_slots=partner,
            )
    return result


@dataclass(frozen=True)
class Stage3DataConfig:
    stage3_dir: Path = Path("data/stage3")
    task_catalog: Path = Path("data/task_catalog.csv")
    artifacts_dir: Path = Path("outputs/v1/stage3/base/prepare/artifacts")
    split_policy: str = "prefer_il"
    split_strategies: dict[str, str] = field(default_factory=dict)
    cv_repeat: int = 1
    cv_repeats: dict[str, int] = field(default_factory=dict)
    seed: int = 42


@dataclass(frozen=True)
class Stage3PreparationConfig:
    encoding_batch_size: int = 256
    cache_dir: Path = Path("outputs/v1/stage3/base/prepare/object_cache")


@dataclass(frozen=True)
class Stage3PluginAdaptationConfig:
    global_scope: bool = False
    groups: tuple[str, ...] = ()
    private_tasks: tuple[str, ...] = ()


@dataclass(frozen=True)
class Stage3PluginConfig:
    checkpoint: Path
    load_scopes: tuple[str, ...] = ("GLOBAL", "GROUP:*", "PRIVATE:*")
    adaptation: Stage3PluginAdaptationConfig = field(
        default_factory=Stage3PluginAdaptationConfig
    )


@dataclass(frozen=True)
class Stage3InitializationConfig:
    stage2_encoder: Path = Path(
        "outputs/v1/stage2/base/train/stage2_encoder.pt"
    )
    plugin: Stage3PluginConfig | None = None


@dataclass(frozen=True)
class Stage3ModelConfig:
    global_experts: int = 2
    group_experts: int = 2
    private_experts: int = 1
    dropout: float = 0.10
    activation: str = "silu"
    expert_hidden_ratio: float = 2.0
    interaction_hidden_ratio: float = 2.0
    film_hidden_ratio: float = 1.0
    tower_hidden_ratio: float = 1.0
    l2_residual: bool = True


@dataclass(frozen=True)
class Stage3GroupConfig:
    enabled: bool = True
    group_weight: float = 1.0


def _base_groups() -> dict[str, Stage3GroupConfig]:
    return {name: Stage3GroupConfig() for name in BASE_GROUP_TASKS}


@dataclass(frozen=True)
class Stage3TrainingConfig:
    composite_batch_size: int = 2048
    microbatch_size: int = 1024
    virtual_min_size: int = 1000
    epochs: int = 100
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.05
    max_grad_norm: float = 1.0
    smooth_l1_beta: float = 1.0
    amp_dtype: str = "bf16"
    optimizer_implementation: str = "single_tensor"
    active_tasks: str | tuple[str, ...] = "auto"
    checkpoint_interval_epochs: int = 10
    device: str = "cuda"
    cpu_threads: int = 4
    cpu_interop_threads: int = 1
    debug_pcgrad_traces: bool = False


@dataclass(frozen=True)
class Stage3Config:
    data: Stage3DataConfig = field(default_factory=Stage3DataConfig)
    preparation: Stage3PreparationConfig = field(default_factory=Stage3PreparationConfig)
    initialization: Stage3InitializationConfig = field(
        default_factory=Stage3InitializationConfig
    )
    model: Stage3ModelConfig = field(default_factory=Stage3ModelConfig)
    groups: dict[str, Stage3GroupConfig] = field(default_factory=_base_groups)
    tasks: dict[str, Stage3TaskConfig] = field(default_factory=_base_task_registry)
    training: Stage3TrainingConfig = field(default_factory=Stage3TrainingConfig)

    def validate(self) -> None:
        if self.data.split_policy != "prefer_il":
            raise ValueError("data.split_policy must be prefer_il")
        if self.data.cv_repeat <= 0 or any(
            value <= 0 for value in self.data.cv_repeats.values()
        ):
            raise ValueError("Stage 3 cv repeats must be positive")
        if self.preparation.encoding_batch_size <= 0:
            raise ValueError("preparation.encoding_batch_size must be positive")
        if not self.groups or not self.tasks:
            raise ValueError("Stage 3 requires groups and tasks")
        if any(group.group_weight <= 0 for group in self.groups.values()):
            raise ValueError("Stage 3 group weights must be positive")
        for task_id, task in self.tasks.items():
            if not task_id or "." in task_id:
                raise ValueError(f"Invalid Stage 3 task id: {task_id}")
            if task.meta_group not in self.groups:
                raise ValueError(f"Unknown meta-group for {task_id}: {task.meta_group}")
            if task.partner_mode not in {"none", "interaction"}:
                raise ValueError(f"Invalid partner_mode for {task_id}")
            if not task.primary_slots:
                raise ValueError(f"Stage 3 task requires primary slots: {task_id}")
            if task.partner_mode == "none" and task.partner_slots:
                raise ValueError(f"Non-partner task has partner slots: {task_id}")
            if task.partner_mode == "interaction" and not task.partner_slots:
                raise ValueError(f"Partner task has no partner slots: {task_id}")
            if task.task_weight <= 0:
                raise ValueError(f"Stage 3 task weight must be positive: {task_id}")
            if task.model_overrides:
                raise ValueError(
                    f"Stage 3 v1 has no task model overrides yet: {task_id}"
                )
        enabled_tasks = [task for task in self.tasks.values() if task.enabled]
        if not enabled_tasks:
            raise ValueError("Stage 3 requires at least one enabled task")
        if any(not self.groups[task.meta_group].enabled for task in enabled_tasks):
            raise ValueError("Enabled task cannot belong to a disabled group")
        model = self.model
        for name in ("global_experts", "group_experts", "private_experts"):
            if getattr(model, name) <= 0:
                raise ValueError(f"model.{name} must be positive")
        if not 0.0 <= model.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if model.activation not in {"silu", "gelu"}:
            raise ValueError("model.activation must be silu or gelu")
        for name in (
            "expert_hidden_ratio", "interaction_hidden_ratio",
            "film_hidden_ratio", "tower_hidden_ratio",
        ):
            if getattr(model, name) <= 0:
                raise ValueError(f"model.{name} must be positive")
        training = self.training
        for name in (
            "composite_batch_size", "microbatch_size", "virtual_min_size",
            "epochs", "checkpoint_interval_epochs", "cpu_threads",
            "cpu_interop_threads",
        ):
            if getattr(training, name) <= 0:
                raise ValueError(f"training.{name} must be positive")
        if training.microbatch_size > training.composite_batch_size:
            raise ValueError("microbatch_size exceeds composite_batch_size")
        if training.learning_rate <= 0 or training.weight_decay < 0:
            raise ValueError("Stage 3 optimizer values are invalid")
        if len(training.betas) != 2 or not all(0 <= x < 1 for x in training.betas):
            raise ValueError("training.betas must contain two values in [0, 1)")
        if training.eps <= 0 or training.max_grad_norm < 0:
            raise ValueError("Stage 3 eps/grad norm values are invalid")
        if not 0 <= training.warmup_ratio < 1:
            raise ValueError("training.warmup_ratio must be in [0, 1)")
        if not 0 < training.min_lr_ratio <= 1:
            raise ValueError("training.min_lr_ratio must be in (0, 1]")
        if training.smooth_l1_beta <= 0:
            raise ValueError("training.smooth_l1_beta must be positive")
        if training.amp_dtype not in {"bf16", "none"}:
            raise ValueError("training.amp_dtype must be bf16 or none")
        if training.optimizer_implementation != "single_tensor":
            raise ValueError(
                "training.optimizer_implementation must be single_tensor"
            )
        if isinstance(training.active_tasks, tuple):
            unknown = set(training.active_tasks) - set(self.tasks)
            if unknown:
                raise ValueError(
                    "Unknown active Stage 3 tasks: " + ", ".join(sorted(unknown))
                )
        elif training.active_tasks not in {"auto", "auto_new"}:
            raise ValueError("training.active_tasks must be auto, auto_new, or a list")
        plugin = self.initialization.plugin
        if plugin is not None:
            if not plugin.load_scopes:
                raise ValueError("Plugin load_scopes cannot be empty")
            if set(plugin.adaptation.groups) - set(self.groups) or set(
                plugin.adaptation.private_tasks
            ) - set(self.tasks):
                raise ValueError("Plugin adaptation references unknown scopes")

    @property
    def enabled_task_ids(self) -> tuple[str, ...]:
        return tuple(task_id for task_id, spec in self.tasks.items() if spec.enabled)

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            return value

        payload = convert(asdict(self))
        plugin = payload["initialization"].get("plugin")
        if plugin is not None:
            adaptation = plugin["adaptation"]
            adaptation["global"] = adaptation.pop("global_scope")
        return payload


def _construct_dataclass(cls: type, raw: dict[str, Any] | None) -> Any:
    values = dict(raw or {})
    unknown = set(values) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} fields: " + ", ".join(sorted(unknown))
        )
    return cls(**values)


def stage3_config_from_dict(raw: dict[str, Any]) -> Stage3Config:
    allowed = {
        "data", "preparation", "initialization", "model", "groups", "tasks", "training"
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError("Unknown Stage 3 config sections: " + ", ".join(sorted(unknown)))
    data_raw = dict(raw.get("data") or {})
    for name in ("stage3_dir", "task_catalog", "artifacts_dir"):
        if name in data_raw:
            data_raw[name] = Path(data_raw[name])
    preparation_raw = dict(raw.get("preparation") or {})
    if "cache_dir" in preparation_raw:
        preparation_raw["cache_dir"] = Path(preparation_raw["cache_dir"])
    initialization_raw = dict(raw.get("initialization") or {})
    if "stage2_encoder" in initialization_raw:
        initialization_raw["stage2_encoder"] = Path(
            initialization_raw["stage2_encoder"]
        )
    plugin_raw = initialization_raw.get("plugin")
    if plugin_raw is not None:
        plugin_values = dict(plugin_raw)
        plugin_values["checkpoint"] = Path(plugin_values["checkpoint"])
        if "load_scopes" in plugin_values:
            plugin_values["load_scopes"] = tuple(plugin_values["load_scopes"])
        adaptation_values = dict(plugin_values.get("adaptation") or {})
        if "global" in adaptation_values:
            adaptation_values["global_scope"] = adaptation_values.pop("global")
        for name in ("groups", "private_tasks"):
            if name in adaptation_values:
                adaptation_values[name] = tuple(adaptation_values[name])
        plugin_values["adaptation"] = _construct_dataclass(
            Stage3PluginAdaptationConfig, adaptation_values
        )
        initialization_raw["plugin"] = _construct_dataclass(
            Stage3PluginConfig, plugin_values
        )
    groups_raw = raw.get("groups")
    groups = _base_groups() if groups_raw is None else {
        name: _construct_dataclass(Stage3GroupConfig, value)
        for name, value in groups_raw.items()
    }
    tasks_raw = raw.get("tasks")
    tasks = _base_task_registry() if tasks_raw is None else {
        task_id: _construct_dataclass(
            Stage3TaskConfig,
            {
                **value,
                **{
                    name: tuple(value[name])
                    for name in ("primary_slots", "partner_slots")
                    if name in value
                },
            },
        )
        for task_id, value in tasks_raw.items()
    }
    training_raw = dict(raw.get("training") or {})
    if "betas" in training_raw:
        training_raw["betas"] = tuple(training_raw["betas"])
    if isinstance(training_raw.get("active_tasks"), list):
        training_raw["active_tasks"] = tuple(training_raw["active_tasks"])
    config = Stage3Config(
        data=_construct_dataclass(Stage3DataConfig, data_raw),
        preparation=_construct_dataclass(Stage3PreparationConfig, preparation_raw),
        initialization=_construct_dataclass(
            Stage3InitializationConfig, initialization_raw
        ),
        model=_construct_dataclass(Stage3ModelConfig, raw.get("model")),
        groups=groups,
        tasks=tasks,
        training=_construct_dataclass(Stage3TrainingConfig, training_raw),
    )
    config.validate()
    return config


def stage3_config_from_checkpoint_dict(raw: dict[str, Any]) -> Stage3Config:
    return stage3_config_from_dict(raw)


def load_stage3_config(path: str | Path) -> Stage3Config:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Stage 3 configuration root must be a mapping")
    return stage3_config_from_dict(raw)


def configure_process_runtime(config: Stage3Config) -> None:
    threads = config.training.cpu_threads
    interop_threads = config.training.cpu_interop_threads
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != interop_threads:
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError as error:
            raise RuntimeError(
                "Stage 3 inter-op threads must be configured before parallel work"
            ) from error
