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
    unique_systems: int | None = None
    size_class: str | None = None
    phase1_private_lr: float | None = None
    phase1_private_epochs: int | None = None
    phase2_private_epochs: int | None = None
    phase3_private_epochs: int | None = None
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
class Stage3RepresentationConfig:
    kind: str
    descriptor_family: str
    adapter: str
    output_dim: int


@dataclass(frozen=True)
class Stage3InitializationConfig:
    stage2_encoder: Path | None = Path(
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
    experts: int | None = None
    expert_hidden_ratio: float | None = None
    phase1: Stage3OwnerBudgetConfig | None = None
    phase2: Stage3OwnerBudgetConfig | None = None


def _base_groups() -> dict[str, Stage3GroupConfig]:
    return {name: Stage3GroupConfig() for name in BASE_GROUP_TASKS}


@dataclass(frozen=True)
class Stage3OwnerBudgetConfig:
    lr: float
    epochs: int


@dataclass(frozen=True)
class Stage3GlobalBudgetConfig:
    lr: float
    epochs: int
    warmup_ratio: float
    min_lr_ratio: float


@dataclass(frozen=True)
class Stage3PrivateClassConfig:
    width_ratio: float
    phase1: Stage3OwnerBudgetConfig
    phase2_epochs: int
    phase3_epochs: int


@dataclass(frozen=True)
class ResolvedStage3PrivateRecipe:
    phase1_lr: float
    phase1_epochs: int
    phase2_lr: float
    phase2_epochs: int
    phase3_lr: float
    phase3_epochs: int
    private_hidden_ratio: float
    tower_hidden_ratio: float
    film_hidden_ratio: float


@dataclass(frozen=True)
class Stage3ThreePhaseConfig:
    global_scope: Stage3GlobalBudgetConfig
    private_classes: dict[str, Stage3PrivateClassConfig]
    phase1_min_lr_ratio: float
    phase2_min_lr_ratio: float
    phase3_min_lr_ratio: float


@dataclass(frozen=True)
class Stage3TrainingConfig:
    seed: int | None = None
    composite_batch_size: int = 2048
    microbatch_size: int = 1024
    sampling_mode: str = "virtual"
    virtual_min_size: int = 1000
    epochs: int = 100
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.05
    max_grad_norm: float = 1.0
    joint_gradient_clip_mode: str = "global"
    smooth_l1_beta: float = 1.0
    amp_dtype: str = "bf16"
    optimizer_implementation: str = "single_tensor"
    active_tasks: str | tuple[str, ...] = "auto"
    checkpoint_interval_epochs: int = 10
    device: str = "cuda"
    cpu_threads: int = 4
    cpu_interop_threads: int = 1
    debug_pcgrad_traces: bool = False
    refinement_ratio: float = 0.20
    refinement_lr_multiplier: float = 0.10
    schedule_mode: str = "legacy_joint_refinement"
    three_phase: Stage3ThreePhaseConfig | None = None


@dataclass(frozen=True)
class Stage3Config:
    data: Stage3DataConfig = field(default_factory=Stage3DataConfig)
    preparation: Stage3PreparationConfig = field(default_factory=Stage3PreparationConfig)
    initialization: Stage3InitializationConfig = field(
        default_factory=Stage3InitializationConfig
    )
    representation: Stage3RepresentationConfig | None = None
    model: Stage3ModelConfig = field(default_factory=Stage3ModelConfig)
    groups: dict[str, Stage3GroupConfig] = field(default_factory=_base_groups)
    tasks: dict[str, Stage3TaskConfig] = field(default_factory=_base_task_registry)
    training: Stage3TrainingConfig = field(default_factory=Stage3TrainingConfig)

    def resolved_private_recipe(
        self, task_id: str
    ) -> ResolvedStage3PrivateRecipe:
        phases = self.training.three_phase
        if self.training.schedule_mode != "three_phase" or phases is None:
            raise ValueError("Resolved PRIVATE recipe requires three-phase training")
        task = self.tasks[task_id]
        class_recipe = phases.private_classes[str(task.size_class)]
        phase1_lr = (
            class_recipe.phase1.lr
            if task.phase1_private_lr is None
            else task.phase1_private_lr
        )
        phase1_epochs = (
            class_recipe.phase1.epochs
            if task.phase1_private_epochs is None
            else task.phase1_private_epochs
        )
        phase2_epochs = (
            class_recipe.phase2_epochs
            if task.phase2_private_epochs is None
            else task.phase2_private_epochs
        )
        phase3_epochs = (
            class_recipe.phase3_epochs
            if task.phase3_private_epochs is None
            else task.phase3_private_epochs
        )
        width = class_recipe.width_ratio
        return ResolvedStage3PrivateRecipe(
            phase1_lr=phase1_lr,
            phase1_epochs=phase1_epochs,
            phase2_lr=phase1_lr * phases.phase1_min_lr_ratio,
            phase2_epochs=phase2_epochs,
            phase3_lr=(
                phase1_lr
                * phases.phase1_min_lr_ratio
                * phases.phase2_min_lr_ratio
            ),
            phase3_epochs=phase3_epochs,
            private_hidden_ratio=float(
                task.model_overrides.get("private_hidden_ratio", width)
            ),
            tower_hidden_ratio=float(
                task.model_overrides.get("tower_hidden_ratio", width)
            ),
            film_hidden_ratio=float(
                task.model_overrides.get("film_hidden_ratio", width)
            ),
        )

    def validate(self) -> None:
        if self.data.split_policy not in {
            "prefer_il", "random", "system", "individual"
        }:
            raise ValueError(
                "data.split_policy must be one of: individual, prefer_il, "
                "random, system"
            )
        if self.data.cv_repeat <= 0 or any(
            value <= 0 for value in self.data.cv_repeats.values()
        ):
            raise ValueError("Stage 3 cv repeats must be positive")
        if self.preparation.encoding_batch_size <= 0:
            raise ValueError("preparation.encoding_batch_size must be positive")
        if self.representation is None:
            if self.initialization.stage2_encoder is None:
                raise ValueError("Stage 2 Object representation requires stage2_encoder")
        else:
            expected = {
                "kind": "rdkit_2d_adapter",
                "descriptor_family": "rdkit_2d",
                "adapter": "linear_layernorm",
                "output_dim": 512,
            }
            if asdict(self.representation) != expected:
                raise ValueError(
                    "RDKit Stage 3 representation must match the registered recipe"
                )
            if self.initialization.stage2_encoder is not None:
                raise ValueError("RDKit representation forbids stage2_encoder")
            if self.initialization.plugin is not None:
                raise ValueError("RDKit representation forbids plugin initialization")
        if not self.groups or not self.tasks:
            raise ValueError("Stage 3 requires groups and tasks")
        if any(group.group_weight <= 0 for group in self.groups.values()):
            raise ValueError("Stage 3 group weights must be positive")
        for group_id, group in self.groups.items():
            if group.experts is not None and group.experts <= 0:
                raise ValueError(f"Stage 3 group experts must be positive: {group_id}")
            if group.expert_hidden_ratio is not None and group.expert_hidden_ratio <= 0:
                raise ValueError(
                    f"Stage 3 group expert_hidden_ratio must be positive: {group_id}"
                )
            for phase_name, budget in (("phase1", group.phase1), ("phase2", group.phase2)):
                if budget is not None and (budget.lr <= 0 or budget.epochs <= 0):
                    raise ValueError(
                        f"Stage 3 {phase_name} group budget is invalid: {group_id}"
                    )
        override_keys = {
            "private_experts", "private_hidden_ratio", "tower_hidden_ratio",
            "film_hidden_ratio",
        }
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
            if task.unique_systems is not None and task.unique_systems <= 0:
                raise ValueError(
                    f"Stage 3 unique_systems must be positive: {task_id}"
                )
            if task.size_class is not None and task.size_class not in {
                "tiny", "small", "medium", "large"
            }:
                raise ValueError(f"Invalid Stage 3 size_class: {task_id}")
            if task.phase1_private_lr is not None and (
                isinstance(task.phase1_private_lr, bool)
                or not isinstance(task.phase1_private_lr, (int, float))
                or task.phase1_private_lr <= 0
            ):
                raise ValueError(
                    f"Stage 3 Phase 1 PRIVATE LR must be positive: {task_id}"
                )
            for name in ("phase1_private_epochs", "phase2_private_epochs"):
                value = getattr(task, name)
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                ):
                    raise ValueError(
                        f"Stage 3 {name} must be a positive integer: {task_id}"
                    )
            if task.phase3_private_epochs is not None and (
                isinstance(task.phase3_private_epochs, bool)
                or not isinstance(task.phase3_private_epochs, int)
                or task.phase3_private_epochs < 0
            ):
                raise ValueError(
                    f"Stage 3 phase3_private_epochs must be a non-negative integer: {task_id}"
                )
            unknown_overrides = set(task.model_overrides) - override_keys
            if unknown_overrides:
                raise ValueError(
                    f"Unknown Stage 3 task model overrides for {task_id}: "
                    + ", ".join(sorted(unknown_overrides))
                )
            if "private_experts" in task.model_overrides:
                value = task.model_overrides["private_experts"]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"Stage 3 private_experts must be a non-negative integer: {task_id}"
                    )
            for name in (
                "private_hidden_ratio", "tower_hidden_ratio",
                "film_hidden_ratio",
            ):
                if name in task.model_overrides:
                    value = task.model_overrides[name]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                        raise ValueError(
                            f"Stage 3 {name} must be positive: {task_id}"
                        )
        enabled_tasks = [task for task in self.tasks.values() if task.enabled]
        if not enabled_tasks:
            raise ValueError("Stage 3 requires at least one enabled task")
        if any(not self.groups[task.meta_group].enabled for task in enabled_tasks):
            raise ValueError("Enabled task cannot belong to a disabled group")
        model = self.model
        if model.global_experts < 0 or model.private_experts < 0:
            raise ValueError("global/private expert counts must be non-negative")
        if model.group_experts <= 0:
            raise ValueError("model.group_experts must be positive")
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
        if training.seed is not None and training.seed < 0:
            raise ValueError("training.seed must be non-negative or null")
        for name in (
            "composite_batch_size", "microbatch_size", "virtual_min_size",
            "epochs", "checkpoint_interval_epochs", "cpu_threads",
            "cpu_interop_threads",
        ):
            if getattr(training, name) <= 0:
                raise ValueError(f"training.{name} must be positive")
        if training.microbatch_size > training.composite_batch_size:
            raise ValueError("microbatch_size exceeds composite_batch_size")
        if training.schedule_mode not in {"legacy_joint_refinement", "three_phase"}:
            raise ValueError(
                "training.schedule_mode must be legacy_joint_refinement or three_phase"
            )
        if training.schedule_mode == "legacy_joint_refinement":
            if training.three_phase is not None:
                raise ValueError("legacy Stage 3 training forbids training.three_phase")
            from common.refinement import refinement_geometry
            refinement_geometry(training.epochs, training.refinement_ratio)
            if training.refinement_lr_multiplier <= 0:
                raise ValueError("refinement_lr_multiplier must be positive")
        else:
            if training.three_phase is None:
                raise ValueError("three-phase Stage 3 training requires training.three_phase")
            if training.sampling_mode != "raw":
                raise ValueError("three-phase Stage 3 training requires raw sampling")
            if training.joint_gradient_clip_mode != "ownership":
                raise ValueError(
                    "three-phase Stage 3 training requires ownership clipping"
                )
            for group_id, group in self.groups.items():
                if group.enabled and (
                    group.experts is None
                    or group.expert_hidden_ratio is None
                    or group.phase1 is None
                    or group.phase2 is None
                ):
                    raise ValueError(
                        f"Three-phase Stage 3 group recipe is incomplete: {group_id}"
                    )
            for task_id, task in self.tasks.items():
                if task.enabled and (
                    task.unique_systems is None
                    or task.size_class is None
                ):
                    raise ValueError(
                        f"Three-phase Stage 3 task recipe is incomplete: {task_id}"
                    )
            phases = training.three_phase
            expected_classes = {"tiny", "small", "medium", "large"}
            if set(phases.private_classes) != expected_classes:
                raise ValueError(
                    "Three-phase Stage 3 private classes must be tiny/small/medium/large"
                )
            global_scope = phases.global_scope
            if global_scope.lr <= 0 or global_scope.epochs <= 0:
                raise ValueError("Three-phase Stage 3 GLOBAL budget is invalid")
            if not 0 <= global_scope.warmup_ratio < 1:
                raise ValueError("Three-phase GLOBAL warmup_ratio must be in [0, 1)")
            if not 0 < global_scope.min_lr_ratio <= 1:
                raise ValueError("Three-phase GLOBAL min_lr_ratio must be in (0, 1]")
            for class_name, recipe in phases.private_classes.items():
                if (
                    recipe.width_ratio <= 0
                    or recipe.phase1.lr <= 0
                    or recipe.phase1.epochs <= 0
                    or recipe.phase2_epochs <= 0
                    or recipe.phase3_epochs < 0
                ):
                    raise ValueError(
                        f"Three-phase Stage 3 private class is invalid: {class_name}"
                    )
            for name, value in (
                ("phase1", phases.phase1_min_lr_ratio),
                ("phase2", phases.phase2_min_lr_ratio),
                ("phase3", phases.phase3_min_lr_ratio),
            ):
                if not 0 < value <= 1:
                    raise ValueError(
                        f"Three-phase {name} min_lr_ratio must be in (0, 1]"
                    )
            if model.private_experts != 1 or any(
                task.model_overrides.get("private_experts", 1) != 1
                for task in enabled_tasks
            ):
                raise ValueError("Three-phase Stage 3 requires one PRIVATE expert")
            for group_id, group in self.groups.items():
                if not group.enabled:
                    continue
                assert group.phase1 is not None and group.phase2 is not None
                if group.phase2.lr != group.phase1.lr * phases.phase1_min_lr_ratio:
                    raise ValueError(
                        f"Three-phase GROUP Phase 2 LR must equal Phase 1 terminal LR: {group_id}"
                    )
            for task_id, task in self.tasks.items():
                if not task.enabled:
                    continue
                private_recipe = self.resolved_private_recipe(task_id)
                group = self.groups[task.meta_group]
                assert group.phase1 is not None
                if not (
                    phases.global_scope.lr
                    > group.phase1.lr
                    > private_recipe.phase1_lr
                ):
                    raise ValueError(
                        f"Three-phase Phase 1 nominal LR ordering is invalid: {task_id}"
                    )
                if private_recipe.phase1_epochs > phases.global_scope.epochs:
                    raise ValueError(
                        f"Three-phase Phase 1 PRIVATE epochs exceed GLOBAL budget: {task_id}"
                    )
        if training.learning_rate <= 0 or training.weight_decay < 0:
            raise ValueError("Stage 3 optimizer values are invalid")
        if len(training.betas) != 2 or not all(0 <= x < 1 for x in training.betas):
            raise ValueError("training.betas must contain two values in [0, 1)")
        if training.eps <= 0 or training.max_grad_norm < 0:
            raise ValueError("Stage 3 eps/grad norm values are invalid")
        if training.sampling_mode not in {"virtual", "raw"}:
            raise ValueError("training.sampling_mode must be virtual or raw")
        if training.joint_gradient_clip_mode not in {"global", "ownership"}:
            raise ValueError(
                "training.joint_gradient_clip_mode must be global or ownership"
            )
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
        if self.representation is None:
            payload.pop("representation")
        plugin = payload["initialization"].get("plugin")
        if plugin is not None:
            adaptation = plugin["adaptation"]
            adaptation["global"] = adaptation.pop("global_scope")
        training = payload["training"]
        if training["schedule_mode"] == "legacy_joint_refinement":
            training.pop("schedule_mode")
            training.pop("three_phase")
        else:
            payload["model"].pop("group_experts")
            training["three_phase"]["global"] = training["three_phase"].pop(
                "global_scope"
            )
            for name in (
                "virtual_min_size", "epochs", "learning_rate", "warmup_ratio",
                "min_lr_ratio", "refinement_ratio", "refinement_lr_multiplier",
            ):
                training.pop(name)
        for group in payload["groups"].values():
            for name in ("experts", "expert_hidden_ratio", "phase1", "phase2"):
                if group[name] is None:
                    group.pop(name)
        for task in payload["tasks"].values():
            for name in (
                "unique_systems", "size_class", "phase1_private_lr",
                "phase1_private_epochs", "phase2_private_epochs",
                "phase3_private_epochs",
            ):
                if task[name] is None:
                    task.pop(name)
        if training["sampling_mode"] == "virtual":
            training.pop("sampling_mode")
        else:
            training.pop("virtual_min_size", None)
        if training["joint_gradient_clip_mode"] == "global":
            training.pop("joint_gradient_clip_mode")
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
        "data", "preparation", "initialization", "representation", "model",
        "groups", "tasks", "training"
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
    if initialization_raw.get("stage2_encoder") is not None:
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
    groups = _base_groups()
    if groups_raw is not None:
        groups = {}
        for name, raw_group in groups_raw.items():
            values = dict(raw_group)
            for phase_name in ("phase1", "phase2"):
                if values.get(phase_name) is not None:
                    values[phase_name] = _construct_dataclass(
                        Stage3OwnerBudgetConfig, values[phase_name]
                    )
            groups[name] = _construct_dataclass(Stage3GroupConfig, values)
    tasks_raw = raw.get("tasks")
    tasks = _base_task_registry()
    if tasks_raw is not None:
        tasks = {}
        for task_id, raw_task in tasks_raw.items():
            value = dict(raw_task)
            if "phase3_epochs" in value:
                raise ValueError(
                    "Stage 3 task phase3_epochs is retired; use "
                    f"phase3_private_epochs for {task_id}"
                )
            if "model_overrides" in value:
                if not isinstance(value["model_overrides"], dict):
                    raise ValueError(
                        f"Stage 3 task model_overrides must be a mapping: {task_id}"
                    )
                value["model_overrides"] = dict(value["model_overrides"])
            tasks[task_id] = _construct_dataclass(
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
    training_raw = dict(raw.get("training") or {})
    schedule_mode = training_raw.get("schedule_mode", "legacy_joint_refinement")
    if schedule_mode == "four_phase" or "four_phase" in training_raw:
        raise ValueError(
            "Four-phase Stage 3 training is retired and cannot be loaded"
        )
    if schedule_mode == "three_phase":
        forbidden = {
            "epochs", "learning_rate", "warmup_ratio", "min_lr_ratio",
            "refinement_ratio", "refinement_lr_multiplier",
        } & set(training_raw)
        if forbidden:
            raise ValueError(
                "Three-phase Stage 3 training forbids legacy fields: "
                + ", ".join(sorted(forbidden))
            )
    three_phase_raw = training_raw.get("three_phase")
    if three_phase_raw is not None:
        if not isinstance(three_phase_raw, dict):
            raise ValueError("training.three_phase must be a mapping")
        values = dict(three_phase_raw)
        if "global" not in values or "private_classes" not in values:
            raise ValueError("training.three_phase is incomplete")
        values["global_scope"] = _construct_dataclass(
            Stage3GlobalBudgetConfig, values.pop("global")
        )
        private_classes = values["private_classes"]
        if not isinstance(private_classes, dict):
            raise ValueError("training.three_phase.private_classes must be a mapping")
        resolved_classes = {}
        for class_name, raw_class in private_classes.items():
            class_values = dict(raw_class)
            class_values["phase1"] = _construct_dataclass(
                Stage3OwnerBudgetConfig, class_values.get("phase1")
            )
            resolved_classes[class_name] = _construct_dataclass(
                Stage3PrivateClassConfig, class_values
            )
        values["private_classes"] = resolved_classes
        training_raw["three_phase"] = _construct_dataclass(
            Stage3ThreePhaseConfig, values
        )
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
        representation=(
            None
            if raw.get("representation") is None
            else _construct_dataclass(
                Stage3RepresentationConfig, raw.get("representation")
            )
        ),
        model=_construct_dataclass(Stage3ModelConfig, raw.get("model")),
        groups=groups,
        tasks=tasks,
        training=_construct_dataclass(Stage3TrainingConfig, training_raw),
    )
    config.validate()
    return config


def effective_training_seed(config: Stage3Config) -> int:
    """Return the training RNG seed while preserving the legacy data-seed default."""
    return config.data.seed if config.training.seed is None else config.training.seed


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
