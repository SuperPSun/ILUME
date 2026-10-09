from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch
from stage1.masking import MultimodalPacker
from stage2.data import (
    Stage2BatchDescriptor, Stage2DeviceTaskData, Stage2EntityDataset,
    Stage2TaskDataset, load_artifact_registry, pack_stage2_batch,
)
from stage2.home_model import SimulationHoME, predict_simulation_task
from stage2.home_contract import source_task_specs
from stage2.home_train import simulation_loss

from .model import Ownership, group_owner, private_owner
from .model import Stage3SparseModel


SIMULATION_TASKS = ("simulation/heat_of_vaporization", "simulation/thermal_expansion")
AUXILIARY_SIMULATION_TASKS = SIMULATION_TASKS


def simulation_tasks(config: Any) -> tuple[str, ...]:
    return AUXILIARY_SIMULATION_TASKS if config.is_entity_home else SIMULATION_TASKS


SIMULATION_BACKBONE_OWNER = Ownership("SIMULATION_BACKBONE")


def resolve_simulation_specs(source_registry: Any, experimental_specs: Mapping[str, Any], recipe: Any, tasks: tuple[str, ...] = SIMULATION_TASKS) -> dict[str, Any]:
    specs = source_task_specs(source_registry, role_policy="formal_charge_v1" if tasks == AUXILIARY_SIMULATION_TASKS else "legacy_slot_v1")
    shared_groups = {spec.meta_group for spec in experimental_specs.values() if spec.enabled}
    return {
        task: replace(
            specs[task],
            task_weight=(recipe.shared_group_task_weight
                         if tasks != AUXILIARY_SIMULATION_TASKS and specs[task].meta_group in shared_groups else 1.0),
        )
        for task in tasks
    }


class SimulationEntityModel(Stage3SparseModel):
    def __init__(self, *args: Any, source_model: SimulationHoME, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.simulation_backbone = source_model.backbone
        self.simulation_tasks = AUXILIARY_SIMULATION_TASKS if source_model.stage2_config.is_entity_home else SIMULATION_TASKS
        self.simulation_atom_adapter = None
        self.simulation_registry = source_model.registry
        self._own_modules(SIMULATION_BACKBONE_OWNER, self.simulation_backbone)
        if self.simulation_atom_adapter is not None:
            self._own_modules(
                private_owner("simulation/partial_atomic_charge"), self.simulation_atom_adapter,
            )
        self._validate_ownership()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.simulation_tasks == AUXILIARY_SIMULATION_TASKS:
            self.simulation_backbone.eval()
        return self

    def predict_simulation(self, task: str, packed: Any, task_data: Any) -> torch.Tensor:
        if task not in self.simulation_tasks:
            raise ValueError(f"Unsupported Stage 3 simulation task: {task}")
        return predict_simulation_task(
            self.simulation_backbone, self, self.simulation_registry,
            task, packed, task_data,
        )


def copy_simulation_owners(model: SimulationEntityModel, source: SimulationHoME) -> tuple[str, ...]:
    source_manifest = source.home.ownership_manifest()
    target_manifest = model.ownership_manifest()
    tasks = model.simulation_tasks
    electronic = "simulation/homo" in tasks
    selected = {
        name for name, owner in target_manifest.items()
        if ((electronic and owner == group_owner("electronic_structure").label)
            or owner in {private_owner(task).label for task in tasks})
        and not name.startswith("simulation_atom_adapter.")
    }
    expected = {
        name for name, owner in source_manifest.items()
        if (electronic and owner == group_owner("electronic_structure").label)
        or owner in {private_owner(task).label for task in tasks}
    }
    if selected != expected:
        raise ValueError("Stage 3 simulation owner tensor set differs from Stage 2")
    target = model.state_dict()
    original = source.home.state_dict()
    for name in selected:
        if target[name].shape != original[name].shape or target[name].dtype != original[name].dtype:
            raise ValueError(f"Stage 3 simulation owner tensor shape differs: {name}")
        target[name] = original[name]
    model.load_state_dict(target, strict=True)
    return tuple(sorted(selected))


@dataclass
class SimulationPlanData:
    train: dict[str, Stage2TaskDataset]
    data_identity: str


@dataclass
class SimulationTrainingData:
    entities: Stage2EntityDataset
    train: dict[str, Stage2TaskDataset]
    valid: dict[str, Stage2TaskDataset]
    packer: MultimodalPacker
    train_device: dict[str, Stage2DeviceTaskData]
    valid_device: dict[str, Stage2DeviceTaskData]
    data_identity: str
    amp_dtype: str = "bf16"

    @torch.no_grad()
    def validate_tasks(
        self, model: SimulationEntityModel, tasks: tuple[str, ...],
        device: torch.device,
    ) -> dict[str, Any]:
        if not tasks:
            return {}
        model.eval()
        result: dict[str, Any] = {}
        for task in tasks:
            dataset = self.valid[task]
            error_sum = 0.0
            count = 0
            for start in range(0, len(dataset), 256):
                indices = torch.arange(start, min(len(dataset), start + 256))
                packed = pack_stage2_batch(
                    Stage2BatchDescriptor(task, indices), {task: dataset},
                    self.entities, self.packer, needs_entities=True,
                    include_raw_atom_targets=False, pin_memory=False,
                ).to(device, non_blocking=False)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=self.amp_dtype == "bf16",
                ):
                    prediction = model.predict_simulation(task, packed, self.valid_device[task])
                if not torch.isfinite(prediction).all():
                    raise RuntimeError(f"Non-finite Stage 3 simulation validation prediction: {task}")
                if task == "simulation/partial_atomic_charge":
                    atoms = packed.atom_targets
                    assert atoms is not None
                    errors = (prediction.float() - atoms.values.float()).abs() * atoms.mask
                    sums = torch.zeros(len(indices), device=device).index_add_(
                        0, atoms.atom_sample_indices, errors,
                    )
                    counts = torch.zeros(len(indices), device=device).index_add_(
                        0, atoms.atom_sample_indices, atoms.mask.float(),
                    )
                    error_sum += float((sums / counts.clamp_min(1)).sum())
                    count += len(indices)
                else:
                    targets = self.valid_device[task].targets[packed.row_indices]
                    mask = self.valid_device[task].target_mask[packed.row_indices]
                    error_sum += float(((prediction.float() - targets.float()).abs() * mask).sum())
                    count += int(mask.sum())
            result[task] = {"count": count, "normalized_mae": error_sum / count}
        return {"simulation_tasks": result}

    def compute_gradient(
        self, model: SimulationEntityModel, task: str, indices: torch.Tensor,
        device: torch.device, microbatch_size: int = 256,
    ) -> tuple[dict[torch.nn.Parameter, torch.Tensor], float]:
        if task not in self.train:
            raise ValueError(f"Unknown Stage 3 simulation task: {task}")
        parameters = tuple(
            parameter for parameter, owner in model.parameter_ownership().items()
            if parameter.requires_grad and (
                model.simulation_tasks != AUXILIARY_SIMULATION_TASKS or owner == private_owner(task)
            )
        )
        full_indices = indices.to(device)
        gradients: dict[torch.nn.Parameter, torch.Tensor] = {}
        total = 0.0
        for micro in indices.split(microbatch_size):
            packed = pack_stage2_batch(
                Stage2BatchDescriptor(task, micro), {task: self.train[task]},
                self.entities, self.packer, needs_entities=True,
                include_raw_atom_targets=False, pin_memory=False,
            ).to(device, non_blocking=False)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=self.amp_dtype == "bf16",
            ):
                prediction = model.predict_simulation(task, packed, self.train_device[task])
                if not torch.isfinite(prediction).all():
                    raise RuntimeError(f"Non-finite Stage 3 simulation prediction: {task}")
                loss = simulation_loss(task, prediction, packed, self.train_device[task], full_indices)
            for parameter, gradient in zip(
                parameters,
                (torch.autograd.grad(loss, parameters, allow_unused=True, materialize_grads=False)
                 if parameters else ()),
                strict=True,
            ):
                if gradient is not None:
                    value = gradient.detach().float()
                    gradients[parameter] = gradients.get(parameter, torch.zeros_like(value)) + value
            total += float(loss.detach().float())
        return gradients, total


def load_simulation_training_data(
    artifacts_dir: Path, source_payload: Mapping[str, Any], vocabulary: Any,
    device: torch.device, *, amp_dtype: str = "bf16",
) -> SimulationTrainingData:
    planned = load_simulation_plan_data(artifacts_dir, source_payload)
    valid = {task: Stage2TaskDataset(artifacts_dir, task, "valid") for task in planned.train}
    entities = Stage2EntityDataset(artifacts_dir)
    if source_payload.get("kind") in {"ilume_stage2_home_final_v4", "ilume_stage2_entity_home_final_v4"}:
        from common.identity import tensor_state_hash

        manifest = entities.frozen_entity_manifest
        if manifest is None or manifest["identity"]["payload"]["stage1_state_hash"] != tensor_state_hash("stage1.encoding-state", source_payload["stage1_backbone"]):
            raise ValueError("Stage 3 simulation frozen entity cache source mismatch")
    return SimulationTrainingData(
        entities, planned.train, valid,
        MultimodalPacker(vocabulary),
        {task: Stage2DeviceTaskData.from_dataset(data, device) for task, data in planned.train.items()},
        {task: Stage2DeviceTaskData.from_dataset(data, device) for task, data in valid.items()},
        planned.data_identity, amp_dtype,
    )


def load_simulation_plan_data(
    artifacts_dir: Path, source_payload: Mapping[str, Any],
) -> SimulationPlanData:
    from stage2.identity import metadata_identity

    metadata = json.loads((artifacts_dir / "metadata.json").read_text(encoding="utf-8"))
    identity = metadata_identity(metadata, "data", context="Stage 3 simulation data")
    registry = load_artifact_registry(artifacts_dir)
    if (identity["hash"] != source_payload["stage2_data_identity"]["hash"]
            or registry.registry_hash != source_payload["registry_hash"]):
        raise ValueError("Stage 3 simulation data differs from the Stage 2 source")
    tasks = AUXILIARY_SIMULATION_TASKS if source_payload["kind"] == "ilume_stage2_entity_home_final_v4" else SIMULATION_TASKS
    train = {task: Stage2TaskDataset(artifacts_dir, task, "train") for task in tasks}
    return SimulationPlanData(train, identity["hash"])


def extend_simulation_plan(
    plan: dict[str, Any], config: Any, model: SimulationEntityModel,
    data: SimulationPlanData | SimulationTrainingData, source_payload: Mapping[str, Any],
) -> None:
    """Add raw-coverage simulation branches without changing Phase 1 exposure."""
    tasks = simulation_tasks(config)
    phases = plan["phases"]
    simulation_recipe = config.training.simulation
    if simulation_recipe is None:
        raise ValueError("Stage 3 simulation training recipe is missing")
    size_class = simulation_recipe.private_size_class
    class_recipe = config.training.three_phase.private_classes[size_class]
    training = config.training.three_phase
    counts = {task: len(data.train[task]) for task in tasks}
    allocation = {task: simulation_recipe.batch_size for task in tasks}
    task_steps = {task: math.ceil(count / simulation_recipe.batch_size) for task, count in counts.items()}
    plan["data"]["N_t"].update(counts)
    plan["data"]["B_t"].update(allocation)
    plan["data"]["task_steps"].update(task_steps)
    plan["data"]["epoch_exposures"].update(counts)
    plan["data"]["effective_composite_batch_size"] += simulation_recipe.batch_size * len(tasks)

    def owner_recipe(lr: float, epochs: int, updates: int, floor: float, **extra: Any) -> dict[str, Any]:
        return {
            "nominal_lr": lr, "terminal_lr": lr * floor,
            "nominal_epochs": epochs, "effective_epochs": epochs,
            "freeze_epoch": epochs, "updates_per_epoch": updates,
            "actual_update_budget": epochs * updates,
            "warmup_updates": 0, **extra,
        }

    by_group = {
        group: tuple(task for task in tasks if model.task_specs[task].meta_group == group)
        for group in ("thermophysical", "electronic_structure")
        if any(model.task_specs[task].meta_group == group for task in tasks)
    }
    for group, group_tasks in by_group.items():
        budget = config.groups[group].phase2
        assert budget is not None
        branches = phases["phase2"]["branches"]
        steps = max(
            branches.get(group, {}).get("steps_per_epoch", 0),
            *(task_steps[task] for task in group_tasks),
        )
        if group not in branches:
            branches[group] = {
                "epochs": budget.epochs, "steps_per_epoch": steps,
                "owners": {
                    group_owner(group).label: owner_recipe(
                        budget.lr, budget.epochs, steps, training.phase2_min_lr_ratio,
                        capacity=model.resolved_capacity_recipe()["groups"][group],
                    ),
                },
            }
        else:
            branch = branches[group]
            branch["steps_per_epoch"] = steps
            group_recipe = branch["owners"][group_owner(group).label]
            if not config.is_entity_home:
                group_recipe["updates_per_epoch"] = steps
                group_recipe["actual_update_budget"] = budget.epochs * steps
        for task in group_tasks:
            effective = min(class_recipe.phase2_epochs, budget.epochs)
            lr = class_recipe.phase1.lr * training.phase1_min_lr_ratio
            recipe = owner_recipe(
                lr, effective, task_steps[task], training.phase2_min_lr_ratio,
                size_class=size_class, train_rows=counts[task],
                capacity=model.resolved_capacity_recipe()["tasks"][task],
            )
            recipe["nominal_epochs"] = class_recipe.phase2_epochs
            branches[group]["owners"][private_owner(task).label] = recipe

    for task in tasks:
        lr = (class_recipe.phase1.lr * training.phase1_min_lr_ratio
              * training.phase2_min_lr_ratio)
        owner = private_owner(task)
        phases["phase3"]["branches"][task] = {
            "epochs": class_recipe.phase3_epochs,
            "steps_per_epoch": task_steps[task],
            "carried_from_anchor": False,
            "owners": {
                owner.label: owner_recipe(
                    lr, class_recipe.phase3_epochs, task_steps[task],
                    training.phase3_min_lr_ratio,
                    size_class=size_class, train_rows=counts[task],
                    capacity=model.resolved_capacity_recipe()["tasks"][task],
                ),
            },
        }
    plan["simulation_training"] = {
        "tasks": list(tasks),
        "groups": {group: list(tasks) for group, tasks in by_group.items()},
        "stage2_data_identity": data.data_identity,
        "stage2_full_state_hash": source_payload["full_model_state_hash"],
        "source_owner_initialization": "stage2_final_private_scalar_entity_v4" if config.is_entity_home else "stage2_final_group_private_atom_adapter_v1",
        "feature_source": "frozen_stage1_raw_entity_slots_v1",
        "loss": "stage2_physics_scalar_smooth_l1_v4" if config.is_entity_home else "stage2_physics_smooth_l1_molecule_equal_charge_v1",
        "sampling": "raw_without_replacement_256_per_task_v1",
        **({"phase2_gradient_policy": "simulation_private_only_v1"} if config.is_entity_home else {}),
        "phase1": "simulation_private_frozen_v4" if config.is_entity_home else "simulation_private_and_electronic_group_frozen_v1",
        "recipe": {
            "batch_size": simulation_recipe.batch_size,
            "microbatch_size": simulation_recipe.microbatch_size,
            "private_size_class": size_class,
            **({"shared_group_task_weight": simulation_recipe.shared_group_task_weight} if not config.is_entity_home else {}),
        },
    }
    plan["format_version"] = 14 if config.is_entity_home else 11 if config.initialization.representation_contract == "dual_view_v4" else 10
