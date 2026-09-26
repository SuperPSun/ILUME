from __future__ import annotations

from typing import Any, Mapping

import torch

from common.identity import tensor_state_hash
from stage3.data import ResolvedTaskSpec
from stage3.model import Stage3SparseModel


SOURCE_GROUPS = {
    "simulation/density": "thermophysical",
    "simulation/heat_capacity": "thermophysical",
    "simulation/thermal_expansion": "thermophysical",
    "simulation/heat_of_vaporization": "thermophysical",
    "simulation/transfer_organic": "solvation",
    "simulation/homo": "electronic_structure",
    "simulation/lumo": "electronic_structure",
    "simulation/simulated_qm_elec_hf": "electronic_structure",
    "simulation/partial_atomic_charge": "electronic_structure",
}
TRANSFER_GROUPS = ("solvation", "thermophysical")
TRANSFER_GLOBAL_PREFIXES = ("l1_global_experts.", "l1_global_gate.", "l2_global_experts.")
TRANSFER_GROUP_PREFIXES = (
    "l1_group_experts.", "l1_group_gates.", "l1_group_normalizations.",
    "l2_group_experts.", "interactions.",
)


def source_task_specs(registry: Any) -> dict[str, ResolvedTaskSpec]:
    if set(registry.task_ids) != set(SOURCE_GROUPS):
        raise ValueError("Stage2-HoME requires the complete nine-task simulation registry")
    result: dict[str, ResolvedTaskSpec] = {}
    for task in registry.tasks:
        task_id = task.task_id
        group = SOURCE_GROUPS[task_id]
        count = len(task.target_columns)
        ids = (
            tuple(f"{task_id}::target_{index}" for index in range(count))
            if count > 1 else (task_id,)
        )
        for index, model_task in enumerate(ids):
            result[model_task] = ResolvedTaskSpec(
                task_id=model_task, target_column=task.target_columns[index],
                identity_columns=(), condition_columns=tuple(task.condition_columns),
                system_type=task.topology, materialized_path="", split_strategy="",
                cv_repeat=1, meta_group=group,
                partner_mode="interaction" if task.topology == "interaction" else "none",
                primary_slots=("solute",) if task.topology == "interaction" else
                    (("cation", "anion") if task.topology == "ionic_liquid" else ("molecule",)),
                partner_slots=("solvent",) if task.topology == "interaction" else (),
                enabled=True, task_weight=1.0, catalog_schema_version=0, provenance={},
            )
    return result


def model_task_ids(task_id: str, target_count: int) -> tuple[str, ...]:
    if task_id not in SOURCE_GROUPS or target_count < 1:
        raise ValueError("Invalid Stage2-HoME source task")
    return (
        tuple(f"{task_id}::target_{index}" for index in range(target_count))
        if target_count > 1 else (task_id,)
    )


def transferable_state(model: Stage3SparseModel) -> dict[str, torch.Tensor]:
    manifest = model.ownership_manifest()
    state = model.state_dict()
    selected = {
        name: value.detach().cpu().clone()
        for name, value in state.items()
        if manifest.get(name) == "GLOBAL"
        or manifest.get(name) in {f"GROUP:{group}" for group in TRANSFER_GROUPS}
    }
    if not selected or set(selected) == set(state):
        raise ValueError("Stage2-HoME transferable state selection is invalid")
    for name in selected:
        if not (name.startswith(TRANSFER_GLOBAL_PREFIXES) or name.startswith(TRANSFER_GROUP_PREFIXES)):
            raise ValueError(f"Unexpected transferable HoME state: {name}")
    return selected


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    return tensor_state_hash("stage2-home-transfer.shared-state.v1", state)


def load_transferable_state(
    model: Stage3SparseModel, state: Mapping[str, torch.Tensor], expected_hash: str,
) -> tuple[str, ...]:
    if state_hash(state) != expected_hash:
        raise ValueError("Stage2-HoME transferable state hash mismatch")
    target = model.state_dict()
    manifest = model.ownership_manifest()
    expected = {
        name for name, owner in manifest.items()
        if owner == "GLOBAL" or owner in {f"GROUP:{group}" for group in TRANSFER_GROUPS}
    }
    if set(state) != expected:
        raise ValueError("Stage2-HoME transferable owner set is incomplete or contains extra tensors")
    for name, value in state.items():
        if name not in target or value.shape != target[name].shape or value.dtype != target[name].dtype:
            raise ValueError(f"Stage2-HoME transferable tensor mismatch: {name}")
        target[name] = value
    model.load_state_dict(target, strict=True)
    return tuple(sorted(expected))
