from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn

from stage1.data import MultimodalBatch
from stage1.features import ROLE_TO_ID
from stage1.model import EncodedEntityStates, MultimodalPretrainModel
from .registry import Stage2Registry, TaskSpec


RECONSTRUCTION_MODULES = (
    "smiles_head", "atom_trunk", "bond_trunk", "atom_heads", "bond_heads",
    "descriptor_heads", "fingerprint_heads",
)


def build_model_contract(
    d_model: int,
    n_heads: int,
    registry: Stage2Registry,
    *,
    object_layers: int,
    object_ffn_dim: int,
    dropout: float,
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for spec in registry.tasks:
        family = "atom" if spec.target_level == "atom" else ("interaction" if spec.topology == "interaction" else "object")
        input_dim = d_model if family == "atom" else d_model + len(spec.condition_columns)
        tasks[spec.task_id] = {
            "topology": spec.topology, "head_family": family,
            "condition_dim": len(spec.condition_columns), "input_dim": input_dim,
            "output_dim": len(spec.target_columns),
            **({"atom_dim": d_model, "object_projection_dim": d_model} if family == "atom" else {}),
        }
    return {
        "d_model": d_model, "n_heads": n_heads,
        "object_encoder": {"layers": object_layers, "ffn_dim": object_ffn_dim, "dropout": dropout},
        "role_to_id": dict(ROLE_TO_ID), "tasks": tasks,
    }


class ObjectEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, *, num_layers: int, feedforward_dim: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.object_cls = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.object_cls, std=0.02)
        self.role_embedding = nn.Embedding(len(ROLE_TO_ID), d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=feedforward_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False,
        )
        self.residual_projection = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)
        self.output_normalization = nn.LayerNorm(d_model)

    def forward(self, entity_cls: torch.Tensor, entity_roles: torch.Tensor) -> torch.Tensor:
        if entity_cls.ndim != 3 or entity_roles.shape != entity_cls.shape[:2] or entity_cls.shape[-1] != self.d_model:
            raise ValueError("ObjectEncoder entity tensor contract mismatch")
        slots = entity_cls.shape[1]
        if slots == 1:
            if entity_roles.device.type == "cpu" and not bool(
                torch.isin(entity_roles, torch.tensor(tuple(ROLE_TO_ID.values()))).all()
            ):
                raise ValueError("Single object has an invalid entity role")
            residual = entity_cls[:, 0]
        elif slots == 2:
            if entity_roles.device.type == "cpu" and not torch.equal(
                entity_roles,
                torch.tensor([ROLE_TO_ID["cation"], ROLE_TO_ID["anion"]]).expand_as(entity_roles),
            ):
                raise ValueError("Ionic-liquid object requires ordered cation and anion")
            residual = entity_cls.mean(dim=1)
        else:
            raise ValueError("ObjectEncoder supports one entity or ordered cation/anion")
        cls = self.object_cls.expand(entity_cls.shape[0], 1, -1)
        encoded = self.encoder(torch.cat((cls, entity_cls + self.role_embedding(entity_roles)), dim=1))
        return self.output_normalization(residual + self.residual_projection(encoded[:, 0]))


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class InteractionHead(nn.Module):
    def __init__(self, d_model: int, condition_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.interaction = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, d_model),
        )
        self.normalization = nn.LayerNorm(d_model)
        self.regressor = RegressionHead(d_model + condition_dim, d_model, output_dim, dropout)

    def forward(self, first: torch.Tensor, second: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        interactions = torch.cat((first, second, torch.abs(first - second), first * second), dim=-1)
        value = self.normalization(first + self.interaction(interactions))
        return self.regressor(torch.cat((value, conditions), dim=-1))


class AtomPropertyHead(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.object_projection = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.object_projection.weight)
        nn.init.zeros_(self.object_projection.bias)
        self.normalization = nn.LayerNorm(d_model)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, atoms: torch.Tensor, object_state: torch.Tensor) -> torch.Tensor:
        values = self.normalization(atoms + self.object_projection(object_state))
        return self.regressor(values).squeeze(-1)


@dataclass(frozen=True)
class Stage2ForwardOutput:
    predictions: torch.Tensor
    physics_loss: torch.Tensor
    teacher_loss: torch.Tensor
    student_slots: torch.Tensor
    teacher_slots: torch.Tensor

    @property
    def property_loss(self) -> torch.Tensor:
        return self.physics_loss


def masked_target_macro_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if predictions.shape != targets.shape or mask.shape != targets.shape:
        raise ValueError("Stage 2 prediction, target, and mask shapes must match")
    values = F.smooth_l1_loss(predictions, targets, reduction="none")
    counts = mask.sum(dim=0)
    valid = counts > 0
    per_target = (values * mask.to(values.dtype)).sum(dim=0) / counts.clamp_min(1).to(values.dtype)
    return (per_target * valid.to(per_target.dtype)).sum() / valid.sum().clamp_min(1)


def element_mean_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if predictions.shape != targets.shape or mask.shape != targets.shape:
        raise ValueError("element_mean target tensor contract mismatch")
    return F.smooth_l1_loss(predictions, targets, reduction="mean")


def molecule_equal_smooth_l1_loss(
    predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
    atom_sample_indices: torch.Tensor, molecule_count: int,
) -> torch.Tensor:
    if predictions.shape != targets.shape or mask.shape != targets.shape:
        raise ValueError("Atom prediction/target shapes must match")
    weights = mask.to(torch.float32)
    losses = F.smooth_l1_loss(predictions, targets, reduction="none").to(torch.float32)
    weighted_losses = losses * weights
    sums = torch.zeros(
        molecule_count, dtype=torch.float32, device=predictions.device,
    ).index_add_(0, atom_sample_indices, weighted_losses)
    counts = torch.zeros_like(sums).index_add_(0, atom_sample_indices, weights)
    return (sums / counts.clamp_min(1.0)).mean()


# Backward-compatible public name; Object v3 callers select the mode explicitly.
masked_smooth_l1_loss = masked_target_macro_smooth_l1_loss


class Stage2ObjectModel(nn.Module):
    def __init__(self, backbone: MultimodalPretrainModel, registry: Stage2Registry, *, object_layers: int = 2, object_ffn_dim: int = 1024, dropout: float = 0.10) -> None:
        super().__init__()
        self.backbone = backbone
        self.registry = registry
        self.specs = {task.task_id: task for task in registry.tasks}
        config = backbone.config.model
        d_model = config.d_model
        self.object_encoder = ObjectEncoder(d_model, config.n_heads, num_layers=object_layers, feedforward_dim=object_ffn_dim, dropout=dropout)
        self.object_heads = nn.ModuleDict()
        self.interaction_heads = nn.ModuleDict()
        self.atom_heads = nn.ModuleDict()
        for task in registry.tasks:
            if task.target_level == "atom":
                self.atom_heads[task.task_id] = AtomPropertyHead(d_model, dropout)
            elif task.topology == "interaction":
                self.interaction_heads[task.task_id] = InteractionHead(d_model, len(task.condition_columns), len(task.target_columns), dropout)
            else:
                self.object_heads[task.task_id] = RegressionHead(d_model + len(task.condition_columns), d_model, len(task.target_columns), dropout)
        self._object_config = {"layers": object_layers, "ffn_dim": object_ffn_dim, "dropout": dropout}
        self.set_backbone_trainable(True)

    @property
    def model_contract(self) -> dict[str, Any]:
        return build_model_contract(
            self.backbone.config.model.d_model,
            self.backbone.config.model.n_heads,
            self.registry,
            object_layers=self._object_config["layers"],
            object_ffn_dim=self._object_config["ffn_dim"],
            dropout=self._object_config["dropout"],
        )

    def encode_entities(self, batch: MultimodalBatch) -> torch.Tensor:
        return self.backbone.encode(batch)

    def encode_entity_states(self, batch: MultimodalBatch) -> EncodedEntityStates:
        return self.backbone.encode_states(batch)

    def encode_object(self, entity_cls: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
        return self.object_encoder(entity_cls, roles)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for name, parameter in self.backbone.named_parameters():
            reconstruction = any(name == prefix or name.startswith(prefix + ".") for prefix in RECONSTRUCTION_MODULES)
            parameter.requires_grad_(trainable and not reconstruction)

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        for name, parameter in self.backbone.named_parameters():
            if not any(name == prefix or name.startswith(prefix + ".") for prefix in RECONSTRUCTION_MODULES):
                yield parameter

    def object_encoder_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.object_encoder.parameters()

    def task_head_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.object_heads.parameters()
        yield from self.interaction_heads.parameters()
        yield from self.atom_heads.parameters()

    def new_module_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.object_encoder_parameters()
        yield from self.task_head_parameters()

    def predict_object(self, spec: TaskSpec, slots: torch.Tensor, roles: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        if spec.topology == "interaction":
            if slots.shape[1] != 2:
                raise ValueError("Interaction task requires two entity slots")
            first = self.encode_object(slots[:, :1], roles[:, :1])
            second = self.encode_object(slots[:, 1:], roles[:, 1:])
            return self.interaction_heads[spec.task_id](first, second, conditions)
        expected = 2 if spec.topology == "ionic_liquid" else 1
        if slots.shape[1] != expected:
            raise ValueError("Stage 2 task slot count does not match topology")
        object_state = self.encode_object(slots, roles)
        return self.object_heads[spec.task_id](torch.cat((object_state, conditions), dim=-1))

    def forward_object_from_slots(self, task: str, student_slots: torch.Tensor, roles: torch.Tensor, conditions: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor, teacher_slots: torch.Tensor, *, loss_mode: str, teacher_loss_is_zero: bool = False) -> Stage2ForwardOutput:
        spec = self.specs[task]
        predictions = self.predict_object(spec, student_slots, roles, conditions)
        if loss_mode == "masked_target_macro":
            physics = masked_target_macro_smooth_l1_loss(predictions, targets, target_mask)
        elif loss_mode == "element_mean":
            physics = element_mean_smooth_l1_loss(predictions, targets, target_mask)
        else:
            raise ValueError(f"Unsupported Stage 2 loss mode: {loss_mode}")
        teacher = predictions.new_zeros(()) if teacher_loss_is_zero else torch.square(student_slots - teacher_slots).mean()
        return Stage2ForwardOutput(predictions, physics, teacher, student_slots, teacher_slots)

    def forward_atom_from_states(
        self, task: str, states: EncodedEntityStates, entity_positions: torch.Tensor,
        roles: torch.Tensor, object_slots: torch.Tensor, teacher_slots: torch.Tensor,
        targets: torch.Tensor, target_mask: torch.Tensor,
        atom_state_indices: torch.Tensor, atom_sample_indices: torch.Tensor,
        *, teacher_loss_is_zero: bool = False,
    ) -> Stage2ForwardOutput:
        spec = self.specs[task]
        if spec.target_level != "atom" or entity_positions.shape[1] != 1:
            raise ValueError("Atom task requires one entity slot")
        objects = self.encode_object(object_slots, roles)
        predictions = self.atom_heads[task](
            states.atom_states[atom_state_indices], objects[atom_sample_indices]
        )
        physics = molecule_equal_smooth_l1_loss(
            predictions, targets, target_mask, atom_sample_indices,
            entity_positions.shape[0],
        )
        teacher = predictions.new_zeros(()) if teacher_loss_is_zero else torch.square(object_slots - teacher_slots).mean()
        return Stage2ForwardOutput(predictions, physics, teacher, object_slots, teacher_slots)


def stage2_optimizer_groups(model: Stage2ObjectModel, *, backbone_learning_rate: float, object_encoder_learning_rate: float, task_head_learning_rate: float, weight_decay: float) -> list[dict[str, Any]]:
    groups = [
        {"params": list(model.backbone_parameters()), "lr": backbone_learning_rate, "weight_decay": weight_decay},
        {"params": list(model.object_encoder_parameters()), "lr": object_encoder_learning_rate, "weight_decay": weight_decay},
        {"params": list(model.task_head_parameters()), "lr": task_head_learning_rate, "weight_decay": weight_decay},
    ]
    if any(not group["params"] for group in groups):
        raise ValueError("Stage 2 optimizer groups cannot be empty")
    return groups


__all__ = [
    "ObjectEncoder", "RECONSTRUCTION_MODULES", "Stage2ForwardOutput", "Stage2ObjectModel",
    "element_mean_smooth_l1_loss", "masked_smooth_l1_loss", "masked_target_macro_smooth_l1_loss",
    "molecule_equal_smooth_l1_loss", "stage2_optimizer_groups",
    "build_model_contract",
]
