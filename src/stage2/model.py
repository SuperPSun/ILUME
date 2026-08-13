from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn

from stage1.data import MultimodalBatch
from stage1.features import ROLE_TO_ID
from stage1.model import MultimodalPretrainModel


IL_TASKS = ("density", "heat_capacity", "thermal_expansion")
QM_TASK = "simulated_qm_elec_hf"
TRANSFER_TASK = "transfer_organic"
RECONSTRUCTION_MODULES = (
    "smiles_head",
    "atom_trunk",
    "bond_trunk",
    "atom_heads",
    "bond_heads",
    "descriptor_heads",
    "fingerprint_heads",
)


class ObjectEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.object_cls = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.object_cls, std=0.02)
        self.role_embedding = nn.Embedding(len(ROLE_TO_ID), d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.residual_projection = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)
        self.output_normalization = nn.LayerNorm(d_model)

    def forward(
        self,
        entity_cls: torch.Tensor,
        entity_roles: torch.Tensor,
    ) -> torch.Tensor:
        if entity_cls.ndim != 3:
            raise ValueError("ObjectEncoder entity_cls must have shape [B, S, D]")
        if entity_roles.shape != entity_cls.shape[:2]:
            raise ValueError("ObjectEncoder entity role shape mismatch")
        if entity_cls.shape[-1] != self.d_model:
            raise ValueError("ObjectEncoder entity dimension mismatch")
        slot_count = entity_cls.shape[1]
        if slot_count == 1:
            expected = entity_roles.new_full(
                entity_roles.shape, ROLE_TO_ID["neutral"]
            )
            if not torch.equal(entity_roles, expected):
                raise ValueError("Molecule object requires one neutral entity")
            residual = entity_cls[:, 0]
        elif slot_count == 2:
            expected = entity_roles.new_tensor(
                [ROLE_TO_ID["cation"], ROLE_TO_ID["anion"]]
            ).expand_as(entity_roles)
            if not torch.equal(entity_roles, expected):
                raise ValueError("IL object requires ordered cation and anion")
            residual = entity_cls.mean(dim=1)
        else:
            raise ValueError("ObjectEncoder supports molecule or IL topology only")
        batch_size = entity_cls.shape[0]
        cls = self.object_cls.expand(batch_size, 1, -1)
        inputs = torch.cat(
            (cls, entity_cls + self.role_embedding(entity_roles)), dim=1
        )
        encoded = self.encoder(inputs)
        delta = self.residual_projection(encoded[:, 0])
        return self.output_normalization(residual + delta)


class RegressionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class TransferHead(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.interaction = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.normalization = nn.LayerNorm(d_model)
        self.regressor = RegressionHead(d_model, d_model, 1, dropout)

    def forward(
        self, solute_object: torch.Tensor, solvent_object: torch.Tensor
    ) -> torch.Tensor:
        interactions = torch.cat(
            (
                solute_object,
                solvent_object,
                torch.abs(solute_object - solvent_object),
                solute_object * solvent_object,
            ),
            dim=-1,
        )
        values = self.normalization(
            solute_object + self.interaction(interactions)
        )
        return self.regressor(values)


@dataclass(frozen=True)
class Stage2ForwardOutput:
    predictions: torch.Tensor
    property_loss: torch.Tensor
    teacher_loss: torch.Tensor
    total_loss: torch.Tensor
    student_slots: torch.Tensor
    teacher_slots: torch.Tensor


def masked_smooth_l1_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    if target_mask.shape != targets.shape or predictions.shape != targets.shape:
        raise ValueError("Stage 2 prediction, target, and mask shapes must match")
    elementwise = F.smooth_l1_loss(predictions, targets, reduction="none")
    counts = target_mask.sum(dim=0)
    valid_columns = counts > 0
    if not bool(valid_columns.any()):
        raise ValueError("Stage 2 batch has no supervised targets")
    per_target = (
        (elementwise * target_mask.to(elementwise.dtype)).sum(dim=0)
        / counts.clamp_min(1).to(elementwise.dtype)
    )
    return per_target[valid_columns].mean()


class Stage2ObjectModel(nn.Module):
    def __init__(
        self,
        backbone: MultimodalPretrainModel,
        *,
        object_layers: int = 2,
        object_ffn_dim: int = 1024,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        model_config = backbone.config.model
        d_model = model_config.d_model
        self.object_encoder = ObjectEncoder(
            d_model,
            model_config.n_heads,
            num_layers=object_layers,
            feedforward_dim=object_ffn_dim,
            dropout=dropout,
        )
        self.property_heads = nn.ModuleDict(
            {
                **{
                    task: RegressionHead(d_model + 1, d_model, 1, dropout)
                    for task in IL_TASKS
                },
                QM_TASK: RegressionHead(d_model, d_model, 11, dropout),
            }
        )
        self.transfer_head = TransferHead(d_model, dropout)
        self.set_backbone_trainable(True)

    @property
    def model_contract(self) -> dict[str, int]:
        return {
            "d_model": self.backbone.config.model.d_model,
            "n_heads": self.backbone.config.model.n_heads,
        }

    def encode_entities(self, batch: MultimodalBatch) -> torch.Tensor:
        return self.backbone.encode(batch)

    def encode_object(
        self,
        entity_cls: torch.Tensor,
        entity_roles: torch.Tensor,
    ) -> torch.Tensor:
        return self.object_encoder(entity_cls, entity_roles)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for name, parameter in self.backbone.named_parameters():
            is_reconstruction = any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in RECONSTRUCTION_MODULES
            )
            parameter.requires_grad_(trainable and not is_reconstruction)

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        for name, parameter in self.backbone.named_parameters():
            if not any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in RECONSTRUCTION_MODULES
            ):
                yield parameter

    def new_module_parameters(self) -> Iterator[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if not name.startswith("backbone."):
                yield parameter

    def predict(
        self,
        task: str,
        slots: torch.Tensor,
        roles: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        if task in IL_TASKS:
            if slots.shape[1] != 2 or conditions.shape[1] != 1:
                raise ValueError("IL tasks require two entities and temperature")
            object_cls = self.encode_object(slots, roles)
            return self.property_heads[task](
                torch.cat((object_cls, conditions), dim=-1)
            )
        if task == QM_TASK:
            if slots.shape[1] != 1 or conditions.shape[1] != 0:
                raise ValueError("QM task requires one entity and no conditions")
            return self.property_heads[task](self.encode_object(slots, roles))
        if task == TRANSFER_TASK:
            if slots.shape[1] != 2 or conditions.shape[1] != 0:
                raise ValueError(
                    "Transfer task requires solute and solvent molecules"
                )
            solute = self.encode_object(slots[:, :1], roles[:, :1])
            solvent = self.encode_object(slots[:, 1:], roles[:, 1:])
            return self.transfer_head(solute, solvent)
        raise ValueError(f"Unknown Stage 2 task: {task}")

    def forward(
        self,
        task: str,
        entities: MultimodalBatch,
        entity_positions: torch.Tensor,
        conditions: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        teacher_embeddings: torch.Tensor,
        *,
        lambda_teacher: float,
    ) -> Stage2ForwardOutput:
        student_unique = self.encode_entities(entities)
        student_slots = student_unique[entity_positions]
        teacher_slots = teacher_embeddings[entity_positions]
        roles = entities.roles[entity_positions]
        predictions = self.predict(task, student_slots, roles, conditions)
        property_loss = masked_smooth_l1_loss(
            predictions, targets, target_mask
        )
        teacher_loss = torch.square(student_slots - teacher_slots).mean()
        total_loss = property_loss + lambda_teacher * teacher_loss
        return Stage2ForwardOutput(
            predictions=predictions,
            property_loss=property_loss,
            teacher_loss=teacher_loss,
            total_loss=total_loss,
            student_slots=student_slots,
            teacher_slots=teacher_slots,
        )


def stage2_optimizer_groups(
    model: Stage2ObjectModel,
    *,
    backbone_learning_rate: float,
    new_module_learning_rate: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    backbone = list(model.backbone_parameters())
    new_modules = list(model.new_module_parameters())
    if not backbone or not new_modules:
        raise ValueError("Stage 2 optimizer groups cannot be empty")
    return [
        {
            "params": backbone,
            "lr": backbone_learning_rate,
            "weight_decay": weight_decay,
        },
        {
            "params": new_modules,
            "lr": new_module_learning_rate,
            "weight_decay": weight_decay,
        },
    ]
