from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from stage1.features import ROLE_TO_ID
from .registry import Stage2Registry


RECONSTRUCTION_MODULES = (
    "smiles_head", "atom_trunk", "bond_trunk", "atom_heads", "bond_heads",
    "descriptor_heads", "descriptor_decoder", "fingerprint_heads",
)


def build_model_contract(
    d_model: int,
    n_heads: int,
    registry: Stage2Registry,
    *,
    atom_dim: int | None = None,
    representation_kind: str = "cls_v1",
    object_layers: int,
    object_ffn_dim: int,
    dropout: float,
) -> dict[str, Any]:
    atom_dim = d_model if atom_dim is None else atom_dim
    tasks: dict[str, Any] = {}
    for spec in registry.tasks:
        family = "atom" if spec.target_level == "atom" else ("interaction" if spec.topology == "interaction" else "object")
        input_dim = (
            atom_dim
            if family == "atom" and representation_kind in {"cls_rdkit_concat_v2", "dual_view_learned_v4"}
            else d_model + (0 if family == "atom" else len(spec.condition_columns))
        )
        task_contract = {
            "topology": spec.topology, "head_family": family,
            "condition_dim": len(spec.condition_columns), "input_dim": input_dim,
            "output_dim": len(spec.target_columns),
        }
        if family == "atom":
            task_contract.update(
                {"atom_dim": atom_dim, "object_projection_dim": atom_dim}
            )
            if representation_kind in {"cls_rdkit_concat_v2", "dual_view_learned_v4"}:
                task_contract["object_context_dim"] = d_model
        tasks[spec.task_id] = task_contract
    contract = {
        "d_model": d_model,
        "n_heads": n_heads,
        "object_encoder": {"layers": object_layers, "ffn_dim": object_ffn_dim, "dropout": dropout},
        "regression_head_hidden_dims": [d_model, d_model // 2],
        "role_to_id": dict(ROLE_TO_ID), "tasks": tasks,
    }
    if representation_kind == "cls_rdkit_concat_v2":
        contract.update(
            {
                "entity_dim": d_model,
                "atom_dim": atom_dim,
                "representation_kind": representation_kind,
            }
        )
    return contract


def ObjectEncoder(*args, **kwargs):
    raise ValueError("ObjectEncoder retired; use the historical Git revision for old models")


def build_object_encoder(*args, **kwargs):
    raise ValueError("ObjectEncoder retired; use the historical Git revision for old models")


def object_encoder_contract(*args, **kwargs):
    raise ValueError("ObjectEncoder retired; use the historical Git revision for old models")


@dataclass(frozen=True)
class ObjectEntityEncoding:
    entity_embedding: torch.Tensor
    atom_states: torch.Tensor
    atom_batch: torch.Tensor


def encode_object_entities(backbone, batch):
    """Descriptors enter only the downstream entity input, never Stage1."""
    if not getattr(backbone.config, "is_dual_view", False):
        return backbone.encode_entity(batch)
    if batch.frozen_entity_embedding is not None:
        return ObjectEntityEncoding(batch.frozen_entity_embedding, batch.frozen_atom_states, batch.graphs.atom_batch)
    backbone.eval()
    with torch.no_grad():
        encoded = backbone.encode_entity(batch)
    return ObjectEntityEncoding(torch.cat((encoded.entity_embedding, batch.descriptors), -1), encoded.atom_states, encoded.atom_batch)


class RDKitDescriptorBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 1024,
        output_dim: int = 512,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim != 1024 or output_dim != 512:
            raise ValueError("RDKit Stage 2 descriptor encoder contract mismatch")
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.config = SimpleNamespace(
            model=SimpleNamespace(d_model=output_dim, n_heads=8)
        )
        self.entity_dim = output_dim
        self.atom_dim = output_dim
        self.representation_kind = "rdkit_2d_stage2"

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.layers[0].in_features:
            raise ValueError("RDKit Stage 2 descriptor input width mismatch")
        return self.layers(values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.encode(values)


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("RegressionHead hidden_dim must be even")
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, output_dim),
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
    def __init__(self, atom_dim: int, object_dim: int, dropout: float) -> None:
        super().__init__()
        self.object_projection = nn.Linear(object_dim, atom_dim)
        nn.init.zeros_(self.object_projection.weight)
        nn.init.zeros_(self.object_projection.bias)
        self.normalization = nn.LayerNorm(atom_dim)
        self.regressor = nn.Sequential(
            nn.Linear(atom_dim, atom_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(atom_dim, 1),
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


def Stage2ObjectModel(*args, **kwargs):
    raise ValueError("ObjectEncoder retired; use the historical Git revision")


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
    "ObjectEncoder", "RDKitDescriptorBackbone", "RECONSTRUCTION_MODULES",
    "Stage2ForwardOutput", "Stage2ObjectModel",
    "element_mean_smooth_l1_loss", "masked_smooth_l1_loss", "masked_target_macro_smooth_l1_loss",
    "molecule_equal_smooth_l1_loss", "stage2_optimizer_groups",
    "build_model_contract",
]
