from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn

from .config import PretrainConfig, config_from_dict
from .data import MultimodalBatch, PreparedCorpusDataset
from .model import MultimodalPretrainModel
from .tokenizer import SmilesTokenizer


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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class LoadedStage1Model:
    model: MultimodalPretrainModel
    config: PretrainConfig
    vocabulary: SmilesTokenizer
    checkpoint_hash: str
    artifact_hash: str


def load_stage1_model(
    checkpoint_path: str | Path,
    artifact_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
    backbone_dropout: float | None = None,
) -> LoadedStage1Model:
    checkpoint_path = Path(checkpoint_path)
    artifact_dir = Path(artifact_dir)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("format_version") != 3:
        raise ValueError("Stage 2 requires a Stage 1 checkpoint in format v3")
    config = config_from_dict(checkpoint["config"])
    artifact_hash = sha256_file(artifact_dir / "metadata.json")
    if checkpoint.get("artifact_hash") != artifact_hash:
        raise ValueError("Stage 1 checkpoint and preprocessing artifact do not match")
    if backbone_dropout is not None:
        config = replace(
            config,
            model=replace(config.model, dropout=backbone_dropout),
        )
    dataset = PreparedCorpusDataset(
        artifact_dir,
        "train",
        config.data.shard_cache_size,
    )
    vocabulary = SmilesTokenizer.load(artifact_dir / "tokenizer.json")
    model = MultimodalPretrainModel(
        config,
        vocabulary,
        dataset.descriptor_schema,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device)
    return LoadedStage1Model(
        model=model,
        config=config,
        vocabulary=vocabulary,
        checkpoint_hash=sha256_file(checkpoint_path),
        artifact_hash=artifact_hash,
    )


class PairEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.skip = nn.Linear(2 * d_model, d_model)
        self.normalization = nn.LayerNorm(d_model)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        interactions = torch.cat(
            (left, right, torch.abs(left - right), left * right),
            dim=-1,
        )
        ordered = torch.cat((left, right), dim=-1)
        return self.normalization(self.main(interactions) + self.skip(ordered))


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


@dataclass(frozen=True)
class Stage2ForwardOutput:
    predictions: torch.Tensor
    supervised_loss: torch.Tensor
    alignment_loss: torch.Tensor
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


class Stage2AlignmentModel(nn.Module):
    def __init__(
        self,
        backbone: MultimodalPretrainModel,
        *,
        head_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        d_model = backbone.config.model.d_model
        self.il_pair_encoder = PairEncoder(d_model, head_dropout)
        self.transfer_pair_encoder = PairEncoder(d_model, head_dropout)
        self.regressors = nn.ModuleDict(
            {
                **{
                    task: RegressionHead(
                        d_model + 1,
                        d_model,
                        1,
                        head_dropout,
                    )
                    for task in IL_TASKS
                },
                QM_TASK: RegressionHead(
                    d_model,
                    d_model,
                    11,
                    head_dropout,
                ),
                TRANSFER_TASK: RegressionHead(
                    d_model,
                    d_model,
                    1,
                    head_dropout,
                ),
            }
        )
        self.set_backbone_trainable(True)

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

    def head_parameters(self) -> Iterator[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if not name.startswith("backbone."):
                yield parameter

    def predict(
        self,
        task: str,
        slots: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        if task in IL_TASKS:
            if slots.shape[1] != 2 or conditions.shape[1] != 1:
                raise ValueError("IL tasks require two entities and temperature")
            pair = self.il_pair_encoder(slots[:, 0], slots[:, 1])
            return self.regressors[task](
                torch.cat((pair, conditions), dim=-1)
            )
        if task == QM_TASK:
            if slots.shape[1] != 1 or conditions.shape[1] != 0:
                raise ValueError("QM task requires one entity and no conditions")
            return self.regressors[task](slots[:, 0])
        if task == TRANSFER_TASK:
            if slots.shape[1] != 2 or conditions.shape[1] != 0:
                raise ValueError(
                    "Transfer task requires solute and solvent embeddings"
                )
            pair = self.transfer_pair_encoder(slots[:, 0], slots[:, 1])
            return self.regressors[task](pair)
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
        lambda_alignment: float,
    ) -> Stage2ForwardOutput:
        student_unique = self.backbone.encode(entities)
        student_slots = student_unique[entity_positions]
        teacher_slots = teacher_embeddings[entity_positions]
        predictions = self.predict(task, student_slots, conditions)
        supervised_loss = masked_smooth_l1_loss(
            predictions,
            targets,
            target_mask,
        )
        per_slot = torch.square(student_slots - teacher_slots).mean(dim=-1)
        alignment_loss = per_slot.mean(dim=1).mean()
        total_loss = supervised_loss + lambda_alignment * alignment_loss
        return Stage2ForwardOutput(
            predictions=predictions,
            supervised_loss=supervised_loss,
            alignment_loss=alignment_loss,
            total_loss=total_loss,
            student_slots=student_slots,
            teacher_slots=teacher_slots,
        )


def stage2_optimizer_groups(
    model: Stage2AlignmentModel,
    *,
    backbone_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    backbone = [parameter for parameter in model.backbone_parameters()]
    heads = [parameter for parameter in model.head_parameters()]
    if not backbone or not heads:
        raise ValueError("Stage 2 optimizer groups cannot be empty")
    return [
        {
            "params": backbone,
            "lr": backbone_learning_rate,
            "weight_decay": weight_decay,
        },
        {
            "params": heads,
            "lr": head_learning_rate,
            "weight_decay": weight_decay,
        },
    ]
