from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import PretrainConfig
from .data import MultimodalBatch
from .encoders import (
    DescriptorEncoder,
    DirectedMessagePassingEncoder,
    SmilesEncoder,
)
from .fusion import (
    FusionTransformer,
    gather_graph_tokens,
    gather_smiles,
)
from .graph import (
    ATOM_CARDINALITIES,
    ATOM_FEATURE_NAMES,
    BOND_CARDINALITIES,
    BOND_FEATURE_NAMES,
)
from .tokenizer import AISVocabulary


@dataclass(frozen=True)
class PretrainOutput:
    loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    logits: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    fused_cls: torch.Tensor


class TiedMLMHead(nn.Module):
    def __init__(self, d_model: int, embedding: nn.Embedding) -> None:
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.embedding = embedding
        self.bias = nn.Parameter(torch.zeros(embedding.num_embeddings))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(self.transform(hidden), self.embedding.weight, self.bias)


def _masked_multitask_cross_entropy(
    logits: dict[str, torch.Tensor],
    targets: torch.Tensor,
    mask: torch.Tensor,
    feature_names: tuple[str, ...],
    zero_reference: torch.Tensor,
) -> torch.Tensor:
    if not bool(mask.any()):
        return zero_reference.sum() * 0.0
    losses = [
        F.cross_entropy(logits[name][mask], targets[mask, column])
        for column, name in enumerate(feature_names)
    ]
    return torch.stack(losses).mean()


class MultimodalPretrainModel(nn.Module):
    def __init__(
        self,
        config: PretrainConfig,
        vocabulary: AISVocabulary,
    ) -> None:
        super().__init__()
        self.config = config
        model_config = config.model
        self.smiles_encoder = SmilesEncoder(
            vocab_size=len(vocabulary.tokens),
            max_length=config.data.max_smiles_tokens,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            num_layers=model_config.smiles_layers,
            feedforward_dim=model_config.feedforward_dim,
            dropout=model_config.dropout,
        )
        self.graph_encoder = DirectedMessagePassingEncoder(
            d_model=model_config.d_model,
            depth=model_config.graph_depth,
            dropout=model_config.dropout,
        )
        self.descriptor_encoder = DescriptorEncoder(
            descriptor_dim=config.data.descriptor_dim,
            hidden_dim=model_config.descriptor_hidden_dim,
            blocks=model_config.descriptor_blocks,
            d_model=model_config.d_model,
            dropout=model_config.dropout,
        )
        self.fusion = FusionTransformer(
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            num_layers=model_config.fusion_layers,
            feedforward_dim=model_config.feedforward_dim,
            dropout=model_config.dropout,
        )
        self.smiles_head = TiedMLMHead(
            model_config.d_model, self.smiles_encoder.token_embedding
        )
        self.atom_heads = nn.ModuleDict(
            {
                name: nn.Linear(model_config.d_model, cardinality)
                for name, cardinality in zip(
                    ATOM_FEATURE_NAMES, ATOM_CARDINALITIES, strict=True
                )
            }
        )
        self.bond_heads = nn.ModuleDict(
            {
                name: nn.Linear(model_config.d_model, cardinality)
                for name, cardinality in zip(
                    BOND_FEATURE_NAMES, BOND_CARDINALITIES, strict=True
                )
            }
        )
        self.descriptor_head = nn.Linear(
            model_config.d_model, config.data.descriptor_dim
        )

    def forward(self, batch: MultimodalBatch) -> PretrainOutput:
        smiles_tokens = self.smiles_encoder(
            batch.token_ids, batch.token_padding_mask
        )
        atom_tokens, bond_tokens = self.graph_encoder(
            batch.graphs,
            batch.masks.atom_mask,
            batch.masks.bond_mask,
        )
        descriptor_tokens = self.descriptor_encoder(
            batch.descriptors, batch.masks.descriptor_indicator
        )
        fused, layout = self.fusion(
            smiles_tokens,
            batch.token_padding_mask,
            atom_tokens,
            bond_tokens,
            descriptor_tokens,
            batch.graphs,
        )

        fused_smiles = gather_smiles(fused, layout)
        fused_atoms = gather_graph_tokens(
            fused,
            layout.atom_indices,
            batch.graphs.atom_batch,
        )
        fused_bonds = gather_graph_tokens(
            fused,
            layout.bond_indices,
            batch.graphs.bond_batch,
        )
        batch_indices = torch.arange(fused.shape[0], device=fused.device)
        fused_descriptors = fused[batch_indices, layout.descriptor_indices]

        smiles_logits = self.smiles_head(fused_smiles)
        atom_logits = {name: head(fused_atoms) for name, head in self.atom_heads.items()}
        bond_logits = {name: head(fused_bonds) for name, head in self.bond_heads.items()}
        descriptor_logits = self.descriptor_head(fused_descriptors)

        if bool((batch.masks.smiles_labels != -100).any()):
            smiles_loss = F.cross_entropy(
                smiles_logits.reshape(-1, smiles_logits.shape[-1]),
                batch.masks.smiles_labels.reshape(-1),
                ignore_index=-100,
            )
        else:
            smiles_loss = fused_smiles.sum() * 0.0
        atom_loss = _masked_multitask_cross_entropy(
            atom_logits,
            batch.graphs.atom_categorical,
            batch.masks.atom_mask,
            ATOM_FEATURE_NAMES,
            fused_atoms,
        )
        bond_loss = _masked_multitask_cross_entropy(
            bond_logits,
            batch.graphs.bond_categorical,
            batch.masks.bond_mask,
            BOND_FEATURE_NAMES,
            fused_bonds,
        )
        if bool(batch.masks.descriptor_loss_mask.any()):
            descriptor_loss = F.smooth_l1_loss(
                descriptor_logits[batch.masks.descriptor_loss_mask],
                batch.descriptors[batch.masks.descriptor_loss_mask],
            )
        else:
            descriptor_loss = descriptor_logits.sum() * 0.0

        losses = {
            "smiles": smiles_loss,
            "descriptor": descriptor_loss,
            "atom": atom_loss,
            "bond": bond_loss,
        }
        weights = self.config.loss
        total = (
            weights.lambda_smiles * smiles_loss
            + weights.lambda_descriptor * descriptor_loss
            + weights.lambda_atom * atom_loss
            + weights.lambda_bond * bond_loss
        )
        return PretrainOutput(
            loss=total,
            losses=losses,
            logits={
                "smiles": smiles_logits,
                "atom": atom_logits,
                "bond": bond_logits,
                "descriptor": descriptor_logits,
            },
            fused_cls=fused[:, 0],
        )
