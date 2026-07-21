from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .graph import ATOM_CARDINALITIES, BOND_CARDINALITIES, PackedGraph


class SmilesEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
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

    def forward(
        self,
        token_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)[
            None, :, :
        ]
        hidden = self.dropout(self.input_norm(hidden))
        return self.encoder(hidden, src_key_padding_mask=padding_mask)


def _categorical_to_one_hot(
    values: torch.Tensor,
    cardinalities: Sequence[int],
) -> torch.Tensor:
    encoded = [
        F.one_hot(values[:, column], num_classes=cardinality)
        for column, cardinality in enumerate(cardinalities)
    ]
    return torch.cat(encoded, dim=-1).float()


class DirectedMessagePassingEncoder(nn.Module):
    def __init__(self, d_model: int, depth: int, dropout: float) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("D-MPNN depth must be at least 1")
        self.depth = depth
        self.atom_feature_dim = sum(ATOM_CARDINALITIES) + 1
        self.bond_feature_dim = sum(BOND_CARDINALITIES)
        self.atom_mask_feature = nn.Parameter(torch.zeros(self.atom_feature_dim))
        self.bond_mask_feature = nn.Parameter(torch.zeros(self.bond_feature_dim))
        nn.init.normal_(self.atom_mask_feature, std=0.02)
        nn.init.normal_(self.bond_mask_feature, std=0.02)

        self.input_projection = nn.Linear(
            self.atom_feature_dim + self.bond_feature_dim,
            d_model,
            bias=False,
        )
        self.message_projection = nn.Linear(d_model, d_model, bias=False)
        self.atom_output = nn.Linear(self.atom_feature_dim + d_model, d_model)
        self.bond_output = nn.Linear(self.bond_feature_dim + d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.atom_norm = nn.LayerNorm(d_model)
        self.bond_norm = nn.LayerNorm(d_model)

    def _feature_inputs(
        self,
        graph: PackedGraph,
        atom_mask: torch.Tensor,
        bond_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atom_features = torch.cat(
            [
                _categorical_to_one_hot(
                    graph.atom_categorical, ATOM_CARDINALITIES
                ),
                graph.atom_continuous,
            ],
            dim=-1,
        )
        bond_features = _categorical_to_one_hot(
            graph.bond_categorical, BOND_CARDINALITIES
        )
        atom_features = torch.where(
            atom_mask[:, None],
            self.atom_mask_feature[None, :],
            atom_features,
        )
        bond_features = torch.where(
            bond_mask[:, None],
            self.bond_mask_feature[None, :],
            bond_features,
        )
        return atom_features, bond_features

    def forward(
        self,
        graph: PackedGraph,
        atom_mask: torch.Tensor,
        bond_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atom_features, bond_features = self._feature_inputs(
            graph, atom_mask, bond_mask
        )
        atom_count = atom_features.shape[0]
        bond_count = bond_features.shape[0]
        if bond_count:
            source, destination = graph.directed_edge_index
            directed_bond_features = bond_features[graph.directed_to_bond]
            initial = self.activation(
                self.input_projection(
                    torch.cat([atom_features[source], directed_bond_features], dim=-1)
                )
            )
            hidden = initial
            for _ in range(self.depth - 1):
                incoming = hidden.new_zeros((atom_count, hidden.shape[-1]))
                incoming.index_add_(0, destination, hidden)
                messages = incoming[source] - hidden[graph.reverse_edge_index]
                hidden = self.dropout(
                    self.activation(initial + self.message_projection(messages))
                )
            atom_messages = hidden.new_zeros((atom_count, hidden.shape[-1]))
            atom_messages.index_add_(0, destination, hidden)
            paired_messages = hidden[0::2] + hidden[1::2]
            bond_tokens = self.bond_norm(
                self.activation(
                    self.bond_output(
                        torch.cat([bond_features, paired_messages], dim=-1)
                    )
                )
            )
        else:
            atom_messages = atom_features.new_zeros((atom_count, self.atom_output.out_features))
            bond_tokens = atom_features.new_empty((0, self.bond_output.out_features))

        atom_tokens = self.atom_norm(
            self.activation(
                self.atom_output(torch.cat([atom_features, atom_messages], dim=-1))
            )
        )
        return atom_tokens, bond_tokens


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.layers(self.norm(values))


class DescriptorEncoder(nn.Module):
    def __init__(
        self,
        descriptor_dim: int,
        hidden_dim: int,
        blocks: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(descriptor_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, dropout) for _ in range(blocks)]
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        values: torch.Tensor,
        mask_indicator: torch.Tensor,
    ) -> torch.Tensor:
        masked_values = values.masked_fill(mask_indicator, 0.0)
        inputs = torch.cat([masked_values, mask_indicator.float()], dim=-1)
        hidden = self.input_projection(inputs)
        hidden = self.residual_blocks(hidden)
        return self.bottleneck(hidden)
