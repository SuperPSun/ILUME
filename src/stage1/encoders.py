from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .descriptors import DescriptorSchema
from .fingerprints import FingerprintBatch
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
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.gradient_checkpointing = gradient_checkpointing
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
        if not (self.gradient_checkpointing and self.training):
            return self.encoder(hidden, src_key_padding_mask=padding_mask)
        for layer in self.encoder.layers:
            hidden = checkpoint(
                lambda value, current_layer=layer: current_layer(
                    value, src_key_padding_mask=padding_mask
                ),
                hidden,
                use_reentrant=False,
            )
        if self.encoder.norm is not None:
            hidden = self.encoder.norm(hidden)
        return hidden


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
        parameter_zero = sum(
            parameter.float().sum() for parameter in self.parameters()
        ) * 0.0
        return atom_tokens + parameter_zero, bond_tokens + parameter_zero


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


class _DescriptorGroupEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        blocks: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
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


class DescriptorEncoder(nn.Module):
    def __init__(
        self,
        schema: DescriptorSchema,
        hidden_dim: int,
        blocks: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.group_encoders = nn.ModuleList(
            [
                _DescriptorGroupEncoder(
                    input_dim=len(indices),
                    hidden_dim=hidden_dim,
                    blocks=blocks,
                    d_model=d_model,
                    dropout=dropout,
                )
                if indices
                else nn.Identity()
                for indices in schema.group_indices
            ]
        )
        self.empty_group_tokens = nn.Parameter(
            torch.empty(len(schema.group_indices), d_model)
        )
        nn.init.normal_(self.empty_group_tokens, std=0.02)

    def forward(
        self,
        values: torch.Tensor,
        mask_indicator: torch.Tensor,
    ) -> torch.Tensor:
        tokens: list[torch.Tensor] = []
        for group_index, (indices, encoder) in enumerate(
            zip(self.schema.group_indices, self.group_encoders, strict=True)
        ):
            if not indices:
                tokens.append(
                    self.empty_group_tokens[group_index][None, :].expand(
                        values.shape[0], -1
                    )
                )
                continue
            index = torch.tensor(indices, dtype=torch.long, device=values.device)
            tokens.append(
                encoder(
                    values.index_select(1, index),
                    mask_indicator.index_select(1, index),
                )
            )
        return torch.stack(tokens, dim=1) + self.empty_group_tokens.sum() * 0.0


class FingerprintEncoder(nn.Module):
    def __init__(
        self,
        families: Sequence[str],
        dimensions: dict[str, int],
        chunk_size: int,
        hidden_dim: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.families = tuple(families)
        self.dimensions = dict(dimensions)
        self.chunk_size = chunk_size
        self.chunk_encoder = nn.Sequential(
            nn.Linear(chunk_size * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.family_embedding = nn.Embedding(max(1, len(self.families)), d_model)
        max_chunks = max(
            (math.ceil(self.dimensions[name] / chunk_size) for name in self.families),
            default=1,
        )
        self.chunk_embedding = nn.Embedding(max_chunks, d_model)

    def forward(
        self,
        batch: FingerprintBatch,
        indicators: dict[str, torch.Tensor],
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, slice]]:
        if not self.families:
            return reference.new_empty((reference.shape[0], 0, reference.shape[-1])), {}
        family_tokens: list[torch.Tensor] = []
        family_slices: dict[str, slice] = {}
        cursor = 0
        for family_index, family in enumerate(self.families):
            values = batch.values[family]
            indicator = indicators[family]
            chunk_count = math.ceil(values.shape[1] / self.chunk_size)
            padded_size = chunk_count * self.chunk_size
            pad = padded_size - values.shape[1]
            masked = values.masked_fill(indicator, 0.0)
            if pad:
                masked = F.pad(masked, (0, pad))
                indicator = F.pad(indicator, (0, pad), value=True)
            chunks = masked.reshape(values.shape[0], chunk_count, self.chunk_size)
            mask_chunks = indicator.reshape(values.shape[0], chunk_count, self.chunk_size)
            encoded = self.chunk_encoder(
                torch.cat([chunks, mask_chunks.float()], dim=-1)
            )
            chunk_ids = torch.arange(chunk_count, device=values.device)
            encoded = (
                encoded
                + self.family_embedding.weight[family_index][None, None, :]
                + self.chunk_embedding(chunk_ids)[None, :, :]
            )
            family_tokens.append(encoded)
            family_slices[family] = slice(cursor, cursor + chunk_count)
            cursor += chunk_count
        return torch.cat(family_tokens, dim=1), family_slices
