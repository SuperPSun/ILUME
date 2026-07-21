from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .graph import PackedGraph


@dataclass(frozen=True)
class FusionLayout:
    smiles_indices: torch.Tensor
    atom_indices: torch.Tensor
    bond_indices: torch.Tensor
    descriptor_indices: torch.Tensor
    padding_mask: torch.Tensor
    sequence_lengths: torch.Tensor


class FusionTransformer(nn.Module):
    CLS_MODALITY = 0
    SMILES_MODALITY = 1
    GRAPH_MODALITY = 2
    DESCRIPTOR_MODALITY = 3

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.modality_embedding = nn.Embedding(4, d_model)
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

    def _assemble(
        self,
        smiles_tokens: torch.Tensor,
        smiles_padding_mask: torch.Tensor,
        atom_tokens: torch.Tensor,
        bond_tokens: torch.Tensor,
        descriptor_tokens: torch.Tensor,
        graph: PackedGraph,
    ) -> tuple[torch.Tensor, FusionLayout]:
        batch_size, smiles_width, d_model = smiles_tokens.shape
        smiles_lengths = (~smiles_padding_mask).sum(dim=1)
        sequence_lengths = torch.empty(
            batch_size, dtype=torch.long, device=smiles_tokens.device
        )
        for row, ((_, atom_count), (_, bond_count)) in enumerate(
            zip(graph.atom_scopes, graph.bond_scopes, strict=True)
        ):
            sequence_lengths[row] = (
                1 + smiles_lengths[row] + atom_count + bond_count + 1
            )
        max_length = int(sequence_lengths.max().item())
        fused_inputs = smiles_tokens.new_zeros((batch_size, max_length, d_model))
        padding_mask = torch.ones(
            (batch_size, max_length),
            dtype=torch.bool,
            device=smiles_tokens.device,
        )
        smiles_indices = torch.full(
            (batch_size, smiles_width),
            -1,
            dtype=torch.long,
            device=smiles_tokens.device,
        )
        atom_indices = torch.empty(
            atom_tokens.shape[0], dtype=torch.long, device=smiles_tokens.device
        )
        bond_indices = torch.empty(
            bond_tokens.shape[0], dtype=torch.long, device=smiles_tokens.device
        )
        descriptor_indices = torch.empty(
            batch_size, dtype=torch.long, device=smiles_tokens.device
        )

        cls_type = self.modality_embedding.weight[self.CLS_MODALITY]
        smiles_type = self.modality_embedding.weight[self.SMILES_MODALITY]
        graph_type = self.modality_embedding.weight[self.GRAPH_MODALITY]
        descriptor_type = self.modality_embedding.weight[self.DESCRIPTOR_MODALITY]
        for row, (
            (atom_start, atom_count),
            (bond_start, bond_count),
        ) in enumerate(zip(graph.atom_scopes, graph.bond_scopes, strict=True)):
            cursor = 0
            fused_inputs[row, cursor] = self.cls_token + cls_type
            cursor += 1

            smiles_count = int(smiles_lengths[row].item())
            smiles_slice = slice(cursor, cursor + smiles_count)
            fused_inputs[row, smiles_slice] = (
                smiles_tokens[row, :smiles_count] + smiles_type
            )
            smiles_indices[row, :smiles_count] = torch.arange(
                cursor, cursor + smiles_count, device=smiles_tokens.device
            )
            cursor += smiles_count

            atom_slice = slice(cursor, cursor + atom_count)
            fused_inputs[row, atom_slice] = (
                atom_tokens[atom_start : atom_start + atom_count] + graph_type
            )
            atom_indices[atom_start : atom_start + atom_count] = torch.arange(
                cursor, cursor + atom_count, device=smiles_tokens.device
            )
            cursor += atom_count

            bond_slice = slice(cursor, cursor + bond_count)
            fused_inputs[row, bond_slice] = (
                bond_tokens[bond_start : bond_start + bond_count] + graph_type
            )
            bond_indices[bond_start : bond_start + bond_count] = torch.arange(
                cursor, cursor + bond_count, device=smiles_tokens.device
            )
            cursor += bond_count

            descriptor_indices[row] = cursor
            fused_inputs[row, cursor] = descriptor_tokens[row] + descriptor_type
            cursor += 1
            padding_mask[row, :cursor] = False

        layout = FusionLayout(
            smiles_indices=smiles_indices,
            atom_indices=atom_indices,
            bond_indices=bond_indices,
            descriptor_indices=descriptor_indices,
            padding_mask=padding_mask,
            sequence_lengths=sequence_lengths,
        )
        return fused_inputs, layout

    def forward(
        self,
        smiles_tokens: torch.Tensor,
        smiles_padding_mask: torch.Tensor,
        atom_tokens: torch.Tensor,
        bond_tokens: torch.Tensor,
        descriptor_tokens: torch.Tensor,
        graph: PackedGraph,
    ) -> tuple[torch.Tensor, FusionLayout]:
        inputs, layout = self._assemble(
            smiles_tokens,
            smiles_padding_mask,
            atom_tokens,
            bond_tokens,
            descriptor_tokens,
            graph,
        )
        fused = self.encoder(inputs, src_key_padding_mask=layout.padding_mask)
        return fused, layout


def gather_smiles(
    fused: torch.Tensor,
    layout: FusionLayout,
) -> torch.Tensor:
    safe_indices = layout.smiles_indices.clamp_min(0)
    batch_indices = torch.arange(fused.shape[0], device=fused.device)[:, None]
    gathered = fused[batch_indices, safe_indices]
    return gathered.masked_fill((layout.smiles_indices < 0)[..., None], 0.0)


def gather_graph_tokens(
    fused: torch.Tensor,
    token_indices: torch.Tensor,
    batch_indices: torch.Tensor,
) -> torch.Tensor:
    if token_indices.numel() == 0:
        return fused.new_empty((0, fused.shape[-1]))
    return fused[batch_indices, token_indices]
