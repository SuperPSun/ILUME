from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .graph import PackedGraph


@dataclass(frozen=True)
class FusionLayout:
    smiles_indices: torch.Tensor
    atom_indices: torch.Tensor
    bond_indices: torch.Tensor
    descriptor_indices: torch.Tensor
    fingerprint_indices: torch.Tensor
    padding_mask: torch.Tensor
    sequence_lengths: torch.Tensor


class FusionTransformer(nn.Module):
    CLS_MODALITY = 0
    SMILES_MODALITY = 1
    GRAPH_MODALITY = 2
    DESCRIPTOR_MODALITY = 3
    FINGERPRINT_MODALITY = 4

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
        role_embedding: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.modality_embedding = nn.Embedding(5, d_model)
        self.role_embedding = nn.Embedding(3, d_model) if role_embedding else None
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

    def _assemble(
        self,
        smiles_tokens: torch.Tensor,
        smiles_padding_mask: torch.Tensor,
        atom_tokens: torch.Tensor,
        bond_tokens: torch.Tensor,
        descriptor_tokens: torch.Tensor,
        fingerprint_tokens: torch.Tensor,
        graph: PackedGraph,
        roles: torch.Tensor,
    ) -> tuple[torch.Tensor, FusionLayout]:
        batch_size, smiles_width, d_model = smiles_tokens.shape
        descriptor_count = descriptor_tokens.shape[1]
        fingerprint_count = fingerprint_tokens.shape[1]
        smiles_lengths = (~smiles_padding_mask).sum(dim=1)
        sequence_lengths = torch.empty(
            batch_size, dtype=torch.long, device=smiles_tokens.device
        )
        for row, ((_, atom_count), (_, bond_count)) in enumerate(
            zip(graph.atom_scopes, graph.bond_scopes, strict=True)
        ):
            sequence_lengths[row] = (
                1
                + smiles_lengths[row]
                + atom_count
                + bond_count
                + descriptor_count
                + fingerprint_count
            )
        max_length = int(sequence_lengths.max().item())
        fused_inputs = smiles_tokens.new_zeros((batch_size, max_length, d_model))
        padding_mask = torch.ones(
            (batch_size, max_length), dtype=torch.bool, device=smiles_tokens.device
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
            (batch_size, descriptor_count),
            dtype=torch.long,
            device=smiles_tokens.device,
        )
        fingerprint_indices = torch.empty(
            (batch_size, fingerprint_count),
            dtype=torch.long,
            device=smiles_tokens.device,
        )

        modality = self.modality_embedding.weight
        for row, ((atom_start, atom_count), (bond_start, bond_count)) in enumerate(
            zip(graph.atom_scopes, graph.bond_scopes, strict=True)
        ):
            role_type = (
                self.role_embedding(roles[row])
                if self.role_embedding is not None
                else smiles_tokens.new_zeros(d_model)
            )
            cursor = 0
            fused_inputs[row, cursor] = (
                self.cls_token + modality[self.CLS_MODALITY] + role_type
            )
            cursor += 1

            smiles_count = int(smiles_lengths[row].item())
            smiles_slice = slice(cursor, cursor + smiles_count)
            fused_inputs[row, smiles_slice] = (
                smiles_tokens[row, :smiles_count]
                + modality[self.SMILES_MODALITY]
                + role_type
            )
            smiles_indices[row, :smiles_count] = torch.arange(
                cursor, cursor + smiles_count, device=smiles_tokens.device
            )
            cursor += smiles_count

            atom_slice = slice(cursor, cursor + atom_count)
            fused_inputs[row, atom_slice] = (
                atom_tokens[atom_start : atom_start + atom_count]
                + modality[self.GRAPH_MODALITY]
                + role_type
            )
            atom_indices[atom_start : atom_start + atom_count] = torch.arange(
                cursor, cursor + atom_count, device=smiles_tokens.device
            )
            cursor += atom_count

            bond_slice = slice(cursor, cursor + bond_count)
            fused_inputs[row, bond_slice] = (
                bond_tokens[bond_start : bond_start + bond_count]
                + modality[self.GRAPH_MODALITY]
                + role_type
            )
            bond_indices[bond_start : bond_start + bond_count] = torch.arange(
                cursor, cursor + bond_count, device=smiles_tokens.device
            )
            cursor += bond_count

            descriptor_slice = slice(cursor, cursor + descriptor_count)
            fused_inputs[row, descriptor_slice] = (
                descriptor_tokens[row]
                + modality[self.DESCRIPTOR_MODALITY]
                + role_type
            )
            descriptor_indices[row] = torch.arange(
                cursor, cursor + descriptor_count, device=smiles_tokens.device
            )
            cursor += descriptor_count

            fingerprint_slice = slice(cursor, cursor + fingerprint_count)
            if fingerprint_count:
                fused_inputs[row, fingerprint_slice] = (
                    fingerprint_tokens[row]
                    + modality[self.FINGERPRINT_MODALITY]
                    + role_type
                )
                fingerprint_indices[row] = torch.arange(
                    cursor, cursor + fingerprint_count, device=smiles_tokens.device
                )
            cursor += fingerprint_count
            padding_mask[row, :cursor] = False

        return fused_inputs, FusionLayout(
            smiles_indices=smiles_indices,
            atom_indices=atom_indices,
            bond_indices=bond_indices,
            descriptor_indices=descriptor_indices,
            fingerprint_indices=fingerprint_indices,
            padding_mask=padding_mask,
            sequence_lengths=sequence_lengths,
        )

    def _encode(self, inputs: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if not (self.gradient_checkpointing and self.training):
            return self.encoder(inputs, src_key_padding_mask=padding_mask)
        hidden = inputs
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

    def forward(
        self,
        smiles_tokens: torch.Tensor,
        smiles_padding_mask: torch.Tensor,
        atom_tokens: torch.Tensor,
        bond_tokens: torch.Tensor,
        descriptor_tokens: torch.Tensor,
        fingerprint_tokens: torch.Tensor,
        graph: PackedGraph,
        roles: torch.Tensor,
    ) -> tuple[torch.Tensor, FusionLayout]:
        inputs, layout = self._assemble(
            smiles_tokens,
            smiles_padding_mask,
            atom_tokens,
            bond_tokens,
            descriptor_tokens,
            fingerprint_tokens,
            graph,
            roles,
        )
        return self._encode(inputs, layout.padding_mask), layout


def gather_smiles(fused: torch.Tensor, layout: FusionLayout) -> torch.Tensor:
    safe_indices = layout.smiles_indices.clamp_min(0)
    batch_indices = torch.arange(fused.shape[0], device=fused.device)[:, None]
    gathered = fused[batch_indices, safe_indices]
    return gathered.masked_fill((layout.smiles_indices < 0)[..., None], 0.0)


def gather_group_tokens(fused: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.shape[1] == 0:
        return fused.new_empty((fused.shape[0], 0, fused.shape[-1]))
    batch_indices = torch.arange(fused.shape[0], device=fused.device)[:, None]
    return fused[batch_indices, indices]


def gather_graph_tokens(
    fused: torch.Tensor,
    token_indices: torch.Tensor,
    batch_indices: torch.Tensor,
) -> torch.Tensor:
    if token_indices.numel() == 0:
        return fused.new_empty((0, fused.shape[-1]))
    return fused[batch_indices, token_indices]
