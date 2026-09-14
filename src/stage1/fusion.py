from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .data import BatchFusionLayout
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
        fingerprint_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.fingerprint_enabled = fingerprint_enabled
        self.modality_embedding = nn.Embedding(
            5 if fingerprint_enabled else 4, d_model
        )
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
        batch_layout: BatchFusionLayout,
    ) -> tuple[torch.Tensor, FusionLayout]:
        batch_size, smiles_width, d_model = smiles_tokens.shape
        descriptor_count = descriptor_tokens.shape[1]
        fingerprint_count = fingerprint_tokens.shape[1]
        if not self.fingerprint_enabled and fingerprint_count:
            raise ValueError("global_rdkit_v2 Fusion forbids fingerprint tokens")
        smiles_lengths = batch_layout.smiles_lengths
        atom_counts = batch_layout.atom_counts
        bond_counts = batch_layout.bond_counts
        sequence_lengths = (
            1
            + smiles_lengths
            + atom_counts
            + bond_counts
            + descriptor_count
            + fingerprint_count
        )
        max_length = batch_layout.max_core_length + descriptor_count + fingerprint_count
        fused_inputs = smiles_tokens.new_zeros((batch_size, max_length, d_model))
        positions = torch.arange(max_length, device=smiles_tokens.device)
        padding_mask = positions[None, :] >= sequence_lengths[:, None]
        modality = self.modality_embedding.weight
        role_types = (
            self.role_embedding(roles)
            if self.role_embedding is not None
            else smiles_tokens.new_zeros((batch_size, d_model))
        )
        rows = torch.arange(batch_size, device=smiles_tokens.device)
        fused_inputs[:, 0] = self.cls_token + modality[self.CLS_MODALITY] + role_types

        smiles_columns = torch.arange(smiles_width, device=smiles_tokens.device)
        smiles_valid = smiles_columns[None, :] < smiles_lengths[:, None]
        smiles_indices = torch.where(
            smiles_valid,
            1 + smiles_columns[None, :],
            smiles_columns.new_full((batch_size, smiles_width), -1),
        )
        smiles_targets = smiles_indices.clamp_min(0)
        smiles_values = (
            smiles_tokens
            + modality[self.SMILES_MODALITY]
            + role_types[:, None, :]
        ).masked_fill(~smiles_valid[..., None], 0.0)
        fused_inputs.scatter_add_(
            1,
            smiles_targets[..., None].expand(-1, -1, d_model),
            smiles_values,
        )
        fused_inputs[:, 0] = self.cls_token + modality[self.CLS_MODALITY] + role_types

        atom_rows = graph.atom_batch
        atom_indices = (
            1
            + smiles_lengths[atom_rows]
            + batch_layout.atom_local_indices
        )
        fused_inputs[atom_rows, atom_indices] = (
            atom_tokens + modality[self.GRAPH_MODALITY] + role_types[atom_rows]
        )

        bond_rows = graph.bond_batch
        bond_indices = (
            1
            + smiles_lengths[bond_rows]
            + atom_counts[bond_rows]
            + batch_layout.bond_local_indices
        )
        fused_inputs[bond_rows, bond_indices] = (
            bond_tokens + modality[self.GRAPH_MODALITY] + role_types[bond_rows]
        )

        descriptor_start = 1 + smiles_lengths + atom_counts + bond_counts
        descriptor_columns = torch.arange(
            descriptor_count, device=smiles_tokens.device
        )
        descriptor_indices = descriptor_start[:, None] + descriptor_columns[None, :]
        fused_inputs[rows[:, None], descriptor_indices] = (
            descriptor_tokens
            + modality[self.DESCRIPTOR_MODALITY]
            + role_types[:, None, :]
        )

        fingerprint_columns = torch.arange(
            fingerprint_count, device=smiles_tokens.device
        )
        fingerprint_indices = (
            descriptor_start[:, None]
            + descriptor_count
            + fingerprint_columns[None, :]
        )
        if fingerprint_count:
            fused_inputs[rows[:, None], fingerprint_indices] = (
                fingerprint_tokens
                + modality[self.FINGERPRINT_MODALITY]
                + role_types[:, None, :]
            )

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
        batch_layout: BatchFusionLayout,
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
            batch_layout,
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
