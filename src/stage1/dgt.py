"""ILUME's 2D DGT adaptation, without PyG or graph-tool dependencies.

Adapted from zhangsy-ryan/DGT, commit 2f5a4c3ef2e376d60238f481c36a87eeb5a581e8.
Copyright (c) 2025 Shuyuan Zhang. MIT license; see docs/licenses/DGT-MIT.txt.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .dgt_topology import build_dgt_topology
from .encoders import _categorical_to_one_hot
from .graph import ATOM_CARDINALITIES, BOND_CARDINALITIES


class PairEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.distance = nn.Embedding(10, width)
        self.ring = nn.Embedding(3, width, padding_idx=0)
        self.edge_type = nn.Embedding(3, width, padding_idx=0)
        self.attention = nn.Linear(width, width)
        self.value = nn.Linear(width, width)

    def components(self, nodes, edges):
        # Distribute the shared linear map over the additive pair inputs. This
        # avoids allocating an N*N*width tensor or repeating matrix projections.
        result = []
        for layer in (self.attention, self.value):
            node_values = F.linear(nodes, layer.weight)
            result.append((node_values, F.linear(edges, layer.weight),
                           F.linear(self.distance.weight, layer.weight),
                           F.linear(self.ring.weight, layer.weight),
                           F.linear(self.edge_type.weight, layer.weight), layer.bias.to(node_values.dtype)))
        return tuple(result)


class DGTStream(nn.Module):
    def __init__(self, width, heads, feedforward, dropout):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(width, feedforward), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(feedforward, width), nn.Dropout(dropout))

    def attend(self, query, key, value, pair_attention, pair_value, valid):
        head_dim = query.shape[-1] // self.heads
        q = query.unflatten(-1, (self.heads, head_dim))
        k = key.unflatten(-1, (self.heads, head_dim))
        scores = torch.einsum("bqhd,bkhd->bqkh", q, k) * (head_dim ** -0.5)
        scores = scores.unsqueeze(-1) + pair_attention.unflatten(-1, (self.heads, head_dim))
        # An empty molecular bond graph gets a dummy key; its outputs are masked
        # away. No all-masked softmax and no communication between molecules.
        scores = scores.float().masked_fill(~valid[:, None, :, None, None], -torch.inf)
        weights = scores.softmax(2).flatten(-2).to(value.dtype)
        pair_dropout = self.attention_dropout(
            torch.ones((*weights.shape[:3], 1), device=value.device, dtype=value.dtype))
        weights = weights * pair_dropout
        return (weights * (value[:, None] + pair_value)).sum(2)

    def forward(self, hidden, mask, pair_components, pair_indices, chunk_size, recompute):
        if hidden.shape[1] == 0:
            return hidden + sum(p.sum() for p in self.parameters()) * 0
        query, key, value = self.query(hidden), self.key(hidden), self.value(hidden)
        safe_mask = mask.clone()
        safe_mask[:, 0] |= ~mask.any(1)
        distance, ring, edge_type, node_index, edge_index, adjacency = pair_indices

        def pair_chunk(parts, start, end):
            nodes, edges, distances, rings, types, bias = parts
            result = (distances[distance[:, start:end].long()]
                      + rings[ring[:, start:end].long()]
                      + types[edge_type[:, start:end].long()] + bias)
            if node_index is None:
                # Atom-pair edge features include both endpoints only for bonds.
                result = result + (nodes[:, start:end, None] + nodes[:, None]) * adjacency[:, start:end, :, None]
            else:
                padded = F.pad(nodes, (0, 0, 1, 0))
                batch = torch.arange(len(nodes), device=nodes.device)[:, None, None]
                result = result + padded[batch, node_index[:, start:end] + 1]
            padded_edges = F.pad(edges, (0, 0, 1, 0))
            batch = torch.arange(len(edges), device=edges.device)[:, None, None]
            return result + padded_edges[batch, edge_index[:, start:end] + 1]

        pieces = []
        for start in range(0, hidden.shape[1], chunk_size):
            end = min(start + chunk_size, hidden.shape[1])

            def compute(q, k, v, begin=start, finish=end):
                attention = pair_chunk(pair_components[0], begin, finish)
                values = pair_chunk(pair_components[1], begin, finish)
                return self.attend(q, k, v, attention, values, safe_mask)

            inputs = (query[:, start:end], key, value)
            if recompute and torch.is_grad_enabled():
                pieces.append(checkpoint(compute, *inputs, use_reentrant=False, preserve_rng_state=True))
            else:
                pieces.append(compute(*inputs))
        hidden = self.norm1(hidden + self.dropout(self.output(torch.cat(pieces, 1))))
        return self.norm2(hidden + self.ffn(hidden)) * mask[:, :, None]


class DualGraphBlock(nn.Module):
    def __init__(self, width, heads, feedforward, dropout):
        super().__init__()
        self.atom = DGTStream(width, heads, feedforward, dropout)
        self.bond = DGTStream(width, heads, feedforward, dropout)


class DualGraphTransformerEncoder(nn.Module):
    def __init__(self, width, depth, heads, feedforward, dropout, *, query_chunk_size=32, recompute=True):
        super().__init__()
        self.query_chunk_size = query_chunk_size
        self.recompute = recompute
        atom_dim, bond_dim = sum(ATOM_CARDINALITIES) + 1, sum(BOND_CARDINALITIES)
        self.atom_mask_feature = nn.Parameter(torch.zeros(atom_dim))
        self.bond_mask_feature = nn.Parameter(torch.zeros(bond_dim))
        nn.init.normal_(self.atom_mask_feature, std=.02)
        nn.init.normal_(self.bond_mask_feature, std=.02)
        self.atom_input = nn.Linear(atom_dim, width)
        self.bond_input = nn.Linear(bond_dim, width)
        self.atom_fusion = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
        self.bond_fusion = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
        self.atom_pairs = PairEncoder(width)
        self.bond_pairs = PairEncoder(width)
        self.blocks = nn.ModuleList(DualGraphBlock(width, heads, feedforward, dropout) for _ in range(depth))
        self.graph_readout = nn.Linear(2 * width, width)

    @staticmethod
    def _dense(values, batch, scopes):
        maximum = max(count for _, count in scopes)
        starts = torch.tensor([start for start, _ in scopes], device=values.device)
        counts = torch.tensor([count for _, count in scopes], device=values.device)
        positions = torch.arange(maximum, device=values.device)
        mask = positions[None] < counts[:, None]
        result = values.new_zeros((len(scopes), maximum, values.shape[-1]))
        local = torch.arange(len(values), device=values.device) - starts[batch]
        result[batch, local] = values
        return result, mask

    def _pair_indices(self, graph, topologies, atom_mask, bond_mask):
        batch_size = len(topologies)
        n, m = atom_mask.shape[1], bond_mask.shape[1]
        device = atom_mask.device
        atom_distance = torch.full((batch_size, n, n), 9, dtype=torch.uint8, device=device)
        atom_ring = torch.zeros_like(atom_distance)
        bond_distance = torch.full((batch_size, m, m), 9, dtype=torch.uint8, device=device)
        bond_ring = torch.zeros_like(bond_distance)
        atom_edges = torch.full((batch_size, n, n), -1, dtype=torch.long, device=device)
        bond_nodes = torch.full((batch_size, m, m), -1, dtype=torch.long, device=device)
        for batch, (topology, (atom_start, atom_count), (_, bond_count)) in enumerate(zip(
                topologies, graph.atom_scopes, graph.bond_scopes, strict=True)):
            for name, target, count in (("atom_spd", atom_distance, atom_count), ("atom_ring", atom_ring, atom_count),
                                        ("bond_spd", bond_distance, bond_count), ("bond_ring", bond_ring, bond_count)):
                target[batch, :count, :count] = topology[name].to(device)
            edges = graph.raw_bond_indices[batch].to(device)
            if bond_count:
                indices = torch.arange(bond_count, device=device)
                atom_edges[batch, edges[0], edges[1]] = indices
                atom_edges[batch, edges[1], edges[0]] = indices
            line = topology["line_index"].to(device=device, dtype=torch.long)
            bond_nodes[batch, line[0], line[1]] = topology["shared_atom"].to(device=device, dtype=torch.long)
        atom_adj = atom_edges >= 0
        bond_adj = bond_nodes >= 0
        atom_type = torch.where(atom_adj, 0, 2)
        bond_type = torch.where(bond_adj, 0, 2)
        atom_type.diagonal(dim1=1, dim2=2).fill_(1)
        bond_type.diagonal(dim1=1, dim2=2).fill_(1)
        return ((atom_distance, atom_ring, atom_type, None, atom_edges, atom_adj),
                (bond_distance, bond_ring, bond_type, bond_nodes,
                 torch.full_like(bond_nodes, -1), bond_adj))

    def forward(self, graph, atom_mask, bond_mask):
        atom_features = torch.cat((_categorical_to_one_hot(graph.atom_categorical, ATOM_CARDINALITIES),
                                   graph.atom_continuous), -1)
        bond_features = _categorical_to_one_hot(graph.bond_categorical, BOND_CARDINALITIES)
        atoms = self.atom_input(torch.where(atom_mask[:, None], self.atom_mask_feature, atom_features))
        bonds = self.bond_input(torch.where(bond_mask[:, None], self.bond_mask_feature, bond_features))
        incident = torch.zeros_like(atoms).index_add(0, graph.bond_index[0], bonds).index_add(0, graph.bond_index[1], bonds)
        atom_hidden = self.atom_fusion(torch.cat((atoms, incident), -1))
        bond_hidden = self.bond_fusion(torch.cat((bonds, atoms[graph.bond_index[0]] + atoms[graph.bond_index[1]]), -1))
        dense_atoms, atom_valid = self._dense(atom_hidden, graph.atom_batch, graph.atom_scopes)
        dense_bonds, bond_valid = self._dense(bond_hidden, graph.bond_batch, graph.bond_scopes)
        original_atoms, _ = self._dense(atoms, graph.atom_batch, graph.atom_scopes)
        original_bonds, _ = self._dense(bonds, graph.bond_batch, graph.bond_scopes)
        context, _ = self._dense(atoms + incident, graph.atom_batch, graph.atom_scopes)
        topologies = graph.dgt_topologies
        if not topologies or any(topology is None for topology in topologies):
            topologies = tuple(topology if topology is not None else build_dgt_topology(count, bonds)
                               for topology, (_, count), bonds in zip(
                                   topologies or (None,) * len(graph.atom_scopes), graph.atom_scopes,
                                   graph.raw_bond_indices, strict=True))
        atom_indices, bond_indices = self._pair_indices(graph, topologies, atom_valid, bond_valid)
        atom_components = self.atom_pairs.components(original_atoms, original_bonds)
        bond_components = self.bond_pairs.components(context, original_bonds[:, :0])
        for block in self.blocks:
            dense_atoms = block.atom(dense_atoms, atom_valid, atom_components, atom_indices,
                                     self.query_chunk_size, self.recompute)
            dense_bonds = block.bond(dense_bonds, bond_valid, bond_components, bond_indices,
                                     self.query_chunk_size, self.recompute)
        # Keep every branch in DDP's graph, including batches without bonds and
        # molecular-only probes whose objective does not use the graph readout.
        zero = sum(p.float().sum() for p in self.parameters()) * 0
        return dense_atoms[atom_valid] + zero, dense_bonds[bond_valid] + zero

    def pool(self, atoms, bonds, graph):
        size = len(graph.atom_scopes)
        atom_mean = atoms.new_zeros((size, atoms.shape[-1])).index_add(0, graph.atom_batch, atoms)
        atom_mean = atom_mean / torch.bincount(graph.atom_batch, minlength=size).clamp_min(1)[:, None]
        bond_mean = bonds.new_zeros((size, atoms.shape[-1])).index_add(0, graph.bond_batch, bonds)
        bond_mean = bond_mean / torch.bincount(graph.bond_batch, minlength=size).clamp_min(1)[:, None]
        return self.graph_readout(torch.cat((atom_mean, bond_mean), -1))
