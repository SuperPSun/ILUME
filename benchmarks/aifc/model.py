from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .preprocessing import AIFCGraph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM


def _segment_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    output = values.new_zeros((size, *values.shape[1:]))
    if len(index):
        output.index_add_(0, index, values)
    return output


def _segment_softmax(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError("AIFC attention scores must have shape [items, 1]")
    if not len(index):
        return values
    maximum = values.new_full((size, 1), -torch.inf)
    maximum.scatter_reduce_(0, index[:, None], values, reduce="amax", include_self=True)
    numerator = torch.exp(values - maximum[index])
    denominator = _segment_sum(numerator, index, size)
    return numerator / denominator[index]


class AtomAFPLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.node_embedding = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU())
        self.edge_embedding = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LeakyReLU())
        self.alignment = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, 1), nn.LeakyReLU())
        self.attend = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self, nodes: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.node_embedding(nodes)
        if not edge_index.numel():
            context = torch.zeros_like(embedded)
        else:
            source, destination = edge_index
            message = self.edge_embedding(torch.cat((embedded[source], edges), dim=-1))
            score = self.alignment(torch.cat((message, embedded[destination]), dim=-1))
            weight = _segment_softmax(score, destination, len(nodes))
            context = _segment_sum(weight * self.attend(message), destination, len(nodes))
        return F.relu(self.gru(F.elu(context), embedded))


class AtomAttentiveFP(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(AtomAFPLayer(hidden_dim, dropout) for _ in range(depth))

    def forward(
        self, nodes: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            nodes = layer(nodes, edges, edge_index)
        return nodes


class MolAFPLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.alignment = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.LeakyReLU())
        self.attend = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))

    def forward(
        self, super_nodes: torch.Tensor, nodes: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        super_nodes = F.leaky_relu(super_nodes)
        score = self.alignment(torch.cat((nodes, super_nodes[batch]), dim=-1))
        attention = _segment_softmax(score, batch, len(super_nodes))
        context = F.elu(_segment_sum(attention * self.attend(nodes), batch, len(super_nodes)))
        return F.relu(self.gru(context, super_nodes)), attention


class MolAttentiveFP(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(MolAFPLayer(hidden_dim, dropout) for _ in range(layers))

    def forward(
        self, nodes: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = int(batch.max().item()) + 1
        super_nodes = _segment_sum(nodes, batch, size)
        attentions = []
        for layer in self.layers:
            super_nodes, attention = layer(super_nodes, nodes, batch)
            attentions.append(attention)
        return super_nodes, torch.stack(attentions, dim=1).mean(dim=1)


class AIFCHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, depth: int, layers: int) -> None:
        super().__init__()
        self.atom = AtomAttentiveFP(hidden_dim, dropout, depth)
        self.readout = MolAttentiveFP(hidden_dim, dropout, layers)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = self.atom(nodes, edges, edge_index)
        return self.readout(nodes, batch)


class JunctionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, depth: int, layers: int) -> None:
        super().__init__()
        self.project = nn.Linear(hidden_dim * 2, hidden_dim)
        self.atom = AtomAttentiveFP(hidden_dim, dropout, depth)
        self.readout = MolAttentiveFP(hidden_dim, dropout, layers)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = self.atom(self.project(nodes), edges, edge_index)
        return self.readout(nodes, batch)


class AIFCEncoder(nn.Module):
    def __init__(
        self,
        *,
        fragment_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        depth: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fragment_node_embedding = nn.Sequential(
            nn.Linear(ATOM_FEATURE_DIM, hidden_dim), nn.LeakyReLU()
        )
        self.fragment_edge_embedding = nn.Sequential(
            nn.Linear(BOND_FEATURE_DIM, hidden_dim), nn.LeakyReLU()
        )
        self.motif_node_embedding = nn.Sequential(
            nn.Linear(fragment_dim, hidden_dim), nn.LeakyReLU()
        )
        self.motif_edge_embedding = nn.Sequential(
            nn.Linear(BOND_FEATURE_DIM, hidden_dim), nn.LeakyReLU()
        )
        self.fragment_heads = nn.ModuleList(
            AIFCHead(hidden_dim, dropout, depth, layers) for _ in range(num_heads)
        )
        self.junction_heads = nn.ModuleList(
            JunctionHead(hidden_dim, dropout, depth, layers) for _ in range(num_heads)
        )
        self.fragment_attention = nn.Sequential(
            nn.Linear(num_heads * hidden_dim, hidden_dim), nn.ReLU()
        )
        self.motif_attention = nn.Sequential(
            nn.Linear(num_heads * hidden_dim, hidden_dim), nn.ReLU()
        )

    def forward(
        self, graph: AIFCGraph, *, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        fragment_nodes = self.fragment_node_embedding(graph.fragment_nodes.float())
        fragment_edges = self.fragment_edge_embedding(graph.fragment_edges.float())
        fragment_outputs = [
            head(
                fragment_nodes,
                fragment_edges,
                graph.fragment_edge_index,
                graph.fragment_batch,
            )[0]
            for head in self.fragment_heads
        ]
        fragment_representation = self.fragment_attention(torch.cat(fragment_outputs, dim=-1))
        motif_nodes = self.motif_node_embedding(graph.motif_nodes.float())
        motif_edges = self.motif_edge_embedding(graph.motif_edges.float())
        junction_outputs = [
            head(
                torch.cat((fragment_representation, motif_nodes), dim=-1),
                motif_edges,
                graph.motif_edge_index,
                graph.motif_batch,
            )
            for head in self.junction_heads
        ]
        representation = torch.stack([value[0] for value in junction_outputs], dim=1).mean(dim=1)
        attention = torch.stack([value[1] for value in junction_outputs], dim=1).mean(dim=1)
        if return_attention:
            return representation, attention
        return representation


class AIFCRegressor(nn.Module):
    def __init__(
        self,
        *,
        fragment_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        depth: int,
        layers: int,
        view_count: int,
        condition_dim: int,
    ) -> None:
        super().__init__()
        self.view_count = int(view_count)
        self.condition_dim = int(condition_dim)
        self.encoder = AIFCEncoder(
            fragment_dim=fragment_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            depth=depth,
            layers=layers,
        )
        self.predictor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(view_count * hidden_dim + condition_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, graph: AIFCGraph, conditions: torch.Tensor) -> torch.Tensor:
        representation = self.encoder(graph)
        batch_size = int(conditions.shape[0])
        if len(representation) != batch_size * self.view_count:
            raise ValueError("AIFC graph batch differs from registry view topology")
        ordered = representation.reshape(batch_size, self.view_count * representation.shape[-1])
        fused = F.relu(torch.cat((ordered, conditions.float()), dim=-1))
        return self.predictor(fused).reshape(-1)


__all__ = ["AIFCEncoder", "AIFCRegressor"]
