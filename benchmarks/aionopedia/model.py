"""AIonopedia downstream architecture with ILUME condition extensions.

The released architecture is adapted from AIonopedia/AIonopedia-public commit
17e2f550f91eadcdec39f467c0443f5446d9713c (MIT license).  Pressure,
frequency, and wavelength projectors and segment embeddings are deliberately
new, randomly initialized ILUME modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import TransformerConv


class GraphTransformerRegressor(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.conv1 = TransformerConv(35, hidden_dim, heads=4, edge_dim=11, dropout=0.1)
        self.conv2 = TransformerConv(hidden_dim * 4, hidden_dim, heads=4, edge_dim=11, dropout=0.1)
        self.conv3 = TransformerConv(hidden_dim * 4, hidden_dim, heads=4, edge_dim=11, dropout=0.1)
        self.conv4 = TransformerConv(hidden_dim * 4, hidden_dim, heads=1, edge_dim=11, concat=False, dropout=0.1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = self.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        x = self.relu(self.conv3(x, edge_index, edge_attr=edge_attr))
        return self.relu(self.conv4(x, edge_index, edge_attr=edge_attr))


def _condition_projector() -> nn.Sequential:
    return nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 256))


class MultiModalRegressor(nn.Module):
    def __init__(self, llm: nn.Module, *, llm_dim: int = 1024) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1,
            activation="relu", batch_first=True,
        )
        self.llm = llm
        self.GNN = GraphTransformerRegressor()
        self.projector_llm = nn.Sequential(
            nn.Linear(llm_dim, llm_dim), nn.ReLU(), nn.Linear(llm_dim, 256)
        )
        self.projector_gnn = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256)
        )
        self.projector_temp = _condition_projector()
        self.embedding_property = nn.Embedding(7, 256)
        self.graph_merge_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.segment_embed_temp = nn.Parameter(torch.randn(1, 256))
        self.segment_embed_solute = nn.Parameter(torch.randn(1, 256))
        self.segment_embed_cation = nn.Parameter(torch.randn(1, 256))
        self.segment_embed_anion = nn.Parameter(torch.randn(1, 256))
        self.segment_embed_property = nn.Parameter(torch.randn(1, 256))
        self.fc_out = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 1))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1,
            activation="relu", batch_first=True,
        )
        self.decoder1 = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.decoder2 = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.condition_projectors = nn.ModuleDict(
            {name: _condition_projector() for name in ("pressure", "frequency", "wavelength")}
        )
        self.condition_segments = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(1, 256)) for name in ("pressure", "frequency", "wavelength")}
        )

    def _graph_nodes(self, graph: Any) -> list[torch.Tensor]:
        encoded = self.projector_gnn(
            self.GNN(graph.x, graph.edge_index, graph.edge_attr)
        )
        count = int(graph.num_graphs)
        return [encoded[graph.batch == index] for index in range(count)]

    def encode_graphs(
        self,
        *,
        solute_graph: Any,
        cation_graph: Any,
        anion_graph: Any,
        temperature: torch.Tensor,
        topology: torch.Tensor,
        extra_conditions: Mapping[str, torch.Tensor],
        active_conditions: tuple[str, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        solute = self._graph_nodes(solute_graph)
        cation = self._graph_nodes(cation_graph)
        anion = self._graph_nodes(anion_graph)
        sequences = []
        for index in range(len(cation)):
            tokens = [
                self.projector_temp(temperature[index].reshape(1, 1))
                + self.segment_embed_temp
            ]
            for name in active_conditions:
                tokens.append(
                    self.condition_projectors[name](
                        extra_conditions[name][index].reshape(1, 1)
                    )
                    + self.condition_segments[name]
                )
            tokens.extend(
                [
                    solute[index] + self.segment_embed_solute,
                    cation[index] + self.segment_embed_cation,
                    anion[index] + self.segment_embed_anion,
                    self.embedding_property(topology[index].reshape(1))
                    + self.segment_embed_property,
                ]
            )
            sequences.append(torch.cat(tokens, dim=0))
        lengths = [len(sequence) for sequence in sequences]
        merged = pad_sequence(sequences, batch_first=True, padding_value=0)
        padding = torch.arange(merged.shape[1], device=merged.device)[None, :] >= torch.tensor(
            lengths, device=merged.device
        )[:, None]
        return self.graph_merge_encoder(merged, src_key_padding_mask=padding), padding

    def forward(
        self,
        *,
        solute_graph: Any,
        cation_graph: Any,
        anion_graph: Any,
        temperature: torch.Tensor,
        topology: torch.Tensor,
        extra_conditions: Mapping[str, torch.Tensor],
        active_conditions: tuple[str, ...],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        device = input_ids.device
        solute_graph = solute_graph.to(device, non_blocking=True)
        cation_graph = cation_graph.to(device, non_blocking=True)
        anion_graph = anion_graph.to(device, non_blocking=True)
        graph, graph_padding = self.encode_graphs(
            solute_graph=solute_graph,
            cation_graph=cation_graph,
            anion_graph=anion_graph,
            temperature=temperature,
            topology=topology,
            extra_conditions=extra_conditions,
            active_conditions=active_conditions,
        )
        output = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        text = self.projector_llm(output.hidden_states[-1])
        text_padding = attention_mask == 0
        causal = torch.triu(
            torch.ones(text.shape[1], text.shape[1], dtype=torch.bool, device=device),
            diagonal=1,
        )
        decoded_text = self.decoder1(
            tgt=text, memory=graph, tgt_mask=causal,
            tgt_key_padding_mask=text_padding,
            memory_key_padding_mask=graph_padding,
        )
        decoded_graph = self.decoder2(
            tgt=graph, memory=text,
            tgt_key_padding_mask=graph_padding,
            memory_key_padding_mask=text_padding,
        )
        batch = torch.arange(input_ids.shape[0], device=device)
        text_last = attention_mask.sum(dim=1) - 1
        graph_last = (~graph_padding).sum(dim=1) - 1
        return self.fc_out(
            torch.cat((decoded_text[batch, text_last], decoded_graph[batch, graph_last]), dim=1)
        )


__all__ = ["GraphTransformerRegressor", "MultiModalRegressor"]
