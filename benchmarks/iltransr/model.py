from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


EMBED_SIZE = 128
FFN_SIZE = 1024
NUM_LAYERS = 3
NUM_HEADS = 4
MAX_LENGTH = 100

TEXTCNN_SPECS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "smiles_only": (
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (100, 200, 200, 200, 200, 100, 100, 100, 100, 100),
    ),
    "temperature": (
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20),
        (100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160),
    ),
    "temperature_pressure": (
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15),
        (100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160),
    ),
}


class ILTransREncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.key = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.value = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.projection = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.attention_norm = nn.LayerNorm(EMBED_SIZE, eps=1.0e-5)
        self.ffn_1 = nn.Linear(EMBED_SIZE, FFN_SIZE)
        self.ffn_2 = nn.Linear(FFN_SIZE, EMBED_SIZE)
        self.ffn_norm = nn.LayerNorm(EMBED_SIZE, eps=1.0e-5)

    def forward(self, inputs: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = inputs.shape
        head_dim = EMBED_SIZE // NUM_HEADS

        def split_heads(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch, length, NUM_HEADS, head_dim).transpose(1, 2)

        query = split_heads(self.query(inputs)) / math.sqrt(head_dim)
        key = split_heads(self.key(inputs))
        value = split_heads(self.value(inputs))
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores.masked_fill(~valid_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attended = torch.matmul(torch.softmax(scores, dim=-1), value)
        attended = attended.transpose(1, 2).reshape(batch, length, EMBED_SIZE)
        hidden = self.attention_norm(inputs + self.projection(attended))
        output = self.ffn_2(F.relu(self.ffn_1(hidden)))
        return self.ffn_norm(hidden + output)


class ILTransRTransformer(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_SIZE)
        self.input_norm = nn.LayerNorm(EMBED_SIZE, eps=1.0e-5)
        self.layers = nn.ModuleList(ILTransREncoderLayer() for _ in range(NUM_LAYERS))
        position = torch.arange(MAX_LENGTH, dtype=torch.float32).reshape(-1, 1)
        divisor = torch.pow(
            torch.tensor(10000.0),
            (2.0 / EMBED_SIZE) * torch.arange(EMBED_SIZE, dtype=torch.float32).reshape(1, -1),
        )
        encoding = position / divisor
        encoding[:, 0::2] = torch.sin(encoding[:, 0::2])
        encoding[:, 1::2] = torch.cos(encoding[:, 1::2])
        self.register_buffer("position_weight", encoding, persistent=True)

    def forward(self, token_ids: torch.Tensor, valid_lengths: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] > MAX_LENGTH:
            raise ValueError("ILTransR token tensor exceeds the 100-token context")
        length = token_ids.shape[1]
        steps = torch.arange(length, device=token_ids.device)
        valid_mask = steps.unsqueeze(0) < valid_lengths.long().unsqueeze(1)
        hidden = self.embedding(token_ids.long()) * math.sqrt(EMBED_SIZE)
        hidden = self.input_norm(hidden + self.position_weight[:length].to(hidden.dtype))
        for layer in self.layers:
            hidden = layer(hidden, valid_mask)
        return hidden.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


class Highway(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width * 2)
        nn.init.xavier_uniform_(self.projection.weight)
        with torch.no_grad():
            self.projection.bias[:width].zero_()
            self.projection.bias[width:].fill_(-2.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        nonlinear, gate = self.projection(inputs).chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        return (1.0 - gate) * inputs + gate * F.relu(nonlinear)


class ILTransRTextCNN(nn.Module):
    def __init__(self, topology: str) -> None:
        super().__init__()
        kernels, filters = TEXTCNN_SPECS[topology]
        self.kernels = kernels
        self.output_dim = sum(filters)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(EMBED_SIZE, channels, kernel_size=kernel)
            for kernel, channels in zip(kernels, filters, strict=True)
        )
        self.highway = Highway(self.output_dim)
        for convolution in self.convolutions:
            nn.init.xavier_uniform_(convolution.weight)
            nn.init.zeros_(convolution.bias)

    @property
    def maximum_kernel(self) -> int:
        return max(self.kernels)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        values = encoded.transpose(1, 2)
        pooled = [F.relu(convolution(values).amax(dim=-1)) for convolution in self.convolutions]
        return self.highway(torch.cat(pooled, dim=-1))


class ILTransRRegressor(nn.Module):
    def __init__(
        self,
        transformer: ILTransRTransformer,
        *,
        topology: str,
        view_count: int,
        condition_dim: int,
        dropout: float,
        load_audit: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.textcnn = ILTransRTextCNN(topology)
        self.topology = topology
        self.view_count = int(view_count)
        self.condition_dim = int(condition_dim)
        self.load_audit = dict(load_audit)
        input_dim = self.view_count * self.textcnn.output_dim + self.condition_dim
        if topology == "smiles_only":
            self.predictor = nn.Sequential(nn.Linear(input_dim, 512), nn.Linear(512, 1))
        else:
            self.predictor = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
            )
        for module in self.predictor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        valid_lengths: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(conditions.shape[0])
        if int(token_ids.shape[0]) != batch_size * self.view_count:
            raise ValueError("ILTransR merged view batch differs from registry topology")
        encoded = self.transformer(token_ids, valid_lengths)
        pooled = self.textcnn(encoded)
        ordered = pooled.reshape(self.view_count, batch_size, -1).transpose(0, 1)
        representation = torch.cat(
            (ordered.reshape(batch_size, -1), conditions.float()), dim=-1
        )
        return self.predictor(representation).reshape(-1)


def load_converted_transformer(
    path: str,
    *,
    vocab_size: int,
) -> tuple[ILTransRTransformer, dict[str, Any]]:
    from safetensors.torch import load_file

    converted = load_file(path, device="cpu")
    model = ILTransRTransformer(vocab_size)
    expected = set(model.state_dict())
    actual = set(converted)
    if expected != actual:
        raise ValueError(
            "ILTransR converted checkpoint structure mismatch: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    model.load_state_dict(converted, strict=True)
    return model, {
        "loaded_tensors": len(actual),
        "loaded_tensor_names": sorted(actual),
        "strict": True,
    }


__all__ = [
    "EMBED_SIZE",
    "MAX_LENGTH",
    "TEXTCNN_SPECS",
    "ILTransRRegressor",
    "ILTransRTextCNN",
    "ILTransRTransformer",
    "load_converted_transformer",
]
