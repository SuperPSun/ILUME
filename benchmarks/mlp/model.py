from __future__ import annotations

import torch
from torch import nn


class DescriptorMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or not hidden_dims:
            raise ValueError("MLP dimensions must be positive")
        layers: list[nn.Module] = []
        width = input_dim
        for hidden in hidden_dims:
            if hidden <= 0:
                raise ValueError("MLP hidden dimensions must be positive")
            layers.extend((nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)))
            width = hidden
        layers.append(nn.Linear(width, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


__all__ = ["DescriptorMLP"]

