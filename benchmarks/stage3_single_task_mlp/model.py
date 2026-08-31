from __future__ import annotations

import torch
from torch import nn


class Stage3SingleTaskMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dims != (512, 256):
            raise ValueError("Stage3 Single-task MLP requires positive input and [512, 256]")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Stage3 Single-task MLP dropout must be in [0, 1)")
        layers: list[nn.Module] = []
        width = input_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.SiLU(), nn.Dropout(dropout)))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


__all__ = ["Stage3SingleTaskMLP"]

