"""Configuration-driven scalar predictors for frozen Stage1 representations."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class PredictorConfig:
    type: str = "linear"
    hidden_dims: tuple[int, ...] = ()
    activation: str = "gelu"
    dropout: float = 0.

    def validate(self):
        if not isinstance(self.type, str) or self.type not in {"linear", "mlp", "residual_mlp"}:
            raise ValueError("Unknown predictor type")
        if not isinstance(self.activation, str) or self.activation not in {"gelu", "relu", "silu"}:
            raise ValueError("Unknown predictor activation")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("predictor.dropout must be finite and in [0,1)")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in self.hidden_dims):
            raise ValueError("predictor.hidden_dims must contain positive integers")
        if self.type == "linear":
            if self.hidden_dims or self.dropout != 0 or self.activation != "gelu":
                raise ValueError("Linear predictor does not accept hidden layers, activation or dropout")
        elif not self.hidden_dims:
            raise ValueError("Nonlinear predictor requires at least one hidden layer")

    def to_dict(self):
        if self.type == "linear":
            return {"type": "linear"}
        return {"type": self.type, "hidden_dims": list(self.hidden_dims),
                "activation": self.activation, "dropout": self.dropout}


def predictor_from_dict(raw):
    if not isinstance(raw, dict) or set(raw) - set(PredictorConfig.__dataclass_fields__):
        raise ValueError("Unknown predictor configuration fields")
    raw = dict(raw)
    if "hidden_dims" in raw:
        if not isinstance(raw["hidden_dims"], (list, tuple)):
            raise ValueError("predictor.hidden_dims must be a list of positive integers")
        raw["hidden_dims"] = tuple(raw["hidden_dims"])
    recipe = PredictorConfig(**raw)
    recipe.validate()
    return recipe


class ResidualPredictorBlock(nn.Module):
    def __init__(self, input_dim, width, activation, dropout):
        super().__init__()
        self.shortcut = nn.Identity() if input_dim == width else nn.Linear(input_dim, width)
        self.residual = nn.Sequential(nn.Linear(input_dim, width), activation(), nn.Dropout(dropout),
                                      nn.Linear(width, width), nn.Dropout(dropout))

    def forward(self, inputs):
        return self.shortcut(inputs) + self.residual(inputs)


def build_predictor(recipe, input_dim):
    recipe.validate()
    if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim < 1:
        raise ValueError("Predictor input dimension must be a positive integer")
    if recipe.type == "linear":
        return nn.Linear(input_dim, 1)
    activation = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[recipe.activation]
    layers = []
    for width in recipe.hidden_dims:
        if recipe.type == "mlp":
            layers.extend([nn.Linear(input_dim, width), activation(), nn.Dropout(recipe.dropout)])
        else:
            layers.append(ResidualPredictorBlock(input_dim, width, activation, recipe.dropout))
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, 1))


@contextmanager
def predictor_rng(recipe, seed, device):
    """Separate nonlinear initialization/dropout streams; leave Linear RNG unchanged."""
    if recipe.type == "linear":
        yield
        return
    devices = []
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        devices = [index]
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.default_generators[index].manual_seed(seed)
        yield
