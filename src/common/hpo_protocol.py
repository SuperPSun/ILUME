"""Explicit research protocol shared by the Stage2/3 configuration contracts."""
from __future__ import annotations

import math
from dataclasses import asdict

HPO_PROTOCOL = "hpo_v4"
SCOPES = ("GLOBAL", "GROUP", "PRIVATE")


def recipe_dict(recipe):
    """Omit optional additions so legacy recipe identities remain unchanged."""
    return {key: value for key, value in asdict(recipe).items()
            if key not in {"global_dropout", "dropout", "private_hidden_ratio"} or value is not None}


def validate_owner_training(values):
    if set(values) != set(SCOPES):
        raise ValueError("HPO owner_training requires GLOBAL/GROUP/PRIVATE")
    for scope, recipe in values.items():
        if set(recipe) != {"lr", "epochs", "weight_decay"}:
            raise ValueError(f"Invalid HPO owner training fields: {scope}")
        if type(recipe["epochs"]) is not int or recipe["epochs"] < 1:
            raise ValueError("HPO owner epochs must be positive integers")
        for key in ("lr", "weight_decay"):
            value = recipe[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid HPO owner {key}")
        if recipe["lr"] <= 0 or recipe["weight_decay"] < 0:
            raise ValueError("Invalid HPO owner LR/weight decay")


def validate_protocol(protocol):
    if protocol not in (None, HPO_PROTOCOL):
        raise ValueError("Unknown research protocol")


def validate_hpo_paths(config_path, output):
    from pathlib import Path
    for value, prefix in ((config_path, ("configs", "HPO_v4")), (output, ("outputs", "HPO_v4"))):
        path = Path(value)
        if path.is_absolute() or path.parts[:2] != prefix or ".." in path.parts:
            raise ValueError("HPO config/output must remain in configs/HPO_v4 and outputs/HPO_v4")
