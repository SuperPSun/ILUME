from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .model import AIFCRegressor
from .preprocessing import FragmentScheme, smiles_to_aifc_graph


def _fill_parameter(name: str, parameter: torch.Tensor) -> torch.Tensor:
    offset = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 101 - 50
    values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
    return ((values % 23) - 11 + offset / 50) * 0.003


def validate_legacy_parity(
    reference_path: str | Path, fragment_path: str | Path
) -> dict[str, Any]:
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    if reference.get("format_version") != 1:
        raise ValueError("Unsupported AIFC legacy parity reference")
    model = AIFCRegressor(fragment_dim=100, **reference["model"])
    with torch.no_grad():
        for name, parameter in model.state_dict().items():
            parameter.copy_(_fill_parameter(name, parameter))
    graph = smiles_to_aifc_graph(
        reference["smiles"], FragmentScheme.load(fragment_path)
    )
    model.eval()
    with torch.inference_mode():
        representation, attention = model.encoder(graph, return_attention=True)
        prediction = model(graph, torch.empty((1, 0)))
    expected_representation = torch.tensor(reference["representation"])
    expected_attention = torch.tensor(reference["attention"])
    errors = {
        "prediction": float(abs(prediction.item() - float(reference["prediction"]))),
        "representation": float(
            (representation.reshape(-1) - expected_representation).abs().max()
        ),
        "attention": float((attention.reshape(-1) - expected_attention).abs().max()),
    }
    threshold = 1.0e-5
    if max(errors.values()) > threshold:
        raise RuntimeError(f"AIFC legacy DGL parity failed: {errors}")
    return {"max_abs_errors": errors, "threshold": threshold}


__all__ = ["validate_legacy_parity"]
