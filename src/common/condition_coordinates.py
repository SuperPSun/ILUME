"""Physical condition coordinates shared by simulation and experimental HoME."""
from collections.abc import Mapping
import math

import torch

CONDITION_SCALES = {
    "temperature_K": 1000.0,
    "pressure_kPa": 1000.0,
    "frequency_MHz": 1000.0,
    "wavelength_nm": 1000.0,
}
CONDITION_CONTRACT = {"kind": "physical_div1000_v1", "scales": CONDITION_SCALES}


def physical_conditions(values, columns, statistics):
    """Undo the prepared train-only z-score, then apply fixed physical scales."""
    if values.ndim != 2 or values.shape[1] != len(columns) or not torch.isfinite(values).all():
        raise ValueError("Invalid normalized HoME conditions")
    if len(set(columns)) != len(columns) or set(columns) - set(CONDITION_SCALES):
        raise ValueError("Unsupported or duplicate physical condition columns")
    if not isinstance(statistics, Mapping) or set(statistics) != set(columns):
        raise ValueError("HoME condition statistics differ from declared columns")
    result = values.new_zeros((len(values), len(CONDITION_SCALES)), dtype=torch.float32)
    mask = torch.zeros_like(result, dtype=torch.bool)
    names = tuple(CONDITION_SCALES)
    for j, name in enumerate(columns):
        stats = statistics[name]
        mean, scale = float(stats["mean"]), float(stats["scale"])
        if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid HoME condition normalization statistics")
        i = names.index(name)
        result[:, i] = (values[:, j].float() * scale + mean) / CONDITION_SCALES[name]
        mask[:, i] = True
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite physical HoME conditions")
    return result, mask
