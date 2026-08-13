from __future__ import annotations

import os

import torch

from .stage3_config import Stage3Config


def configure_stage3_runtime(config: Stage3Config) -> None:
    """Apply Stage 3 CPU thread limits before any parallel tensor work."""

    threads = config.training.cpu_threads
    interop_threads = config.training.cpu_interop_threads
    value = str(threads)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        os.environ[name] = value
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != interop_threads:
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError as error:
            raise RuntimeError(
                "Stage 3 inter-op threads must be configured before "
                "parallel PyTorch work starts"
            ) from error
