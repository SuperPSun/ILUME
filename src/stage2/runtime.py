from __future__ import annotations

from typing import Any

import torch


def configure_stage2_math(device: torch.device) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "device_type": device.type,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "teacher_dtype": "float32",
        "fp32_matmul_precision": "ieee",
        "cudnn_conv_fp32_precision": "ieee",
        "cuda_capability": None,
    }
    if device.type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        contract.update(
            {
                "fp32_matmul_precision": "tf32",
                "cudnn_conv_fp32_precision": "tf32",
                "cuda_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return contract
