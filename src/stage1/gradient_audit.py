"""Read-only, fixed-validation-probe encoder gradients for Stage1-v4."""
from __future__ import annotations

import random

import numpy as np
import torch

from common.identity import semantic_identity, tensor_state_hash
from .masking import MultimodalMasker, MultimodalPacker


AUDIT_LOSSES = {
    "smiles": "smiles", "atom": "atom", "bond": "bond",
    "alignment": "alignment", "rdkit": "descriptor",
    "unimol": "unimol", "electronic": "electronic",
}


def prepare_gradient_probe(dataset, vocabulary, config):
    sample_seed = config.data.seed + 400000
    mask_seed = config.data.seed + 400001
    indices = np.random.default_rng(sample_seed).choice(
        len(dataset), size=min(config.training.gradient_audit_batch_size, len(dataset)),
        replace=False,
    ).tolist()
    if not indices:
        raise ValueError("Gradient audit requires a nonempty validation split")
    batch = MultimodalPacker(vocabulary)(dataset.__getitems__(indices))
    batch = MultimodalMasker(vocabulary, config.masking, mask_seed).apply(batch, evaluation=True)
    metadata = {
        "probe_ids": list(batch.sample_ids), "probe_size": len(indices),
        "sample_seed": sample_seed, "mask_seed": mask_seed,
        "corpus_identity": dataset.metadata["semantic"]["identities"]["corpus"]["hash"],
        "feature_identity": dataset.metadata["semantic"]["identities"]["feature"]["hash"],
        "mask_hash": tensor_state_hash("stage1.gradient-audit.mask", {
            "token_ids": batch.token_ids, "smiles_labels": batch.masks.smiles_labels,
            "atom_mask": batch.masks.atom_mask, "bond_mask": batch.masks.bond_mask,
            "modality_dropped": batch.masks.modality_dropped,
        }),
    }
    metadata["probe_hash"] = semantic_identity("stage1.gradient-audit.probe", metadata)["hash"]
    return batch, metadata


def audit_gradient_norms(model, probe, config, device, *, amp_enabled, amp_dtype):
    """No backward/optimizer/collective; leave state, .grad and RNG untouched."""
    parameters = tuple(dict.fromkeys(
        p for encoder in (model.smiles_encoder, model.graph_encoder, model.fusion)
        for p in encoder.parameters() if p.requires_grad
    ))
    modes = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            model.eval()
            batch = probe.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                # Direct eager forward: bypass compiled call and the DDP wrapper.
                output = model.forward(batch)
            result = {"weighted_grad_norms": {}, "coefficients": {}, "coverage": {}}
            role_weights = torch.as_tensor(config.loss.role_weights, device=device)
            active = [name for name, loss in AUDIT_LOSSES.items()
                      if output.loss_statistics[loss].denominators.sum().item() > 0]
            for name, loss in AUDIT_LOSSES.items():
                stats = output.loss_statistics[loss]
                coefficient = getattr(config.loss, f"lambda_{loss}")
                valid_weight = stats.denominators.sum().item()
                result["coefficients"][name] = coefficient
                result["coverage"][name] = {
                    "valid_molecules": round((stats.role_denominators / role_weights).sum().item()),
                    "valid_role_weight_sum": valid_weight,
                    "status": "ok" if valid_weight > 0 else "no_valid_targets",
                }
                norm = None
                if name in active:
                    gradients = torch.autograd.grad(
                        output.losses[loss], parameters, allow_unused=True,
                        retain_graph=name != active[-1],
                    )
                    squared = torch.zeros((), dtype=torch.float32, device=device)
                    for gradient in gradients:
                        if gradient is not None:
                            squared += gradient.detach().float().square().sum()
                    norm = squared.sqrt().item()
                    del gradients, gradient
                    if not np.isfinite(norm):
                        raise RuntimeError(f"Non-finite gradient audit norm: {name}")
                result[f"{name}_grad_norm"] = norm
                result["weighted_grad_norms"][name] = None if norm is None else abs(coefficient) * norm
            return result
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
