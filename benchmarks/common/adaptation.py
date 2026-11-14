"""Frozen-pretrained parameter boundaries shared by the six Headft adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import torch

from common.identity import tensor_state_hash
from common.io import atomic_json, sha256_file
from .config import HEADFT_PROTOCOL


TASK_MODULES = {
    "molformer": ("fusion", "classifier"),
    "ilbert": ("textcnn", "predictor"),
    "spmm": ("predictor",),
    "llasmol": ("predictor",),
    "iltransr": ("textcnn", "predictor"),
    "aionopedia": ("fc_out",),
}
CONDITION_NAMES = {"pressure_kPa": "pressure", "frequency_MHz": "frequency",
                   "wavelength_nm": "wavelength", "x_water_unitless": "water_activity"}


def task_modules(name: str, conditions: tuple[str, ...] = ()) -> tuple[str, ...]:
    modules = TASK_MODULES[name]
    if name == "aionopedia":
        modules += tuple(f"{parent}.{CONDITION_NAMES[c]}" for c in conditions if c in CONDITION_NAMES
                         for parent in ("condition_projectors", "condition_segments"))
    return modules


def _inside(name: str, modules: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in modules)


def freeze_pretrained(config, model: torch.nn.Module, conditions: tuple[str, ...] = ()) -> torch.nn.Module:
    if config.protocol != HEADFT_PROTOCOL:
        return model
    modules = task_modules(config.name, conditions)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_inside(name, modules))
    if config.name == "llasmol":
        model.load_audit["trainable_lora_parameters"] = 0
    model.eval()
    return model


def training_mode(config, model: torch.nn.Module, conditions: tuple[str, ...] = ()) -> None:
    if config.protocol != HEADFT_PROTOCOL:
        model.train()
        return
    model.eval()
    for name in task_modules(config.name, conditions):
        value = getattr(model, name, None) if "." not in name else None
        if "." in name:
            parent, leaf = name.rsplit(".", 1)
            value = getattr(model, parent)[leaf]
        if isinstance(value, torch.nn.Module):
            value.train()


class FrozenAudit:
    def __init__(self, config, model: torch.nn.Module, identity: dict, conditions: tuple[str, ...]):
        self.model, self.identity = model, identity
        self.modules = task_modules(config.name, conditions)
        self.roles = {name: {"trainable": p.requires_grad, "numel": p.numel(),
                             "shape": list(p.shape), "dtype": str(p.dtype)}
                      for name, p in model.named_parameters()}
        self._check_boundary()
        self.before = self._hash()

    def _check_boundary(self) -> None:
        if any(module.training for name, module in self.model.named_modules() if not _inside(name, self.modules)):
            raise ValueError("Frozen modules must remain in evaluation mode")
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad != _inside(name, self.modules):
                raise ValueError("Headft parameter boundary changed: " + name)
            if not parameter.requires_grad and parameter.grad is not None:
                raise ValueError("Frozen parameter received a gradient: " + name)
        if not any(r["trainable"] for r in self.roles.values()):
            raise ValueError("Headft has no trainable task parameters")

    def _hash(self) -> str:
        state = {k: v for k, v in self.model.state_dict().items() if not _inside(k, self.modules)}
        state.update({"buffer:" + k: v for k, v in self.model.named_buffers() if not _inside(k, self.modules)})
        return tensor_state_hash("benchmark.frozen-pretrained-state.v1", state)

    def check_optimizer(self, optimizer) -> list[str]:
        expected = {id(p): n for n, p in self.model.named_parameters() if p.requires_grad}
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise ValueError("Headft optimizer must contain exactly the task parameters")
        return sorted(expected.values())

    def finish(self, root: Path, optimizer) -> dict[str, dict[str, Any]]:
        self._check_boundary()
        after = self._hash()
        if after != self.before:
            raise ValueError("Frozen pretrained parameters or buffers changed")
        payload = {"version": 1, "adaptation_mode": "frozen_pretrained",
                   "training_identity_hash": self.identity["hash"], "task_modules": list(self.modules),
                   "parameters": self.roles, "optimizer_parameters": self.check_optimizer(optimizer),
                   "trainable_parameters": sum(r["numel"] for r in self.roles.values() if r["trainable"]),
                   "frozen_state_before": self.before, "frozen_state_after": after}
        path = root / "frozen_audit.json"
        atomic_json(path, payload)
        return {path.name: {"size": path.stat().st_size, "sha256": sha256_file(path)}}


def begin_frozen_audit(config, model, identity, conditions, optimizer):
    if config.protocol != HEADFT_PROTOCOL:
        return None
    audit = FrozenAudit(config, model, identity, conditions)
    audit.check_optimizer(optimizer)
    return audit


def validate_frozen_audit(config, task, manifest: dict, payload: dict) -> None:
    if config.protocol != HEADFT_PROTOCOL:
        return
    modules = task_modules(config.name, task.condition_columns)
    if (payload.get("version") != 1 or payload.get("adaptation_mode") != "frozen_pretrained"
        or payload.get("training_identity_hash") != manifest["training_identity"]["hash"]
        or payload.get("task_modules") != list(modules)
        or not payload.get("frozen_state_before")
        or payload.get("frozen_state_before") != payload.get("frozen_state_after")):
        raise ValueError("Invalid Headft frozen-state audit")
    roles = payload["parameters"]
    if (not roles or not any(not r["trainable"] for r in roles.values())
        or any(r["trainable"] != _inside(n, modules) for n, r in roles.items())):
        raise ValueError("Invalid Headft parameter roles")
    expected = sorted(n for n, r in roles.items() if r["trainable"])
    count = sum(r["numel"] for r in roles.values() if r["trainable"])
    if not expected or payload.get("optimizer_parameters") != expected or payload.get("trainable_parameters") != count:
        raise ValueError("Invalid Headft optimizer or trainable-parameter audit")


def load_frozen_audit(config, task, manifest: dict, root: Path) -> dict:
    if config.protocol != HEADFT_PROTOCOL:
        return {}
    path = root / "frozen_audit.json"
    entry = manifest.get("integrity", {}).get(path.name)
    if not entry or path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
        raise ValueError("Missing or changed Headft frozen audit")
    payload = json.loads(path.read_text())
    validate_frozen_audit(config, task, manifest, payload)
    return payload


def verify_frozen_model(config, model, task, manifest: dict, root: Path) -> None:
    if config.protocol != HEADFT_PROTOCOL:
        return
    payload = load_frozen_audit(config, task, manifest, root)
    freeze_pretrained(config, model, task.condition_columns)
    restored = FrozenAudit(config, model, manifest["training_identity"], task.condition_columns)
    if restored.before != payload["frozen_state_after"] or restored.roles != payload["parameters"]:
        raise ValueError("Restored Headft frozen model differs from its audit")
