from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from common.identity import require_compatible_identity, tensor_state_hash
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.training import capture_rng_state, resolve_device, restore_rng_state, seed_everything
from .config import Stage3Config, effective_training_seed
from .data import (
    Stage3RepresentationStore,
    Stage3TaskDataset,
    sanitize_task,
    shuffled_epoch_indices,
    stable_seed,
)
from .evaluate import (
    _load_model,
    _three_phase_final_path,
    _validate_gate_calibrated_manifest,
    _validate_three_phase_manifest,
)
from .identity import build_stage3_gate_calibration_identity
from .model import Stage3SparseModel, summarize_task_gate_observations, task_gate_observations
from .prepare import load_prepared_stage3
from .train import _batch, compute_task_gradient, resolve_stage3_training_identity, validate_tasks


GATE_CALIBRATION_CHECKPOINT_VERSION = 1
GATE_CALIBRATED_FORMAT_VERSION = 1
GATE_CALIBRATED_KIND = "ilume_stage3_three_phase_gate_calibrated"
GATE_CALIBRATED_RDKIT_KIND = "ilume_stage3_rdkit_home_three_phase_gate_calibrated"


def _model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def calibrated_model_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    return tensor_state_hash("stage3.three-phase-model-state", state)


def _gate_names(model: Stage3SparseModel, task: str) -> tuple[str, ...]:
    prefix = f"task_gates.{sanitize_task(task)}."
    names = tuple(
        name for name, _ in model.named_parameters() if name.startswith(prefix)
    )
    if len(names) != 2:
        raise RuntimeError(f"Stage 3 task gate must contain weight and bias: {task}")
    return names


def _gate_state(
    model: Stage3SparseModel, task: str
) -> dict[str, torch.Tensor]:
    names = set(_gate_names(model, task))
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name in names
    }


def gate_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    return tensor_state_hash("stage3.gate-calibration-task-gate-state", state)


def _load_gate_state(
    model: Stage3SparseModel,
    task: str,
    state: Mapping[str, torch.Tensor],
) -> None:
    expected = set(_gate_names(model, task))
    if set(state) != expected:
        raise ValueError(f"Stage 3 calibration delta has wrong parameters: {task}")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            if value.shape != parameters[name].shape:
                raise ValueError(f"Stage 3 calibration delta shape mismatch: {name}")
            parameters[name].copy_(value.to(parameters[name].device))


def _require_only_gate_changed(
    model: Stage3SparseModel,
    anchor: Mapping[str, torch.Tensor],
    task: str,
) -> None:
    allowed = set(_gate_names(model, task))
    current = model.state_dict()
    if set(current) != set(anchor):
        raise ValueError("Stage 3 calibration anchor state keys changed")
    changed = [
        name
        for name in current
        if name not in allowed
        and not torch.equal(current[name].detach().cpu(), anchor[name])
    ]
    if changed:
        raise RuntimeError(
            "Stage 3 calibration changed non-gate state: " + ", ".join(changed[:5])
        )


def stitch_gate_deltas(
    model: Stage3SparseModel,
    anchor: Mapping[str, torch.Tensor],
    deltas: Mapping[str, tuple[Mapping[str, torch.Tensor], str]],
) -> dict[str, torch.Tensor]:
    model.load_state_dict(anchor, strict=True)
    seen: set[str] = set()
    for task, (state, state_hash) in deltas.items():
        names = set(_gate_names(model, task))
        if seen & names:
            raise RuntimeError(f"Stage 3 calibration deltas overlap: {task}")
        if gate_state_hash(state) != state_hash:
            raise ValueError(f"Stage 3 calibration delta hash mismatch: {task}")
        _load_gate_state(model, task, state)
        seen.update(names)
    return _model_state(model)


def _optimizer(
    model: Stage3SparseModel, task: str, config: Stage3Config, lr: float
) -> torch.optim.AdamW:
    parameters = model.task_gate_parameters(task)
    decay = [parameter for parameter in parameters if parameter.ndim >= 2]
    no_decay = [parameter for parameter in parameters if parameter.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.training.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=config.training.betas,
        eps=config.training.eps,
        foreach=False,
        fused=False,
    )


@torch.no_grad()
def _gate_weights(
    model: Stage3SparseModel,
    task: str,
    dataset: Stage3TaskDataset,
    representations: Stage3RepresentationStore | torch.Tensor,
    normalization: Mapping[str, Any],
    config: Stage3Config,
    device: torch.device,
) -> torch.Tensor:
    result: list[torch.Tensor] = []
    for start in range(0, len(dataset), config.training.microbatch_size):
        indices = torch.arange(
            start, min(len(dataset), start + config.training.microbatch_size)
        )
        primary, conditions, partner, _ = _batch(
            dataset,
            indices,
            representations,
            model.task_specs[task],
            normalization,
            device,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=config.training.amp_dtype == "bf16",
        ):
            output = model(task, primary, conditions, partner_embedding=partner)
        result.append(output.diagnostics["task_gate"].detach().float().cpu())
    return torch.cat(result)


def gate_kl_divergence(
    anchor_weights: torch.Tensor, calibrated_weights: torch.Tensor
) -> float:
    if anchor_weights.shape != calibrated_weights.shape:
        raise ValueError("Stage 3 gate KL tensors have different shapes")
    anchor = anchor_weights.detach().double().cpu()
    calibrated = calibrated_weights.detach().double().cpu()
    if not torch.isfinite(anchor).all() or not torch.isfinite(calibrated).all():
        raise ValueError("Stage 3 gate KL tensors must be finite")
    tiny = torch.finfo(torch.float64).tiny
    value = (
        anchor
        * (anchor.clamp_min(tiny).log() - calibrated.clamp_min(tiny).log())
    ).sum(dim=1).mean()
    return max(0.0, float(value))


def _gate_delta_norm(
    anchor: Mapping[str, torch.Tensor], state: Mapping[str, torch.Tensor]
) -> float:
    total = sum(
        (state[name].double() - anchor[name].double()).square().sum()
        for name in state
    )
    return float(total.sqrt())


def _comparison(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    gate_delta_norm: float,
    gate_kl: float,
) -> dict[str, Any]:
    before_metrics = before["tasks"]
    after_metrics = after["tasks"]
    task = next(iter(after_metrics))
    return {
        "task": task,
        "before": {
            "mae": before_metrics[task]["mae"],
            "normalized_mae": before_metrics[task]["normalized_mae"],
            "gate_diagnostics": before["gate_diagnostics"][task],
        },
        "after": {
            "mae": after_metrics[task]["mae"],
            "normalized_mae": after_metrics[task]["normalized_mae"],
            "gate_diagnostics": after["gate_diagnostics"][task],
        },
        "delta_mae": after_metrics[task]["mae"] - before_metrics[task]["mae"],
        "delta_normalized_mae": (
            after_metrics[task]["normalized_mae"]
            - before_metrics[task]["normalized_mae"]
        ),
        "gate_parameter_delta_norm": gate_delta_norm,
        "pre_post_gate_kl": gate_kl,
    }


def build_resolved_calibration_plan(
    config: Stage3Config,
    fold: int,
    base_artifact: Mapping[str, Any],
    base_artifact_sha256: str,
    model: Stage3SparseModel,
    train_data: Mapping[str, Stage3TaskDataset],
) -> dict[str, Any]:
    calibration = config.gate_calibration
    base_plan = base_artifact["resolved_training_plan"]
    tasks: dict[str, Any] = {}
    for task in sorted(train_data):
        phase3 = base_plan["phases"]["phase3"]["branches"][task]
        private = phase3["owners"][f"PRIVATE:{task}"]
        count = len(train_data[task])
        batch_size = int(base_plan["data"]["B_t"][task])
        updates_per_epoch = math.ceil(count / batch_size)
        tasks[task] = {
            "epochs": calibration.epochs,
            "phase3_private_lr": float(private["nominal_lr"]),
            "lr": float(private["nominal_lr"]) * calibration.lr_scale,
            "N_t": count,
            "B_t": batch_size,
            "updates_per_epoch": updates_per_epoch,
            "actual_update_budget": updates_per_epoch * calibration.epochs,
            "seed": stable_seed(
                effective_training_seed(config), fold, "gate_calibration", task
            ),
            "parameters": list(_gate_names(model, task)),
            "carried_from_anchor": calibration.epochs == 0,
        }
    return {
        "format_version": 1,
        "fold": fold,
        "base_artifact_sha256": base_artifact_sha256,
        "base_model_state_hash": base_artifact["model_state_hash"],
        "base_training_identity": base_artifact["training_identity"],
        "prepared_identity": base_plan["prepared_identity"],
        "normalization_hash": base_artifact["normalization_hash"],
        "ownership_manifest": base_artifact["ownership_manifest"],
        "calibration": {
            "epochs": calibration.epochs,
            "lr_scale": calibration.lr_scale,
            "optimizer": "AdamW",
            "betas": list(config.training.betas),
            "eps": config.training.eps,
            "weight_decay": config.training.weight_decay,
            "max_grad_norm": config.training.max_grad_norm,
            "learning_rate_schedule": "constant",
            "loss": "smooth_l1",
            "smooth_l1_beta": config.training.smooth_l1_beta,
            "sampling": "raw_without_replacement",
            "pcgrad": "off",
            "seed": effective_training_seed(config),
        },
        "tasks": tasks,
    }


def _history_epochs(path: Path) -> list[int]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if any(not isinstance(row, dict) or not isinstance(row.get("epoch"), int) for row in rows):
        raise ValueError(f"Malformed Stage 3 gate calibration history: {path}")
    return [int(row["epoch"]) for row in rows]


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, allow_nan=True, sort_keys=True) + "\n")


def _checkpoint_path(root: Path, epoch: int) -> Path:
    return root / f"checkpoint_epoch_{epoch:05d}.pt"


def _checkpoint_epochs(root: Path) -> list[int]:
    return sorted(
        int(path.stem.rsplit("_", 1)[1])
        for path in root.glob("checkpoint_epoch_*.pt")
    )


def _run_task_branch(
    *,
    root: Path,
    model: Stage3SparseModel,
    task: str,
    recipe: Mapping[str, Any],
    anchor_state: Mapping[str, torch.Tensor],
    anchor_hash: str,
    calibration_identity: Mapping[str, Any],
    train_data: Stage3TaskDataset,
    valid_data: Stage3TaskDataset,
    representations: Stage3RepresentationStore | torch.Tensor,
    normalization: Mapping[str, Any],
    config: Stage3Config,
    device: torch.device,
    resume: bool,
) -> tuple[dict[str, torch.Tensor], str, dict[str, Any]]:
    model.load_state_dict(anchor_state, strict=True)
    model.set_task_gate_calibration_mode(task)
    anchor_gate_state = _gate_state(model, task)
    before_validation = validate_tasks(
        model,
        {task: valid_data},
        representations,
        {task: normalization},
        config,
        device,
    )
    before_weights = _gate_weights(
        model, task, valid_data, representations, normalization, config, device
    )
    epochs = int(recipe["epochs"])
    if epochs == 0:
        comparison = _comparison(
            before_validation,
            before_validation,
            gate_delta_norm=0.0,
            gate_kl=0.0,
        )
        state = _gate_state(model, task)
        return state, gate_state_hash(state), comparison

    root.mkdir(parents=True, exist_ok=True)
    metrics_epochs = _history_epochs(root / "metrics.jsonl")
    diagnostics_epochs = _history_epochs(root / "diagnostics.jsonl")
    if metrics_epochs != diagnostics_epochs or metrics_epochs != list(
        range(1, len(metrics_epochs) + 1)
    ):
        raise ValueError(f"Stage 3 calibration histories are inconsistent: {task}")
    if _checkpoint_epochs(root) != metrics_epochs:
        raise ValueError(f"Stage 3 calibration checkpoints are inconsistent: {task}")
    if metrics_epochs and not resume:
        raise FileExistsError(f"Stage 3 calibration branch already exists: {task}")
    start = len(metrics_epochs) + 1
    optimizer = _optimizer(model, task, config, float(recipe["lr"]))
    updates = 0
    if metrics_epochs:
        checkpoint = torch.load(
            _checkpoint_path(root, metrics_epochs[-1]),
            map_location="cpu",
            weights_only=False,
        )
        if (
            checkpoint.get("kind") != "ilume_stage3_gate_calibration_delta"
            or checkpoint.get("format_version") != GATE_CALIBRATION_CHECKPOINT_VERSION
            or checkpoint.get("task") != task
            or checkpoint.get("completed_epoch") != metrics_epochs[-1]
            or checkpoint.get("base_model_state_hash") != anchor_hash
            or checkpoint.get("calibration_identity") != calibration_identity
            or checkpoint.get("gate_state_hash")
            != gate_state_hash(checkpoint.get("gate_state", {}))
            or checkpoint.get("updates")
            != metrics_epochs[-1] * int(recipe["updates_per_epoch"])
        ):
            raise ValueError(f"Stage 3 calibration checkpoint mismatch: {task}")
        _load_gate_state(model, task, checkpoint["gate_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng_state(checkpoint["rng"])
        updates = int(checkpoint["updates"])
    else:
        seed_everything(int(recipe["seed"]) % (2**32))

    last_validation: dict[str, Any] | None = None
    last_comparison: dict[str, Any] | None = None
    for epoch in range(start, epochs + 1):
        model.set_task_gate_calibration_mode(task)
        sequence = shuffled_epoch_indices(
            int(recipe["N_t"]),
            seed=int(recipe["seed"]),
            epoch=epoch,
            task_id=task,
        )
        loss_sum = 0.0
        samples = 0
        last_pre_norm = last_post_norm = 0.0
        parameters = model.task_gate_parameters(task)
        for begin in range(0, int(recipe["N_t"]), int(recipe["B_t"])):
            indices = sequence[begin : begin + int(recipe["B_t"])]
            gradients, loss = compute_task_gradient(
                model,
                task,
                train_data,
                indices,
                representations,
                normalization,
                config,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in gradients.items():
                parameter.grad = gradient.to(parameter.device, dtype=parameter.dtype)
            last_pre_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    parameters, float("inf"), error_if_nonfinite=True
                )
            )
            if config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    config.training.max_grad_norm,
                    error_if_nonfinite=True,
                )
            last_post_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    parameters, float("inf"), error_if_nonfinite=True
                )
            )
            optimizer.step()
            updates += 1
            loss_sum += loss * len(indices)
            samples += len(indices)
        if samples != int(recipe["N_t"]):
            raise RuntimeError(f"Stage 3 calibration raw coverage is incomplete: {task}")
        _require_only_gate_changed(model, anchor_state, task)
        last_validation = validate_tasks(
            model,
            {task: valid_data},
            representations,
            {task: normalization},
            config,
            device,
        )
        after_weights = _gate_weights(
            model, task, valid_data, representations, normalization, config, device
        )
        state = _gate_state(model, task)
        last_comparison = _comparison(
            before_validation,
            last_validation,
            gate_delta_norm=_gate_delta_norm(anchor_gate_state, state),
            gate_kl=gate_kl_divergence(before_weights, after_weights),
        )
        _append_jsonl(
            root / "metrics.jsonl",
            {
                "epoch": epoch,
                "task": task,
                "updates": updates,
                "lr": float(recipe["lr"]),
                "training_loss": loss_sum / int(recipe["N_t"]),
                "validation": last_validation,
                "comparison": last_comparison,
            },
        )
        _append_jsonl(
            root / "diagnostics.jsonl",
            {
                "epoch": epoch,
                "task": task,
                "updates": updates,
                "gradient_norm": last_pre_norm,
                "clip_post_norm": last_post_norm,
                "gate_parameter_delta_norm": last_comparison[
                    "gate_parameter_delta_norm"
                ],
                "pre_post_gate_kl": last_comparison["pre_post_gate_kl"],
            },
        )
        checkpoint_path = _checkpoint_path(root, epoch)
        if checkpoint_path.exists():
            raise FileExistsError(
                f"Stage 3 calibration checkpoint already exists: {checkpoint_path}"
            )
        atomic_torch_save(
            checkpoint_path,
            {
                "kind": "ilume_stage3_gate_calibration_delta",
                "format_version": GATE_CALIBRATION_CHECKPOINT_VERSION,
                "task": task,
                "completed_epoch": epoch,
                "updates": updates,
                "base_model_state_hash": anchor_hash,
                "calibration_identity": calibration_identity,
                "gate_state": state,
                "gate_state_hash": gate_state_hash(state),
                "optimizer": optimizer.state_dict(),
                "rng": capture_rng_state(),
            },
        )
    if last_comparison is None:
        row = json.loads((root / "metrics.jsonl").read_text().splitlines()[-1])
        last_comparison = row["comparison"]
    state = _gate_state(model, task)
    _require_only_gate_changed(model, anchor_state, task)
    return state, gate_state_hash(state), last_comparison


def run_gate_calibration(
    config: Stage3Config,
    fold: int,
    *,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    resume: bool = False,
) -> dict[str, Any]:
    if not config.gate_calibration.enabled:
        raise ValueError("Stage 3 gate calibration is not enabled")
    if config.training.schedule_mode != "three_phase":
        raise ValueError("Stage 3 gate calibration requires three-phase training")
    device = resolve_device(config.training.device)
    if config.training.amp_dtype == "bf16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("Stage 3 BF16 gate calibration requires capable CUDA")
    prepared = load_prepared_stage3(config)
    base_path = _three_phase_final_path(Path(checkpoint_dir), fold)
    if not base_path.is_file():
        raise FileNotFoundError(f"Missing Stage 3 three-phase final: {base_path}")
    model, base_artifact, representations = _load_model(
        config,
        prepared,
        base_path,
        fold,
        config.training.epochs,
        device,
        three_phase_final=True,
    )
    _validate_three_phase_manifest(base_path, base_artifact)
    require_compatible_identity(
        resolve_stage3_training_identity(config, fold),
        base_artifact["training_identity"],
        context="Stage 3 gate calibration base artifact",
    )
    active = tuple(base_artifact["resolved_training_plan"]["active_tasks"])
    train_data = {
        task: Stage3TaskDataset(config.data.artifacts_dir, fold, task, "train")
        for task in active
    }
    valid_data = {
        task: Stage3TaskDataset(config.data.artifacts_dir, fold, task, "valid")
        for task in active
    }
    normalizations = base_artifact["normalization"]
    anchor_state = {
        name: value.detach().cpu().clone()
        for name, value in base_artifact["model"].items()
    }
    anchor_hash = base_artifact["model_state_hash"]
    rdkit = "representation" in base_artifact
    kind = GATE_CALIBRATED_RDKIT_KIND if rdkit else GATE_CALIBRATED_KIND
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan = build_resolved_calibration_plan(
        config,
        fold,
        base_artifact,
        sha256_file(base_path),
        model,
        train_data,
    )
    calibration_identity = build_stage3_gate_calibration_identity(plan)
    plan_path = output / "resolved_calibration_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        require_compatible_identity(
            calibration_identity,
            build_stage3_gate_calibration_identity(existing),
            context="Existing Stage 3 gate calibration plan",
        )
    else:
        atomic_json(plan_path, plan)

    artifact_path = output / "gate_calibrated.pt"
    manifest_path = output / "gate_calibrated.json"
    if artifact_path.exists() or manifest_path.exists():
        if not resume or not artifact_path.is_file() or not manifest_path.is_file():
            raise FileExistsError("Stage 3 gate calibration final artifact is incomplete")
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        manifest = json.loads(manifest_path.read_text())
        if (
            artifact.get("kind") != kind
            or artifact.get("format_version") != GATE_CALIBRATED_FORMAT_VERSION
            or artifact.get("fold") != fold
            or artifact.get("base_artifact_sha256") != sha256_file(base_path)
            or artifact.get("base_model_state_hash") != anchor_hash
            or artifact.get("model_state_hash")
            != calibrated_model_state_hash(artifact.get("model", {}))
            or set(artifact.get("model", {})) != set(anchor_state)
            or artifact.get("ownership_manifest")
            != base_artifact["ownership_manifest"]
        ):
            raise ValueError("Stage 3 gate calibration final artifact is corrupt")
        require_compatible_identity(
            calibration_identity,
            artifact.get("calibration_identity", {}),
            context="Stage 3 gate calibration final artifact",
        )
        require_compatible_identity(
            base_artifact["training_identity"],
            artifact.get("base_training_identity", {}),
            context="Stage 3 gate calibration base training identity",
        )
        _validate_gate_calibrated_manifest(artifact_path, artifact)
        allowed = set().union(*(_gate_names(model, task) for task in active))
        if any(
            name not in allowed and not torch.equal(value, anchor_state[name])
            for name, value in artifact["model"].items()
        ):
            raise ValueError("Stage 3 gate calibration changed non-gate state")
        return manifest

    deltas: dict[str, tuple[Mapping[str, torch.Tensor], str]] = {}
    task_records: dict[str, Any] = {}
    for task in sorted(active):
        state, state_hash, comparison = _run_task_branch(
            root=output / "tasks" / sanitize_task(task),
            model=model,
            task=task,
            recipe=plan["tasks"][task],
            anchor_state=anchor_state,
            anchor_hash=anchor_hash,
            calibration_identity=calibration_identity,
            train_data=train_data[task],
            valid_data=valid_data[task],
            representations=representations,
            normalization=normalizations[task],
            config=config,
            device=device,
            resume=resume,
        )
        deltas[task] = (state, state_hash)
        task_records[task] = {
            "recipe": plan["tasks"][task],
            "base_model_state_hash": anchor_hash,
            "task_gate_state_hash": state_hash,
            "comparison": comparison,
        }
    if set(deltas) != set(active):
        raise RuntimeError("Stage 3 gate calibration task set is incomplete")
    final_state = stitch_gate_deltas(model, anchor_state, deltas)
    allowed = set().union(*(_gate_names(model, task) for task in active))
    changed_outside = [
        name
        for name in final_state
        if name not in allowed and not torch.equal(final_state[name], anchor_state[name])
    ]
    if changed_outside:
        raise RuntimeError("Stage 3 calibrated model changed non-gate state")
    final_hash = calibrated_model_state_hash(final_state)
    if config.gate_calibration.epochs == 0 and any(
        not torch.equal(final_state[name], anchor_state[name]) for name in final_state
    ):
        raise RuntimeError("Zero-epoch Stage 3 calibration changed model state")
    final_validation = validate_tasks(
        model,
        valid_data,
        representations,
        normalizations,
        config,
        device,
    )
    artifact = {
        "kind": kind,
        "format_version": GATE_CALIBRATED_FORMAT_VERSION,
        "fold": fold,
        "model": final_state,
        "model_state_hash": final_hash,
        "base_artifact_sha256": plan["base_artifact_sha256"],
        "base_model_state_hash": anchor_hash,
        "base_training_identity": base_artifact["training_identity"],
        "training_identity": base_artifact["training_identity"],
        "calibration_identity": calibration_identity,
        "resolved_calibration_plan": plan,
        "resolved_training_plan": base_artifact["resolved_training_plan"],
        "resolved_registry": base_artifact["resolved_registry"],
        "normalization": normalizations,
        "normalization_hash": base_artifact["normalization_hash"],
        "ownership_manifest": base_artifact["ownership_manifest"],
        "task_gate_state_hashes": {
            task: task_records[task]["task_gate_state_hash"] for task in active
        },
        "task_comparisons": {
            task: task_records[task]["comparison"] for task in active
        },
        "validation": final_validation,
        **(
            {"representation": base_artifact["representation"]}
            if rdkit
            else {"stage2_encoder_identity": base_artifact["stage2_encoder_identity"]}
        ),
    }
    atomic_torch_save(artifact_path, artifact)
    manifest = {
        "kind": kind,
        "format_version": GATE_CALIBRATED_FORMAT_VERSION,
        "fold": fold,
        "artifact": artifact_path.name,
        "artifact_sha256": sha256_file(artifact_path),
        "model_state_hash": final_hash,
        "base_artifact_sha256": plan["base_artifact_sha256"],
        "base_model_state_hash": anchor_hash,
        "base_training_identity": base_artifact["training_identity"],
        "calibration_identity": calibration_identity,
        "calibration": plan["calibration"],
        "ownership_manifest": base_artifact["ownership_manifest"],
        "tasks": task_records,
        "validation": final_validation,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def resolve_gate_calibration_identity(
    config: Stage3Config,
    fold: int,
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    if not config.gate_calibration.enabled:
        raise ValueError("Stage 3 gate calibration is not enabled")
    prepared = load_prepared_stage3(config)
    base_path = _three_phase_final_path(Path(checkpoint_dir), fold)
    if not base_path.is_file():
        raise FileNotFoundError(f"Missing Stage 3 three-phase final: {base_path}")
    model, base_artifact, _ = _load_model(
        config,
        prepared,
        base_path,
        fold,
        config.training.epochs,
        torch.device("cpu"),
        three_phase_final=True,
    )
    _validate_three_phase_manifest(base_path, base_artifact)
    require_compatible_identity(
        resolve_stage3_training_identity(config, fold),
        base_artifact["training_identity"],
        context="Stage 3 gate calibration base artifact",
    )
    active = tuple(base_artifact["resolved_training_plan"]["active_tasks"])
    train_data = {
        task: Stage3TaskDataset(config.data.artifacts_dir, fold, task, "train")
        for task in active
    }
    plan = build_resolved_calibration_plan(
        config,
        fold,
        base_artifact,
        sha256_file(base_path),
        model,
        train_data,
    )
    return build_stage3_gate_calibration_identity(plan)


__all__ = [
    "GATE_CALIBRATED_FORMAT_VERSION",
    "GATE_CALIBRATED_KIND",
    "GATE_CALIBRATED_RDKIT_KIND",
    "build_resolved_calibration_plan",
    "calibrated_model_state_hash",
    "gate_kl_divergence",
    "gate_state_hash",
    "resolve_gate_calibration_identity",
    "run_gate_calibration",
    "stitch_gate_deltas",
]
