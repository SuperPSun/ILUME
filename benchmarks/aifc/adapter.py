from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from common.identity import require_compatible_identity, semantic_identity, tensor_state_hash
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.outputs import repository_path
from common.progress import ProgressReporter

from benchmarks.common.config import BenchmarkConfig, BenchmarkName
from benchmarks.common.data import BenchmarkTask, RawDataset, load_split, resolve_task
from benchmarks.common.engine import EvaluationResult, TargetStats, seed_benchmark
from benchmarks.common.metrics import target_metrics

from .model import AIFCRegressor
from .preprocessing import (
    AIFCGraph,
    FRAGMENT_BLOB,
    FRAGMENT_COMMIT,
    FragmentScheme,
    batch_aifc_graphs,
    canonicalize_view,
    graph_audit,
    smiles_to_aifc_graph,
)


AIFC_INPUT_CONTRACT = {
    "atom_features": 46,
    "bond_features": 12,
    "fragment_rows": 100,
    "unknown_fragment": "one_atom_motif_with_zero_100d_type",
    "ordinary_il_view": "canonical_cation_dot_anion",
    "partner_view": "registry_slot_ordered_components",
    "fusion": "shared_encoder_ordered_concat_then_registry_conditions",
}


@dataclass(frozen=True)
class ConditionStats:
    columns: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    constant: tuple[bool, ...]

    @classmethod
    def fit(cls, columns: Sequence[str], values: np.ndarray) -> "ConditionStats":
        names = tuple(columns)
        if values.ndim != 2 or values.shape[1] != len(names) or not np.isfinite(values).all():
            raise ValueError("AIFC train conditions must be a finite matrix")
        if not len(values):
            if names:
                raise ValueError("AIFC train conditions are empty")
            return cls((), (), (), ())
        mean = values.mean(axis=0)
        raw_scale = values.std(axis=0, ddof=0)
        constant = np.ptp(values, axis=0) == 0
        scale = np.where(constant, 1.0, raw_scale)
        if not np.isfinite(scale).all() or bool((scale <= 0).any()):
            raise ValueError("AIFC train condition variance is invalid")
        return cls(
            names,
            tuple(map(float, mean)),
            tuple(map(float, scale)),
            tuple(map(bool, constant)),
        )

    def normalize(self, values: np.ndarray) -> np.ndarray:
        if values.ndim != 2 or values.shape[1] != len(self.columns):
            raise ValueError("AIFC condition shape differs from registry")
        if not self.columns:
            return np.empty((len(values), 0), dtype=np.float32)
        return ((values - np.asarray(self.mean)) / np.asarray(self.scale)).astype(np.float32)


@dataclass(frozen=True)
class PreparedSplit:
    raw: RawDataset
    view_names: tuple[str, ...]
    model_views: tuple[tuple[str, ...], ...]
    graphs: tuple[tuple[AIFCGraph, ...], ...]
    normalized_conditions: np.ndarray
    audit: dict[str, Any]

    @property
    def view_count(self) -> int:
        return len(self.view_names)


@dataclass
class AIFCTrainingBundle:
    task: BenchmarkTask
    train: PreparedSplit
    valid: PreparedSplit
    target_stats: TargetStats
    condition_stats: ConditionStats
    source_hashes: dict[str, Any]
    training_identity: dict[str, Any]
    scheme: FragmentScheme
    architecture: dict[str, Any]


def aifc_model_views(
    task: BenchmarkTask, components: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if len(components) != len(task.slots):
        raise ValueError("AIFC components differ from registry slots")
    if task.slots == ("cation", "anion"):
        return (canonicalize_view(".".join(components)),), ("ionic_liquid",)
    if task.slots == ("cation", "anion", "solute"):
        return tuple(canonicalize_view(value) for value in components), task.slots
    if task.slots == ("solute", "solvent"):
        return tuple(canonicalize_view(value) for value in components), task.slots
    raise ValueError(f"Unsupported AIFC Stage 3 registry topology: {task.slots}")


def resolve_aifc_architecture(config: BenchmarkConfig, task_id: str) -> dict[str, Any]:
    specific = config.model["property_architectures"].get(task_id)
    resolved = dict(config.model["fallback_architecture"] if specific is None else specific)
    resolved["source"] = "fallback" if specific is None else "official_public_script"
    return resolved


def _prepare_split(
    task: BenchmarkTask,
    raw: RawDataset,
    stats: ConditionStats,
    scheme: FragmentScheme,
    cache: dict[str, AIFCGraph],
) -> PreparedSplit:
    rows: list[tuple[str, ...]] = []
    graph_rows: list[tuple[AIFCGraph, ...]] = []
    names: tuple[str, ...] | None = None
    role_graphs: dict[str, list[AIFCGraph]] = {}
    for components in raw.components:
        views, current_names = aifc_model_views(task, components)
        if names is None:
            names = current_names
            role_graphs = {name: [] for name in names}
        elif names != current_names:
            raise RuntimeError("AIFC registry view order changed within a split")
        resolved_graphs = []
        for view in views:
            if view not in cache:
                cache[view] = smiles_to_aifc_graph(view, scheme)
            resolved_graphs.append(cache[view])
        graphs = tuple(resolved_graphs)
        rows.append(views)
        graph_rows.append(graphs)
        for name, graph in zip(current_names, graphs, strict=True):
            role_graphs[name].append(graph)
    resolved_names = names or (
        ("ionic_liquid",) if task.slots == ("cation", "anion") else task.slots
    )
    return PreparedSplit(
        raw=raw,
        view_names=resolved_names,
        model_views=tuple(rows),
        graphs=tuple(graph_rows),
        normalized_conditions=stats.normalize(raw.conditions),
        audit={
            "rows": len(raw),
            "view_names": list(resolved_names),
            "view_order": list(resolved_names),
            "registry_slots": list(task.slots),
            "condition_columns": list(task.condition_columns),
            "condition_statistics": asdict(stats),
            "roles": {name: graph_audit(graphs) for name, graphs in role_graphs.items()},
            "input_contract": AIFC_INPUT_CONTRACT,
            "fragment_scheme": {
                "commit": FRAGMENT_COMMIT,
                "blob": FRAGMENT_BLOB,
                "sha256": scheme.sha256,
                "entries": len(scheme.names),
                "order_sha256": scheme.order_sha256,
            },
        },
    )


def prepare_aifc_training(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
) -> AIFCTrainingBundle:
    if config.name != "aifc":
        raise ValueError("AIFC adapter requires name=aifc")
    task = resolve_task(config, benchmark, task_id, fold)
    if len(task.target_columns) != 1:
        raise ValueError("AIFC requires one scalar target")
    train_raw = load_split(task, "train")
    valid_raw = load_split(task, "valid")
    condition_stats = ConditionStats.fit(task.condition_columns, train_raw.conditions)
    target_stats = TargetStats.fit(train_raw.targets)
    scheme_path = repository_path(config.model["fragment_scheme"])
    scheme = FragmentScheme.load(scheme_path)
    cache: dict[str, AIFCGraph] = {}
    train = _prepare_split(task, train_raw, condition_stats, scheme, cache)
    valid = _prepare_split(task, valid_raw, condition_stats, scheme, cache)
    architecture = resolve_aifc_architecture(config, task_id)
    source_hashes = {
        "train": [sha256_file(path) for path in task.train_paths],
        "valid": [sha256_file(path) for path in task.valid_paths],
    }
    identity = semantic_identity(
        "benchmark.training.v1",
        {
            "benchmark_model": config.name,
            "domain": benchmark,
            "task_id": task_id,
            "fold": fold,
            "registry": task.registry_payload,
            "source_hashes": source_hashes,
            "view_order": list(train.view_names),
            "input_contract": AIFC_INPUT_CONTRACT,
            "condition_statistics": asdict(condition_stats),
            "target_statistics": target_stats.to_dict(),
            "architecture": architecture,
            "model": config.model,
            "training": config.training,
            "seed": config.seed,
        },
    )
    return AIFCTrainingBundle(
        task, train, valid, target_stats, condition_stats, source_hashes,
        identity, scheme, architecture,
    )


def build_aifc_model(bundle: AIFCTrainingBundle) -> AIFCRegressor:
    architecture = bundle.architecture
    model = AIFCRegressor(
        fragment_dim=len(bundle.scheme.names),
        hidden_dim=int(architecture["hidden_dim"]),
        num_heads=int(architecture["num_heads"]),
        dropout=float(architecture["dropout"]),
        depth=int(architecture["depth"]),
        layers=int(architecture["layers"]),
        view_count=bundle.train.view_count,
        condition_dim=len(bundle.task.condition_columns),
    )
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("AIFC unexpectedly contains frozen parameters")
    return model


def _batch(
    prepared: PreparedSplit,
    indices: Sequence[int],
    target_stats: TargetStats,
    *,
    include_labels: bool,
) -> tuple[AIFCGraph, torch.Tensor, torch.Tensor | None]:
    graphs = [graph for index in indices for graph in prepared.graphs[index]]
    conditions = torch.from_numpy(prepared.normalized_conditions[np.asarray(indices)])
    labels = None
    if include_labels:
        labels = torch.from_numpy(
            target_stats.normalize(prepared.raw.targets[np.asarray(indices)])[:, 0]
        )
    return batch_aifc_graphs(graphs), conditions, labels


def _predict(
    model: AIFCRegressor,
    prepared: PreparedSplit,
    target_stats: TargetStats,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(prepared.raw), batch_size):
            indices = list(range(start, min(start + batch_size, len(prepared.raw))))
            graph, conditions, _ = _batch(prepared, indices, target_stats, include_labels=False)
            values.append(model(graph.to(device), conditions.to(device)).cpu().numpy())
    normalized = np.concatenate(values) if values else np.empty((0,), dtype=np.float32)
    return target_stats.denormalize(normalized.reshape(-1, 1))


def _configure_fp32() -> None:
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"


def train_aifc_bundle(
    config: BenchmarkConfig,
    bundle: AIFCTrainingBundle,
    output_dir: str | Path,
    *,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(config.training["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("AIFC requires CUDA; no silent CPU fallback")
    seed_benchmark(config.seed)
    _configure_fp32()
    model = build_aifc_model(bundle).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.training["learning_rate"]),
        betas=tuple(map(float, config.training["betas"])),
        eps=float(config.training["eps"]),
        weight_decay=float(config.training["weight_decay"]),
    )
    epochs = int(config.training["max_epochs"])
    batch_size = int(config.training["batch_size"])
    generator = torch.Generator().manual_seed(config.seed)
    watched = {
        "fragment_node_embedding": model.encoder.fragment_node_embedding[0].weight.detach().cpu().clone(),
        "fragment_message": model.encoder.fragment_heads[0].atom.layers[0].node_embedding[0].weight.detach().cpu().clone(),
        "junction_message": model.encoder.junction_heads[0].atom.layers[0].node_embedding[0].weight.detach().cpu().clone(),
        "regression_head": model.predictor[1].weight.detach().cpu().clone(),
    }
    history: list[dict[str, float | int]] = []
    progress = (reporter or ProgressReporter()).bar(
        total=epochs,
        desc=f"AIFC {bundle.task.task_id} fold{bundle.task.fold}",
        unit="epoch",
    )
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(bundle.train.raw), generator=generator).tolist()
            loss_sum = 0.0
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                graph, conditions, labels = _batch(
                    bundle.train, indices, bundle.target_stats, include_labels=True
                )
                assert labels is not None
                optimizer.zero_grad(set_to_none=True)
                prediction = model(graph.to(device), conditions.to(device))
                loss = torch.nn.functional.mse_loss(prediction, labels.to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("AIFC training loss is non-finite")
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach().cpu()) * len(indices)
            predictions = _predict(
                model, bundle.valid, bundle.target_stats, device, batch_size
            )
            valid_mae = float(
                np.abs(predictions[:, 0] - bundle.valid.raw.targets[:, 0]).mean()
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_normalized_mse": loss_sum / len(bundle.train.raw),
                    "valid_raw_mae": valid_mae,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            progress.set_postfix(
                {"train_mse": f"{history[-1]['train_normalized_mse']:.4f}", "val_mae": f"{valid_mae:.4f}"}
            )
            progress.update(1)
    finally:
        progress.close()
    if len(history) != epochs:
        raise RuntimeError("AIFC did not complete its fixed 20-epoch budget")
    current = {
        "fragment_node_embedding": model.encoder.fragment_node_embedding[0].weight.detach().cpu(),
        "fragment_message": model.encoder.fragment_heads[0].atom.layers[0].node_embedding[0].weight.detach().cpu(),
        "junction_message": model.encoder.junction_heads[0].atom.layers[0].node_embedding[0].weight.detach().cpu(),
        "regression_head": model.predictor[1].weight.detach().cpu(),
    }
    parameter_audit = {
        "fully_trainable": all(parameter.requires_grad for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "selected_parameter_max_abs_delta": {
            name: float((current[name] - initial).abs().max())
            for name, initial in watched.items()
        },
    }
    if not parameter_audit["fully_trainable"] or any(
        value <= 0 for value in parameter_audit["selected_parameter_max_abs_delta"].values()
    ):
        raise RuntimeError("AIFC trainable-parameter audit failed")
    atomic_json(root / "training_history.json", history)
    model_file = root / f"checkpoint_epoch_{epochs:05d}.pt"
    final_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    atomic_torch_save(model_file, final_state)
    state_hash = tensor_state_hash("benchmark.aifc-state.v1", final_state)
    integrity = {
        path.name: {"sha256": sha256_file(path), "size": path.stat().st_size}
        for path in (root / "training_history.json", model_file)
    }
    manifest = {
        "format_version": 2,
        "kind": "ilume_baseline_model",
        "model_kind": "aifc",
        "training_identity": bundle.training_identity,
        "target_statistics": bundle.target_stats.to_dict(),
        "condition_statistics": asdict(bundle.condition_stats),
        "target_columns": list(bundle.task.target_columns),
        "architecture": bundle.architecture,
        "effective_seed": config.seed,
        "model_selection": "final_training_state",
        "final_epoch": epochs,
        "final_valid_raw_mae": history[-1]["valid_raw_mae"],
        "final_model_file": model_file.name,
        "model_state_hash": state_hash,
        "parameter_audit": parameter_audit,
        "input_audit": {"train": bundle.train.audit, "valid": bundle.valid.audit},
        "single_model": True,
        "validation_drives_training": False,
        "test_seen_during_training": False,
        "integrity": integrity,
    }
    atomic_json(root / "checkpoint.json", manifest)
    return {
        "final_epoch": epochs,
        "epochs_ran": len(history),
        "final_valid_raw_mae": history[-1]["valid_raw_mae"],
        "model_selector": "final_training_state",
    }


def _load_state(path: Path) -> Mapping[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def evaluate_aifc_checkpoint(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
    checkpoint_dir: str | Path,
    split: str,
) -> EvaluationResult:
    if split not in {"valid", "test"}:
        raise ValueError("AIFC evaluation split must be valid or test")
    root = Path(checkpoint_dir)
    manifest = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    if (
        manifest.get("model_kind") != "aifc"
        or manifest.get("model_selection") != "final_training_state"
        or manifest.get("final_epoch") != 10
        or manifest.get("effective_seed") != 1000
        or manifest.get("single_model") is not True
    ):
        raise ValueError("Unsupported AIFC checkpoint")
    for filename, expected in manifest["integrity"].items():
        path = root / filename
        if not path.is_file() or path.stat().st_size != expected["size"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"AIFC checkpoint integrity mismatch: {filename}")
    bundle = prepare_aifc_training(config, benchmark, task_id, fold)
    require_compatible_identity(
        bundle.training_identity,
        manifest["training_identity"],
        context="AIFC evaluation checkpoint",
    )
    raw = load_split(bundle.task, split)
    prepared = _prepare_split(bundle.task, raw, bundle.condition_stats, bundle.scheme, {})
    device = torch.device(str(config.training["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("AIFC evaluation requires CUDA")
    _configure_fp32()
    model = build_aifc_model(bundle)
    state = _load_state(root / manifest["final_model_file"])
    if tensor_state_hash("benchmark.aifc-state.v1", state) != manifest["model_state_hash"]:
        raise ValueError("AIFC model state hash mismatch")
    model.load_state_dict(state, strict=True)
    model.to(device)
    predictions = _predict(
        model,
        prepared,
        bundle.target_stats,
        device,
        int(config.training["batch_size"]),
    )
    return EvaluationResult(
        predictions=predictions,
        targets=raw.targets,
        source_rows=raw.source_rows,
        metrics=target_metrics(
            predictions, raw.targets, bundle.task.target_columns, bundle.target_stats.scale
        ),
        target_stats=bundle.target_stats,
        training_identity=bundle.training_identity,
        components=raw.components,
        conditions=raw.conditions,
        audit_rows=raw.audit_rows,
        input_audit=prepared.audit,
    )


def aifc_evaluation_audit(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
    split: str,
) -> dict[str, Any]:
    task = resolve_task(config, benchmark, task_id, fold)
    train = load_split(task, "train")
    stats = ConditionStats.fit(task.condition_columns, train.conditions)
    scheme = FragmentScheme.load(repository_path(config.model["fragment_scheme"]))
    return _prepare_split(task, load_split(task, split), stats, scheme, {}).audit


__all__ = [
    "AIFCTrainingBundle",
    "ConditionStats",
    "PreparedSplit",
    "_prepare_split",
    "aifc_evaluation_audit",
    "aifc_model_views",
    "build_aifc_model",
    "evaluate_aifc_checkpoint",
    "prepare_aifc_training",
    "resolve_aifc_architecture",
    "train_aifc_bundle",
]
