"""AIonopedia adapter for ILUME Stage 3 tasks and evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from common.identity import require_compatible_identity, semantic_identity
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.outputs import repository_path
from common.progress import ProgressReporter

from benchmarks.common.config import BenchmarkConfig, BenchmarkName
from benchmarks.common.data import BenchmarkTask, RawDataset, load_split, resolve_task
from benchmarks.common.engine import EvaluationResult, TargetStats, seed_benchmark
from benchmarks.common.metrics import target_metrics

from .graph import empty_graph, smiles_to_graph


UPSTREAM_REVISION = "17e2f550f91eadcdec39f467c0443f5446d9713c"
OFFICIAL_MODULE_FILES = {
    "GNN_state_dict.pt": "GNN",
    "projector_gnn_state_dict.pt": "projector_gnn",
    "projector_llm_state_dict.pt": "projector_llm",
    "projector_temp_state_dict.pt": "projector_temp",
    "embedding_property_state_dict.pt": "embedding_property",
    "graph_merge_encoder_state_dict.pt": "graph_merge_encoder",
    "decoder1_state_dict.pt": "decoder1",
    "decoder2_state_dict.pt": "decoder2",
}
OFFICIAL_SEGMENTS = (
    "segment_embed_temp",
    "segment_embed_solute",
    "segment_embed_cation",
    "segment_embed_anion",
    "segment_embed_property",
)
EXTRA_CONDITION_COLUMNS = {
    "pressure_kPa": "pressure",
    "frequency_MHz": "frequency",
    "wavelength_nm": "wavelength",
}
AIONOPEDIA_INPUT_CONTRACT = {
    "canonical_identity": "ilume_isomeric_smiles",
    "target_in_prompt": False,
    "graph_atom_features": 35,
    "graph_edge_features": 11,
    "explicit_hydrogen_expansion": False,
    "temperature_transform": "temperature_K/1000",
    "pressure_transform": "train_only_sample_zscore",
    "frequency_transform": "frequency_MHz/1000",
    "wavelength_transform": "wavelength_nm/1000",
    "graph_condition_order": "stage3_registry_order",
    "topology_ids": {
        "solute_solvent_conditions": 0,
        "solute_il_conditions": 1,
        "il_conditions": 2,
        "il": 3,
    },
}


@dataclass(frozen=True)
class SampleStats:
    mean: float
    scale: float
    constant: bool

    @classmethod
    def fit(cls, values: np.ndarray, *, allow_constant: bool) -> "SampleStats":
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(array) < 2 or not np.isfinite(array).all():
            raise ValueError("AIonopedia sample statistics require at least two finite train rows")
        mean = float(array.mean())
        scale = float(array.std(ddof=1))
        constant = not math.isfinite(scale) or scale <= 0
        if constant and not allow_constant:
            raise ValueError("AIonopedia train target has zero or invalid sample variance")
        return cls(mean=mean, scale=1.0 if constant else scale, constant=constant)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values) - self.mean) / self.scale).astype(np.float32)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * self.scale + self.mean


@dataclass(frozen=True)
class PreparedSplit:
    raw: RawDataset
    prompts: tuple[str, ...]
    topology: int
    temperature: np.ndarray
    extra_conditions: dict[str, np.ndarray]
    active_conditions: tuple[str, ...]
    graph_roles: tuple[tuple[str, str, str], ...]
    audit: dict[str, Any]


@dataclass
class AIonopediaTrainingBundle:
    task: BenchmarkTask
    train: PreparedSplit
    valid: PreparedSplit
    target_stats: SampleStats
    pressure_stats: SampleStats | None
    source_hashes: dict[str, Any]
    training_identity: dict[str, Any]
    assets: dict[str, Any]
    tokenizer: Any
    graph_cache: dict[str, Data]


def _number(value: float) -> str:
    return format(float(value), ".12g")


def _topology(task: BenchmarkTask) -> tuple[int, str]:
    conditions = bool(task.condition_columns)
    if task.slots == ("cation", "anion"):
        return (2, "il_conditions") if conditions else (3, "il")
    if task.slots == ("cation", "anion", "solute") and conditions:
        return 1, "solute_il_conditions"
    if task.slots == ("solute", "solvent") and conditions:
        return 0, "solute_solvent_conditions"
    raise ValueError(
        f"AIonopedia does not support registry topology {task.slots} "
        f"with conditions {task.condition_columns}"
    )


def _condition_phrases(task: BenchmarkTask, row: np.ndarray) -> list[str]:
    phrases = []
    units = {
        "temperature_K": ("temperature", "K"),
        "pressure_kPa": ("pressure", "kPa"),
        "frequency_MHz": ("frequency", "MHz"),
        "wavelength_nm": ("wavelength", "nm"),
    }
    for column, value in zip(task.condition_columns, row, strict=True):
        if column not in units:
            raise ValueError(f"AIonopedia has no condition representation for {column}")
        label, unit = units[column]
        phrases.append(f"{label} {_number(float(value))}{unit}")
    return phrases


def _prepare_split(
    task: BenchmarkTask,
    raw: RawDataset,
    pressure_stats: SampleStats | None,
) -> PreparedSplit:
    topology, topology_name = _topology(task)
    prompts: list[str] = []
    graph_roles: list[tuple[str, str, str]] = []
    temperature = np.zeros((len(raw),), dtype=np.float32)
    extras: dict[str, np.ndarray] = {}
    active = tuple(
        EXTRA_CONDITION_COLUMNS[column]
        for column in task.condition_columns
        if column in EXTRA_CONDITION_COLUMNS
    )
    for name in active:
        extras[name] = np.zeros((len(raw),), dtype=np.float32)
    for index, (components, conditions) in enumerate(
        zip(raw.components, raw.conditions, strict=True)
    ):
        component = dict(zip(task.slots, components, strict=True))
        condition = dict(zip(task.condition_columns, conditions, strict=True))
        phrases = _condition_phrases(task, conditions)
        if topology == 0:
            prompts.append(
                " ".join(
                    ["solute", component["solute"], *phrases, "solvent", component["solvent"]]
                )
            )
            graph_roles.append((component["solute"], component["solvent"], ""))
        elif topology == 1:
            prompts.append(
                " ".join(
                    [
                        "solute", component["solute"], *phrases,
                        "cation", component["cation"], "anion", component["anion"],
                    ]
                )
            )
            graph_roles.append((component["solute"], component["cation"], component["anion"]))
        else:
            prompts.append(
                " ".join(
                    [*phrases, "cation", component["cation"], "anion", component["anion"]]
                )
            )
            graph_roles.append(("", component["cation"], component["anion"]))
        if "temperature_K" in condition:
            temperature[index] = float(condition["temperature_K"]) / 1000.0
        if "pressure_kPa" in condition:
            if pressure_stats is None:
                raise ValueError("AIonopedia pressure task lacks train-only pressure statistics")
            extras["pressure"][index] = pressure_stats.normalize(
                np.asarray([condition["pressure_kPa"]])
            )[0]
        if "frequency_MHz" in condition:
            extras["frequency"][index] = float(condition["frequency_MHz"]) / 1000.0
        if "wavelength_nm" in condition:
            extras["wavelength"][index] = float(condition["wavelength_nm"]) / 1000.0
    prompt_hash = hashlib.sha256("\n".join(prompts).encode()).hexdigest()
    return PreparedSplit(
        raw=raw,
        prompts=tuple(prompts),
        topology=topology,
        temperature=temperature,
        extra_conditions=extras,
        active_conditions=active,
        graph_roles=tuple(graph_roles),
        audit={
            "topology": topology_name,
            "topology_id": topology,
            "condition_columns": list(task.condition_columns),
            "active_graph_conditions": list(active),
            "prompt_sha256": prompt_hash,
            "rows": len(raw),
            "pressure_constant_in_train": (
                pressure_stats.constant if pressure_stats is not None else None
            ),
        },
    )


def _load_tokenizer(config: BenchmarkConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        repository_path(config.model["base_snapshot"]),
        local_files_only=True,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_aionopedia_training(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
) -> AIonopediaTrainingBundle:
    from benchmarks.common.environment import aionopedia_asset_snapshot

    task = resolve_task(config, benchmark, task_id, fold)
    train_raw, valid_raw = load_split(task, "train"), load_split(task, "valid")
    target_stats = SampleStats.fit(train_raw.targets[:, 0], allow_constant=False)
    pressure_stats = None
    if "pressure_kPa" in task.condition_columns:
        column = task.condition_columns.index("pressure_kPa")
        pressure_stats = SampleStats.fit(train_raw.conditions[:, column], allow_constant=True)
    train = _prepare_split(task, train_raw, pressure_stats)
    valid = _prepare_split(task, valid_raw, pressure_stats)
    source_hashes = {
        "train": [sha256_file(path) for path in task.train_paths],
        "valid": [sha256_file(path) for path in task.valid_paths],
    }
    assets = aionopedia_asset_snapshot(config)
    effective_seed = config.seed + int(fold or 1) - 1
    identity = semantic_identity(
        "benchmark.aionopedia-training.v1",
        {
            "benchmark_model": config.name,
            "task": task.to_dict(),
            "source_hashes": source_hashes,
            "target_statistics": asdict(target_stats),
            "pressure_statistics": asdict(pressure_stats) if pressure_stats else None,
            "input_contract": AIONOPEDIA_INPUT_CONTRACT,
            "model": config.model,
            "training": config.training,
            "effective_seed": effective_seed,
            "assets": assets,
        },
    )
    graph_cache: dict[str, Data] = {}
    for split in (train, valid):
        for roles in split.graph_roles:
            for smiles in roles:
                if smiles and smiles not in graph_cache:
                    graph_cache[smiles] = smiles_to_graph(smiles)
    return AIonopediaTrainingBundle(
        task=task,
        train=train,
        valid=valid,
        target_stats=target_stats,
        pressure_stats=pressure_stats,
        source_hashes=source_hashes,
        training_identity=identity,
        assets=assets,
        tokenizer=_load_tokenizer(config),
        graph_cache=graph_cache,
    )


def _torch_load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=True)


def _build_model(config: BenchmarkConfig, *, device: torch.device) -> Any:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from .model import MultiModalRegressor

    base = AutoModelForCausalLM.from_pretrained(
        repository_path(config.model["base_snapshot"]),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    if int(base.config.hidden_size) != 1024:
        raise ValueError("AIonopedia released checkpoint requires Qwen hidden size 1024")
    llm = PeftModel.from_pretrained(
        base,
        repository_path(config.model["pretrained_snapshot"]),
        is_trainable=True,
    )
    for peft_config in llm.peft_config.values():
        peft_config.base_model_name_or_path = config.model["base_repository"]
    model = MultiModalRegressor(llm, llm_dim=1024)
    pretrained = repository_path(config.model["pretrained_snapshot"])
    for filename, attribute in OFFICIAL_MODULE_FILES.items():
        getattr(model, attribute).load_state_dict(
            _torch_load(pretrained / filename), strict=True
        )
    segments = _torch_load(pretrained / "segment_embeddings.pt")
    for name in OFFICIAL_SEGMENTS:
        if name not in segments:
            raise KeyError(f"AIonopedia segment checkpoint lacks {name}")
        getattr(model, name).data.copy_(segments[name])
    for name, parameter in model.llm.named_parameters():
        parameter.requires_grad = "lora" in name.lower()
    return model.to(device)


def _collator(bundle: AIonopediaTrainingBundle, split: PreparedSplit):
    tokenizer = bundle.tokenizer

    def collate(indices: Sequence[int]) -> dict[str, Any]:
        prompts = [split.prompts[index] for index in indices]
        text = tokenizer(prompts, padding="longest", truncation=False, return_tensors="pt")
        roles = [split.graph_roles[index] for index in indices]

        def graphs(position: int) -> Batch:
            return Batch.from_data_list(
                [
                    bundle.graph_cache[role[position]]
                    if role[position]
                    else empty_graph()
                    for role in roles
                ]
            )

        return {
            "input_ids": text["input_ids"],
            "attention_mask": text["attention_mask"],
            "solute_graph": graphs(0),
            "cation_graph": graphs(1),
            "anion_graph": graphs(2),
            "temperature": torch.from_numpy(split.temperature[list(indices)]),
            "topology": torch.full((len(indices),), split.topology, dtype=torch.long),
            "extra_conditions": {
                name: torch.from_numpy(values[list(indices)])
                for name, values in split.extra_conditions.items()
            },
            "active_conditions": split.active_conditions,
            "labels": torch.from_numpy(
                bundle.target_stats.normalize(split.raw.targets[list(indices), 0])
            ),
        }

    return collate


def _loader(
    bundle: AIonopediaTrainingBundle,
    split: PreparedSplit,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        list(range(len(split.raw))),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=_collator(bundle, split),
        num_workers=0,
    )


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (
            {name: value.to(device, non_blocking=True) for name, value in item.items()}
            if key == "extra_conditions"
            else item.to(device, non_blocking=True)
            if isinstance(item, torch.Tensor)
            else item
        )
        for key, item in batch.items()
        if key != "labels"
    }


def _predict(
    model: Any,
    loader: DataLoader,
    target_stats: SampleStats,
    device: torch.device,
) -> np.ndarray:
    values = []
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            values.append(model(**_move(batch, device)).float().cpu().numpy())
    normalized = np.concatenate(values).reshape(-1) if values else np.empty((0,))
    return target_stats.denormalize(normalized).reshape(-1, 1)


def _scheduled_factor(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _checkpoint_payload(model: Any) -> dict[str, Any]:
    from peft import get_peft_model_state_dict

    modules = {
        attribute: {
            name: value.detach().cpu().clone()
            for name, value in getattr(model, attribute).state_dict().items()
        }
        for attribute in (*OFFICIAL_MODULE_FILES.values(), "condition_projectors", "fc_out")
    }
    return {
        "lora": {
            name: value.detach().cpu().clone()
            for name, value in get_peft_model_state_dict(model.llm).items()
        },
        "modules": modules,
        "segments": {
            name: getattr(model, name).detach().cpu().clone()
            for name in (*OFFICIAL_SEGMENTS,)
        },
        "condition_segments": {
            name: value.detach().cpu().clone()
            for name, value in model.condition_segments.items()
        },
    }


def _restore_checkpoint(model: Any, payload: Mapping[str, Any]) -> None:
    from peft import set_peft_model_state_dict

    result = set_peft_model_state_dict(model.llm, payload["lora"])
    if getattr(result, "unexpected_keys", None):
        raise ValueError("AIonopedia LoRA checkpoint has unexpected keys")
    for attribute, state in payload["modules"].items():
        getattr(model, attribute).load_state_dict(state, strict=True)
    for name, value in payload["segments"].items():
        getattr(model, name).data.copy_(value)
    for name, value in payload["condition_segments"].items():
        model.condition_segments[name].data.copy_(value)


def train_aionopedia_bundle(
    config: BenchmarkConfig,
    bundle: AIonopediaTrainingBundle,
    output_dir: str | Path,
    *,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(config.training["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("AIonopedia requires CUDA; no silent CPU fallback")
    effective_seed = config.seed + int(bundle.task.fold or 1) - 1
    seed_benchmark(effective_seed)
    model = _build_model(config, device=device)
    head, decoders, other = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("fc_out."):
            head.append(parameter)
        elif name.startswith(("decoder1.", "decoder2.")):
            decoders.append(parameter)
        else:
            other.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": head, "lr": float(config.training["head_learning_rate"])},
            {"params": decoders, "lr": float(config.training["decoder_learning_rate"])},
            {"params": other, "lr": float(config.training["other_learning_rate"])},
        ],
        weight_decay=float(config.training["weight_decay"]),
    )
    batch_size = int(config.training["batch_size"])
    epochs = int(config.training["max_epochs"])
    steps_per_epoch = math.ceil(len(bundle.train.raw) / batch_size)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(config.training["warmup_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _scheduled_factor(
            step, total_steps=total_steps, warmup_steps=warmup_steps
        ),
    )
    valid_loader = _loader(
        bundle, bundle.valid, batch_size=batch_size, shuffle=False, seed=effective_seed
    )
    history = []
    integrity = {}
    progress = (reporter or ProgressReporter()).bar(
        total=epochs,
        desc=f"AIonopedia {bundle.task.task_id} fold{bundle.task.fold}",
        unit="epoch",
    )
    try:
        for epoch in range(1, epochs + 1):
            train_loader = _loader(
                bundle,
                bundle.train,
                batch_size=batch_size,
                shuffle=True,
                seed=effective_seed + epoch - 1,
            )
            model.train()
            loss_sum = 0.0
            for batch in train_loader:
                labels = batch.pop("labels").to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predictions = model(**_move(batch, device)).view(-1)
                    loss = torch.nn.functional.mse_loss(predictions, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("AIonopedia training loss is non-finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for group in optimizer.param_groups for parameter in group["params"]],
                    float(config.training["max_grad_norm"]),
                )
                optimizer.step()
                scheduler.step()
                loss_sum += float(loss.detach().cpu()) * len(labels)
            raw_predictions = _predict(model, valid_loader, bundle.target_stats, device)
            valid_mae = float(np.abs(raw_predictions[:, 0] - bundle.valid.raw.targets[:, 0]).mean())
            row = {
                "epoch": epoch,
                "train_normalized_mse": loss_sum / len(bundle.train.raw),
                "valid_raw_mae": valid_mae,
                "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            }
            history.append(row)
            snapshot = root / f"checkpoint_epoch_{epoch:05d}.pt"
            atomic_torch_save(snapshot, _checkpoint_payload(model))
            integrity[snapshot.name] = {
                "sha256": sha256_file(snapshot),
                "size": snapshot.stat().st_size,
            }
            progress.set_postfix({"train_mse": f"{row['train_normalized_mse']:.4f}", "val_mae": f"{valid_mae:.4f}"})
            progress.update(1)
    finally:
        progress.close()
    atomic_json(root / "training_history.json", history)
    integrity["training_history.json"] = {
        "sha256": sha256_file(root / "training_history.json"),
        "size": (root / "training_history.json").stat().st_size,
    }
    manifest = {
        "format_version": 2,
        "kind": "ilume_baseline_model",
        "model_kind": "aionopedia",
        "training_identity": bundle.training_identity,
        "target_statistics": asdict(bundle.target_stats),
        "pressure_statistics": asdict(bundle.pressure_stats) if bundle.pressure_stats else None,
        "target_columns": list(bundle.task.target_columns),
        "effective_seed": effective_seed,
        "model_selection": "final_training_state",
        "final_epoch": epochs,
        "final_valid_raw_mae": float(history[-1]["valid_raw_mae"]),
        "final_model_file": f"checkpoint_epoch_{epochs:05d}.pt",
        "epoch_snapshots_resumable": False,
        "pretrained_modules": sorted(OFFICIAL_MODULE_FILES),
        "randomly_initialized_modules": [
            "fc_out", "condition_projectors.pressure", "condition_projectors.frequency",
            "condition_projectors.wavelength", "condition_segments.pressure",
            "condition_segments.frequency", "condition_segments.wavelength",
        ],
        "integrity": integrity,
    }
    atomic_json(root / "checkpoint.json", manifest)
    return {
        "final_epoch": epochs,
        "epochs_ran": len(history),
        "final_valid_raw_mae": float(history[-1]["valid_raw_mae"]),
        "model_selector": "final_training_state",
    }


def evaluate_aionopedia_checkpoint(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
    checkpoint_dir: str | Path,
    split: str,
) -> EvaluationResult:
    if split not in {"valid", "test"}:
        raise ValueError("AIonopedia evaluation split must be valid or test")
    root = Path(checkpoint_dir)
    manifest = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    if manifest.get("model_kind") != "aionopedia" or manifest.get("final_epoch") != 10:
        raise ValueError("Unsupported AIonopedia checkpoint")
    for filename, expected in manifest["integrity"].items():
        artifact = root / filename
        if not artifact.is_file() or artifact.stat().st_size != int(expected["size"]) or sha256_file(artifact) != expected["sha256"]:
            raise ValueError(f"AIonopedia checkpoint integrity mismatch: {filename}")
    bundle = prepare_aionopedia_training(config, benchmark, task_id, fold)
    require_compatible_identity(
        bundle.training_identity,
        manifest["training_identity"],
        context="AIonopedia evaluation checkpoint",
    )
    raw = load_split(bundle.task, split)
    prepared = _prepare_split(bundle.task, raw, bundle.pressure_stats)
    for roles in prepared.graph_roles:
        for smiles in roles:
            if smiles and smiles not in bundle.graph_cache:
                bundle.graph_cache[smiles] = smiles_to_graph(smiles)
    device = torch.device(str(config.training["device"]))
    model = _build_model(config, device=device)
    _restore_checkpoint(
        model,
        _torch_load(root / str(manifest["final_model_file"])),
    )
    loader = _loader(
        bundle,
        prepared,
        batch_size=int(config.training["batch_size"]),
        shuffle=False,
        seed=int(manifest["effective_seed"]),
    )
    predictions = _predict(model, loader, bundle.target_stats, device)
    stats = TargetStats((bundle.target_stats.mean,), (bundle.target_stats.scale,))
    return EvaluationResult(
        predictions=predictions,
        targets=raw.targets,
        source_rows=raw.source_rows,
        metrics=target_metrics(
            predictions, raw.targets, bundle.task.target_columns, stats.scale
        ),
        target_stats=stats,
        training_identity=bundle.training_identity,
        components=raw.components,
        conditions=raw.conditions,
        audit_rows=raw.audit_rows,
        input_audit=prepared.audit,
    )


def aionopedia_evaluation_audit(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
    split: str,
) -> dict[str, Any]:
    task = resolve_task(config, benchmark, task_id, fold)
    train = load_split(task, "train")
    pressure_stats = None
    if "pressure_kPa" in task.condition_columns:
        column = task.condition_columns.index("pressure_kPa")
        pressure_stats = SampleStats.fit(train.conditions[:, column], allow_constant=True)
    return _prepare_split(task, load_split(task, split), pressure_stats).audit


__all__ = [
    "AIONOPEDIA_INPUT_CONTRACT",
    "AIonopediaTrainingBundle",
    "SampleStats",
    "_prepare_split",
    "_scheduled_factor",
    "aionopedia_evaluation_audit",
    "evaluate_aionopedia_checkpoint",
    "prepare_aionopedia_training",
    "train_aionopedia_bundle",
]
