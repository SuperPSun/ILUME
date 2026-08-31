from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from common.identity import require_compatible_identity, semantic_identity, tensor_state_hash
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.outputs import repository_path
from common.progress import ProgressReporter
from common.reporting import role_mae_diagnostics
from stage2.registry import ORBITAL_TASK_TARGETS

from benchmarks.common.config import BenchmarkConfig, BenchmarkName
from benchmarks.common.data import BenchmarkTask, RawDataset, load_split, resolve_task
from benchmarks.common.engine import EvaluationResult, TargetStats, seed_benchmark
from benchmarks.common.environment import ilbert_asset_snapshot
from benchmarks.common.metrics import target_metrics


ILBERT_INPUT_CONTRACT = {
    "canonical_identity": "ilume_isomeric_smiles",
    "tokenizer": "official_ais_atomwise",
    "native_ionic_liquid": "cation_dot_anion",
    "multiview_forward": "merged_sequence_view_forward",
    "max_length": 100,
    "truncation": True,
    "padding": "max_length",
    "token_length_includes_special_tokens": True,
    "input_cache": "unique_sequence_memory_token_cache",
    "conditions": "raw_physical_units_in_registry_order",
}
TokenCache = dict[str, tuple[torch.Tensor, int]]


@dataclass(frozen=True)
class PreparedSplit:
    raw: RawDataset
    model_sequences: tuple[tuple[str, ...], ...]
    raw_conditions: np.ndarray
    audit: dict[str, Any]

    @property
    def view_count(self) -> int:
        return len(self.model_sequences[0]) if self.model_sequences else 0


@dataclass
class ILBERTTrainingBundle:
    task: BenchmarkTask
    train: PreparedSplit
    valid: PreparedSplit
    target_stats: TargetStats
    source_hashes: dict[str, Any]
    training_identity: dict[str, Any]
    assets: dict[str, Any]
    token_cache: TokenCache
    pad_token_id: int


class SharedILBERTRegressor(torch.nn.Module):
    def __init__(
        self,
        roberta: torch.nn.Module,
        textcnn: torch.nn.Module,
        predictor: torch.nn.Module,
        *,
        view_count: int,
        condition_dim: int,
        hidden_dim: int,
        load_audit: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.roberta = roberta
        self.textcnn = textcnn
        self.predictor = predictor
        self.view_count = int(view_count)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.load_audit = dict(load_audit)

    def forward(self, input_ids: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        batch_size = int(conditions.shape[0])
        if int(input_ids.shape[0]) != self.view_count * batch_size:
            raise ValueError("ILBERT merged sequence batch differs from task topology")
        attention_mask = (input_ids != 0).long()
        states = self.roberta(
            input_ids=input_ids.long(), attention_mask=attention_mask
        ).last_hidden_state
        pooled = self.textcnn(states.permute(1, 0, 2))
        ordered = pooled.reshape(self.view_count, batch_size, self.hidden_dim).transpose(0, 1)
        representation = torch.cat(
            (ordered.reshape(batch_size, -1), conditions.float()), dim=1
        )
        return self.predictor(representation)


class EpochBatchSampler(torch.utils.data.Sampler[list[int]]):
    def __init__(self, row_count: int, *, batch_size: int, seed: int) -> None:
        self.row_count = int(row_count)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(self.row_count / self.batch_size)

    def __iter__(self) -> Any:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.row_count, generator=generator).tolist()
        return iter(
            order[start : start + self.batch_size]
            for start in range(0, len(order), self.batch_size)
        )


def _upstream_paths(config: BenchmarkConfig) -> tuple[Path, Path, Path, Path]:
    checkout = repository_path(str(config.model["checkout"]))
    return (
        checkout / "ILBERT" / "model.py",
        checkout / "ILBERT" / "ILtokenizer.py",
        checkout / "ILBERT" / "merged_vocab.txt",
        repository_path(str(config.model["pretrained_checkpoint"])),
    )


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot dynamically load pinned ILBERT source: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tokenizer(config: BenchmarkConfig) -> Any:
    _, tokenizer_source, vocab, _ = _upstream_paths(config)
    tokenizer_class = _load_module(
        tokenizer_source, "ilume_pinned_ilbert_tokenizer_adapter"
    ).SMILES_Atomwise_Tokenizer
    return tokenizer_class(str(vocab))


def ilbert_model_sequences(
    task: BenchmarkTask, components: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    slots = tuple(task.slots)
    values = tuple(components)
    if len(values) != len(slots):
        raise ValueError("ILBERT row component count differs from registry topology")
    if slots == ("cation", "anion"):
        return (f"{values[0]}.{values[1]}",), ("ionic_liquid",)
    if slots == ("cation", "anion", "solute"):
        return (f"{values[0]}.{values[1]}", values[2]), ("ionic_liquid", "solute")
    if slots == ("solute", "solvent"):
        return values, slots
    if len(slots) == 1 and slots[0] in {"SMILES", "smiles", "molecule"}:
        return values, (slots[0],)
    raise ValueError(f"ILBERT does not support registry topology: {slots}")


def _view_source_slots(task: BenchmarkTask, view: str) -> tuple[str, ...]:
    if view == "ionic_liquid":
        return tuple(task.slots[:2])
    if view in task.slots:
        return (view,)
    raise ValueError(f"ILBERT view does not map to registry slots: {view}")


def _populate_token_cache(
    tokenizer: Any,
    sequences: Sequence[str],
    token_cache: TokenCache,
    *,
    max_length: int,
) -> None:
    for sequence in sorted(set(sequences) - set(token_cache)):
        full = tokenizer.encode(sequence, add_special_tokens=True, truncation=False)
        encoded = tokenizer.encode(
            sequence,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        ids = torch.as_tensor(encoded, dtype=torch.long)
        if ids.ndim != 1 or len(ids) != max_length:
            raise ValueError("ILBERT tokenizer violated fixed-length padding contract")
        token_cache[sequence] = (ids, len(full))


def _token_cache_audit(token_cache: TokenCache) -> dict[str, int | str]:
    lengths = [length for _, length in token_cache.values()]
    return {
        "contract": "unique_sequence_memory_token_cache",
        "unique_model_inputs": len(lengths),
        "total_pre_truncation_tokens": sum(lengths),
        "maximum_pre_truncation_token_length": max(lengths, default=0),
    }


def _prepare_split(
    raw: RawDataset,
    task: BenchmarkTask,
    split: str,
    tokenizer: Any,
    token_cache: TokenCache,
    *,
    max_length: int,
) -> PreparedSplit:
    model_sequences: list[tuple[str, ...]] = []
    view_names: tuple[str, ...] | None = None
    for components in raw.components:
        sequences, names = ilbert_model_sequences(task, components)
        if view_names is not None and names != view_names:
            raise ValueError("ILBERT sequence views changed within one task")
        view_names = names
        model_sequences.append(sequences)
    flat = tuple(value for row in model_sequences for value in row)
    _populate_token_cache(tokenizer, flat, token_cache, max_length=max_length)
    truncated: list[dict[str, Any]] = []
    truncated_rows: set[str] = set()
    for source_row, sequences in zip(raw.source_rows, model_sequences, strict=True):
        for view, sequence in zip(view_names or (), sequences, strict=True):
            length = token_cache[sequence][1]
            if length > max_length:
                truncated_rows.add(source_row)
                truncated.append(
                    {
                        "source_row": source_row,
                        "view": view,
                        "source_slots": list(_view_source_slots(task, view)),
                        "sequence": sequence,
                        "pre_truncation_token_length": length,
                    }
                )
    conditions = raw.conditions.astype(np.float32, copy=True)
    if conditions.ndim != 2 or not np.isfinite(conditions).all():
        raise ValueError("ILBERT raw numeric conditions must be a finite matrix")
    lengths = [token_cache[value][1] for value in set(flat)]
    audit = {
        "contract": ILBERT_INPUT_CONTRACT,
        "task": task.task_id,
        "split": split,
        "view_order": list(view_names or ()),
        "view_source_slots": {
            view: list(_view_source_slots(task, view)) for view in view_names or ()
        },
        "condition_columns": list(task.condition_columns),
        "total_rows": len(raw),
        "unique_sequences": len(set(flat)),
        "unique_sequence_tokens": sum(lengths),
        "maximum_pre_truncation_token_length": max(lengths, default=0),
        "truncated_sequence_count": len(truncated),
        "truncated_row_count": len(truncated_rows),
        "truncated_rows": sorted(truncated_rows),
        "truncated_sequences": truncated,
    }
    return PreparedSplit(raw, tuple(model_sequences), conditions, audit)


def _lock_sha(config: BenchmarkConfig) -> str:
    if config.environment is None:
        raise ValueError("ILBERT environment contract is missing")
    return sha256_file(repository_path(config.environment.lock))


def prepare_ilbert_training(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
) -> ILBERTTrainingBundle:
    if config.name != "ilbert":
        raise ValueError("ILBERT adapter requires name=ilbert")
    task = resolve_task(config, benchmark, task_id, fold)
    if len(task.target_columns) != 1:
        raise ValueError("ILBERT baseline requires one scalar target per task")
    assets = ilbert_asset_snapshot(config)
    tokenizer = _tokenizer(config)
    if int(tokenizer.pad_token_id) != 0:
        raise ValueError("ILBERT tokenizer pad token ID must be zero")
    token_cache: TokenCache = {}
    train = _prepare_split(
        load_split(task, "train"), task, "train", tokenizer, token_cache,
        max_length=int(config.model["max_length"]),
    )
    valid = _prepare_split(
        load_split(task, "valid"), task, "valid", tokenizer, token_cache,
        max_length=int(config.model["max_length"]),
    )
    if not len(train.raw):
        raise ValueError("ILBERT requires non-empty training rows")
    if not len(valid.raw):
        raise ValueError("ILBERT requires non-empty validation rows")
    if train.view_count != valid.view_count:
        raise ValueError("ILBERT train and valid sequence topology differ")
    target_stats = TargetStats.fit(train.raw.targets)
    source_hashes = {
        "train": [sha256_file(path) for path in task.train_paths],
        "valid": [sha256_file(path) for path in task.valid_paths],
    }
    identity = semantic_identity(
        "benchmark.training.v1",
        {
            "benchmark_model": "ilbert",
            "domain": task.benchmark,
            "task_id": task.task_id,
            "fold": task.fold,
            "registry": task.registry_payload,
            "component_order": list(task.slots),
            "sequence_view_order": train.audit["view_order"],
            "source_hashes": source_hashes,
            "input_contract": ILBERT_INPUT_CONTRACT,
            "token_cache_audit": _token_cache_audit(token_cache),
            "train_input_audit": train.audit,
            "valid_input_audit": valid.audit,
            "condition_contract": {
                "columns": list(task.condition_columns),
                "transform": "raw_physical_units",
            },
            "target_statistics": target_stats.to_dict(),
            "upstream_assets": assets,
            "model": config.model,
            "training": config.training,
            "seed": config.seed,
            "environment_lock_sha256": _lock_sha(config),
        },
    )
    return ILBERTTrainingBundle(
        task=task,
        train=train,
        valid=valid,
        target_stats=target_stats,
        source_hashes=source_hashes,
        training_identity=identity,
        assets=assets,
        token_cache=token_cache,
        pad_token_id=int(tokenizer.pad_token_id),
    )


def build_ilbert_model(
    config: BenchmarkConfig, bundle: ILBERTTrainingBundle
) -> SharedILBERTRegressor:
    model_source, _, _, checkpoint_path = _upstream_paths(config)
    official_class = _load_module(
        model_source, "ilume_pinned_ilbert_model_adapter"
    ).ILBERT
    official = official_class(
        int(config.model["vocab_size"]),
        int(config.model["hidden_dim"]),
        int(config.model["heads"]),
        int(config.model["ffn_hidden_dim"]),
        int(config.model["layers"]),
        float(config.model["dropout"]),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("ILBERT pretrained checkpoint is not a state dictionary")
    incompatible = official.load_state_dict(checkpoint, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    bad_missing = [
        key
        for key in missing
        if not key.startswith(("CNN.", "pred_head.", "roberta.pooler."))
    ]
    bad_unexpected = [key for key in unexpected if not key.startswith("lm_head.")]
    encoder_keys = {
        key
        for key in official.state_dict()
        if key.startswith("roberta.") and not key.startswith("roberta.pooler.")
    }
    checkpoint_encoder = {key for key in checkpoint if key.startswith("roberta.")}
    unloaded_encoder = sorted(encoder_keys - checkpoint_encoder)
    if bad_missing or bad_unexpected or unloaded_encoder:
        raise RuntimeError(
            "ILBERT pretrained checkpoint load contract mismatch: "
            + json.dumps(
                {
                    "bad_missing": bad_missing,
                    "bad_unexpected": bad_unexpected,
                    "unloaded_encoder": unloaded_encoder,
                },
                sort_keys=True,
            )
        )
    view_count = bundle.train.view_count
    condition_dim = int(bundle.train.raw_conditions.shape[1])
    hidden_dim = int(config.model["hidden_dim"])
    predictor_input = view_count * hidden_dim + condition_dim
    if predictor_input != int(official.pred_head[0].in_features):
        official.pred_head[0] = torch.nn.Linear(predictor_input, hidden_dim // 2)
    load_audit = {
        "loaded_encoder_parameter_keys": len(encoder_keys),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "allowed_missing_prefixes": ["CNN.", "pred_head.", "roberta.pooler."],
        "allowed_unexpected_prefixes": ["lm_head."],
    }
    return SharedILBERTRegressor(
        official.roberta,
        official.CNN,
        official.pred_head,
        view_count=view_count,
        condition_dim=condition_dim,
        hidden_dim=hidden_dim,
        load_audit=load_audit,
    )


def _collate(
    prepared: PreparedSplit, token_cache: TokenCache, target_stats: TargetStats
) -> Any:
    def collate(indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = torch.stack(
            [
                token_cache[prepared.model_sequences[index][view]][0]
                for view in range(prepared.view_count)
                for index in indices
            ]
        )
        selected = np.asarray(indices, dtype=np.int64)
        return (
            input_ids,
            torch.from_numpy(prepared.raw_conditions[selected]),
            torch.from_numpy(target_stats.normalize(prepared.raw.targets[selected])),
        )

    return collate


def _data_loader(
    config: BenchmarkConfig,
    prepared: PreparedSplit,
    token_cache: TokenCache,
    target_stats: TargetStats,
    *,
    batch_sampler: EpochBatchSampler | None = None,
) -> torch.utils.data.DataLoader:
    common = {
        "num_workers": int(config.runtime["num_workers"]),
        "pin_memory": bool(config.runtime["pin_memory"]),
        "persistent_workers": bool(config.runtime["persistent_workers"]),
        "prefetch_factor": int(config.runtime["prefetch_factor"]),
        "collate_fn": _collate(prepared, token_cache, target_stats),
    }
    if batch_sampler is not None:
        return torch.utils.data.DataLoader(
            range(len(prepared.raw)), batch_sampler=batch_sampler, **common
        )
    return torch.utils.data.DataLoader(
        range(len(prepared.raw)),
        batch_size=int(config.training["batch_size"]),
        shuffle=False,
        **common,
    )


def _configure_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _predict_normalized(
    model: SharedILBERTRegressor,
    loader: torch.utils.data.DataLoader,
    row_count: int,
    *,
    device: torch.device,
    non_blocking: bool,
) -> np.ndarray:
    if not row_count:
        return np.empty((0, 1), dtype=np.float32)
    values: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for input_ids, conditions, _ in loader:
            prediction = model(
                input_ids.to(device, non_blocking=non_blocking),
                conditions.to(device, non_blocking=non_blocking),
            )
            if not torch.isfinite(prediction).all():
                raise RuntimeError("ILBERT evaluation produced non-finite predictions")
            values.append(prediction.float().cpu())
    return torch.cat(values).numpy()


def train_ilbert_bundle(
    config: BenchmarkConfig,
    bundle: ILBERTTrainingBundle,
    output_dir: str | Path,
    *,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seed_benchmark(config.seed)
    _configure_tf32(bool(config.training["tf32"]))
    device = torch.device(str(config.training["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ILBERT requires CUDA; no silent CPU fallback")
    model = build_ilbert_model(config, bundle).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.training["learning_rate"]),
        weight_decay=float(config.training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=int(config.training["scheduler_patience"]),
        factor=float(config.training["scheduler_factor"]),
        min_lr=float(config.training["minimum_learning_rate"]),
    )
    sampler = EpochBatchSampler(
        len(bundle.train.raw),
        batch_size=int(config.training["batch_size"]),
        seed=config.seed,
    )
    train_loader = _data_loader(
        config,
        bundle.train,
        bundle.token_cache,
        bundle.target_stats,
        batch_sampler=sampler,
    )
    valid_loader = _data_loader(
        config, bundle.valid, bundle.token_cache, bundle.target_stats
    )
    non_blocking = bool(config.runtime["non_blocking_transfer"])
    best_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    max_epochs = int(config.training["max_epochs"])
    progress = (reporter or ProgressReporter()).bar(
        total=max_epochs, desc=f"ILBERT {bundle.task.task_id}", unit="epoch"
    )
    try:
        for epoch in range(1, max_epochs + 1):
            sampler.set_epoch(epoch - 1)
            model.train()
            loss_sum = 0.0
            seen = 0
            for input_ids, conditions, targets in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(
                    input_ids.to(device, non_blocking=non_blocking),
                    conditions.to(device, non_blocking=non_blocking),
                )
                loss = torch.nn.functional.mse_loss(
                    prediction, targets.to(device, non_blocking=non_blocking)
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("ILBERT training produced a non-finite loss")
                loss.backward()
                optimizer.step()
                size = len(targets)
                loss_sum += float(loss.detach().cpu()) * size
                seen += size
            normalized = _predict_normalized(
                model,
                valid_loader,
                len(bundle.valid.raw),
                device=device,
                non_blocking=non_blocking,
            )
            raw = bundle.target_stats.denormalize(normalized)
            errors = raw[:, 0] - bundle.valid.raw.targets[:, 0]
            raw_mae = float(np.abs(errors).mean())
            raw_rmse = float(np.sqrt(np.square(errors).mean()))
            normalized_mae = float(
                np.abs(
                    normalized - bundle.target_stats.normalize(bundle.valid.raw.targets)
                ).mean()
            )
            scheduler.step(raw_rmse)
            history.append(
                {
                    "epoch": epoch,
                    "train_normalized_mse": loss_sum / seen,
                    "valid_normalized_mae": normalized_mae,
                    "valid_raw_mae": raw_mae,
                    "valid_raw_rmse": raw_rmse,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if raw_mae < best_mae:
                best_mae = raw_mae
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            progress.set_postfix(
                {
                    "train_mse": f"{loss_sum / seen:.4f}",
                    "val_mae": f"{raw_mae:.4f}",
                    "best": f"{best_mae:.4f}@{best_epoch}",
                    "patience": f"{stale}/{config.training['early_stopping_patience']}",
                }
            )
            progress.update(1)
            if stale >= int(config.training["early_stopping_patience"]):
                break
    finally:
        progress.close()
    if best_state is None:
        raise RuntimeError("ILBERT training did not produce a best checkpoint")
    state_hash = tensor_state_hash("benchmark.ilbert-state.v1", best_state)
    model_path = root / "model.pt"
    history_path = root / "history.json"
    audit_path = root / "input_audit.json"
    atomic_torch_save(model_path, {"state_dict": best_state, "state_hash": state_hash})
    atomic_json(history_path, history)
    atomic_json(audit_path, {"train": bundle.train.audit, "valid": bundle.valid.audit})
    manifest = {
        "format_version": 1,
        "kind": "ilume_baseline_model",
        "model_kind": "ilbert",
        "training_identity": bundle.training_identity,
        "target_statistics": bundle.target_stats.to_dict(),
        "target_columns": list(bundle.task.target_columns),
        "view_count": bundle.train.view_count,
        "condition_dim": int(bundle.train.raw_conditions.shape[1]),
        "best_epoch": best_epoch,
        "best_valid_raw_mae": best_mae,
        "model_state_hash": state_hash,
        "pretrained_load_audit": model.load_audit,
        "upstream_assets": bundle.assets,
        "token_cache_audit": _token_cache_audit(bundle.token_cache),
        "runtime": config.runtime,
        "input_audit": {"train": bundle.train.audit, "valid": bundle.valid.audit},
        "integrity": {
            path.name: {"sha256": sha256_file(path), "size": path.stat().st_size}
            for path in (model_path, history_path, audit_path)
        },
    }
    atomic_json(root / "checkpoint.json", manifest)
    return {
        "best_epoch": best_epoch,
        "best_valid_raw_mae": best_mae,
        "epochs_ran": len(history),
        "input_audit": manifest["input_audit"],
    }


def _manifest(root: Path) -> dict[str, Any]:
    path = root / "checkpoint.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing ILBERT checkpoint manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("format_version") != 1
        or payload.get("kind") != "ilume_baseline_model"
        or payload.get("model_kind") != "ilbert"
    ):
        raise ValueError("Unsupported ILBERT checkpoint")
    for filename, expected in payload.get("integrity", {}).items():
        artifact = root / filename
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(expected["size"])
            or sha256_file(artifact) != expected["sha256"]
        ):
            raise ValueError(f"ILBERT checkpoint integrity mismatch: {filename}")
    return payload


def ilbert_evaluation_audit(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
    split: str,
) -> dict[str, Any]:
    if split not in {"valid", "test"}:
        raise ValueError("ILBERT evaluation split must be valid or test")
    task = resolve_task(config, benchmark, task_id, fold)
    raw = load_split(task, split)
    prepared = _prepare_split(
        raw,
        task,
        split,
        _tokenizer(config),
        {},
        max_length=int(config.model["max_length"]),
    )
    return prepared.audit


def evaluate_ilbert_checkpoint(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
    checkpoint_dir: str | Path,
    split: str,
) -> EvaluationResult:
    if split not in {"valid", "test"}:
        raise ValueError("ILBERT evaluation split must be valid or test")
    root = Path(checkpoint_dir)
    manifest = _manifest(root)
    bundle = prepare_ilbert_training(config, benchmark, task_id, fold)
    require_compatible_identity(
        bundle.training_identity,
        manifest["training_identity"],
        context="ILBERT evaluation checkpoint",
    )
    seed_benchmark(config.seed)
    _configure_tf32(bool(config.training["tf32"]))
    raw = load_split(bundle.task, split)
    prepared = _prepare_split(
        raw,
        bundle.task,
        split,
        _tokenizer(config),
        bundle.token_cache,
        max_length=int(config.model["max_length"]),
    )
    model = build_ilbert_model(config, bundle)
    payload = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    if tensor_state_hash(
        "benchmark.ilbert-state.v1", payload["state_dict"]
    ) != manifest["model_state_hash"]:
        raise ValueError("ILBERT checkpoint state hash mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    device = torch.device(str(config.training["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ILBERT evaluation requires CUDA; no silent CPU fallback")
    model.to(device)
    loader = _data_loader(config, prepared, bundle.token_cache, bundle.target_stats)
    normalized = _predict_normalized(
        model,
        loader,
        len(prepared.raw),
        device=device,
        non_blocking=bool(config.runtime["non_blocking_transfer"]),
    )
    predictions = bundle.target_stats.denormalize(normalized)
    metrics = target_metrics(
        predictions,
        raw.targets,
        bundle.task.target_columns,
        bundle.target_stats.scale,
    )
    if task_id in ORBITAL_TASK_TARGETS:
        target = ORBITAL_TASK_TARGETS[task_id]
        metrics[target]["role_diagnostics"] = role_mae_diagnostics(
            predictions[:, 0],
            raw.targets[:, 0],
            [row["ion_role"] for row in raw.audit_rows],
        )
    return EvaluationResult(
        predictions=predictions,
        targets=raw.targets,
        source_rows=raw.source_rows,
        metrics=metrics,
        target_stats=bundle.target_stats,
        training_identity=bundle.training_identity,
        components=raw.components,
        conditions=raw.conditions,
        audit_rows=raw.audit_rows,
        input_audit=prepared.audit,
    )


__all__ = [
    "EpochBatchSampler",
    "ILBERT_INPUT_CONTRACT",
    "ILBERTTrainingBundle",
    "PreparedSplit",
    "SharedILBERTRegressor",
    "build_ilbert_model",
    "evaluate_ilbert_checkpoint",
    "ilbert_evaluation_audit",
    "ilbert_model_sequences",
    "prepare_ilbert_training",
    "train_ilbert_bundle",
]
