from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from rdkit import Chem

from common.identity import require_compatible_identity, semantic_identity, tensor_state_hash
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.outputs import repository_path
from common.progress import ProgressReporter
from common.reporting import role_mae_diagnostics
from stage2.registry import ORBITAL_TASK_TARGETS

from benchmarks.common.config import BenchmarkConfig, BenchmarkName
from benchmarks.common.data import BenchmarkTask, RawDataset, load_split, resolve_task
from benchmarks.common.engine import EvaluationResult, TargetStats, seed_benchmark
from benchmarks.common.environment import spmm_asset_snapshot
from benchmarks.common.metrics import target_metrics


SPMM_INPUT_CONTRACT = {
    "canonical_identity": "ilume_isomeric_smiles",
    "model_input": "rdkit_canonical_isomeric_false",
    "tokenizer": "official_bert_wordpiece",
    "wordpiece_max_input_chars_per_word": 350,
    "manual_prefix": "[CLS]",
    "tokenizer_add_special_tokens": True,
    "tokenizer_max_length": 100,
    "encoder_slice": "remove_outer_leading_special_token",
    "encoder_max_length": 99,
    "truncation": True,
    "padding": "dynamic_longest",
    "input_cache": "unique_smiles_memory_token_cache",
    "component_forward": "merged_component_backbone_forward",
    "conditions": "train_only_zscore_in_registry_order",
}
TokenCache = dict[str, tuple[torch.Tensor, int]]


@dataclass(frozen=True)
class ConditionStats:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, values: np.ndarray) -> "ConditionStats":
        if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
            raise ValueError("SPMM train conditions must be a non-empty finite matrix")
        if values.shape[1] == 0:
            return cls((), ())
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        return cls(tuple(map(float, mean)), tuple(map(float, scale)))

    def normalize(self, values: np.ndarray) -> np.ndarray:
        if (
            values.ndim != 2
            or values.shape[1] != len(self.mean)
            or not np.isfinite(values).all()
        ):
            raise ValueError("SPMM condition shape or values differ from training")
        if not self.mean:
            return np.empty((len(values), 0), dtype=np.float32)
        return (
            (values - np.asarray(self.mean)) / np.asarray(self.scale)
        ).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedSplit:
    raw: RawDataset
    model_components: tuple[tuple[str, ...], ...]
    normalized_conditions: np.ndarray
    audit: dict[str, Any]
    component_count: int


@dataclass
class SPMMTrainingBundle:
    task: BenchmarkTask
    train: PreparedSplit
    valid: PreparedSplit
    target_stats: TargetStats
    condition_stats: ConditionStats
    source_hashes: dict[str, Any]
    training_identity: dict[str, Any]
    assets: dict[str, Any]
    token_cache: TokenCache
    pad_token_id: int


class SharedSPMMRegressor(torch.nn.Module):
    def __init__(
        self,
        text_encoder: torch.nn.Module,
        predictor: torch.nn.Module,
        *,
        component_count: int,
        condition_dim: int,
        hidden_dim: int,
        load_audit: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.predictor = predictor
        self.component_count = int(component_count)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.load_audit = dict(load_audit)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(conditions.shape[0])
        if int(input_ids.shape[0]) != self.component_count * batch_size:
            raise ValueError("SPMM merged molecular batch differs from task topology")
        states = self.text_encoder.bert(
            input_ids=input_ids.long(),
            attention_mask=attention_mask.long(),
            return_dict=True,
            mode="text",
        ).last_hidden_state
        pooled = states[:, 0, :]
        ordered = pooled.reshape(
            self.component_count, batch_size, self.hidden_dim
        ).transpose(0, 1)
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
        checkout / "xbert.py",
        checkout / "vocab_bpe_300.txt",
        checkout / "config_bert.json",
        repository_path(str(config.model["pretrained_checkpoint"])),
    )


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot dynamically load pinned SPMM source: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tokenizer(config: BenchmarkConfig) -> Any:
    from transformers import BertTokenizer, WordpieceTokenizer

    _, vocab, _, _ = _upstream_paths(config)
    tokenizer = BertTokenizer(
        vocab_file=str(vocab), do_lower_case=False, do_basic_tokenize=False
    )
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(
        vocab=tokenizer.vocab,
        unk_token=tokenizer.unk_token,
        max_input_chars_per_word=int(
            config.model["wordpiece_max_input_chars_per_word"]
        ),
    )
    return tokenizer


def spmm_model_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid ILUME canonical SMILES for SPMM: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _populate_token_cache(
    tokenizer: Any,
    smiles_values: Sequence[str],
    token_cache: TokenCache,
    *,
    max_length: int,
) -> None:
    missing = sorted(set(smiles_values) - set(token_cache))
    for smiles in missing:
        sequence = "[CLS]" + smiles
        full = tokenizer.encode(
            sequence, add_special_tokens=True, truncation=False
        )
        truncated = tokenizer.encode(
            sequence,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        if not full or not truncated or len(truncated) > max_length:
            raise ValueError("SPMM tokenizer returned an invalid token sequence")
        encoder_ids = torch.as_tensor(truncated[1:], dtype=torch.long)
        if encoder_ids.ndim != 1 or not len(encoder_ids):
            raise ValueError("SPMM official leading-token slice produced an empty sequence")
        if int(encoder_ids[0]) != int(tokenizer.cls_token_id):
            raise ValueError("SPMM manual [CLS] token was not preserved after official slicing")
        if len(encoder_ids) > max_length - 1:
            raise ValueError("SPMM encoder input exceeds the official 99-token contract")
        token_cache[smiles] = (encoder_ids, len(full))


def _token_cache_audit(token_cache: TokenCache) -> dict[str, int | str]:
    lengths = [length for _, length in token_cache.values()]
    return {
        "contract": "unique_smiles_memory_token_cache",
        "unique_model_inputs": len(lengths),
        "total_pre_truncation_tokens": sum(lengths),
        "maximum_pre_truncation_token_length": max(lengths, default=0),
    }


def _input_audit(
    raw: RawDataset,
    task: BenchmarkTask,
    split: str,
    model_components: tuple[tuple[str, ...], ...],
    token_cache: TokenCache,
    *,
    max_length: int,
) -> dict[str, Any]:
    model_to_benchmark: dict[str, set[str]] = {}
    for benchmark_row, model_row in zip(raw.components, model_components, strict=True):
        for benchmark_smiles, model_smiles in zip(
            benchmark_row, model_row, strict=True
        ):
            model_to_benchmark.setdefault(model_smiles, set()).add(benchmark_smiles)
    collisions = {
        model_smiles: sorted(benchmark_smiles)
        for model_smiles, benchmark_smiles in model_to_benchmark.items()
        if len(benchmark_smiles) > 1
    }
    collision_inputs = set(collisions)
    collision_rows: set[str] = set()
    truncated_rows: set[str] = set()
    truncated_components: list[dict[str, Any]] = []
    for source_row, benchmark_row, model_row in zip(
        raw.source_rows, raw.components, model_components, strict=True
    ):
        for slot, benchmark_smiles, model_smiles in zip(
            task.slots, benchmark_row, model_row, strict=True
        ):
            if model_smiles in collision_inputs:
                collision_rows.add(source_row)
            length = token_cache[model_smiles][1]
            if length > max_length:
                truncated_rows.add(source_row)
                truncated_components.append(
                    {
                        "source_row": source_row,
                        "slot": slot,
                        "benchmark_canonical_smiles": benchmark_smiles,
                        "spmm_input_smiles": model_smiles,
                        "pre_truncation_token_length": length,
                    }
                )
    affected_molecules = {
        smiles for values in collisions.values() for smiles in values
    }
    unique = set(value for row in model_components for value in row)
    lengths = [token_cache[value][1] for value in unique]
    return {
        "contract": SPMM_INPUT_CONTRACT,
        "task": task.task_id,
        "split": split,
        "component_order": list(task.slots),
        "condition_columns": list(task.condition_columns),
        "total_rows": len(raw),
        "unique_model_inputs": len(unique),
        "unique_model_input_tokens": sum(lengths),
        "maximum_pre_truncation_token_length": max(lengths, default=0),
        "truncated_component_count": len(truncated_components),
        "truncated_row_count": len(truncated_rows),
        "truncated_rows": sorted(truncated_rows),
        "truncated_components": truncated_components,
        "collision_group_count": len(collisions),
        "collision_affected_molecules": len(affected_molecules),
        "collision_affected_rows": len(collision_rows),
        "collision_groups": collisions,
        "collision_rows": sorted(collision_rows),
    }


def _prepare_split(
    raw: RawDataset,
    task: BenchmarkTask,
    split: str,
    tokenizer: Any,
    token_cache: TokenCache,
    condition_stats: ConditionStats | None,
    *,
    max_length: int,
) -> PreparedSplit:
    model_components = tuple(
        tuple(spmm_model_smiles(value) for value in row) for row in raw.components
    )
    if any(len(row) != len(task.slots) for row in model_components):
        raise ValueError("SPMM row component count differs from registry topology")
    flat = tuple(value for row in model_components for value in row)
    _populate_token_cache(tokenizer, flat, token_cache, max_length=max_length)
    normalized_conditions = (
        raw.conditions.astype(np.float32, copy=True)
        if condition_stats is None
        else condition_stats.normalize(raw.conditions)
    )
    if (
        normalized_conditions.ndim != 2
        or normalized_conditions.shape[1] != len(task.condition_columns)
        or not np.isfinite(normalized_conditions).all()
    ):
        raise ValueError("SPMM numeric conditions must be a finite matrix")
    return PreparedSplit(
        raw=raw,
        model_components=model_components,
        normalized_conditions=normalized_conditions,
        audit=_input_audit(
            raw,
            task,
            split,
            model_components,
            token_cache,
            max_length=max_length,
        ),
        component_count=len(task.slots),
    )


def _lock_sha(config: BenchmarkConfig) -> str:
    if config.environment is None:
        raise ValueError("SPMM environment contract is missing")
    return sha256_file(repository_path(config.environment.lock))


def prepare_spmm_training(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
) -> SPMMTrainingBundle:
    if config.name != "spmm":
        raise ValueError("SPMM adapter requires name=spmm")
    task = resolve_task(config, benchmark, task_id, fold)
    if len(task.target_columns) != 1:
        raise ValueError("SPMM baseline requires one scalar target per task")
    assets = spmm_asset_snapshot(config)
    tokenizer = _tokenizer(config)
    if int(tokenizer.pad_token_id) != 0:
        raise ValueError("SPMM tokenizer pad token ID must be zero")
    token_cache: TokenCache = {}
    train_raw = load_split(task, "train")
    valid_raw = load_split(task, "valid")
    if not len(train_raw):
        raise ValueError("SPMM requires non-empty training rows")
    if not len(valid_raw):
        raise ValueError("SPMM requires non-empty validation rows")
    condition_stats = ConditionStats.fit(train_raw.conditions)
    max_length = int(config.model["max_length"])
    train = _prepare_split(
        train_raw,
        task,
        "train",
        tokenizer,
        token_cache,
        condition_stats,
        max_length=max_length,
    )
    valid = _prepare_split(
        valid_raw,
        task,
        "valid",
        tokenizer,
        token_cache,
        condition_stats,
        max_length=max_length,
    )
    if train.component_count != valid.component_count:
        raise ValueError("SPMM train and valid component topology differ")
    target_stats = TargetStats.fit(train.raw.targets)
    source_hashes = {
        "train": [sha256_file(path) for path in task.train_paths],
        "valid": [sha256_file(path) for path in task.valid_paths],
    }
    identity = semantic_identity(
        "benchmark.training.v1",
        {
            "benchmark_model": "spmm",
            "domain": task.benchmark,
            "task_id": task.task_id,
            "fold": task.fold,
            "registry": task.registry_payload,
            "component_order": list(task.slots),
            "source_hashes": source_hashes,
            "input_contract": SPMM_INPUT_CONTRACT,
            "token_cache_audit": _token_cache_audit(token_cache),
            "train_input_audit": train.audit,
            "valid_input_audit": valid.audit,
            "condition_statistics": condition_stats.to_dict(),
            "target_statistics": target_stats.to_dict(),
            "upstream_assets": assets,
            "model": config.model,
            "training": config.training,
            "seed": config.seed,
            "environment_lock_sha256": _lock_sha(config),
        },
    )
    return SPMMTrainingBundle(
        task=task,
        train=train,
        valid=valid,
        target_stats=target_stats,
        condition_stats=condition_stats,
        source_hashes=source_hashes,
        training_identity=identity,
        assets=assets,
        token_cache=token_cache,
        pad_token_id=int(tokenizer.pad_token_id),
    )


def _load_pretrained_encoder(
    config: BenchmarkConfig, text_encoder: torch.nn.Module
) -> dict[str, Any]:
    _, _, _, checkpoint_path = _upstream_paths(config)
    if (
        checkpoint_path.stat().st_size != int(config.model["pretrained_size"])
        or sha256_file(checkpoint_path) != str(config.model["pretrained_sha256"])
    ):
        raise RuntimeError("SPMM checkpoint trust boundary changed before deserialization")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("state_dict"), Mapping
    ):
        raise ValueError("SPMM checkpoint does not contain a state_dict")
    source = checkpoint["state_dict"]
    if len(source) != 758:
        raise RuntimeError(
            f"SPMM checkpoint state entry count differs from upstream: {len(source)}"
        )
    target = text_encoder.state_dict()
    if len(target) != 102:
        raise RuntimeError(
            f"SPMM text encoder state entry count differs from contract: {len(target)}"
        )
    selected = {
        key: source[f"text_encoder.{key}"]
        for key in target
        if f"text_encoder.{key}" in source
    }
    missing = sorted(set(target) - set(selected))
    shape_mismatches = {
        key: {"expected": list(target[key].shape), "actual": list(selected[key].shape)}
        for key in selected
        if tuple(target[key].shape) != tuple(selected[key].shape)
    }
    if missing or shape_mismatches or len(selected) != 102:
        raise RuntimeError(
            "SPMM pretrained text encoder load contract mismatch: "
            + json.dumps(
                {"missing": missing, "shape_mismatches": shape_mismatches},
                sort_keys=True,
            )
        )
    text_encoder.load_state_dict(selected, strict=True)
    ignored: dict[str, int] = {}
    selected_source = {f"text_encoder.{key}" for key in selected}
    for key in set(source) - selected_source:
        prefix = key.split(".", 1)[0]
        ignored[prefix] = ignored.get(prefix, 0) + 1
    return {
        "checkpoint_format": "trusted_pinned_lightning_pickle",
        "source_state_entries": len(source),
        "loaded_text_encoder_entries": len(selected),
        "loaded_scope": "text_encoder.bert embeddings and layers 0-5",
        "ignored_namespace_counts": dict(sorted(ignored.items())),
    }


def build_spmm_model(
    config: BenchmarkConfig,
    bundle: SPMMTrainingBundle,
    *,
    load_pretrained: bool = True,
    load_audit: Mapping[str, Any] | None = None,
) -> SharedSPMMRegressor:
    xbert_source, _, bert_config_path, _ = _upstream_paths(config)
    upstream = _load_module(xbert_source, "ilume_pinned_spmm_xbert_adapter")
    bert_config = upstream.BertConfig.from_json_file(str(bert_config_path))
    if (
        int(bert_config.vocab_size) != int(config.model["vocab_size"])
        or int(bert_config.hidden_size) != int(config.model["hidden_dim"])
        or int(bert_config.num_hidden_layers) != 12
        or int(bert_config.fusion_layer) != int(config.model["text_layers"])
        or int(bert_config.num_attention_heads) != int(config.model["heads"])
        or int(bert_config.intermediate_size) != int(config.model["ffn_hidden_dim"])
        or float(bert_config.hidden_dropout_prob) != float(config.model["dropout"])
        or float(bert_config.attention_probs_dropout_prob)
        != float(config.model["dropout"])
    ):
        raise RuntimeError("SPMM upstream BERT config differs from registered architecture")
    text_encoder = upstream.BertForMaskedLM(config=bert_config)
    for index in range(int(bert_config.fusion_layer), int(bert_config.num_hidden_layers)):
        text_encoder.bert.encoder.layer[index] = torch.nn.Identity()
    text_encoder.cls = torch.nn.Identity()
    audit = (
        _load_pretrained_encoder(config, text_encoder)
        if load_pretrained
        else dict(load_audit or {})
    )
    if not audit:
        raise ValueError("SPMM model requires a pretrained load audit")
    hidden_dim = int(config.model["hidden_dim"])
    input_dim = len(bundle.task.slots) * hidden_dim + len(bundle.task.condition_columns)
    predictor = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim * 2),
        torch.nn.GELU(),
        torch.nn.Linear(hidden_dim * 2, 1),
    )
    return SharedSPMMRegressor(
        text_encoder,
        predictor,
        component_count=len(bundle.task.slots),
        condition_dim=len(bundle.task.condition_columns),
        hidden_dim=hidden_dim,
        load_audit=audit,
    )


def _collate(
    prepared: PreparedSplit,
    token_cache: TokenCache,
    target_stats: TargetStats,
    *,
    pad_token_id: int,
) -> Any:
    def collate(
        indices: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = [
            token_cache[prepared.model_components[index][component]][0]
            for component in range(prepared.component_count)
            for index in indices
        ]
        maximum = max(map(len, tokens))
        input_ids = torch.full(
            (len(tokens), maximum), int(pad_token_id), dtype=torch.long
        )
        attention_mask = torch.zeros((len(tokens), maximum), dtype=torch.long)
        for row, values in enumerate(tokens):
            input_ids[row, : len(values)] = values
            attention_mask[row, : len(values)] = 1
        selected = np.asarray(indices, dtype=np.int64)
        return (
            input_ids,
            attention_mask,
            torch.from_numpy(prepared.normalized_conditions[selected]),
            torch.from_numpy(target_stats.normalize(prepared.raw.targets[selected])),
        )

    return collate


def _data_loader(
    config: BenchmarkConfig,
    prepared: PreparedSplit,
    token_cache: TokenCache,
    target_stats: TargetStats,
    *,
    pad_token_id: int,
    batch_sampler: EpochBatchSampler | None = None,
) -> torch.utils.data.DataLoader:
    common = {
        "num_workers": int(config.runtime["num_workers"]),
        "pin_memory": bool(config.runtime["pin_memory"]),
        "persistent_workers": bool(config.runtime["persistent_workers"]),
        "prefetch_factor": int(config.runtime["prefetch_factor"]),
        "collate_fn": _collate(
            prepared, token_cache, target_stats, pad_token_id=pad_token_id
        ),
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


def _configure_backend(config: BenchmarkConfig) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(
        config.training["cuda_matmul_tf32"]
    )
    torch.backends.cudnn.allow_tf32 = bool(config.training["cudnn_tf32"])
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = bool(config.training["cudnn_benchmark"])


def _scheduled_learning_rate(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    warmup_learning_rate: float,
    peak_learning_rate: float,
    minimum_learning_rate: float,
) -> float:
    if total_steps <= 0 or warmup_steps <= 0 or warmup_steps >= total_steps:
        raise ValueError("SPMM scheduler step contract is invalid")
    if step < warmup_steps:
        progress = step / max(1, warmup_steps - 1)
        return warmup_learning_rate + progress * (
            peak_learning_rate - warmup_learning_rate
        )
    decay_steps = total_steps - warmup_steps
    progress = (step - warmup_steps) / max(1, decay_steps - 1)
    return minimum_learning_rate + 0.5 * (
        peak_learning_rate - minimum_learning_rate
    ) * (1.0 + math.cos(math.pi * progress))


def _predict_normalized(
    model: SharedSPMMRegressor,
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
        for input_ids, attention_mask, conditions, _ in loader:
            prediction = model(
                input_ids.to(device, non_blocking=non_blocking),
                attention_mask.to(device, non_blocking=non_blocking),
                conditions.to(device, non_blocking=non_blocking),
            )
            if not torch.isfinite(prediction).all():
                raise RuntimeError("SPMM evaluation produced non-finite predictions")
            values.append(prediction.float().cpu())
    return torch.cat(values).numpy()


def train_spmm_bundle(
    config: BenchmarkConfig,
    bundle: SPMMTrainingBundle,
    output_dir: str | Path,
    *,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seed_benchmark(config.seed)
    _configure_backend(config)
    device = torch.device(str(config.training["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SPMM requires CUDA; no silent CPU fallback")
    model = build_spmm_model(config, bundle).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training["warmup_learning_rate"]),
        weight_decay=float(config.training["weight_decay"]),
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
        pad_token_id=bundle.pad_token_id,
        batch_sampler=sampler,
    )
    valid_loader = _data_loader(
        config,
        bundle.valid,
        bundle.token_cache,
        bundle.target_stats,
        pad_token_id=bundle.pad_token_id,
    )
    steps_per_epoch = len(sampler)
    max_epochs = int(config.training["max_epochs"])
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = steps_per_epoch * int(config.training["warmup_epochs"])
    non_blocking = bool(config.runtime["non_blocking_transfer"])
    best_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    progress = (reporter or ProgressReporter()).bar(
        total=max_epochs, desc=f"SPMM {bundle.task.task_id}", unit="epoch"
    )
    try:
        for epoch in range(1, max_epochs + 1):
            sampler.set_epoch(epoch - 1)
            model.train()
            loss_sum = 0.0
            seen = 0
            for input_ids, attention_mask, conditions, targets in train_loader:
                learning_rate = _scheduled_learning_rate(
                    global_step,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    warmup_learning_rate=float(
                        config.training["warmup_learning_rate"]
                    ),
                    peak_learning_rate=float(config.training["learning_rate"]),
                    minimum_learning_rate=float(
                        config.training["minimum_learning_rate"]
                    ),
                )
                optimizer.param_groups[0]["lr"] = learning_rate
                optimizer.zero_grad(set_to_none=True)
                prediction = model(
                    input_ids.to(device, non_blocking=non_blocking),
                    attention_mask.to(device, non_blocking=non_blocking),
                    conditions.to(device, non_blocking=non_blocking),
                )
                target = targets.to(device, non_blocking=non_blocking)
                loss = torch.nn.functional.mse_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise RuntimeError("SPMM training produced a non-finite loss")
                loss.backward()
                optimizer.step()
                size = len(targets)
                loss_sum += float(loss.detach().cpu()) * size
                seen += size
                global_step += 1
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
            history.append(
                {
                    "epoch": epoch,
                    "optimizer_steps": global_step,
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
        raise RuntimeError("SPMM training did not produce a best checkpoint")
    state_hash = tensor_state_hash("benchmark.spmm-state.v1", best_state)
    model_path = root / "model.pt"
    history_path = root / "history.json"
    audit_path = root / "input_audit.json"
    atomic_torch_save(model_path, {"state_dict": best_state, "state_hash": state_hash})
    atomic_json(history_path, history)
    atomic_json(audit_path, {"train": bundle.train.audit, "valid": bundle.valid.audit})
    manifest = {
        "format_version": 1,
        "kind": "ilume_baseline_model",
        "model_kind": "spmm",
        "training_identity": bundle.training_identity,
        "target_statistics": bundle.target_stats.to_dict(),
        "condition_statistics": bundle.condition_stats.to_dict(),
        "target_columns": list(bundle.task.target_columns),
        "component_count": bundle.train.component_count,
        "condition_dim": len(bundle.task.condition_columns),
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
        raise FileNotFoundError(f"Missing SPMM checkpoint manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("format_version") != 1
        or payload.get("kind") != "ilume_baseline_model"
        or payload.get("model_kind") != "spmm"
    ):
        raise ValueError("Unsupported SPMM checkpoint")
    for filename, expected in payload.get("integrity", {}).items():
        artifact = root / filename
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(expected["size"])
            or sha256_file(artifact) != expected["sha256"]
        ):
            raise ValueError(f"SPMM checkpoint integrity mismatch: {filename}")
    return payload


def spmm_evaluation_audit(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
    split: str,
) -> dict[str, Any]:
    if split not in {"valid", "test"}:
        raise ValueError("SPMM evaluation split must be valid or test")
    task = resolve_task(config, benchmark, task_id, fold)
    raw = load_split(task, split)
    prepared = _prepare_split(
        raw,
        task,
        split,
        _tokenizer(config),
        {},
        None,
        max_length=int(config.model["max_length"]),
    )
    return prepared.audit


def evaluate_spmm_checkpoint(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
    checkpoint_dir: str | Path,
    split: str,
) -> EvaluationResult:
    if split not in {"valid", "test"}:
        raise ValueError("SPMM evaluation split must be valid or test")
    root = Path(checkpoint_dir)
    manifest = _manifest(root)
    bundle = prepare_spmm_training(config, benchmark, task_id, fold)
    require_compatible_identity(
        bundle.training_identity,
        manifest["training_identity"],
        context="SPMM evaluation checkpoint",
    )
    seed_benchmark(config.seed)
    _configure_backend(config)
    raw = load_split(bundle.task, split)
    prepared = _prepare_split(
        raw,
        bundle.task,
        split,
        _tokenizer(config),
        bundle.token_cache,
        bundle.condition_stats,
        max_length=int(config.model["max_length"]),
    )
    model = build_spmm_model(
        config,
        bundle,
        load_pretrained=False,
        load_audit=manifest["pretrained_load_audit"],
    )
    payload = torch.load(root / "model.pt", map_location="cpu")
    if tensor_state_hash(
        "benchmark.spmm-state.v1", payload["state_dict"]
    ) != manifest["model_state_hash"]:
        raise ValueError("SPMM checkpoint state hash mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    device = torch.device(str(config.training["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SPMM evaluation requires CUDA; no silent CPU fallback")
    model.to(device)
    loader = _data_loader(
        config,
        prepared,
        bundle.token_cache,
        bundle.target_stats,
        pad_token_id=bundle.pad_token_id,
    )
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
    "ConditionStats",
    "EpochBatchSampler",
    "PreparedSplit",
    "SPMMTrainingBundle",
    "SPMM_INPUT_CONTRACT",
    "SharedSPMMRegressor",
    "build_spmm_model",
    "evaluate_spmm_checkpoint",
    "prepare_spmm_training",
    "spmm_evaluation_audit",
    "spmm_model_smiles",
    "train_spmm_bundle",
]
