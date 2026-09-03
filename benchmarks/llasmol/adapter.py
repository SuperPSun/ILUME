from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
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
from benchmarks.common.environment import llasmol_asset_snapshot
from benchmarks.common.metrics import target_metrics


LLASMOL_INPUT_CONTRACT = {
    "canonical_identity": "ilume_isomeric_smiles",
    "model_input": "uppercase_task_leaf_newline_smiles",
    "native_ionic_liquid": "cation_dot_anion",
    "multiview_forward": "merged_sequence_view_forward",
    "tokenizer": "pinned_mistral_base_tokenizer",
    "add_special_tokens": True,
    "adds_bos": True,
    "adds_eos": False,
    "pad_token_id": 0,
    "padding_side": "left",
    "truncation_side": "right",
    "max_length": 512,
    "padding": "dynamic_longest_multiple_of_8",
    "input_cache": "unique_sequence_memory_token_cache",
    "conditions": "train_only_zscore_in_registry_order",
}
LLASMOL_TRAINING_ORDER_CONTRACT = "sortish_length_bucketing_v1"
TokenCache = dict[str, tuple[torch.Tensor, torch.Tensor, int]]


@dataclass(frozen=True)
class ConditionStats:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, values: np.ndarray) -> "ConditionStats":
        if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
            raise ValueError("LlaSMol train conditions must be a non-empty finite matrix")
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
            raise ValueError("LlaSMol condition shape or values differ from training")
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
    model_views: tuple[tuple[str, ...], ...]
    view_names: tuple[str, ...]
    normalized_conditions: np.ndarray
    audit: dict[str, Any]

    @property
    def view_count(self) -> int:
        return len(self.view_names)


@dataclass
class LlaSMolTrainingBundle:
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


class SharedLlaSMolRegressor(torch.nn.Module):
    def __init__(
        self,
        llasmol: torch.nn.Module,
        predictor: torch.nn.Module,
        *,
        view_count: int,
        condition_dim: int,
        hidden_dim: int,
        load_audit: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.llasmol = llasmol
        self.predictor = predictor
        self.view_count = int(view_count)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.load_audit = dict(load_audit)

    def _backbone(self) -> torch.nn.Module:
        try:
            return self.llasmol.base_model.model.model
        except AttributeError as error:
            raise RuntimeError("Pinned PEFT Mistral backbone path changed") from error

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(conditions.shape[0])
        if int(input_ids.shape[0]) != self.view_count * batch_size:
            raise ValueError("LlaSMol merged sequence batch differs from task topology")
        states = self._backbone()(
            input_ids=input_ids.long(),
            attention_mask=attention_mask.long(),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        mask = attention_mask.to(dtype=states.dtype).unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        pooled = (states * mask).sum(dim=1) / denominator
        ordered = pooled.reshape(
            self.view_count, batch_size, self.hidden_dim
        ).transpose(0, 1)
        representation = torch.cat(
            (ordered.reshape(batch_size, -1).float(), conditions.float()), dim=1
        )
        return self.predictor(representation)


class SortishBatchSampler(torch.utils.data.Sampler[list[int]]):
    def __init__(
        self,
        lengths: Sequence[int],
        *,
        batch_size: int,
        window_batches: int,
        seed: int,
    ) -> None:
        self.lengths = tuple(int(value) for value in lengths)
        self.batch_size = int(batch_size)
        self.window_batches = int(window_batches)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self) -> Any:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(len(self.lengths), generator=generator).tolist()
        window_size = self.window_batches * self.batch_size
        batches: list[list[int]] = []
        for start in range(0, len(permutation), window_size):
            window = permutation[start : start + window_size]
            window.sort(key=self.lengths.__getitem__)
            batches.extend(
                window[offset : offset + self.batch_size]
                for offset in range(0, len(window), self.batch_size)
            )
        batch_order = torch.randperm(len(batches), generator=generator).tolist()
        return iter(batches[index] for index in batch_order)


def llasmol_task_prefix(task_id: str) -> str:
    leaf = task_id.rsplit("/", 1)[-1]
    if not leaf or not re.fullmatch(r"[a-z0-9_]+", leaf):
        raise ValueError(f"LlaSMol cannot derive a task marker from: {task_id}")
    return f"<{leaf.upper()}>"


def llasmol_model_views(
    task: BenchmarkTask, components: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    slots = tuple(task.slots)
    values = tuple(components)
    if len(values) != len(slots):
        raise ValueError("LlaSMol row component count differs from registry topology")
    if slots == ("cation", "anion"):
        return (f"{values[0]}.{values[1]}",), ("ionic_liquid",)
    if slots == ("cation", "anion", "solute"):
        return (f"{values[0]}.{values[1]}", values[2]), ("ionic_liquid", "solute")
    if slots == ("solute", "solvent"):
        return values, slots
    if len(slots) == 1 and slots[0] in {"SMILES", "smiles", "molecule"}:
        return values, (slots[0],)
    raise ValueError(f"LlaSMol does not support registry topology: {slots}")


def _view_source_slots(task: BenchmarkTask, view: str) -> tuple[str, ...]:
    if view == "ionic_liquid":
        return tuple(task.slots[:2])
    if view in task.slots:
        return (view,)
    raise ValueError(f"LlaSMol view does not map to registry slots: {view}")


def _tokenizer(config: BenchmarkConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(repository_path(str(config.model["base_snapshot"]))),
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is not None:
        raise RuntimeError("Pinned Mistral tokenizer unexpectedly defines a pad token")
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    if (
        tokenizer.pad_token_id != 0
        or tokenizer.unk_token_id != 0
        or tokenizer.bos_token_id != 1
        or tokenizer.eos_token_id != 2
    ):
        raise RuntimeError("LlaSMol tokenizer special IDs differ from contract")
    return tokenizer


def _populate_token_cache(
    tokenizer: Any,
    sequences: Sequence[str],
    token_cache: TokenCache,
    *,
    max_length: int,
) -> None:
    for sequence in sorted(set(sequences) - set(token_cache)):
        full = tokenizer(
            sequence,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=True,
        )
        encoded = tokenizer(
            sequence,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )
        input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
        attention_mask = torch.as_tensor(encoded["attention_mask"], dtype=torch.long)
        if (
            input_ids.ndim != 1
            or not len(input_ids)
            or len(input_ids) > max_length
            or attention_mask.shape != input_ids.shape
            or int(input_ids[0]) != 1
            or not bool(attention_mask.all())
        ):
            raise ValueError("LlaSMol tokenizer violated the registered sequence contract")
        token_cache[sequence] = (input_ids, attention_mask, len(full["input_ids"]))


def _token_cache_audit(token_cache: TokenCache) -> dict[str, int | str]:
    lengths = [length for _, _, length in token_cache.values()]
    return {
        "contract": "unique_sequence_memory_token_cache",
        "unique_model_inputs": len(lengths),
        "total_pre_truncation_tokens": sum(lengths),
        "maximum_pre_truncation_token_length": max(lengths, default=0),
    }


def _row_token_lengths(
    prepared: PreparedSplit, token_cache: TokenCache
) -> tuple[int, ...]:
    return tuple(
        max(len(token_cache[value][0]) for value in row)
        for row in prepared.model_views
    )


def _input_audit(
    raw: RawDataset,
    task: BenchmarkTask,
    split: str,
    model_views: tuple[tuple[str, ...], ...],
    view_names: tuple[str, ...],
    token_cache: TokenCache,
    *,
    max_length: int,
) -> dict[str, Any]:
    truncated_rows: set[str] = set()
    truncated_views: list[dict[str, Any]] = []
    for source_row, row in zip(raw.source_rows, model_views, strict=True):
        for view, sequence in zip(view_names, row, strict=True):
            length = token_cache[sequence][2]
            if length > max_length:
                truncated_rows.add(source_row)
                truncated_views.append(
                    {
                        "source_row": source_row,
                        "view": view,
                        "source_slots": list(_view_source_slots(task, view)),
                        "pre_truncation_token_length": length,
                    }
                )
    unique = {value for row in model_views for value in row}
    lengths = [token_cache[value][2] for value in unique]
    return {
        "contract": LLASMOL_INPUT_CONTRACT,
        "task": task.task_id,
        "task_prefix": llasmol_task_prefix(task.task_id),
        "split": split,
        "component_order": list(task.slots),
        "view_order": list(view_names),
        "view_source_slots": {
            view: list(_view_source_slots(task, view)) for view in view_names
        },
        "condition_columns": list(task.condition_columns),
        "total_rows": len(raw),
        "unique_model_inputs": len(unique),
        "unique_model_input_tokens": sum(lengths),
        "maximum_pre_truncation_token_length": max(lengths, default=0),
        "truncated_view_count": len(truncated_views),
        "truncated_row_count": len(truncated_rows),
        "truncated_rows": sorted(truncated_rows),
        "truncated_views": truncated_views,
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
    prefix = llasmol_task_prefix(task.task_id)
    view_names: tuple[str, ...] | None = None
    rows: list[tuple[str, ...]] = []
    for components in raw.components:
        views, names = llasmol_model_views(task, components)
        if view_names is None:
            view_names = names
        elif view_names != names:
            raise ValueError("LlaSMol split contains inconsistent view topology")
        rows.append(tuple(f"{prefix}\n{value}" for value in views))
    if view_names is None:
        _, view_names = llasmol_model_views(task, tuple("" for _ in task.slots))
    model_views = tuple(rows)
    flat = tuple(value for row in model_views for value in row)
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
        raise ValueError("LlaSMol numeric conditions must be a finite matrix")
    return PreparedSplit(
        raw=raw,
        model_views=model_views,
        view_names=view_names,
        normalized_conditions=normalized_conditions,
        audit=_input_audit(
            raw,
            task,
            split,
            model_views,
            view_names,
            token_cache,
            max_length=max_length,
        ),
    )


def _lock_sha(config: BenchmarkConfig) -> str:
    if config.environment is None:
        raise ValueError("LlaSMol environment contract is missing")
    return sha256_file(repository_path(config.environment.lock))


def prepare_llasmol_training(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
) -> LlaSMolTrainingBundle:
    if config.name != "llasmol":
        raise ValueError("LlaSMol adapter requires name=llasmol")
    task = resolve_task(config, benchmark, task_id, fold)
    if len(task.target_columns) != 1:
        raise ValueError("LlaSMol baseline requires one scalar target per task")
    assets = llasmol_asset_snapshot(config)
    tokenizer = _tokenizer(config)
    token_cache: TokenCache = {}
    train_raw = load_split(task, "train")
    valid_raw = load_split(task, "valid")
    if not len(train_raw) or not len(valid_raw):
        raise ValueError("LlaSMol requires non-empty train and validation rows")
    condition_stats = ConditionStats.fit(train_raw.conditions)
    max_length = int(config.model["max_length"])
    train = _prepare_split(
        train_raw, task, "train", tokenizer, token_cache, condition_stats,
        max_length=max_length,
    )
    valid = _prepare_split(
        valid_raw, task, "valid", tokenizer, token_cache, condition_stats,
        max_length=max_length,
    )
    if train.view_names != valid.view_names:
        raise ValueError("LlaSMol train and valid view topology differ")
    target_stats = TargetStats.fit(train.raw.targets)
    source_hashes = {
        "train": [sha256_file(path) for path in task.train_paths],
        "valid": [sha256_file(path) for path in task.valid_paths],
    }
    identity = semantic_identity(
        "benchmark.training.v1",
        {
            "benchmark_model": "llasmol",
            "domain": task.benchmark,
            "task_id": task.task_id,
            "fold": task.fold,
            "registry": task.registry_payload,
            "component_order": list(task.slots),
            "view_order": list(train.view_names),
            "source_hashes": source_hashes,
            "input_contract": LLASMOL_INPUT_CONTRACT,
            "training_order_contract": LLASMOL_TRAINING_ORDER_CONTRACT,
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
    return LlaSMolTrainingBundle(
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


def _official_adapter_state(
    config: BenchmarkConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path = repository_path(str(config.model["adapter_snapshot"])) / "adapter_model.bin"
    if (
        path.stat().st_size != int(config.model["adapter_model_size"])
        or sha256_file(path) != str(config.model["adapter_model_sha256"])
    ):
        raise RuntimeError("LlaSMol adapter trust boundary changed before deserialization")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or len(state) != int(config.model["adapter_state_entries"]):
        raise RuntimeError("LlaSMol adapter state entry count differs from contract")
    tensors = {str(name): value for name, value in state.items()}
    if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise RuntimeError("LlaSMol adapter contains non-tensor state")
    modules = tuple(str(value) for value in config.model["lora_target_modules"])
    counts = {
        module: sum(f".{module}." in name for name in tensors) for module in modules
    }
    layers = {
        int(match.group(1))
        for name in tensors
        if (match := re.search(r"\.layers\.(\d+)\.", name)) is not None
    }
    if (
        any(value.dtype != torch.bfloat16 for value in tensors.values())
        or any(count != 64 for count in counts.values())
        or layers != set(range(32))
        or any(".lora_A.weight" not in name and ".lora_B.weight" not in name for name in tensors)
    ):
        raise RuntimeError("LlaSMol adapter tensor namespace or dtype differs from contract")
    return tensors, {
        "format": "pinned_official_weights_only_pickle",
        "state_entries": len(tensors),
        "dtype": "torch.bfloat16",
        "layers": 32,
        "module_entry_counts": counts,
    }


def build_llasmol_model(
    config: BenchmarkConfig,
    bundle: LlaSMolTrainingBundle,
    *,
    device: torch.device,
) -> SharedLlaSMolRegressor:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        set_peft_model_state_dict,
    )
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("LlaSMol requires CUDA; no silent CPU fallback")
    quantization = BitsAndBytesConfig(
        load_in_4bit=bool(config.model["load_in_4bit"]),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        str(repository_path(str(config.model["base_snapshot"]))),
        local_files_only=True,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": device.index if device.index is not None else 0},
    )
    base.config.use_cache = bool(config.model["use_cache"])
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=bool(config.model["gradient_checkpointing"])
    )
    lora = LoraConfig(
        r=int(config.model["lora_rank"]),
        lora_alpha=int(config.model["lora_alpha"]),
        lora_dropout=float(config.model["lora_dropout"]),
        target_modules=list(config.model["lora_target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    adapted = get_peft_model(base, lora)
    official_state, audit = _official_adapter_state(config)
    incompatible = set_peft_model_state_dict(adapted, official_state)
    missing_lora = sorted(
        name for name in incompatible.missing_keys if ".lora_" in name
    )
    unexpected = sorted(incompatible.unexpected_keys)
    if missing_lora or unexpected:
        raise RuntimeError(
            "LlaSMol official adapter load mismatch: "
            + json.dumps(
                {"missing_lora": missing_lora, "unexpected": unexpected}, sort_keys=True
            )
        )
    non_lora_trainable = sorted(
        name
        for name, parameter in adapted.named_parameters()
        if parameter.requires_grad and ".lora_" not in name
    )
    if non_lora_trainable:
        raise RuntimeError("LlaSMol frozen base exposes non-LoRA trainable parameters")
    audit["trainable_lora_parameters"] = sum(
        parameter.numel() for parameter in adapted.parameters() if parameter.requires_grad
    )
    hidden_dim = int(config.model["hidden_dim"])
    input_dim = bundle.train.view_count * hidden_dim + len(bundle.task.condition_columns)
    predictor = torch.nn.Sequential(
        torch.nn.Linear(input_dim, int(config.model["head_hidden_dim"])),
        torch.nn.SiLU(),
        torch.nn.Linear(int(config.model["head_hidden_dim"]), 1),
    ).to(device=device, dtype=torch.float32)
    return SharedLlaSMolRegressor(
        adapted,
        predictor,
        view_count=bundle.train.view_count,
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
        cached = [
            token_cache[prepared.model_views[index][view]]
            for view in range(prepared.view_count)
            for index in indices
        ]
        maximum = min(512, 8 * math.ceil(max(len(value[0]) for value in cached) / 8))
        input_ids = torch.full(
            (len(cached), maximum), int(pad_token_id), dtype=torch.long
        )
        attention_mask = torch.zeros((len(cached), maximum), dtype=torch.long)
        for row, (tokens, mask, _) in enumerate(cached):
            input_ids[row, -len(tokens) :] = tokens
            attention_mask[row, -len(mask) :] = mask
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
    batch_sampler: SortishBatchSampler | None = None,
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
    torch.backends.cuda.matmul.allow_tf32 = bool(config.training["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config.training["tf32"])
    torch.set_float32_matmul_precision("highest")


def _scheduled_factor(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 1 or warmup_steps <= 0 or warmup_steps >= total_steps:
        raise ValueError("LlaSMol scheduler step contract is invalid")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _predict_normalized(
    model: SharedLlaSMolRegressor,
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
                raise RuntimeError("LlaSMol evaluation produced non-finite predictions")
            values.append(prediction.float().cpu())
    return torch.cat(values).numpy()


def _trainable_state(model: SharedLlaSMolRegressor) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _load_trainable_state(
    model: SharedLlaSMolRegressor, state: Mapping[str, torch.Tensor]
) -> None:
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(state) != set(trainable):
        raise ValueError("LlaSMol checkpoint trainable parameter set mismatch")
    with torch.no_grad():
        for name, parameter in trainable.items():
            value = state[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"LlaSMol checkpoint tensor shape mismatch: {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def train_llasmol_bundle(
    config: BenchmarkConfig,
    bundle: LlaSMolTrainingBundle,
    output_dir: str | Path,
    *,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seed_benchmark(config.seed)
    _configure_backend(config)
    device = torch.device(str(config.training["device"]))
    model = build_llasmol_model(config, bundle, device=device)
    lora_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("predictor.")
    ]
    head_parameters = list(model.predictor.parameters())
    if not lora_parameters or not head_parameters:
        raise RuntimeError("LlaSMol optimizer parameter groups are incomplete")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_parameters,
                "lr": float(config.training["lora_learning_rate"]),
                "weight_decay": float(config.training["weight_decay"]),
            },
            {
                "params": head_parameters,
                "lr": float(config.training["head_learning_rate"]),
                "weight_decay": float(config.training["weight_decay"]),
            },
        ]
    )
    sampler = SortishBatchSampler(
        _row_token_lengths(bundle.train, bundle.token_cache),
        batch_size=int(config.training["batch_size"]),
        window_batches=int(config.training["bucket_window_batches"]),
        seed=config.seed,
    )
    train_loader = _data_loader(
        config, bundle.train, bundle.token_cache, bundle.target_stats,
        pad_token_id=bundle.pad_token_id, batch_sampler=sampler,
    )
    valid_loader = _data_loader(
        config, bundle.valid, bundle.token_cache, bundle.target_stats,
        pad_token_id=bundle.pad_token_id,
    )
    accumulation = int(config.training["gradient_accumulation_steps"])
    optimizer_steps_per_epoch = math.ceil(len(sampler) / accumulation)
    max_epochs = int(config.training["max_epochs"])
    total_steps = optimizer_steps_per_epoch * max_epochs
    warmup_steps = math.ceil(float(config.training["warmup_fraction"]) * total_steps)
    non_blocking = bool(config.runtime["non_blocking_transfer"])
    peaks = (
        float(config.training["lora_learning_rate"]),
        float(config.training["head_learning_rate"]),
    )
    best_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    progress = (reporter or ProgressReporter()).bar(
        total=max_epochs, desc=f"LlaSMol {bundle.task.task_id}", unit="epoch"
    )
    try:
        for epoch in range(1, max_epochs + 1):
            sampler.set_epoch(epoch - 1)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            seen = 0
            for batch_index, (input_ids, attention_mask, conditions, targets) in enumerate(
                train_loader
            ):
                group_start = (batch_index // accumulation) * accumulation
                window = min(accumulation, len(train_loader) - group_start)
                prediction = model(
                    input_ids.to(device, non_blocking=non_blocking),
                    attention_mask.to(device, non_blocking=non_blocking),
                    conditions.to(device, non_blocking=non_blocking),
                )
                target = targets.to(device, non_blocking=non_blocking)
                loss = torch.nn.functional.mse_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise RuntimeError("LlaSMol training produced a non-finite loss")
                (loss / window).backward()
                size = len(targets)
                loss_sum += float(loss.detach().cpu()) * size
                seen += size
                should_step = (
                    (batch_index + 1) % accumulation == 0
                    or batch_index + 1 == len(train_loader)
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        lora_parameters + head_parameters,
                        float(config.training["max_grad_norm"]),
                    )
                    factor = _scheduled_factor(
                        global_step, total_steps=total_steps, warmup_steps=warmup_steps
                    )
                    for group, peak in zip(optimizer.param_groups, peaks, strict=True):
                        group["lr"] = peak * factor
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
            normalized = _predict_normalized(
                model, valid_loader, len(bundle.valid.raw),
                device=device, non_blocking=non_blocking,
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
                    "lora_learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "head_learning_rate": float(optimizer.param_groups[1]["lr"]),
                }
            )
            if raw_mae < best_mae:
                best_mae = raw_mae
                best_epoch = epoch
                best_state = _trainable_state(model)
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
        raise RuntimeError("LlaSMol training did not produce a best checkpoint")
    state_hash = tensor_state_hash("benchmark.llasmol-state.v1", best_state)
    model_path = root / "model.pt"
    history_path = root / "history.json"
    audit_path = root / "input_audit.json"
    atomic_torch_save(model_path, {"state_dict": best_state, "state_hash": state_hash})
    atomic_json(history_path, history)
    atomic_json(audit_path, {"train": bundle.train.audit, "valid": bundle.valid.audit})
    manifest = {
        "format_version": 1,
        "kind": "ilume_baseline_model",
        "model_kind": "llasmol",
        "training_identity": bundle.training_identity,
        "target_statistics": bundle.target_stats.to_dict(),
        "condition_statistics": bundle.condition_stats.to_dict(),
        "target_columns": list(bundle.task.target_columns),
        "view_count": bundle.train.view_count,
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
        raise FileNotFoundError(f"Missing LlaSMol checkpoint manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("format_version") != 1
        or payload.get("kind") != "ilume_baseline_model"
        or payload.get("model_kind") != "llasmol"
    ):
        raise ValueError("Unsupported LlaSMol checkpoint")
    for filename, expected in payload.get("integrity", {}).items():
        artifact = root / filename
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(expected["size"])
            or sha256_file(artifact) != expected["sha256"]
        ):
            raise ValueError(f"LlaSMol checkpoint integrity mismatch: {filename}")
    return payload


def llasmol_evaluation_audit(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
    split: str,
) -> dict[str, Any]:
    if split not in {"valid", "test"}:
        raise ValueError("LlaSMol evaluation split must be valid or test")
    task = resolve_task(config, benchmark, task_id, fold)
    raw = load_split(task, split)
    prepared = _prepare_split(
        raw, task, split, _tokenizer(config), {}, None,
        max_length=int(config.model["max_length"]),
    )
    return prepared.audit


def evaluate_llasmol_checkpoint(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
    checkpoint_dir: str | Path,
    split: str,
) -> EvaluationResult:
    if split not in {"valid", "test"}:
        raise ValueError("LlaSMol evaluation split must be valid or test")
    root = Path(checkpoint_dir)
    manifest = _manifest(root)
    bundle = prepare_llasmol_training(config, benchmark, task_id, fold)
    require_compatible_identity(
        bundle.training_identity,
        manifest["training_identity"],
        context="LlaSMol evaluation checkpoint",
    )
    seed_benchmark(config.seed)
    _configure_backend(config)
    raw = load_split(bundle.task, split)
    prepared = _prepare_split(
        raw, bundle.task, split, _tokenizer(config), bundle.token_cache,
        bundle.condition_stats, max_length=int(config.model["max_length"]),
    )
    device = torch.device(str(config.training["device"]))
    model = build_llasmol_model(config, bundle, device=device)
    payload = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    state = payload.get("state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping) or tensor_state_hash(
        "benchmark.llasmol-state.v1", state
    ) != manifest["model_state_hash"]:
        raise ValueError("LlaSMol checkpoint state hash mismatch")
    _load_trainable_state(model, state)
    loader = _data_loader(
        config, prepared, bundle.token_cache, bundle.target_stats,
        pad_token_id=bundle.pad_token_id,
    )
    normalized = _predict_normalized(
        model, loader, len(prepared.raw), device=device,
        non_blocking=bool(config.runtime["non_blocking_transfer"]),
    )
    predictions = bundle.target_stats.denormalize(normalized)
    metrics = target_metrics(
        predictions, raw.targets, bundle.task.target_columns, bundle.target_stats.scale
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
    "LLASMOL_INPUT_CONTRACT",
    "LLASMOL_TRAINING_ORDER_CONTRACT",
    "LlaSMolTrainingBundle",
    "PreparedSplit",
    "SharedLlaSMolRegressor",
    "SortishBatchSampler",
    "build_llasmol_model",
    "evaluate_llasmol_checkpoint",
    "llasmol_evaluation_audit",
    "llasmol_model_views",
    "llasmol_task_prefix",
    "prepare_llasmol_training",
    "train_llasmol_bundle",
]
