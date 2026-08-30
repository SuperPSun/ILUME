from __future__ import annotations

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
from benchmarks.common.metrics import target_metrics


MOLFORMER_INPUT_CONTRACT = {
    "canonical_identity": "ilume_isomeric_smiles",
    "model_input": "rdkit_canonical_isomeric_false",
    "token_length_includes_special_tokens": True,
    "max_input_tokens": 202,
    "train_overlength": "skip_row",
    "validation_overlength": "truncate_component",
    "test_overlength": "truncate_component",
    "truncation_keeps_row_eligible": True,
    "input_cache": "unique_smiles_memory_token_cache",
    "component_forward": "merged_component_backbone_forward",
}
MOLFORMER_TRAINING_ORDER_CONTRACT = "sortish_length_bucketing_v1"
TokenCache = dict[
    str,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]
_SNAPSHOT_FILES = (
    "config.json",
    "configuration_molformer.py",
    "model.safetensors",
    "modeling_molformer.py",
    "tokenizer.json",
    "tokenizer_config.json",
)


@dataclass(frozen=True)
class ConditionStats:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, values: np.ndarray) -> "ConditionStats":
        if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
            raise ValueError("MoLFormer train conditions must be a non-empty finite matrix")
        if values.shape[1] == 0:
            return cls((), ())
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        return cls(tuple(map(float, mean)), tuple(map(float, scale)))

    def normalize(self, values: np.ndarray) -> np.ndarray:
        if values.ndim != 2 or values.shape[1] != len(self.mean) or not np.isfinite(values).all():
            raise ValueError("MoLFormer condition shape or values differ from training")
        if not self.mean:
            return np.empty((len(values), 0), dtype=np.float32)
        return (
            (values - np.asarray(self.mean)) / np.asarray(self.scale)
        ).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ConditionStats":
        return cls(
            tuple(float(value) for value in raw["mean"]),
            tuple(float(value) for value in raw["scale"]),
        )


@dataclass(frozen=True)
class PreparedSplit:
    raw: RawDataset
    model_components: tuple[tuple[str, ...], ...]
    normalized_conditions: np.ndarray
    audit: dict[str, Any]


@dataclass
class MolFormerTrainingBundle:
    task: BenchmarkTask
    train: PreparedSplit
    valid: PreparedSplit
    target_stats: TargetStats
    condition_stats: ConditionStats
    source_hashes: dict[str, Any]
    training_identity: dict[str, Any]
    snapshot: dict[str, Any]
    token_cache: TokenCache
    pad_token_id: int


class SharedMolFormerRegressor(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        classifier: torch.nn.Module,
        *,
        component_count: int,
        condition_dim: int,
        hidden_dim: int,
        initializer_range: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.component_count = component_count
        self.condition_dim = condition_dim
        fusion_dim = component_count * hidden_dim + condition_dim
        self.fusion = (
            None
            if component_count == 1 and condition_dim == 0
            else torch.nn.Linear(fusion_dim, hidden_dim)
        )
        if self.fusion is not None:
            torch.nn.init.normal_(self.fusion.weight, mean=0.0, std=initializer_range)
            torch.nn.init.zeros_(self.fusion.bias)

    def forward(
        self,
        component_inputs: Mapping[str, torch.Tensor],
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(conditions.shape[0])
        pooled = self.backbone(**component_inputs).pooler_output
        if int(pooled.shape[0]) != self.component_count * batch_size:
            raise ValueError("MoLFormer merged molecular batch differs from model topology")
        ordered = pooled.reshape(self.component_count, batch_size, -1).transpose(0, 1)
        representation = ordered[:, 0]
        if self.fusion is not None:
            representation = self.fusion(
                torch.cat([ordered.reshape(batch_size, -1), conditions], dim=1)
            )
        return self.classifier(representation)


def _snapshot(config: BenchmarkConfig) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    repository = str(config.model["repository"])
    revision = str(config.model["revision"])
    try:
        root = Path(
            snapshot_download(
                repo_id=repository,
                revision=revision,
                local_files_only=True,
            )
        )
    except Exception as error:
        raise RuntimeError(
            f"MoLFormer snapshot {repository}@{revision} is not available locally"
        ) from error
    missing = [name for name in _SNAPSHOT_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("MoLFormer snapshot is incomplete: " + ", ".join(missing))
    payload = {
        "repository": repository,
        "revision": revision,
        "files": {
            name: {
                "sha256": sha256_file(root / name),
                "size": (root / name).stat().st_size,
            }
            for name in _SNAPSHOT_FILES
        },
    }
    return root, payload


def _tokenizer(config: BenchmarkConfig) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(config.model["repository"]),
        revision=str(config.model["revision"]),
        trust_remote_code=bool(config.model["trust_remote_code"]),
        local_files_only=True,
    )


def model_input_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid ILUME canonical SMILES for MoLFormer: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _populate_token_cache(
    tokenizer: Any,
    smiles_values: Sequence[str],
    token_cache: TokenCache,
    *,
    max_tokens: int = 202,
) -> None:
    missing = sorted(set(smiles_values) - set(token_cache))
    for start in range(0, len(missing), 4096):
        values = missing[start : start + 4096]
        encoded = tokenizer(
            values,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=True,
        )
        masks = encoded.get("attention_mask")
        for index, smiles in enumerate(values):
            input_ids = torch.as_tensor(encoded["input_ids"][index], dtype=torch.long)
            attention_mask = (
                torch.ones_like(input_ids)
                if masks is None
                else torch.as_tensor(masks[index], dtype=torch.long)
            )
            if input_ids.ndim != 1 or not len(input_ids):
                raise ValueError("MoLFormer tokenizer returned an empty or non-vector input")
            if attention_mask.shape != input_ids.shape:
                raise ValueError("MoLFormer tokenizer returned a mismatched attention mask")
            truncated_ids = input_ids
            truncated_mask = attention_mask
            if len(input_ids) > max_tokens:
                if (
                    tokenizer.truncation_side != "right"
                    or tokenizer.eos_token_id is None
                    or int(input_ids[-1]) != int(tokenizer.eos_token_id)
                ):
                    raise ValueError("MoLFormer tokenizer truncation contract is unsupported")
                truncated_ids = torch.cat(
                    (input_ids[: max_tokens - 1], input_ids[-1:])
                )
                truncated_mask = torch.cat(
                    (attention_mask[: max_tokens - 1], attention_mask[-1:])
                )
            token_cache[smiles] = (
                input_ids,
                attention_mask,
                truncated_ids,
                truncated_mask,
            )


def _token_cache_audit(token_cache: TokenCache) -> dict[str, int | str]:
    lengths = [len(values[0]) for values in token_cache.values()]
    return {
        "contract": "unique_smiles_memory_token_cache",
        "unique_model_inputs": len(lengths),
        "total_tokens": sum(lengths),
        "maximum_token_length": max(lengths, default=0),
    }


def _input_audit(
    raw: RawDataset,
    task: BenchmarkTask,
    split: str,
    tokenizer: Any,
    token_cache: TokenCache,
    *,
    max_tokens: int,
) -> tuple[tuple[tuple[str, ...], ...], dict[str, Any]]:
    model_components = tuple(
        tuple(model_input_smiles(smiles) for smiles in components)
        for components in raw.components
    )
    unique_benchmark = sorted({value for row in raw.components for value in row})
    mapping: dict[str, set[str]] = {}
    for benchmark_row, model_row in zip(raw.components, model_components, strict=True):
        for benchmark_smiles, model_smiles in zip(benchmark_row, model_row, strict=True):
            mapping.setdefault(model_smiles, set()).add(benchmark_smiles)
    collisions = {
        model_smiles: sorted(benchmark_smiles)
        for model_smiles, benchmark_smiles in mapping.items()
        if len(benchmark_smiles) > 1
    }
    collision_inputs = set(collisions)
    _populate_token_cache(
        tokenizer, tuple(mapping), token_cache, max_tokens=max_tokens
    )
    lengths = {smiles: len(token_cache[smiles][0]) for smiles in mapping}
    overlength: list[dict[str, Any]] = []
    collision_rows: set[str] = set()
    overlength_rows: set[str] = set()
    for source_row, benchmark_row, model_row in zip(
        raw.source_rows, raw.components, model_components, strict=True
    ):
        for slot, benchmark_smiles, model_smiles in zip(
            task.slots, benchmark_row, model_row, strict=True
        ):
            if model_smiles in collision_inputs:
                collision_rows.add(source_row)
            length = lengths[model_smiles]
            if length > max_tokens:
                overlength_rows.add(source_row)
                overlength.append(
                    {
                        "source_row": source_row,
                        "slot": slot,
                        "benchmark_canonical_smiles": benchmark_smiles,
                        "molformer_input_smiles": model_smiles,
                        "token_length": length,
                    }
                )
    affected_molecules = {
        smiles for values in collisions.values() for smiles in values
    }
    audit = {
        "contract": MOLFORMER_INPUT_CONTRACT,
        "task": task.task_id,
        "split": split,
        "total_rows": len(raw),
        "unique_benchmark_molecules": len(unique_benchmark),
        "unique_model_inputs": len(mapping),
        "unique_model_input_tokens": sum(lengths.values()),
        "maximum_token_length": max(lengths.values(), default=0),
        "collision_group_count": len(collisions),
        "collision_affected_molecules": len(affected_molecules),
        "collision_affected_rows": len(collision_rows),
        "collision_groups": collisions,
        "overlength_component_count": len(overlength),
        "overlength_row_count": len(overlength_rows),
        "overlength_components": overlength,
        "retained_rows": len(raw) - len(overlength_rows) if split == "train" else len(raw),
        "skipped_rows": sorted(overlength_rows) if split == "train" else [],
        "truncated_rows": sorted(overlength_rows) if split != "train" else [],
    }
    return model_components, audit


def _select_rows(raw: RawDataset, indices: Sequence[int]) -> RawDataset:
    selected = np.asarray(indices, dtype=np.int64)
    return RawDataset(
        components=tuple(raw.components[index] for index in indices),
        component_count=raw.component_count,
        conditions=raw.conditions[selected],
        targets=raw.targets[selected],
        source_rows=tuple(raw.source_rows[index] for index in indices),
        audit_rows=tuple(raw.audit_rows[index] for index in indices),
    )


def _prepare_split(
    raw: RawDataset,
    task: BenchmarkTask,
    split: str,
    tokenizer: Any,
    token_cache: TokenCache,
    condition_stats: ConditionStats | None,
    *,
    max_tokens: int,
) -> PreparedSplit:
    model_components, audit = _input_audit(
        raw, task, split, tokenizer, token_cache, max_tokens=max_tokens
    )
    if split == "train" and audit["skipped_rows"]:
        skipped = set(audit["skipped_rows"])
        indices = [
            index for index, source_row in enumerate(raw.source_rows)
            if source_row not in skipped
        ]
        raw = _select_rows(raw, indices)
        model_components = tuple(model_components[index] for index in indices)
    if split == "train" and not len(raw):
        raise ValueError("MoLFormer retained no training rows after token-length filtering")
    normalized_conditions = (
        np.empty((len(raw), raw.conditions.shape[1]), dtype=np.float32)
        if condition_stats is None
        else condition_stats.normalize(raw.conditions)
    )
    return PreparedSplit(raw, model_components, normalized_conditions, audit)


def _lock_sha(config: BenchmarkConfig) -> str:
    if config.environment is None:
        raise ValueError("MoLFormer environment contract is missing")
    return sha256_file(repository_path(config.environment.lock))


def prepare_molformer_training(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
) -> MolFormerTrainingBundle:
    if config.name != "molformer":
        raise ValueError("MoLFormer adapter requires name=molformer")
    task = resolve_task(config, benchmark, task_id, fold)
    if len(task.target_columns) != 1:
        raise ValueError("MoLFormer baseline requires one scalar target per task")
    tokenizer = _tokenizer(config)
    if tokenizer.pad_token_id is None:
        raise ValueError("MoLFormer tokenizer does not define a pad token")
    token_cache: TokenCache = {}
    max_tokens = int(config.model["max_input_tokens"])
    raw_train = load_split(task, "train")
    provisional_train = _prepare_split(
        raw_train,
        task,
        "train",
        tokenizer,
        token_cache,
        None,
        max_tokens=max_tokens,
    )
    condition_stats = ConditionStats.fit(provisional_train.raw.conditions)
    train = PreparedSplit(
        provisional_train.raw,
        provisional_train.model_components,
        condition_stats.normalize(provisional_train.raw.conditions),
        provisional_train.audit,
    )
    target_stats = TargetStats.fit(train.raw.targets)
    valid = _prepare_split(
        load_split(task, "valid"),
        task,
        "valid",
        tokenizer,
        token_cache,
        condition_stats,
        max_tokens=max_tokens,
    )
    _, snapshot = _snapshot(config)
    source_hashes = {
        "train": [sha256_file(path) for path in task.train_paths],
        "valid": [sha256_file(path) for path in task.valid_paths],
    }
    identity = semantic_identity(
        "benchmark.training.v1",
        {
            "benchmark_model": "molformer",
            "domain": task.benchmark,
            "task_id": task.task_id,
            "fold": task.fold,
            "registry": task.registry_payload,
            "component_order": list(task.slots),
            "source_hashes": source_hashes,
            "input_contract": MOLFORMER_INPUT_CONTRACT,
            "training_order_contract": MOLFORMER_TRAINING_ORDER_CONTRACT,
            "token_cache_audit": _token_cache_audit(token_cache),
            "train_input_audit": train.audit,
            "valid_input_audit": valid.audit,
            "condition_statistics": condition_stats.to_dict(),
            "target_statistics": target_stats.to_dict(),
            "pretrained_snapshot": snapshot,
            "model": config.model,
            "training": config.training,
            "seed": config.seed,
            "environment_lock_sha256": _lock_sha(config),
        },
    )
    return MolFormerTrainingBundle(
        task=task,
        train=train,
        valid=valid,
        target_stats=target_stats,
        condition_stats=condition_stats,
        source_hashes=source_hashes,
        training_identity=identity,
        snapshot=snapshot,
        token_cache=token_cache,
        pad_token_id=int(tokenizer.pad_token_id),
    )


def build_molformer_model(
    config: BenchmarkConfig,
    bundle: MolFormerTrainingBundle,
) -> SharedMolFormerRegressor:
    from transformers import AutoModelForSequenceClassification

    official = AutoModelForSequenceClassification.from_pretrained(
        str(config.model["repository"]),
        revision=str(config.model["revision"]),
        trust_remote_code=bool(config.model["trust_remote_code"]),
        local_files_only=True,
        num_labels=1,
        problem_type="regression",
        deterministic_eval=bool(config.model["deterministic_eval"]),
    )
    if int(official.config.hidden_size) != int(config.model["hidden_dim"]):
        raise ValueError("MoLFormer hidden size differs from the registered contract")
    if int(official.config.max_position_embeddings) != int(config.model["max_input_tokens"]):
        raise ValueError("MoLFormer positional length differs from the registered contract")
    if not bool(official.config.deterministic_eval):
        raise ValueError("MoLFormer deterministic_eval must be enabled")
    return SharedMolFormerRegressor(
        official.molformer,
        official.classifier,
        component_count=bundle.train.raw.component_count,
        condition_dim=bundle.train.raw.conditions.shape[1],
        hidden_dim=int(config.model["hidden_dim"]),
        initializer_range=float(official.config.initializer_range),
    )


def _collate(
    prepared: PreparedSplit,
    token_cache: TokenCache,
    target_stats: TargetStats,
    *,
    truncate: bool,
    max_tokens: int,
    pad_token_id: int,
) -> Any:
    def collate(
        indices: Sequence[int],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        input_ids: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []
        for component_index in range(prepared.raw.component_count):
            for index in indices:
                smiles = prepared.model_components[index][component_index]
                cached = token_cache[smiles]
                ids, mask = (
                    (cached[2], cached[3])
                    if truncate
                    else (cached[0], cached[1])
                )
                if len(ids) > max_tokens:
                    raise ValueError(
                        "MoLFormer cached input exceeded the registered sequence length"
                    )
                input_ids.append(ids)
                attention_masks.append(mask)
        encoded = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=pad_token_id
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                attention_masks, batch_first=True, padding_value=0
            ),
        }
        conditions = torch.from_numpy(prepared.normalized_conditions[np.asarray(indices)])
        targets = torch.from_numpy(
            target_stats.normalize(prepared.raw.targets[np.asarray(indices)])
        )
        return encoded, conditions, targets

    return collate


def _move_inputs(
    components: Mapping[str, torch.Tensor],
    device: torch.device,
    *,
    non_blocking: bool,
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=non_blocking)
        for name, value in components.items()
    }


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
        self.batch_size = batch_size
        self.window_batches = window_batches
        self.seed = seed
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
        order = torch.randperm(len(batches), generator=generator).tolist()
        return iter(batches[index] for index in order)


def _row_token_lengths(
    prepared: PreparedSplit,
    token_cache: TokenCache,
    *,
    max_tokens: int,
) -> tuple[int, ...]:
    return tuple(
        max(min(len(token_cache[smiles][0]), max_tokens) for smiles in components)
        for components in prepared.model_components
    )


def _data_loader(
    config: BenchmarkConfig,
    prepared: PreparedSplit,
    token_cache: TokenCache,
    target_stats: TargetStats,
    *,
    pad_token_id: int,
    batch_sampler: SortishBatchSampler | None = None,
) -> torch.utils.data.DataLoader:
    runtime = config.runtime
    common = {
        "num_workers": int(runtime["num_workers"]),
        "pin_memory": bool(runtime["pin_memory"]),
        "persistent_workers": bool(runtime["persistent_workers"]),
        "prefetch_factor": int(runtime["prefetch_factor"]),
        "collate_fn": _collate(
            prepared,
            token_cache,
            target_stats,
            truncate=prepared.audit["split"] != "train",
            max_tokens=int(config.model["max_input_tokens"]),
            pad_token_id=pad_token_id,
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


def _predict_normalized(
    model: SharedMolFormerRegressor,
    loader: torch.utils.data.DataLoader,
    row_count: int,
    *,
    device: torch.device,
    non_blocking: bool,
) -> np.ndarray:
    if not row_count:
        return np.empty((0, 1), dtype=np.float32)
    values = []
    model.eval()
    with torch.inference_mode():
        for components, conditions, _ in loader:
            prediction = model(
                _move_inputs(components, device, non_blocking=non_blocking),
                conditions.to(device, non_blocking=non_blocking),
            )
            values.append(prediction.float().cpu())
    return torch.cat(values).numpy()


def _configure_tf32(enabled: bool) -> None:
    precision = "tf32" if enabled else "ieee"
    torch.backends.cuda.matmul.fp32_precision = precision
    torch.backends.cudnn.conv.fp32_precision = precision


def train_molformer_bundle(
    config: BenchmarkConfig,
    bundle: MolFormerTrainingBundle,
    output_dir: str | Path,
    *,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    from transformers.optimization import get_cosine_schedule_with_warmup

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seed_benchmark(config.seed)
    _configure_tf32(bool(config.training["tf32"]))
    device = torch.device(str(config.training["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("MoLFormer requires CUDA; no silent CPU fallback")
    model = build_molformer_model(config, bundle).to(device)
    head_parameters = [*model.classifier.parameters()]
    if model.fusion is not None:
        head_parameters.extend(model.fusion.parameters())
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.backbone.parameters()),
                "lr": float(config.training["encoder_learning_rate"]),
            },
            {
                "params": head_parameters,
                "lr": float(config.training["new_parameter_learning_rate"]),
            },
        ],
        weight_decay=float(config.training["weight_decay"]),
    )
    batch_size = int(config.training["batch_size"])
    max_epochs = int(config.training["max_epochs"])
    train_sampler = SortishBatchSampler(
        _row_token_lengths(
            bundle.train,
            bundle.token_cache,
            max_tokens=int(config.model["max_input_tokens"]),
        ),
        batch_size=batch_size,
        window_batches=int(config.training["bucket_window_batches"]),
        seed=config.seed,
    )
    steps_per_epoch = len(train_sampler)
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = math.ceil(float(config.training["warmup_fraction"]) * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5,
    )
    train_loader = _data_loader(
        config,
        bundle.train,
        bundle.token_cache,
        bundle.target_stats,
        pad_token_id=bundle.pad_token_id,
        batch_sampler=train_sampler,
    )
    valid_loader = _data_loader(
        config,
        bundle.valid,
        bundle.token_cache,
        bundle.target_stats,
        pad_token_id=bundle.pad_token_id,
    )
    non_blocking = bool(config.runtime["non_blocking_transfer"])
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    progress = (reporter or ProgressReporter()).bar(
        total=max_epochs,
        desc=f"MoLFormer {bundle.task.task_id}",
        unit="epoch",
    )
    try:
        for epoch in range(1, max_epochs + 1):
            train_sampler.set_epoch(epoch - 1)
            model.train()
            loss_sum = 0.0
            seen = 0
            for components, conditions, targets in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(
                    _move_inputs(components, device, non_blocking=non_blocking),
                    conditions.to(device, non_blocking=non_blocking),
                )
                loss = torch.nn.functional.mse_loss(
                    prediction, targets.to(device, non_blocking=non_blocking)
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("MoLFormer training produced a non-finite loss")
                loss.backward()
                optimizer.step()
                scheduler.step()
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
            normalized_mae = float(
                np.abs(
                    normalized - bundle.target_stats.normalize(bundle.valid.raw.targets)
                ).mean()
            )
            raw_mae = normalized_mae * bundle.target_stats.scale[0]
            history.append(
                {
                    "epoch": epoch,
                    "train_normalized_mse": loss_sum / seen,
                    "valid_normalized_mae": normalized_mae,
                    "valid_raw_mae": raw_mae,
                    "encoder_learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "new_parameter_learning_rate": float(optimizer.param_groups[1]["lr"]),
                }
            )
            if normalized_mae < best_score:
                best_score = normalized_mae
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
                    "best": f"{best_score:.4f}@{best_epoch}",
                    "patience": f"{stale}/{config.training['early_stopping_patience']}",
                }
            )
            progress.update(1)
            if stale >= int(config.training["early_stopping_patience"]):
                break
    finally:
        progress.close()
    if best_state is None:
        raise RuntimeError("MoLFormer training did not produce a best checkpoint")
    state_hash = tensor_state_hash("benchmark.molformer-state.v1", best_state)
    model_path = root / "model.pt"
    atomic_torch_save(model_path, {"state_dict": best_state, "state_hash": state_hash})
    history_path = root / "history.json"
    audit_path = root / "input_audit.json"
    atomic_json(history_path, history)
    atomic_json(audit_path, {"train": bundle.train.audit, "valid": bundle.valid.audit})
    manifest = {
        "format_version": 1,
        "kind": "ilume_baseline_model",
        "model_kind": "molformer",
        "training_identity": bundle.training_identity,
        "target_statistics": bundle.target_stats.to_dict(),
        "condition_statistics": bundle.condition_stats.to_dict(),
        "target_columns": list(bundle.task.target_columns),
        "component_count": bundle.train.raw.component_count,
        "condition_dim": bundle.train.raw.conditions.shape[1],
        "best_epoch": best_epoch,
        "best_valid_normalized_mae": best_score,
        "best_valid_raw_mae": best_score * bundle.target_stats.scale[0],
        "warmup_steps": warmup_steps,
        "max_total_optimizer_steps": total_steps,
        "model_state_hash": state_hash,
        "pretrained_snapshot": bundle.snapshot,
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
        "best_valid_raw_mae": manifest["best_valid_raw_mae"],
        "epochs_ran": len(history),
        "input_audit": manifest["input_audit"],
    }


def _manifest(root: Path) -> dict[str, Any]:
    path = root / "checkpoint.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing MoLFormer checkpoint manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("format_version") != 1
        or payload.get("kind") != "ilume_baseline_model"
        or payload.get("model_kind") != "molformer"
    ):
        raise ValueError("Unsupported MoLFormer checkpoint")
    for filename, expected in payload.get("integrity", {}).items():
        artifact = root / filename
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(expected["size"])
            or sha256_file(artifact) != expected["sha256"]
        ):
            raise ValueError(f"MoLFormer checkpoint integrity mismatch: {filename}")
    return payload


def molformer_evaluation_audit(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
    split: str,
) -> dict[str, Any]:
    if split not in {"valid", "test"}:
        raise ValueError("MoLFormer evaluation split must be valid or test")
    task = resolve_task(config, benchmark, task_id, fold)
    raw = load_split(task, split)  # Test is opened only by the evaluation entrypoint.
    token_cache: TokenCache = {}
    _, audit = _input_audit(
        raw,
        task,
        split,
        _tokenizer(config),
        token_cache,
        max_tokens=int(config.model["max_input_tokens"]),
    )
    return audit


def evaluate_molformer_checkpoint(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
    checkpoint_dir: str | Path,
    split: str,
) -> EvaluationResult:
    if split not in {"valid", "test"}:
        raise ValueError("MoLFormer evaluation split must be valid or test")
    root = Path(checkpoint_dir)
    manifest = _manifest(root)
    bundle = prepare_molformer_training(config, benchmark, task_id, fold)
    require_compatible_identity(
        bundle.training_identity,
        manifest["training_identity"],
        context="MoLFormer evaluation checkpoint",
    )
    seed_benchmark(config.seed)
    _configure_tf32(bool(config.training["tf32"]))
    raw = load_split(bundle.task, split)
    tokenizer = _tokenizer(config)
    prepared = _prepare_split(
        raw,
        bundle.task,
        split,
        tokenizer,
        bundle.token_cache,
        bundle.condition_stats,
        max_tokens=int(config.model["max_input_tokens"]),
    )
    model = build_molformer_model(config, bundle)
    payload = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    if tensor_state_hash("benchmark.molformer-state.v1", payload["state_dict"]) != manifest["model_state_hash"]:
        raise ValueError("MoLFormer checkpoint state hash mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    device = torch.device(str(config.training["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("MoLFormer evaluation requires CUDA; no silent CPU fallback")
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
    "MOLFORMER_INPUT_CONTRACT",
    "MolFormerTrainingBundle",
    "PreparedSplit",
    "SharedMolFormerRegressor",
    "build_molformer_model",
    "evaluate_molformer_checkpoint",
    "model_input_smiles",
    "molformer_evaluation_audit",
    "prepare_molformer_training",
    "train_molformer_bundle",
]
