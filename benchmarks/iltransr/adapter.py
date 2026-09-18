from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import DataLoader, Sampler

from common.identity import require_compatible_identity, semantic_identity
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.outputs import repository_path
from common.progress import ProgressReporter

from benchmarks.common.config import BenchmarkConfig, BenchmarkName
from benchmarks.common.data import BenchmarkTask, RawDataset, load_split, resolve_task
from benchmarks.common.engine import EvaluationResult, TargetStats, seed_benchmark
from benchmarks.common.metrics import target_metrics
from benchmarks.iltransr.environment import iltransr_asset_snapshot

from .model import ILTransRRegressor, load_converted_transformer


ILTRANSR_INPUT_CONTRACT = {
    "canonical_identity": "ilume_isomeric_smiles",
    "model_view": "rdkit_canonical_non_isomeric",
    "tokenizer": "official_character_vocab",
    "bos": False,
    "eos": True,
    "padding_value": 0,
    "max_length": 100,
    "overflow": "official_clip_sequence_100",
    "condition_transform": "task_global_stage3_population_zscore",
    "condition_population": "five_folds_plus_test_covariates_only",
    "constant_condition": "center_then_scale_one",
}


@dataclass(frozen=True)
class ConditionStats:
    columns: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    constant: tuple[bool, ...]
    population_rows: int
    population_sha256: str

    @classmethod
    def fit(
        cls,
        columns: Sequence[str],
        values: np.ndarray,
        population_sha256: str,
    ) -> "ConditionStats":
        names = tuple(columns)
        if values.ndim != 2 or values.shape[1] != len(names) or not np.isfinite(values).all():
            raise ValueError("ILTransR condition population is not a finite matrix")
        if values.shape[0] == 0:
            if names:
                raise ValueError("ILTransR condition population is empty")
            return cls((), (), (), (), 0, population_sha256)
        mean = values.mean(axis=0)
        raw_scale = values.std(axis=0, ddof=0)
        constant = np.ptp(values, axis=0) == 0
        scale = np.where(constant, 1.0, raw_scale)
        if not np.isfinite(scale).all() or bool((scale <= 0).any()):
            raise ValueError("ILTransR condition population has invalid variance")
        return cls(
            names,
            tuple(map(float, mean)),
            tuple(map(float, scale)),
            tuple(map(bool, constant)),
            int(values.shape[0]),
            population_sha256,
        )

    def normalize(self, values: np.ndarray) -> np.ndarray:
        if values.ndim != 2 or values.shape[1] != len(self.columns):
            raise ValueError("ILTransR condition shape differs from registry")
        if not self.columns:
            return np.empty((len(values), 0), dtype=np.float32)
        return (
            (values - np.asarray(self.mean)) / np.asarray(self.scale)
        ).astype(np.float32)


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
class ILTransRTrainingBundle:
    task: BenchmarkTask
    train: PreparedSplit
    valid: PreparedSplit
    target_stats: TargetStats
    condition_stats: ConditionStats
    source_hashes: dict[str, Any]
    training_identity: dict[str, Any]
    assets: dict[str, Any]
    vocabulary: "CharacterVocabulary"
    token_cache: dict[str, tuple[torch.Tensor, int, dict[str, int]]]
    recipe: dict[str, Any]


class CharacterVocabulary:
    def __init__(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.token_to_idx = {str(key): int(value) for key, value in payload["token_to_idx"].items()}
        if len(self.token_to_idx) != 72:
            raise ValueError("ILTransR source vocabulary must contain 72 tokens")
        expected = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
        if any(self.token_to_idx.get(key) != value for key, value in expected.items()):
            raise ValueError("ILTransR source vocabulary special-token IDs changed")
        self.unk_id = 0
        self.eos_id = 3

    def encode(self, smiles: str) -> tuple[torch.Tensor, int, dict[str, int]]:
        unknown = sum(character not in self.token_to_idx for character in smiles)
        original_length = len(smiles) + 1
        values = [self.token_to_idx.get(character, self.unk_id) for character in smiles]
        values.append(self.eos_id)
        values = values[:100]
        return (
            torch.tensor(values, dtype=torch.long),
            len(values),
            {
                "characters": len(smiles),
                "tokens_before_clip": original_length,
                "clipped": int(original_length > 100),
                "eos_clipped": int(original_length > 100),
                "unknown_tokens": unknown,
            },
        )


def _canonical_non_isomeric(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("ILTransR could not parse an ILUME canonical SMILES")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def iltransr_model_views(
    task: BenchmarkTask,
    components: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    slots = tuple(task.slots)
    values = dict(zip(slots, components, strict=True))
    if slots == ("cation", "anion"):
        return (
            (_canonical_non_isomeric(f"{values['cation']}.{values['anion']}"),),
            ("ionic_liquid",),
        )
    if slots == ("cation", "anion", "solute"):
        return (
            (
                _canonical_non_isomeric(f"{values['cation']}.{values['anion']}"),
                _canonical_non_isomeric(values["solute"]),
            ),
            ("ionic_liquid", "solute"),
        )
    if slots == ("solute", "solvent"):
        return (
            (_canonical_non_isomeric(values["solute"]), _canonical_non_isomeric(values["solvent"])),
            ("solute", "solvent"),
        )
    raise ValueError(f"ILTransR does not support registry topology: {slots}")


def _topology(task: BenchmarkTask) -> str:
    if not task.condition_columns:
        return "smiles_only"
    if "temperature_K" not in task.condition_columns:
        raise ValueError("ILTransR conditioned topology requires registry temperature")
    return "temperature_pressure" if "pressure_kPa" in task.condition_columns else "temperature"


def resolve_iltransr_recipe(config: BenchmarkConfig, task_id: str) -> dict[str, Any]:
    official = config.training["official_recipes"]
    if task_id in official:
        return {**official[task_id], "source": "official_notebook"}
    return {**config.training["fallback_recipe"], "source": "registered_fallback"}


def _condition_population(task: BenchmarkTask) -> tuple[np.ndarray, str]:
    paths = sorted({*task.train_paths, *task.valid_paths, task.test_path}, key=lambda path: path.as_posix())
    rows: list[list[float]] = []
    row_count = 0
    digest = hashlib.sha256()
    digest.update("\t".join(task.condition_columns).encode())
    for path in paths:
        digest.update(path.as_posix().encode())
        if not path.is_file():
            digest.update(b"<missing-empty-split>")
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = set(task.condition_columns) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"ILTransR condition source missing columns: {sorted(missing)}")
            for row_number, row in enumerate(reader, start=2):
                row_count += 1
                values = [float(row[column]) for column in task.condition_columns]
                if not np.isfinite(values).all():
                    raise ValueError(f"ILTransR condition is non-finite in {path}:{row_number}")
                rows.append(values)
                digest.update(str(row_number).encode())
                digest.update(np.asarray(values, dtype=np.float64).tobytes())
    values = (
        np.empty((row_count, 0), dtype=np.float64)
        if not task.condition_columns
        else np.asarray(rows, dtype=np.float64)
    )
    return values, digest.hexdigest()


def _prepare_split(
    task: BenchmarkTask,
    raw: RawDataset,
    condition_stats: ConditionStats,
    vocabulary: CharacterVocabulary,
    token_cache: dict[str, tuple[torch.Tensor, int, dict[str, int]]],
) -> PreparedSplit:
    model_views: list[tuple[str, ...]] = []
    view_names: tuple[str, ...] | None = None
    role_audit: dict[str, dict[str, Any]] = {}
    role_lengths: dict[str, list[int]] = {}
    for components in raw.components:
        views, names = iltransr_model_views(task, components)
        if view_names is None:
            view_names = names
        elif names != view_names:
            raise ValueError("ILTransR view topology changed within one split")
        model_views.append(views)
        for role, smiles in zip(names, views, strict=True):
            if smiles not in token_cache:
                token_cache[smiles] = vocabulary.encode(smiles)
            details = token_cache[smiles][2]
            role_lengths.setdefault(role, []).append(details["tokens_before_clip"])
            totals = role_audit.setdefault(
                role,
                {"rows": 0, "max_characters": 0, "max_tokens_before_clip": 0, "clipped": 0, "eos_clipped": 0, "unknown_tokens": 0},
            )
            totals["rows"] += 1
            totals["max_characters"] = max(totals["max_characters"], details["characters"])
            totals["max_tokens_before_clip"] = max(totals["max_tokens_before_clip"], details["tokens_before_clip"])
            for key in ("clipped", "eos_clipped", "unknown_tokens"):
                totals[key] += details[key]
    names = view_names or ()
    for role, lengths in role_lengths.items():
        values = np.asarray(lengths, dtype=np.float64)
        role_audit[role]["token_length_distribution"] = {
            "min": int(values.min()),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": int(values.max()),
        }
    return PreparedSplit(
        raw=raw,
        model_views=tuple(model_views),
        view_names=names,
        normalized_conditions=condition_stats.normalize(raw.conditions),
        audit={
            "rows": len(raw),
            "view_names": list(names),
            "view_order": list(names),
            "topology": _topology(task),
            "condition_columns": list(task.condition_columns),
            "condition_statistics": asdict(condition_stats),
            "roles": role_audit,
            "input_contract": ILTRANSR_INPUT_CONTRACT,
        },
    )


def prepare_iltransr_training(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
) -> ILTransRTrainingBundle:
    if config.name != "iltransr":
        raise ValueError("ILTransR adapter requires name=iltransr")
    task = resolve_task(config, benchmark, task_id, fold)
    if len(task.target_columns) != 1:
        raise ValueError("ILTransR requires one scalar target")
    train_raw = load_split(task, "train")
    valid_raw = load_split(task, "valid")
    population, population_hash = _condition_population(task)
    condition_stats = ConditionStats.fit(task.condition_columns, population, population_hash)
    assets = iltransr_asset_snapshot(config)
    vocabulary = CharacterVocabulary(repository_path(config.model["source_vocab"]))
    cache: dict[str, tuple[torch.Tensor, int, dict[str, int]]] = {}
    train = _prepare_split(task, train_raw, condition_stats, vocabulary, cache)
    valid = _prepare_split(task, valid_raw, condition_stats, vocabulary, cache)
    target_stats = TargetStats.fit(train_raw.targets)
    recipe = resolve_iltransr_recipe(config, task_id)
    source_hashes = {
        "train": [sha256_file(path) for path in task.train_paths],
        "valid": [sha256_file(path) for path in task.valid_paths],
        "condition_population_sources": [
            path.as_posix()
            for path in sorted(
                {*task.train_paths, *task.valid_paths, task.test_path},
                key=lambda value: value.as_posix(),
            )
        ],
        "condition_population_sha256": population_hash,
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
            "component_order": list(task.slots),
            "view_order": list(train.view_names),
            "input_contract": ILTRANSR_INPUT_CONTRACT,
            "condition_statistics": asdict(condition_stats),
            "target_statistics": target_stats.to_dict(),
            "recipe": recipe,
            "model": config.model,
            "training": config.training,
            "seed": config.seed,
            "upstream_assets": assets,
        },
    )
    return ILTransRTrainingBundle(
        task, train, valid, target_stats, condition_stats, source_hashes,
        identity, assets, vocabulary, cache, recipe,
    )


class TwoBucketBatchSampler(Sampler[list[int]]):
    def __init__(self, lengths: Sequence[int], *, batch_size: int, seed: int) -> None:
        self.lengths = tuple(map(int, lengths))
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        if not self.lengths or min(self.lengths) <= 0 or self.batch_size <= 0:
            raise ValueError("ILTransR bucket sampling requires positive lengths and batch size")

    def _buckets(self) -> list[list[int]]:
        minimum = min(self.lengths)
        maximum = max(self.lengths)
        width = max((1 + maximum - minimum) // 2, 1)
        short_limit = max(maximum - width, minimum)
        if short_limit == maximum:
            return [list(range(len(self.lengths)))]
        return [
            [index for index, length in enumerate(self.lengths) if length <= short_limit],
            [index for index, length in enumerate(self.lengths) if length > short_limit],
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return sum(
            math.ceil(len(bucket) / self.batch_size)
            for bucket in self._buckets()
            if bucket
        )

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batches: list[list[int]] = []
        for bucket in self._buckets():
            if bucket:
                order = torch.randperm(len(bucket), generator=generator).tolist()
                shuffled = [bucket[index] for index in order]
                batches.extend(
                    shuffled[start : start + self.batch_size]
                    for start in range(0, len(shuffled), self.batch_size)
                )
        batch_order = torch.randperm(len(batches), generator=generator).tolist()
        return iter(batches[index] for index in batch_order)


def _collate(
    indices: Sequence[int],
    *,
    prepared: PreparedSplit,
    target_stats: TargetStats,
    cache: Mapping[str, tuple[torch.Tensor, int, dict[str, int]]],
    minimum_length: int,
    include_labels: bool,
) -> dict[str, torch.Tensor]:
    flattened = [
        prepared.model_views[index][view]
        for view in range(prepared.view_count)
        for index in indices
    ]
    encoded = [cache[smiles] for smiles in flattened]
    padded_length = max(minimum_length, max((value[1] for value in encoded), default=0))
    token_ids = torch.zeros((len(encoded), padded_length), dtype=torch.long)
    valid_lengths = torch.empty((len(encoded),), dtype=torch.long)
    for row, (tokens, length, _) in enumerate(encoded):
        token_ids[row, :length] = tokens
        valid_lengths[row] = length
    batch = {
        "token_ids": token_ids,
        "valid_lengths": valid_lengths,
        "conditions": torch.from_numpy(prepared.normalized_conditions[np.asarray(indices)]),
    }
    if include_labels:
        normalized = target_stats.normalize(prepared.raw.targets[np.asarray(indices)])[:, 0]
        batch["labels"] = torch.from_numpy(normalized)
    return batch


def _loader(
    prepared: PreparedSplit,
    bundle: ILTransRTrainingBundle,
    *,
    batch_size: int,
    model: ILTransRRegressor,
    seed: int,
    epoch: int,
    training: bool,
) -> DataLoader[Any]:
    indices = list(range(len(prepared.raw)))
    if training and model.topology == "smiles_only":
        lengths = [max(bundle.token_cache[value][1] for value in views) for views in prepared.model_views]
        sampler: Sampler[list[int]] = TwoBucketBatchSampler(lengths, batch_size=batch_size, seed=seed)
        sampler.set_epoch(epoch)  # type: ignore[attr-defined]
    else:
        sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(indices), batch_size=batch_size, drop_last=False
        )
    return DataLoader(
        indices,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=lambda rows: _collate(
            rows,
            prepared=prepared,
            target_stats=bundle.target_stats,
            cache=bundle.token_cache,
            minimum_length=model.textcnn.maximum_kernel,
            include_labels=training,
        ),
    )


def build_iltransr_model(
    config: BenchmarkConfig,
    bundle: ILTransRTrainingBundle,
) -> ILTransRRegressor:
    transformer, load_audit = load_converted_transformer(
        str(repository_path(config.model["converted_checkpoint"])),
        vocab_size=72,
    )
    model = ILTransRRegressor(
        transformer,
        topology=_topology(bundle.task),
        view_count=bundle.train.view_count,
        condition_dim=len(bundle.task.condition_columns),
        dropout=float(bundle.recipe["dropout"]),
        load_audit=load_audit,
    )
    frozen = [name for name, parameter in model.transformer.named_parameters() if not parameter.requires_grad]
    if frozen:
        raise RuntimeError(f"ILTransR pretrained transformer is frozen: {frozen}")
    return model


def _move(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _configure_fp32() -> None:
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"


def _predict(
    model: ILTransRRegressor,
    loader: DataLoader[Any],
    target_stats: TargetStats,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    values = []
    with torch.inference_mode():
        for batch in loader:
            values.append(model(**_move(batch, device)).cpu().numpy())
    normalized = np.concatenate(values) if values else np.empty((0,), dtype=np.float32)
    return target_stats.denormalize(normalized.reshape(-1, 1))


def train_iltransr_bundle(
    config: BenchmarkConfig,
    bundle: ILTransRTrainingBundle,
    output_dir: str | Path,
    *,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(config.training["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ILTransR requires CUDA; no silent CPU fallback")
    effective_seed = config.seed + int(bundle.task.fold or 1) - 1
    seed_benchmark(effective_seed)
    _configure_fp32()
    model = build_iltransr_model(config, bundle).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.training["learning_rate"]),
        betas=tuple(map(float, config.training["betas"])),
        eps=float(config.training["eps"]),
        weight_decay=float(config.training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config.training["lr_decay_epochs"]),
        gamma=float(config.training["lr_decay_factor"]),
    )
    epochs = int(bundle.recipe["epochs"])
    batch_size = int(bundle.recipe["batch_size"])
    valid_loader = _loader(
        bundle.valid, bundle, batch_size=batch_size, model=model,
        seed=effective_seed, epoch=0, training=False,
    )
    watched = {
        "embedding": model.transformer.embedding.weight.detach().cpu().clone(),
        "textcnn": model.textcnn.convolutions[0].weight.detach().cpu().clone(),
        "head": next(module.weight for module in model.predictor if isinstance(module, torch.nn.Linear)).detach().cpu().clone(),
    }
    for index, layer in enumerate(model.transformer.layers):
        watched[f"layer{index}_attention"] = layer.query.weight.detach().cpu().clone()
        watched[f"layer{index}_ffn"] = layer.ffn_1.weight.detach().cpu().clone()
    history = []
    progress = (reporter or ProgressReporter()).bar(
        total=epochs,
        desc=f"ILTransR {bundle.task.task_id} fold{bundle.task.fold}",
        unit="epoch",
    )
    try:
        for epoch in range(1, epochs + 1):
            train_loader = _loader(
                bundle.train, bundle, batch_size=batch_size, model=model,
                seed=effective_seed, epoch=epoch - 1, training=True,
            )
            model.train()
            loss_sum = 0.0
            for batch in train_loader:
                labels = batch.pop("labels").to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(**_move(batch, device))
                absolute_error = torch.abs(predictions - labels).sum()
                loss = absolute_error / batch_size
                if not torch.isfinite(loss):
                    raise FloatingPointError("ILTransR training loss is non-finite")
                loss.backward()
                optimizer.step()
                loss_sum += float(absolute_error.detach().cpu())
            raw_predictions = _predict(model, valid_loader, bundle.target_stats, device)
            valid_mae = float(np.abs(raw_predictions[:, 0] - bundle.valid.raw.targets[:, 0]).mean())
            history.append(
                {
                    "epoch": epoch,
                    "train_normalized_l1": loss_sum / len(bundle.train.raw),
                    "valid_raw_mae": valid_mae,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            scheduler.step()
            progress.set_postfix({"train_l1": f"{history[-1]['train_normalized_l1']:.4f}", "val_mae": f"{valid_mae:.4f}"})
            progress.update(1)
    finally:
        progress.close()
    current = {
        "embedding": model.transformer.embedding.weight.detach().cpu(),
        "textcnn": model.textcnn.convolutions[0].weight.detach().cpu(),
        "head": next(module.weight for module in model.predictor if isinstance(module, torch.nn.Linear)).detach().cpu(),
    }
    for index, layer in enumerate(model.transformer.layers):
        current[f"layer{index}_attention"] = layer.query.weight.detach().cpu()
        current[f"layer{index}_ffn"] = layer.ffn_1.weight.detach().cpu()
    parameter_audit = {
        "pretrained_transformer_fully_trainable": all(
            parameter.requires_grad for parameter in model.transformer.parameters()
        ),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "selected_parameter_max_abs_delta": {
            name: float((current[name] - initial).abs().max()) for name, initial in watched.items()
        },
    }
    if not parameter_audit["pretrained_transformer_fully_trainable"] or any(
        value <= 0 for value in parameter_audit["selected_parameter_max_abs_delta"].values()
    ):
        raise RuntimeError("ILTransR full-fine-tuning parameter audit failed")
    atomic_json(root / "training_history.json", history)
    model_file = root / f"checkpoint_epoch_{epochs:05d}.pt"
    atomic_torch_save(model_file, {name: value.detach().cpu() for name, value in model.state_dict().items()})
    integrity = {
        path.name: {"sha256": sha256_file(path), "size": path.stat().st_size}
        for path in (root / "training_history.json", model_file)
    }
    manifest = {
        "format_version": 2,
        "kind": "ilume_baseline_model",
        "model_kind": "iltransr",
        "training_identity": bundle.training_identity,
        "target_statistics": bundle.target_stats.to_dict(),
        "condition_statistics": asdict(bundle.condition_stats),
        "target_columns": list(bundle.task.target_columns),
        "recipe": bundle.recipe,
        "effective_seed": effective_seed,
        "model_selection": "final_training_state",
        "final_epoch": epochs,
        "final_valid_raw_mae": history[-1]["valid_raw_mae"],
        "final_model_file": model_file.name,
        "parameter_audit": parameter_audit,
        "input_audit": {"train": bundle.train.audit, "valid": bundle.valid.audit},
        "upstream_assets": bundle.assets,
        "pretrained_modules": ["source_embedding", "transformer_encoder"],
        "ignored_pretraining_modules": ["decoder", "one_step_ahead_decoder", "target_embedding", "target_projection"],
        "property_specific_weights_loaded": False,
        "integrity": integrity,
    }
    atomic_json(root / "checkpoint.json", manifest)
    return {
        "final_epoch": epochs,
        "epochs_ran": len(history),
        "final_valid_raw_mae": history[-1]["valid_raw_mae"],
        "model_selector": "final_training_state",
    }


def _torch_load(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def evaluate_iltransr_checkpoint(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
    checkpoint_dir: str | Path,
    split: str,
) -> EvaluationResult:
    if split not in {"valid", "test"}:
        raise ValueError("ILTransR evaluation split must be valid or test")
    root = Path(checkpoint_dir)
    manifest = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    if manifest.get("model_kind") != "iltransr" or manifest.get("model_selection") != "final_training_state":
        raise ValueError("Unsupported ILTransR checkpoint")
    for filename, expected in manifest["integrity"].items():
        path = root / filename
        if not path.is_file() or path.stat().st_size != expected["size"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"ILTransR checkpoint integrity mismatch: {filename}")
    bundle = prepare_iltransr_training(config, benchmark, task_id, fold)
    require_compatible_identity(bundle.training_identity, manifest["training_identity"], context="ILTransR evaluation checkpoint")
    raw = load_split(bundle.task, split)
    prepared = _prepare_split(bundle.task, raw, bundle.condition_stats, bundle.vocabulary, bundle.token_cache)
    device = torch.device(str(config.training["device"]))
    _configure_fp32()
    model = build_iltransr_model(config, bundle)
    model.load_state_dict(_torch_load(root / manifest["final_model_file"]), strict=True)
    model.to(device)
    loader = _loader(
        prepared, bundle, batch_size=int(bundle.recipe["batch_size"]), model=model,
        seed=int(manifest["effective_seed"]), epoch=0, training=False,
    )
    predictions = _predict(model, loader, bundle.target_stats, device)
    return EvaluationResult(
        predictions=predictions,
        targets=raw.targets,
        source_rows=raw.source_rows,
        metrics=target_metrics(predictions, raw.targets, bundle.task.target_columns, bundle.target_stats.scale),
        target_stats=bundle.target_stats,
        training_identity=bundle.training_identity,
        components=raw.components,
        conditions=raw.conditions,
        audit_rows=raw.audit_rows,
        input_audit=prepared.audit,
    )


def iltransr_evaluation_audit(
    config: BenchmarkConfig,
    benchmark: BenchmarkName,
    task_id: str,
    fold: int | None,
    split: str,
) -> dict[str, Any]:
    task = resolve_task(config, benchmark, task_id, fold)
    population, population_hash = _condition_population(task)
    stats = ConditionStats.fit(task.condition_columns, population, population_hash)
    vocabulary = CharacterVocabulary(repository_path(config.model["source_vocab"]))
    return _prepare_split(task, load_split(task, split), stats, vocabulary, {}).audit


__all__ = [
    "CharacterVocabulary",
    "ConditionStats",
    "ILTRANSR_INPUT_CONTRACT",
    "ILTransRTrainingBundle",
    "PreparedSplit",
    "TwoBucketBatchSampler",
    "_collate",
    "_prepare_split",
    "build_iltransr_model",
    "evaluate_iltransr_checkpoint",
    "iltransr_evaluation_audit",
    "iltransr_model_views",
    "prepare_iltransr_training",
    "resolve_iltransr_recipe",
    "train_iltransr_bundle",
]
