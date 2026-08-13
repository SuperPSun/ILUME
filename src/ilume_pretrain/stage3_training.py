from __future__ import annotations

import hashlib
import json
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from .progress import ProgressReporter
from .stage2_model import sha256_file
from .stage2_prepare import resolve_device
from .stage3_config import Stage3Config
from .stage3_data import Stage3TaskDataset, SystemCursor, TASK_REGISTRY
from .stage3_model import Stage3MultiDomainModel
from .stage3_prepare import load_frozen_embeddings


STAGE3_CHECKPOINT_VERSION = 2
STAGE3_TRAINING_KIND = "ilume_stage3_domain_training"
STAGE3_DOMAIN_MODEL_KIND = "ilume_stage3_domain_model"
STAGE3_MODEL_KIND = "ilume_stage3_model"


@dataclass(frozen=True)
class ResidentTaskData:
    base_kind: str
    base_indices: torch.Tensor
    conditions: torch.Tensor
    phase_ids: torch.Tensor
    targets: torch.Tensor
    solute_indices: torch.Tensor | None


class FrozenRepresentationStore:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        device: torch.device = torch.device("cpu"),
        resident: bool = False,
    ) -> None:
        self.device = device if resident else torch.device("cpu")
        self.resident = resident
        self.entity = payload["entity_embeddings"].float().to(self.device)
        self.il = payload["il_pair_embeddings"].float().to(self.device)
        self.neutral = payload["neutral_pair_embeddings"].float().to(
            self.device
        )
        self.il_index = {
            tuple(map(int, key)): index
            for index, key in enumerate(payload["il_pair_keys"].tolist())
        }
        self.neutral_index = {
            tuple(map(int, key)): index
            for index, key in enumerate(
                payload["neutral_pair_keys"].tolist()
            )
        }
        self.prepared: dict[tuple[str, int, str, str], ResidentTaskData] = {}

    @staticmethod
    def _key(dataset: Stage3TaskDataset) -> tuple[str, int, str, str]:
        return (
            dataset.domain,
            dataset.fold,
            dataset.task,
            dataset.split,
        )

    def prepare_dataset(self, dataset: Stage3TaskDataset) -> ResidentTaskData:
        key = self._key(dataset)
        existing = self.prepared.get(key)
        if existing is not None:
            return existing
        entity_ids = dataset.entity_ids
        topology = TASK_REGISTRY[dataset.task].topology
        if topology in {"il", "il_solute"}:
            base_kind = "il"
            base_indices = torch.tensor(
                [
                    self.il_index[tuple(map(int, row[:2]))]
                    for row in entity_ids.tolist()
                ],
                dtype=torch.long,
            )
        elif topology == "neutral_pair":
            base_kind = "neutral"
            base_indices = torch.tensor(
                [
                    self.neutral_index[tuple(map(int, row[:2]))]
                    for row in entity_ids.tolist()
                ],
                dtype=torch.long,
            )
        else:
            base_kind = "entity"
            base_indices = entity_ids[:, 0].clone()
        solute_indices = (
            entity_ids[:, 2].clone() if topology == "il_solute" else None
        )
        prepared = ResidentTaskData(
            base_kind=base_kind,
            base_indices=base_indices.to(self.device),
            conditions=dataset.conditions.to(self.device),
            phase_ids=dataset.phase_ids.to(self.device),
            targets=dataset.targets.to(self.device),
            solute_indices=(
                solute_indices.to(self.device)
                if solute_indices is not None
                else None
            ),
        )
        self.prepared[key] = prepared
        return prepared

    def view_for(self, dataset: Stage3TaskDataset) -> ResidentTaskData:
        return self.prepare_dataset(dataset)

    def batch(
        self,
        task: str,
        dataset: Stage3TaskDataset,
        indices: torch.Tensor,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        if task != dataset.task:
            raise ValueError("Stage 3 store task/dataset mismatch")
        prepared = self.prepare_dataset(dataset)
        resident_indices = indices.to(self.device)
        base_store = getattr(self, prepared.base_kind)
        base = base_store[prepared.base_indices[resident_indices]]
        solute = (
            self.entity[prepared.solute_indices[resident_indices]]
            if prepared.solute_indices is not None
            else None
        )
        def on_training_device(value: torch.Tensor) -> torch.Tensor:
            return value if value.device == device else value.to(device)

        return (
            on_training_device(base),
            on_training_device(prepared.conditions[resident_indices]),
            on_training_device(prepared.phase_ids[resident_indices]),
            on_training_device(prepared.targets[resident_indices]),
            on_training_device(solute) if solute is not None else None,
        )


class DomainTorchRNG:
    """Swap a domain-owned torch RNG stream into global PyTorch state."""

    def __init__(self, seed: int) -> None:
        external_cpu = torch.get_rng_state()
        external_cuda = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.cpu_state = torch.get_rng_state()
        self.cuda_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        torch.set_rng_state(external_cpu)
        if external_cuda is not None:
            torch.cuda.set_rng_state_all(external_cuda)

    @contextmanager
    def activate(self) -> Iterator[None]:
        external_cpu = torch.get_rng_state()
        external_cuda = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        torch.set_rng_state(self.cpu_state)
        if self.cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_state)
        try:
            yield
        finally:
            self.cpu_state = torch.get_rng_state()
            self.cuda_state = (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            )
            torch.set_rng_state(external_cpu)
            if external_cuda is not None:
                torch.cuda.set_rng_state_all(external_cuda)

    def state_dict(self) -> dict[str, Any]:
        return {
            "cpu": self.cpu_state.clone(),
            "cuda": (
                [value.clone() for value in self.cuda_state]
                if self.cuda_state is not None
                else None
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.cpu_state = state["cpu"].clone()
        self.cuda_state = (
            [value.clone() for value in state["cuda"]]
            if state.get("cuda") is not None
            else None
        )


@dataclass
class DomainRuntime:
    name: str
    tasks: tuple[str, ...]
    store: FrozenRepresentationStore
    metadata: dict[str, Any]
    train: dict[str, Stage3TaskDataset]
    valid: dict[str, Stage3TaskDataset]
    cursors: dict[str, SystemCursor]
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: torch.amp.GradScaler
    task_order_rng: random.Random
    torch_rng: DomainTorchRNG
    block: int = 0
    micro_step: int = 0
    best_metric: float = float("inf")
    validations_without_improvement: int = 0
    stopped: bool = False


def _config_hash(config: Stage3Config, fold: int) -> str:
    payload = _semantic_config(config)
    payload["fold"] = fold
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _checkpoint_config(config: Stage3Config) -> dict[str, Any]:
    payload = config.to_dict()
    payload["training"]["resume_from"] = None
    return payload


def _semantic_config(config: Stage3Config) -> dict[str, Any]:
    payload = _checkpoint_config(config)
    training = payload["training"]
    for name in ("cpu_threads", "cpu_interop_threads", "resident_data"):
        training.pop(name, None)
    for domain in ("il21", "aux6"):
        domain_config = training[domain]
        if domain_config.get("backward_mode") == "per_task":
            domain_config.pop("backward_mode")
    return payload


def _lr_lambda(block: int, total: int, warmup_fraction: float) -> float:
    warmup = max(1, round(total * warmup_fraction))
    if block < warmup:
        return (block + 1) / warmup
    progress = (block - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate_domain(
    module: torch.nn.Module,
    runtime: DomainRuntime,
    config: Stage3Config,
    device: torch.device,
) -> dict[str, float | int]:
    was_training = module.training
    module.eval()
    result: dict[str, float | int] = {}
    task_statistics: list[
        tuple[str, int, torch.Tensor]
    ] = []
    training = config.domain_training(runtime.name)
    for task, dataset in runtime.valid.items():
        if len(dataset) == 0:
            continue
        statistics = torch.zeros(3, dtype=torch.float32, device=device)
        scale = float(
            dataset.scalers[f"fold{dataset.fold}"][task]["target"][
                "scale"
            ]
        )
        for start in range(0, len(dataset), training.batch_size):
            indices = torch.arange(
                start,
                min(len(dataset), start + training.batch_size),
                device=runtime.store.device,
            )
            base, conditions, phase_ids, targets, solute = runtime.store.batch(
                task, dataset, indices, device
            )
            predictions = module(
                task,
                base,
                conditions,
                phase_ids,
                solute_cls=solute,
            ).predictions
            delta = predictions.float() - targets
            statistics += torch.stack(
                (
                    delta.abs().sum(),
                    delta.square().sum(),
                    (delta * scale).abs().sum(),
                )
            )
        task_statistics.append((task, len(dataset), statistics))
    synchronized = (
        torch.stack([row[2] for row in task_statistics]).cpu().tolist()
        if task_statistics
        else []
    )
    normalized_maes: list[float] = []
    for (task, count, _), (absolute, squared, raw_absolute) in zip(
        task_statistics, synchronized, strict=True
    ):
        key = task.replace("/", "_")
        result[f"valid_{key}_count"] = count
        result[f"valid_{key}_normalized_mae"] = absolute / count
        result[f"valid_{key}_normalized_rmse"] = math.sqrt(squared / count)
        result[f"valid_{key}_mae"] = raw_absolute / count
        normalized_maes.append(absolute / count)
    result[f"valid_{runtime.name}_macro_normalized_mae"] = (
        sum(normalized_maes) / len(normalized_maes)
    )
    if was_training:
        module.train()
    return result


def _atomic_save(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _domain_best_payload(
    *,
    domain: str,
    module: torch.nn.Module,
    runtime: DomainRuntime,
    fold: int,
    config: Stage3Config,
    config_hash: str,
) -> dict[str, Any]:
    return {
        "format_version": STAGE3_CHECKPOINT_VERSION,
        "kind": STAGE3_DOMAIN_MODEL_KIND,
        "domain": domain,
        "model": module.state_dict(),
        "block": runtime.block,
        "fold": fold,
        "best_metric": runtime.best_metric,
        "config": _checkpoint_config(config),
        "config_hash": config_hash,
        "data_metadata_hash": sha256_file(
            config.data.artifacts_dir / domain / "metadata.json"
        ),
        "stage2_checkpoint_hash": runtime.metadata["provenance"][
            "stage2_checkpoint_hash"
        ],
        "source_hashes": runtime.metadata["source_hashes"],
    }


def _training_checkpoint_payload(
    *,
    model: Stage3MultiDomainModel,
    runtimes: dict[str, DomainRuntime],
    cycle: int,
    fold: int,
    config: Stage3Config,
    config_hash: str,
) -> dict[str, Any]:
    return {
        "format_version": STAGE3_CHECKPOINT_VERSION,
        "kind": STAGE3_TRAINING_KIND,
        "model": model.state_dict(),
        "cycle": cycle,
        "fold": fold,
        "active_domains": list(config.active_domains),
        "config": _checkpoint_config(config),
        "config_hash": config_hash,
        "domains": {
            domain: {
                "optimizer": runtime.optimizer.state_dict(),
                "scheduler": runtime.scheduler.state_dict(),
                "scaler": runtime.scaler.state_dict(),
                "cursors": {
                    task: cursor.state_dict()
                    for task, cursor in runtime.cursors.items()
                },
                "task_order_rng": runtime.task_order_rng.getstate(),
                "torch_rng": runtime.torch_rng.state_dict(),
                "block": runtime.block,
                "micro_step": runtime.micro_step,
                "best_metric": runtime.best_metric,
                "validations_without_improvement": (
                    runtime.validations_without_improvement
                ),
                "stopped": runtime.stopped,
                "data_metadata_hash": sha256_file(
                    config.data.artifacts_dir / domain / "metadata.json"
                ),
                "stage2_checkpoint_hash": runtime.metadata["provenance"][
                    "stage2_checkpoint_hash"
                ],
                "source_hashes": runtime.metadata["source_hashes"],
            }
            for domain, runtime in runtimes.items()
        },
    }


def _load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("format_version") != STAGE3_CHECKPOINT_VERSION
        or payload.get("kind") != STAGE3_TRAINING_KIND
    ):
        raise ValueError("Unsupported Stage 3 v2 training checkpoint")
    return payload


def _reconcile_metrics_for_resume(
    path: Path, checkpoint: dict[str, Any]
) -> None:
    if not path.is_file():
        return
    domain_blocks = {
        domain: int(state["block"])
        for domain, state in checkpoint["domains"].items()
    }
    retained: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise ValueError(
                f"Malformed Stage 3 metrics before final line: {path}"
            )
        domain = row.get("domain")
        if domain not in domain_blocks:
            raise ValueError(f"Unexpected Stage 3 metrics domain: {domain}")
        if int(row["block"]) <= domain_blocks[domain]:
            retained.append(json.dumps(row, sort_keys=True, allow_nan=True))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(f"{line}\n" for line in retained), encoding="utf-8"
    )
    temporary.replace(path)


def _build_runtime(
    config: Stage3Config,
    domain: str,
    fold: int,
    model: Stage3MultiDomainModel,
    device: torch.device,
) -> DomainRuntime:
    frozen, metadata = load_frozen_embeddings(config, domain)
    tasks = config.tasks_for_domain(domain)
    train = {
        task: Stage3TaskDataset(
            config.data.artifacts_dir, domain, fold, task, "train"
        )
        for task in tasks
    }
    valid = {
        task: Stage3TaskDataset(
            config.data.artifacts_dir, domain, fold, task, "valid"
        )
        for task in tasks
    }
    domain_offset = 100000 if domain == "il21" else 200000
    cursors = {
        task: SystemCursor(
            dataset.system_offsets,
            dataset.system_rows,
            seed=config.data.seed + fold * 1000 + domain_offset + index,
        )
        for index, (task, dataset) in enumerate(train.items())
    }
    training = config.domain_training(domain)
    module = model.domain_module(domain)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda block: _lr_lambda(
            block, training.max_blocks, training.warmup_fraction
        ),
    )
    fp16 = training.amp_dtype == "fp16" and device.type == "cuda"
    resident = config.training.resident_data and device.type == "cuda"
    try:
        store = FrozenRepresentationStore(
            frozen,
            device=device,
            resident=resident,
        )
        for dataset in (*train.values(), *valid.values()):
            store.prepare_dataset(dataset)
    except torch.OutOfMemoryError as error:
        if not resident:
            raise
        raise RuntimeError(
            f"Stage 3 {domain} resident_data does not fit on {device}; "
            "no CPU fallback was applied"
        ) from error
    return DomainRuntime(
        name=domain,
        tasks=tasks,
        store=store,
        metadata=metadata,
        train=train,
        valid=valid,
        cursors=cursors,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=torch.amp.GradScaler("cuda", enabled=fp16),
        task_order_rng=random.Random(
            config.data.seed + fold * 100000 + domain_offset
        ),
        torch_rng=DomainTorchRNG(
            config.data.seed + fold * 1000000 + domain_offset
        ),
    )


def _restore_runtime(
    runtime: DomainRuntime,
    state: dict[str, Any],
    config: Stage3Config,
) -> None:
    domain = runtime.name
    expected = {
        "data_metadata_hash": sha256_file(
            config.data.artifacts_dir / domain / "metadata.json"
        ),
        "stage2_checkpoint_hash": runtime.metadata["provenance"][
            "stage2_checkpoint_hash"
        ],
        "source_hashes": runtime.metadata["source_hashes"],
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(
                f"Stage 3 checkpoint {domain}.{key} does not match"
            )
    runtime.optimizer.load_state_dict(state["optimizer"])
    runtime.scheduler.load_state_dict(state["scheduler"])
    runtime.scaler.load_state_dict(state["scaler"])
    for task, cursor in runtime.cursors.items():
        cursor.load_state_dict(state["cursors"][task])
    runtime.task_order_rng.setstate(state["task_order_rng"])
    runtime.torch_rng.load_state_dict(state["torch_rng"])
    runtime.block = int(state["block"])
    runtime.micro_step = int(state["micro_step"])
    if runtime.micro_step != runtime.block * len(runtime.tasks):
        raise ValueError(
            f"Stage 3 checkpoint {domain} micro-step is not a full block"
        )
    runtime.best_metric = float(state["best_metric"])
    runtime.validations_without_improvement = int(
        state["validations_without_improvement"]
    )
    runtime.stopped = bool(state["stopped"])
    if runtime.block >= config.domain_training(domain).max_blocks:
        runtime.stopped = True


def _assemble_best(
    *,
    config: Stage3Config,
    model: Stage3MultiDomainModel,
    runtimes: dict[str, DomainRuntime],
    fold: int,
    config_hash: str,
    output_dir: Path,
) -> None:
    if len(config.active_domains) == 1:
        return
    domain_metrics: dict[str, float] = {}
    for domain in config.active_domains:
        path = output_dir / f"best_{domain}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("format_version") != STAGE3_CHECKPOINT_VERSION
            or payload.get("kind") != STAGE3_DOMAIN_MODEL_KIND
            or payload.get("domain") != domain
        ):
            raise ValueError(f"Invalid Stage 3 domain best checkpoint: {path}")
        model.domain_module(domain).load_state_dict(
            payload["model"], strict=True
        )
        domain_metrics[domain] = float(payload["best_metric"])
    _atomic_save(
        output_dir / "best.pt",
        {
            "format_version": STAGE3_CHECKPOINT_VERSION,
            "kind": STAGE3_MODEL_KIND,
            "model": model.state_dict(),
            "fold": fold,
            "active_domains": list(config.active_domains),
            "domain_best_metrics": domain_metrics,
            "config": _checkpoint_config(config),
            "config_hash": config_hash,
            "domains": {
                domain: {
                    "data_metadata_hash": sha256_file(
                        config.data.artifacts_dir / domain / "metadata.json"
                    ),
                    "stage2_checkpoint_hash": runtime.metadata[
                        "provenance"
                    ]["stage2_checkpoint_hash"],
                    "source_hashes": runtime.metadata["source_hashes"],
                }
                for domain, runtime in runtimes.items()
            },
        },
    )


def _train_domain_block(
    *,
    config: Stage3Config,
    model: Stage3MultiDomainModel,
    runtime: DomainRuntime,
    device: torch.device,
) -> dict[str, Any]:
    domain = runtime.name
    training = config.domain_training(domain)
    module = model.domain_module(domain)
    module.train()
    runtime.optimizer.zero_grad(set_to_none=True)
    order = list(runtime.tasks)
    runtime.task_order_rng.shuffle(order)
    sampled = [
        runtime.cursors[task].next_indices(training.batch_size)
        for task in order
    ]
    resident_indices = torch.cat(sampled).to(runtime.store.device)
    sampled_by_task = {
        task: resident_indices[
            index * training.batch_size : (index + 1) * training.batch_size
        ]
        for index, task in enumerate(order)
    }
    task_losses: list[tuple[str, torch.Tensor]] = []
    domain_loss: torch.Tensor | None = None
    amp_enabled = training.amp_dtype != "none" and device.type == "cuda"
    amp_dtype = (
        torch.float16 if training.amp_dtype == "fp16" else torch.bfloat16
    )
    with runtime.torch_rng.activate():
        for task in order:
            indices = sampled_by_task[task]
            base, conditions, phase_ids, targets, solute = runtime.store.batch(
                task, runtime.train[task], indices, device
            )
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                predictions = module(
                    task,
                    base,
                    conditions,
                    phase_ids,
                    solute_cls=solute,
                ).predictions
                task_loss = F.smooth_l1_loss(predictions, targets)
                loss = task_loss / len(runtime.tasks)
            if training.backward_mode == "per_task":
                runtime.scaler.scale(loss).backward()
            else:
                domain_loss = (
                    loss if domain_loss is None else domain_loss + loss
                )
            task_losses.append((task, task_loss.detach()))
            runtime.micro_step += 1
        if training.backward_mode == "domain":
            if domain_loss is None:
                raise RuntimeError(f"Stage 3 {domain} block has no task loss")
            runtime.scaler.scale(domain_loss).backward()
    runtime.scaler.unscale_(runtime.optimizer)
    if training.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            module.parameters(), training.max_grad_norm
        )
    scale_before = runtime.scaler.get_scale()
    runtime.scaler.step(runtime.optimizer)
    runtime.scaler.update()
    optimizer_stepped = (
        not runtime.scaler.is_enabled()
        or runtime.scaler.get_scale() >= scale_before
    )
    if optimizer_stepped:
        runtime.scheduler.step()
    runtime.block += 1
    synchronized_losses = torch.stack(
        [value for _, value in task_losses]
    ).float().cpu().tolist()
    losses = {
        task: float(value)
        for (task, _), value in zip(
            task_losses, synchronized_losses, strict=True
        )
    }
    return {
        "domain": domain,
        "block": runtime.block,
        "micro_step": runtime.micro_step,
        "loss": sum(losses.values()) / len(losses),
        "learning_rate": runtime.optimizer.param_groups[0]["lr"],
        "optimizer_step": int(optimizer_stepped),
        **{
            f"loss_{task.replace('/', '_')}": value
            for task, value in losses.items()
        },
    }


def run_stage3_training(
    config: Stage3Config,
    fold: int,
    *,
    reporter: ProgressReporter | None = None,
) -> list[dict[str, Any]]:
    config.validate()
    if fold not in range(1, 6):
        raise ValueError("Stage 3 fold must be in 1..5")
    reporter = reporter or ProgressReporter()
    device = resolve_device(config.training.device)
    metadata_by_domain = {
        domain: load_frozen_embeddings(config, domain)[1]
        for domain in config.active_domains
    }
    dimensions = {
        int(metadata["embedding_dim"])
        for metadata in metadata_by_domain.values()
    }
    if len(dimensions) != 1:
        raise ValueError("Stage 3 domain embedding dimensions do not match")
    model = Stage3MultiDomainModel(
        config,
        dimensions.pop(),
        seed=config.data.seed + fold,
    ).to(device)
    runtimes = {
        domain: _build_runtime(config, domain, fold, model, device)
        for domain in config.active_domains
    }
    output_dir = config.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    config_hash = _config_hash(config, fold)
    cycle = 0
    if config.training.resume_from is not None:
        checkpoint = _load_training_checkpoint(config.training.resume_from)
        if checkpoint.get("fold") != fold:
            raise ValueError("Stage 3 checkpoint fold does not match")
        if checkpoint.get("config_hash") != config_hash:
            raise ValueError("Stage 3 checkpoint config_hash does not match")
        if tuple(checkpoint.get("active_domains", ())) != config.active_domains:
            raise ValueError("Stage 3 checkpoint active domains do not match")
        model.load_state_dict(checkpoint["model"], strict=True)
        for domain, runtime in runtimes.items():
            _restore_runtime(runtime, checkpoint["domains"][domain], config)
        cycle = int(checkpoint["cycle"])
        _reconcile_metrics_for_resume(metrics_path, checkpoint)
    elif metrics_path.is_file() and metrics_path.stat().st_size:
        raise FileExistsError(
            f"Stage 3 output already contains metrics: {metrics_path}"
        )
    results: list[dict[str, Any]] = []
    total_remaining = sum(
        max(
            0,
            config.domain_training(domain).max_blocks - runtime.block,
        )
        for domain, runtime in runtimes.items()
        if not runtime.stopped
    )
    with reporter.bar(
        total=total_remaining,
        desc=f"Stage 3 v2 fold {fold}",
        unit="domain-block",
    ) as progress:
        while any(not runtime.stopped for runtime in runtimes.values()):
            cycle += 1
            should_checkpoint = False
            for domain in config.active_domains:
                runtime = runtimes[domain]
                if runtime.stopped:
                    continue
                training = config.domain_training(domain)
                row = _train_domain_block(
                    config=config,
                    model=model,
                    runtime=runtime,
                    device=device,
                )
                should_validate = (
                    runtime.block % training.validation_interval_blocks == 0
                    or runtime.block == training.max_blocks
                )
                if should_validate:
                    row.update(
                        evaluate_domain(
                            model.domain_module(domain),
                            runtime,
                            config,
                            device,
                        )
                    )
                    metric = float(
                        row[f"valid_{domain}_macro_normalized_mae"]
                    )
                    if not math.isfinite(metric):
                        raise RuntimeError(
                            f"Non-finite Stage 3 {domain} validation metric"
                        )
                    improved = (
                        not math.isfinite(runtime.best_metric)
                        or metric
                        < runtime.best_metric
                        - training.early_stopping_min_delta
                    )
                    if improved:
                        runtime.best_metric = metric
                        runtime.validations_without_improvement = 0
                        _atomic_save(
                            output_dir / f"best_{domain}.pt",
                            _domain_best_payload(
                                domain=domain,
                                module=model.domain_module(domain),
                                runtime=runtime,
                                fold=fold,
                                config=config,
                                config_hash=config_hash,
                            ),
                        )
                    else:
                        runtime.validations_without_improvement += 1
                    row[f"best_{domain}_macro_normalized_mae"] = (
                        runtime.best_metric
                    )
                    if (
                        runtime.validations_without_improvement
                        >= training.early_stopping_patience
                    ):
                        runtime.stopped = True
                    should_checkpoint = True
                if runtime.block >= training.max_blocks:
                    runtime.stopped = True
                row["stopped"] = int(runtime.stopped)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(row, sort_keys=True, allow_nan=True) + "\n"
                    )
                results.append(row)
                progress.set_postfix(
                    {
                        "domain": domain,
                        "loss": f"{row['loss']:.4f}",
                    },
                    refresh=False,
                )
                progress.update(1)
            if should_checkpoint or all(
                runtime.stopped for runtime in runtimes.values()
            ):
                checkpoint_path = (
                    output_dir / f"checkpoint_cycle_{cycle:08d}.pt"
                )
                _atomic_save(
                    checkpoint_path,
                    _training_checkpoint_payload(
                        model=model,
                        runtimes=runtimes,
                        cycle=cycle,
                        fold=fold,
                        config=config,
                        config_hash=config_hash,
                    ),
                )
                checkpoints = sorted(
                    output_dir.glob("checkpoint_cycle_*.pt")
                )
                for stale in checkpoints[
                    : -config.training.keep_last_checkpoints
                ]:
                    stale.unlink()
    _assemble_best(
        config=config,
        model=model,
        runtimes=runtimes,
        fold=fold,
        config_hash=config_hash,
        output_dir=output_dir,
    )
    return results


def with_stage3_overrides(
    config: Stage3Config,
    *,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> Stage3Config:
    training = config.training
    if output_dir is not None:
        training = replace(training, output_dir=Path(output_dir))
    if resume_from is not None:
        training = replace(training, resume_from=Path(resume_from))
    result = replace(config, training=training)
    result.validate()
    return result
