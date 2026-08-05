from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .masking import MultimodalPacker
from .progress import ProgressReporter
from .stage2_config import Stage2Config
from .stage2_data import Stage2EntityDataset, prepare_stage2_data
from .stage2_model import load_stage1_model, sha256_file


TEACHER_CACHE_VERSION = 1


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable")
    return device


def teacher_cache_dir(config: Stage2Config, checkpoint_hash: str) -> Path:
    return config.data.artifacts_dir / "teachers" / checkpoint_hash


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def prepare_teacher_cache(
    config: Stage2Config,
    *,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    config.validate()
    reporter = reporter or ProgressReporter()
    data_metadata = prepare_stage2_data(config, reporter=reporter)
    device = resolve_device(config.training.device)
    loaded = load_stage1_model(
        config.initialization.checkpoint,
        config.data.pretrain_artifacts_dir,
        device=device,
    )
    if loaded.artifact_hash != data_metadata["pretrain_artifact_hash"]:
        raise ValueError("Teacher checkpoint does not match Stage 2 entity features")
    output_dir = teacher_cache_dir(config, loaded.checkpoint_hash)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    embeddings_path = output_dir / "embeddings.pt"
    data_metadata_hash = sha256_file(config.data.artifacts_dir / "metadata.json")
    if metadata_path.is_file() and embeddings_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("format_version") == TEACHER_CACHE_VERSION
            and metadata.get("checkpoint_hash") == loaded.checkpoint_hash
            and metadata.get("data_metadata_hash") == data_metadata_hash
            and metadata.get("embeddings_hash") == sha256_file(embeddings_path)
        ):
            embeddings = torch.load(
                embeddings_path,
                map_location="cpu",
                weights_only=True,
            )
            if tuple(embeddings.shape) == (
                int(metadata["entity_count"]),
                int(metadata["embedding_dim"]),
            ):
                return metadata

    entity_dataset = Stage2EntityDataset(
        config.data.artifacts_dir,
        config.data.shard_cache_size,
    )
    packer = MultimodalPacker(loaded.vocabulary)
    embeddings = torch.empty(
        (len(entity_dataset), loaded.config.model.d_model),
        dtype=torch.float32,
    )
    loaded.model.eval()
    with torch.no_grad(), reporter.bar(
        total=len(entity_dataset),
        desc="Stage 2 teacher embeddings",
        unit="entity",
    ) as progress:
        for start in range(0, len(entity_dataset), config.data.teacher_batch_size):
            end = min(len(entity_dataset), start + config.data.teacher_batch_size)
            batch = packer(
                [entity_dataset[index] for index in range(start, end)]
            ).to(device)
            encoded = loaded.model.encode(batch).float().cpu()
            if not torch.isfinite(encoded).all():
                raise RuntimeError(
                    f"Non-finite teacher embedding in entity rows {start}:{end}"
                )
            embeddings[start:end] = encoded
            progress.update(end - start)
    _atomic_torch_save(embeddings_path, embeddings)
    metadata = {
        "format_version": TEACHER_CACHE_VERSION,
        "checkpoint": str(config.initialization.checkpoint),
        "checkpoint_hash": loaded.checkpoint_hash,
        "pretrain_artifact_hash": loaded.artifact_hash,
        "data_metadata_hash": data_metadata_hash,
        "entity_count": len(entity_dataset),
        "embedding_dim": loaded.config.model.d_model,
        "dtype": "float32",
        "embeddings_hash": sha256_file(embeddings_path),
    }
    _atomic_json(metadata_path, metadata)
    reporter.emit_json(
        {
            "event": "stage2_teacher_cache_complete",
            "entity_count": len(entity_dataset),
            "embedding_dim": loaded.config.model.d_model,
            "checkpoint_hash": loaded.checkpoint_hash,
        }
    )
    return metadata


def load_teacher_embeddings(
    config: Stage2Config,
    *,
    checkpoint_hash: str,
    expected_count: int,
    expected_dim: int,
) -> torch.Tensor:
    output_dir = teacher_cache_dir(config, checkpoint_hash)
    metadata_path = output_dir / "metadata.json"
    embeddings_path = output_dir / "embeddings.pt"
    if not metadata_path.is_file() or not embeddings_path.is_file():
        raise FileNotFoundError(
            "Missing Stage 2 teacher cache; run ilume-stage2-prepare first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != TEACHER_CACHE_VERSION:
        raise ValueError("Unsupported Stage 2 teacher cache format")
    if metadata.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError("Stage 2 teacher checkpoint hash mismatch")
    if metadata.get("data_metadata_hash") != sha256_file(
        config.data.artifacts_dir / "metadata.json"
    ):
        raise ValueError("Stage 2 teacher cache does not match the data artifact")
    if metadata.get("embeddings_hash") != sha256_file(embeddings_path):
        raise ValueError("Stage 2 teacher embedding hash mismatch")
    embeddings = torch.load(
        embeddings_path,
        map_location="cpu",
        weights_only=True,
    )
    if tuple(embeddings.shape) != (expected_count, expected_dim):
        raise ValueError("Stage 2 teacher embedding shape mismatch")
    if embeddings.dtype != torch.float32 or not torch.isfinite(embeddings).all():
        raise ValueError("Stage 2 teacher embeddings must be finite FP32")
    return embeddings
