from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from rdkit import rdBase

from common.io import sha256_file
from common.progress import ProgressReporter
from .config import Stage3Config
from .data import STAGE3_ARTIFACT_VERSION, source_hashes


STAGE2_OBJECT_CHECKPOINT_VERSION = 1
STAGE2_OBJECT_CHECKPOINT_KIND = "ilume_stage2_object"
STAGE3_MIGRATION_MESSAGE = "Stage 3 object contract migration pending"


def load_frozen_stage2(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    del device
    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False
    )
    if (
        checkpoint.get("format_version") == STAGE2_OBJECT_CHECKPOINT_VERSION
        and checkpoint.get("kind") == STAGE2_OBJECT_CHECKPOINT_KIND
    ):
        raise ValueError(STAGE3_MIGRATION_MESSAGE)
    raise ValueError("Stage 3 rejects legacy Stage 2 checkpoints")


def prepare_stage3(
    config: Stage3Config,
    *,
    reporter: ProgressReporter | None = None,
) -> dict[str, dict[str, Any]]:
    del reporter
    config.validate()
    load_frozen_stage2(config.initialization.stage2_checkpoint)
    raise AssertionError("unreachable")


def load_frozen_embeddings(
    config: Stage3Config,
    domain: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tasks = config.tasks_for_domain(domain)
    root = config.data.artifacts_dir / domain
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("format_version") != STAGE3_ARTIFACT_VERSION
        or metadata.get("kind") != "ilume_stage3_data"
    ):
        raise ValueError("Unsupported Stage 3 artifact format")
    if metadata.get("domain") != domain or tuple(metadata.get("tasks", ())) != tasks:
        raise ValueError("Stage 3 artifact domain/task registry mismatch")
    if metadata.get("rdkit_version") != rdBase.rdkitVersion:
        raise ValueError("Stage 3 artifact RDKit version mismatch")
    if metadata.get("source_hashes") != source_hashes(config, tasks):
        raise ValueError("Stage 3 source data hash mismatch")
    if metadata.get("provenance", {}).get("stage2_checkpoint") != str(
        config.initialization.stage2_checkpoint
    ):
        raise ValueError("Stage 3 frozen cache Stage 2 checkpoint path mismatch")
    path = root / "frozen_embeddings.pt"
    if metadata["artifact_hashes"].get("frozen_embeddings.pt") != sha256_file(path):
        raise ValueError("Stage 3 frozen embedding hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload, metadata
