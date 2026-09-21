from __future__ import annotations

import math
from collections import Counter
from typing import Any

import torch

from common.identity import semantic_identity, tensor_state_hash
from stage2.data import Stage2TaskDataset
from stage3.data import stable_seed

from .config import TransferExperimentConfig


def experiment_contract(experiment: TransferExperimentConfig) -> dict[str, Any]:
    """Omit the default to preserve existing full-data artifact identities."""
    if experiment.stage2.sampling_mode == "full":
        return {}
    return {"sampling_mode": "balanced_rows", "balanced_experiment_identity": semantic_identity(
        "stage2-stage3.transfer-balanced.v1", experiment.to_dict()
    )["hash"]}


def require_experiment_contract(experiment: TransferExperimentConfig, payload: dict[str, Any]) -> None:
    expected = experiment_contract(experiment).get("balanced_experiment_identity")
    if payload.get("balanced_experiment_identity") != expected:
        raise ValueError("Transfer full/balanced experiment identity mismatch")


def resolve_balanced_rows(
    experiment: TransferExperimentConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Resolve one fixed, no-replacement subset per source before any training."""
    counts = {}
    for task in sorted(experiment.stage2.sources):
        counts[task] = len(Stage2TaskDataset(experiment.stage2.prepared_artifacts, task, "train"))
    size = min(counts.values())
    if size <= 0:
        raise ValueError("Balanced transfer requires nonempty training sources")
    selections, audits = {}, {}
    for task in sorted(counts):
        dataset = Stage2TaskDataset(experiment.stage2.prepared_artifacts, task, "train")
        seed = stable_seed(experiment.seed, "transfer-balanced-rows-v1", task)
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:size].sort().values
        dataset.select_train_rows(indices)
        systems = Counter(tuple(row) for row in dataset.entity_indices.tolist())
        entities = dataset.entity_indices[dataset.entity_indices >= 0].unique()
        selections[task] = indices
        audits[task] = {
            "available_train_rows": counts[task], "selected_train_rows": size,
            "selection_seed": seed, "selected_indices": indices.tolist(),
            "selected_source_rows": dataset.source_rows.tolist(),
            "selection_hash": tensor_state_hash("stage2.transfer-selected-rows.v1", {
                "indices": indices, "source_rows": dataset.source_rows,
            }),
            "unique_ordered_entity_systems": len(systems), "unique_entities": len(entities),
            "rows_per_system_histogram": {
                str(rows): count for rows, count in sorted(Counter(systems.values()).items())
            },
        }
    steps = math.ceil(size / experiment.stage2.batch_size)
    plan = {
        "sampling_mode": "balanced_rows", "algorithm": "fixed-randperm-subset-v1",
        "normalization": "unchanged-full-prepared-train-scalers",
        "system_definition": "ordered prepared entity-index tuple; conditions excluded",
        "rows_per_source": size, "batch_size": experiment.stage2.batch_size,
        "epochs": experiment.stage2.epochs, "updates_per_epoch": steps,
        "optimizer_updates": steps * experiment.stage2.epochs,
        "backbone_frozen_updates": steps * experiment.stage2.backbone_frozen_epochs,
        "sources": audits, **experiment_contract(experiment),
    }
    return selections, plan
