from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from rdkit import Chem

from common.io import sha256_file
from common.training import canonical_json_sha256
from stage1.descriptors import calculate_descriptors, rdkit_descriptor_names
from stage1.features import (
    ROLE_TO_ID,
    build_entity_sample,
    inspect_entity_qc,
    load_stage1_feature_inputs,
)
from stage1.masking import MultimodalPacker
from stage1.model import load_stage1_model
from .config import (
    STAGE2_CHECKPOINT_KIND,
    STAGE2_CHECKPOINT_VERSION,
    stage2_config_from_checkpoint_dict,
)
from .model import Stage2ObjectModel
from .registry import Stage2Registry


@dataclass(frozen=True)
class FrozenObjectSpec:
    topology: str
    slots: tuple[tuple[str, str], ...]


@dataclass
class FrozenStage2ObjectEncoder:
    model: Stage2ObjectModel
    packer: MultimodalPacker
    pretrain_config: Any
    descriptor_schema: Any
    descriptor_standardizer: Any
    checkpoint_hash: str
    checkpoint_kind: str
    checkpoint_version: int
    device: torch.device

    @property
    def embedding_dim(self) -> int:
        return int(self.model.model_contract["d_model"])

    def _sample(self, role: str, canonical_smiles: str) -> dict[str, Any]:
        if role not in ROLE_TO_ID:
            raise ValueError(f"Unsupported frozen Stage 2 role: {role}")
        record = {
            "sample_id": f"stage3:{role}:{canonical_smiles}",
            "role": role,
            "role_id": ROLE_TO_ID[role],
            "canonical_smiles": canonical_smiles,
            "sources": ("stage3",),
            "split": "stage3",
            "is_augmented": False,
            "seed_smiles": (),
        }
        qc = inspect_entity_qc(record)
        token_count = self.packer.vocabulary.token_count(canonical_smiles)
        if token_count > self.pretrain_config.data.max_smiles_tokens:
            qc.reasons.append("smiles_overlength")
        if qc.reasons:
            raise ValueError(
                f"Stage 3 object is incompatible with Stage 2 features: "
                f"{role}/{canonical_smiles}: {','.join(qc.reasons)}"
            )
        molecule = Chem.MolFromSmiles(canonical_smiles)
        if molecule is None:
            raise ValueError(f"Invalid canonical Stage 3 SMILES: {canonical_smiles}")
        raw = calculate_descriptors(molecule, rdkit_descriptor_names())
        return build_entity_sample(
            record,
            np.asarray(raw),
            self.descriptor_schema,
            self.descriptor_standardizer,
            self.packer.vocabulary,
            self.pretrain_config,
        )

    @torch.inference_mode()
    def encode(self, objects: Sequence[FrozenObjectSpec]) -> torch.Tensor:
        if not objects:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)
        slot_counts = {len(item.slots) for item in objects}
        if len(slot_counts) != 1:
            raise ValueError("Frozen Stage 2 object batch must share topology")
        slot_count = slot_counts.pop()
        expected_topology = "molecule" if slot_count == 1 else "il"
        if any(item.topology != expected_topology for item in objects):
            raise ValueError("Frozen Stage 2 object topology/slot mismatch")
        samples = [
            self._sample(role, smiles)
            for item in objects
            for role, smiles in item.slots
        ]
        packed = self.packer(samples).to(self.device)
        entity_cls = self.model.encode_entities(packed).reshape(
            len(objects), slot_count, self.embedding_dim
        )
        roles = torch.tensor(
            [ROLE_TO_ID[role] for item in objects for role, _ in item.slots],
            dtype=torch.long,
            device=self.device,
        ).reshape(len(objects), slot_count)
        values = self.model.encode_object(entity_cls, roles).float().cpu()
        if not torch.isfinite(values).all():
            raise RuntimeError("Frozen Stage 2 produced non-finite object embeddings")
        return values


def load_frozen_object_encoder(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> FrozenStage2ObjectEncoder:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("format_version") != STAGE2_CHECKPOINT_VERSION
        or checkpoint.get("kind") != STAGE2_CHECKPOINT_KIND
    ):
        raise ValueError("Stage 3 requires a Stage 2 Object v3 checkpoint")
    config = stage2_config_from_checkpoint_dict(checkpoint["config"])
    if checkpoint.get("config_hash") != canonical_json_sha256(
        config.experiment_dict()
    ):
        raise ValueError("Frozen Stage 2 checkpoint config hash mismatch")
    registry = Stage2Registry.from_snapshot(
        checkpoint["registry"],
        registry_hash=checkpoint["registry_hash"],
        catalog_sha256=checkpoint["catalog_sha256"],
    )
    config.validate_registry(registry)
    target_device = torch.device(device)
    loaded = load_stage1_model(
        config.initialization.checkpoint,
        config.data.pretrain_artifacts_dir,
        device=target_device,
        backbone_dropout=0.0,
    )
    model = Stage2ObjectModel(
        loaded.model,
        registry,
        object_layers=config.model.object_layers,
        object_ffn_dim=config.model.object_ffn_dim,
        dropout=config.model.dropout,
    )
    if checkpoint.get("model_contract") != model.model_contract:
        raise ValueError("Frozen Stage 2 checkpoint model contract mismatch")
    data_metadata = config.data.artifacts_dir / "metadata.json"
    if (
        not data_metadata.is_file()
        or checkpoint.get("data_metadata_hash") != sha256_file(data_metadata)
    ):
        raise ValueError("Frozen Stage 2 checkpoint data artifact mismatch")
    if checkpoint.get("normalized_task_weights") != config.normalized_task_weights(registry):
        raise ValueError("Frozen Stage 2 checkpoint normalized task weights mismatch")
    completed_epoch = checkpoint.get("completed_epoch")
    if not isinstance(completed_epoch, int) or not 1 <= completed_epoch <= config.training.epochs:
        raise ValueError("Frozen Stage 2 checkpoint epoch is invalid")
    for required in ("optimizer", "scheduler", "rng", "optimizer_implementation", "math_contract"):
        if required not in checkpoint:
            raise ValueError(f"Frozen Stage 2 checkpoint lacks {required}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(target_device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    pretrain_config, vocabulary, schema, standardizer, artifact_hash = (
        load_stage1_feature_inputs(
            config.initialization.checkpoint,
            config.data.pretrain_artifacts_dir,
        )
    )
    if artifact_hash != loaded.artifact_hash:
        raise ValueError("Frozen Stage 2 feature artifact identity mismatch")
    del checkpoint
    return FrozenStage2ObjectEncoder(
        model=model,
        packer=MultimodalPacker(vocabulary),
        pretrain_config=pretrain_config,
        descriptor_schema=schema,
        descriptor_standardizer=standardizer,
        checkpoint_hash=sha256_file(checkpoint_path),
        checkpoint_kind=STAGE2_CHECKPOINT_KIND,
        checkpoint_version=STAGE2_CHECKPOINT_VERSION,
        device=target_device,
    )
