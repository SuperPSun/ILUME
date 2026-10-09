from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from rdkit import Chem

from common.identity import semantic_identity
from common.io import sha256_file
from common.entity_roles import molecular_role
from stage1.descriptors import (
    DescriptorSchema,
    DescriptorStandardizer,
    calculate_descriptors,
    rdkit_descriptor_names,
)
from stage1.features import ROLE_TO_ID, build_entity_sample, inspect_entity_qc
from stage1.masking import MultimodalPacker
from .model import encode_object_entities
from stage1.model import load_stage1_model
from stage1.identity import metadata_identity
import json

STAGE2_ENCODER_VERSION = 1
STAGE2_ENCODER_KIND = "ilume_stage2_encoder"
STAGE2_HOME_ENCODER_KIND = "ilume_stage2_home_encoder_v1"
STAGE2_ZERO_UPDATE_HOME_ENCODER_KIND = "ilume_stage2_home_zero_update_encoder_v1"

@dataclass(frozen=True)
class FrozenObjectSpec:
    topology: str
    slots: tuple[tuple[str, str], ...]

@dataclass
class FrozenStage1Entities:
    backbone: Any
    packer: Any
    pretrain_config: Any
    descriptor_schema: Any
    descriptor_standardizer: Any
    encoder_identity: dict
    artifact_hash: str
    device: torch.device
    role_policy: str = "formal_charge_v1"
    entity_input_dim: int = 1241
    embedding_dim: int = 1024

    def _sample(self, role: str, canonical_smiles: str) -> dict[str, Any]:
        if role not in ROLE_TO_ID:
            raise ValueError(f"Unsupported frozen Stage 2 role: {role}")
        if self.role_policy == "formal_charge_v1" and molecular_role(canonical_smiles) != role:
            raise ValueError("Frozen Stage2 entity role/formal-charge mismatch")
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
        if self.packer.vocabulary.token_count(canonical_smiles) > (
            self.pretrain_config.data.max_smiles_tokens
        ):
            qc.reasons.append("smiles_overlength")
        if qc.reasons:
            raise ValueError(
                "Stage 3 object is incompatible with Stage 2 features: "
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

    def input_sample(self, role: str, canonical_smiles: str) -> dict[str, Any]:
        """Build the audited Stage 1 input for one Stage 3 entity slot."""
        return self._sample(role, canonical_smiles)

    @torch.inference_mode()
    def _encode_slots_device(
        self, objects: Sequence[FrozenObjectSpec]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not objects:
            return (
                torch.empty((0, 0, self.entity_input_dim), dtype=torch.float32),
                torch.empty((0, 0), dtype=torch.long),
            )
        slot_counts = {len(item.slots) for item in objects}
        if len(slot_counts) != 1:
            raise ValueError("Frozen Stage 2 object batch must share topology")
        slot_count = slot_counts.pop()
        expected_topology = "molecule" if slot_count == 1 else "il"
        if any(item.topology != expected_topology for item in objects):
            raise ValueError("Frozen Stage 2 object topology/slot mismatch")
        packed = self.packer(
            [
                self._sample(role, smiles)
                for item in objects
                for role, smiles in item.slots
            ]
        ).to(self.device)
        entity_cls = (
            encode_object_entities(self.backbone, packed).entity_embedding
            if self.pretrain_config.is_global_rdkit or self.pretrain_config.is_dual_view
            else self.backbone.encode(packed)
        ).reshape(
            len(objects), slot_count, self.entity_input_dim
        )
        roles = torch.tensor(
            [ROLE_TO_ID[role] for item in objects for role, _ in item.slots],
            dtype=torch.long,
            device=self.device,
        ).reshape(len(objects), slot_count)
        if not torch.isfinite(entity_cls).all():
            raise RuntimeError("Frozen Stage 2 produced non-finite entity slots")
        return entity_cls, roles

    @torch.inference_mode()
    def encode_slots(
        self, objects: Sequence[FrozenObjectSpec]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slots, roles = self._encode_slots_device(objects)
        return slots.float().cpu(), roles.cpu()


def load_frozen_stage1_entities(checkpoint, artifacts_dir, *, device="cpu"):
    from stage1.identity import encoding_state_hash
    root = Path(artifacts_dir)
    loaded = load_stage1_model(checkpoint, root)
    if not loaded.config.is_dual_view or loaded.model.entity_dim != 1024:
        raise ValueError("Entity HoME requires the frozen v4 Stage1 encoder")
    metadata = json.loads((root / "metadata.json").read_text())
    features = {name: sha256_file(root / name) for name in (
        "tokenizer.json", "descriptor_schema.json", "descriptor_scaler.json")}
    identity = semantic_identity("stage3.frozen-stage1-entities.v4", {
        "checkpoint_sha256": sha256_file(checkpoint), "stage1_state_hash": encoding_state_hash(loaded.model),
        "feature_identity": metadata_identity(metadata, "feature", context="Stage1 features")["hash"],
        "feature_files": features, "slot_contract": "ordered_entity_slots_v1", "role_policy": "formal_charge_v1"})
    target = torch.device(device)
    loaded.model.requires_grad_(False).to(target).eval()
    schema = DescriptorSchema.from_payload(json.loads((root / "descriptor_schema.json").read_text()), expected_raw_names=rdkit_descriptor_names())
    scaler = DescriptorStandardizer.from_payload(json.loads((root / "descriptor_scaler.json").read_text()))
    return FrozenStage1Entities(loaded.model, MultimodalPacker(loaded.vocabulary), loaded.config,
        schema, scaler, identity, loaded.artifact_hash, target)

def load_frozen_object_encoder(*args, **kwargs):
    raise ValueError("ObjectEncoder loading retired; use the historical Git revision")

def load_stage2_encoder_identity(*args, **kwargs):
    raise ValueError("ObjectEncoder loading retired; use the historical Git revision")
