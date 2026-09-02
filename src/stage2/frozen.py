from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from rdkit import Chem

from common.identity import (
    IDENTITY_CONTRACT_VERSION,
    require_compatible_identity,
    tensor_state_hash,
)
from common.io import sha256_file
from common.descriptor_preprocessing import FeaturePreprocessor
from stage1.config import config_from_dict
from stage1.descriptors import (
    DescriptorSchema,
    DescriptorStandardizer,
    calculate_descriptors,
    rdkit_descriptor_names,
)
from rdkit import rdBase
from stage1.features import ROLE_TO_ID, build_entity_sample, inspect_entity_qc
from stage1.identity import validate_feature_generation_runtime
from stage1.masking import MultimodalPacker
from stage1.model import MultimodalPretrainModel
from stage1.tokenizer import SmilesTokenizer
from .identity import build_stage2_encoder_identity
from .model import ObjectEncoder, RDKitDescriptorBackbone, RECONSTRUCTION_MODULES
from .rdkit_train import (
    STAGE2_RDKIT_ENCODER_KIND,
    load_rdkit_stage2_encoder_artifact,
)


STAGE2_ENCODER_VERSION = 1
STAGE2_ENCODER_KIND = "ilume_stage2_encoder"


@dataclass(frozen=True)
class FrozenObjectSpec:
    topology: str
    slots: tuple[tuple[str, str], ...]


@dataclass
class FrozenStage2ObjectEncoder:
    backbone: MultimodalPretrainModel
    object_encoder: ObjectEncoder
    packer: MultimodalPacker
    pretrain_config: Any
    descriptor_schema: DescriptorSchema
    descriptor_standardizer: DescriptorStandardizer
    encoder_identity: dict[str, Any]
    artifact_hash: str
    device: torch.device

    @property
    def embedding_dim(self) -> int:
        return int(self.pretrain_config.model.d_model)

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
        packed = self.packer(
            [
                self._sample(role, smiles)
                for item in objects
                for role, smiles in item.slots
            ]
        ).to(self.device)
        entity_cls = self.backbone.encode(packed).reshape(
            len(objects), slot_count, self.embedding_dim
        )
        roles = torch.tensor(
            [ROLE_TO_ID[role] for item in objects for role, _ in item.slots],
            dtype=torch.long,
            device=self.device,
        ).reshape(len(objects), slot_count)
        values = self.object_encoder(entity_cls, roles).float().cpu()
        if not torch.isfinite(values).all():
            raise RuntimeError("Frozen Stage 2 produced non-finite object embeddings")
        return values


@dataclass
class FrozenRDKitStage2ObjectEncoder:
    descriptor_encoder: RDKitDescriptorBackbone
    object_encoder: ObjectEncoder
    preprocessor: FeaturePreprocessor
    encoder_identity: dict[str, Any]
    artifact_hash: str
    device: torch.device

    @property
    def embedding_dim(self) -> int:
        return 512

    @torch.inference_mode()
    def encode(self, objects: Sequence[FrozenObjectSpec]) -> torch.Tensor:
        if not objects:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)
        slot_counts = {len(item.slots) for item in objects}
        if len(slot_counts) != 1:
            raise ValueError("Frozen RDKit Stage 2 batch must share topology")
        slot_count = slot_counts.pop()
        expected_topology = "molecule" if slot_count == 1 else "il"
        if any(item.topology != expected_topology for item in objects):
            raise ValueError("Frozen RDKit Stage 2 object topology mismatch")
        raw_rows = []
        role_values = []
        names = rdkit_descriptor_names()
        for item in objects:
            for role, smiles in item.slots:
                if role not in ROLE_TO_ID:
                    raise ValueError(f"Unsupported frozen Stage 2 role: {role}")
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    raise ValueError(f"Invalid canonical Stage 3 SMILES: {smiles}")
                raw_rows.append(calculate_descriptors(molecule, names))
                role_values.append(ROLE_TO_ID[role])
        features = torch.from_numpy(
            self.preprocessor.transform(np.stack(raw_rows))
        ).to(self.device)
        slots = self.descriptor_encoder.encode(features).reshape(
            len(objects), slot_count, self.embedding_dim
        )
        roles = torch.tensor(
            role_values, dtype=torch.long, device=self.device
        ).reshape(len(objects), slot_count)
        values = self.object_encoder(slots, roles).float().cpu()
        if not torch.isfinite(values).all():
            raise RuntimeError("Frozen RDKit Stage 2 produced non-finite embeddings")
        return values


def _load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("kind") != STAGE2_ENCODER_KIND
        or payload.get("format_version") != STAGE2_ENCODER_VERSION
    ):
        raise ValueError("Stage 3 requires a Stage 2 encoder artifact v1")
    if payload.get("identity_contract_version") != IDENTITY_CONTRACT_VERSION:
        raise ValueError(
            "Stage 2 encoder predates identity contract v1; retrain Stage 2"
        )
    required = {
        "semantic_identity",
        "stage1_backbone",
        "object_encoder",
        "stage1_config",
        "stage1_feature_identity",
        "stage1_encoding_contract",
        "feature_artifacts",
        "object_encoder_config",
        "role_to_id",
        "state_hashes",
    }
    if not required.issubset(payload):
        raise ValueError("Stage 2 encoder artifact contract is incomplete")
    state_hashes = {
        "stage1_backbone": tensor_state_hash(
            "stage2.encoder-state", payload["stage1_backbone"]
        ),
        "object_encoder": tensor_state_hash(
            "stage2.encoder-state", payload["object_encoder"]
        ),
    }
    if payload["state_hashes"] != state_hashes:
        raise ValueError("Stage 2 encoder state integrity mismatch")
    expected = build_stage2_encoder_identity(
        stage1_feature_identity=payload["stage1_feature_identity"],
        stage1_encoding_contract=payload["stage1_encoding_contract"],
        stage1_state_hash=state_hashes["stage1_backbone"],
        object_encoder_contract=payload["object_encoder_config"],
        object_encoder_state_hash=state_hashes["object_encoder"],
        role_to_id=payload["role_to_id"],
    )
    require_compatible_identity(
        expected,
        payload["semantic_identity"],
        context="Stage 2 encoder artifact",
    )
    if payload["role_to_id"] != ROLE_TO_ID:
        raise ValueError("Stage 2 encoder role mapping mismatch")
    return payload


def load_frozen_object_encoder(
    encoder_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> FrozenStage2ObjectEncoder | FrozenRDKitStage2ObjectEncoder:
    path = Path(encoder_path)
    kind = torch.load(path, map_location="cpu", weights_only=False).get("kind")
    if kind == STAGE2_RDKIT_ENCODER_KIND:
        payload = load_rdkit_stage2_encoder_artifact(path)
        contract = payload["descriptor_contract"]
        schema = contract.get("schema", {})
        if schema.get("descriptor_names") != list(rdkit_descriptor_names()):
            raise ValueError("RDKit Stage 2 descriptor schema mismatch")
        if schema.get("rdkit_version") != rdBase.rdkitVersion:
            raise ValueError("RDKit Stage 2 runtime version mismatch")
        if payload.get("role_to_id") != ROLE_TO_ID:
            raise ValueError("RDKit Stage 2 encoder role mapping mismatch")
        encoder_spec = contract["encoder"]
        target_device = torch.device(device)
        descriptor_encoder = RDKitDescriptorBackbone(
            int(contract["retained_width"]),
            hidden_dim=int(encoder_spec["hidden_dim"]),
            output_dim=int(encoder_spec["output_dim"]),
            dropout=float(encoder_spec["dropout"]),
        )
        descriptor_encoder.load_state_dict(
            payload["descriptor_encoder"], strict=True
        )
        object_config = payload["object_encoder_config"]
        object_encoder = ObjectEncoder(
            512,
            8,
            num_layers=int(object_config["layers"]),
            feedforward_dim=int(object_config["ffn_dim"]),
            dropout=float(object_config["dropout"]),
        )
        object_encoder.load_state_dict(payload["object_encoder"], strict=True)
        descriptor_encoder.to(target_device).eval()
        object_encoder.to(target_device).eval()
        for parameter in [*descriptor_encoder.parameters(), *object_encoder.parameters()]:
            parameter.requires_grad_(False)
        return FrozenRDKitStage2ObjectEncoder(
            descriptor_encoder=descriptor_encoder,
            object_encoder=object_encoder,
            preprocessor=FeaturePreprocessor.from_dict(contract["preprocessing"]),
            encoder_identity=payload["semantic_identity"],
            artifact_hash=sha256_file(path),
            device=target_device,
        )
    payload = _load_payload(path)
    config = config_from_dict(payload["stage1_config"])
    feature_artifacts = payload["feature_artifacts"]
    validate_feature_generation_runtime(
        {
            "feature_generation_contract": payload["stage1_encoding_contract"][
                "feature_generation_contract"
            ]
        }
    )
    vocabulary = SmilesTokenizer.from_payload(feature_artifacts["tokenizer.json"])
    schema = DescriptorSchema.from_payload(
        feature_artifacts["descriptor_schema.json"],
        expected_raw_names=rdkit_descriptor_names(),
    )
    standardizer = DescriptorStandardizer.from_payload(
        feature_artifacts["descriptor_scaler.json"],
        expected_names=schema.selected_names,
    )
    target_device = torch.device(device)
    backbone = MultimodalPretrainModel(config, vocabulary, schema)
    missing, unexpected = backbone.load_state_dict(
        payload["stage1_backbone"], strict=False
    )
    if unexpected or any(
        not any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in RECONSTRUCTION_MODULES
        )
        for name in missing
    ):
        raise ValueError("Stage 2 encoder Stage 1 state contract mismatch")
    object_config = payload["object_encoder_config"]
    object_encoder = ObjectEncoder(
        config.model.d_model,
        config.model.n_heads,
        num_layers=int(object_config["layers"]),
        feedforward_dim=int(object_config["ffn_dim"]),
        dropout=float(object_config["dropout"]),
    )
    object_encoder.load_state_dict(payload["object_encoder"], strict=True)
    backbone.to(target_device).eval()
    object_encoder.to(target_device).eval()
    for parameter in [*backbone.parameters(), *object_encoder.parameters()]:
        parameter.requires_grad_(False)
    return FrozenStage2ObjectEncoder(
        backbone=backbone,
        object_encoder=object_encoder,
        packer=MultimodalPacker(vocabulary),
        pretrain_config=config,
        descriptor_schema=schema,
        descriptor_standardizer=standardizer,
        encoder_identity=payload["semantic_identity"],
        artifact_hash=sha256_file(path),
        device=target_device,
    )


def load_stage2_encoder_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    kind = torch.load(resolved, map_location="cpu", weights_only=False).get("kind")
    if kind == STAGE2_RDKIT_ENCODER_KIND:
        return dict(load_rdkit_stage2_encoder_artifact(resolved)["semantic_identity"])
    return dict(_load_payload(resolved)["semantic_identity"])


__all__ = [
    "FrozenObjectSpec",
    "FrozenRDKitStage2ObjectEncoder",
    "FrozenStage2ObjectEncoder",
    "load_frozen_object_encoder",
    "load_stage2_encoder_identity",
]
