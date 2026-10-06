from __future__ import annotations

import json

import torch

from common.identity import semantic_identity
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.progress import ProgressReporter
from common.training import resolve_device
from stage1.identity import encoding_state_hash
from stage1.masking import MultimodalPacker

from .data import Stage2EntityDataset
from .home_train import load_backbone
from .model import encode_object_entities


@torch.no_grad()
def prepare_frozen_entities(recipe):
    if not recipe.freeze_stage1:
        return None
    loaded = load_backbone(recipe)
    if not loaded.config.is_dual_view:
        return None
    config = recipe.stage2
    root = config.data.artifacts_dir
    entities = Stage2EntityDataset(root)
    data_identity = entities.metadata["semantic"]["identities"]["data"]["hash"]
    source_hash = encoding_state_hash(loaded.model)
    identity = semantic_identity("stage2.frozen-entities.v4", {"data_identity": data_identity, "stage1_state_hash": source_hash, "feature_identity": loaded.artifact_hash, "entity_input_dim": loaded.model.entity_dim + 217, "atom_dim": loaded.model.atom_dim})
    path = root / "frozen_entities.pt"
    manifest_path = root / "frozen_entities.json"
    if path.exists():
        validate_frozen_entity_source(entities, loaded.model)
        if entities.frozen_entity_manifest["identity"] != identity:
            raise ValueError("Frozen entity cache identity mismatch")
        return entities.frozen_entity_manifest
    device = resolve_device(config.training.device)
    loaded.model.requires_grad_(False).to(device).eval()
    packer = MultimodalPacker(loaded.vocabulary)
    embeddings, atoms = [], []
    with ProgressReporter().bar(total=len(entities), desc="Freeze Stage1 entities", unit="entity") as progress:
        for start in range(0, len(entities), config.training.batch_size):
            samples = entities.samples[start:start + config.training.batch_size]
            batch = packer(samples).to(device)
            encoded = encode_object_entities(loaded.model, batch)
            embeddings.extend(encoded.entity_embedding.float().cpu().unbind())
            atoms.extend(encoded.atom_states.float().cpu().split([len(sample["atom_categorical"]) for sample in samples]))
            progress.update(len(samples))
    payload = {"kind": "ilume_stage2_frozen_entities_v4", "identity": identity, "samples": [{"sample_id": sample["sample_id"], "entity": embedding, "atoms": atom} for sample, embedding, atom in zip(entities.samples, embeddings, atoms, strict=True)]}
    atomic_torch_save(path, payload)
    manifest = {"kind": payload["kind"], "identity": identity, "artifact_sha256": sha256_file(path)}
    atomic_json(manifest_path, manifest)
    return manifest


def validate_frozen_entity_source(entities, backbone):
    manifest = entities.frozen_entity_manifest
    if manifest is None or manifest["identity"]["payload"]["stage1_state_hash"] != encoding_state_hash(backbone):
        raise ValueError("Frozen Stage1 entity cache source mismatch; run v4 Stage2 prepare")
