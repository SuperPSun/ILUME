"""Reference-row-preserving materialization without Stage1 features."""
from copy import deepcopy
import json
import shutil

import torch

from common.identity import require_compatible_identity, semantic_identity
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.training import canonical_json_sha256
from stage2.data import load_artifact_registry, Stage2TaskDataset, STAGE2_FROZEN_ARTIFACT_KIND
from stage2.identity import build_stage2_data_identity
from stage2.frozen import FrozenObjectSpec


def reference_file(root, metadata, name):
    path = root / name
    if sha256_file(path) != metadata["artifact_hashes"][name]:
        raise ValueError(f"Reference artifact SHA mismatch: {name}")
    return path


def prepare_stage2(source, config):
    reference = config.preparation.reference_artifacts
    metadata = json.loads((reference / "metadata.json").read_text())
    registry = load_artifact_registry(reference)
    config.validate_registry(registry)
    expected = build_stage2_data_identity(config, registry, metadata["stage1_feature_identity"])
    require_compatible_identity(expected, metadata["semantic"]["identities"]["data"], context="ChemBERTa reference rows")
    identity = build_stage2_data_identity(config, registry, source.feature_identity)
    root = config.data.artifacts_dir
    if root.exists():
        raise FileExistsError(f"External frozen entity output already exists: {root}")
    entries = json.loads(reference_file(reference, metadata, "entity_index.json").read_text())["entries"]
    entries = [{key: entry[key] for key in ("entity_id", "role", "role_id", "canonical_smiles")} for entry in entries]
    if any(entry["entity_id"] != i for i, entry in enumerate(entries)):
        raise ValueError("Reference entity IDs are not contiguous")
    copied = []
    for task in registry.task_ids:
        for split in ("train", "valid"):
            Stage2TaskDataset(reference, task, split)
            copied.append(f"tasks/{task}/{split}.pt")
    copied += [name for name in ("scalers.json", "excluded_entities.csv", "excluded_rows.csv") if name in metadata["artifact_hashes"]]
    root.mkdir(parents=True, exist_ok=False)
    for relative in copied:
        path = reference_file(reference, metadata, relative)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith(".pt"):
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["kind"] = STAGE2_FROZEN_ARTIFACT_KIND
            atomic_torch_save(target, payload)
        else:
            shutil.copyfile(path, target)
    atomic_json(root / "entity_index.json", {"kind": STAGE2_FROZEN_ARTIFACT_KIND,
                                             "format_version": metadata["format_version"], "entries": entries})
    values = []
    for start in range(0, len(entries), config.training.batch_size):
        selected = entries[start:start + config.training.batch_size]
        objects = [FrozenObjectSpec("molecule", ((e["role"], e["canonical_smiles"]),)) for e in selected]
        values.append(source.encode_slots(objects)[0][:, 0])
    features = torch.cat(values)
    frozen_identity = semantic_identity("ablation.external-frozen-cache.v1", {
        "data_identity": identity["hash"], "stage1_state_hash": source.encoder_identity["payload"]["stage1_state_hash"],
        "feature_identity": source.feature_identity["hash"], "entity_input_dim": 1241, "atom_dim": 0,
        "frozen_encoder_identity": source.encoder_identity["hash"]})
    atomic_torch_save(root / "entity_features.pt", {"kind": STAGE2_FROZEN_ARTIFACT_KIND,
                                                   "identity": frozen_identity, "features": features})
    manifest = {"kind": STAGE2_FROZEN_ARTIFACT_KIND, "identity": frozen_identity,
                "artifact_sha256": sha256_file(root / "entity_features.pt")}
    atomic_json(root / "frozen_entities.json", manifest)
    updated = deepcopy(metadata)
    updated.update(kind=STAGE2_FROZEN_ARTIFACT_KIND, data_signature=identity["hash"],
                   frozen_encoder_identity=source.encoder_identity,
                   frozen_encoder=config.frozen_encoder.to_dict(), feature_generation_contract=source.encoder_identity["payload"]["extraction"])
    updated["semantic"]["identities"].update(data=identity, entity=frozen_identity)
    updated["stage1_feature_identity"] = source.feature_identity
    updated["pretrain_artifact_hash"] = source.feature_identity["hash"]
    updated["entity_artifact_hash"] = manifest["artifact_sha256"]
    files = [*copied, "entity_index.json", "entity_features.pt", "frozen_entities.json"]
    updated["artifact_hashes"] = {name: sha256_file(root / name) for name in files}
    updated["reference_data_identity"] = metadata["semantic"]["identities"]["data"]
    updated.pop("entity_index", None)
    updated["locator"] = {"files": {name: name for name in files}}
    updated["integrity"] = {"files": {name: {"sha256": digest, "size_bytes": (root / name).stat().st_size}
                                       for name, digest in updated["artifact_hashes"].items()}}
    atomic_json(root / "metadata.json", updated)
    return {**updated["summary"], "frozen_encoder_identity": source.encoder_identity["hash"]}


def validate_stage3_reference(config, registry, objects, normalization):
    from stage3.data import source_hashes
    root = config.preparation.reference_artifacts
    metadata = json.loads((root / "metadata.json").read_text())
    registry_hash = canonical_json_sha256({task: spec.prepared_dict() for task, spec in registry.items()})
    original = torch.load(reference_file(root, metadata, "object_index.pt"), map_location="cpu", weights_only=True)
    norms = json.loads(reference_file(root, metadata, "normalization.json").read_text())
    if (metadata["registry_hash"] != registry_hash or metadata["source_hashes"] != source_hashes(config, registry)
            or original["objects"] != [key.to_dict() for key in objects] or norms != normalization):
        raise ValueError("ChemBERTa Stage3 reference row/order/normalization mismatch")
