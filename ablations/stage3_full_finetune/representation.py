from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import nn

from common.identity import tensor_state_hash
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.training import canonical_json_sha256
from stage2 import load_frozen_object_encoder
from stage2.model import RECONSTRUCTION_MODULES
from stage3.config import Stage3Config, stage3_config_from_dict
from stage3.data import ObjectKey, Stage3RepresentationStore
from stage3.identity import metadata_identity
from stage3.model import Ownership, Stage3SparseModel
from stage3.prepare import load_prepared_stage3


STAGE1_OWNER = Ownership("ENCODER_STAGE1")
STAGE2_OWNER = Ownership("ENCODER_STAGE2")
FEATURE_KIND = "ilume_stage3_full_finetune_features"


@dataclass(frozen=True)
class FinetuneRecipe:
    stage1_lr: float
    stage2_lr: float
    epochs: int
    warmup_ratio: float
    min_lr_ratio: float

    def validate(self, config: Stage3Config) -> None:
        if config.training.schedule_mode != "three_phase" or config.training.three_phase is None or config.representation is not None:
            raise ValueError("Encoder fine-tuning requires Object-backed three-phase Stage 3")
        if self.stage1_lr <= 0 or self.stage2_lr <= 0:
            raise ValueError("Encoder fine-tuning learning rates must be positive")
        if self.epochs != config.training.three_phase.global_scope.epochs:
            raise ValueError("Encoder fine-tuning must span exactly Phase 1")
        if not 0 <= self.warmup_ratio < 1 or not 0 < self.min_lr_ratio <= 1:
            raise ValueError("Invalid encoder fine-tuning scheduler ratio")
        if config.initialization.plugin is not None or config.transfer_knowledge is not None:
            raise ValueError("Encoder fine-tuning cannot be combined with another Stage 3 ablation")
        if config.training.sampling_mode != "raw" or config.training.joint_gradient_clip_mode != "ownership":
            raise ValueError("Encoder fine-tuning requires Base raw sampling and ownership clipping")
        if config.training.amp_dtype != "bf16":
            raise ValueError("Encoder fine-tuning requires Base BF16 precision")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def load_config(path: str | Path) -> tuple[Stage3Config, FinetuneRecipe]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("encoder_finetune"), dict):
        raise ValueError("Fine-tuning YAML requires encoder_finetune mapping")
    base = dict(raw)
    recipe_raw = base.pop("encoder_finetune")
    if set(recipe_raw) != set(FinetuneRecipe.__dataclass_fields__):
        raise ValueError("Encoder fine-tuning recipe has missing or unknown fields")
    recipe = FinetuneRecipe(**recipe_raw)
    config = stage3_config_from_dict(base)
    recipe.validate(config)
    return config, recipe


def object_keys(prepared: Mapping[str, Any]) -> tuple[ObjectKey, ...]:
    return tuple(
        ObjectKey(item["topology"], tuple(tuple(slot) for slot in item["slots"]))
        for item in prepared["objects"]["objects"]
    )


def prepare_features(config: Stage3Config, output: str | Path) -> dict[str, Any]:
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"Fine-tuning feature directory already exists: {root}")
    prepared = load_prepared_stage3(config)
    keys = object_keys(prepared)
    encoder_path = config.initialization.stage2_encoder
    assert encoder_path is not None
    encoder = load_frozen_object_encoder(encoder_path, device="cpu")
    if not hasattr(encoder, "input_sample"):
        raise ValueError("Fine-tuning requires the Stage 2 Object encoder")
    samples: dict[tuple[str, str], dict[str, Any]] = {}
    for key in keys:
        for role, smiles in key.slots:
            if (role, smiles) not in samples:
                samples[(role, smiles)] = encoder.input_sample(role, smiles)
    payload = {
        "kind": FEATURE_KIND,
        "format_version": 1,
        "prepared_identity": metadata_identity(prepared["metadata"], "prepared", context="Fine-tuning prepare")["hash"],
        "stage2_encoder_identity": encoder.encoder_identity["hash"],
        "stage2_encoder_sha256": sha256_file(encoder_path),
        "object_keys_hash": canonical_json_sha256([key.to_dict() for key in keys]),
        "samples": samples,
    }
    root.mkdir(parents=True)
    atomic_torch_save(root / "features.pt", payload)
    manifest = {key: value for key, value in payload.items() if key != "samples"}
    manifest["sample_count"] = len(samples)
    manifest["artifact_sha256"] = sha256_file(root / "features.pt")
    atomic_json(root / "features.json", manifest)
    return manifest


def load_features(config: Stage3Config, root: str | Path, prepared: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(root)
    manifest = json.loads((root / "features.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != FEATURE_KIND or manifest.get("artifact_sha256") != sha256_file(root / "features.pt"):
        raise ValueError("Fine-tuning feature artifact integrity mismatch")
    payload = torch.load(root / "features.pt", map_location="cpu", weights_only=False)
    encoder_path = config.initialization.stage2_encoder
    assert encoder_path is not None
    expected = {
        "kind": FEATURE_KIND,
        "format_version": 1,
        "prepared_identity": metadata_identity(prepared["metadata"], "prepared", context="Fine-tuning features")["hash"],
        "stage2_encoder_identity": prepared["metadata"]["stage2_encoder_identity"]["hash"],
        "stage2_encoder_sha256": sha256_file(encoder_path),
        "object_keys_hash": canonical_json_sha256([key.to_dict() for key in object_keys(prepared)]),
    }
    if any(payload.get(name) != value or manifest.get(name) != value for name, value in expected.items()):
        raise ValueError("Fine-tuning features do not match prepared data or encoder")
    if manifest.get("sample_count") != len(payload.get("samples", {})):
        raise ValueError("Fine-tuning feature sample count mismatch")
    return payload


class FinetuneStage3Model(Stage3SparseModel):
    def __init__(self, *args: Any, backbone: nn.Module, object_encoder: nn.Module, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for name in RECONSTRUCTION_MODULES:
            if hasattr(backbone, name):
                delattr(backbone, name)
        self.stage1_encoder = backbone
        self.stage2_object_encoder = object_encoder
        self.joint_upstream_owners = (STAGE1_OWNER, STAGE2_OWNER)
        self._own_modules(STAGE1_OWNER, self.stage1_encoder)
        self._own_modules(STAGE2_OWNER, self.stage2_object_encoder)
        self._validate_ownership()


class LiveRepresentationStore(Stage3RepresentationStore):
    def __init__(
        self, model: FinetuneStage3Model, packer: Any,
        keys: tuple[ObjectKey, ...], samples: Mapping[tuple[str, str], dict[str, Any]],
    ) -> None:
        self.model = model
        self.packer = packer
        self.keys = keys
        self.samples = samples
        self.output_dim = model.d_model
        self.input_dims = None
        self.knowledge_bank = None
        self._embeddings: torch.Tensor | None = None
        self._phase1_hash: str | None = None

    def values(self, object_ids: torch.Tensor, topology: str) -> torch.Tensor:
        ids = object_ids.cpu().long().tolist()
        device = next(self.model.parameters()).device
        if self._embeddings is not None:
            return self._embeddings[ids].to(device)
        if not ids:
            return torch.empty((0, self.output_dim), device=device)
        keys = [self.keys[index] for index in ids]
        if any(key.topology != topology for key in keys):
            raise ValueError("Fine-tuning object topology mismatch")
        slot_count = len(keys[0].slots)
        if any(len(key.slots) != slot_count for key in keys):
            raise ValueError("Fine-tuning object slot count mismatch")
        from stage1.features import ROLE_TO_ID

        batch = self.packer([
            self.samples[(role, smiles)]
            for key in keys for role, smiles in key.slots
        ]).to(device)
        roles = torch.tensor(
            [ROLE_TO_ID[role] for key in keys for role, _ in key.slots],
            dtype=torch.long, device=device,
        ).reshape(len(keys), slot_count)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda" and torch.is_grad_enabled(),
        ):
            slots = self.model.stage1_encoder.encode_entity(batch).entity_embedding
            slots = slots.reshape(len(keys), slot_count, self.output_dim)
            result = self.model.stage2_object_encoder(slots, roles)
        return result.float()

    @torch.no_grad()
    def freeze_after_phase1(self, model: FinetuneStage3Model, phase1_hash: str) -> None:
        if model is not self.model:
            raise ValueError("Fine-tuning representation model mismatch")
        model.eval()
        values = torch.empty((len(self.keys), self.output_dim), dtype=torch.float32)
        for topology in ("il", "molecule"):
            indices = [index for index, key in enumerate(self.keys) if key.topology == topology]
            for start in range(0, len(indices), 8):
                selected = indices[start:start + 8]
                values[selected] = self.values(torch.tensor(selected), topology).cpu()
        if not torch.isfinite(values).all():
            raise RuntimeError("Fine-tuned representation contains non-finite values")
        self._embeddings = values
        self._phase1_hash = phase1_hash


def encoder_state_hashes(model: FinetuneStage3Model) -> dict[str, str]:
    return {
        "stage1": tensor_state_hash("stage3.full-finetune.stage1", model.stage1_encoder.state_dict()),
        "stage2": tensor_state_hash("stage3.full-finetune.stage2", model.stage2_object_encoder.state_dict()),
    }
