from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from common.identity import tensor_state_hash
from common.io import sha256_file
from common.training import seed_everything
from stage2 import load_frozen_object_encoder
from stage2.frozen import STAGE2_ENCODER_KIND
from stage2.rdkit_train import STAGE2_RDKIT_ENCODER_KIND

from .config import Stage3Config, effective_training_seed
from .model import Ownership, Stage3SparseModel


OBJECT_ENCODER_OWNER = Ownership("ENCODER_STAGE2")


def load_object_phase1_source(path: Any) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("kind") not in {STAGE2_ENCODER_KIND, STAGE2_RDKIT_ENCODER_KIND}
        or not isinstance(payload.get("provenance"), Mapping)
        or not isinstance(payload.get("state_hashes"), Mapping)):
        raise ValueError("ObjectEncoder Phase 1 requires a supported Stage 2 encoder")
    return payload


def validate_encoder_source(config: Stage3Config) -> Mapping[str, Any]:
    path = config.initialization.stage2_encoder
    recipe = config.training.object_encoder_phase1
    assert path is not None and recipe is not None
    payload = load_object_phase1_source(path)
    provenance = payload["provenance"]
    if recipe.source_variant == "zero_update":
        if (provenance.get("zero_stage2_training") is not True
            or provenance.get("optimizer_updates") != 0
            or provenance.get("initialization_seed") != config.data.seed
            or provenance.get("stage2_checkpoint_hash") is not None):
            raise ValueError("No-Stage2 arm requires the matching zero-update encoder")
        assert recipe.paired_trained_encoder is not None
        trained = load_object_phase1_source(recipe.paired_trained_encoder)
        if (
            trained["provenance"].get("stage2_checkpoint_hash") is None
            or trained["provenance"].get("refinement_boundary_epoch") != 10
            or provenance.get("paired_trained_encoder_sha256") != sha256_file(recipe.paired_trained_encoder)
            or provenance.get("stage1_checkpoint_hash") != trained["provenance"].get("stage1_checkpoint_hash")
            or payload.get("model_contract") != trained.get("model_contract")
        ):
            raise ValueError("No-Stage2 encoder does not match its trained control")
        manifest_path = Path(path).with_name("manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("kind") != "ilume_stage2_zero_update_control"
            or manifest.get("optimizer_updates") != 0
            or manifest.get("stage2_encoder_sha256") != sha256_file(path)
            or manifest.get("paired_trained_encoder_sha256") != sha256_file(recipe.paired_trained_encoder)
            or manifest.get("initial_shared_state_hash") != provenance.get("initial_shared_state_hash")
        ):
            raise ValueError("No-Stage2 zero-update manifest mismatch")
    elif provenance.get("stage2_checkpoint_hash") is None:
        raise ValueError("Trained arm requires a completed Stage 2 encoder")
    elif payload.get("kind") == STAGE2_ENCODER_KIND and provenance.get("refinement_boundary_epoch") != 10:
        raise ValueError("Trained arm requires the final ten-epoch Stage 2 encoder")
    return {"state_hashes": dict(payload["state_hashes"]), "provenance": dict(provenance)}


class ObjectPhase1Model(Stage3SparseModel):
    def __init__(self, *args: Any, object_encoder: nn.Module, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.object_phase1 = True
        self.stage2_object_encoder = object_encoder
        self.joint_upstream_owners = (OBJECT_ENCODER_OWNER,)
        self._own_modules(OBJECT_ENCODER_OWNER, object_encoder)
        self._validate_ownership()


class ObjectPhase1Representations:
    def __init__(self, model: ObjectPhase1Model, payload: Mapping[str, Any]) -> None:
        self.model = model
        self.slots = payload["slots"]
        self.roles = payload["roles"]
        self.counts = payload["counts"]
        self.output_dim = int(self.slots.shape[-1])
        self.input_dims = None
        self.knowledge_bank = None
        self._embeddings: torch.Tensor | None = None
        self.final_embedding_hash: str | None = None

    def values(self, object_ids: torch.Tensor, topology: str) -> torch.Tensor:
        indices = object_ids.detach().cpu().long()
        device = next(self.model.stage2_object_encoder.parameters()).device
        expected = 2 if topology == "il" else 1 if topology == "molecule" else 0
        if not expected or not torch.all(self.counts[indices] == expected):
            raise ValueError("Stage 3 frozen slot topology mismatch")
        if self._embeddings is not None:
            return self._embeddings[indices].to(device)
        slots = self.slots[indices, :expected].to(device)
        roles = self.roles[indices, :expected].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and torch.is_grad_enabled(),
        ):
            return self.model.stage2_object_encoder(slots, roles).float()

    @torch.no_grad()
    def freeze_after_phase1(self, model: ObjectPhase1Model, phase1_hash: str) -> None:
        if model is not self.model:
            raise ValueError("Stage 3 ObjectEncoder representation model mismatch")
        model.eval()
        result = torch.empty((len(self.counts), self.output_dim), dtype=torch.float32)
        for topology, count in (("il", 2), ("molecule", 1)):
            indices = torch.where(self.counts == count)[0]
            for part in indices.split(256):
                result[part] = self.values(part, topology).cpu()
        if not torch.isfinite(result).all():
            raise ValueError("Stage 3 Phase 1 final ObjectEncoder emitted non-finite values")
        self._embeddings = result
        self.final_embedding_hash = tensor_state_hash(
            "stage3.object-phase1.final-embeddings.v1", {"embeddings": result}
        )


@torch.no_grad()
def validate_initial_object_embeddings(
    model: ObjectPhase1Model,
    representations: ObjectPhase1Representations,
    prepared: Mapping[str, Any],
) -> None:
    """Catch slot/order/role drift before the first optimizer update."""
    reference = prepared["objects"]["embeddings"]
    model.eval()
    for topology, count in (("il", 2), ("molecule", 1)):
        indices = torch.where(representations.counts == count)[0]
        if not len(indices):
            continue
        selected = indices[::max(1, len(indices) // 16)][:16]
        observed = representations.values(selected, topology).cpu()
        if not torch.allclose(observed, reference[selected], rtol=1e-3, atol=1e-3):
            raise ValueError("Frozen Stage 1 slots differ from prepared initial Object embeddings")


def build_object_phase1_model(
    config: Stage3Config, prepared: Mapping[str, Any], *, fold: int, device: torch.device
) -> tuple[ObjectPhase1Model, ObjectPhase1Representations]:
    if prepared.get("slots") is None:
        raise ValueError("Stage 3 ObjectEncoder Phase 1 requires prepared frozen entity slots")
    validate_encoder_source(config)
    path = config.initialization.stage2_encoder
    assert path is not None
    source = load_frozen_object_encoder(path, device="cpu")
    seed_everything(effective_training_seed(config) + fold)
    model = ObjectPhase1Model(
        config.model, prepared["registry"], source.embedding_dim,
        group_configs=config.groups, task_configs=config.tasks,
        task_private_recipes={
            task: config.resolved_private_recipe(task)
            for task, spec in config.tasks.items() if spec.enabled
        },
        object_encoder=source.object_encoder,
    ).to(device)
    model.set_trainable_owners(set(model.parameter_ownership().values()))
    return model, ObjectPhase1Representations(model, prepared["slots"])
