from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from common.identity import semantic_hash, tensor_state_hash, validate_semantic_identity
from common.io import sha256_file
from stage1.config import config_from_dict
from stage1.descriptors import DescriptorSchema, rdkit_descriptor_names
from stage1.model import build_stage1_model
from stage1.tokenizer import SmilesTokenizer

from .home_config import home_recipe_from_dict
from .home_contract import (
    BATCH_SAMPLE_AGGREGATION, BATCH_SAMPLE_WEIGHTING,
    source_groups, state_hash, transferable_state,
)
from .home_model import SimulationHoME
from .model import RECONSTRUCTION_MODULES
from .registry import Stage2Registry


STAGE2_HOME_FINAL_KIND = "ilume_stage2_home_final_v2"
STAGE2_HOME_V4_FINAL_KIND = "ilume_stage2_home_final_v4"


def final_kind(recipe):
    if recipe.stage2.is_entity_home:
        return "ilume_stage2_entity_home_final_v4"
    return STAGE2_HOME_V4_FINAL_KIND if recipe.freeze_stage1 else STAGE2_HOME_FINAL_KIND


def full_state_hash(state: Mapping[str, torch.Tensor], *, version: int | None = None) -> str:
    return tensor_state_hash("stage2.entity-home.full-model.v4", state)


def full_owner_manifest(model: SimulationHoME) -> dict[str, str]:
    home = model.home.ownership_manifest()
    result: dict[str, str] = {}
    for name, _ in model.named_parameters():
        if name.startswith("backbone."):
            result[name] = "STAGE1"
        elif name.startswith("object_encoder."):
            result[name] = "OBJECT"
        elif name.startswith("home."):
            result[name] = home[name.removeprefix("home.")]
        elif name.startswith("atom_adapter."):
            result[name] = "ATOM_ADAPTER"
        else:
            raise ValueError(f"Unexpected Stage 2 full model parameter: {name}")
    return result


def load_home_final(path: str | Path) -> tuple[dict[str, Any], SimulationHoME, SmilesTokenizer]:
    """Load the self-contained, fixed-final simulation model with its manifest."""
    artifact = Path(path)
    manifest = json.loads(artifact.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest.get("artifact") != artifact.name or manifest.get("artifact_sha256") != sha256_file(artifact):
        raise ValueError("Stage 2 final artifact SHA mismatch")
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    if (payload.get("kind") != "ilume_stage2_entity_home_final_v4"
            or manifest.get("kind") != payload.get("kind") or payload.get("format_version") != 4):
        raise ValueError("Unsupported Stage 2 full HoME artifact kind")
    identity = payload["training_identity"]
    validate_semantic_identity(identity)
    math_contract = identity["payload"].get("math_contract", {})
    if (math_contract.get("gradient_aggregation") != BATCH_SAMPLE_AGGREGATION
            or math_contract.get("gradient_weighting") != BATCH_SAMPLE_WEIGHTING):
        raise ValueError("Stage 2 final requires batch_sample_weighted_owner_raw_v1 artifacts")
    validate_semantic_identity(payload["stage2_data_identity"])
    validate_semantic_identity(payload["stage1_feature_identity"])
    validate_semantic_identity(manifest["training_identity"])
    validate_semantic_identity(manifest["stage2_data_identity"])
    if (manifest["training_identity"]["hash"] != identity["hash"]
            or manifest["stage2_data_identity"]["hash"] != payload["stage2_data_identity"]["hash"]
            or identity["payload"]["stage2_data_identity"] != payload["stage2_data_identity"]["hash"]
            or manifest.get("registry_hash") != payload.get("registry_hash")
            or manifest.get("full_model_state_hash") != payload.get("full_model_state_hash")
            or manifest.get("owner_manifest") != payload.get("owner_manifest")
            or manifest.get("shared_state_hash") != payload.get("shared_state_hash")
            or manifest.get("checkpoint_sha256") != payload.get("checkpoint_sha256")
            or manifest.get("feature_artifacts_hash") != payload.get("feature_artifacts_hash")
            or manifest.get("scalers_hash") != payload.get("scalers_hash")):
        raise ValueError("Stage 2 final manifest identity mismatch")
    state = payload["full_model_state"]
    if not isinstance(state, dict) or full_state_hash(state, version=payload["format_version"]) != payload["full_model_state_hash"]:
        raise ValueError("Stage 2 full model state hash mismatch")
    registry = Stage2Registry.from_snapshot(
        payload["registry"], registry_hash=payload["registry_hash"],
        catalog_sha256=payload["catalog_sha256"],
    )
    source_groups(registry)
    if set(payload["scalers"]) != set(registry.task_ids):
        raise ValueError("Stage 2 full model registry/scaler mismatch")
    if semantic_hash("stage2.home.scalers.v1", payload["scalers"]) != payload["scalers_hash"]:
        raise ValueError("Stage 2 full model scaler hash mismatch")
    for spec in registry.tasks:
        stats = payload["scalers"][spec.task_id]
        if (set(stats["conditions"]) != set(spec.condition_columns)
                or set(stats["targets"]) != set(spec.target_columns)):
            raise ValueError("Stage 2 full model scaler columns mismatch")
        for value in (*stats["conditions"].values(), *stats["targets"].values()):
            if not math.isfinite(float(value["mean"])) or not math.isfinite(float(value["scale"])) or float(value["scale"]) <= 0:
                raise ValueError("Stage 2 full model scaler values are invalid")
    recipe = home_recipe_from_dict(payload["recipe"])
    recipe.stage2.validate_registry(registry)
    if payload["kind"] != final_kind(recipe):
        raise ValueError("Stage 2 final representation family mismatch")
    if (recipe.stage2.experiment_dict() != identity["payload"]["stage2_config"]
            or asdict(recipe.stage3.model) != identity["payload"]["stage3_model"]
            or identity["payload"].get("epochs") != recipe.stage2_epochs
            or manifest.get("fixed_final_epoch") != recipe.stage2_epochs):
        raise ValueError("Stage 2 full model recipe mismatch")
    features = payload["feature_artifacts"]
    if semantic_hash("stage2.home.feature-artifacts.v1", features) != payload["feature_artifacts_hash"]:
        raise ValueError("Stage 2 full model feature artifact hash mismatch")
    vocabulary = SmilesTokenizer.from_payload(features["tokenizer.json"])
    schema = DescriptorSchema.from_payload(
        features["descriptor_schema.json"], expected_raw_names=rdkit_descriptor_names(),
    )
    stage1_config = config_from_dict(payload["stage1_config"])
    backbone = build_stage1_model(stage1_config, vocabulary, schema, encoder_only=stage1_config.is_dual_view)
    model = SimulationHoME(backbone, registry, recipe.stage3, recipe.stage2)
    model.load_state_dict(state, strict=True)
    if payload["owner_manifest"] != full_owner_manifest(model):
        raise ValueError("Stage 2 full model owner manifest mismatch")
    shared = transferable_state(model.home)
    if (set(shared) != set(payload["shared_state"])
            or any(not torch.equal(value, payload["shared_state"][name]) for name, value in shared.items())
            or state_hash(shared) != payload["shared_state_hash"]):
        raise ValueError("Stage 2 full model transferable state mismatch")
    stage1_state = {
        name.removeprefix("backbone."): value for name, value in state.items()
        if name.startswith("backbone.") and not any(
            name == "backbone." + prefix or name.startswith("backbone." + prefix + ".")
            for prefix in RECONSTRUCTION_MODULES
        )
    }
    hashes = payload["encoder_state_hashes"]
    if (tensor_state_hash("stage1.encoding-state", stage1_state) != hashes["stage1"]
            or set(stage1_state) != set(payload["stage1_backbone"])
            or any(not torch.equal(value, payload["stage1_backbone"][name]) for name, value in stage1_state.items())):
        raise ValueError("Stage2 frozen Stage1 state mismatch")
    if {k:v for k,v in payload["stage1_encoding_contract"].items() if k != "feature_generation_contract"} != model.model_contract:
        raise ValueError("Stage2 entity input contract mismatch")
    model.eval()
    return payload, model, vocabulary
