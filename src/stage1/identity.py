from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any, Mapping

from rdkit import rdBase

from common.identity import (
    semantic_identity,
    tensor_state_hash,
    validate_semantic_identity,
)
from .config import PretrainConfig
from .tokenizer import tokenizer_backend_version


STAGE1_FEATURE_CONTRACT_VERSION = 1
STAGE1_ENCODING_CONTRACT_VERSION = 1
STAGE1_SAMPLER_LAYOUT_CONTRACT_VERSION = 1

_RECONSTRUCTION_MODULES = (
    "smiles_head",
    "atom_trunk",
    "bond_trunk",
    "atom_heads",
    "bond_heads",
    "descriptor_heads",
    "descriptor_decoder",
    "fingerprint_heads",
)


def _source_identity(source_audit: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        identity = source_audit["semantic"]["identities"]["source"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Stage 1 source audit predates identity contract v1; regenerate it"
        ) from error
    validate_semantic_identity(identity)
    source_ids = set(identity["payload"].get("sources", {}))
    locator_ids = set(source_audit.get("locator", {}).get("files", {}))
    integrity_ids = set(source_audit.get("integrity", {}).get("files", {}))
    if not source_ids or source_ids != locator_ids or source_ids != integrity_ids:
        raise ValueError("Stage 1 source audit logical file set does not match")
    return identity


def feature_generation_contract(config: PretrainConfig) -> dict[str, Any]:
    contract = {
        "contract_version": STAGE1_FEATURE_CONTRACT_VERSION,
        "rdkit_version": rdBase.rdkitVersion,
        "rdkit_descriptor_names_contract": "rdkit-runtime-order-v1",
        "tokenizer_backend": config.tokenizer.backend,
        "tokenizer_backend_version": tokenizer_backend_version(
            config.tokenizer.backend
        ),
        "atom_in_smiles_version": importlib.metadata.version("atomInSmiles"),
        "graph_contract": "stage1-rdkit-graph-v1",
    }
    if config.is_global_rdkit:
        contract["architecture"] = config.architecture.kind
    return contract


def build_stage1_corpus_identity(
    config: PretrainConfig, source_audit: Mapping[str, Any]
) -> dict[str, Any]:
    raw = config.to_dict()
    data = dict(raw["data"])
    for name in (
        "stage1_dir",
        "artifacts_dir",
        "shard_size",
        "shard_cache_size",
    ):
        data.pop(name, None)
    payload = {
        "source_identity": _source_identity(source_audit)["hash"],
        "data": data,
        "tokenizer": raw["tokenizer"],
        "descriptor": raw["descriptor"],
        "feature_generation_contract": feature_generation_contract(config),
        "corpus_kind": "ilume_stage1_corpus",
        "corpus_format_version": 3 if config.is_global_rdkit else 2,
    }
    if not config.is_global_rdkit:
        payload["fingerprint"] = raw["fingerprint"]
    return semantic_identity(
        "stage1.corpus",
        payload,
    )


def build_stage1_sampler_layout_identity(
    shards: list[dict[str, Any]], *, shard_size: int
) -> dict[str, Any]:
    return semantic_identity(
        "stage1.sampler-layout",
        {
            "contract_version": STAGE1_SAMPLER_LAYOUT_CONTRACT_VERSION,
            "configured_shard_size": shard_size,
            "ordered_shards": [
                {"split": item["split"], "count": int(item["count"])}
                for item in shards
            ],
        },
    )


def build_stage1_feature_identity(
    artifact_dir: str | Path, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(artifact_dir)
    payloads = {
        name: json.loads((root / name).read_text(encoding="utf-8"))
        for name in (
            "tokenizer.json",
            "descriptor_schema.json",
            "descriptor_scaler.json",
        )
    }
    generation = metadata.get("feature_generation_contract")
    if not isinstance(generation, Mapping):
        raise ValueError(
            "Stage 1 corpus predates identity contract v1; regenerate the corpus"
        )
    payload = {
        "contract_version": STAGE1_FEATURE_CONTRACT_VERSION,
        "generation": dict(generation),
        "tokenizer": payloads["tokenizer.json"],
        "descriptor_schema": payloads["descriptor_schema.json"],
        "descriptor_scaler": payloads["descriptor_scaler.json"],
        "max_smiles_tokens": int(metadata["max_smiles_tokens"]),
    }
    if generation.get("architecture") != "global_rdkit_v2":
        payload["fingerprint"] = metadata["fingerprint_contract"]
    return semantic_identity(
        "stage1.feature-artifact",
        payload,
    )


def build_stage1_training_identity(
    config: PretrainConfig,
    corpus_identity: Mapping[str, Any],
    sampler_layout_identity: Mapping[str, Any],
) -> dict[str, Any]:
    raw = config.to_dict()
    training = dict(raw["training"])
    for name in (
        "num_workers",
        "device",
        "compile",
        "validation_interval_steps",
        "quick_validation_samples_per_role",
    ):
        training.pop(name, None)
    return semantic_identity(
        "stage1.training",
        {
            "corpus_identity": corpus_identity["hash"],
            "sampler_layout_identity": sampler_layout_identity["hash"],
            "model": raw["model"],
            "masking": raw["masking"],
            "loss": raw["loss"],
            "training": training,
            "seed": config.data.seed,
            "optimizer": "AdamW",
            "scheduler": "linear-warmup-cosine-v1",
        },
    )


def resolve_stage1_training_identity(config: PretrainConfig) -> dict[str, Any]:
    metadata_path = config.data.artifacts_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    corpus = metadata_identity(metadata, "corpus", context="Stage 1 corpus")
    layout = metadata_identity(
        metadata, "sampler_layout", context="Stage 1 corpus"
    )
    return build_stage1_training_identity(config, corpus, layout)


def metadata_identity(
    metadata: Mapping[str, Any], name: str, *, context: str
) -> Mapping[str, Any]:
    try:
        identity = metadata["semantic"]["identities"][name]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"{context} predates identity contract v1; regenerate it"
        ) from error
    if not isinstance(identity, Mapping):
        raise ValueError(f"Malformed {context} {name} identity")
    return identity


def validate_feature_generation_runtime(metadata: Mapping[str, Any]) -> None:
    contract = metadata.get("feature_generation_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(
            "Stage 1 feature artifact predates identity contract v1; regenerate it"
        )
    expected = {
        "rdkit_version": rdBase.rdkitVersion,
        "tokenizer_backend_version": tokenizer_backend_version(
            str(contract["tokenizer_backend"])
        ),
        "atom_in_smiles_version": importlib.metadata.version("atomInSmiles"),
    }
    for name, value in expected.items():
        if contract.get(name) != value:
            raise ValueError(
                f"Stage 1 feature-generation contract mismatch: {name}"
            )


def encoding_state_hash(model: Any) -> str:
    state = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if not any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in _RECONSTRUCTION_MODULES
        )
    }
    return tensor_state_hash("stage1.encoding-state", state)


def build_stage1_encoder_identity(
    *,
    model: Any,
    config: PretrainConfig,
    feature_identity: Mapping[str, Any],
) -> dict[str, Any]:
    raw = config.to_dict()
    model_config = raw["model"]
    payload = {
            "contract_version": STAGE1_ENCODING_CONTRACT_VERSION,
            "feature_identity": feature_identity["hash"],
            "encoding_state_hash": encoding_state_hash(model),
            "encoding_api": "encode-states-v1",
            "role_to_id": {"cation": 0, "anion": 1, "neutral": 2},
            "model": {
                name: model_config[name]
                for name in (
                    "d_model",
                    "n_heads",
                    "smiles_layers",
                    "graph_depth",
                    "descriptor_hidden_dim",
                    "descriptor_blocks",
                    "fusion_layers",
                    "feedforward_dim",
                    "dropout",
                    "role_embedding",
                    "gradient_checkpointing",
                )
            },
        }
    if config.is_global_rdkit:
        payload["contract_version"] = 2
        payload["encoding_api"] = "encode-entity-v2"
        payload["representation"] = {
            "kind": model.representation_kind,
            "token_dim": model.token_dim,
            "atom_dim": model.atom_dim,
            "entity_dim": model.entity_dim,
        }
    return semantic_identity("stage1.encoder", payload)


__all__ = [
    "STAGE1_ENCODING_CONTRACT_VERSION",
    "STAGE1_FEATURE_CONTRACT_VERSION",
    "build_stage1_corpus_identity",
    "build_stage1_encoder_identity",
    "build_stage1_feature_identity",
    "build_stage1_sampler_layout_identity",
    "build_stage1_training_identity",
    "encoding_state_hash",
    "feature_generation_contract",
    "metadata_identity",
    "resolve_stage1_training_identity",
    "validate_feature_generation_runtime",
]
