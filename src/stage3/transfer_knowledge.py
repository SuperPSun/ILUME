"""Read-only, identity-bound Stage 2 transfer knowledge for Stage 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from common.identity import semantic_identity, tensor_state_hash
from common.io import sha256_file
from common.training import canonical_json_sha256


KNOWLEDGE_KIND = "ilume_stage3_transfer_knowledge_bank"
KNOWLEDGE_VERSION = 1
SOURCES = (
    "simulation/density", "simulation/heat_capacity",
    "simulation/heat_of_vaporization", "simulation/homo", "simulation/lumo",
    "simulation/partial_atomic_charge", "simulation/simulated_qm_elec_hf",
    "simulation/thermal_expansion", "simulation/transfer_organic",
)


class TransferKnowledgeBank:
    def __init__(
        self, path: str | Path, *, prepared_identity: str,
        objects: Mapping[str, Any], d_model: int,
    ) -> None:
        path = Path(path)
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if manifest.get("artifact_sha256") != sha256_file(path):
            raise ValueError("Transfer knowledge bank artifact SHA mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if any(payload.get(key) != manifest.get(key) for key in (
            "kind", "format_version", "identity", "prepared_identity",
            "object_list_hash", "tensor_hash", "source_artifacts",
        )):
            raise ValueError("Transfer knowledge bank manifest mismatch")
        if payload.get("kind") != KNOWLEDGE_KIND or payload.get("format_version") != KNOWLEDGE_VERSION:
            raise ValueError("Unsupported transfer knowledge bank")
        if payload["prepared_identity"] != prepared_identity:
            raise ValueError("Transfer knowledge prepared identity mismatch")
        object_hash = canonical_json_sha256(objects["objects"])
        if payload["object_list_hash"] != object_hash:
            raise ValueError("Transfer knowledge ObjectKey order mismatch")
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, dict) or set(embeddings) != {"baseline", *SOURCES}:
            raise ValueError("Transfer knowledge variants are incomplete")
        if set(payload.get("source_artifacts", {})) != {"baseline", *SOURCES}:
            raise ValueError("Transfer knowledge source provenance is incomplete")
        shape = (len(objects["objects"]), d_model)
        if any(
            not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or tuple(value.shape) != shape or not torch.isfinite(value).all()
            for value in embeddings.values()
        ):
            raise ValueError("Transfer knowledge tensors are malformed")
        if payload["tensor_hash"] != tensor_state_hash(
            "stage3.transfer-knowledge-embeddings.v1", embeddings
        ):
            raise ValueError("Transfer knowledge tensor hash mismatch")
        expected_identity = semantic_identity("stage3.transfer-knowledge", {
            "contract_version": KNOWLEDGE_VERSION,
            "prepared_identity": prepared_identity,
            "object_list_hash": object_hash,
            "source_artifacts": payload["source_artifacts"],
            "tensor_hash": payload["tensor_hash"],
            "shape": list(shape),
        })
        if payload["identity"] != expected_identity:
            raise ValueError("Transfer knowledge semantic identity mismatch")
        self.embeddings = embeddings
        self.identity = payload["identity"]
        self.manifest = manifest

    def deltas(self, object_ids: torch.Tensor, sources: tuple[str, ...]) -> dict[str, torch.Tensor]:
        ids = object_ids.cpu().long()
        if bool((ids < 0).any()) or bool((ids >= len(self.embeddings["baseline"])).any()):
            raise ValueError("Transfer knowledge ObjectKey ID out of range")
        baseline = self.embeddings["baseline"][ids]
        return {source: self.embeddings[source][ids] - baseline for source in sources}
