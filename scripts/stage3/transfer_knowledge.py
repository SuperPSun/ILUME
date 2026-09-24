"""Materialize the isolated Stage 2 transfer knowledge bank."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ablations.stage2_stage3_transfer.stage3 import (  # noqa: E402
    encode_transfer_objects, load_representation_bank,
)
from common.identity import semantic_identity, tensor_state_hash  # noqa: E402
from common.io import atomic_json, atomic_torch_save, sha256_file  # noqa: E402
from common.training import canonical_json_sha256, resolve_device  # noqa: E402
from stage2.model import ObjectEncoder  # noqa: E402
from stage3.config import load_stage3_config  # noqa: E402
from stage3.data import sanitize_task  # noqa: E402
from stage3.identity import metadata_identity  # noqa: E402
from stage3.prepare import load_prepared_stage3  # noqa: E402
from stage3.transfer_knowledge import KNOWLEDGE_KIND, KNOWLEDGE_VERSION, SOURCES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage 3 transfer knowledge embeddings.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--representations", required=True)
    parser.add_argument("--stage2-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    config = load_stage3_config(args.config)
    if config.transfer_knowledge is None:
        parser.error("config.transfer_knowledge is required")
    destination = config.transfer_knowledge.bank
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError(f"Transfer knowledge bank already exists: {destination}")
    prepared = load_prepared_stage3(config)
    prepared_identity = metadata_identity(
        prepared["metadata"], "prepared", context="Stage 3 transfer knowledge"
    )
    objects = prepared["objects"]["objects"]
    object_hash = canonical_json_sha256(objects)
    device = resolve_device(args.device)
    representation_root = Path(args.representations)
    embeddings: dict[str, torch.Tensor] = {}
    source_artifacts: dict[str, dict[str, str]] = {}
    initial_shared_hash: str | None = None
    for source in ("baseline", *SOURCES):
        path = (
            representation_root / "baseline.pt" if source == "baseline"
            else representation_root / "sources" / f"{sanitize_task(source)}.pt"
        )
        bank = load_representation_bank(path, expected_prepared_identity=prepared_identity)
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if bank.get("variant") != source or bank.get("object_list_hash") != object_hash or bank.get("objects") != objects:
            raise ValueError(f"Transfer representation variant/ObjectKey mismatch: {source}")
        if manifest.get("variant") != source or manifest.get("identity") != bank.get("identity"):
            raise ValueError(f"Transfer representation manifest mismatch: {source}")
        # The source Stage 2 manifests bind every variant to one common initialization.
        stage2_root = Path(args.stage2_root)
        encoder_manifest = (
            stage2_root / "baseline" / "manifest.json"
            if source == "baseline" else
            stage2_root / "sources" / sanitize_task(source) / "manifest.json"
        )
        if not encoder_manifest.is_file():
            raise FileNotFoundError(f"Missing transfer encoder manifest: {encoder_manifest}")
        encoder_record = json.loads(encoder_manifest.read_text(encoding="utf-8"))
        if source == "baseline":
            if encoder_record.get("variant") != "baseline" or encoder_record.get("source_task") is not None or encoder_record.get("optimizer_updates") != 0:
                raise ValueError("Transfer baseline is not the zero-update variant")
        elif (
            encoder_record.get("variant") != "source"
            or encoder_record.get("source_task") != source
            or encoder_record.get("physics_only") is not True
            or int(encoder_record.get("optimizer_updates", 0)) <= 0
        ):
            raise ValueError(f"Transfer source training contract mismatch: {source}")
        shared_hash = encoder_record.get("initial_shared_state_hash")
        if not isinstance(shared_hash, str) or not shared_hash:
            raise ValueError("Transfer encoder lacks initial shared-state hash")
        if initial_shared_hash is None:
            initial_shared_hash = shared_hash
        elif shared_hash != initial_shared_hash:
            raise ValueError("Transfer variants do not share the same initialization")
        if encoder_record.get("encoder_sha256") != bank["identity"]["payload"]["encoder_artifact_sha256"]:
            raise ValueError(f"Transfer encoder/representation SHA mismatch: {source}")
        contract = bank["object_encoder_contract"]
        encoder = ObjectEncoder(
            int(contract["d_model"]), int(contract["n_heads"]),
            num_layers=int(contract["layers"]),
            feedforward_dim=int(contract["ffn_dim"]),
            dropout=float(contract["dropout"]),
        ).to(device)
        encoder.load_state_dict(bank["object_encoder_state"], strict=True)
        encoder.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(objects), args.batch_size):
                ids = torch.arange(start, min(start + args.batch_size, len(objects)))
                chunks.append(encode_transfer_objects(encoder, bank, ids, device=device).float().cpu())
        value = torch.cat(chunks).contiguous()
        if tuple(value.shape) != (len(objects), 1024) or not torch.isfinite(value).all():
            raise ValueError(f"Malformed transfer encoder output: {source}")
        embeddings[source] = value
        source_artifacts[source] = {
            "identity": bank["identity"]["hash"],
            "sha256": sha256_file(path),
            "encoder_identity": bank["stage2_encoder_identity"]["hash"],
            "initial_shared_state_hash": shared_hash,
        }
    tensor_hash = tensor_state_hash("stage3.transfer-knowledge-embeddings.v1", embeddings)
    identity = semantic_identity("stage3.transfer-knowledge", {
        "contract_version": KNOWLEDGE_VERSION,
        "prepared_identity": prepared_identity["hash"],
        "object_list_hash": object_hash,
        "source_artifacts": source_artifacts,
        "tensor_hash": tensor_hash,
        "shape": [len(objects), 1024],
    })
    payload = {
        "kind": KNOWLEDGE_KIND, "format_version": KNOWLEDGE_VERSION,
        "identity": identity, "prepared_identity": prepared_identity["hash"],
        "object_list_hash": object_hash, "source_artifacts": source_artifacts,
        "tensor_hash": tensor_hash, "embeddings": embeddings,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(destination, payload)
    atomic_json(destination.with_suffix(".json"), {
        key: value for key, value in payload.items() if key != "embeddings"
    } | {"artifact": destination.name, "artifact_sha256": sha256_file(destination)})


if __name__ == "__main__":
    main()
