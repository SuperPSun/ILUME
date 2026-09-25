from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.identity import semantic_identity, tensor_state_hash
from common.io import atomic_json, sha256_file
from common.training import seed_everything
from stage1.model import load_stage1_model
from stage2.config import load_stage2_config
from stage2.data import load_artifact_registry
from stage2.identity import metadata_identity
from stage2.model import Stage2ObjectModel
from stage2.train import export_stage2_encoder_artifact, load_stage2_encoder_artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the Stage 2 initialized encoder with zero optimizer updates."
    )
    parser.add_argument("--config", required=True, help="Formal Stage 2 v2 configuration")
    parser.add_argument("--trained-encoder", required=True, help="Paired formal Stage 2 final encoder")
    parser.add_argument("--output", required=True, help="New isolated artifact directory")
    args = parser.parse_args()
    config = load_stage2_config(args.config)
    if config.representation is not None:
        raise ValueError("Zero-update control requires Object-backed Stage 2")
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(f"Zero-update artifact already exists: {root}")
    trained = load_stage2_encoder_artifact(args.trained_encoder)
    if (
        trained["provenance"].get("stage2_checkpoint_hash") is None
        or trained["provenance"].get("refinement_boundary_epoch") != 10
        or trained["provenance"].get("stage1_checkpoint_hash") != sha256_file(config.initialization.checkpoint)
        or trained["object_encoder_config"] != {
            "layers": config.model.object_layers,
            "ffn_dim": config.model.object_ffn_dim,
            "dropout": config.model.dropout,
        }
    ):
        raise ValueError("Formal Stage 2 encoder is not the matching trained control")
    seed_everything(config.data.seed)
    loaded = load_stage1_model(
        config.initialization.checkpoint,
        config.data.pretrain_artifacts_dir,
        device="cpu",
        backbone_dropout=0.0,
    )
    registry = load_artifact_registry(config.data.artifacts_dir)
    config.validate_registry(registry)
    model = Stage2ObjectModel(
        loaded.model, registry,
        object_layers=config.model.object_layers,
        object_ffn_dim=config.model.object_ffn_dim,
        dropout=config.model.dropout,
    )
    if model.model_contract != trained["model_contract"]:
        raise ValueError("Zero-update encoder differs from formal Stage 2 architecture")
    initial_state = {
        **{f"backbone.{key}": value for key, value in model.backbone.state_dict().items()},
        **{f"object_encoder.{key}": value for key, value in model.object_encoder.state_dict().items()},
    }
    initial_hash = tensor_state_hash("stage2.transfer-initial-shared-state.v1", initial_state)
    metadata = json.loads((config.data.artifacts_dir / "metadata.json").read_text(encoding="utf-8"))
    data_identity = dict(metadata_identity(metadata, "data", context="Stage 2 prepared data"))
    if trained["provenance"].get("stage2_data_identity") != data_identity["hash"]:
        raise ValueError("Formal Stage 2 encoder used different prepared data")
    control_identity = semantic_identity("stage2.zero-update-control", {
        "contract_version": 1,
        "stage1_checkpoint_sha256": sha256_file(config.initialization.checkpoint),
        "stage2_prepared_identity": data_identity["hash"],
        "initialization_seed": config.data.seed,
        "initial_shared_state_hash": initial_hash,
        "optimizer_updates": 0,
    })
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stage2-zero-update-", dir=root.parent) as staging_name:
        staging = Path(staging_name)
        encoder_path = staging / "stage2_encoder.pt"
        export_stage2_encoder_artifact(
            encoder_path, model=model, config=config, registry=registry,
            data_identity=data_identity,
            provenance={
                "zero_stage2_training": True,
                "optimizer_updates": 0,
                "initialization_seed": config.data.seed,
                "initial_shared_state_hash": initial_hash,
                "control_identity": control_identity["hash"],
                "paired_trained_encoder_sha256": sha256_file(args.trained_encoder),
            },
        )
        atomic_json(staging / "manifest.json", {
            "kind": "ilume_stage2_zero_update_control",
            "identity": control_identity,
            "optimizer_updates": 0,
            "initial_shared_state_hash": initial_hash,
            "stage1_checkpoint_sha256": sha256_file(config.initialization.checkpoint),
            "stage2_encoder_sha256": sha256_file(encoder_path),
            "paired_trained_encoder_sha256": sha256_file(args.trained_encoder),
        })
        staging.replace(root)


if __name__ == "__main__":
    main()
