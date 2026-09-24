from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from common.identity import require_compatible_identity, tensor_state_hash
from common.io import sha256_file
from common.training import canonical_json_sha256
from stage3.config import Stage3Config
from stage3.evaluate import evaluate_checkpoints
from stage3.identity import build_stage3_training_identity, metadata_identity
from stage3.prepare import load_prepared_stage3

from .representation import FinetuneRecipe, encoder_state_hashes, load_features
from .train import build_model_and_store


FINAL_KIND = "ilume_stage3_encoder_finetune_three_phase_final"


def evaluate_finetuned(
    config: Stage3Config, recipe: FinetuneRecipe, *,
    feature_dir: str | Path, checkpoint_dir: str | Path,
    split: str, fold: int | None = None,
    predictions_dir: str | Path | None = None,
    task_subset: Sequence[str] | None = None,
    historical_base_root: str | Path | None = None,
) -> dict[str, Any]:
    prepared = load_prepared_stage3(config)
    features = load_features(config, feature_dir, prepared)

    def load_model(
        requested_config: Stage3Config, requested_prepared: Mapping[str, Any],
        path: Path, current_fold: int, epoch: int, device: torch.device, *,
        taskwise_refined: bool, three_phase_final: bool,
    ) -> tuple[Any, dict[str, Any], Any]:
        if requested_config != config or not three_phase_final or taskwise_refined:
            raise ValueError("Fine-tuning evaluator requires its three-phase final artifact")
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        plan = artifact.get("resolved_training_plan")
        if (
            artifact.get("kind") != FINAL_KIND
            or artifact.get("format_version") != 1
            or artifact.get("fold") != current_fold
            or manifest.get("kind") != FINAL_KIND
            or manifest.get("artifact_sha256") != sha256_file(path)
            or not isinstance(plan, dict)
            or plan.get("format_version") != 6
            or plan.get("encoder_finetune", {}).get("recipe") != recipe.to_dict()
            or plan.get("encoder_finetune", {}).get("feature_artifact_sha256") != features["artifact_sha256"]
            or plan.get("encoder_finetune", {}).get("source_encoder_sha256") != features["stage2_encoder_sha256"]
            or plan.get("prepared_identity") != metadata_identity(requested_prepared["metadata"], "prepared", context="Fine-tuning evaluation")["hash"]
            or plan.get("normalization_hash") != canonical_json_sha256(artifact.get("normalization"))
        ):
            raise ValueError("Fine-tuned final artifact is incompatible or corrupt")
        require_compatible_identity(
            build_stage3_training_identity(plan), artifact.get("training_identity", {}),
            context="Fine-tuning evaluation training identity",
        )
        model, store = build_model_and_store(
            config, requested_prepared, features, fold=current_fold, device=device
        )
        if artifact.get("ownership_manifest") != model.ownership_manifest():
            raise ValueError("Fine-tuned final ownership mismatch")
        if artifact.get("model_state_hash") != tensor_state_hash(
            "stage3.three-phase-model-state", artifact["model"]
        ):
            raise ValueError("Fine-tuned final model state hash mismatch")
        model.load_state_dict(artifact["model"], strict=True)
        if (
            artifact.get("encoder_state_hashes") != encoder_state_hashes(model)
            or manifest.get("encoder_state_hashes") != artifact["encoder_state_hashes"]
        ):
            raise ValueError("Fine-tuned final encoder state hash mismatch")
        store.freeze_after_phase1(model, artifact["phase1_model_state_hash"])
        return model.eval(), artifact, store

    result = evaluate_checkpoints(
        config, checkpoint_dir, split=split, ensemble_folds=split == "test",
        fold=fold, task_subset=task_subset,
        predictions_dir=predictions_dir,
        reporting_study_id="ilume-stage3-full-finetune-v1",
        model_loader=load_model,
    )
    result["ablation"] = "stage3_full_finetune"
    result["source_encoder_sha256"] = features["stage2_encoder_sha256"]
    if historical_base_root is not None:
        result["historical_base_comparison"] = compare_historical_base(
            result, Path(historical_base_root), split=split, fold=fold,
            training_tasks=tuple(task for task, item in config.tasks.items() if item.enabled),
            prepared_identity=metadata_identity(
                prepared["metadata"], "prepared", context="Fine-tuning comparison"
            )["hash"],
        )
    return result


def compare_historical_base(
    result: Mapping[str, Any], root: Path, *, split: str, fold: int | None,
    training_tasks: Sequence[str], prepared_identity: str,
) -> dict[str, Any]:
    path = (
        root / "evaluate_valid" / f"fold{fold}" / "summary.json"
        if split == "valid" else root / "evaluate_test" / "summary.json"
    )
    baseline = json.loads(path.read_text(encoding="utf-8"))
    if (
        baseline.get("split") != split
        or baseline.get("model_selector") != "three_phase_final"
        or baseline.get("reporting", {}).get("comparison_identity", {}).get("hash")
        != result["reporting"]["comparison_identity"]["hash"]
    ):
        raise ValueError("Historical Base summary does not share the evaluation data contract")
    manifest_folds = (fold,) if fold is not None else range(1, 6)
    for current_fold in manifest_folds:
        manifest = json.loads(
            (root / "train" / f"fold{current_fold}" / "three_phase_final.json").read_text(encoding="utf-8")
        )
        plan = manifest.get("training_identity", {}).get("payload", {}).get("plan", {})
        if (
            manifest.get("training_identity", {}).get("payload", {}).get("contract_version") != 6
            or plan.get("math", {}).get("gradient_aggregation") != "weighted_owner_raw_v1"
            or set(plan.get("active_tasks", ())) != set(training_tasks)
            or plan.get("prepared_identity") != prepared_identity
            or not set(result["reporting"]["protocol"]["expected_tasks"]).issubset(
                plan.get("active_tasks", ())
            )
        ):
            raise ValueError("Historical Base training contract is not the current 20-task Base")

    def scope_delta(after: Mapping[str, Any], before: Mapping[str, Any]) -> dict[str, Any]:
        if set(after["tasks"]) != set(before["tasks"]):
            raise ValueError("Historical Base task set differs from fine-tuning")
        before_macro = before["macro_task_equal"]["normalized_mae"]["value"]
        after_macro = after["macro_task_equal"]["normalized_mae"]["value"]
        return {
            "macro_normalized_mae": {
                "base": before_macro,
                "finetuned": after_macro,
                "delta": after_macro - before_macro,
            },
            "tasks": {
                task: {
                    "base_normalized_mae": before["tasks"][task]["normalized_mae"],
                    "finetuned_normalized_mae": after["tasks"][task]["normalized_mae"],
                    "delta_normalized_mae": after["tasks"][task]["normalized_mae"] - before["tasks"][task]["normalized_mae"],
                }
                for task in after["tasks"]
            },
        }

    comparison = (
        scope_delta(result, baseline)
        if split == "valid" else {
            "folds": {
                name: scope_delta(result["folds"][name], baseline["folds"][name])
                for name in result["folds"]
            },
            "ensemble": scope_delta(result["ensemble"], baseline["ensemble"]),
        }
    )
    return {
        "source_summary": str(path),
        "interpretation": "Historical comparison; runtime representation path is not paired-controlled",
        **comparison,
    }
