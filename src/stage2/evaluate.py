from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from rdkit import Chem

from common.progress import ProgressReporter
from common.identity import (
    IDENTITY_CONTRACT_VERSION,
    require_compatible_identity,
    semantic_identity,
    tensor_state_hash,
)
from common.io import sha256_file
from common.reporting import (
    STAGE2_BENCHMARK_SUITE_CONTRACT,
    comparison_identity,
    role_mae_diagnostics,
    sanitize_task_id,
    stage2_full_comparison_identity,
    write_prediction_csv,
)
from common.training import resolve_device
from stage1.descriptors import calculate_descriptors, rdkit_descriptor_names
from stage1.features import (
    ROLE_TO_ID,
    build_entity_sample,
    inspect_entity_qc,
    load_stage1_feature_inputs,
)
from stage1.masking import MultimodalPacker
from stage1.model import load_stage1_model
from .config import STAGE2_CHECKPOINT_KIND, STAGE2_CHECKPOINT_VERSION, Stage2Config
from .data import load_artifact_registry
from .identity import metadata_identity
from .model import Stage2ObjectModel
from .train import STAGE2_REFINED_KIND
from .registry import (
    ORBITAL_TASK_TARGETS,
    orbital_audit_columns,
    validate_orbital_audit_row,
)
from .atom_evaluation import (
    PARTIAL_CHARGE_TASK,
    PARTIAL_CHARGE_UNIT,
    PartialChargeBenchmark,
    build_partial_charge_benchmark,
    public_partial_charge_score,
    score_partial_charge_predictions,
    write_partial_charge_predictions,
)


STAGE2_CORE_TASKS = (
    "simulation/heat_of_vaporization",
    "simulation/homo",
    "simulation/lumo",
)


def _checkpoint_epoch(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint_epoch_(\d{5})\.pt", path.name)
    return int(match.group(1)) if match else None


def resolve_checkpoint_path(
    checkpoint_dir: str | Path, checkpoint_epoch: int | None = None
) -> Path:
    root = Path(checkpoint_dir)
    if checkpoint_epoch is not None:
        if checkpoint_epoch <= 0:
            raise ValueError("Stage 2 checkpoint epoch must be positive")
        path = root / f"checkpoint_epoch_{checkpoint_epoch:05d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Stage 2 checkpoint: {path}")
        return path
    candidates = [
        (epoch, path)
        for path in root.glob("checkpoint_epoch_*.pt")
        if (epoch := _checkpoint_epoch(path)) is not None
    ]
    if not candidates:
        raise FileNotFoundError(f"No Stage 2 epoch checkpoint under {root}")
    return max(candidates)[1]


def _load_checkpoint_contract(
    config: Stage2Config, checkpoint_path: Path
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("kind") != STAGE2_CHECKPOINT_KIND
        or checkpoint.get("format_version") != STAGE2_CHECKPOINT_VERSION
        or checkpoint.get("identity_contract_version") != IDENTITY_CONTRACT_VERSION
    ):
        raise ValueError("Unsupported Stage 2 evaluation checkpoint")
    epoch = _checkpoint_epoch(checkpoint_path)
    if epoch is None or checkpoint.get("completed_epoch") != epoch:
        raise ValueError("Stage 2 checkpoint filename/completed epoch mismatch")
    registry = load_artifact_registry(config.data.artifacts_dir)
    if (
        checkpoint.get("registry") != registry.snapshot()
        or checkpoint.get("registry_hash") != registry.registry_hash
        or checkpoint.get("catalog_sha256") != registry.catalog_sha256
    ):
        raise ValueError("Stage 2 evaluation checkpoint registry mismatch")
    metadata = json.loads(
        (config.data.artifacts_dir / "metadata.json").read_text(encoding="utf-8")
    )
    stored_data = checkpoint.get("data_identity")
    if not isinstance(stored_data, Mapping):
        raise ValueError("Stage 2 checkpoint lacks its prepared-data identity")
    require_compatible_identity(
        metadata_identity(metadata, "data", context="Stage 2 prepared artifact"),
        stored_data,
        context="Stage 2 evaluation prepared-data identity",
    )
    training_identity = checkpoint.get("training_identity")
    if not isinstance(training_identity, Mapping):
        raise ValueError("Stage 2 checkpoint lacks its training identity")
    return checkpoint, registry, metadata


def _scalers(config: Stage2Config) -> dict[str, Any]:
    path = config.data.artifacts_dir / "scalers.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Malformed Stage 2 scaler artifact")
    return payload


def _core_units(registry: Any) -> tuple[str, ...]:
    for task in STAGE2_CORE_TASKS:
        if len(registry.by_id(task).target_columns) != 1:
            raise ValueError(f"Stage 2 Core task must be scalar: {task}")
    return STAGE2_CORE_TASKS


def _comparison(
    config: Stage2Config, registry: Any, scalers: Mapping[str, Any]
) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    normalization: dict[str, Any] = {}
    for task in STAGE2_CORE_TASKS:
        spec = registry.by_id(task)
        for split in ("train", "test"):
            path = spec.dataset.split_path(config.data.data_root, split)
            sources[f"{task}:{split}"] = sha256_file(path)
        target = spec.target_columns[0]
        stats = scalers[task]["targets"][target]
        normalization[task] = {"scale": float(stats["scale"])}
    return comparison_identity(
        "stage2_physics",
        split="test",
        expected=_core_units(registry),
        sources=sources,
        normalization=normalization,
    )


def _partial_benchmark(
    config: Stage2Config, registry: Any, scalers: Mapping[str, Any]
) -> PartialChargeBenchmark:
    spec = registry.by_id(PARTIAL_CHARGE_TASK)
    manifest = spec.dataset.resource_manifest_path(config.data.data_root)
    if manifest is None:
        raise ValueError("Partial-charge task is missing its structure manifest")
    return build_partial_charge_benchmark(
        spec.dataset.split_path(config.data.data_root, "test"),
        manifest,
        scalers[PARTIAL_CHARGE_TASK]["targets"][spec.target_columns[0]],
    )


def _full_comparison(
    registry: Any,
    core: Mapping[str, Any],
    partial_charge: Mapping[str, Any],
) -> dict[str, Any]:
    return stage2_full_comparison_identity(
        core,
        partial_charge,
        ordered_units=(*_core_units(registry), PARTIAL_CHARGE_UNIT),
    )


def resolve_stage2_evaluation_identity(
    config: Stage2Config,
    checkpoint_dir: str | Path,
    *,
    checkpoint_epoch: int | None = None,
    taskwise_refined: bool = False,
) -> dict[str, Any]:
    identity, _ = resolve_stage2_evaluation_contract(
        config, checkpoint_dir, checkpoint_epoch=checkpoint_epoch,
        taskwise_refined=taskwise_refined,
    )
    return identity


def _load_refined_artifact(
    checkpoint_dir: str | Path,
    checkpoint: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], str]:
    root = Path(checkpoint_dir)
    artifact_path = root / "taskwise_refined.pt"
    manifest_path = root / "taskwise_refinement.json"
    if not artifact_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Missing Stage 2 task-wise refined artifact or manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Stage 2 refinement manifest is unreadable") from error
    refined = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(manifest, dict)
        or manifest.get("kind") != STAGE2_REFINED_KIND
        or manifest.get("format_version") != 1
        or manifest.get("artifact") != artifact_path.name
        or manifest.get("artifact_sha256") != sha256_file(artifact_path)
    ):
        raise ValueError("Stage 2 refinement manifest/artifact integrity mismatch")
    if refined.get("kind") != STAGE2_REFINED_KIND or refined.get("format_version") != 1:
        raise ValueError("Unsupported Stage 2 refined artifact")
    require_compatible_identity(
        checkpoint["training_identity"],
        refined.get("training_identity", {}),
        context="Stage 2 refined artifact training identity",
    )
    if manifest.get("ordinary_final_epoch") != checkpoint.get("completed_epoch"):
        raise ValueError("Stage 2 refinement manifest final epoch mismatch")
    for key in (
        "boundary_epoch", "shared_state_hash", "private_state_hashes",
        "selected_tasks", "validation",
    ):
        if json.dumps(manifest.get(key), sort_keys=True) != json.dumps(
            refined.get(key), sort_keys=True
        ):
            raise ValueError(f"Stage 2 refinement manifest mismatch: {key}")
    if refined.get("model_state_hash") != tensor_state_hash(
        "stage2.taskwise-refined-state", refined["model"]
    ):
        raise ValueError("Stage 2 refined artifact model state hash mismatch")
    return artifact_path, refined, sha256_file(manifest_path)


def resolve_stage2_evaluation_contract(
    config: Stage2Config,
    checkpoint_dir: str | Path,
    *,
    checkpoint_epoch: int | None = None,
    taskwise_refined: bool = False,
) -> tuple[dict[str, Any], PartialChargeBenchmark]:
    if taskwise_refined and checkpoint_epoch is not None:
        raise ValueError("taskwise_refined forbids checkpoint_epoch")
    path = resolve_checkpoint_path(checkpoint_dir, checkpoint_epoch)
    checkpoint, registry, _ = _load_checkpoint_contract(config, path)
    model_path = path
    model_state_hash: str | None = None
    selection_manifest_hash: str | None = None
    if taskwise_refined:
        model_path, refined, selection_manifest_hash = _load_refined_artifact(
            checkpoint_dir, checkpoint
        )
        model_state_hash = refined.get("model_state_hash")
    scalers = _scalers(config)
    core_comparison = _comparison(config, registry, scalers)
    partial_benchmark = _partial_benchmark(config, registry, scalers)
    full_comparison = _full_comparison(
        registry, core_comparison, partial_benchmark.comparison_identity
    )
    identity = _evaluation_identity(
        model_path, checkpoint, core_comparison,
        partial_benchmark.comparison_identity, full_comparison,
        model_state_hash=model_state_hash,
        selection_manifest_hash=selection_manifest_hash,
        model_selector="taskwise_refined" if taskwise_refined else "epoch_checkpoint",
    )
    return identity, partial_benchmark


def _evaluation_identity(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    core_comparison: Mapping[str, Any],
    partial_comparison: Mapping[str, Any],
    full_comparison: Mapping[str, Any],
    *,
    model_state_hash: str | None = None,
    selection_manifest_hash: str | None = None,
    model_selector: str = "epoch_checkpoint",
) -> dict[str, Any]:
    return semantic_identity(
        "stage2.evaluation.v1",
        {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["completed_epoch"]),
            "model_selector": model_selector,
            "model_state_hash": model_state_hash,
            "selection_manifest_sha256": selection_manifest_hash,
            "training_identity": checkpoint["training_identity"]["hash"],
            "prepared_identity": checkpoint["data_identity"]["hash"],
            "tasks": [*STAGE2_CORE_TASKS, PARTIAL_CHARGE_TASK],
            "comparison_identities": {
                "stage2_core_physics": core_comparison["hash"],
                "stage2_partial_charge": partial_comparison["hash"],
                "stage2_physics_full": full_comparison["hash"],
            },
        },
    )


def _canonical(raw: str, context: str) -> str:
    molecule = Chem.MolFromSmiles((raw or "").strip())
    if molecule is None:
        raise ValueError(f"Invalid Stage 2 test SMILES in {context}: {raw}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _finite(raw: str | None, context: str) -> float:
    try:
        value = float(raw or "")
    except ValueError as error:
        raise ValueError(f"Non-numeric Stage 2 test value in {context}: {raw}") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite Stage 2 test value in {context}: {raw}")
    return value


def _entity_role(canonical: str, policy: str, context: str) -> str:
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:
        raise ValueError(f"Invalid canonical Stage 2 test SMILES: {canonical}")
    charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    inferred = "cation" if charge > 0 else "anion" if charge < 0 else "neutral"
    if policy == "formal_charge":
        return inferred
    if policy != inferred:
        raise ValueError(
            f"Stage 2 test role mismatch in {context}: {policy} != {inferred}"
        )
    return policy


def _read_test_rows(config: Stage2Config, spec: Any) -> list[dict[str, Any]]:
    path = spec.dataset.split_path(config.data.data_root, "test")
    required = (
        *spec.entity_columns,
        *spec.condition_columns,
        *orbital_audit_columns(spec.task_id),
        *spec.target_columns,
        "source_list",
    )
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != required:
            raise ValueError(
                f"Unexpected Stage 2 test columns in {path}: {reader.fieldnames}"
            )
        for source_row, raw in enumerate(reader, start=2):
            canonicals = tuple(
                _canonical(raw[name], f"{spec.task_id}:{source_row}/{name}")
                for name in spec.entity_columns
            )
            roles = tuple(
                _entity_role(
                    canonical,
                    policy,
                    f"{spec.task_id}:{source_row}",
                )
                for canonical, policy in zip(
                    canonicals, spec.role_policy, strict=True
                )
            )
            if roles:
                validate_orbital_audit_row(
                    spec.task_id,
                    raw,
                    inferred_role=roles[0],
                    context=f"{spec.task_id}:{source_row}",
                )
            rows.append(
                {
                    "source_row": source_row,
                    "raw": dict(raw),
                    "canonicals": canonicals,
                    "roles": roles,
                    "conditions": tuple(
                        _finite(raw[name], f"{spec.task_id}:{source_row}/{name}")
                        for name in spec.condition_columns
                    ),
                    "targets": tuple(
                        _finite(raw[name], f"{spec.task_id}:{source_row}/{name}")
                        for name in spec.target_columns
                    ),
                }
            )
    if not rows:
        raise ValueError(f"Stage 2 test split is empty: {spec.task_id}")
    return rows


def _entity_sample(
    role: str,
    canonical_smiles: str,
    *,
    feature_config: Any,
    vocabulary: Any,
    schema: Any,
    standardizer: Any,
) -> dict[str, Any]:
    record = {
        "sample_id": f"stage2-evaluate:{role}:{canonical_smiles}",
        "role": role,
        "role_id": ROLE_TO_ID[role],
        "canonical_smiles": canonical_smiles,
        "sources": ("stage2-evaluate",),
        "split": "test",
        "is_augmented": False,
        "seed_smiles": (),
    }
    qc = inspect_entity_qc(record)
    if vocabulary.token_count(canonical_smiles) > feature_config.data.max_smiles_tokens:
        qc.reasons.append("smiles_overlength")
    if qc.reasons:
        raise ValueError(
            f"Stage 2 test entity fails feature QC: {role}/{canonical_smiles}: "
            + ",".join(qc.reasons)
        )
    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None:
        raise ValueError(f"Invalid canonical Stage 2 test SMILES: {canonical_smiles}")
    raw = calculate_descriptors(molecule, rdkit_descriptor_names())
    return build_entity_sample(
        record, np.asarray(raw), schema, standardizer, vocabulary, feature_config
    )


def _metrics(
    predictions: np.ndarray, targets: np.ndarray, scale: float
) -> dict[str, Any]:
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    actual = np.asarray(targets, dtype=np.float64).reshape(-1)
    if predicted.shape != actual.shape or not len(actual):
        raise ValueError("Stage 2 evaluation metric vectors must be matching and non-empty")
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("Stage 2 evaluation metrics require finite values")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Stage 2 normalized metrics require a positive train scale")
    delta = predicted - actual
    denominator = float(np.square(actual - actual.mean()).sum())
    mae = float(np.abs(delta).mean())
    rmse = float(np.sqrt(np.square(delta).mean()))
    return {
        "count": len(actual),
        "mae": mae,
        "rmse": rmse,
        "r2": (
            float("nan")
            if denominator == 0
            else 1.0 - float(np.square(delta).sum()) / denominator
        ),
        "r2_reason": "constant_target" if denominator == 0 else None,
        "normalized_mae": mae / scale,
        "normalized_rmse": rmse / scale,
    }


@torch.inference_mode()
def evaluate_stage2_checkpoints(
    config: Stage2Config,
    checkpoint_dir: str | Path,
    *,
    checkpoint_epoch: int | None = None,
    predictions_dir: str | Path | None = None,
    expected_evaluation_identity: Mapping[str, Any] | None = None,
    partial_charge_benchmark: PartialChargeBenchmark | None = None,
    reporter: ProgressReporter | None = None,
    taskwise_refined: bool = False,
) -> dict[str, Any]:
    if taskwise_refined and checkpoint_epoch is not None:
        raise ValueError("taskwise_refined forbids checkpoint_epoch")
    checkpoint_path = resolve_checkpoint_path(checkpoint_dir, checkpoint_epoch)
    checkpoint, registry, _ = _load_checkpoint_contract(config, checkpoint_path)
    model_payload = checkpoint
    model_path = checkpoint_path
    model_state_hash: str | None = None
    selection_manifest_hash: str | None = None
    if taskwise_refined:
        model_path, model_payload, selection_manifest_hash = _load_refined_artifact(
            checkpoint_dir, checkpoint
        )
        model_state_hash = model_payload.get("model_state_hash")
    scalers = _scalers(config)
    core_comparison = _comparison(config, registry, scalers)
    partial_benchmark = partial_charge_benchmark or _partial_benchmark(
        config, registry, scalers
    )
    full_comparison = _full_comparison(
        registry, core_comparison, partial_benchmark.comparison_identity
    )
    evaluation_identity = _evaluation_identity(
        model_path, checkpoint, core_comparison,
        partial_benchmark.comparison_identity, full_comparison,
        model_state_hash=model_state_hash,
        selection_manifest_hash=selection_manifest_hash,
        model_selector="taskwise_refined" if taskwise_refined else "epoch_checkpoint",
    )
    if expected_evaluation_identity is not None:
        require_compatible_identity(
            expected_evaluation_identity,
            evaluation_identity,
            context="Stage 2 run-directory evaluation identity",
        )

    device = resolve_device(config.training.device)
    if config.training.amp_dtype == "bf16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("Stage 2 BF16 evaluation requires capable CUDA")
    loaded = load_stage1_model(
        config.initialization.checkpoint,
        config.data.pretrain_artifacts_dir,
        device=device,
        backbone_dropout=0.0,
    )
    model = Stage2ObjectModel(
        loaded.model,
        registry,
        object_layers=config.model.object_layers,
        object_ffn_dim=config.model.object_ffn_dim,
        dropout=config.model.dropout,
    ).to(device)
    if checkpoint.get("model_contract") != model.model_contract:
        raise ValueError("Stage 2 evaluation checkpoint model contract mismatch")
    model.load_state_dict(model_payload["model"], strict=True)
    model.eval()
    feature_config, vocabulary, schema, standardizer, feature_hash = (
        load_stage1_feature_inputs(
            config.initialization.checkpoint, config.data.pretrain_artifacts_dir
        )
    )
    if feature_hash != loaded.artifact_hash:
        raise ValueError("Stage 2 evaluation feature identity mismatch")
    packer = MultimodalPacker(vocabulary)
    sample_cache: dict[tuple[str, str], dict[str, Any]] = {}
    task_metrics: dict[str, Any] = {}
    prediction_manifests: list[dict[str, Any]] = []

    progress_reporter = reporter or ProgressReporter()

    task_rows = {
        task: _read_test_rows(config, registry.by_id(task))
        for task in STAGE2_CORE_TASKS
    }

    total_batches = sum(
        math.ceil(len(rows) / config.training.batch_size)
        for rows in task_rows.values()
    ) + math.ceil(len(partial_benchmark.evaluated) / config.training.batch_size)

    evaluation_progress = progress_reporter.bar(
        total=total_batches,
        desc="Stage 2 test evaluation",
        unit="batch",
    )
    try:
        for task in STAGE2_CORE_TASKS:
            spec = registry.by_id(task)
            rows = task_rows[task]
            raw_predictions: list[np.ndarray] = []

            task_batches = math.ceil(
                len(rows) / config.training.batch_size
            )

            for batch_index, start in enumerate(
                range(0, len(rows), config.training.batch_size),
                start=1,
            ):
                evaluation_progress.set_postfix_str(
                    f"task={task} batch={batch_index}/{task_batches}",
                    refresh=False,
                )

                chunk = rows[
                    start : start + config.training.batch_size
                ]
                samples = []
                role_rows = []
                for row in chunk:
                    roles = []
                    for role, canonical in zip(
                        row["roles"], row["canonicals"], strict=True
                    ):
                        key = (role, canonical)
                        if key not in sample_cache:
                            sample_cache[key] = _entity_sample(
                                role,
                                canonical,
                                feature_config=feature_config,
                                vocabulary=vocabulary,
                                schema=schema,
                                standardizer=standardizer,
                            )
                        samples.append(sample_cache[key])
                        roles.append(ROLE_TO_ID[role])
                    role_rows.append(roles)
                packed = packer(samples).to(device)
                slots = model.encode_entities(packed).reshape(
                    len(chunk), len(spec.entity_columns), -1
                )
                roles = torch.tensor(role_rows, dtype=torch.long, device=device)
                condition_values = []
                for row in chunk:
                    condition_values.append(
                        [
                            (value - float(scalers[task]["conditions"][name]["mean"]))
                            / float(scalers[task]["conditions"][name]["scale"])
                            for name, value in zip(
                                spec.condition_columns, row["conditions"], strict=True
                            )
                        ]
                    )
                conditions = torch.tensor(
                    condition_values, dtype=torch.float32, device=device
                ).reshape(len(chunk), len(spec.condition_columns))
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=config.training.amp_dtype == "bf16",
                ):
                    normalized = model.predict_object(spec, slots, roles, conditions)
                if not torch.isfinite(normalized).all():
                    raise RuntimeError(f"Non-finite Stage 2 evaluation prediction: {task}")
                prediction = normalized.float().cpu().numpy()
                for column, target in enumerate(spec.target_columns):
                    stats = scalers[task]["targets"][target]
                    prediction[:, column] = (
                        prediction[:, column] * float(stats["scale"])
                        + float(stats["mean"])
                    )
                raw_predictions.append(prediction.astype(np.float64))
                evaluation_progress.update(1)
            predictions = np.concatenate(raw_predictions, axis=0)
            targets = np.asarray([row["targets"] for row in rows], dtype=np.float64)
            task_metrics[task] = {
                target: _metrics(
                    predictions[:, column],
                    targets[:, column],
                    float(scalers[task]["targets"][target]["scale"]),
                )
                for column, target in enumerate(spec.target_columns)
            }
            if task in ORBITAL_TASK_TARGETS:
                target = ORBITAL_TASK_TARGETS[task]
                values = task_metrics[task][target]
                values["role_diagnostics"] = role_mae_diagnostics(
                    predictions[:, 0],
                    targets[:, 0],
                    [row["raw"]["ion_role"] for row in rows],
                )
            if predictions_dir is not None:
                output_rows: list[dict[str, Any]] = []
                fields = [
                    "source_row",
                    *spec.entity_columns,
                    *spec.condition_columns,
                    *orbital_audit_columns(spec.task_id),
                ]
                if len(spec.target_columns) == 1:
                    fields.extend(("target", "prediction", "absolute_error"))
                else:
                    for target in spec.target_columns:
                        fields.extend(
                            (
                                f"{target}_target",
                                f"{target}_prediction",
                                f"{target}_absolute_error",
                            )
                        )
                for row_index, row in enumerate(rows):
                    output: dict[str, Any] = {"source_row": row["source_row"]}
                    for name in (
                        *spec.entity_columns,
                        *spec.condition_columns,
                        *orbital_audit_columns(spec.task_id),
                    ):
                        output[name] = row["raw"][name]
                    for column, target in enumerate(spec.target_columns):
                        actual = float(targets[row_index, column])
                        predicted = float(predictions[row_index, column])
                        if len(spec.target_columns) == 1:
                            output["target"] = actual
                            output["prediction"] = predicted
                            output["absolute_error"] = abs(predicted - actual)
                        else:
                            output[f"{target}_target"] = actual
                            output[f"{target}_prediction"] = predicted
                            output[f"{target}_absolute_error"] = abs(predicted - actual)
                    output_rows.append(output)
                manifest = write_prediction_csv(
                    Path(predictions_dir) / f"{sanitize_task_id(task)}.csv",
                    output_rows,
                    fields,
                )
                manifest["path"] = f"predictions/{sanitize_task_id(task)}.csv"
                manifest["task"] = task
                prediction_manifests.append(manifest)

        partial_predictions: dict[str, np.ndarray] = {}
        partial_rows = partial_benchmark.evaluated
        partial_batches = math.ceil(
            len(partial_rows) / config.training.batch_size
        )
        for batch_index, start in enumerate(
            range(0, len(partial_rows), config.training.batch_size), start=1
        ):
            evaluation_progress.set_postfix_str(
                f"task={PARTIAL_CHARGE_TASK} batch={batch_index}/{partial_batches}",
                refresh=False,
            )
            chunk = partial_rows[start : start + config.training.batch_size]
            samples = []
            for molecule in chunk:
                key = (molecule.role, molecule.canonical_smiles)
                if key not in sample_cache:
                    sample_cache[key] = _entity_sample(
                        molecule.role,
                        molecule.canonical_smiles,
                        feature_config=feature_config,
                        vocabulary=vocabulary,
                        schema=schema,
                        standardizer=standardizer,
                    )
                sample = dict(sample_cache[key])
                sample["sample_id"] = f"stage2-evaluate:{molecule.mol_id}"
                samples.append(sample)
            packed = packer(samples).to(device)
            states = model.encode_entity_states(packed)
            positions = torch.arange(
                len(chunk), dtype=torch.long, device=device
            ).reshape(-1, 1)
            atom_state_indices = torch.arange(
                states.atom_states.shape[0], dtype=torch.long, device=device
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=config.training.amp_dtype == "bf16",
            ):
                normalized = model.predict_atom_from_states(
                    PARTIAL_CHARGE_TASK,
                    states,
                    positions,
                    packed.roles[positions],
                    states.entity_cls[positions],
                    atom_state_indices,
                    states.atom_batch,
                )
            normalized = normalized.reshape(-1)
            if not torch.isfinite(normalized).all():
                raise RuntimeError("Non-finite partial-charge evaluation prediction")
            raw = (
                normalized.float().cpu().numpy() * partial_benchmark.target_scale
                + partial_benchmark.target_mean
            )
            atom_batch = states.atom_batch.cpu().numpy()
            for row_index, molecule in enumerate(chunk):
                values = raw[atom_batch == row_index]
                partial_predictions[molecule.mol_id] = values.astype(np.float64)
            evaluation_progress.update(1)

        partial_score = score_partial_charge_predictions(
            partial_benchmark, partial_predictions
        )
        task_metrics[PARTIAL_CHARGE_TASK] = public_partial_charge_score(
            partial_score
        )
        if predictions_dir is not None:
            manifest = write_partial_charge_predictions(
                Path(predictions_dir) / f"{sanitize_task_id(PARTIAL_CHARGE_TASK)}.csv",
                partial_benchmark,
                partial_score,
            )
            manifest["path"] = (
                f"predictions/{sanitize_task_id(PARTIAL_CHARGE_TASK)}.csv"
            )
            prediction_manifests.append(manifest)
    finally:
        evaluation_progress.close()
    scalar_values = [
        float(task_metrics[task][target]["normalized_mae"])
        for task in STAGE2_CORE_TASKS
        for target in registry.by_id(task).target_columns
    ]
    epoch = int(checkpoint["completed_epoch"])
    core_value = sum(scalar_values) / len(scalar_values)
    partial_public = task_metrics[PARTIAL_CHARGE_TASK]
    partial_complete = partial_public["status"] == "complete"
    partial_value = (
        None
        if not partial_complete
        else float(partial_public["primary"]["molecule_macro_normalized_mae"])
    )
    full_value = (
        None
        if partial_value is None
        else (sum(scalar_values) + partial_value) / (len(scalar_values) + 1)
    )
    study_id = f"ilume-stage2-{checkpoint['training_identity']['hash']}"
    reporting_protocol = {
        "split": "test",
        "ensemble": False,
        "checkpoint_epoch": epoch,
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    return {
        "split": "test",
        "checkpoint_epoch": epoch,
        "tasks": task_metrics,
        "core_macro_normalized_mae": {
            "value": core_value,
            "valid_tasks": len(scalar_values),
            "total_tasks": len(_core_units(registry)),
        },
        "full_macro_normalized_mae": {
            "value": full_value,
            "valid_units": len(scalar_values) + (1 if partial_complete else 0),
            "total_units": len(scalar_values) + 1,
        },
        "reporting": {
            "schema_version": 1,
            "contract": STAGE2_BENCHMARK_SUITE_CONTRACT,
            "model_id": "ilume",
            "model_display_name": "ILUME",
            "study_id": study_id,
            "capabilities": {
                "stage2_core_physics": "supported",
                "stage2_partial_charge": "supported",
                "stage2_physics_full": "supported",
            },
            "benchmarks": {
                "stage2_core_physics": {
                    "status": "complete",
                    "benchmark": "stage2_physics",
                    "protocol": {
                        **reporting_protocol,
                        "expected_tasks": list(STAGE2_CORE_TASKS),
                    },
                    "comparison_identity": core_comparison,
                },
                "stage2_partial_charge": {
                    "status": partial_public["status"],
                    "benchmark": "stage2_partial_charge",
                    "protocol": {
                        **reporting_protocol,
                        "expected_tasks": [PARTIAL_CHARGE_TASK],
                        "expected_units": [PARTIAL_CHARGE_UNIT],
                        "weighting": "molecule_equal",
                    },
                    "comparison_identity": partial_benchmark.comparison_identity,
                    "issues": partial_public["coverage"]["issues"],
                },
                "stage2_physics_full": {
                    "status": "complete" if partial_complete else "incomplete",
                    "benchmark": "stage2_physics_full",
                    "protocol": {
                        **reporting_protocol,
                        "ordered_units": [*_core_units(registry), PARTIAL_CHARGE_UNIT],
                        "unit_weighting": "equal",
                    },
                    "comparison_identity": full_comparison,
                    "issues": [] if partial_complete else ["partial_charge_incomplete"],
                },
            },
            "predictions": prediction_manifests,
        },
    }


__all__ = [
    "STAGE2_CORE_TASKS",
    "evaluate_stage2_checkpoints",
    "resolve_checkpoint_path",
    "resolve_stage2_evaluation_contract",
    "resolve_stage2_evaluation_identity",
]
