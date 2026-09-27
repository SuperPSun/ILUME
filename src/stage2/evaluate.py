from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from rdkit import Chem

from common.identity import require_compatible_identity, semantic_identity
from common.io import sha256_file
from common.reporting import comparison_identity, reporting_block, sanitize_task_id, write_prediction_csv
from common.training import resolve_device
from stage1.config import config_from_dict
from stage1.descriptors import DescriptorSchema, DescriptorStandardizer, calculate_descriptors, rdkit_descriptor_names
from stage1.features import ROLE_TO_ID, build_entity_sample, inspect_entity_qc
from stage1.identity import validate_feature_generation_runtime
from stage1.masking import MultimodalPacker
from stage1.tokenizer import SmilesTokenizer

from .atom_targets import load_structure_manifest, load_verify_parse_and_map
from .data import (
    Stage2BatchDescriptor, Stage2DeviceTaskData, Stage2EntityDataset,
    Stage2TaskDataset, load_artifact_registry, pack_stage2_batch,
)
from .home_artifact import load_home_final
from .home_config import HomeRecipe, home_recipe_from_dict
from .identity import metadata_identity
from .prepare import _canonicalize, _finite_float, _role_for, _target_value
from .registry import Stage2Registry, TaskSpec, orbital_audit_columns, validate_orbital_audit_row


EVALUATION_TASKS = (
    "simulation/heat_of_vaporization",
    "simulation/thermal_expansion",
    "simulation/homo",
    "simulation/lumo",
    "simulation/partial_atomic_charge",
)


@dataclass
class TestEntities:
    samples: list[dict[str, Any]]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


@dataclass
class TestTask:
    spec: TaskSpec
    entity_indices: torch.Tensor
    conditions: torch.Tensor
    targets: torch.Tensor | None
    target_mask: torch.Tensor | None
    raw_targets: torch.Tensor | None
    source_rows: torch.Tensor
    atom_target_values: torch.Tensor | None
    atom_target_offsets: torch.Tensor | None
    atom_target_mask: torch.Tensor | None
    raw_atom_target_values: torch.Tensor | None
    mol_ids: tuple[str, ...]
    canonical_smiles: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.source_rows)


def _test_sources(recipe: HomeRecipe, registry: Any) -> tuple[tuple[str, ...], dict[str, str]]:
    tasks: list[str] = []
    hashes: dict[str, str] = {}
    for task in EVALUATION_TASKS:
        spec = registry.by_id(task)
        path = spec.dataset.split_path(recipe.stage2.data.data_root, "test")
        if not path.is_file() or path.stat().st_size == 0:
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            if not any(csv.DictReader(handle)):
                continue
        tasks.append(task)
        hashes[task] = sha256_file(path)
        manifest = spec.dataset.resource_manifest_path(recipe.stage2.data.data_root)
        if manifest is not None:
            hashes[task + ":structure_manifest"] = sha256_file(manifest)
    if not tasks:
        raise ValueError("No nonempty Stage 2 test splits for the five evaluation tasks")
    return tuple(tasks), hashes


def _evaluation_identity(recipe: HomeRecipe, artifact: Path, payload: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    if split not in {"valid", "test"}:
        raise ValueError("Stage 2 evaluation split must be valid or test")
    registry = Stage2Registry.from_snapshot(
        payload["registry"], registry_hash=payload["registry_hash"],
        catalog_sha256=payload["catalog_sha256"],
    )
    if sha256_file(recipe.stage2.data.task_catalog_path) != payload["catalog_sha256"]:
        raise ValueError("Stage 2 evaluation task catalog differs from the trained model")
    if split == "valid":
        prepared_registry = load_artifact_registry(recipe.stage2.data.artifacts_dir)
        metadata = json.loads((recipe.stage2.data.artifacts_dir / "metadata.json").read_text(encoding="utf-8"))
        prepared_identity = metadata_identity(metadata, "data", context="Stage 2 evaluation data")
        if (prepared_registry.registry_hash != registry.registry_hash
                or prepared_identity["hash"] != payload["stage2_data_identity"]["hash"]):
            raise ValueError("Stage 2 evaluation prepared data differs from the trained model")
        tasks, source_hashes = EVALUATION_TASKS, {"prepared_artifacts": metadata["artifact_hashes"]}
    else:
        tasks, source_hashes = _test_sources(recipe, registry)
    saved_recipe = home_recipe_from_dict(payload["recipe"])
    if (recipe.stage2.experiment_dict() != saved_recipe.stage2.experiment_dict()
            or recipe.stage3 != saved_recipe.stage3
            or recipe.initialization != saved_recipe.initialization
            or recipe.random_seed != saved_recipe.random_seed):
        raise ValueError("Stage 2 evaluation recipe differs from the trained model")
    return semantic_identity("stage2.home-evaluation.v1", {
        "split": split,
        "tasks": list(tasks),
        "final_artifact_sha256": sha256_file(artifact),
        "full_model_state_hash": payload["full_model_state_hash"],
        "prepared_data_identity": payload["stage2_data_identity"]["hash"],
        "source_hashes": source_hashes,
        "scalers": {task: payload["scalers"][task] for task in tasks},
        "atom_order": "canonical_rdkit_atom_index_v1",
    })


def resolve_evaluation_identity(recipe: HomeRecipe, checkpoint_dir: str | Path, *, split: str) -> dict[str, Any]:
    artifact = Path(checkpoint_dir) / "stage2_final.pt"
    payload, _, _ = load_home_final(artifact)
    return _evaluation_identity(recipe, artifact, payload, split=split)


def _feature_builder(payload: Mapping[str, Any]):
    config = config_from_dict(payload["stage1_config"])
    validate_feature_generation_runtime({"feature_generation_contract": payload["stage1_encoding_contract"]["feature_generation_contract"]})
    features = payload["feature_artifacts"]
    vocabulary = SmilesTokenizer.from_payload(features["tokenizer.json"])
    schema = DescriptorSchema.from_payload(features["descriptor_schema.json"], expected_raw_names=rdkit_descriptor_names())
    standardizer = DescriptorStandardizer.from_payload(features["descriptor_scaler.json"], expected_names=schema.selected_names)

    def build(role: str, smiles: str) -> dict[str, Any]:
        record = {
            "sample_id": f"stage2:test:{role}:{smiles}", "role": role,
            "role_id": ROLE_TO_ID[role], "canonical_smiles": smiles,
            "sources": ("stage2",), "split": "test", "is_augmented": False,
            "seed_smiles": (),
        }
        qc = inspect_entity_qc(record)
        if vocabulary.token_count(smiles) > config.data.max_smiles_tokens:
            qc.reasons.append("smiles_overlength")
        if qc.reasons:
            raise ValueError("Stage 2 test entity failed feature QC: " + ",".join(qc.reasons))
        molecule = Chem.MolFromSmiles(smiles)
        assert molecule is not None
        raw = calculate_descriptors(molecule, rdkit_descriptor_names())
        return build_entity_sample(record, raw, schema, standardizer, vocabulary, config)

    return build


def _load_test_task(recipe: HomeRecipe, spec: TaskSpec, scalers: Mapping[str, Any], build: Any) -> tuple[TestEntities, TestTask, list[dict[str, Any]]]:
    source = spec.dataset.split_path(recipe.stage2.data.data_root, "test")
    expected = (
        ("mol_id", *spec.entity_columns, "role", "formal_charge", "source_list")
        if spec.target_level == "atom" else
        (*spec.entity_columns, *spec.condition_columns, *orbital_audit_columns(spec.task_id), *spec.target_columns, "source_list")
    )
    manifest_path = spec.dataset.resource_manifest_path(recipe.stage2.data.data_root)
    structures = load_structure_manifest(manifest_path) if manifest_path is not None else {}
    samples: list[dict[str, Any]] = []
    entity_ids: dict[tuple[str, str], int] = {}
    slots: list[list[int]] = []
    conditions: list[list[float]] = []
    targets: list[list[float]] = []
    rows: list[int] = []
    mol_ids: list[str] = []
    canonicals: list[str] = []
    atom_values: list[float] = []
    atom_offsets = [0]
    audit: list[dict[str, Any]] = []
    role_cache: dict[str, tuple[str, int, int]] = {}
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"Unexpected Stage 2 test columns: {source}")
        for source_row, row in enumerate(reader, start=2):
            context = f"{spec.task_id}/test:{source_row}"
            canonical = tuple(_canonicalize(row[name].strip(), context) for name in spec.entity_columns)
            roles = tuple(_role_for(smiles, policy, row, context, role_cache) for smiles, policy in zip(canonical, spec.role_policy, strict=True))
            validate_orbital_audit_row(spec.task_id, row, inferred_role=roles[0], context=context)
            raw_conditions = [_finite_float(row[name], context + "/" + name) for name in spec.condition_columns]
            raw_targets: list[float] = []
            if spec.target_level == "atom":
                mol_id = row["mol_id"].strip()
                structure = structures.get(mol_id)
                if structure is None:
                    audit.append({"source_row": source_row, "mol_id": mol_id, "status": "excluded", "reason": "missing_structure_manifest_entry"})
                    continue
                try:
                    mapped = load_verify_parse_and_map(structure, canonical[0])
                except (OSError, ValueError) as error:
                    audit.append({"source_row": source_row, "mol_id": mol_id, "status": "excluded", "reason": "mapping_error:" + type(error).__name__})
                    continue
                raw_targets = list(mapped.charges)
            else:
                mol_id = ""
                raw_targets = [
                    _target_value(row[name], context + "/" + name, allow_missing=False)
                    for name in spec.target_columns
                ]
                if any(value is None for value in raw_targets):
                    raise ValueError(f"Stage 2 test target is missing: {context}")
            pending: list[tuple[tuple[str, str], dict[str, Any]]] = []
            try:
                for role, smiles in zip(roles, canonical, strict=True):
                    key = (role, smiles)
                    if key not in entity_ids:
                        pending.append((key, build(role, smiles)))
            except (RuntimeError, ValueError, OverflowError) as error:
                audit.append({"source_row": source_row, "mol_id": mol_id, "status": "excluded", "reason": "feature_qc:" + type(error).__name__})
                continue
            for key, sample in pending:
                if key not in entity_ids:
                    entity_ids[key] = len(samples)
                    samples.append(sample)
            slots.append([entity_ids[(role, smiles)] for role, smiles in zip(roles, canonical, strict=True)])
            conditions.append([(value - float(scalers["conditions"][name]["mean"])) / float(scalers["conditions"][name]["scale"]) for name, value in zip(spec.condition_columns, raw_conditions, strict=True)])
            targets.append(raw_targets if spec.target_level == "object" else [])
            if spec.target_level == "atom":
                atom_values.extend(raw_targets)
                atom_offsets.append(len(atom_values))
            rows.append(source_row)
            mol_ids.append(mol_id)
            canonicals.append(canonical[0])
            audit.append({"source_row": source_row, "mol_id": mol_id, "status": "evaluated", "reason": ""})
    if not rows:
        raise ValueError(f"Stage 2 test has no evaluable rows: {spec.task_id}")
    is_atom = spec.target_level == "atom"
    raw = None if is_atom else torch.tensor(targets, dtype=torch.float32)
    normalized = None if raw is None else raw.clone()
    if normalized is not None:
        for column, name in enumerate(spec.target_columns):
            stats = scalers["targets"][name]
            normalized[:, column] = (normalized[:, column] - float(stats["mean"])) / float(stats["scale"])
    atom_raw = torch.tensor(atom_values, dtype=torch.float32) if is_atom else None
    atom_normalized = None if atom_raw is None else (atom_raw - float(scalers["targets"][spec.target_columns[0]]["mean"])) / float(scalers["targets"][spec.target_columns[0]]["scale"])
    dataset = TestTask(
        spec, torch.tensor(slots, dtype=torch.long), torch.tensor(conditions, dtype=torch.float32).reshape(len(rows), len(spec.condition_columns)),
        normalized, None if normalized is None else torch.ones_like(normalized, dtype=torch.bool), raw,
        torch.tensor(rows, dtype=torch.long), atom_normalized,
        torch.tensor(atom_offsets, dtype=torch.long) if is_atom else None,
        torch.ones_like(atom_raw, dtype=torch.bool) if is_atom else None,
        atom_raw, tuple(mol_ids), tuple(canonicals),
    )
    return TestEntities(samples), dataset, audit


def _metrics(target: np.ndarray, prediction: np.ndarray, scale: float, *, molecule_offsets: list[int] | None = None) -> dict[str, Any]:
    if target.shape != prediction.shape or not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Stage 2 evaluation predictions/targets are invalid")
    error = prediction - target
    if molecule_offsets is None:
        mae = float(np.abs(error).mean())
        count = len(error)
    else:
        molecule_errors = [float(np.abs(error[start:end]).mean()) for start, end in zip(molecule_offsets[:-1], molecule_offsets[1:], strict=True)]
        mae = float(np.mean(molecule_errors))
        count = len(molecule_errors)
    rmse = float(np.sqrt(np.mean(error ** 2)))
    denominator = float(np.sum((target - target.mean()) ** 2))
    return {
        "count": count, "atom_count": len(error) if molecule_offsets is not None else None,
        "mae": mae, "rmse": rmse,
        "r2": 1.0 - float(np.sum(error ** 2)) / denominator if denominator > 0 else None,
        "normalized_mae": mae / scale, "normalized_rmse": rmse / scale,
        **({"atom_micro_mae": float(np.abs(error).mean())} if molecule_offsets is not None else {}),
    }


@torch.inference_mode()
def evaluate_home_final(
    recipe: HomeRecipe, checkpoint_dir: str | Path, *, split: str,
    predictions_dir: str | Path, expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = Path(checkpoint_dir) / "stage2_final.pt"
    payload, model, vocabulary = load_home_final(artifact)
    identity = _evaluation_identity(recipe, artifact, payload, split=split)
    if expected_identity is not None:
        require_compatible_identity(expected_identity, identity, context="Stage 2 evaluation")
    device = resolve_device(recipe.stage2.training.device)
    model.to(device).eval()
    registry = Stage2Registry.from_snapshot(
        payload["registry"], registry_hash=payload["registry_hash"],
        catalog_sha256=payload["catalog_sha256"],
    )
    tasks = tuple(identity["payload"]["tasks"])
    packer = MultimodalPacker(vocabulary)
    build = _feature_builder(payload) if split == "test" else None
    shared_entities = Stage2EntityDataset(recipe.stage2.data.artifacts_dir) if split == "valid" else None
    metrics: dict[str, dict[str, Any]] = {}
    manifests: list[dict[str, Any]] = []
    mapping_audit: dict[str, list[dict[str, Any]]] = {}
    output = Path(predictions_dir)
    for task in tasks:
        spec = registry.by_id(task)
        if split == "valid":
            assert shared_entities is not None
            entities: Any = shared_entities
            dataset: Any = Stage2TaskDataset(recipe.stage2.data.artifacts_dir, task, "valid")
            audit: list[dict[str, Any]] = []
        else:
            assert build is not None
            entities, dataset, audit = _load_test_task(recipe, spec, payload["scalers"][task], build)
        mapping_audit[task] = audit
        device_data = Stage2DeviceTaskData.from_dataset(dataset, device)
        rows: list[dict[str, Any]] = []
        observed: list[float] = []
        predicted: list[float] = []
        offsets = [0]
        for start in range(0, len(dataset), recipe.stage2_microbatch_size):
            indices = torch.arange(start, min(len(dataset), start + recipe.stage2_microbatch_size))
            packed = pack_stage2_batch(
                Stage2BatchDescriptor(task, indices), {task: dataset}, entities, packer,
                needs_entities=True, include_raw_atom_targets=spec.target_level == "atom", pin_memory=False,
            ).to(device, non_blocking=False)
            values = model.predict(task, packed, device_data).float().cpu()
            if spec.target_level == "atom":
                atom = packed.atom_targets
                assert atom is not None and atom.raw_values is not None
                raw = atom.raw_values.cpu()
                scale_stats = payload["scalers"][task]["targets"][spec.target_columns[0]]
                inverse = values * float(scale_stats["scale"]) + float(scale_stats["mean"])
                local_offsets = atom.molecule_offsets.cpu().tolist()
                for index, dataset_index in enumerate(indices.tolist()):
                    left, right = local_offsets[index:index + 2]
                    for atom_index in range(right - left):
                        target = float(raw[left + atom_index])
                        prediction = float(inverse[left + atom_index])
                        canonical = (
                            dataset.canonical_smiles[dataset_index]
                            if split == "test" else
                            entities.entries[int(dataset.entity_indices[dataset_index, 0])]["canonical_smiles"]
                        )
                        rows.append({"source_row": int(dataset.source_rows[dataset_index]), "mol_id": dataset.mol_ids[dataset_index], "canonical_smiles": canonical, "atom_index": atom_index, "target": target, "prediction": prediction, "absolute_error": abs(prediction - target)})
                        observed.append(target)
                        predicted.append(prediction)
                    offsets.append(len(observed))
            else:
                assert dataset.raw_targets is not None
                for index, dataset_index in enumerate(indices.tolist()):
                    for column, name in enumerate(spec.target_columns):
                        stats = payload["scalers"][task]["targets"][name]
                        target = float(dataset.raw_targets[dataset_index, column])
                        prediction = float(values[index, column]) * float(stats["scale"]) + float(stats["mean"])
                        rows.append({"source_row": int(dataset.source_rows[dataset_index]), "target_column": name, "target": target, "prediction": prediction, "absolute_error": abs(prediction - target)})
                        observed.append(target)
                        predicted.append(prediction)
        scales = [float(payload["scalers"][task]["targets"][name]["scale"]) for name in spec.target_columns]
        if len(scales) != 1:
            raise ValueError("Formal Stage 2 evaluator requires scalar selected tasks")
        metrics[task] = _metrics(np.asarray(observed), np.asarray(predicted), scales[0], molecule_offsets=offsets if spec.target_level == "atom" else None)
        fields = ("source_row", "mol_id", "canonical_smiles", "atom_index", "target", "prediction", "absolute_error") if spec.target_level == "atom" else ("source_row", "target_column", "target", "prediction", "absolute_error")
        manifest = write_prediction_csv(output / f"{sanitize_task_id(task)}.csv", rows, fields)
        manifest.update({"path": f"predictions/{sanitize_task_id(task)}.csv", "task": task})
        manifests.append(manifest)
    comparison = comparison_identity(
        "stage2_property", split=split, expected=tasks,
        sources={"prepared_data_identity": payload["stage2_data_identity"]["hash"], "source_hashes": identity["payload"]["source_hashes"]},
        normalization={task: {"scale": payload["scalers"][task]["targets"][registry.by_id(task).target_columns[0]]["scale"]} for task in tasks},
    )
    model_id, display_name = (
        ("ilume_stage2_no_stage1", "ILUME Stage2-HoME w/o Stage1")
        if recipe.initialization == "random_stage1" else
        ("ilume_stage2", "ILUME Stage2-HoME")
    )
    report = reporting_block(
        model_id=model_id, model_display_name=display_name, benchmark="stage2_property",
        protocol={"split": split, "expected_tasks": list(tasks), "ensemble": False, "folds": []},
        comparison=comparison, study_id="ilume_stage2_home_final_v2", predictions=manifests,
    )
    return {
        "split": split, "checkpoint_epoch": 10, "tasks": metrics,
        "macro_normalized_mae": sum(float(value["normalized_mae"]) for value in metrics.values()) / len(metrics),
        "mapping_audit": mapping_audit, "evaluation_identity": identity, "reporting": report,
    }
