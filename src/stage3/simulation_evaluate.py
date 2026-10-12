"""Protocol-specific scalar simulation ensemble from five Stage3 final models."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from rdkit import Chem

from common.identity import require_compatible_identity, semantic_identity
from common.io import sha256_file
from common.reporting import reporting_block, sanitize_task_id, write_prediction_csv
from stage3.simulation_reporting import (
    scalar_simulation_tasks, scalar_metrics, scalar_split_rows, simulation_comparison,
    simulation_scale, simulation_task_sources,
)
from common.training import resolve_device
from stage1.config import config_from_dict
from stage1.descriptors import DescriptorSchema, DescriptorStandardizer, calculate_descriptors, rdkit_descriptor_names
from stage1.features import ROLE_TO_ID, build_entity_sample, inspect_entity_qc
from stage1.identity import validate_feature_generation_runtime
from stage1.masking import MultimodalPacker
from stage1.tokenizer import SmilesTokenizer
from stage2.data import Stage2BatchDescriptor, Stage2DeviceTaskData, Stage2EntityDataset, Stage2TaskDataset, pack_stage2_batch
from stage2.identity import metadata_identity
from stage2.registry import load_stage2_registry, validate_orbital_audit_row
from .home import load_simulation_final, load_source


def _sources(config: Any, split: str) -> tuple[Any, dict[str, Any]]:
    if split not in {"valid", "test"} or config.training.simulation is None:
        raise ValueError("Simulation evaluation requires a simulation-trained Stage3 and valid/test split")
    source = load_source(config)
    assert source is not None
    registry = load_stage2_registry(config.data.task_catalog,
        task_ids=tuple(source["recipe"]["data"]["tasks"]) if config.is_entity_home else None)
    if registry.registry_hash != source["registry_hash"] or registry.catalog_sha256 != source["catalog_sha256"]:
        raise ValueError("Simulation evaluation catalog differs from the trained source")
    data_root = Path(source["recipe"]["data"]["data_root"])
    content = source["stage2_data_identity"]["payload"]["source_content"]
    for task in scalar_simulation_tasks(config):
        for part in ("train", "valid"):
            path = registry.by_id(task).dataset.split_path(data_root, part)
            if sha256_file(path) != content[f"{task}:{part}"]["sha256"]:
                raise ValueError("Simulation source train/valid changed since Stage2 preparation")
    return registry, source


def resolve_simulation_evaluation_identity(config: Any, checkpoint_dir: str | Path, *, split: str) -> dict[str, Any]:
    registry, source = _sources(config, split)
    root = Path(checkpoint_dir)
    checkpoints = {}
    for fold in range(1, 6):
        final = root / f"fold{fold}" / "three_phase_final.pt"
        checkpoints[f"fold{fold}"] = {"artifact": sha256_file(final), "manifest": sha256_file(final.with_suffix(".json"))}
    sources = {
        task: {part: sha256_file(registry.by_id(task).dataset.split_path(Path(source["recipe"]["data"]["data_root"]), part)) for part in ("train", split)}
        for task in scalar_simulation_tasks(config)
    }
    return semantic_identity("stage3.simulation-evaluation.v1", {
        "tasks": list(scalar_simulation_tasks(config)), "split": split, "checkpoints": checkpoints,
        "sources": sources, "stage2_full_state_hash": source["full_model_state_hash"],
        "stage2_scalers": source["scalers"], "simulation_recipe": config.to_dict()["training"]["simulation"],
        "prepared_data_identity": source["stage2_data_identity"], "ensemble": "raw_unit_prediction_mean_v1",
    })


def _feature_builder(source: dict[str, Any]):
    config = config_from_dict(source["stage1_config"])
    validate_feature_generation_runtime({"feature_generation_contract": source["stage1_encoding_contract"]["feature_generation_contract"]})
    features = source["feature_artifacts"]
    vocabulary = SmilesTokenizer.from_payload(features["tokenizer.json"])
    schema = DescriptorSchema.from_payload(features["descriptor_schema.json"], expected_raw_names=rdkit_descriptor_names())
    scaler = DescriptorStandardizer.from_payload(features["descriptor_scaler.json"], expected_names=schema.selected_names)

    def build(role: str, smiles: str) -> dict[str, Any]:
        record = {"sample_id": f"simulation:test:{role}:{smiles}", "role": role, "role_id": ROLE_TO_ID[role],
                  "canonical_smiles": smiles, "sources": ("stage2",), "split": "test", "is_augmented": False, "seed_smiles": ()}
        qc = inspect_entity_qc(record)
        if vocabulary.token_count(smiles) > config.data.max_smiles_tokens:
            qc.reasons.append("smiles_overlength")
        if qc.reasons:
            raise ValueError("Simulation test feature QC failed: " + ",".join(qc.reasons))
        return build_entity_sample(record, calculate_descriptors(Chem.MolFromSmiles(smiles), rdkit_descriptor_names()), schema, scaler, vocabulary, config)
    return build, vocabulary


def _test_data(spec: Any, rows: list[dict[str, str]], stats: dict[str, Any], build: Any):
    samples, ids, slots = [], {}, []
    for row in rows:
        current = []
        for column, policy in zip(spec.entity_columns, spec.role_policy, strict=True):
            mol = Chem.MolFromSmiles(row[column])
            if mol is None:
                raise ValueError("Invalid simulation test SMILES")
            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            charge = Chem.GetFormalCharge(mol)
            role = "cation" if charge > 0 else "anion" if charge < 0 else "neutral"
            if policy != "formal_charge" and role != policy:
                raise ValueError("Simulation test role differs from registry")
            validate_orbital_audit_row(spec.task_id, row, inferred_role=role, context="simulation test")
            key = (role, smiles)
            if key not in ids:
                ids[key] = len(samples)
                samples.append(build(role, smiles))
            current.append(ids[key])
        slots.append(current)
    conditions = [[(float(row[name]) - stats["conditions"][name]["mean"]) / stats["conditions"][name]["scale"] for name in spec.condition_columns] for row in rows]
    targets = torch.tensor([[float(row[spec.target_columns[0]])] for row in rows])
    target_stats = stats["targets"][spec.target_columns[0]]
    dataset = SimpleNamespace(spec=spec, entity_indices=torch.tensor(slots),
        conditions=torch.tensor(conditions).reshape(len(rows), len(spec.condition_columns)),
        targets=(targets - target_stats["mean"]) / target_stats["scale"], raw_targets=targets, target_mask=torch.ones_like(targets, dtype=torch.bool),
        atom_target_values=None, atom_target_offsets=None, atom_target_mask=None)
    return samples, dataset


@torch.inference_mode()
def evaluate_simulation_checkpoints(config: Any, checkpoint_dir: str | Path, *, split: str,
                                    predictions_dir: str | Path, expected_evaluation_identity: Any = None,
                                    reporting_study_id: str | None = None) -> dict[str, Any]:
    identity = resolve_simulation_evaluation_identity(config, checkpoint_dir, split=split)
    if expected_evaluation_identity is not None:
        require_compatible_identity(expected_evaluation_identity, identity, context="Stage3 simulation evaluation")
    registry, source = _sources(config, split)
    source.pop("_loaded_model", None)
    source.pop("full_model_state", None)
    build, vocabulary = _feature_builder(source)
    data_root = Path(source["recipe"]["data"]["data_root"])
    raw, datasets, entities, scales, comparison_sources = {}, {}, {}, {}, {}
    prepared_entities = None
    if split == "valid":
        prepared_entities = Stage2EntityDataset(config.initialization.simulation_artifacts_dir)
        data_identity = metadata_identity(prepared_entities.metadata, "data", context="simulation evaluation")
        require_compatible_identity(source["stage2_data_identity"], data_identity, context="simulation prepared source")
    for task in scalar_simulation_tasks(config):
        spec = registry.by_id(task)
        train_path = spec.dataset.split_path(data_root, "train")
        path = spec.dataset.split_path(data_root, split)
        rows = scalar_split_rows(path, (*spec.entity_columns, *spec.condition_columns, *spec.target_columns))
        raw[task] = rows
        if split == "valid":
            dataset = Stage2TaskDataset(config.initialization.simulation_artifacts_dir, task, "valid")
            row_ids = [f"{path.as_posix()}:{number}" for number in dataset.source_rows.tolist()]
            entities[task] = prepared_entities
            expected_targets = np.asarray([[float(row[spec.target_columns[0]])] for row in rows])
            if dataset.raw_targets.shape != expected_targets.shape or not np.allclose(dataset.raw_targets.numpy(), expected_targets, rtol=1e-6, atol=1e-6):
                raise ValueError("Simulation prepared targets differ from the raw split")
        else:
            entities[task], dataset = _test_data(spec, rows, source["scalers"][task], build)
            row_ids = [f"{path.as_posix()}:{number}" for number in range(2, len(rows) + 2)]
        datasets[task] = dataset
        scales[task] = simulation_scale(train_path, spec.target_columns[0])
        comparison_sources.update(simulation_task_sources(task, train_path, path, row_ids))
    device = resolve_device(config.training.device)
    predictions = {task: [] for task in scalar_simulation_tasks(config)}
    for fold in range(1, 6):
        model = load_simulation_final(config, checkpoint_dir, fold=fold, device=device).eval()
        for task, dataset in datasets.items():
            values = []
            task_data = Stage2DeviceTaskData.from_dataset(dataset, device)
            for start in range(0, len(raw[task]), 256):
                indices = torch.arange(start, min(start + 256, len(raw[task])))
                packed = pack_stage2_batch(Stage2BatchDescriptor(task, indices), {task: dataset}, entities[task],
                    MultimodalPacker(vocabulary), needs_entities=True, include_raw_atom_targets=False, pin_memory=False).to(device, non_blocking=False)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.training.amp_dtype == "bf16"):
                    predicted = model.predict_simulation(task, packed, task_data).float().cpu().numpy()
                stats = source["scalers"][task]["targets"][registry.by_id(task).target_columns[0]]
                values.append(predicted.astype(np.float64) * stats["scale"] + stats["mean"])
            predictions[task].append(np.concatenate(values).reshape(-1))
        del model
    metrics, manifests = {}, []
    for task in scalar_simulation_tasks(config):
        prediction = np.stack(predictions[task]).mean(axis=0)
        spec = registry.by_id(task)
        targets = np.asarray([float(row[spec.target_columns[0]]) for row in raw[task]])
        metrics[task] = scalar_metrics(targets, prediction, scales[task])
        rows = [{"source_row": number, "target": actual, "prediction": predicted, "absolute_error": abs(predicted - actual)}
                for number, actual, predicted in zip(range(2, len(targets) + 2), targets, prediction, strict=True)]
        path = Path(predictions_dir) / f"{sanitize_task_id(task)}.csv"
        manifest = write_prediction_csv(path, rows, ("source_row", "target", "prediction", "absolute_error"))
        manifest.update({"task": task, "path": f"predictions/{path.name}"})
        manifests.append(manifest)
    require_compatible_identity(identity, resolve_simulation_evaluation_identity(config, checkpoint_dir, split=split), context="Simulation sources changed during evaluation")
    comparison = simulation_comparison(split=split, tasks=scalar_simulation_tasks(config), sources=comparison_sources, scales=scales)
    study = reporting_study_id or "ilume-simulation-" + identity["hash"]
    no_stage1 = source["recipe"]["home"]["initialization"] == "random_stage1"
    model_id, display = ("ilume_no_stage1", "ILUME w/o Stage1") if no_stage1 else ("ilume", "ILUME")
    return {"split": split, "domain": "simulation", "checkpoint_epoch": None, "tasks": metrics,
        "macro_normalized_mae": sum(value["normalized_mae"] for value in metrics.values()) / len(metrics),
        "reporting": reporting_block(model_id=model_id, model_display_name=display, benchmark="simulation_property",
            protocol={"split": split, "expected_tasks": list(scalar_simulation_tasks(config)), "folds": list(range(1, 6)), "ensemble": True,
                      "ensemble_method": "raw_unit_prediction_mean_v1", "model_selector": "three_phase_final", "model_sources": identity["payload"]["checkpoints"]},
            comparison=comparison, study_id=study, predictions=manifests)}
