from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from rdkit import Chem

from common.io import sha256_file
from stage1.config import (
    STAGE1_CHECKPOINT_KIND, STAGE1_CHECKPOINT_VERSION, DataConfig,
    DescriptorConfig, FingerprintConfig, ModelConfig, PretrainConfig,
)
from stage1.data import PreparedCorpusDataset
from stage1.masking import MultimodalPacker
from stage1.model import MultimodalPretrainModel, load_stage1_model
from stage1.prepare import prepare_corpus
from stage1.tokenizer import SmilesTokenizer
from stage2.atom_targets import (
    StructureManifestEntry, map_partial_charges, parse_mol2,
    verify_structure,
)
from stage2.config import (
    Stage2Config, Stage2DataConfig, Stage2InitializationConfig,
    Stage2PreparationConfig, Stage2TrainingConfig,
)
from stage2.data import (
    Stage2BatchDescriptor, Stage2EntityDataset, Stage2TaskDataset,
    epoch_batch_schedule, load_artifact_registry, pack_stage2_window,
)
from stage2.model import (
    ObjectEncoder, Stage2ObjectModel, masked_target_macro_smooth_l1_loss,
    molecule_equal_smooth_l1_loss,
)
from stage2.prepare import prepare_stage2_data, prepare_teacher_cache
from stage2.registry import load_stage2_registry
from stage2.train import (
    load_stage2_encoder_artifact, run_stage2_training,
    task_compensation_scale,
)
from stage3.prepare import STAGE3_MIGRATION_MESSAGE, load_frozen_stage2


TASKS = (
    "simulation/density",
    "simulation/heat_capacity",
    "simulation/heat_of_vaporization",
    "simulation/partial_atomic_charge",
    "simulation/pbe_tzvp_anion_orbitals",
    "simulation/pbe_tzvp_cation_orbitals",
    "simulation/simulated_qm_elec_hf",
    "simulation/thermal_expansion",
    "simulation/transfer_organic",
)


def _write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_mol2(path: Path, atoms: list[tuple[str, str, float]], bonds: list[tuple[int, int, str]]) -> None:
    lines = ["@<TRIPOS>MOLECULE", "MOL", f"{len(atoms)} {len(bonds)} 1 0 0", "SMALL", "resp", "@<TRIPOS>ATOM"]
    for index, (name, atom_type, charge) in enumerate(atoms, start=1):
        lines.append(f"{index} {name} {index}.0 0.0 0.0 {atom_type} 1 MOL {charge}")
    lines.append("@<TRIPOS>BOND")
    for index, (first, second, kind) in enumerate(bonds, start=1):
        lines.append(f"{index} {first} {second} {kind}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stage2_sources(data_root: Path) -> None:
    root = data_root / "stage2"
    il_tasks = (
        ("density", "density_g/cm^3", 1.0, 100.0),
        ("heat_capacity", "heat_capacity_J/mol/K", 200.0, 300.0),
        ("heat_of_vaporization", "heat_of_vaporization_kJ/mol", 10.0, 11.0),
        ("thermal_expansion", "thermal_expansion_K^-1", 0.001, 0.002),
    )
    for name, target, train_value, valid_value in il_tasks:
        fields = ["cation", "anion", "temperature_K", target, "source_list"]
        train = [{"cation": "[Na+]", "anion": "[Cl-]", "temperature_K": 298, target: train_value, "source_list": "simulation"}]
        if name == "density":
            train.append({"cation": "[Na+]", "anion": "[Cl-]", "temperature_K": 298, target: 2.0, "source_list": "simulation"})
        valid = [{"cation": "C[NH3+]", "anion": "C(=O)[O-]", "temperature_K": 310, target: valid_value, "source_list": "simulation"}]
        _write_csv(root / name / "train.csv", fields, train)
        _write_csv(root / name / "valid.csv", fields, valid)
    for name, column, train_smiles, valid_smiles in (
        ("pbe_tzvp_cation_orbitals", "cation", "[Na+]", "C[NH3+]"),
        ("pbe_tzvp_anion_orbitals", "anion", "[Cl-]", "C(=O)[O-]"),
    ):
        fields = [column, "HOMO_eV", "LUMO_eV", "source_list"]
        _write_csv(root / name / "train.csv", fields, [{column: train_smiles, "HOMO_eV": -1.0, "LUMO_eV": 1.0, "source_list": "simulation"}])
        _write_csv(root / name / "valid.csv", fields, [{column: valid_smiles, "HOMO_eV": -2.0, "LUMO_eV": 2.0, "source_list": "simulation"}])
    qm_targets = ("ESP_max", "ESP_min", "ESP_std", "ESP_pos_frac", "Dipole", "Quadrupole", "q_max", "q_min", "q_std", "q_pos_frac", "gap_eV")
    qm_fields = ["SMILES", *qm_targets, "source_list"]
    _write_csv(root / "simulated_qm_elec_hf/train.csv", qm_fields, [{"SMILES": "CC", **{name: index + 0.5 for index, name in enumerate(qm_targets)}, "source_list": "simulation"}])
    _write_csv(root / "simulated_qm_elec_hf/valid.csv", qm_fields, [{"SMILES": "CCC", **{name: index + 10.5 for index, name in enumerate(qm_targets)}, "source_list": "simulation"}])
    transfer_fields = ["solute", "solvent", "transfer_organic_kcal/mol", "source_list"]
    _write_csv(root / "transfer_organic/train.csv", transfer_fields, [{"solute": "CC", "solvent": "O", "transfer_organic_kcal/mol": -1.0, "source_list": "simulation"}])
    _write_csv(root / "transfer_organic/valid.csv", transfer_fields, [{"solute": "CCC", "solvent": "O", "transfer_organic_kcal/mol": -2.0, "source_list": "simulation"}])

    structure_root = root / "partial_atomic_charge/charge_20260514"
    structure_root.mkdir(parents=True)
    structures = {
        "mol_train": ("CCO", [("C1", "c3", -0.1), ("C2", "c3", 0.2), ("O1", "os", -0.1)], [(1, 2, "1"), (2, 3, "1")]),
        "mol_valid": ("O", [("O1", "os", -0.2)], []),
    }
    manifest = []
    for mol_id, (_, atoms, bonds) in structures.items():
        path = structure_root / f"{mol_id}.mol2"
        _write_mol2(path, atoms, bonds)
        manifest.append({"mol_id": mol_id, "relative_path": path.name, "format": "mol2", "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "referenced_by_charge": "True"})
    _write_csv(structure_root / "structure_manifest.csv", ["mol_id", "relative_path", "format", "size_bytes", "sha256", "referenced_by_charge"], manifest)
    for split, mol_id in (("train", "mol_train"), ("valid", "mol_valid")):
        _write_csv(root / "partial_atomic_charge" / f"{split}.csv", ["mol_id", "SMILES", "role", "formal_charge", "source_list"], [{"mol_id": mol_id, "SMILES": structures[mol_id][0], "role": "neutral", "formal_charge": 0, "source_list": "simulation"}])

    fields = ["catalog_schema_version", "stage", "task_id", "task_kind", "target_level", "source_file", "target_columns", "identity_columns", "condition_columns", "system_type", "split_unit", "sample_unit", "simulation_method", "experiment_reference", "materialized_path", "label_source", "resource_manifest", "raw_rows", "rows", "unique_systems", "tier", "test_systems", "reserved_systems", "development_systems", "strategies", "repeats", "strategy_units"]
    definitions = (
        ("density", "object_property", "object", "density_g/cm^3", "cation;anion", "temperature_K", "il", "materialized_csv", ""),
        ("heat_capacity", "object_property", "object", "heat_capacity_J/mol/K", "cation;anion", "temperature_K", "il", "materialized_csv", ""),
        ("heat_of_vaporization", "object_property", "object", "heat_of_vaporization_kJ/mol", "cation;anion", "temperature_K", "il", "materialized_csv", ""),
        ("partial_atomic_charge", "atom_property", "atom", "partial_atomic_charge", "SMILES", "", "molecule", "structure_resource", "stage2/partial_atomic_charge/charge_20260514/structure_manifest.csv"),
        ("pbe_tzvp_anion_orbitals", "object_property", "object", "HOMO_eV;LUMO_eV", "anion", "", "anion", "materialized_csv", ""),
        ("pbe_tzvp_cation_orbitals", "object_property", "object", "HOMO_eV;LUMO_eV", "cation", "", "cation", "materialized_csv", ""),
        ("simulated_qm_elec_hf", "object_property", "object", ";".join(qm_targets), "SMILES", "", "molecule", "materialized_csv", ""),
        ("thermal_expansion", "object_property", "object", "thermal_expansion_K^-1", "cation;anion", "temperature_K", "il", "materialized_csv", ""),
        ("transfer_organic", "object_property", "object", "transfer_organic_kcal/mol", "solute;solvent", "", "solute_solvent", "materialized_csv", ""),
    )
    rows = []
    for name, kind, level, targets, identities, conditions, system_type, label_source, resource_manifest in definitions:
        row = {field: "" for field in fields}
        row.update({"catalog_schema_version": 1, "stage": 2, "task_id": f"simulation/{name}", "task_kind": kind, "target_level": level, "source_file": f"simulation/{name}.csv", "target_columns": targets, "identity_columns": identities, "condition_columns": conditions, "system_type": system_type, "materialized_path": f"stage2/{name}", "label_source": label_source, "resource_manifest": resource_manifest})
        rows.append(row)
    _write_csv(data_root / "task_catalog.csv", fields, rows)


@pytest.fixture
def tiny_stage2_setup(tmp_path: Path) -> Stage2Config:
    stage1 = tmp_path / "stage1"
    _write_csv(stage1 / "cation.csv", ["SMILES"], [{"SMILES": "[Na+]"}, {"SMILES": "C[NH3+]"}])
    _write_csv(stage1 / "anion.csv", ["SMILES"], [{"SMILES": "[Cl-]"}, {"SMILES": "C(=O)[O-]"}])
    _write_csv(stage1 / "molecule.csv", ["SMILES"], [{"SMILES": value} for value in ("O", "CC", "CCC", "CCO")])
    corpus = tmp_path / "pretrain"
    pretrain = PretrainConfig(
        data=DataConfig(stage1_dir=stage1, artifacts_dir=corpus, valid_fraction=0.5, max_smiles_tokens=64, shard_size=4),
        descriptor=DescriptorConfig(mode="full", token_count=8),
        fingerprint=FingerprintConfig(kind="both"),
        model=ModelConfig(d_model=16, n_heads=4, smiles_layers=1, graph_depth=2, descriptor_hidden_dim=32, descriptor_blocks=1, fusion_layers=1, feedforward_dim=32, dropout=0.0),
    )
    prepare_corpus(pretrain)
    vocabulary = SmilesTokenizer.load(corpus / "tokenizer.json")
    dataset = PreparedCorpusDataset(corpus, "train")
    model = MultimodalPretrainModel(pretrain, vocabulary, dataset.descriptor_schema)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"kind": STAGE1_CHECKPOINT_KIND, "format_version": STAGE1_CHECKPOINT_VERSION, "model": model.state_dict(), "config": pretrain.to_dict(), "artifact_hash": sha256_file(corpus / "metadata.json")}, checkpoint)
    _stage2_sources(tmp_path)
    return Stage2Config(
        data=Stage2DataConfig(data_root=tmp_path, task_catalog_path=tmp_path / "task_catalog.csv", pretrain_artifacts_dir=corpus, artifacts_dir=tmp_path / "stage2_artifacts", entity_shard_size=3),
        preparation=Stage2PreparationConfig(workers=1, teacher_batch_size=4),
        initialization=Stage2InitializationConfig(checkpoint=checkpoint),
        training=Stage2TrainingConfig(batch_size=2, epochs=2, backbone_frozen_epochs=1, packing_workers=2, packing_prefetch_windows=2, log_every_batches=3, device="cpu", amp_dtype="none"),
    )


def test_registry_is_catalog_driven_and_model_independent(tiny_stage2_setup):
    registry = load_stage2_registry(tiny_stage2_setup.data.task_catalog_path)
    assert registry.task_ids == TASKS
    original = registry.registry_hash
    loaded = load_stage1_model(tiny_stage2_setup.initialization.checkpoint, tiny_stage2_setup.data.pretrain_artifacts_dir, backbone_dropout=0.0)
    model = Stage2ObjectModel(loaded.model, registry, object_layers=1, object_ffn_dim=32, dropout=0.0)
    assert registry.registry_hash == original
    assert model.model_contract["d_model"] == 16
    assert model.model_contract["tasks"]["simulation/partial_atomic_charge"]["head_family"] == "atom"


def test_config_normalizes_relative_weights_and_rejects_accumulation(tiny_stage2_setup):
    registry = load_stage2_registry(tiny_stage2_setup.data.task_catalog_path)
    normalized = tiny_stage2_setup.normalized_task_weights(registry)
    scaled = replace(tiny_stage2_setup, loss=replace(tiny_stage2_setup.loss, task_weights={key: value * 10 for key, value in tiny_stage2_setup.loss.task_weights.items()}))
    assert scaled.normalized_task_weights(registry) == pytest.approx(normalized)
    with pytest.raises(ValueError, match="gradient_accumulation_steps == 1"):
        replace(tiny_stage2_setup, training=replace(tiny_stage2_setup.training, gradient_accumulation_steps=2)).validate()


def test_prepare_v3_task_local_scalers_and_ragged_atoms(tiny_stage2_setup):
    metadata = prepare_stage2_data(tiny_stage2_setup)
    assert metadata["format_version"] == 3
    assert metadata["summary"]["rows"]["simulation/density"]["train"] == 2
    density = metadata["scalers"]["simulation/density"]["targets"]["density_g/cm^3"]
    assert density["mean"] == pytest.approx(1.5)
    assert density["scale"] == pytest.approx(0.5)
    atom = Stage2TaskDataset(tiny_stage2_setup.data.artifacts_dir, "simulation/partial_atomic_charge", "train")
    assert atom.mol_ids == ("mol_train",)
    assert atom.atom_target_offsets.tolist() == [0, 3]
    assert metadata["scalers"]["simulation/partial_atomic_charge"]["targets"]["partial_atomic_charge"]["weighting"] == "molecule_equal"
    audit = list(csv.DictReader((tiny_stage2_setup.data.artifacts_dir / "partial_charge_mapping_audit.csv").open()))
    assert {row["status"] for row in audit} == {"mapped"}


def test_partial_charge_duplicate_smiles_remain_distinct_molecules(tiny_stage2_setup):
    root = tiny_stage2_setup.data.data_root / "stage2/partial_atomic_charge"
    structure_root = root / "charge_20260514"
    duplicate = structure_root / "mol_train_2.mol2"
    _write_mol2(
        duplicate,
        [("C1", "c3", -0.2), ("C2", "c3", 0.3), ("O1", "os", -0.1)],
        [(1, 2, "1"), (2, 3, "1")],
    )
    manifest_path = structure_root / "structure_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest_rows.append({"mol_id": "mol_train_2", "relative_path": duplicate.name, "format": "mol2", "size_bytes": duplicate.stat().st_size, "sha256": sha256_file(duplicate), "referenced_by_charge": "True"})
    _write_csv(manifest_path, list(manifest_rows[0]), manifest_rows)
    train_path = root / "train.csv"
    with train_path.open(newline="", encoding="utf-8") as handle:
        train_rows = list(csv.DictReader(handle))
    train_rows.append({**train_rows[0], "mol_id": "mol_train_2"})
    _write_csv(train_path, list(train_rows[0]), train_rows)

    metadata = prepare_stage2_data(tiny_stage2_setup)
    atom = Stage2TaskDataset(tiny_stage2_setup.data.artifacts_dir, "simulation/partial_atomic_charge", "train")
    assert atom.mol_ids == ("mol_train", "mol_train_2")
    assert atom.entity_indices[0].equal(atom.entity_indices[1])
    assert atom.atom_target_offsets.tolist() == [0, 3, 6]
    assert metadata["scalers"]["simulation/partial_atomic_charge"]["targets"]["partial_atomic_charge"]["count"] == 2


def test_stage1_encode_states_uses_fused_atom_order(tiny_stage2_setup):
    prepare_stage2_data(tiny_stage2_setup)
    loaded = load_stage1_model(tiny_stage2_setup.initialization.checkpoint, tiny_stage2_setup.data.pretrain_artifacts_dir, backbone_dropout=0.0)
    entities = Stage2EntityDataset(tiny_stage2_setup.data.artifacts_dir)
    batch = MultimodalPacker(loaded.vocabulary)([entities[0], entities[1]])
    states = loaded.model.encode_states(batch)
    assert torch.equal(states.entity_cls, loaded.model.encode(batch))
    assert states.atom_states.shape[0] == batch.graphs.atom_batch.shape[0]
    assert torch.equal(states.atom_batch, batch.graphs.atom_batch)


def test_mol2_typed_mapping_and_explicit_connectivity_fallback(tmp_path):
    typed_path = tmp_path / "typed.mol2"
    _write_mol2(typed_path, [("C1", "c3", 0.1), ("O1", "os", -0.1), ("H1", "h1", 0.0)], [(1, 2, "1"), (1, 3, "1")])
    typed = map_partial_charges("CO", parse_mol2(typed_path))
    assert typed.bond_match_mode == "typed"
    assert typed.charges == pytest.approx((0.1, -0.1))
    fallback_path = tmp_path / "fallback.mol2"
    _write_mol2(fallback_path, [("C1", "c3", 0.1), ("O1", "os", -0.1)], [(1, 2, "du")])
    fallback = map_partial_charges("CO", parse_mol2(fallback_path))
    assert fallback.bond_match_mode == "connectivity_only"
    assert fallback.unparsed_bond_types == ("du",)
    symmetric_path = tmp_path / "symmetric.mol2"
    _write_mol2(symmetric_path, [("C1", "c3", 0.2), ("C2", "c3", -0.2)], [(1, 2, "1")])
    symmetric = map_partial_charges("CC", parse_mol2(symmetric_path))
    assert symmetric.mapping_count == 2
    assert symmetric.charges == pytest.approx((0.2, -0.2))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_structure(StructureManifestEntry("bad", typed_path, typed_path.stat().st_size, "0" * 64))


def test_object_encoder_roles_dynamic_heads_and_losses(tiny_stage2_setup):
    registry = load_stage2_registry(tiny_stage2_setup.data.task_catalog_path)
    loaded = load_stage1_model(tiny_stage2_setup.initialization.checkpoint, tiny_stage2_setup.data.pretrain_artifacts_dir, backbone_dropout=0.0)
    model = Stage2ObjectModel(loaded.model, registry, object_layers=1, object_ffn_dim=32, dropout=0.0)
    values = torch.randn(2, 1, 16)
    for role in range(3):
        assert model.encode_object(values, torch.full((2, 1), role)).shape == (2, 16)
    ions = torch.randn(2, 2, 16)
    assert model.encode_object(ions, torch.tensor([[0, 1], [0, 1]])).shape == (2, 16)
    with pytest.raises(ValueError, match="ordered"):
        model.encode_object(ions, torch.tensor([[1, 0], [1, 0]]))
    assert model.object_heads["simulation/pbe_tzvp_cation_orbitals"] is not model.object_heads["simulation/pbe_tzvp_anion_orbitals"]
    predictions = torch.tensor([[2.0, 2.0], [0.0, 2.0]], requires_grad=True)
    loss = masked_target_macro_smooth_l1_loss(predictions, torch.zeros_like(predictions), torch.tensor([[True, True], [False, True]]))
    assert loss.item() == pytest.approx(1.5)
    atom_loss = molecule_equal_smooth_l1_loss(torch.tensor([2.0, 2.0, 2.0]), torch.zeros(3), torch.ones(3, dtype=torch.bool), torch.tensor([0, 1, 3]))
    assert atom_loss.item() == pytest.approx(1.5)


def test_round_robin_schedule_is_complete_deterministic_and_no_cycle(tiny_stage2_setup):
    prepare_stage2_data(tiny_stage2_setup)
    datasets = {task: Stage2TaskDataset(tiny_stage2_setup.data.artifacts_dir, task, "train") for task in TASKS}
    first = epoch_batch_schedule(datasets, 1, seed=42, epoch=1)
    second = epoch_batch_schedule(datasets, 1, seed=42, epoch=1)
    assert [(item.task, item.indices.tolist()) for item in first] == [(item.task, item.indices.tolist()) for item in second]
    for task, dataset in datasets.items():
        observed = torch.cat([item.indices for item in first if item.task == task]).tolist()
        assert sorted(observed) == list(range(len(dataset)))
    assert len({item.task for item in first[:len(TASKS)]}) == len(TASKS)


def test_task_compensation_only_scales_physics():
    compensation = task_compensation_scale(0.25, 20, 4, 10)
    physics = torch.tensor(2.0)
    teacher = torch.tensor(3.0)
    step = compensation * physics + 0.1 * teacher
    assert step.item() == pytest.approx(4.3)


def test_prepare_train_checkpoint_and_encoder_export(tiny_stage2_setup, tmp_path):
    prepare_teacher_cache(tiny_stage2_setup)
    legacy = tmp_path / "object_v2.pt"
    torch.save({"kind": "ilume_stage2_object", "format_version": 2}, legacy)
    with pytest.raises(ValueError, match="Object v2 is not migrated"):
        run_stage2_training(
            tiny_stage2_setup,
            output_dir=tmp_path / "legacy_resume",
            resume_from=legacy,
        )
    output = tmp_path / "train"
    run_stage2_training(tiny_stage2_setup, output_dir=output)
    assert (output / "checkpoint_epoch_00001.pt").is_file()
    final_checkpoint = torch.load(output / "checkpoint_epoch_00002.pt", map_location="cpu", weights_only=False)
    assert final_checkpoint["format_version"] == 3
    assert final_checkpoint["completed_epoch"] == 2
    assert final_checkpoint["registry_hash"] == load_artifact_registry(tiny_stage2_setup.data.artifacts_dir).registry_hash
    encoder_path = output / "stage2_encoder.pt"
    encoder = load_stage2_encoder_artifact(encoder_path)
    assert encoder["kind"] == "ilume_stage2_encoder"
    assert not any("head" in key for key in encoder["stage1_backbone"])
    assert set(encoder) >= {"stage1_backbone", "object_encoder", "model_contract", "state_hashes", "provenance"}
    with pytest.raises(ValueError, match=STAGE3_MIGRATION_MESSAGE):
        load_frozen_stage2(encoder_path)

    resume_output = tmp_path / "resume"
    resume_output.mkdir()
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    epoch_one = [row for row in rows if int(row.get("epoch", 0)) <= 1]
    (resume_output / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in epoch_one) + "\n",
        encoding="utf-8",
    )
    run_stage2_training(
        tiny_stage2_setup,
        output_dir=resume_output,
        resume_from=output / "checkpoint_epoch_00001.pt",
    )
    resumed = torch.load(resume_output / "checkpoint_epoch_00002.pt", map_location="cpu", weights_only=False)
    assert resumed["scheduler_geometry"]["gradient_accumulation_steps"] == 1
    assert (resume_output / "stage2_encoder.pt").is_file()
