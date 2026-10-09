from __future__ import annotations

import copy
import csv
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch


from common.identity import IDENTITY_CONTRACT_VERSION
from common.io import sha256_file
from stage1.config import (
    ArchitectureConfig,
    GLOBAL_RDKIT_STAGE1_CHECKPOINT_VERSION,
    STAGE1_CHECKPOINT_KIND,
    STAGE1_CHECKPOINT_VERSION,
    DataConfig,
    DescriptorConfig,
    FingerprintConfig,
    ModelConfig,
    PretrainConfig,
)
from stage1.data import PreparedCorpusDataset
from stage1.descriptors import DescriptorSchema, rdkit_descriptor_names
from stage1.identity import metadata_identity
from stage1.model import MultimodalPretrainModel, load_stage1_model
from stage1.prepare import prepare_corpus
from stage1.tokenizer import SmilesTokenizer
from stage2 import FrozenObjectSpec, load_frozen_object_encoder
from stage2.atom_targets import map_partial_charges, parse_mol2
from stage2.config import (
    DEFAULT_REFINEMENT_TASKS,
    STAGE2_CHECKPOINT_VERSION,
    Stage2Config,
    Stage2DataConfig,
    Stage2InitializationConfig,
    Stage2PreparationConfig,
    Stage2RepresentationConfig,
    Stage2TrainingConfig,
    load_stage2_config,
    stage2_config_from_dict,
)
from stage2.data import (
    STAGE2_PREPARATION_CONTRACT_VERSION,
    Stage2TaskDataset,
    load_artifact_registry,
    epoch_batch_schedule,
)
from stage2.identity import build_stage2_training_identity
from stage2.model import (
    RDKitDescriptorBackbone,
    Stage2ObjectModel,
    molecule_equal_smooth_l1_loss,
    masked_target_macro_smooth_l1_loss,
)
from stage2.prepare import (
    prepare_stage2_data,
    prepare_teacher_cache,
    stage1_encoder_identity,
    teacher_cache_identity,
)
from stage2.rdkit_train import (
    STAGE2_RDKIT_CHECKPOINT_KIND,
    STAGE2_RDKIT_ENCODER_KIND,
    STAGE2_RDKIT_REFINED_KIND,
    load_rdkit_stage2_encoder_artifact,
)
from stage2.registry import load_stage2_registry
from stage2.train import (
    load_stage2_encoder_artifact,
    run_stage2_training,
    stage2_training_optimizer_groups,
    joint_stage2_loss,
    task_compensation_scale,
)


# --- Configuration, preparation, training, and artifact contracts ---

TASKS = (
    "simulation/density",
    "simulation/heat_capacity",
    "simulation/heat_of_vaporization",
    "simulation/homo",
    "simulation/lumo",
    "simulation/partial_atomic_charge",
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
    orbital_fields = [
        "SMILES", "ion_role", "provenance_source_file",
        "provenance_source_row",
    ]
    role_rows = {
        "train": (("cation", "[Na+]"), ("anion", "[Cl-]")),
        "valid": (("cation", "C[NH3+]"), ("anion", "C(=O)[O-]")),
    }
    for name, target, values in (
        ("homo", "HOMO_eV", (-9.0, 1.0)),
        ("lumo", "LUMO_eV", (-4.0, 4.0)),
    ):
        fields = [*orbital_fields, target, "source_list"]
        for split, entities in role_rows.items():
            _write_csv(
                root / name / f"{split}.csv",
                fields,
                [
                    {
                        "SMILES": smiles,
                        "ion_role": role,
                        "provenance_source_file": (
                            "simulation/simulated_HOMO+LUMO_PBE_TZVP_"
                            f"{role}s_structured.csv"
                        ),
                        "provenance_source_row": 2,
                        target: values[index] + (index if split == "valid" else 0),
                        "source_list": "simulation",
                    }
                    for index, (role, smiles) in enumerate(entities)
                ],
            )
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
        ("homo", "object_property", "object", "HOMO_eV", "SMILES", "", "molecule", "materialized_csv", ""),
        ("lumo", "object_property", "object", "LUMO_eV", "SMILES", "", "molecule", "materialized_csv", ""),
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
    corpus_metadata = json.loads((corpus / "metadata.json").read_text())
    torch.save({
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "kind": STAGE1_CHECKPOINT_KIND,
        "format_version": STAGE1_CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "config": pretrain.to_dict(),
        "corpus_identity": dict(metadata_identity(
            corpus_metadata, "corpus", context="test Stage 1 corpus"
        )),
    }, checkpoint)
    _stage2_sources(tmp_path)
    return Stage2Config(
        data=Stage2DataConfig(
            data_root=tmp_path,
            task_catalog_path=tmp_path / "task_catalog.csv",
            pretrain_artifacts_dir=corpus,
            artifacts_dir=tmp_path / "stage2_artifacts",
            entity_shard_size=3,
            target_materialization_modes={
                "simulation/simulated_qm_elec_hf":
                    "allow_partial_drop_all_missing"
            },
        ),
        preparation=Stage2PreparationConfig(workers=1, teacher_batch_size=4),
        initialization=Stage2InitializationConfig(checkpoint=checkpoint),
        training=Stage2TrainingConfig(batch_size=2, epochs=2, backbone_frozen_epochs=1, packing_workers=2, packing_prefetch_batches=2, cuda_prefetch_batches=1, log_every_batches=3, device="cpu", amp_dtype="none", refinement_epochs=2),
    )



def test_registry_is_catalog_driven_and_model_independent(tiny_stage2_setup):
    registry = load_stage2_registry(tiny_stage2_setup.data.task_catalog_path)
    assert registry.task_ids == TASKS
    original = registry.registry_hash
    assert registry.registry_hash == original
    assert registry.by_id("simulation/partial_atomic_charge").target_level == "atom"
    assert registry.by_id("simulation/homo").task_id != registry.by_id("simulation/lumo").task_id


def test_stage2_refinement_config_contract(tiny_stage2_setup):
    active = load_stage2_config(Path("configs/v3/stage2/base.yaml"))
    assert active.model.object_layers == 2
    assert active.model.object_ffn_dim == 2048
    assert active.loss.lambda_teacher == 0.0
    assert active.preparation.teacher_batch_size is None
    assert active.training.epochs == 10
    assert active.training.refinement_epochs == 0
    assert active.training.refinement_tasks == ()
    assert active.loss.teacher_weighting == "task_compensated"
    assert active.to_dict()["loss"]["teacher_weighting"] == "task_compensated"

    paths = [
        Path("configs/v1/stage2/base.yaml"),
        *sorted(Path("configs/experiments_v1/stage2").glob("*.yaml")),
    ]
    for path in paths:
        config = load_stage2_config(path)
        assert config.training.epochs == (5 if path == Path("configs/v1/stage2/base.yaml") else 10)
        assert config.training.refinement_epochs == 10
        assert config.training.refinement_tasks == DEFAULT_REFINEMENT_TASKS
        assert config.loss.teacher_weighting == "uncompensated"
        assert "teacher_weighting" not in config.to_dict()["loss"]
        assert config.to_dict()["training"]["refinement_tasks"] == list(
            DEFAULT_REFINEMENT_TASKS
        )

    invalid = active.to_dict()
    invalid["loss"]["teacher_weighting"] = "unknown"
    with pytest.raises(ValueError, match="teacher_weighting"):
        stage2_config_from_dict(invalid)

    registry = load_stage2_registry(tiny_stage2_setup.data.task_catalog_path)
    identity_args = {
        "data_identity": {"hash": "data"},
        "teacher_identity": {"hash": "teacher"},
        "stage1_encoder_identity": {"hash": "stage1"},
        "registry": registry,
        "model_contract": {"kind": "test"},
        "normalized_task_weights": active.normalized_task_weights(registry),
        "math_contract": {"precision": "test"},
        "optimizer_implementation": "single_tensor",
    }
    legacy = replace(
        active,
        loss=replace(active.loss, teacher_weighting="uncompensated"),
    )
    assert build_stage2_training_identity(active, **identity_args)["hash"] != (
        build_stage2_training_identity(legacy, **identity_args)["hash"]
    )

    with pytest.raises(ValueError, match="disabled together"):
        replace(
            tiny_stage2_setup,
            training=replace(tiny_stage2_setup.training, refinement_epochs=0),
        ).validate()
    with pytest.raises(ValueError, match="duplicates"):
        replace(
            tiny_stage2_setup,
            training=replace(
                tiny_stage2_setup.training,
                refinement_tasks=("simulation/homo", "simulation/homo"),
            ),
        ).validate()
    with pytest.raises(ValueError, match="disabled together"):
        replace(
            tiny_stage2_setup,
            training=replace(tiny_stage2_setup.training, refinement_tasks=()),
        ).validate()
    unknown = replace(
        tiny_stage2_setup,
        training=replace(
            tiny_stage2_setup.training,
            refinement_tasks=("simulation/unknown",),
        ),
    )
    unknown.validate()
    with pytest.raises(ValueError, match="unknown"):
        unknown.validate_registry(load_stage2_registry(unknown.data.task_catalog_path))

def test_prepare_v3_task_local_scalers_and_ragged_atoms(tiny_stage2_setup):
    metadata = prepare_stage2_data(tiny_stage2_setup)
    assert metadata["format_version"] == 3
    assert metadata["preparation_contract_version"] == STAGE2_PREPARATION_CONTRACT_VERSION
    assert "model_contract" not in metadata
    assert metadata["summary"]["rows"]["simulation/density"]["train"] == 2
    density = metadata["scalers"]["simulation/density"]["targets"]["density_g/cm^3"]
    assert density["mean"] == pytest.approx(1.5)
    assert density["scale"] == pytest.approx(0.5)
    homo = metadata["scalers"]["simulation/homo"]["targets"]["HOMO_eV"]
    lumo = metadata["scalers"]["simulation/lumo"]["targets"]["LUMO_eV"]
    assert (homo["count"], homo["mean"], homo["scale"]) == pytest.approx(
        (2, -4.0, 5.0)
    )
    assert (lumo["count"], lumo["mean"], lumo["scale"]) == pytest.approx(
        (2, 0.0, 4.0)
    )
    atom = Stage2TaskDataset(tiny_stage2_setup.data.artifacts_dir, "simulation/partial_atomic_charge", "train")
    assert atom.mol_ids == ("mol_train",)
    assert atom.atom_target_offsets.tolist() == [0, 3]
    assert metadata["scalers"]["simulation/partial_atomic_charge"]["targets"]["partial_atomic_charge"]["weighting"] == "molecule_equal"
    audit = list(csv.DictReader((tiny_stage2_setup.data.artifacts_dir / "partial_charge_mapping_audit.csv").open()))
    assert {row["status"] for row in audit} == {"mapped"}

def test_prepare_rejects_orbital_role_provenance_mismatch(tiny_stage2_setup):
    path = tiny_stage2_setup.data.data_root / "stage2/homo/train.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows[0]["ion_role"] = "anion"
    _write_csv(path, list(rows[0]), rows)
    with pytest.raises(ValueError, match="formal-charge mismatch"):
        prepare_stage2_data(tiny_stage2_setup)

def test_data_and_teacher_identity_ignore_stage2_model_contract(tiny_stage2_setup):
    first_data = prepare_stage2_data(tiny_stage2_setup)
    first_teacher = prepare_teacher_cache(tiny_stage2_setup)
    changed = replace(
        tiny_stage2_setup,
        model=replace(
            tiny_stage2_setup.model,
            object_layers=tiny_stage2_setup.model.object_layers + 1,
            object_ffn_dim=tiny_stage2_setup.model.object_ffn_dim * 2,
            dropout=0.2,
        ),
    )

    second_data = prepare_stage2_data(changed)
    second_teacher = prepare_teacher_cache(changed)

    assert second_data["data_signature"] == first_data["data_signature"]
    assert "model_contract" not in second_data
    assert second_teacher["identity"] == first_teacher["identity"]
    assert second_teacher["cache_reused"] is True
    assert "model_contract" not in second_teacher
    teacher_identity = second_teacher["semantic"]["identities"]["teacher"]
    assert set(teacher_identity["payload"]) == {
        "extraction_contract_version", "entity_identity",
        "stage1_encoder_identity",
    }
    assert second_teacher["dtype"] == "float32"
    assert "math_contract" in second_teacher

def test_teacher_identity_binds_stage1_encoding_contract_and_entity_data(tiny_stage2_setup):
    data_metadata = prepare_stage2_data(tiny_stage2_setup)
    loaded = load_stage1_model(
        tiny_stage2_setup.initialization.checkpoint,
        tiny_stage2_setup.data.pretrain_artifacts_dir,
        backbone_dropout=0.0,
    )
    identity = teacher_cache_identity(data_metadata, loaded)
    changed_loaded = replace(
        loaded,
        config=replace(
            loaded.config,
            model=replace(loaded.config.model, n_heads=2),
        ),
    )
    changed_identity = teacher_cache_identity(data_metadata, changed_loaded)
    changed_feature_identity = teacher_cache_identity(
        data_metadata, replace(loaded, artifact_hash="different"),
    )
    changed_data = copy.deepcopy(data_metadata)
    changed_data["semantic"]["identities"]["entity"]["hash"] = "different"
    changed_entity_identity = teacher_cache_identity(changed_data, loaded)

    assert identity != changed_identity
    assert identity != changed_feature_identity
    assert identity != changed_entity_identity
    assert identity["payload"]["stage1_encoder_identity"] == stage1_encoder_identity(loaded)["hash"]

    with torch.no_grad():
        next(loaded.model.smiles_encoder.parameters()).add_(1.0)
    changed_state_identity = teacher_cache_identity(data_metadata, loaded)
    assert identity != changed_state_identity

def test_prepare_worker_count_preserves_atom_semantics(tiny_stage2_setup):
    first_config = replace(
        tiny_stage2_setup,
        data=replace(
            tiny_stage2_setup.data,
            artifacts_dir=tiny_stage2_setup.data.data_root / "stage2_workers_1",
        ),
        preparation=replace(tiny_stage2_setup.preparation, workers=1),
    )
    second_config = replace(
        tiny_stage2_setup,
        data=replace(
            tiny_stage2_setup.data,
            artifacts_dir=tiny_stage2_setup.data.data_root / "stage2_workers_2",
        ),
        preparation=replace(tiny_stage2_setup.preparation, workers=2),
    )
    first_metadata = prepare_stage2_data(first_config)
    second_metadata = prepare_stage2_data(second_config)
    assert first_metadata["summary"] == second_metadata["summary"]
    assert first_metadata["scalers"] == second_metadata["scalers"]
    task = "simulation/partial_atomic_charge"
    first = Stage2TaskDataset(first_config.data.artifacts_dir, task, "train")
    second = Stage2TaskDataset(second_config.data.artifacts_dir, task, "train")
    assert first.mol_ids == second.mol_ids
    assert torch.equal(first.atom_target_offsets, second.atom_target_offsets)
    assert torch.equal(first.atom_target_values, second.atom_target_values)
    assert (
        first_config.data.artifacts_dir / "partial_charge_mapping_audit.csv"
    ).read_text(encoding="utf-8") == (
        second_config.data.artifacts_dir / "partial_charge_mapping_audit.csv"
    ).read_text(encoding="utf-8")





# --- Partial-charge mapping behavior ---

def _write_mol2(path: Path, atoms, bonds) -> None:
    lines = [
        "@<TRIPOS>MOLECULE", "MOL",
        f"{len(atoms)} {len(bonds)} 1 0 0",
        "SMALL", "resp", "@<TRIPOS>ATOM",
    ]
    for index, (name, atom_type, charge) in enumerate(atoms, start=1):
        lines.append(f"{index} {name} 0 0 0 {atom_type} 1 MOL {charge}")
    lines.append("@<TRIPOS>BOND")
    for index, (first, second, kind) in enumerate(bonds, start=1):
        lines.append(f"{index} {first} {second} {kind}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def test_typed_mapping_explicit_h_and_deterministic_automorphism(tmp_path: Path) -> None:
    path = tmp_path / "typed.mol2"
    _write_mol2(
        path,
        [("C1", "c3", 0.2), ("C2", "c3", -0.2), ("H1", "h1", 0.0)],
        [(1, 2, "1"), (1, 3, "du")],
    )
    result = map_partial_charges("CC", parse_mol2(path))
    assert result.bond_match_mode == "typed"
    assert result.unparsed_bond_types == ()
    assert result.mapping_status == "ambiguous"
    assert result.mapping_count_lower_bound == 2
    assert result.charges == pytest.approx((0.2, -0.2))
    from common.atom_targets import MOL2_PREFIX_ALIASES, MOL2_SECTION_ALIASES, parse_mol2_text
    text = path.read_text()
    alias = text.replace("@<TRIPOS>ATOM", "@<TRIPOS>MOLM")
    assert parse_mol2_text(alias) == parse_mol2_text(text)
    assert map_partial_charges("CC", parse_mol2_text(alias)) == result
    for bond_alias in (text.replace("@<TRIPOS>BOND", "@<TRIPOS>MOLD"),
                       alias.replace("@<TRIPOS>BOND", "@<TRIPOS>MOLD")):
        assert parse_mol2_text(bond_alias) == parse_mol2_text(text)
        assert map_partial_charges("CC", parse_mol2_text(bond_alias)) == result
    for typo, section in MOL2_SECTION_ALIASES.items():
        corrected = text.replace(f"@<TRIPOS>{section}", f"@<TRIPOS>{typo}")
        assert parse_mol2_text(corrected) == parse_mol2_text(text)
        assert map_partial_charges("CC", parse_mol2_text(corrected)) == result
        with pytest.raises(ValueError, match="counts do not match"):
            parse_mol2_text(corrected.replace("3 2 1 0 0", "4 2 1 0 0"))
    for prefix in MOL2_PREFIX_ALIASES:
        for source in (text, alias):
            renamed = source.replace("@<TRIPOS>", prefix)
            assert parse_mol2_text(renamed) == parse_mol2_text(text)
            assert map_partial_charges("CC", parse_mol2_text(renamed)) == result
            with pytest.raises(ValueError, match="counts do not match"):
                parse_mol2_text(renamed.replace("3 2 1 0 0", "4 2 1 0 0"))
    with pytest.raises(ValueError, match="ATOM, and BOND sections"):
        parse_mol2_text(text.replace("@<TRIPOS>", "@<UNKNOWN>"))
    with pytest.raises(ValueError, match="ATOM, and BOND sections"):
        parse_mol2_text(text.replace("@<TRIPOS>MOLECULE", "@<TRIPOS>UNKNOWN"))
    with pytest.raises(ValueError, match="ATOM, and BOND sections"):
        parse_mol2_text(text.replace("@<TRIPOS>BOND", "@<TRIPOS>UNKNOWN"))
    with pytest.raises(ValueError, match="ATOM, and BOND sections"):
        parse_mol2_text(text.replace("@<TRIPOS>ATOM", "@<TRIPOS>UNKNOWN"))
    with pytest.raises(ValueError, match="counts do not match"):
        parse_mol2_text(alias.replace("3 2 1 0 0", "4 2 1 0 0"))

def test_unknown_bond_is_auditable_connectivity_fallback(tmp_path: Path) -> None:
    path = tmp_path / "fallback.mol2"
    _write_mol2(path, [("C1", "c3", 0.1), ("O1", "os", -0.1)], [(1, 2, "du")])
    result = map_partial_charges("CO", parse_mol2(path))
    assert result.bond_match_mode == "connectivity_only"
    assert result.unparsed_bond_types == ("du",)
    assert result.bond_fallback_reason == "unparsed_bond_type"

# --- Batch scheduling and scientific loss behavior ---

class _SizedDataset:
    def __init__(self, rows: int) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return self.rows

def test_round_robin_is_complete_deterministic_and_does_not_cycle() -> None:
    datasets = {"a": _SizedDataset(5), "b": _SizedDataset(3), "c": _SizedDataset(1)}
    first = epoch_batch_schedule(datasets, 2, seed=17, epoch=3)  # type: ignore[arg-type]
    second = epoch_batch_schedule(datasets, 2, seed=17, epoch=3)  # type: ignore[arg-type]
    assert [(item.task, item.indices.tolist()) for item in first] == [
        (item.task, item.indices.tolist()) for item in second
    ]
    for task, dataset in datasets.items():
        observed = torch.cat([item.indices for item in first if item.task == task]).tolist()
        assert sorted(observed) == list(range(len(dataset)))
    assert len({item.task for item in first[:len(datasets)]}) == len(datasets)
    assert [item.task for item in first].count("c") == 1

def test_loss_reductions_and_teacher_weighting() -> None:
    predictions = torch.tensor([[2.0, 2.0], [0.0, 2.0]])
    target = torch.zeros_like(predictions)
    macro = masked_target_macro_smooth_l1_loss(
        predictions, target, torch.tensor([[True, True], [False, True]])
    )
    assert macro.item() == pytest.approx(1.5)
    molecule = molecule_equal_smooth_l1_loss(
        torch.tensor([2.0, 2.0, 2.0]), torch.zeros(3),
        torch.ones(3, dtype=torch.bool), torch.tensor([0, 1, 1]), 2,
    )
    assert molecule.item() == pytest.approx(1.5)
    compensation = task_compensation_scale(0.25, 20, 4, 10)
    physics_loss = torch.tensor(2.0)
    teacher_loss = torch.tensor(3.0)
    assert joint_stage2_loss(
        physics_loss,
        teacher_loss,
        compensation=compensation,
        lambda_teacher=0.1,
        teacher_weighting="task_compensated",
    ).item() == pytest.approx(4.6)
    assert joint_stage2_loss(
        physics_loss,
        teacher_loss,
        compensation=compensation,
        lambda_teacher=0.1,
        teacher_weighting="uncompensated",
    ).item() == pytest.approx(4.3)


def test_entity_inputs_order_roles_and_parallel_expert_learning():
    from common.entity_inputs import EntityInputs
    from stage3.model import Stage3SparseModel, GLOBAL, group_owner
    from stage3.data import ResolvedTaskSpec, object_key_from_row
    from stage3.config import Stage3ModelConfig
    spec = ResolvedTaskSpec(task_id="test", target_column="y", identity_columns=(),
        condition_columns=(), system_type="il", materialized_path="", split_strategy="",
        cv_repeat=1, meta_group="g", partner_mode="interaction", primary_slots=("cation","anion"),
        partner_slots=("solute",), enabled=True, task_weight=1., catalog_schema_version=0, provenance={})
    model = Stage3SparseModel(Stage3ModelConfig(global_experts=1, group_experts=1,
        expert_hidden_ratio=.015625, tower_hidden_ratio=.015625, interaction_hidden_ratio=.015625),
        {"test": spec}, 1024, entity_inputs=True)
    values = torch.randn(2, 2, 1241)
    roles = torch.tensor([[0,1],[0,1]])
    primary = EntityInputs.from_slots(values, roles)
    partner = EntityInputs.from_slots(values[:,:1], torch.full((2,1),2))
    observed = []
    handles = [module.register_forward_pre_hook(lambda m, args: observed.append(args[0]))
        for module in (model.l1_global_experts[0], model.l1_group_experts['g'][0])]
    out = model('test',primary,torch.empty(2,0),partner_embedding=partner)
    assert out.predictions.shape == (2,)
    assert torch.equal(observed[0],observed[1]) and observed[0].shape == (2,2490)
    optimizer = torch.optim.SGD(model.parameters(),lr=.1)
    before = {owner: [p.detach().clone() for p in model.parameters_for_owner(owner)] for owner in (GLOBAL,group_owner('g'))}
    out.predictions.square().sum().backward(); optimizer.step()
    for owner in before:
        assert any(not torch.equal(p,v) for p,v in zip(model.parameters_for_owner(owner),before[owner]))
    for handle in handles: handle.remove()
    for role in range(3):
        assert EntityInputs.from_slots(values[:,:1],torch.full((2,1),role)).pack()[0].shape == (2,2490)
    with pytest.raises(ValueError,match='ordered'): EntityInputs.from_slots(values,roles.flip(1))
    with pytest.raises(ValueError,match='role'): EntityInputs.from_slots(values[:,:1],torch.full((2,1),9))
    with pytest.raises(ValueError,match='contract'): EntityInputs.from_slots(values[:,:,:1024],roles)
    with pytest.raises(ValueError,match='mask'): EntityInputs(values,roles,torch.zeros(2,2,dtype=torch.bool)).pack()
    assert object_key_from_row('test',2,{'solute':'[NH4+]'},('solute',),'formal_charge_v1').slots == (('cation','[NH4+]'),)
    assert object_key_from_row('test',2,{'solvent':'[Cl-]'},('solvent',),'formal_charge_v1').slots == (('anion','[Cl-]'),)


def test_entity_home_base_contracts() -> None:
    from stage2.home_config import load_home_recipe
    from stage3.config import load_stage3_config
    from stage3.data import resolve_task_registry
    from stage3.simulation import simulation_tasks

    stage3 = load_stage3_config("configs/v4/stage3/base.yaml")
    resolved = resolve_task_registry(stage3)
    assert len(resolved) == 24 and "experiment/x_co2" not in resolved
    assert set(simulation_tasks(stage3)) == {"simulation/heat_of_vaporization", "simulation/thermal_expansion"}
    enthalpy = resolved["experiment/enthalpy_of_vaporization_or_sublimation"]
    assert enthalpy.condition_columns == ("temperature_K",) and enthalpy.condition_width == 1
    assert resolved["experiment/hydration"].split_strategy == "random"
    assert resolved["experiment/gas_solubility"].partner_slots == ("solute",)
    with pytest.raises(ValueError, match="Selected condition"):
        resolve_task_registry(replace(stage3, tasks={enthalpy.task_id: replace(
            stage3.tasks[enthalpy.task_id], condition_columns=("missing",)
        )}, data=replace(stage3.data, split_strategies={})))
    recipe = load_home_recipe("configs/v4/stage2/base.yaml")
    with pytest.raises(ValueError, match="missing"):
        load_stage2_registry(recipe.stage2.data.task_catalog_path, task_ids=(*recipe.stage2.data.tasks, "simulation/missing"))
    with pytest.raises(ValueError, match="unique"):
        load_stage2_registry(recipe.stage2.data.task_catalog_path, task_ids=(recipe.stage2.data.tasks[0],) * 2)
    with pytest.raises(ValueError, match="task"):
        replace(recipe.stage2, loss=replace(recipe.stage2.loss, task_weights={})).validate()


def test_entity_home_prepare_train_transfer_three_phase_and_predictions(tiny_stage2_setup, tmp_path, monkeypatch):
    variant = "base"
    from stage1.config import AuxiliaryConfig, MaskingConfig
    from stage1.model import build_stage1_model
    from stage2.entity_cache import prepare_frozen_entities
    from stage2.home_artifact import load_home_final
    from stage2.home_config import load_home_recipe
    from stage2.home_train import build_model, train_stage2_home
    from stage3.config import load_stage3_config
    from stage3.data import Stage3TaskDataset, resolve_task_registry, source_path, test_path
    from stage3.prepare import prepare_stage3, load_prepared_stage3
    from stage3.home import build_model_and_store, load_source, train_fold, load_simulation_final
    from stage3.model import GLOBAL, group_owner, private_owner
    from stage3.evaluate import evaluate_checkpoints

    recipe = load_home_recipe(f"configs/v4/stage2/{variant}.yaml")
    stage3 = load_stage3_config(f"configs/v4/stage3/{variant}.yaml")
    stage1_root = tmp_path / "dual_features"
    stage1 = PretrainConfig(
        architecture=ArchitectureConfig(kind="dual_view_v4"),
        data=replace(PretrainConfig().data, stage1_dir=tmp_path / "stage1", artifacts_dir=stage1_root,
                     valid_fraction=0.5, max_smiles_tokens=64, shard_size=4),
        descriptor=DescriptorConfig(mode="full", token_count=1),
        model=ModelConfig(d_model=512, n_heads=8, smiles_layers=1, graph_depth=2,
                          feedforward_dim=32, dropout=0., role_embedding=False),
        masking=MaskingConfig(fusion_only_dropout=True, descriptor_dropout=0.),
        auxiliary=AuxiliaryConfig(simulation_dir=tmp_path / "stage2"),
    )
    prepare_corpus(stage1)
    vocabulary = SmilesTokenizer.load(stage1_root / "tokenizer.json")
    backbone = build_stage1_model(stage1, vocabulary, PreparedCorpusDataset(stage1_root, "train").descriptor_schema)
    checkpoint = tmp_path / "dual.pt"
    stage1_metadata = json.loads((stage1_root / "metadata.json").read_text())
    torch.save({"identity_contract_version": IDENTITY_CONTRACT_VERSION, "kind": STAGE1_CHECKPOINT_KIND,
                "format_version": 4, "model": backbone.state_dict(), "config": stage1.to_dict(),
                "corpus_identity": dict(metadata_identity(stage1_metadata, "corpus", context="test corpus"))}, checkpoint)
    config = replace(recipe.stage2,
        data=replace(recipe.stage2.data, data_root=tmp_path, task_catalog_path=tmp_path / "task_catalog.csv",
                     pretrain_artifacts_dir=stage1_root, artifacts_dir=tmp_path / "v5_stage2"),
        preparation=replace(recipe.stage2.preparation, workers=1),
        initialization=replace(recipe.stage2.initialization, checkpoint=checkpoint),
        model=recipe.stage2.model,
        training=replace(recipe.stage2.training, device="cpu", amp_dtype="none", packing_workers=1),
    )
    recipe = replace(recipe, stage2=config, stage3=replace(recipe.stage3, model=replace(recipe.stage3.model, expert_hidden_ratio=.015625, interaction_hidden_ratio=.015625, film_hidden_ratio=.015625, tower_hidden_ratio=.015625), groups={
        group: replace(spec, expert_hidden_ratio=.015625, phase1=replace(spec.phase1, epochs=1), phase2=replace(spec.phase2, epochs=1))
        for group, spec in recipe.stage3.groups.items()}))
    # Move only the fixture catalog provenance, exactly as the current producer does.
    with config.data.task_catalog_path.open() as handle:
        sim_rows = list(csv.DictReader(handle))
    with Path("data/task_catalog.csv").open() as handle:
        experiment_rows = [row for row in csv.DictReader(handle) if row["task_id"] in stage3.tasks]
    for row in sim_rows:
        if row["task_id"] not in {"simulation/density", "simulation/heat_capacity", "simulation/thermal_expansion", "simulation/heat_of_vaporization", "simulation/transfer_organic"}:
            row["stage"] = "1"
    for row in experiment_rows:
        row["unique_systems"] = "5"
    fields = sorted({key for row in sim_rows + experiment_rows for key in row})
    with config.data.task_catalog_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(sim_rows + experiment_rows)
    prepare_stage2_data(config)
    prepare_frozen_entities(recipe)
    registry = load_artifact_registry(config.data.artifacts_dir)
    initial, _ = build_model(recipe, registry)
    source_backbone = {name: value.clone() for name, value in initial.backbone.state_dict().items()}
    output = tmp_path / "v5_stage2_train"
    train_stage2_home(recipe, output)
    payload, source, _ = load_home_final(output / "stage2_final.pt")
    assert payload["kind"] == "ilume_stage2_entity_home_final_v4" and payload["format_version"] == 4
    assert all(torch.equal(value, source.backbone.state_dict()[name]) for name, value in source_backbone.items())
    assert len(source.registry.tasks) == len(config.data.tasks)
    corrupt = tmp_path / "corrupt_final.pt"
    corrupt.write_bytes((output / "stage2_final.pt").read_bytes() + b"tamper")
    corrupt.with_suffix(".json").write_text((output / "stage2_final.json").read_text())
    with pytest.raises(ValueError, match="SHA"):
        load_home_final(corrupt)
    trained_checkpoint = torch.load(output / "checkpoint_epoch_00010.pt", weights_only=False)
    assert trained_checkpoint["owner_manifest"] == payload["owner_manifest"]
    assert not any("object_encoder" in name or "electronic_structure" in name or "atom_adapter" in name for name in source.state_dict())
    assert not (output / "stage2_encoder.pt").exists()
    from stage2 import load_frozen_stage1_entities
    frozen = load_frozen_stage1_entities(checkpoint, stage1_root)
    assert frozen.encode_slots([FrozenObjectSpec("molecule", (("cation", "[NH4+]"),))])[0].shape == (1,1,1241)
    with pytest.raises(ValueError,match="role"):
        frozen.encode_slots([FrozenObjectSpec("molecule", (("neutral", "[NH4+]"),))])
    with pytest.raises(ValueError,match="retired"):
        load_frozen_object_encoder(output / "stage2_final.pt")
    for task in config.data.tasks:
        spec = registry.by_id(task)
        spec.dataset.split_path(tmp_path, "test").write_bytes(spec.dataset.split_path(tmp_path, "valid").read_bytes())
    stage3 = replace(stage3,
        data=replace(stage3.data, task_catalog=config.data.task_catalog_path, stage3_dir=tmp_path / "experiment", artifacts_dir=tmp_path / "v5_stage3"),
        preparation=replace(stage3.preparation, cache_dir=tmp_path / "object_cache", encoding_batch_size=8),
        initialization=replace(stage3.initialization, stage2_encoder=None, stage1_encoder=checkpoint, stage1_artifacts_dir=stage1_root, stage2_final=output / "stage2_final.pt", simulation_artifacts_dir=config.data.artifacts_dir),
        model=recipe.stage3.model,
        tasks={task: replace(spec, unique_systems=5, phase1_private_epochs=1, phase2_private_epochs=1, phase3_private_epochs=1) for task, spec in stage3.tasks.items()},
        groups={group: replace(spec, expert_hidden_ratio=.015625, phase1=replace(spec.phase1, epochs=1), phase2=replace(spec.phase2, epochs=1)) for group, spec in stage3.groups.items()},
        training=replace(stage3.training, device="cpu", amp_dtype="none", cpu_threads=1, cpu_interop_threads=1,
            three_phase=replace(stage3.training.three_phase, global_scope=replace(stage3.training.three_phase.global_scope, epochs=1),
                private_classes={name: replace(spec, width_ratio=.015625, phase1=replace(spec.phase1, epochs=1), phase2_epochs=1, phase3_epochs=1) for name, spec in stage3.training.three_phase.private_classes.items()})),
    )
    resolved = resolve_task_registry(stage3)
    for spec in resolved.values():
        for fold in range(1, 6):
            directory = {"il": "IL", "il_solute": "IL-solute", "solute_solvent": "solute-solvent"}.get(spec.split_strategy, spec.split_strategy)
            path = stage3.data.stage3_dir / Path(spec.materialized_path).relative_to("stage3") / directory / "cv1" / f"fold{fold}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {slot: {"cation": ["[Na+]", "[K+]", "[Li+]", "[Cs+]", "[Rb+]"],
                          "anion": ["[Cl-]"] * 5, "solute": ["[NH4+]", "CO", "CN", "CCO", "CCN"],
                          "solvent": ["C", "CC", "CCC", "CCCC", "CCCCC"]}[slot][fold - 1] for slot in spec.identity_columns}
            row.update({name: "Liquid | Gas" if name == "phase" else str(280 + fold) for name in spec.condition_columns})
            if spec.task_id == "experiment/enthalpy_of_vaporization_or_sublimation":
                row["phase"] = "Liquid | Gas"
            row[spec.target_column] = str(fold + 0.1)
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
        if spec.task_id != "experiment/hydration":
            test_path(stage3, spec).write_bytes(path.read_bytes())
    prepare_stage3(stage3)
    prepared = load_prepared_stage3(stage3)
    assert prepared["metadata"]["prepared_contract_version"] == 4 and prepared["slots"]["slots"].shape[-1] == 1241
    slot_path = stage3.data.artifacts_dir / "object_slots.pt"
    original_slots = slot_path.read_bytes()
    slot_path.write_bytes(original_slots + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        load_prepared_stage3(stage3)
    slot_path.write_bytes(original_slots)
    enthalpy = "experiment/enthalpy_of_vaporization_or_sublimation"
    dataset = Stage3TaskDataset(stage3.data.artifacts_dir, 1, enthalpy, "train")
    assert dataset.conditions.shape == (4, 1)
    stats = prepared["normalization"]["fold1"][enthalpy]
    assert stats["conditions"]["temperature_K"]["mean"] == pytest.approx(283.5)
    assert "phase" not in stats["conditions"] and "categorical_conditions" not in stats
    model, store, _ = build_model_and_store(stage3, prepared, fold=1, device=torch.device("cpu"), source=load_source(stage3))
    for name, value in payload["shared_state"].items():
        assert torch.equal(model.state_dict()[name], value)
    assert len(model.task_specs) == 26 and model.simulation_atom_adapter is None
    model.set_trainable_owners({GLOBAL})
    model.train()
    assert not model.simulation_backbone.training
    assert all(not p.requires_grad for p in model.simulation_backbone.parameters())
    assert not any("object_encoder" in name for name in model.state_dict())
    assert all(not p.requires_grad for p in model.parameters_for_owner(private_owner("simulation/heat_of_vaporization")))
    model.set_trainable_owners({group_owner("thermophysical"), private_owner("simulation/heat_of_vaporization")})
    # Simulation gradients and updates are PRIVATE-only even with shared owners enabled.
    from stage3.simulation import load_simulation_training_data
    from stage3.gradient_assembly import assemble_owner_gradients
    simulation = load_simulation_training_data(config.data.artifacts_dir, payload, vocabulary, torch.device("cpu"), amp_dtype="none")
    sim_tasks = model.simulation_tasks
    model.set_trainable_owners({GLOBAL, group_owner("thermophysical"), *(private_owner(task) for task in sim_tasks)})
    model.train()
    shared = (*model.parameters_for_owner(GLOBAL), *model.parameters_for_owner(group_owner("thermophysical")))
    before_shared = [p.detach().clone() for p in shared]
    gradients = {}
    for task in sim_tasks:
        gradients[task], _ = simulation.compute_gradient(model, task, torch.arange(len(simulation.train[task])), torch.device("cpu"))
        assert gradients[task] and any(g.abs().sum() > 0 for g in gradients[task].values())
        assert all(model.parameter_ownership()[p] == private_owner(task) for p in gradients[task])
    assembled = assemble_owner_gradients(model, gradients, model.task_specs, {"thermophysical": 1.0})
    assert all(p not in assembled.gradients for p in shared)
    experiment = "experiment/density"
    with_experiment = assemble_owner_gradients(model, {**gradients, experiment: {p: torch.ones_like(p) for p in shared}}, model.task_specs, {"thermophysical": 1.0})
    assert all(torch.equal(with_experiment.gradients[p], torch.ones_like(p)) for p in shared)
    before_private = {task: [p.detach().clone() for p in model.parameters_for_owner(private_owner(task))] for task in sim_tasks}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    for p, g in assembled.gradients.items():
        p.grad = g.to(p.dtype)
    optimizer.step()
    assert all(torch.equal(p, before) for p, before in zip(shared, before_shared))
    for task in sim_tasks:
        assert any(not torch.equal(p, before) for p, before in zip(model.parameters_for_owner(private_owner(task)), before_private[task]))
    model.set_trainable_owners({GLOBAL, group_owner("thermophysical")})
    frozen_gradient, _ = simulation.compute_gradient(model, sim_tasks[0], torch.arange(len(simulation.train[sim_tasks[0]])), torch.device("cpu"))
    assert frozen_gradient == {}
    train_root = tmp_path / "v5_stage3_train"
    rows = train_fold(stage3, 1, output_dir=train_root)
    assert rows[-1]["phase"] == "three_phase_final"
    final = torch.load(train_root / "three_phase_final.pt", weights_only=False)
    assert final["kind"] == "ilume_stage3_entity_home_three_phase_final_v4"
    assert len(final["private_state_hashes"]) == 26
    phase1 = torch.load(train_root / "phase_1" / "checkpoint_epoch_00001.pt", weights_only=False)
    assert all(torch.equal(value, phase1["model"][name]) for name, value in final["model"].items() if name.startswith("simulation_backbone."))
    manifest = model.ownership_manifest()
    assert all(torch.equal(value, phase1["model"][name]) for name, value in final["model"].items()
               if manifest.get(name) == "GLOBAL")
    loaded = load_simulation_final(stage3, train_root, fold=1)
    assert all(torch.equal(value, loaded.state_dict()[name]) for name, value in final["model"].items())
    resumed = train_fold(stage3, 1, output_dir=train_root, resume_from=train_root)
    assert resumed[-1]["phase"] == "three_phase_final"
    assert torch.load(train_root / "three_phase_final.pt", weights_only=False)["model_state_hash"] == final["model_state_hash"]
    metrics = evaluate_checkpoints(stage3, train_root, split="valid", ensemble_folds=False, fold=1, predictions_dir=tmp_path / "predictions")
    assert len(metrics["tasks"]) == 24
    from stage3.evaluate import _reporting_model
    assert _reporting_model(stage3, prepared["metadata"]) == ("ilume", "ILUME")
    if variant == "base":
        from stage1.masking import MultimodalPacker
        from stage2.data import Stage2BatchDescriptor, Stage2DeviceTaskData, Stage2EntityDataset, pack_stage2_batch
        from stage3.simulation_evaluate import evaluate_simulation_checkpoints
        from stage3.simulation_reporting import ENTITY_SCALAR_SIMULATION_TASKS
        # Isolate ensemble arithmetic with five temporary selectors and controlled model offsets.
        ensemble_root = tmp_path / "ensemble"
        for fold in range(1, 6):
            root = ensemble_root / f"fold{fold}"
            root.mkdir(parents=True)
            for filename in ("three_phase_final.pt", "three_phase_final.json"):
                (root / filename).write_bytes((train_root / filename).read_bytes())
        loaded.eval()
        predictions = {}
        entities = Stage2EntityDataset(config.data.artifacts_dir)
        for task in ENTITY_SCALAR_SIMULATION_TASKS:
            data = Stage2TaskDataset(config.data.artifacts_dir, task, "valid")
            packed = pack_stage2_batch(Stage2BatchDescriptor(task, torch.arange(len(data))), {task: data},
                entities, MultimodalPacker(vocabulary), needs_entities=True, include_raw_atom_targets=False, pin_memory=False)
            with torch.no_grad():
                predictions[task] = loaded.predict_simulation(task, packed, Stage2DeviceTaskData.from_dataset(data, torch.device("cpu"))).numpy().reshape(-1)
        def selected_model(_config, _root, *, fold, device):
            result = copy.deepcopy(loaded)
            predict = result.predict_simulation
            result.predict_simulation = lambda task, batch, data: predict(task, batch, data) + fold
            return result
        monkeypatch.setattr("stage3.simulation_evaluate.load_simulation_final", selected_model)
        for split in ("valid", "test"):
            result = evaluate_simulation_checkpoints(stage3, ensemble_root, split=split, predictions_dir=tmp_path / split / "predictions")
            assert tuple(result["tasks"]) == ENTITY_SCALAR_SIMULATION_TASKS
            for task in ENTITY_SCALAR_SIMULATION_TASKS:
                stats = payload["scalers"][task]["targets"][registry.by_id(task).target_columns[0]]
                with (tmp_path / split / "predictions" / (task.replace("/", "__") + ".csv")).open() as handle:
                    actual = [float(row["prediction"]) for row in csv.DictReader(handle)]
                np.testing.assert_allclose(actual, (predictions[task] + 3) * stats["scale"] + stats["mean"], rtol=1e-5, atol=1e-5)
    # Ignored source columns remain SHA-bound; never edit metadata hashes.
    bad = source_path(stage3, resolved[enthalpy], 1)
    bad.write_text(bad.read_text().replace("Liquid | Gas", "Solid | Gas"))
    with pytest.raises(ValueError, match="source hash"):
        load_prepared_stage3(stage3)
    from stage3.data import fit_normalization
    ignored_phase_stats = fit_normalization(stage3, {enthalpy: resolved[enthalpy]}, 2)
    assert set(ignored_phase_stats[enthalpy]["conditions"]) == {"temperature_K"}
