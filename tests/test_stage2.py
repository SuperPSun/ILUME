from __future__ import annotations

import csv

import json

import copy

import shutil

from dataclasses import replace

from pathlib import Path

from unittest.mock import patch

import pytest

import torch

from rdkit import Chem

import scripts.stage2.evaluate as stage2_evaluate_launcher

from common.io import sha256_file

from common.identity import IDENTITY_CONTRACT_VERSION

from stage1.identity import metadata_identity

from stage1.config import (
    ArchitectureConfig,
    GLOBAL_RDKIT_STAGE1_CHECKPOINT_VERSION,
    STAGE1_CHECKPOINT_KIND, STAGE1_CHECKPOINT_VERSION, DataConfig,
    DescriptorConfig, FingerprintConfig, ModelConfig, PretrainConfig,
)

from stage1.data import PreparedCorpusDataset

from stage1.descriptors import DescriptorSchema, rdkit_descriptor_names

from stage1.masking import MultimodalPacker

from stage1.model import EncodedEntityStates, MultimodalPretrainModel, load_stage1_model

from stage1.prepare import prepare_corpus

from stage1.tokenizer import SmilesTokenizer

from stage2.config import (
    DEFAULT_REFINEMENT_TASKS, STAGE2_CHECKPOINT_VERSION,
    Stage2Config, Stage2DataConfig, Stage2InitializationConfig,
    Stage2PreparationConfig, Stage2RepresentationConfig, Stage2TrainingConfig,
    load_stage2_config,
)

from stage2.data import (
    STAGE2_PREPARATION_CONTRACT_VERSION, Stage2BatchDescriptor,
    Stage2DeviceTaskData, Stage2EntityDataset,
    Stage2TaskDataset, load_artifact_registry,
    pack_stage2_batch,
)

from stage2.model import (
    ObjectEncoder, RDKitDescriptorBackbone, RegressionHead, Stage2ObjectModel,
    molecule_equal_smooth_l1_loss,
)

from stage2.evaluate import evaluate_stage2_checkpoints, resolve_checkpoint_path

from stage2.prepare import (
    prepare_stage2_data, prepare_teacher_cache,
    stage1_encoder_identity, teacher_cache_identity,
)

from stage2.registry import load_stage2_registry

from stage2.train import (
    STAGE2_REFINED_VERSION, _batch_output,
    load_stage2_encoder_artifact, run_stage2_training,
)
from stage2.rdkit_train import (
    STAGE2_RDKIT_CHECKPOINT_KIND, STAGE2_RDKIT_ENCODER_KIND,
    STAGE2_RDKIT_REFINED_KIND, load_rdkit_stage2_encoder_artifact,
)

from stage2 import FrozenObjectSpec, load_frozen_object_encoder

from stage2.atom_targets import (
    map_partial_charges, parse_mol2,
)

import numpy as np

from stage2.atom_evaluation import (
    PARTIAL_CHARGE_PREDICTION_FIELDS,
    build_partial_charge_benchmark,
    public_partial_charge_score,
    score_partial_charge_predictions,
    write_partial_charge_predictions,
)

from stage2.atom_targets import PARTIAL_CHARGE_MAPPING_CONTRACT

from stage2.data import epoch_batch_schedule

from stage2.model import (
    masked_target_macro_smooth_l1_loss, molecule_equal_smooth_l1_loss,
)

from stage2.train import task_compensation_scale

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
    loaded = load_stage1_model(tiny_stage2_setup.initialization.checkpoint, tiny_stage2_setup.data.pretrain_artifacts_dir, backbone_dropout=0.0)
    model = Stage2ObjectModel(loaded.model, registry, object_layers=1, object_ffn_dim=32, dropout=0.0)
    assert registry.registry_hash == original
    assert model.model_contract["d_model"] == 16
    assert model.model_contract["regression_head_hidden_dims"] == [16, 8]
    assert model.model_contract["tasks"]["simulation/partial_atomic_charge"]["head_family"] == "atom"


def test_global_rdkit_v2_stage2_width_contract(tiny_stage2_setup) -> None:
    registry = load_stage2_registry(tiny_stage2_setup.data.task_catalog_path)
    loaded = load_stage1_model(
        tiny_stage2_setup.initialization.checkpoint,
        tiny_stage2_setup.data.pretrain_artifacts_dir,
        backbone_dropout=0.0,
    )
    config = replace(
        loaded.config,
        architecture=ArchitectureConfig(kind="global_rdkit_v2"),
        descriptor=DescriptorConfig(mode="full", token_count=1),
    )
    schema = DescriptorSchema.fit(
        np.zeros((2, 217), dtype=np.float64),
        rdkit_descriptor_names(),
        "full",
        1,
    )
    backbone = MultimodalPretrainModel(config, loaded.vocabulary, schema)
    model = Stage2ObjectModel(
        backbone,
        registry,
        object_layers=2,
        object_ffn_dim=backbone.entity_dim * 2,
        dropout=0.0,
    )
    atom_head = model.atom_heads["simulation/partial_atomic_charge"]

    assert (backbone.token_dim, backbone.atom_dim, backbone.entity_dim) == (
        16,
        16,
        32,
    )
    assert model.object_encoder.d_model == 32
    assert model.object_encoder.encoder.layers[0].linear1.out_features == 64
    assert atom_head.object_projection.in_features == 32
    assert atom_head.object_projection.out_features == 16
    assert model.model_contract["representation_kind"] == "cls_rdkit_concat_v2"
    assert model.model_contract["tasks"]["simulation/partial_atomic_charge"] == {
        "topology": "single_entity",
        "head_family": "atom",
        "condition_dim": 0,
        "input_dim": 16,
        "output_dim": 1,
        "atom_dim": 16,
        "object_projection_dim": 16,
        "object_context_dim": 32,
    }


def test_global_rdkit_v2_teacher_cache_uses_entity_embedding(
    tiny_stage2_setup, tmp_path: Path
) -> None:
    legacy = load_stage1_model(
        tiny_stage2_setup.initialization.checkpoint,
        tiny_stage2_setup.data.pretrain_artifacts_dir,
        backbone_dropout=0.0,
    )
    v2_artifacts = tmp_path / "v2_pretrain"
    v2_config = replace(
        legacy.config,
        architecture=ArchitectureConfig(kind="global_rdkit_v2"),
        data=replace(legacy.config.data, artifacts_dir=v2_artifacts),
        descriptor=DescriptorConfig(mode="full", token_count=1),
    )
    prepare_corpus(v2_config)
    vocabulary = SmilesTokenizer.load(v2_artifacts / "tokenizer.json")
    dataset = PreparedCorpusDataset(v2_artifacts, "train")
    backbone = MultimodalPretrainModel(
        v2_config, vocabulary, dataset.descriptor_schema
    )
    corpus_metadata = json.loads((v2_artifacts / "metadata.json").read_text())
    checkpoint = tmp_path / "v2_stage1.pt"
    torch.save(
        {
            "identity_contract_version": IDENTITY_CONTRACT_VERSION,
            "kind": STAGE1_CHECKPOINT_KIND,
            "format_version": GLOBAL_RDKIT_STAGE1_CHECKPOINT_VERSION,
            "model": backbone.state_dict(),
            "config": v2_config.to_dict(),
            "corpus_identity": dict(
                metadata_identity(
                    corpus_metadata, "corpus", context="test v2 Stage 1 corpus"
                )
            ),
        },
        checkpoint,
    )
    stage2_config = replace(
        tiny_stage2_setup,
        data=replace(
            tiny_stage2_setup.data,
            pretrain_artifacts_dir=v2_artifacts,
            artifacts_dir=tmp_path / "v2_stage2_artifacts",
        ),
        initialization=Stage2InitializationConfig(checkpoint=checkpoint),
        model=replace(
            tiny_stage2_setup.model,
            object_ffn_dim=backbone.entity_dim * 2,
        ),
    )

    teacher = prepare_teacher_cache(stage2_config)
    embeddings = torch.load(
        stage2_config.data.artifacts_dir
        / "teachers"
        / teacher["identity"]
        / teacher["locator"]["files"]["embeddings"],
        map_location="cpu",
        weights_only=True,
    )
    assert teacher["embedding_dim"] == backbone.entity_dim == 32
    assert embeddings.shape[1] == 32
    assert (
        teacher["semantic"]["identities"]["teacher"]["payload"][
            "extraction_contract_version"
        ]
        == 3
    )
    training_config = replace(
        stage2_config,
        training=replace(
            stage2_config.training,
            epochs=1,
            backbone_frozen_epochs=0,
            refinement_epochs=1,
        ),
    )
    output = tmp_path / "v2_stage2_train"
    run_stage2_training(training_config, output_dir=output)
    encoder_payload = load_stage2_encoder_artifact(output / "stage2_encoder.pt")
    frozen = load_frozen_object_encoder(output / "stage2_encoder.pt")

    assert encoder_payload["model_contract"]["d_model"] == 32
    assert (
        encoder_payload["model_contract"]["representation_kind"]
        == "cls_rdkit_concat_v2"
    )
    assert frozen.embedding_dim == 32
    assert frozen.encode(
        [FrozenObjectSpec(topology="molecule", slots=(("neutral", "CC"),))]
    ).shape == (1, 32)

def test_stage2_refinement_config_contract(tiny_stage2_setup):
    paths = [
        Path("configs/v1/stage2/base.yaml"),
        *sorted(Path("configs/experiments_v1/stage2").glob("*.yaml")),
    ]
    for path in paths:
        config = load_stage2_config(path)
        assert config.training.epochs == (5 if path == Path("configs/v1/stage2/base.yaml") else 10)
        assert config.training.refinement_epochs == 10
        assert config.training.refinement_tasks == DEFAULT_REFINEMENT_TASKS
        assert config.to_dict()["training"]["refinement_tasks"] == list(
            DEFAULT_REFINEMENT_TASKS
        )

    with pytest.raises(ValueError, match="positive"):
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
    with pytest.raises(ValueError, match="non-empty"):
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

def test_object_encoder_roles_and_dynamic_heads(tiny_stage2_setup):
    registry = load_stage2_registry(tiny_stage2_setup.data.task_catalog_path)
    loaded = load_stage1_model(tiny_stage2_setup.initialization.checkpoint, tiny_stage2_setup.data.pretrain_artifacts_dir, backbone_dropout=0.0)
    model = Stage2ObjectModel(loaded.model, registry, object_layers=1, object_ffn_dim=32, dropout=0.0)
    values = torch.randn(2, 1, 16)
    for role in range(3):
        assert model.encode_object(values, torch.full((2, 1), role)).shape == (2, 16)
    ions = torch.randn(2, 2, 16)
    assert model.encode_object(ions, torch.tensor([[0, 1], [0, 1]])).shape == (2, 16)
    assert model.object_heads["simulation/homo"] is not model.object_heads["simulation/lumo"]

def test_prepare_train_checkpoint_and_encoder_export(tiny_stage2_setup, tmp_path):
    prepare_teacher_cache(tiny_stage2_setup)
    output = tmp_path / "train"
    run_stage2_training(tiny_stage2_setup, output_dir=output)
    assert (output / "checkpoint_epoch_00001.pt").is_file()
    boundary_checkpoint = torch.load(
        output / "checkpoint_epoch_00002.pt", map_location="cpu", weights_only=False
    )
    assert boundary_checkpoint["optimizer"]["state"]
    assert boundary_checkpoint["refinement"]["optimizers"] == {}
    assert boundary_checkpoint["phase"] == "boundary"
    assert boundary_checkpoint["refinement"]["task_updates"] == {
        task: 0 for task in DEFAULT_REFINEMENT_TASKS
    }
    final_checkpoint = torch.load(output / "checkpoint_epoch_00004.pt", map_location="cpu", weights_only=False)
    assert final_checkpoint["format_version"] == STAGE2_CHECKPOINT_VERSION
    assert final_checkpoint["completed_epoch"] == 4
    task_batches = final_checkpoint["task_batches"]
    steps_per_epoch = sum(task_batches.values())
    refinement_steps_per_epoch = sum(
        task_batches[task] for task in DEFAULT_REFINEMENT_TASKS
    )
    assert final_checkpoint["scheduler_geometry"] == {
        "gradient_accumulation_steps": 1,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": 2 * steps_per_epoch,
        "backbone_unfreeze_step": steps_per_epoch,
        "joint_epochs": 2,
        "refinement_epochs": 2,
        "refinement_steps_per_epoch": refinement_steps_per_epoch,
        "total_epochs": 4,
    }
    assert final_checkpoint["refinement"]["task_updates"] == {
        task: 2 * task_batches[task] for task in DEFAULT_REFINEMENT_TASKS
    }
    assert (
        final_checkpoint["refinement"]["shared_state_hash"]
        == boundary_checkpoint["refinement"]["shared_state_hash"]
    )
    assert final_checkpoint["registry_hash"] == load_artifact_registry(tiny_stage2_setup.data.artifacts_dir).registry_hash
    assert final_checkpoint["model_contract"]["object_encoder"] == {
        "layers": tiny_stage2_setup.model.object_layers,
        "ffn_dim": tiny_stage2_setup.model.object_ffn_dim,
        "dropout": tiny_stage2_setup.model.dropout,
    }
    encoder_path = output / "stage2_encoder.pt"
    assert (output / "taskwise_refined.pt").is_file()
    assert (output / "taskwise_refinement.json").is_file()
    refined_payload = torch.load(
        output / "taskwise_refined.pt", map_location="cpu", weights_only=False
    )
    assert refined_payload["format_version"] == STAGE2_REFINED_VERSION
    assert tuple(refined_payload["refined_tasks"]) == DEFAULT_REFINEMENT_TASKS
    assert set(refined_payload["unrefined_tasks"]) == set(TASKS) - set(
        DEFAULT_REFINEMENT_TASKS
    )
    assert set(refined_payload["private_state_hashes"]) == set(
        load_artifact_registry(tiny_stage2_setup.data.artifacts_dir).task_ids
    )
    assert set(refined_payload["selected_tasks"]) == set(DEFAULT_REFINEMENT_TASKS)
    for selection in refined_payload["selected_tasks"].values():
        assert selection["selected_refinement_epoch"] in {0, 1, 2}
        assert [candidate["refinement_epoch"] for candidate in selection["candidates"]] == [0, 1, 2]
    for task in refined_payload["unrefined_tasks"]:
        assert (
            refined_payload["private_state_hashes"][task]
            == boundary_checkpoint["refinement"]["unrefined_task_state_hashes"][task]
        )
    assert resolve_checkpoint_path(output) == output / "checkpoint_epoch_00004.pt"
    assert resolve_checkpoint_path(output, 2) == output / "checkpoint_epoch_00002.pt"
    frozen = load_frozen_object_encoder(encoder_path, device="cpu")
    assert not frozen.backbone.training
    assert not frozen.object_encoder.training
    assert not any(
        parameter.requires_grad
        for module in (frozen.backbone, frozen.object_encoder)
        for parameter in module.parameters()
    )
    encoded = frozen.encode(
        (
            FrozenObjectSpec("molecule", (("neutral", "CC"),)),
            FrozenObjectSpec("molecule", (("neutral", "O"),)),
        )
    )
    assert encoded.shape == (2, 16)
    assert encoded.dtype == torch.float32
    il_encoded = frozen.encode(
        (FrozenObjectSpec("il", (("cation", "[Na+]"), ("anion", "[Cl-]"))),)
    )
    assert il_encoded.shape == (1, 16)
    encoder = load_stage2_encoder_artifact(encoder_path)
    assert encoder["kind"] == "ilume_stage2_encoder"
    assert not any("head" in key for key in encoder["stage1_backbone"])
    assert set(encoder) >= {"stage1_backbone", "object_encoder", "model_contract", "state_hashes", "provenance"}
    assert encoder["provenance"]["stage2_checkpoint_hash"] == sha256_file(
        output / "checkpoint_epoch_00002.pt"
    )
    resume_output = tmp_path / "resume"
    resume_output.mkdir()
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    boundary_rows = [row for row in rows if int(row.get("epoch", 0)) <= 2]
    (resume_output / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in boundary_rows) + "\n",
        encoding="utf-8",
    )
    resume_config = replace(
        tiny_stage2_setup,
        training=replace(
            tiny_stage2_setup.training,
            packing_workers=1,
            packing_prefetch_batches=1,
            log_every_batches=1,
        ),
    )
    assert resume_config.experiment_dict() == tiny_stage2_setup.experiment_dict()
    run_stage2_training(
        resume_config,
        output_dir=resume_output,
        resume_from=output / "checkpoint_epoch_00002.pt",
    )
    resumed = torch.load(resume_output / "checkpoint_epoch_00004.pt", map_location="cpu", weights_only=False)
    assert resumed["scheduler_geometry"]["gradient_accumulation_steps"] == 1
    assert resumed["refinement"]["task_updates"] == {
        task: 2 * task_batches[task] for task in DEFAULT_REFINEMENT_TASKS
    }
    assert (resume_output / "stage2_encoder.pt").is_file()
    resumed_refined = torch.load(
        resume_output / "taskwise_refined.pt", map_location="cpu", weights_only=False
    )
    assert resumed_refined["model_state_hash"] == refined_payload["model_state_hash"]

    mid_resume_output = tmp_path / "resume_mid_refinement"
    mid_resume_output.mkdir()
    mid_rows = [row for row in rows if int(row.get("epoch", 0)) <= 3]
    (mid_resume_output / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in mid_rows) + "\n",
        encoding="utf-8",
    )
    run_stage2_training(
        resume_config,
        output_dir=mid_resume_output,
        resume_from=output / "checkpoint_epoch_00003.pt",
    )
    mid_resumed_refined = torch.load(
        mid_resume_output / "taskwise_refined.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert mid_resumed_refined["model_state_hash"] == refined_payload["model_state_hash"]


def test_no_stage1_rdkit_stage2_prepare_train_and_frozen_export(
    tiny_stage2_setup, tmp_path: Path,
) -> None:
    task_weights = dict(tiny_stage2_setup.loss.task_weights)
    task_weights.pop("simulation/partial_atomic_charge")
    config = replace(
        tiny_stage2_setup,
        data=replace(
            tiny_stage2_setup.data,
            pretrain_artifacts_dir=None,
            artifacts_dir=tmp_path / "rdkit_stage2_artifacts",
        ),
        initialization=Stage2InitializationConfig(checkpoint=None),
        preparation=replace(
            tiny_stage2_setup.preparation,
            teacher_batch_size=None,
        ),
        representation=Stage2RepresentationConfig(
            kind="rdkit_2d_mlp",
            descriptor_family="rdkit_2d",
            raw_width=217,
            hidden_dim=1024,
            output_dim=512,
            activation="gelu",
            dropout=0.10,
            normalization="layernorm",
            learning_rate=3.0e-5,
            unsupported_tasks=("simulation/partial_atomic_charge",),
        ),
        model=replace(
            tiny_stage2_setup.model,
            object_layers=1,
            object_ffn_dim=32,
            dropout=0.0,
        ),
        loss=replace(
            tiny_stage2_setup.loss,
            lambda_teacher=0.0,
            task_weights=task_weights,
        ),
        training=replace(
            tiny_stage2_setup.training,
            epochs=1,
            backbone_frozen_epochs=0,
            packing_workers=1,
            packing_prefetch_batches=1,
            refinement_epochs=1,
            refinement_tasks=(
                "simulation/heat_of_vaporization",
                "simulation/homo",
                "simulation/lumo",
            ),
        ),
    )
    config.validate()
    with patch("stage2.prepare.load_stage1_model") as stage1_model_loader, patch(
        "stage2.prepare.load_stage1_feature_inputs"
    ) as stage1_feature_loader:
        prepared = prepare_teacher_cache(config)
    stage1_model_loader.assert_not_called()
    stage1_feature_loader.assert_not_called()

    metadata = json.loads(
        (config.data.artifacts_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert prepared["task_count"] == 8
    assert metadata["descriptor_contract"]["raw_width"] == 217
    assert metadata["descriptor_contract"]["fit_occurrences"] == 17
    assert metadata["descriptor_contract"]["retained_width"] <= 217
    def keys(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                yield str(key).lower()
                yield from keys(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from keys(nested)

    metadata_keys = tuple(keys(metadata))
    assert not any(
        key.startswith("stage1_") or key.startswith("teacher_")
        for key in metadata_keys
    )
    assert load_artifact_registry(config.data.artifacts_dir).task_ids == tuple(
        task for task in TASKS if task != "simulation/partial_atomic_charge"
    )

    output = tmp_path / "rdkit_stage2_train"
    run_stage2_training(config, output_dir=output)
    boundary = torch.load(
        output / "checkpoint_epoch_00001.pt",
        map_location="cpu",
        weights_only=False,
    )
    final = torch.load(
        output / "checkpoint_epoch_00002.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert boundary["kind"] == STAGE2_RDKIT_CHECKPOINT_KIND
    assert final["refinement"]["shared_state_hash"] == boundary["refinement"]["shared_state_hash"]
    assert "teacher_cache_identity" not in final
    assert "teacher_embeddings_hash" not in final
    refined = torch.load(
        output / "taskwise_refined.pt", map_location="cpu", weights_only=False
    )
    assert refined["kind"] == STAGE2_RDKIT_REFINED_KIND

    encoder_path = output / "stage2_encoder.pt"
    encoder = load_rdkit_stage2_encoder_artifact(encoder_path)
    assert encoder["kind"] == STAGE2_RDKIT_ENCODER_KIND
    with pytest.raises(ValueError, match="Unsupported Stage 2 encoder"):
        load_stage2_encoder_artifact(encoder_path)
    frozen = load_frozen_object_encoder(encoder_path, device="cpu")
    assert isinstance(frozen.descriptor_encoder, RDKitDescriptorBackbone)
    assert not any(
        parameter.requires_grad
        for module in (frozen.descriptor_encoder, frozen.object_encoder)
        for parameter in module.parameters()
    )
    encoded = frozen.encode(
        (
            FrozenObjectSpec("molecule", (("neutral", "CC"),)),
            FrozenObjectSpec("molecule", (("neutral", "O"),)),
        )
    )
    assert encoded.shape == (2, 512)
    for task in (
        "heat_of_vaporization",
        "homo",
        "lumo",
    ):
        root = config.data.data_root / "stage2" / task
        shutil.copy(root / "valid.csv", root / "test.csv")
    evaluation = evaluate_stage2_checkpoints(config, output)
    reporting = evaluation["reporting"]
    assert reporting["model_id"] == "rdkit_2d_stage2"
    assert reporting["model_display_name"] == "RDKit 2D MLP + Stage2"
    assert reporting["capabilities"] == {
        "stage2_core_physics": "supported",
        "stage2_partial_charge": "unsupported",
        "stage2_physics_full": "unsupported",
    }
    assert reporting["benchmarks"]["stage2_core_physics"]["status"] == "complete"

    resumed_output = tmp_path / "rdkit_stage2_resumed"
    resumed_output.mkdir()
    boundary_rows = [
        row
        for row in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if int(json.loads(row).get("epoch", 0)) <= 1
    ]
    (resumed_output / "metrics.jsonl").write_text(
        "\n".join(boundary_rows) + "\n", encoding="utf-8"
    )
    run_stage2_training(
        config,
        output_dir=resumed_output,
        resume_from=output / "checkpoint_epoch_00001.pt",
    )
    resumed_refined = torch.load(
        resumed_output / "taskwise_refined.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert resumed_refined["model_state_hash"] == refined["model_state_hash"]

def test_stage2_evaluate_defaults_to_refined_and_rejects_removed_flag() -> None:
    parser = stage2_evaluate_launcher._build_parser()
    parsed = parser.parse_args([
        "--config", "base.yaml", "--checkpoint-dir", "train",
        "--output", "evaluate",
    ])
    assert parsed.checkpoint_epoch is None
    parsed_epoch = parser.parse_args([
        "--config", "base.yaml", "--checkpoint-dir", "train",
        "--checkpoint-epoch", "5", "--output", "evaluate",
    ])
    assert parsed_epoch.checkpoint_epoch == 5
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--config", "base.yaml", "--checkpoint-dir", "train",
            "--taskwise-refined", "--output", "evaluate",
        ])

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

def test_unknown_bond_is_auditable_connectivity_fallback(tmp_path: Path) -> None:
    path = tmp_path / "fallback.mol2"
    _write_mol2(path, [("C1", "c3", 0.1), ("O1", "os", -0.1)], [(1, 2, "du")])
    result = map_partial_charges("CO", parse_mol2(path))
    assert result.bond_match_mode == "connectivity_only"
    assert result.unparsed_bond_types == ("du",)
    assert result.bond_fallback_reason == "unparsed_bond_type"

# --- Partial-charge evaluation contract ---

def _mol2(path: Path, atoms, bonds, *, valid: bool = True) -> None:
    lines = ["@<TRIPOS>MOLECULE", "MOL", f"{len(atoms)} {len(bonds)} 1 0 0", "SMALL", "resp", "@<TRIPOS>ATOM"]
    for index, (name, kind, charge) in enumerate(atoms, start=1):
        lines.append(f"{index} {name} 0 0 0 {kind} 1 MOL {charge}")
    if valid:
        lines.append("@<TRIPOS>BOND")
        for index, (first, second, kind) in enumerate(bonds, start=1):
            lines.append(f"{index} {first} {second} {kind}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _benchmark_files(root: Path) -> tuple[Path, Path]:
    structures = root / "structures"
    structures.mkdir(parents=True)
    definitions = {
        "unique": (
            [("C1", "c3", 0.1), ("C2", "c3", 0.1), ("O1", "o", -0.2)],
            [(1, 2, "1"), (2, 3, "1")],
            True,
        ),
        "ambiguous": (
            [("C1", "c3", 0.3), ("C2", "c3", -0.3)],
            [(1, 2, "1")],
            True,
        ),
        "fallback": (
            [("C1", "c3", 0.1), ("N1", "n3", -0.1)],
            [(1, 2, "du")],
            True,
        ),
        "excluded": (
            [("C1", "c3", 0.0)],
            [],
            False,
        ),
    }
    smiles = {"unique": "CCO", "ambiguous": "CC", "fallback": "CN", "excluded": "C"}
    with (structures / "structure_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("mol_id", "relative_path", "format", "size_bytes", "sha256", "referenced_by_charge"),
        )
        writer.writeheader()
        for mol_id, (atoms, bonds, valid) in definitions.items():
            path = structures / f"{mol_id}.mol2"
            _mol2(path, atoms, bonds, valid=valid)
            writer.writerow(
                {
                    "mol_id": mol_id,
                    "relative_path": path.name,
                    "format": "mol2",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "referenced_by_charge": "true",
                }
            )
    test = root / "test.csv"
    with test.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("mol_id", "SMILES", "role", "formal_charge", "source_list"),
        )
        writer.writeheader()
        for mol_id in definitions:
            writer.writerow(
                {
                    "mol_id": mol_id,
                    "SMILES": smiles[mol_id],
                    "role": "neutral",
                    "formal_charge": 0,
                    "source_list": "simulation",
                }
            )
    return test, structures / "structure_manifest.csv"

def _stats(scale: float = 2.0) -> dict[str, object]:
    return {"mean": 0.0, "scale": scale, "weighting": "molecule_equal"}

def test_partial_charge_scorer_requires_exact_coverage_and_molecule_macro(tmp_path: Path) -> None:
    test, manifest = _benchmark_files(tmp_path)
    benchmark = build_partial_charge_benchmark(test, manifest, _stats())
    predictions = {
        molecule.mol_id: np.asarray(molecule.target_charges) + index
        for index, molecule in enumerate(benchmark.evaluated, start=1)
    }
    score = score_partial_charge_predictions(benchmark, predictions)
    assert score["status"] == "complete"
    assert score["primary"]["molecule_macro_mae"] == pytest.approx(2.0)
    assert score["primary"]["molecule_macro_normalized_mae"] == pytest.approx(1.0)
    assert score["atom_micro"]["mae"] == pytest.approx(13 / 7)
    assert score["atom_micro"]["mae"] != pytest.approx(
        score["primary"]["molecule_macro_mae"]
    )
    assert score["subsets"]["all_mapped"]["molecule_count"] == 3
    assert score["subsets"]["unique"]["molecule_count"] == 2
    assert score["subsets"]["ambiguous"]["molecule_count"] == 1
    assert score["subsets"]["typed"]["molecule_count"] == 2
    assert score["subsets"]["connectivity_only"]["molecule_count"] == 1

    incomplete = score_partial_charge_predictions(
        benchmark, {"unique": predictions["unique"], "unknown": [0.0]}
    )
    assert incomplete["status"] == "incomplete"
    assert incomplete["primary"] is None
    assert incomplete["coverage"]["missing_prediction_count"] == 2
    assert incomplete["coverage"]["extra_prediction_count"] == 1

    wrong_length = dict(predictions)
    wrong_length["unique"] = [float("nan")]
    invalid = score_partial_charge_predictions(benchmark, wrong_length)
    assert invalid["status"] == "incomplete"
    assert invalid["coverage"]["invalid_prediction_count"] == 1

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

def test_loss_reductions_and_teacher_independence() -> None:
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
    assert (compensation * torch.tensor(2.0) + 0.1 * torch.tensor(3.0)).item() == pytest.approx(4.3)
