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

@pytest.mark.parametrize("final_epoch", [8, 10, 12])
@pytest.mark.parametrize("architecture", ["global_rdkit_v2", "dual_view_v4"])
def test_full_home_final_roundtrip_and_transfer(tiny_stage2_setup, tmp_path, monkeypatch, final_epoch, architecture):
    from stage1.masking import MultimodalPacker
    from stage2.data import Stage2BatchDescriptor, Stage2DeviceTaskData, Stage2EntityDataset, pack_stage2_batch
    from stage2.home_artifact import load_home_final
    from stage2.home_config import HomeRecipe, load_home_recipe
    from stage2.home_contract import state_hash, transferable_state
    from stage2.home_train import _export, build_model, training_identity

    base = load_home_recipe("configs/v3/stage2/base.yaml")
    stage1_root = tmp_path / "global_pretrain"
    stage1_config = PretrainConfig(
        architecture=ArchitectureConfig(kind=architecture),
        data=replace(
            PretrainConfig().data, stage1_dir=tmp_path / "stage1",
            artifacts_dir=stage1_root, valid_fraction=0.5,
            max_smiles_tokens=64, shard_size=4,
        ),
        descriptor=DescriptorConfig(mode="full", token_count=1),
        model=ModelConfig(
            d_model=16, n_heads=4, smiles_layers=1, graph_depth=2,
            descriptor_hidden_dim=32, descriptor_blocks=1,
            fusion_layers=1, feedforward_dim=32, dropout=0.0,
        ),
    )
    dual_view = architecture == "dual_view_v4"
    if dual_view:
        from stage1.config import AuxiliaryConfig, MaskingConfig
        stage1_config = replace(
            stage1_config, model=replace(stage1_config.model, role_embedding=False),
            masking=MaskingConfig(fusion_only_dropout=True, descriptor_dropout=0.),
            auxiliary=AuxiliaryConfig(simulation_dir=tmp_path / "stage2"),
        )
    prepare_corpus(stage1_config)
    vocabulary = SmilesTokenizer.load(stage1_root / "tokenizer.json")
    from stage1.model import build_stage1_model
    stage1_model = build_stage1_model(
        stage1_config, vocabulary, PreparedCorpusDataset(stage1_root, "train").descriptor_schema,
    )
    stage1_checkpoint = tmp_path / "global_stage1.pt"
    stage1_metadata = json.loads((stage1_root / "metadata.json").read_text())
    torch.save({
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "kind": STAGE1_CHECKPOINT_KIND,
        "format_version": stage1_config.checkpoint_version,
        "model": stage1_model.state_dict(), "config": stage1_config.to_dict(),
        "corpus_identity": dict(metadata_identity(stage1_metadata, "corpus", context="test corpus")),
    }, stage1_checkpoint)
    config = replace(
        tiny_stage2_setup,
        data=replace(tiny_stage2_setup.data, pretrain_artifacts_dir=stage1_root, artifacts_dir=tmp_path / "home_data"),
        initialization=replace(tiny_stage2_setup.initialization, checkpoint=stage1_checkpoint),
        model=replace(tiny_stage2_setup.model, object_ffn_dim=64, dropout=0.0),
        loss=replace(tiny_stage2_setup.loss, lambda_teacher=0.0),
        training=replace(
            tiny_stage2_setup.training, batch_size=256, epochs=final_epoch,
            backbone_frozen_epochs=0, refinement_epochs=0, refinement_tasks=(),
        ),
    )
    recipe = HomeRecipe(config, base.stage3, 256, final_epoch, "pretrained", None, freeze_stage1=dual_view)
    prepared = prepare_stage2_data(config)
    if dual_view:
        from stage2.entity_cache import prepare_frozen_entities
        prepare_frozen_entities(recipe)
    registry = load_artifact_registry(config.data.artifacts_dir)
    model, _ = build_model(recipe, registry)
    model.eval()
    if dual_view:
        assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
        assert model.object_encoder.input_dim == 249
        frozen_state = {key: value.clone() for key, value in model.backbone.state_dict().items()}
        model.train()
        assert not model.backbone.training
        entities = Stage2EntityDataset(config.data.artifacts_dir)
        dataset = Stage2TaskDataset(config.data.artifacts_dir, "simulation/density", "train")
        packed = pack_stage2_batch(
            Stage2BatchDescriptor(dataset.spec.task_id, torch.arange(len(dataset))),
            {dataset.spec.task_id: dataset}, entities, MultimodalPacker(vocabulary),
            needs_entities=True, include_raw_atom_targets=False, pin_memory=False,
        )
        optimizer = torch.optim.AdamW(tuple(model.object_encoder.parameters()) + model.home_parameters(), lr=1e-4)
        model.predict(dataset.spec.task_id, packed, Stage2DeviceTaskData.from_dataset(dataset, torch.device("cpu"))).square().mean().backward()
        assert model.object_encoder.input_projection[0].weight.grad.abs().sum() > 0
        optimizer.step()
        assert all(parameter.grad is None for parameter in model.backbone.parameters())
        assert all(torch.equal(value, model.backbone.state_dict()[key]) for key, value in frozen_state.items())
        model.zero_grad(set_to_none=True)
        model.eval()
    output = tmp_path / "home_train"
    output.mkdir()
    checkpoint_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    checkpoint_path = output / f"checkpoint_epoch_{final_epoch:05d}.pt"
    torch.save({"model": checkpoint_state}, checkpoint_path)
    data_identity = prepared["semantic"]["identities"]["data"]
    _export(output, recipe, model, registry, training_identity(recipe, data_identity, {}), data_identity)
    payload, reloaded, vocabulary = load_home_final(output / "stage2_final.pt")
    assert json.loads((output / "stage2_final.json").read_text())["fixed_final_epoch"] == final_epoch
    assert payload["checkpoint_sha256"] == sha256_file(checkpoint_path)
    assert payload["full_model_state"].keys() == checkpoint_state.keys()
    assert all(torch.equal(value, payload["full_model_state"][name]) for name, value in checkpoint_state.items())
    assert state_hash(transferable_state(reloaded.home)) == payload["shared_state_hash"]
    entities = Stage2EntityDataset(config.data.artifacts_dir)
    for task in ("simulation/heat_of_vaporization", "simulation/partial_atomic_charge"):
        dataset = Stage2TaskDataset(config.data.artifacts_dir, task, "valid")
        packed = pack_stage2_batch(
            Stage2BatchDescriptor(task, torch.arange(len(dataset))), {task: dataset},
            entities, MultimodalPacker(vocabulary), needs_entities=True,
            include_raw_atom_targets=True, pin_memory=False,
        )
        task_data = Stage2DeviceTaskData.from_dataset(dataset, torch.device("cpu"))
        with torch.no_grad():
            assert torch.equal(model.predict(task, packed, task_data), reloaded.predict(task, packed, task_data))
    checkpoint_path.unlink()
    load_home_final(output / "stage2_final.pt")
    from stage3.config import load_stage3_config
    from stage3.home import load_source

    stage3_config = load_stage3_config("configs/v3/stage3/base.yaml")
    stage3_config = replace(stage3_config, initialization=replace(
        stage3_config.initialization,
        stage2_final=output / "stage2_final.pt",
        stage2_encoder=output / "stage2_encoder.pt",
        representation_contract="dual_view_v4" if dual_view else "object_v3",
    ))
    assert load_source(stage3_config)["shared_state_hash"] == payload["shared_state_hash"]
    with pytest.raises(ValueError, match="incompatible"):
        load_source(replace(stage3_config, model=replace(stage3_config.model, global_experts=3)))
    from stage2.home_contract import load_transferable_state, source_task_specs
    from stage3.model import group_owner, private_owner
    from stage3.simulation import (
        SIMULATION_TASKS, SimulationObjectPhase1Model,
        copy_simulation_owners, load_simulation_training_data,
    )

    source_specs = source_task_specs(registry)
    combined = SimulationObjectPhase1Model(
        stage3_config.model,
        {**{task: source_specs[task] for task in SIMULATION_TASKS},
         "experiment/transfer_organic": replace(
             source_specs["simulation/transfer_organic"], task_id="experiment/transfer_organic",
         )},
        reloaded.backbone.entity_dim,
        group_configs=stage3_config.groups,
        object_encoder=reloaded.object_encoder, source_model=reloaded,
    )
    copy_simulation_owners(combined, reloaded)
    load_transferable_state(combined, payload["shared_state"], payload["shared_state_hash"])
    combined.eval()
    for task in ("simulation/heat_of_vaporization", "simulation/partial_atomic_charge"):
        dataset = Stage2TaskDataset(config.data.artifacts_dir, task, "valid")
        packed = pack_stage2_batch(
            Stage2BatchDescriptor(task, torch.arange(len(dataset))), {task: dataset},
            entities, MultimodalPacker(vocabulary), needs_entities=True,
            include_raw_atom_targets=True, pin_memory=False,
        )
        task_data = Stage2DeviceTaskData.from_dataset(dataset, torch.device("cpu"))
        with torch.no_grad():
            assert torch.equal(
                combined.predict_simulation(task, packed, task_data),
                reloaded.predict(task, packed, task_data),
            )
    simulation_data = load_simulation_training_data(
        config.data.artifacts_dir, payload, vocabulary, torch.device("cpu"),
        amp_dtype="fp32",
    )
    simulation_validation = simulation_data.validate_tasks(
        combined, SIMULATION_TASKS, torch.device("cpu"),
    )
    assert set(simulation_validation["simulation_tasks"]) == set(SIMULATION_TASKS)
    assert all(
        torch.isfinite(torch.tensor(row["normalized_mae"]))
        for row in simulation_validation["simulation_tasks"].values()
    )
    combined.set_trainable_owners({
        group_owner("electronic_structure"), private_owner("simulation/partial_atomic_charge"),
    })
    gradients, loss = simulation_data.compute_gradient(
        combined, "simulation/partial_atomic_charge", torch.arange(1), torch.device("cpu"),
    )
    assert torch.isfinite(torch.tensor(loss))
    assert any(parameter in gradients for parameter in combined.simulation_atom_adapter.parameters())
    import random
    from stage3.three_phase import _OwnerScheduler, _joint_epoch, _optimizer, _task_epoch

    tasks = ("simulation/homo", "simulation/lumo", "simulation/partial_atomic_charge")
    owners = (group_owner("electronic_structure"), *(private_owner(task) for task in tasks))
    combined.set_trainable_owners(owners)
    optimizer = _optimizer(combined, stage3_config, {owner: 1e-4 for owner in owners})
    recipes = {
        owner.label: {"nominal_lr": 1e-4, "terminal_lr": 5e-5, "actual_update_budget": 1}
        for owner in owners
    }
    scheduler = _OwnerScheduler(optimizer, recipes)
    losses, _ = _joint_epoch(
        model=combined, tasks=tasks, epoch=1, phase_seed=42,
        steps_per_epoch=1, allocation={task: 1 for task in tasks},
        counts={task: 1 for task in tasks}, train_data={},
        representations=torch.empty(0), normalizations={},
        config=stage3_config, device=torch.device("cpu"),
        optimizer=optimizer, scheduler=scheduler,
        registry=combined.task_specs,
        group_weights={group: item.group_weight for group, item in stage3_config.groups.items()},
        task_order_rng=random.Random(42), simulation_data=simulation_data,
    )
    assert set(losses) == set(tasks)
    assert set(scheduler.updates.values()) == {1}
    charge_owner = private_owner("simulation/partial_atomic_charge")
    group_before = {
        name: value.clone() for name, value in combined.state_dict().items()
        if combined.ownership_manifest().get(name) == group_owner("electronic_structure").label
    }
    combined.set_trainable_owners((charge_owner,))
    optimizer = _optimizer(combined, stage3_config, {charge_owner: 1e-4})
    scheduler = _OwnerScheduler(optimizer, {charge_owner.label: recipes[charge_owner.label]})
    _task_epoch(
        model=combined, task="simulation/partial_atomic_charge", epoch=1,
        phase_seed=43, allocation=1, count=1, dataset=None,
        representations=torch.empty(0), normalization=None,
        config=stage3_config, device=torch.device("cpu"),
        optimizer=optimizer, scheduler=scheduler, simulation_data=simulation_data,
    )
    assert scheduler.updates[charge_owner.label] == 1
    assert all(torch.equal(combined.state_dict()[name], value) for name, value in group_before.items())
    from common.identity import semantic_identity, tensor_state_hash
    from common.training import canonical_json_sha256
    from stage3.home import build_model_and_store, load_simulation_final, source_plan
    from stage3.identity import build_stage3_training_identity
    from stage3.simulation import extend_simulation_plan
    from stage3.train import build_resolved_training_plan

    stage3_config = replace(stage3_config, initialization=replace(
        stage3_config.initialization, simulation_artifacts_dir=config.data.artifacts_dir,
    ))
    prepared_identity = semantic_identity("tiny-prepared", {})
    source_encoder = torch.load(output / "stage2_encoder.pt", map_location="cpu", weights_only=False)
    prepared_stage3 = {
        "registry": {
            "experiment/density": replace(source_specs["simulation/heat_of_vaporization"], task_id="experiment/density"),
            "experiment/transfer_organic": replace(source_specs["simulation/transfer_organic"], task_id="experiment/transfer_organic"),
        },
        "slots": {
            "slots": torch.zeros((2, 2, reloaded.object_encoder.input_dim)),
            "roles": torch.tensor([[0, 1], [0, 1]]),
            "counts": torch.tensor([2, 2]),
        },
        "metadata": {
            "kind": "ilume_stage3_sparse_data",
            "stage2_encoder_sha256": sha256_file(output / "stage2_encoder.pt"),
            "source_encoder_state_hashes": source_encoder["state_hashes"],
            "source_stage1_checkpoint_sha256": sha256_file(stage1_checkpoint),
            "object_slots_sha256": "tiny-slots",
            "semantic": {"identities": {
                "prepared": prepared_identity,
                "stage2_encoder": semantic_identity("tiny-encoder", {}),
            }},
        },
    }
    inference_source = load_source(stage3_config)
    inference_model, inference_store, loaded_names = build_model_and_store(
        stage3_config, prepared_stage3, fold=1, device=torch.device("cpu"), source=inference_source,
    )
    from stage3.object_phase1 import build_object_phase1_model

    experimental_model, _ = build_object_phase1_model(
        stage3_config, prepared_stage3, fold=1, device=torch.device("cpu"),
    )
    load_transferable_state(experimental_model, payload["shared_state"], payload["shared_state_hash"])
    assert all(
        torch.equal(inference_model.state_dict()[name], value)
        for name, value in experimental_model.state_dict().items()
    )
    assert inference_model.task_specs["simulation/heat_of_vaporization"].task_weight == 0.1
    assert inference_model.task_specs["simulation/thermal_expansion"].task_weight == 0.1
    assert all(inference_model.task_specs[task].task_weight == 1.0 for task in (
        "simulation/homo", "simulation/lumo", "simulation/partial_atomic_charge",
    ))
    from stage3.gradient_assembly import assemble_owner_gradients

    group_parameters = inference_model.parameters_for_owner(group_owner("thermophysical"))
    raw = {
        task: {parameter: torch.full_like(parameter, value) for parameter in group_parameters}
        for task, value in (("experiment/density", 1.0), ("simulation/heat_of_vaporization", 3.0))
    }
    weighted = assemble_owner_gradients(
        inference_model, raw, inference_model.task_specs,
        {group: item.group_weight for group, item in stage3_config.groups.items()},
    )
    assert all(torch.allclose(value, torch.full_like(value, 1.3 / 1.1)) for value in weighted.gradients.values())
    experimental = tuple(prepared_stage3["registry"])
    normalization = {task: {"target": {"mean": 0.0, "scale": 1.0}} for task in experimental}
    plan = build_resolved_training_plan(
        stage3_config, 1, inference_model, {task: range(2) for task in experimental},
        experimental, prepared_stage3, {}, normalization,
    )
    plan["stage2_pretraining"] = source_plan(stage3_config, inference_source, loaded_names)
    extend_simulation_plan(plan, stage3_config, inference_model, simulation_data, payload)
    inference_store.freeze_after_phase1(inference_model, "tiny-phase1")
    inference_root = tmp_path / "stage3-inference"
    inference_root.mkdir()
    inference_path = inference_root / "three_phase_final.pt"
    state = {name: value.clone() for name, value in inference_model.state_dict().items()}
    final = {
        "kind": "ilume_stage3_dual_view_three_phase_final_v4" if dual_view else "ilume_stage3_home_simulation_three_phase_final_v2",
        "format_version": 4 if dual_view else 1, "fold": 1, "model": state,
        "model_state_hash": tensor_state_hash("stage3.dual-view-model-state.v4" if dual_view else "stage3.object-phase1-model-state.v1", state),
        "resolved_registry": plan["resolved_registry"], "resolved_training_plan": plan,
        "training_identity": build_stage3_training_identity(plan),
        "normalization": normalization, "normalization_hash": canonical_json_sha256(normalization),
        "ownership_manifest": inference_model.ownership_manifest(),
        "stage2_encoder_identity": plan["stage2_encoder_identity"],
        "stage2_pretraining": plan["stage2_pretraining"],
        "simulation_training": plan["simulation_training"],
        "object_encoder_state_hash": tensor_state_hash("stage3.object-phase1.final-encoder.v1", inference_model.stage2_object_encoder.state_dict()),
        "phase1_model_state_hash": "tiny-phase1", "final_embedding_hash": inference_store.final_embedding_hash,
        "validation": {}, "simulation_validation": {},
    }
    torch.save(final, inference_path)
    manifest = {key: final[key] for key in (
        "kind", "format_version", "fold", "model_state_hash", "training_identity",
        "validation", "simulation_validation", "stage2_pretraining", "simulation_training",
        "object_encoder_state_hash", "final_embedding_hash",
    )}
    manifest.update({
        "artifact": inference_path.name, "artifact_sha256": sha256_file(inference_path),
        "object_encoder_phase1": plan["object_encoder_phase1"],
    })
    inference_path.with_suffix(".json").write_text(json.dumps(manifest))
    monkeypatch.setattr("stage3.home.load_prepared_stage3", lambda _config: prepared_stage3)
    loaded_final = load_simulation_final(stage3_config, inference_root, fold=1)
    assert set(loaded_final.task_specs) == set(experimental) | set(SIMULATION_TASKS)
    assert all(torch.equal(loaded_final.state_dict()[name], value) for name, value in state.items())
    from stage3.simulation_reporting import SCALAR_SIMULATION_TASKS
    from stage3.simulation_evaluate import evaluate_simulation_checkpoints, resolve_simulation_evaluation_identity

    # Real packed features and predictions; only the five-fold final selection is synthetic.
    for fold in range(1, 6):
        fold_root = inference_root / f"fold{fold}"
        fold_root.mkdir()
        torch.save(final, fold_root / "three_phase_final.pt")
        (fold_root / "three_phase_final.json").write_text(json.dumps(manifest))
    for task in SCALAR_SIMULATION_TASKS:
        spec = registry.by_id(task)
        spec.dataset.split_path(config.data.data_root, "test").write_bytes(spec.dataset.split_path(config.data.data_root, "valid").read_bytes())
    stage3_config = replace(stage3_config, data=replace(stage3_config.data, task_catalog=config.data.task_catalog_path), training=replace(stage3_config.training, device="cpu", amp_dtype="fp32"))
    loaded_final.eval()
    base_predictions = {}
    for task in SCALAR_SIMULATION_TASKS:
        dataset = Stage2TaskDataset(config.data.artifacts_dir, task, "valid")
        packed = pack_stage2_batch(Stage2BatchDescriptor(task, torch.arange(len(dataset))), {task: dataset}, entities,
                                  MultimodalPacker(vocabulary), needs_entities=True, include_raw_atom_targets=False, pin_memory=False)
        with torch.no_grad():
            base_predictions[task] = loaded_final.predict_simulation(task, packed, Stage2DeviceTaskData.from_dataset(dataset, torch.device("cpu"))).numpy()
    def fake_final(_config, _root, *, fold, device):
        selected = copy.deepcopy(loaded_final)
        original_predict = selected.predict_simulation
        selected.predict_simulation = lambda task, packed, data: original_predict(task, packed, data) + fold
        return selected
    monkeypatch.setattr("stage3.simulation_evaluate.load_simulation_final", fake_final)
    for split in ("valid", "test"):
        evaluation = evaluate_simulation_checkpoints(stage3_config, inference_root, split=split, predictions_dir=tmp_path / split / "predictions")
        assert tuple(evaluation["tasks"]) == SCALAR_SIMULATION_TASKS
        assert evaluation["reporting"]["benchmark"] == "simulation_property"
        for task in SCALAR_SIMULATION_TASKS:
            stats = payload["scalers"][task]["targets"][registry.by_id(task).target_columns[0]]
            prediction_path = tmp_path / split / "predictions" / (task.replace("/", "__") + ".csv")
            with prediction_path.open() as handle:
                actual = [float(row["prediction"]) for row in csv.DictReader(handle)]
            expected = (base_predictions[task].reshape(-1) + 3.0) * stats["scale"] + stats["mean"]
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert resolve_simulation_evaluation_identity(stage3_config, inference_root, split="valid")["hash"] != resolve_simulation_evaluation_identity(stage3_config, inference_root, split="test")["hash"]
    without_simulation = replace(stage3_config, training=replace(stage3_config.training, simulation=None))
    with pytest.raises(ValueError, match="simulation-trained"):
        resolve_simulation_evaluation_identity(without_simulation, inference_root, split="valid")
    final_path = output / "stage2_final.pt"
    manifest_path = output / "stage2_final.json"
    original = torch.load(final_path, map_location="cpu", weights_only=False)
    manifest = json.loads(manifest_path.read_text())
    wrong_epoch_manifest = {**manifest, "fixed_final_epoch": 10 if final_epoch != 10 else 8}
    manifest_path.write_text(json.dumps(wrong_epoch_manifest))
    with pytest.raises(ValueError, match="recipe mismatch"):
        load_home_final(final_path)
    manifest_path.write_text(json.dumps(manifest))
    changed = copy.deepcopy(original)
    changed["kind"] = "ilume_stage2_home_final_v1"
    torch.save(changed, final_path)
    manifest["kind"] = changed["kind"]
    manifest["artifact_sha256"] = sha256_file(final_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="kind"):
        load_home_final(final_path)
    changed = copy.deepcopy(original)
    first_name = next(iter(changed["full_model_state"]))
    changed["full_model_state"][first_name].flatten()[0] += 1
    torch.save(changed, final_path)
    manifest["kind"] = changed["kind"]
    manifest["artifact_sha256"] = sha256_file(final_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="state hash"):
        load_home_final(final_path)
    changed = copy.deepcopy(original)
    owner_name = next(name for name in changed["owner_manifest"] if name.startswith("home."))
    changed["owner_manifest"][owner_name] = "INVALID_OWNER"
    torch.save(changed, final_path)
    manifest["owner_manifest"] = changed["owner_manifest"]
    manifest["artifact_sha256"] = sha256_file(final_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="owner manifest"):
        load_home_final(final_path)
    changed = copy.deepcopy(original)
    changed["scalers"]["simulation/heat_of_vaporization"]["targets"]["heat_of_vaporization_kJ/mol"]["mean"] += 1
    torch.save(changed, final_path)
    manifest["owner_manifest"] = changed["owner_manifest"]
    manifest["artifact_sha256"] = sha256_file(final_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="scaler hash"):
        load_home_final(final_path)


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

    values = torch.randn(2, 1, 16)
    for role in range(3):
        assert model.encode_object(values, torch.full((2, 1), role)).shape == (2, 16)
    ions = torch.randn(2, 2, 16)
    assert model.encode_object(ions, torch.tensor([[0, 1], [0, 1]])).shape == (2, 16)
    assert model.object_heads["simulation/homo"] is not model.object_heads["simulation/lumo"]


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


def test_prepare_train_checkpoint_and_encoder_export(tiny_stage2_setup, tmp_path):
    config = replace(
        tiny_stage2_setup,
        training=replace(
            tiny_stage2_setup.training,
            epochs=5,
            refinement_epochs=0,
            refinement_tasks=(),
        ),
    )
    prepare_teacher_cache(config)
    output = tmp_path / "train"
    run_stage2_training(config, output_dir=output)
    assert (output / "checkpoint_epoch_00001.pt").is_file()
    final_checkpoint = torch.load(
        output / "checkpoint_epoch_00005.pt", map_location="cpu", weights_only=False
    )
    assert final_checkpoint["format_version"] == STAGE2_CHECKPOINT_VERSION
    assert final_checkpoint["completed_epoch"] == 5
    assert final_checkpoint["phase"] == "boundary"
    assert final_checkpoint["optimizer"]["state"]
    assert final_checkpoint["refinement"]["optimizers"] == {}
    assert final_checkpoint["refinement"]["task_updates"] == {}
    task_batches = final_checkpoint["task_batches"]
    steps_per_epoch = sum(task_batches.values())
    assert final_checkpoint["scheduler_geometry"] == {
        "gradient_accumulation_steps": 1,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": 5 * steps_per_epoch,
        "backbone_unfreeze_step": steps_per_epoch,
        "joint_epochs": 5,
        "refinement_epochs": 0,
        "refinement_steps_per_epoch": 0,
        "total_epochs": 5,
    }
    assert final_checkpoint["registry_hash"] == load_artifact_registry(config.data.artifacts_dir).registry_hash
    assert final_checkpoint["model_contract"]["object_encoder"] == {
        "layers": config.model.object_layers,
        "ffn_dim": config.model.object_ffn_dim,
        "dropout": config.model.dropout,
    }
    encoder_path = output / "stage2_encoder.pt"
    assert encoder_path.is_file()
    assert not (output / "taskwise_refined.pt").exists()
    assert not (output / "taskwise_refinement.json").exists()
    final_metrics = json.loads((output / "final_metrics.json").read_text(encoding="utf-8"))
    assert final_metrics["final_epoch"] == 5
    assert final_metrics["final_validation"]
    assert final_metrics["stage2_encoder"]["artifact_sha256"] == sha256_file(encoder_path)
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
        output / "checkpoint_epoch_00005.pt"
    )



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
