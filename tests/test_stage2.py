from __future__ import annotations

import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import stage2.prepare as stage2_prepare
import stage2.train as stage2_train

from common.io import sha256_file
from stage1.config import (
    STAGE1_CHECKPOINT_KIND,
    STAGE1_CHECKPOINT_VERSION,
    DataConfig,
    DescriptorConfig,
    FingerprintConfig,
    ModelConfig,
    PretrainConfig,
)
from stage1.data import PreparedCorpusDataset
from stage1.features import load_stage1_feature_inputs
from stage1.model import load_stage1_model
from stage1.prepare import prepare_corpus
from stage1.masking import MultimodalPacker
from stage1.model import MultimodalPretrainModel
from stage2.config import (
    Stage2Config,
    Stage2DataConfig,
    Stage2InitializationConfig,
    Stage2PreparationConfig,
    Stage2TrainingConfig,
    load_stage2_config,
)
from stage2.data import (
    Stage2EntityDataset,
    Stage2BatchDescriptor,
    Stage2DeviceTaskData,
    Stage2TaskDataset,
    epoch_batch_schedule,
    pack_stage2_window,
    task_batch_counts,
)
from stage2.model import (
    ObjectEncoder,
    Stage2ObjectModel,
    masked_smooth_l1_loss,
    stage2_optimizer_groups,
)
from stage2.prepare import (
    load_teacher_embeddings,
    prepare_stage2_data,
    prepare_teacher_cache,
    teacher_cache_dir,
    teacher_cache_identity,
)
from stage2.train import (
    evaluate_stage2,
    run_stage2_training,
    task_compensation_scale,
)
from stage1.tokenizer import SmilesTokenizer
from stage2.runtime import configure_stage2_math


def _write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stage2_sources(root: Path) -> None:
    _write_csv(
        root / "density/train.csv",
        ["cation", "anion", "temperature_K", "density_g/cm^3", "source_list"],
        [
            {
                "cation": "[Na+]",
                "anion": "[Cl-]",
                "temperature_K": 298,
                "density_g/cm^3": 1.0,
                "source_list": "simulation",
            },
            {
                "cation": "[Na+]",
                "anion": "[Cl-]",
                "temperature_K": 298,
                "density_g/cm^3": 2.0,
                "source_list": "simulation",
            },
        ],
    )
    _write_csv(
        root / "density/valid.csv",
        ["cation", "anion", "temperature_K", "density_g/cm^3", "source_list"],
        [
            {
                "cation": "C[NH3+]",
                "anion": "C(=O)[O-]",
                "temperature_K": 310,
                "density_g/cm^3": 100.0,
                "source_list": "simulation",
            }
        ],
    )
    for task, target, train_value, valid_value in (
        ("heat_capacity", "heat_capacity_J/mol/K", 200.0, 300.0),
        ("thermal_expansion", "thermal_expansion_K^-1", 0.001, 0.002),
    ):
        fields = ["cation", "anion", "temperature_K", target, "source_list"]
        _write_csv(
            root / task / "train.csv",
            fields,
            [
                {
                    "cation": "[Na+]",
                    "anion": "C(=O)[O-]",
                    "temperature_K": 300,
                    target: train_value,
                    "source_list": "simulation",
                }
            ],
        )
        _write_csv(
            root / task / "valid.csv",
            fields,
            [
                {
                    "cation": "C[NH3+]",
                    "anion": "[Cl-]",
                    "temperature_K": 310,
                    target: valid_value,
                    "source_list": "simulation",
                }
            ],
        )

    qm_fields = [
        "SMILES",
        "ESP_max",
        "ESP_min",
        "ESP_std",
        "ESP_pos_frac",
        "Dipole",
        "Quadrupole",
        "q_max",
        "q_min",
        "q_std",
        "q_pos_frac",
        "gap_eV",
        "source_list",
    ]
    qm_values = {
        name: index + 0.5 for index, name in enumerate(qm_fields[1:-1])
    }
    _write_csv(
        root / "simulated_qm_elec_hf/train.csv",
        qm_fields,
        [{"SMILES": "CC", **qm_values, "source_list": "simulation"}],
    )
    _write_csv(
        root / "simulated_qm_elec_hf/valid.csv",
        qm_fields,
        [
            {
                "SMILES": "CCC",
                **{name: value + 10 for name, value in qm_values.items()},
                "source_list": "simulation",
            }
        ],
    )
    transfer_fields = [
        "solute",
        "solvent",
        "transfer_organic_kcal/mol",
        "source_list",
    ]
    _write_csv(
        root / "transfer_organic/train.csv",
        transfer_fields,
        [
            {
                "solute": "CC",
                "solvent": "O",
                "transfer_organic_kcal/mol": -1.0,
                "source_list": "simulation",
            }
        ],
    )
    _write_csv(
        root / "transfer_organic/valid.csv",
        transfer_fields,
        [
            {
                "solute": "CCC",
                "solvent": "O",
                "transfer_organic_kcal/mol": -2.0,
                "source_list": "simulation",
            }
        ],
    )


@pytest.fixture
def tiny_stage2_setup(tmp_path):
    stage1 = tmp_path / "stage1"
    stage1.mkdir()
    _write_csv(stage1 / "cation.csv", ["SMILES"], [{"SMILES": "[Na+]"}, {"SMILES": "C[NH3+]"}])
    _write_csv(stage1 / "anion.csv", ["SMILES"], [{"SMILES": "[Cl-]"}, {"SMILES": "C(=O)[O-]"}])
    _write_csv(
        stage1 / "molecule.csv",
        ["SMILES"],
        [{"SMILES": value} for value in ("O", "CC", "CCC", "CCO")],
    )
    corpus = tmp_path / "pretrain"
    pretrain = PretrainConfig(
        data=DataConfig(
            stage1_dir=stage1,
            artifacts_dir=corpus,
            valid_fraction=0.5,
            max_smiles_tokens=64,
            shard_size=4,
        ),
        descriptor=DescriptorConfig(mode="full", token_count=8),
        fingerprint=FingerprintConfig(kind="both"),
        model=ModelConfig(
            d_model=16,
            n_heads=4,
            smiles_layers=1,
            graph_depth=2,
            descriptor_hidden_dim=32,
            descriptor_blocks=1,
            fusion_layers=1,
            feedforward_dim=32,
            dropout=0.0,
        ),
    )
    prepare_corpus(pretrain)
    vocabulary = SmilesTokenizer.load(corpus / "tokenizer.json")
    dataset = PreparedCorpusDataset(corpus, "train")
    model = MultimodalPretrainModel(
        pretrain, vocabulary, dataset.descriptor_schema
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "kind": STAGE1_CHECKPOINT_KIND,
            "format_version": STAGE1_CHECKPOINT_VERSION,
            "model": model.state_dict(),
            "config": pretrain.to_dict(),
            "artifact_hash": sha256_file(corpus / "metadata.json"),
        },
        checkpoint,
    )
    stage2 = tmp_path / "stage2"
    _stage2_sources(stage2)
    config = Stage2Config(
        data=Stage2DataConfig(
            stage2_dir=stage2,
            pretrain_artifacts_dir=corpus,
            artifacts_dir=tmp_path / "stage2_artifacts",
            entity_shard_size=3,
        ),
        preparation=Stage2PreparationConfig(workers=1, teacher_batch_size=2),
        initialization=Stage2InitializationConfig(checkpoint=checkpoint),
        training=Stage2TrainingConfig(
            batch_size=2,
            gradient_accumulation_steps=1,
            epochs=2,
            backbone_frozen_epochs=1,
            packing_workers=2,
            packing_prefetch_windows=2,
            log_every_batches=2,
            device="cpu",
            amp_dtype="none",
        ),
    )
    return config


def _load_teacher_for_test(config, loaded, entities, metadata, device="cpu"):
    return load_teacher_embeddings(
        config,
        loaded,
        metadata,
        configure_stage2_math(torch.device(device)),
        expected_count=len(entities),
        expected_dim=loaded.config.model.d_model,
    )


def test_stage2_loaders_reject_legacy_stage1_checkpoint(tiny_stage2_setup, tmp_path):
    payload = torch.load(
        tiny_stage2_setup.initialization.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    payload["format_version"] = 3
    payload.pop("kind")
    legacy = tmp_path / "legacy_stage1.pt"
    torch.save(payload, legacy)
    with pytest.raises(ValueError, match="checkpoint v2"):
        load_stage1_model(
            legacy,
            tiny_stage2_setup.data.pretrain_artifacts_dir,
        )
    with pytest.raises(ValueError, match="checkpoint v2"):
        load_stage1_feature_inputs(
            legacy,
            tiny_stage2_setup.data.pretrain_artifacts_dir,
        )


def test_stage2_prepare_preserves_duplicates_and_uses_train_only_scalers(
    tiny_stage2_setup,
):
    metadata = prepare_stage2_data(tiny_stage2_setup)
    assert metadata["summary"]["rows"]["density"]["train"] == 2
    assert metadata["summary"]["duplicate_conditions"] == 1
    density = metadata["scalers"]["targets"]["density_g/cm^3"]
    assert density["mean"] == pytest.approx(1.5)
    assert density["scale"] == pytest.approx(0.5)
    duplicate_rows = list(
        csv.DictReader(
            (tiny_stage2_setup.data.artifacts_dir / "duplicate_conditions.csv").open()
        )
    )
    assert len(duplicate_rows) == 1
    assert duplicate_rows[0]["first_targets"] == "1"
    assert duplicate_rows[0]["duplicate_targets"] == "2"
    density_train = Stage2TaskDataset(
        tiny_stage2_setup.data.artifacts_dir, "density", "train"
    )
    assert density_train.targets[:, 0].tolist() == pytest.approx([-1.0, 1.0])
    assert density_train.raw_targets is None


def test_stage2_prepare_scans_each_csv_once(tiny_stage2_setup, monkeypatch):
    calls: list[tuple[str, str]] = []
    original = stage2_prepare._iter_rows

    def tracked(stage2_dir, task, split):
        calls.append((task, split))
        yield from original(stage2_dir, task, split)

    monkeypatch.setattr(stage2_prepare, "_iter_rows", tracked)
    prepare_stage2_data(tiny_stage2_setup)
    assert sorted(calls) == sorted(
        (task, split)
        for task in tiny_stage2_setup.loss.task_weights
        for split in ("train", "valid")
    )


def test_stage2_prepare_workers_preserve_artifact_content(
    tiny_stage2_setup, tmp_path
):
    serial = prepare_stage2_data(tiny_stage2_setup)
    parallel_config = replace(
        tiny_stage2_setup,
        data=replace(
            tiny_stage2_setup.data,
            artifacts_dir=tmp_path / "parallel_artifacts",
        ),
        preparation=replace(tiny_stage2_setup.preparation, workers=8),
    )
    parallel = prepare_stage2_data(parallel_config)
    assert parallel["entity_artifact_hash"] == serial["entity_artifact_hash"]
    for relative, digest in serial["artifact_hashes"].items():
        if not relative.startswith("entities/"):
            assert parallel["artifact_hashes"][relative] == digest
    assert parallel["summary"] == serial["summary"]


def test_stage2_entity_worker_failure_has_entity_context(
    tiny_stage2_setup, monkeypatch
):
    def fail(_item):
        raise RuntimeError("worker failure")

    monkeypatch.setattr(stage2_prepare, "_compute_entity_feature", fail)
    with pytest.raises(
        RuntimeError,
        match=r"candidate_id=0, role=.*canonical_smiles=",
    ):
        prepare_stage2_data(tiny_stage2_setup)


def test_qm_partial_targets_are_masked_audited_and_validated(
    tiny_stage2_setup,
):
    root = tiny_stage2_setup.data.stage2_dir
    qm_fields = [
        "SMILES",
        "ESP_max",
        "ESP_min",
        "ESP_std",
        "ESP_pos_frac",
        "Dipole",
        "Quadrupole",
        "q_max",
        "q_min",
        "q_std",
        "q_pos_frac",
        "gap_eV",
        "source_list",
    ]
    targets = qm_fields[1:-1]
    complete = {name: index + 0.5 for index, name in enumerate(targets)}
    partial = {name: value + 1 for name, value in complete.items()}
    partial["ESP_max"] = ""
    partial["gap_eV"] = "NaN"
    all_missing = {name: "NA" for name in targets}
    _write_csv(
        root / "simulated_qm_elec_hf/train.csv",
        qm_fields,
        [
            {"SMILES": "CC", **complete, "source_list": "simulation"},
            {"SMILES": "CCO", **partial, "source_list": "simulation"},
            {"SMILES": "CO", **all_missing, "source_list": "simulation"},
        ],
    )
    valid = {name: value + 10 for name, value in complete.items()}
    valid["gap_eV"] = "null"
    _write_csv(
        root / "simulated_qm_elec_hf/valid.csv",
        qm_fields,
        [{"SMILES": "CCC", **valid, "source_list": "simulation"}],
    )

    prepare_teacher_cache(tiny_stage2_setup)
    metadata = json.loads(
        (tiny_stage2_setup.data.artifacts_dir / "metadata.json").read_text()
    )
    assert metadata["format_version"] == 2
    assert metadata["kind"] == "ilume_stage2_object_data"
    assert metadata["model_contract"] == {"d_model": 16, "n_heads": 4}
    assert metadata["summary"]["partial_target_rows"] == 2
    assert metadata["summary"]["all_target_missing_rows"] == 1
    qm_train = Stage2TaskDataset(
        tiny_stage2_setup.data.artifacts_dir,
        "simulated_qm_elec_hf",
        "train",
    )
    qm_valid = Stage2TaskDataset(
        tiny_stage2_setup.data.artifacts_dir,
        "simulated_qm_elec_hf",
        "valid",
    )
    assert len(qm_train) == 2
    assert qm_train.target_mask.sum(dim=1).tolist() == [11, 9]
    assert not bool(qm_valid.target_mask[0, -1])
    assert metadata["scalers"]["targets"]["gap_eV"]["count"] == 1
    audit = list(
        csv.DictReader(
            (tiny_stage2_setup.data.artifacts_dir / "missing_targets.csv").open()
        )
    )
    assert [row["action"] for row in audit].count("retained") == 2
    assert [row["action"] for row in audit].count("excluded") == 1

    loaded = load_stage1_model(
        tiny_stage2_setup.initialization.checkpoint,
        tiny_stage2_setup.data.pretrain_artifacts_dir,
        backbone_dropout=0.0,
    )
    entity_dataset = Stage2EntityDataset(tiny_stage2_setup.data.artifacts_dir)
    teacher, _ = _load_teacher_for_test(
        tiny_stage2_setup, loaded, entity_dataset, metadata
    )
    valid_datasets = {
        task: Stage2TaskDataset(
            tiny_stage2_setup.data.artifacts_dir, task, "valid"
        )
        for task in tiny_stage2_setup.loss.task_weights
    }
    valid_device = {
        task: Stage2DeviceTaskData.from_dataset(dataset, torch.device("cpu"))
        for task, dataset in valid_datasets.items()
    }
    roles = torch.tensor(
        [entry["role_id"] for entry in entity_dataset.entries], dtype=torch.long
    )
    metrics = evaluate_stage2(
        Stage2ObjectModel(
            loaded.model,
            object_layers=1,
            object_ffn_dim=32,
            dropout=0.0,
        ),
        valid_datasets,
        valid_device,
        entity_dataset,
        MultimodalPacker(loaded.vocabulary),
        teacher,
        roles,
        metadata["scalers"],
        tiny_stage2_setup,
        torch.device("cpu"),
        backbone_frozen=True,
    )
    prefix = "valid_simulated_qm_elec_hf_gap_eV"
    assert metrics[f"{prefix}_count"] == 0
    assert torch.isnan(torch.tensor(metrics[f"{prefix}_mae"]))
    assert torch.isfinite(
        torch.tensor(metrics["valid_simulated_qm_elec_hf_normalized_mae"])
    )


def test_qm_masked_loss_weights_available_labels_equally():
    predictions = torch.tensor(
        [[2.0, 2.0], [0.0, 2.0]],
        requires_grad=True,
    )
    targets = torch.zeros_like(predictions)
    mask = torch.tensor([[True, True], [False, True]])
    loss = masked_smooth_l1_loss(predictions, targets, mask)

    assert loss.item() == pytest.approx(1.5)
    loss.backward()
    assert predictions.grad[1, 0].item() == 0.0


def test_qm_train_column_cannot_be_entirely_missing(tiny_stage2_setup):
    path = (
        tiny_stage2_setup.data.stage2_dir
        / "simulated_qm_elec_hf/train.csv"
    )
    rows = list(csv.DictReader(path.open()))
    rows[0]["gap_eV"] = "N/A"
    _write_csv(path, tuple(rows[0]), rows)

    with pytest.raises(ValueError, match="gap_eV"):
        prepare_stage2_data(tiny_stage2_setup)


def test_teacher_cache_and_object_model_freeze_boundary(tiny_stage2_setup):
    prepare_teacher_cache(tiny_stage2_setup)
    loaded = load_stage1_model(
        tiny_stage2_setup.initialization.checkpoint,
        tiny_stage2_setup.data.pretrain_artifacts_dir,
        backbone_dropout=0.0,
    )
    entities = Stage2EntityDataset(tiny_stage2_setup.data.artifacts_dir)
    metadata = json.loads(
        (tiny_stage2_setup.data.artifacts_dir / "metadata.json").read_text()
    )
    teacher, _ = _load_teacher_for_test(
        tiny_stage2_setup, loaded, entities, metadata
    )
    task = Stage2TaskDataset(
        tiny_stage2_setup.data.artifacts_dir, "density", "train"
    )
    descriptor = Stage2BatchDescriptor("density", torch.tensor([0, 1]))
    packed = pack_stage2_window(
        [descriptor],
        {"density": task},
        entities,
        MultimodalPacker(loaded.vocabulary),
        pin_memory=False,
    )
    model = Stage2ObjectModel(
        loaded.model,
        object_layers=1,
        object_ffn_dim=32,
        dropout=0.0,
    )
    model.eval()
    model.set_backbone_trainable(False)
    global_slots = task.entity_indices[descriptor.indices]
    teacher_slots = teacher[global_slots]
    roles = torch.tensor(
        [entry["role_id"] for entry in entities.entries], dtype=torch.long
    )[global_slots]
    output = model.forward_from_slots(
        "density",
        teacher_slots,
        roles,
        task.conditions[descriptor.indices],
        task.targets[descriptor.indices],
        task.target_mask[descriptor.indices],
        teacher_slots,
        lambda_teacher=0.1,
        teacher_loss_is_zero=True,
    )
    assert output.teacher_loss.item() == 0.0
    output.total_loss.backward()
    assert loaded.model.fusion.cls_token.grad is None
    assert model.object_encoder.object_cls.grad is not None
    assert model.property_heads["density"].layers[0].weight.grad is not None
    assert loaded.model.smiles_head.bias.grad is None

    model.zero_grad(set_to_none=True)
    model.set_backbone_trainable(True)
    unique_teacher = teacher[packed.unique_entity_ids]
    output = model(
        "density",
        packed.entities,
        packed.entity_positions[0],
        task.conditions[descriptor.indices],
        task.targets[descriptor.indices],
        task.target_mask[descriptor.indices],
        unique_teacher,
        lambda_teacher=0.1,
    )
    output.total_loss.backward()
    assert loaded.model.fusion.cls_token.grad is not None
    assert loaded.model.smiles_head.bias.grad is None


def test_frozen_validation_never_calls_packer_or_backbone(
    tiny_stage2_setup, monkeypatch
):
    prepare_teacher_cache(tiny_stage2_setup)
    loaded = load_stage1_model(
        tiny_stage2_setup.initialization.checkpoint,
        tiny_stage2_setup.data.pretrain_artifacts_dir,
        backbone_dropout=0.0,
    )
    entities = Stage2EntityDataset(tiny_stage2_setup.data.artifacts_dir)
    metadata = json.loads(
        (tiny_stage2_setup.data.artifacts_dir / "metadata.json").read_text()
    )
    teacher, _ = _load_teacher_for_test(
        tiny_stage2_setup, loaded, entities, metadata
    )
    datasets = {
        task: Stage2TaskDataset(
            tiny_stage2_setup.data.artifacts_dir, task, "valid"
        )
        for task in tiny_stage2_setup.loss.task_weights
    }
    device_data = {
        task: Stage2DeviceTaskData.from_dataset(dataset, torch.device("cpu"))
        for task, dataset in datasets.items()
    }
    roles = torch.tensor(
        [entry["role_id"] for entry in entities.entries], dtype=torch.long
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("frozen validation called the Stage 1 backbone")

    monkeypatch.setattr(loaded.model, "encode", forbidden)
    metrics = evaluate_stage2(
        Stage2ObjectModel(
            loaded.model, object_layers=1, object_ffn_dim=32, dropout=0.0
        ),
        datasets,
        device_data,
        entities,
        None,
        teacher,
        roles,
        metadata["scalers"],
        tiny_stage2_setup,
        torch.device("cpu"),
        backbone_frozen=True,
    )
    assert all(
        metrics[f"valid_{task}_teacher_mse"] == 0.0
        for task in tiny_stage2_setup.loss.task_weights
    )


def test_object_encoder_zero_init_roles_and_independent_heads(tiny_stage2_setup):
    loaded = load_stage1_model(
        tiny_stage2_setup.initialization.checkpoint,
        tiny_stage2_setup.data.pretrain_artifacts_dir,
        backbone_dropout=0.0,
    )
    model = Stage2ObjectModel(
        loaded.model,
        object_layers=1,
        object_ffn_dim=32,
        dropout=0.0,
    )
    d_model = loaded.config.model.d_model
    values = torch.randn(3, 1, d_model)
    neutral = torch.full((3, 1), 2, dtype=torch.long)
    expected = model.object_encoder.output_normalization(values[:, 0])
    assert torch.equal(model.encode_object(values, neutral), expected)
    with pytest.raises(ValueError, match="neutral"):
        model.encode_object(values, torch.zeros_like(neutral))
    ions = torch.randn(3, 2, d_model)
    roles = torch.tensor([[0, 1]]).expand(3, -1)
    expected_il = model.object_encoder.output_normalization(ions.mean(dim=1))
    assert torch.equal(model.encode_object(ions, roles), expected_il)
    with pytest.raises(ValueError, match="ordered"):
        model.encode_object(ions, roles.flip(1))
    parameter_sets = [
        {id(parameter) for parameter in model.property_heads[task].parameters()}
        for task in ("simulated_qm_elec_hf", "density", "heat_capacity", "thermal_expansion")
    ]
    assert all(
        left_parameters.isdisjoint(right_parameters)
        for index, left_parameters in enumerate(parameter_sets)
        for right_parameters in parameter_sets[index + 1 :]
    )

    calls = []
    hook = model.object_encoder.register_forward_hook(
        lambda _module, _inputs, _output: calls.append(1)
    )
    transfer_slots = torch.randn(3, 2, d_model)
    transfer_roles = torch.full((3, 2), 2, dtype=torch.long)
    prediction = model.predict(
        "transfer_organic",
        transfer_slots,
        transfer_roles,
        torch.empty(3, 0),
    )
    hook.remove()
    assert prediction.shape == (3, 1)
    assert len(calls) == 2


def test_compatible_stage1_checkpoint_is_not_blocked_by_lineage_sha(
    tiny_stage2_setup,
) -> None:
    checkpoint_path = tiny_stage2_setup.initialization.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["artifact_hash"] = "historical-lineage-value"
    torch.save(checkpoint, checkpoint_path)
    loaded = load_stage1_model(
        checkpoint_path,
        tiny_stage2_setup.data.pretrain_artifacts_dir,
        backbone_dropout=0.0,
    )
    assert loaded.model.config.model.d_model == 16


def test_teacher_cache_identity_tracks_encoding_state_not_checkpoint_path(
    tiny_stage2_setup, tmp_path
) -> None:
    first = prepare_teacher_cache(tiny_stage2_setup)
    assert first["cache_reused"] is False
    assert not Path(first["checkpoint"]).is_absolute()
    original = torch.load(
        tiny_stage2_setup.initialization.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    original["optimizer"] = {"ignored": True}
    copied_checkpoint = tmp_path / "copied_checkpoint.pt"
    torch.save(original, copied_checkpoint)
    copied_config = replace(
        tiny_stage2_setup,
        initialization=Stage2InitializationConfig(checkpoint=copied_checkpoint),
    )
    copied = prepare_teacher_cache(copied_config)
    assert copied["identity"] == first["identity"]
    assert copied["cache_reused"] is True
    assert teacher_cache_dir(copied_config, copied["identity"]).is_dir()

    model_key = next(
        name
        for name in original["model"]
        if name.startswith("smiles_encoder.")
        and name != "smiles_encoder.token_embedding.weight"
    )
    changed_tensor = original["model"][model_key].clone()
    changed_tensor.view(-1)[0] += 1.0
    original["model"][model_key] = changed_tensor
    changed_checkpoint = tmp_path / "changed_checkpoint.pt"
    torch.save(original, changed_checkpoint)
    changed_config = replace(
        tiny_stage2_setup,
        initialization=Stage2InitializationConfig(checkpoint=changed_checkpoint),
    )
    changed = prepare_teacher_cache(changed_config)
    assert changed["identity"] != first["identity"]

    loaded = load_stage1_model(
        changed_config.initialization.checkpoint,
        changed_config.data.pretrain_artifacts_dir,
        backbone_dropout=0.0,
    )
    metadata = json.loads(
        (changed_config.data.artifacts_dir / "metadata.json").read_text()
    )
    math_contract = configure_stage2_math(torch.device("cpu"))
    base_identity, payload = teacher_cache_identity(
        metadata, loaded, math_contract
    )
    changed_math = {**math_contract, "fp32_matmul_precision": "changed"}
    assert teacher_cache_identity(metadata, loaded, changed_math)[0] != base_identity
    changed_entities = {**metadata, "entity_artifact_hash": "changed"}
    assert (
        teacher_cache_identity(changed_entities, loaded, math_contract)[0]
        != base_identity
    )
    assert payload["dtype"] == "float32"


def test_teacher_cache_corruption_is_rejected(tiny_stage2_setup) -> None:
    cache = prepare_teacher_cache(tiny_stage2_setup)
    path = teacher_cache_dir(tiny_stage2_setup, cache["identity"]) / "embeddings.pt"
    with path.open("ab") as handle:
        handle.write(b"corrupt")
    loaded = load_stage1_model(
        tiny_stage2_setup.initialization.checkpoint,
        tiny_stage2_setup.data.pretrain_artifacts_dir,
        backbone_dropout=0.0,
    )
    entities = Stage2EntityDataset(tiny_stage2_setup.data.artifacts_dir)
    metadata = json.loads(
        (tiny_stage2_setup.data.artifacts_dir / "metadata.json").read_text()
    )
    with pytest.raises(ValueError, match="embedding hash mismatch"):
        _load_teacher_for_test(
            tiny_stage2_setup, loaded, entities, metadata
        )


def test_full_coverage_schedule_is_deterministic_and_complete(tiny_stage2_setup):
    prepare_stage2_data(tiny_stage2_setup)
    datasets = {
        task: Stage2TaskDataset(tiny_stage2_setup.data.artifacts_dir, task, "train")
        for task in tiny_stage2_setup.loss.task_weights
    }
    first = epoch_batch_schedule(datasets, 2, seed=42, epoch=1)
    repeated = epoch_batch_schedule(datasets, 2, seed=42, epoch=1)
    second = epoch_batch_schedule(datasets, 2, seed=42, epoch=2)
    assert [(x.task, x.indices.tolist()) for x in first] == [
        (x.task, x.indices.tolist()) for x in repeated
    ]
    assert [(x.task, x.indices.tolist()) for x in first] != [
        (x.task, x.indices.tolist()) for x in second
    ]
    for task, dataset in datasets.items():
        visited = torch.cat([x.indices for x in first if x.task == task])
        assert sorted(visited.tolist()) == list(range(len(dataset)))
    assert sum(task_batch_counts(datasets, 2).values()) == len(first)


def test_window_packing_deduplicates_across_tasks(tiny_stage2_setup):
    prepare_stage2_data(tiny_stage2_setup)
    entities = Stage2EntityDataset(tiny_stage2_setup.data.artifacts_dir)
    datasets = {
        task: Stage2TaskDataset(
            tiny_stage2_setup.data.artifacts_dir, task, "train"
        )
        for task in ("density", "heat_capacity")
    }
    descriptors = (
        Stage2BatchDescriptor("density", torch.tensor([0])),
        Stage2BatchDescriptor("heat_capacity", torch.tensor([0])),
    )
    packed = pack_stage2_window(
        descriptors,
        datasets,
        entities,
        MultimodalPacker(
            load_stage1_model(
                tiny_stage2_setup.initialization.checkpoint,
                tiny_stage2_setup.data.pretrain_artifacts_dir,
                backbone_dropout=0.0,
            ).vocabulary
        ),
        pin_memory=False,
    )
    flattened = torch.cat(
        [datasets[item.task].entity_indices[item.indices].flatten() for item in descriptors]
    )
    assert len(packed.unique_entity_ids) == len(set(flattened.tolist()))
    for descriptor, positions in zip(
        descriptors, packed.entity_positions, strict=True
    ):
        recovered = packed.unique_entity_ids[positions]
        assert torch.equal(
            recovered, datasets[descriptor.task].entity_indices[descriptor.indices]
        )

    loaded = load_stage1_model(
        tiny_stage2_setup.initialization.checkpoint,
        tiny_stage2_setup.data.pretrain_artifacts_dir,
        backbone_dropout=0.0,
    )
    model = Stage2ObjectModel(
        loaded.model, object_layers=1, object_ffn_dim=32, dropout=0.0
    )
    device_data = {
        task: Stage2DeviceTaskData.from_dataset(dataset, torch.device("cpu"))
        for task, dataset in datasets.items()
    }
    teacher = torch.randn(len(entities), loaded.config.model.d_model)
    calls = []
    hook = model.backbone.fusion.register_forward_pre_hook(
        lambda _module, _inputs: calls.append(1)
    )
    outputs = stage2_train._unfrozen_outputs(
        model,
        packed,
        device_data,
        teacher,
        tiny_stage2_setup,
        torch.device("cpu"),
    )
    hook.remove()
    assert len(outputs) == 2
    assert calls == [1]


def test_thread_prefetch_preserves_window_order(monkeypatch):
    windows = [
        (Stage2BatchDescriptor("density", torch.tensor([0])),),
        (Stage2BatchDescriptor("heat_capacity", torch.tensor([0])),),
    ]

    def delayed(window, *_args, **_kwargs):
        if window[0].task == "density":
            time.sleep(0.05)
        return window[0].task

    monkeypatch.setattr(stage2_train, "pack_stage2_window", delayed)
    yielded = list(
        stage2_train._ordered_packed_windows(
            windows,
            {},
            None,
            None,
            workers=2,
            prefetch_windows=1,
            pin_memory=False,
        )
    )
    assert yielded == ["density", "heat_capacity"]


def test_batch_size_aware_task_compensation_is_partition_invariant():
    total_batches = 4
    sizes = [256, 256, 256, 1]
    scales = [
        task_compensation_scale(0.25, total_batches, size, 769)
        for size in sizes
    ]
    assert sum(scales) == pytest.approx(1.0)
    assert scales[-1] == pytest.approx(scales[0] / 256)


def test_formal_stage2_has_one_base_profile():
    config_dir = Path("configs/v1/stage2")
    assert sorted(path.name for path in config_dir.glob("*.yaml")) == ["base.yaml"]
    config = load_stage2_config(config_dir / "base.yaml")
    assert config.training.epochs == 5
    assert config.training.backbone_frozen_epochs == 1
    assert config.training.batch_size == 256
    assert config.training.gradient_accumulation_steps == 1
    assert config.preparation.workers == 8
    assert config.preparation.teacher_batch_size == 512
    assert config.training.object_encoder_learning_rate == pytest.approx(3.0e-5)
    assert config.training.task_head_learning_rate == pytest.approx(1.0e-4)
    assert config.training.packing_workers == 8
    assert not hasattr(config.data, "shard_cache_size")
    assert config.model.object_layers == 2
    assert config.model.object_ffn_dim == 1024
    assert not hasattr(config.model, "d_model")
    assert not hasattr(config.model, "n_heads")
    assert config.initialization.checkpoint == Path(
        "outputs/v1/stage1/base/train/checkpoint_epoch_00005.pt"
    )


def test_stage2_trainer_saves_only_epoch_checkpoints_and_resumes(
    tiny_stage2_setup, capsys
):
    config = tiny_stage2_setup
    prepare_teacher_cache(config)
    capsys.readouterr()
    output = config.data.artifacts_dir.parent / "training"
    first = run_stage2_training(config, output_dir=output)
    capsys.readouterr()
    intervals = [row for row in first if row["event"] == "stage2_train_interval"]
    assert {row["backbone_trainable"] for row in intervals if row["epoch"] == 1} == {0}
    assert {row["backbone_trainable"] for row in intervals if row["epoch"] == 2} == {1}
    assert {row["backbone_learning_rate"] for row in intervals if row["epoch"] == 1} == {0.0}
    assert any(
        row["backbone_learning_rate"] > 0.0
        for row in intervals
        if row["epoch"] == 2
    )
    assert any(row["object_encoder_learning_rate"] > 0.0 for row in intervals)
    assert any(row["task_head_learning_rate"] > 0.0 for row in intervals)
    checkpoint = output / "checkpoint_epoch_00001.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 2
    assert payload["kind"] == "ilume_stage2_object"
    assert payload["completed_epoch"] == 1
    assert len(payload["optimizer"]["param_groups"]) == 3
    assert payload["optimizer_implementation"] == "single_tensor"
    assert set(path.name for path in output.glob("*.pt")) == {
        "checkpoint_epoch_00001.pt", "checkpoint_epoch_00002.pt"
    }
    (output / "checkpoint_epoch_00002.pt").unlink()
    (output / "final_metrics.json").unlink()
    resumed_config = replace(
        config,
        preparation=replace(config.preparation, workers=2),
        training=replace(
            config.training,
            packing_workers=1,
            packing_prefetch_windows=1,
        ),
    )
    second = run_stage2_training(
        resumed_config, output_dir=output, resume_from=checkpoint
    )
    capsys.readouterr()
    expected = [
        row for row in first
        if row["epoch"] == 2 and row["event"] == "stage2_train_interval"
    ]
    resumed = [
        row for row in second if row["event"] == "stage2_train_interval"
    ]
    assert len(resumed) == len(expected)
    for actual, original in zip(resumed, expected, strict=True):
        for key in original:
            if key.endswith("loss_weighted"):
                assert actual[key] == pytest.approx(original[key], abs=1.0e-7)
    legacy_checkpoint = output / "legacy.pt"
    payload["format_version"] = 1
    torch.save(payload, legacy_checkpoint)
    with pytest.raises(ValueError, match="Unsupported Stage 2 object"):
        run_stage2_training(config, output_dir=output, resume_from=legacy_checkpoint)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_stage2_cuda_minimal_forward_backward(tiny_stage2_setup):
    config = replace(
        tiny_stage2_setup,
        training=replace(tiny_stage2_setup.training, device="cuda"),
    )
    prepare_teacher_cache(config)
    loaded = load_stage1_model(
        config.initialization.checkpoint,
        config.data.pretrain_artifacts_dir,
        device="cuda",
        backbone_dropout=0.0,
    )
    entities = Stage2EntityDataset(config.data.artifacts_dir)
    metadata = json.loads(
        (config.data.artifacts_dir / "metadata.json").read_text()
    )
    teacher, teacher_metadata = _load_teacher_for_test(
        config, loaded, entities, metadata, "cuda"
    )
    teacher = teacher.cuda()
    assert teacher.dtype == torch.float32
    dataset = Stage2TaskDataset(config.data.artifacts_dir, "density", "train")
    descriptor = Stage2BatchDescriptor("density", torch.tensor([0, 1]))
    packed = pack_stage2_window(
        [descriptor],
        {"density": dataset},
        entities,
        MultimodalPacker(loaded.vocabulary),
        pin_memory=True,
    )
    assert packed.unique_entity_ids.is_pinned()
    batch = packed.entities.to("cuda", non_blocking=True)
    positions = packed.entity_positions[0].to("cuda", non_blocking=True)
    unique_teacher = teacher[packed.unique_entity_ids.cuda()]
    model = Stage2ObjectModel(
        loaded.model, object_layers=1, object_ffn_dim=32, dropout=0.0
    ).cuda()
    optimizer = torch.optim.AdamW(
        stage2_optimizer_groups(
            model,
            backbone_learning_rate=1.0e-5,
            object_encoder_learning_rate=3.0e-5,
            task_head_learning_rate=1.0e-4,
            weight_decay=0.01,
        ),
        fused=True,
    )
    output = model(
        "density",
        batch,
        positions,
        dataset.conditions[descriptor.indices].cuda(),
        dataset.targets[descriptor.indices].cuda(),
        dataset.target_mask[descriptor.indices].cuda(),
        unique_teacher,
        lambda_teacher=0.1,
    )
    assert torch.isfinite(output.total_loss)
    output.total_loss.backward()
    assert model.backbone.fusion.cls_token.grad is not None
    optimizer.step()
    assert optimizer.defaults["fused"] is True
    assert teacher_metadata["identity_payload"]["math_contract"][
        "fp32_matmul_precision"
    ] == "tf32"
