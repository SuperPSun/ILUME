from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

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
    Stage2TrainingConfig,
    backbone_unfreeze_step,
    load_stage2_config,
)
from stage2.data import (
    ILSystemCursor,
    Stage2EntityDataset,
    Stage2TaskDataset,
    TaskBlockSampler,
    TaskCursor,
    build_stage2_batch,
)
from stage2.model import (
    PairEncoder,
    Stage2AlignmentModel,
    masked_smooth_l1_loss,
)
from stage2.prepare import (
    load_teacher_embeddings,
    prepare_stage2_data,
    prepare_teacher_cache,
)
from stage2.train import evaluate_stage2, run_stage2_training
from stage1.tokenizer import SmilesTokenizer


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
            teacher_batch_size=2,
        ),
        initialization=Stage2InitializationConfig(checkpoint=checkpoint),
        training=Stage2TrainingConfig(
            batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=20,
            device="cpu",
            amp_dtype="none",
            validation_interval_steps=20,
            early_stopping_patience=1,
            save_every_n_steps=20,
        ),
    )
    return config


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
    with pytest.raises(ValueError, match="checkpoint v1"):
        load_stage1_model(
            legacy,
            tiny_stage2_setup.data.pretrain_artifacts_dir,
        )
    with pytest.raises(ValueError, match="checkpoint v1"):
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
    teacher = load_teacher_embeddings(
        tiny_stage2_setup,
        expected_count=len(entity_dataset),
        expected_dim=loaded.config.model.d_model,
    )
    metrics = evaluate_stage2(
        Stage2AlignmentModel(loaded.model, head_dropout=0.0),
        {
            task: Stage2TaskDataset(
                tiny_stage2_setup.data.artifacts_dir, task, "valid"
            )
            for task in (
                "simulated_qm_elec_hf",
                "density",
                "heat_capacity",
                "thermal_expansion",
                "transfer_organic",
            )
        },
        entity_dataset,
        MultimodalPacker(loaded.vocabulary),
        teacher,
        metadata["scalers"],
        tiny_stage2_setup,
        torch.device("cpu"),
        full=True,
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


def test_teacher_cache_and_stage2_backward_start_from_aligned_embeddings(
    tiny_stage2_setup,
):
    prepare_teacher_cache(tiny_stage2_setup)
    loaded = load_stage1_model(
        tiny_stage2_setup.initialization.checkpoint,
        tiny_stage2_setup.data.pretrain_artifacts_dir,
        backbone_dropout=0.0,
    )
    entities = Stage2EntityDataset(tiny_stage2_setup.data.artifacts_dir)
    teacher = load_teacher_embeddings(
        tiny_stage2_setup,
        expected_count=len(entities),
        expected_dim=loaded.config.model.d_model,
    )
    task = Stage2TaskDataset(
        tiny_stage2_setup.data.artifacts_dir, "density", "train"
    )
    metadata = json.loads(
        (tiny_stage2_setup.data.artifacts_dir / "metadata.json").read_text()
    )
    batch = build_stage2_batch(
        task,
        [0, 1],
        entities,
        MultimodalPacker(loaded.vocabulary),
        teacher,
        metadata["scalers"],
    )
    model = Stage2AlignmentModel(loaded.model, head_dropout=0.0)
    model.eval()
    model.set_backbone_trainable(False)
    output = model(
        batch.task,
        batch.entities,
        batch.entity_positions,
        batch.conditions,
        batch.targets,
        batch.target_mask,
        batch.teacher_embeddings,
        lambda_alignment=0.1,
    )
    assert output.alignment_loss.item() == pytest.approx(0.0, abs=1.0e-12)
    output.total_loss.backward()
    assert loaded.model.fusion.cls_token.grad is None
    assert model.il_pair_encoder.main[0].weight.grad is not None
    assert model.regressors["density"].layers[0].weight.grad is not None
    assert loaded.model.smiles_head.bias.grad is None

    model.zero_grad(set_to_none=True)
    model.set_backbone_trainable(True)
    output = model(
        batch.task,
        batch.entities,
        batch.entity_positions,
        batch.conditions,
        batch.targets,
        batch.target_mask,
        batch.teacher_embeddings,
        lambda_alignment=0.1,
    )
    output.total_loss.backward()
    assert loaded.model.fusion.cls_token.grad is not None
    assert loaded.model.smiles_head.bias.grad is None


def test_pair_encoders_preserve_order_and_regression_heads_are_independent(
    tiny_stage2_setup,
):
    loaded = load_stage1_model(
        tiny_stage2_setup.initialization.checkpoint,
        tiny_stage2_setup.data.pretrain_artifacts_dir,
        backbone_dropout=0.0,
    )
    model = Stage2AlignmentModel(loaded.model, head_dropout=0.0)
    d_model = loaded.config.model.d_model
    left = torch.randn(3, d_model)
    right = torch.randn(3, d_model)

    encoded = model.il_pair_encoder(left, right)
    reversed_encoded = model.il_pair_encoder(right, left)
    assert encoded.shape == (3, d_model)
    assert not torch.equal(encoded, reversed_encoded)
    assert model.il_pair_encoder is not model.transfer_pair_encoder
    parameter_sets = [
        {id(parameter) for parameter in model.regressors[task].parameters()}
        for task in (
            "simulated_qm_elec_hf",
            "density",
            "heat_capacity",
            "thermal_expansion",
            "transfer_organic",
        )
    ]
    assert all(
        left_parameters.isdisjoint(right_parameters)
        for index, left_parameters in enumerate(parameter_sets)
        for right_parameters in parameter_sets[index + 1 :]
    )


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


def test_pair_encoder_standalone_shape():
    encoder = PairEncoder(8, dropout=0.0)
    assert encoder(torch.randn(2, 8), torch.randn(2, 8)).shape == (2, 8)


def test_stage2_task_schedule_and_cursor_resume_are_exact():
    probabilities = {
        "simulated_qm_elec_hf": 0.35,
        "density": 0.20,
        "heat_capacity": 0.15,
        "thermal_expansion": 0.15,
        "transfer_organic": 0.15,
    }
    sampler = TaskBlockSampler(probabilities, 20, seed=42)
    block = [sampler.task_for_step(step) for step in range(20)]
    assert {task: block.count(task) for task in probabilities} == {
        "simulated_qm_elec_hf": 7,
        "density": 4,
        "heat_capacity": 3,
        "thermal_expansion": 3,
        "transfer_organic": 3,
    }

    cursor = TaskCursor(5, seed=9)
    cursor.next_indices(7)
    state = cursor.state_dict()
    expected = cursor.next_indices(6)
    restored = TaskCursor(5, seed=9)
    restored.load_state_dict(state)
    assert torch.equal(restored.next_indices(6), expected)


def test_il_system_cursor_balances_systems_and_cycles_rows_without_replacement():
    offsets = torch.tensor([0, 100, 102])
    rows = torch.arange(102)
    cursor = ILSystemCursor(offsets, rows, seed=17)
    selected = cursor.next_indices(200)

    for start in range(0, 200, 2):
        systems = {0 if value < 100 else 1 for value in selected[start : start + 2]}
        assert systems == {0, 1}
    large_system = selected[selected < 100]
    assert len(set(large_system.tolist())) == 100
    small_system = selected[selected >= 100]
    for start in range(0, 100, 2):
        assert len(set(small_system[start : start + 2].tolist())) == 2

    resumed_source = ILSystemCursor(offsets, rows, seed=17)
    resumed_source.next_indices(37)
    state = resumed_source.state_dict()
    expected = resumed_source.next_indices(29)
    restored = ILSystemCursor(offsets, rows, seed=17)
    restored.load_state_dict(state)
    assert torch.equal(restored.next_indices(29), expected)


def test_formal_stage2_sampling_configs_preserve_transfer_coverage():
    profiles = {
        "reference": (
            load_stage2_config("configs/v1/stage2/base.yaml"),
            (7, 4, 3, 3, 3),
        ),
    }
    task_order = (
        "simulated_qm_elec_hf",
        "density",
        "heat_capacity",
        "thermal_expansion",
        "transfer_organic",
    )
    checkpoints = set()
    expected_unfreeze_steps = {"reference": 2340}
    for name, (config, expected_quotas) in profiles.items():
        effective_batch = (
            config.training.batch_size
            * config.training.gradient_accumulation_steps
        )
        quotas = tuple(
            round(
                config.sampling.probabilities[task]
                * config.sampling.block_size
            )
            for task in task_order
        )
        transfer_steps = (
            config.training.max_steps
            // config.sampling.block_size
            * quotas[-1]
        )
        assert quotas == expected_quotas
        assert effective_batch == 256
        assert transfer_steps == 3516
        assert transfer_steps * effective_batch == 900096
        assert config.data.shard_cache_size == 10
        assert config.data.seed == 42
        assert config.loss.lambda_alignment == pytest.approx(0.1)
        assert config.training.amp_dtype == "bf16"
        assert config.training.backbone_freeze_fraction == pytest.approx(0.10)
        assert backbone_unfreeze_step(config) == expected_unfreeze_steps[name]
        assert config.data.artifacts_dir == Path(
            "outputs/v1/stage2/base/prepare/artifacts"
        )
        checkpoints.add(config.initialization.checkpoint)
    assert len(checkpoints) == 1


def test_formal_stage2_has_one_base_profile():
    config_dir = Path("configs/v1/stage2")
    assert sorted(path.name for path in config_dir.glob("*.yaml")) == ["base.yaml"]
    config = load_stage2_config(config_dir / "base.yaml")
    assert config.training.max_steps == 23440
    assert config.training.batch_size == 256
    assert config.training.gradient_accumulation_steps == 1
    assert backbone_unfreeze_step(config) == 2340
    assert config.initialization.checkpoint == Path(
        "outputs/v1/stage1/base/train/checkpoint_epoch_00005.pt"
    )


def test_stage2_trainer_checkpoints_and_resumes_exactly(
    tiny_stage2_setup,
    capsys,
):
    config = replace(
        tiny_stage2_setup,
        training=replace(
            tiny_stage2_setup.training,
            max_steps=40,
            backbone_freeze_fraction=0.50,
            validation_interval_steps=20,
            early_stopping_patience=3,
            save_every_n_steps=20,
        ),
    )
    prepare_teacher_cache(config)
    capsys.readouterr()
    output = config.data.artifacts_dir.parent / "training"
    first = run_stage2_training(config, output_dir=output)
    capsys.readouterr()
    assert len(first) == 40
    assert [row["global_step"] for row in first] == list(range(1, 41))
    assert {row["training_phase"] for row in first[:20]} == {"heads_only"}
    assert {row["training_phase"] for row in first[20:]} == {
        "backbone_trainable"
    }
    assert {row["backbone_learning_rate"] for row in first[:20]} == {0.0}
    assert first[20]["backbone_learning_rate"] > 0.0
    checkpoint = output / "checkpoint_step_00000020.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 2
    assert payload["kind"] == "ilume_stage2_alignment"
    assert payload["global_step"] == 20
    assert payload["micro_step"] == 20
    assert payload["backbone_unfreeze_step"] == 20
    assert payload["training_phase"] == "backbone_trainable"
    assert sum(payload["task_counts"].values()) == 40

    second = run_stage2_training(
        config, output_dir=output, resume_from=checkpoint
    )
    capsys.readouterr()
    assert [row["task"] for row in second] == [row["task"] for row in first[20:]]
    assert [row["loss"] for row in second] == pytest.approx(
        [row["loss"] for row in first[20:]], abs=1.0e-7
    )

    legacy_checkpoint = output / "legacy_v1.pt"
    payload["format_version"] = 1
    torch.save(payload, legacy_checkpoint)
    with pytest.raises(ValueError, match="Unsupported Stage 2 checkpoint"):
        run_stage2_training(
            config, output_dir=output, resume_from=legacy_checkpoint
        )


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
    teacher = load_teacher_embeddings(
        config,
        expected_count=len(entities),
        expected_dim=loaded.config.model.d_model,
    )
    dataset = Stage2TaskDataset(config.data.artifacts_dir, "density", "train")
    metadata = json.loads(
        (config.data.artifacts_dir / "metadata.json").read_text()
    )
    batch = build_stage2_batch(
        dataset,
        [0, 1],
        entities,
        MultimodalPacker(loaded.vocabulary),
        teacher,
        metadata["scalers"],
    ).to("cuda")
    model = Stage2AlignmentModel(loaded.model, head_dropout=0.0).cuda()
    output = model(
        batch.task,
        batch.entities,
        batch.entity_positions,
        batch.conditions,
        batch.targets,
        batch.target_mask,
        batch.teacher_embeddings,
        lambda_alignment=0.1,
    )
    assert torch.isfinite(output.total_loss)
    output.total_loss.backward()
    assert model.backbone.fusion.cls_token.grad is not None
