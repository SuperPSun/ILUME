from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ilume_pretrain.config import (
    DataConfig,
    DescriptorConfig,
    FingerprintConfig,
    ModelConfig,
    PretrainConfig,
)
from ilume_pretrain.data import PreparedCorpusDataset, prepare_corpus
from ilume_pretrain.masking import MultimodalPacker
from ilume_pretrain.model import MultimodalPretrainModel
from ilume_pretrain.stage2_config import (
    Stage2Config,
    Stage2DataConfig,
    Stage2InitializationConfig,
    Stage2TrainingConfig,
    load_stage2_config,
)
from ilume_pretrain.stage2_data import (
    Stage2EntityDataset,
    Stage2TaskDataset,
    TaskBlockSampler,
    TaskCursor,
    build_stage2_batch,
    prepare_stage2_data,
)
from ilume_pretrain.stage2_model import (
    Stage2AlignmentModel,
    load_stage1_model,
    sha256_file,
)
from ilume_pretrain.stage2_prepare import (
    load_teacher_embeddings,
    prepare_teacher_cache,
)
from ilume_pretrain.stage2_training import run_stage2_training
from ilume_pretrain.tokenizer import SmilesTokenizer


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
            "format_version": 3,
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
            output_dir=tmp_path / "stage2_training",
        ),
    )
    return config


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
        checkpoint_hash=loaded.checkpoint_hash,
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
    output = model(
        batch.task,
        batch.entities,
        batch.entity_positions,
        batch.conditions,
        batch.targets,
        batch.teacher_embeddings,
        lambda_alignment=0.1,
    )
    assert output.alignment_loss.item() == pytest.approx(0.0, abs=1.0e-12)
    output.total_loss.backward()
    assert loaded.model.fusion.cls_token.grad is not None
    assert loaded.model.smiles_head.bias.grad is None


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


def test_formal_stage2_sampling_configs_preserve_transfer_coverage():
    profiles = {
        "reference": (
            load_stage2_config("configs/stage2_base.yaml"),
            (7, 4, 3, 3, 3),
        ),
        "balanced": (
            load_stage2_config(
                "configs/stage2_base_sampling_balanced.yaml"
            ),
            (4, 4, 4, 4, 4),
        ),
        "il_heavy": (
            load_stage2_config(
                "configs/stage2_base_sampling_il_heavy.yaml"
            ),
            (2, 6, 5, 5, 2),
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
    for config, expected_quotas in profiles.values():
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
        checkpoints.add(config.initialization.checkpoint)
    assert len(checkpoints) == 1


def test_formal_stage2_model_size_configs_share_training_budget():
    configs = {
        "base": load_stage2_config("configs/stage2_base.yaml"),
        "large": load_stage2_config("configs/stage2_large.yaml"),
        "xlarge": load_stage2_config("configs/stage2_xlarge.yaml"),
    }
    expected_micro_batches = {"base": 256, "large": 128, "xlarge": 64}
    expected_accumulation = {"base": 1, "large": 2, "xlarge": 4}
    reference = configs["base"]
    assert Stage2Config() == reference
    for name, config in configs.items():
        assert config.data.artifacts_dir == reference.data.artifacts_dir
        assert config.sampling.probabilities == reference.sampling.probabilities
        assert config.training.max_steps == 23440
        assert config.training.batch_size == expected_micro_batches[name]
        assert (
            config.training.gradient_accumulation_steps
            == expected_accumulation[name]
        )
        assert (
            config.training.batch_size
            * config.training.gradient_accumulation_steps
            == 256
        )
    assert len(
        {config.initialization.checkpoint for config in configs.values()}
    ) == len(configs)


def test_formal_stage2_comparison_outputs_are_unique():
    paths = (
        "configs/stage2_base.yaml",
        "configs/stage2_base_sampling_balanced.yaml",
        "configs/stage2_base_sampling_il_heavy.yaml",
        "configs/stage2_large.yaml",
        "configs/stage2_xlarge.yaml",
    )
    output_dirs = {
        load_stage2_config(path).training.output_dir for path in paths
    }
    assert len(output_dirs) == len(paths)


def test_stage2_trainer_checkpoints_and_resumes_exactly(
    tiny_stage2_setup,
    capsys,
):
    config = replace(
        tiny_stage2_setup,
        training=replace(
            tiny_stage2_setup.training,
            validation_interval_steps=10,
            early_stopping_patience=3,
            keep_last_checkpoints=2,
        ),
    )
    prepare_teacher_cache(config)
    capsys.readouterr()
    first = run_stage2_training(config)
    capsys.readouterr()
    assert len(first) == 20
    assert [row["global_step"] for row in first] == list(range(1, 21))
    checkpoint = config.training.output_dir / "checkpoint_step_00000010.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 1
    assert payload["kind"] == "ilume_stage2_alignment"
    assert payload["global_step"] == 10
    assert payload["micro_step"] == 10
    assert sum(payload["task_counts"].values()) == 20

    resumed = replace(
        config,
        training=replace(config.training, resume_from=checkpoint),
    )
    second = run_stage2_training(resumed)
    capsys.readouterr()
    assert [row["task"] for row in second] == [row["task"] for row in first[10:]]
    assert [row["loss"] for row in second] == pytest.approx(
        [row["loss"] for row in first[10:]], abs=1.0e-7
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
        checkpoint_hash=loaded.checkpoint_hash,
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
        batch.teacher_embeddings,
        lambda_alignment=0.1,
    )
    assert torch.isfinite(output.total_loss)
    output.total_loss.backward()
    assert model.backbone.fusion.cls_token.grad is not None
