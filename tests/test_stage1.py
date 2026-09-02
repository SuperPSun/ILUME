from __future__ import annotations

from pathlib import Path

from dataclasses import replace

import hashlib

import json

import pytest

from stage1.config import (
    ArchitectureConfig,
    GLOBAL_RDKIT_STAGE1_CHECKPOINT_VERSION,
    STAGE1_CHECKPOINT_KIND,
    STAGE1_CHECKPOINT_VERSION,
    config_from_dict,
    load_config,
)

import common.outputs as outputs_module

from common.data_identity import write_data_identity

from common.identity import semantic_identity

from common.outputs import open_run_directory

import csv

import torch

import torch.distributed as dist

import torch.multiprocessing as mp

from stage1.config import (
    DataConfig,
    DescriptorConfig,
    FingerprintConfig,
    ModelConfig,
    PreparationConfig,
    PretrainConfig,
    TrainingConfig,
)

from stage1.prepare import prepare_corpus

from stage1.model import LossStatistics, PretrainOutput

import stage1.train as train_module

from stage1.train import (
    _DistributedContext,
    _global_training_losses,
    run_training,
)

from collections import Counter

import numpy as np

from rdkit import Chem

from stage1.config import DataConfig, PreparationConfig, PretrainConfig

import stage1.features as features_module

import stage1.prepare as prepare_module

from stage1.data import (
    CORPUS_FORMAT_VERSION,
    CORPUS_KIND,
    GLOBAL_RDKIT_CORPUS_FORMAT_VERSION,
    PreparedCorpusDataset,
)

from stage1.features import IPC_SQUARE_OVERFLOW_LIMIT, inspect_entity_qc

from stage1.masking import MultimodalPacker

from stage1.prepare import (
    _descriptor_batch,
    _csv_data_row_count,
    _ordered_batch_map,
    _stage1_shard_sample,
    preparation_source_paths,
    prepare_corpus,
)

from stage1.descriptors import rdkit_descriptor_names

from stage1.descriptors import DescriptorSchema, DescriptorStandardizer

from stage1.masking import mask_smiles_tokens

from stage1.tokenizer import SmilesTokenizer, ais_tokenize

import torch.nn.functional as F

from stage1.config import MaskingConfig

from stage1.graph import ATOM_FEATURE_NAMES, BOND_FEATURE_NAMES

from stage1.data import BatchFusionLayout

from stage1.masking import (
    MultimodalCollator,
    MultimodalMasker,
    MultimodalPacker,
    curriculum_dropout_probability,
)

from stage1.model import MultimodalPretrainModel, _weighted_component

from stage1.encoders import DirectedMessagePassingEncoder

from stage1.fusion import FusionTransformer

from stage1.graph import featurize_mol, pack_graphs


def test_global_rdkit_v2_representation_and_losses(
    tiny_config, tiny_samples
) -> None:
    vocabulary, legacy_samples = tiny_samples
    config = replace(
        tiny_config,
        architecture=ArchitectureConfig(kind="global_rdkit_v2"),
        descriptor=DescriptorConfig(mode="full", token_count=1),
        model=replace(tiny_config.model, descriptor_blocks=2),
    )
    config.validate()
    samples = [
        {key: value for key, value in sample.items() if key != "fingerprints"}
        for sample in legacy_samples
    ]
    schema = DescriptorSchema.fit(
        np.stack([sample["descriptors"].numpy() for sample in samples]),
        rdkit_descriptor_names(),
        "full",
        1,
    )
    model = MultimodalPretrainModel(config, vocabulary, schema)
    packed = MultimodalPacker(vocabulary)(samples)
    encoded = model.encode_entity(packed)

    assert config.checkpoint_version == GLOBAL_RDKIT_STAGE1_CHECKPOINT_VERSION
    assert "fingerprint" not in config.to_dict()
    assert (model.token_dim, model.atom_dim, model.entity_dim) == (32, 32, 64)
    assert model.fusion.modality_embedding.num_embeddings == 4
    assert encoded.cls_embedding.shape == (3, 32)
    assert encoded.rdkit_embedding.shape == (3, 32)
    assert encoded.entity_embedding.shape == (3, 64)
    assert torch.equal(
        encoded.entity_embedding,
        torch.cat((encoded.cls_embedding, encoded.rdkit_embedding), dim=-1),
    )
    assert torch.equal(model.encode(packed), encoded.cls_embedding)
    assert not any("fingerprint" in name for name, _ in model.named_parameters())

    masked = MultimodalMasker(vocabulary, config.masking).apply(packed)
    output = model(masked)
    assert masked.masks.modality_dropped.shape == (3, 3)
    assert set(output.losses) == {"smiles", "descriptor", "atom", "bond"}
    assert set(output.logits) == {"smiles", "descriptor", "atom", "bond"}
    assert output.logits["descriptor"].shape == (3, 217)
    invalid = config.to_dict()
    invalid["fingerprint"] = {"kind": "none"}
    with pytest.raises(ValueError, match="forbids fingerprint"):
        config_from_dict(invalid)

# --- Configuration, identity, and checkpoint/resume contracts ---

ROOT = Path(__file__).resolve().parents[1]

def test_formal_stage1_has_one_large_capacity_base_profile() -> None:
    assert sorted(path.name for path in (ROOT / "configs/v1/stage1").glob("*.yaml")) == [
        "base.yaml"
    ]
    base = load_config(ROOT / "configs/v1/stage1/base.yaml")
    assert (
        base.model.d_model,
        base.model.n_heads,
        base.model.smiles_layers,
        base.model.graph_depth,
        base.model.descriptor_hidden_dim,
        base.model.fusion_layers,
        base.model.feedforward_dim,
    ) == (512, 8, 8, 6, 1024, 8, 2048)
    assert base.model.role_embedding is True
    assert base.model.gradient_checkpointing is False
    assert base.data.include_augmentation is True
    assert base.loss.role_weights == (2.0, 2.0, 1.0)
    assert base.training.batch_size == 128
    assert base.training.num_workers == 8
    assert base.training.epochs == 5
    assert base.training.learning_rate == pytest.approx(1.0e-4)
    assert base.training.compile is False
    assert base.training.validation_interval_steps == 5000
    assert base.tokenizer.min_frequency == 1
    assert (
        base.preparation.workers,
        base.preparation.catalog_batch_size,
        base.preparation.qc_batch_size,
        base.preparation.tokenizer_batch_size,
        base.preparation.descriptor_batch_size,
    ) == (16, 10000, 2048, 2048, 512)
    assert "preparation" in base.to_dict()
    assert "preparation" not in base.experiment_dict()

def test_run_directory_allows_execution_change_on_resume(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(outputs_module, "REPOSITORY_ROOT", tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("training:\n  compile: true\n", encoding="utf-8")
    original = {"training": {"epochs": 2, "compile": True}}
    identity = semantic_identity("test.train", {"epochs": 2})
    run = open_run_directory(
        stage="stage1",
        operation="train",
        config_path="config.yaml",
        config_payload=original,
        semantic_identity=identity,
        output="outputs/train",
        seed=42,
    )
    run.fail()
    resumed = open_run_directory(
        stage="stage1",
        operation="train",
        config_path="config.yaml",
        config_payload={"training": {"epochs": 2, "compile": False}},
        semantic_identity=identity,
        output="outputs/train",
        seed=42,
        resume="outputs/train/last.pt",
    )
    assert resumed.metadata["attempt_id"] != run.metadata["attempt_id"]
    assert resumed.metadata["locator"]["resume"] == "outputs/train/last.pt"

def test_data_identity_records_relative_hash_size_and_rows(tmp_path) -> None:
    source = tmp_path / "data" / "stage1" / "cation.csv"
    source.parent.mkdir(parents=True)
    source.write_text("SMILES\n[Na+]\nC[NH3+]\n", encoding="utf-8")
    identity = write_data_identity(tmp_path, "stage1", [source])
    logical_id = next(iter(identity["locator"]["files"]))
    assert identity["locator"]["files"][logical_id] == "data/stage1/cation.csv"
    source_payload = identity["semantic"]["identities"]["source"]["payload"]
    assert source_payload["sources"][logical_id]["rows"] == 2
    record = identity["integrity"]["files"][logical_id]
    assert record["size"] == source.stat().st_size
    assert record["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert identity["provenance"]["source_repository_commit"] is None

def _write_smiles(path, values) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["SMILES"])
        writer.writeheader()
        writer.writerows({"SMILES": value} for value in values)


def test_global_rdkit_v2_corpus_uses_format3_without_fingerprints(
    tmp_path, capsys
) -> None:
    source = tmp_path / "stage1"
    source.mkdir()
    _write_smiles(source / "cation.csv", ["[Na+]", "C[NH3+]"])
    _write_smiles(source / "anion.csv", ["[Cl-]", "C(=O)[O-]"])
    _write_smiles(source / "molecule.csv", ["O", "CC"])
    artifacts = tmp_path / "prepared"
    base = PretrainConfig()
    config = replace(
        base,
        architecture=ArchitectureConfig(kind="global_rdkit_v2"),
        data=replace(
            base.data,
            stage1_dir=source,
            artifacts_dir=artifacts,
            valid_fraction=0.5,
            max_smiles_tokens=64,
            shard_size=2,
        ),
        descriptor=DescriptorConfig(mode="full", token_count=1),
        model=replace(
            base.model,
            d_model=16,
            n_heads=4,
            smiles_layers=1,
            graph_depth=2,
            descriptor_hidden_dim=32,
            descriptor_blocks=2,
            fusion_layers=1,
            feedforward_dim=32,
            dropout=0.0,
        ),
    )

    prepare_corpus(config)
    capsys.readouterr()
    metadata = json.loads((artifacts / "metadata.json").read_text())
    dataset = PreparedCorpusDataset(artifacts, "train")

    assert metadata["format_version"] == GLOBAL_RDKIT_CORPUS_FORMAT_VERSION
    assert metadata["descriptor_dim"] == 217
    assert metadata["descriptor_token_count"] == 1
    assert "fingerprint_kind" not in metadata
    assert "fingerprint_contract" not in metadata
    assert dataset.format_version == GLOBAL_RDKIT_CORPUS_FORMAT_VERSION
    assert "fingerprints" not in dataset[0]

def _ddp_training_worker(
    rank: int,
    world_size: int,
    init_path: str,
    config: PretrainConfig,
    output_dir: str,
    resume_from: str | None,
    stop_after_first_epoch: bool,
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        parameter = torch.tensor(1.0, requires_grad=True)
        local_numerator = parameter * (2.0 if rank == 0 else 9.0)
        statistics = LossStatistics(
            numerators=local_numerator.reshape(1),
            denominators=torch.tensor([2.0 if rank == 0 else 3.0]),
            role_numerators=torch.zeros((1, 3)),
            role_denominators=torch.zeros((1, 3)),
        )
        reduced_loss, _ = _global_training_losses(
            PretrainOutput(
                loss=local_numerator,
                losses={"smiles": local_numerator},
                loss_statistics={"smiles": statistics},
                logits={},
                fused_cls=torch.empty(0),
            ),
            _DistributedContext(rank, world_size, rank),
            config,
        )
        reduced_loss.backward()
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad /= world_size
        assert parameter.grad.item() == pytest.approx(11.0 / 5.0)

        if stop_after_first_epoch:
            real_save = train_module._save_checkpoint

            class StopAfterCheckpoint(RuntimeError):
                pass

            def save_then_stop(paths, **kwargs):
                real_save(paths, **kwargs)
                if kwargs["completed_epoch"] == 1:
                    raise StopAfterCheckpoint

            train_module._save_checkpoint = save_then_stop
            try:
                run_training(config, output_dir=output_dir)
            except StopAfterCheckpoint:
                pass
        else:
            run_training(
                config,
                output_dir=output_dir,
                resume_from=resume_from,
            )
    finally:
        dist.destroy_process_group()

def test_two_rank_gloo_checkpoint_ownership_and_cross_world_resume(tmp_path, capsys) -> None:
    source = tmp_path / "stage1"
    source.mkdir()
    _write_smiles(source / "cation.csv", ["[Na+]", "[K+]", "C[NH3+]", "C[NH2+]C"])
    _write_smiles(source / "anion.csv", ["[Cl-]", "[Br-]", "[I-]", "C(=O)[O-]"])
    _write_smiles(source / "molecule.csv", ["O", "N", "CC", "CCO"])
    artifacts = tmp_path / "prepared"
    output = tmp_path / "ddp_train"
    config = PretrainConfig(
        data=DataConfig(
            stage1_dir=source,
            artifacts_dir=artifacts,
            valid_fraction=0.5,
            max_smiles_tokens=64,
            shard_size=2,
        ),
        descriptor=DescriptorConfig(mode="clean", token_count=1),
        fingerprint=FingerprintConfig(kind="maccs"),
        model=ModelConfig(
            d_model=8,
            n_heads=2,
            smiles_layers=1,
            graph_depth=1,
            descriptor_hidden_dim=16,
            descriptor_blocks=1,
            fusion_layers=1,
            feedforward_dim=16,
            dropout=0.0,
        ),
        training=TrainingConfig(
            batch_size=2,
            epochs=1,
            learning_rate=1.0e-3,
            num_workers=0,
            device="cpu",
            amp_dtype="none",
            compile=False,
            validation_interval_steps=100,
            quick_validation_samples_per_role=1,
        ),
    )
    prepare_corpus(config)
    capsys.readouterr()

    world_size = 2
    mp.spawn(
        _ddp_training_worker,
        args=(
            world_size,
            str(tmp_path / "first_init"),
            config,
            str(output),
            None,
            True,
        ),
        nprocs=world_size,
        join=True,
    )
    mid = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert mid["world_size_at_save"] == 2
    assert mid["global_step"] == 3
    assert mid["completed_epoch"] == 1
    assert "rank_rng" not in mid
    assert "epoch_cursor" not in mid

    mp.spawn(
        _ddp_training_worker,
        args=(
            world_size,
            str(tmp_path / "resume_init"),
            config,
            str(output),
            str(output / "last.pt"),
            False,
        ),
        nprocs=world_size,
        join=True,
    )
    completed = torch.load(
        output / "last.pt", map_location="cpu", weights_only=False
    )
    assert completed["completed_epoch"] == 1
    assert sorted(path.name for path in output.glob("*.pt")) == [
        "checkpoint_epoch_00001.pt",
        "last.pt",
    ]
    metric_steps = [
        json.loads(line)["global_step"]
        for line in (output / "metrics.jsonl").read_text().splitlines()
        if json.loads(line).get("event") != "attempt_start"
    ]
    assert metric_steps == [3]

    assert run_training(
        config, output_dir=output, resume_from=output / "last.pt",
        attempt_id="single-rank-resume",
    ) == []

def test_stage1_epoch_checkpoint_resume_and_attempt_log_preservation(
    tmp_path, capsys, monkeypatch
) -> None:
    source = tmp_path / "stage1"
    source.mkdir()
    _write_smiles(source / "cation.csv", ["[Na+]", "[K+]", "C[NH3+]", "C[NH2+]C"])
    _write_smiles(source / "anion.csv", ["[Cl-]", "[Br-]", "[I-]", "C(=O)[O-]"])
    _write_smiles(source / "molecule.csv", ["O", "N", "CC", "CCO"])
    artifacts = tmp_path / "prepared"
    baseline_output = tmp_path / "baseline"
    output = tmp_path / "train"
    config = PretrainConfig(
        data=DataConfig(
            stage1_dir=source, artifacts_dir=artifacts, valid_fraction=0.5,
            max_smiles_tokens=64, shard_size=2,
        ),
        descriptor=DescriptorConfig(mode="clean", token_count=8),
        fingerprint=FingerprintConfig(kind="both"),
        model=ModelConfig(
            d_model=16, n_heads=4, smiles_layers=1, graph_depth=2,
            descriptor_hidden_dim=32, descriptor_blocks=1, fusion_layers=1,
            feedforward_dim=32, dropout=0.1,
        ),
        training=TrainingConfig(
            batch_size=2, epochs=2,
            learning_rate=1.0e-3, num_workers=0, device="cpu",
            amp_dtype="none", compile=False,
            validation_interval_steps=2, quick_validation_samples_per_role=1,
        ),
    )
    prepare_corpus(config)
    capsys.readouterr()
    validation_calls: list[bool] = []
    real_validate = train_module._validate

    def record_validation(*args, **kwargs):
        validation_calls.append(bool(kwargs["quick"]))
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(train_module, "_validate", record_validation)
    baseline = run_training(config, output_dir=baseline_output, attempt_id="baseline")
    monkeypatch.setattr(train_module, "_validate", real_validate)
    assert [row["global_step"] for row in baseline] == [2, 3, 4, 6]
    assert validation_calls == [True, False, True, False]

    real_save = train_module._save_checkpoint

    class Interrupted(RuntimeError):
        pass

    def interrupt_after_epoch(paths, **kwargs):
        real_save(paths, **kwargs)
        if kwargs["completed_epoch"] == 1:
            raise Interrupted

    monkeypatch.setattr(train_module, "_save_checkpoint", interrupt_after_epoch)
    try:
        run_training(config, output_dir=output, attempt_id="attempt-1")
    except Interrupted:
        pass
    monkeypatch.setattr(train_module, "_save_checkpoint", real_save)

    mid = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert mid["kind"] == STAGE1_CHECKPOINT_KIND
    assert mid["format_version"] == STAGE1_CHECKPOINT_VERSION
    assert mid["completed_epoch"] == 1
    assert mid["global_step"] == 3
    assert set(mid).isdisjoint({"epoch_index", "epoch_cursor", "rank_rng", "micro_step"})
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"global_step": 999, "loss": 0}\n')

    rows = run_training(
        config, output_dir=output, resume_from=output / "last.pt",
        attempt_id="attempt-2",
    )
    assert [row["global_step"] for row in rows] == [4, 6]
    assert [row["loss"] for row in rows] == pytest.approx(
        [row["loss"] for row in baseline[2:]]
    )
    assert (output / "checkpoint_epoch_00001.pt").is_file()
    assert (output / "checkpoint_epoch_00002.pt").is_file()
    assert (output / "last.pt").is_file()
    last = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert last["completed_epoch"] == 2
    metric_rows = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    attempt_rows = [row for row in metric_rows if row.get("event") == "attempt_start"]
    assert attempt_rows == [
        {
            "event": "attempt_start",
            "attempt_id": "attempt-2",
            "resumed_from_attempt_id": "attempt-1",
            "completed_epoch": 1,
            "global_step": 3,
            "world_size": 1,
            "compile": False,
        }
    ]
    training_rows = [row for row in metric_rows if row.get("event") != "attempt_start"]
    assert [row["global_step"] for row in training_rows] == [2, 3, 999, 4, 6]
    assert [row.get("attempt_id") for row in training_rows] == [
        "attempt-1", "attempt-1", None, "attempt-2", "attempt-2"
    ]

    assert run_training(
        config, output_dir=output, resume_from=output / "last.pt"
    ) == []
    changed_preparation = replace(
        config, preparation=PreparationConfig(workers=4)
    )
    assert run_training(
        changed_preparation, output_dir=output, resume_from=output / "last.pt"
    ) == []
    changed_compile = replace(config, training=replace(config.training, compile=True))
    assert run_training(
        changed_compile, output_dir=output, resume_from=output / "last.pt"
    ) == []
    changed_tokenizer = replace(
        config, tokenizer=replace(config.tokenizer, min_frequency=2)
    )
    assert run_training(
        changed_tokenizer, output_dir=output, resume_from=output / "last.pt"
    ) == []

# --- Data preparation and artifact contracts ---

HEADER = [
    "SMILES",
    "formal_charge",
    "origin_list",
    "seed_smiles_list",
    "rule_list",
    "pubchem_cid_list",
    "mol_id_list",
]

def _write_role_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for row in rows:
            smiles, charge, *seed = row
            writer.writerow(
                [smiles, charge, "test", seed[0] if seed else "", "", "", ""]
            )

def test_prepare_uses_new_original_sources_and_sharded_artifacts(tmp_path, monkeypatch):
    stage1 = tmp_path / "stage1"
    artifacts = tmp_path / "artifacts"
    stage1.mkdir()
    _write_role_csv(stage1 / "cation.csv", [("[Na+]", 1), ("C[NH3+]", 1)])
    _write_role_csv(stage1 / "anion.csv", [("[Cl-]", -1), ("C(=O)[O-]", -1)])
    _write_role_csv(stage1 / "molecule.csv", [("CCO", 0), ("O", 0)])
    for ignored in ("simulation_mol.csv", "solute.csv", "solvent.csv"):
        (stage1 / ignored).write_text("not,a,valid,csv\n", encoding="utf-8")
    (stage1 / "IL.csv").write_text("cation,anion\n[K+],[Br-]\n", encoding="utf-8")

    summary = prepare_corpus(
        DataConfig(
            stage1_dir=stage1,
            artifacts_dir=artifacts,
            valid_fraction=0.5,
            seed=3,
            shard_size=2,
        )
    )
    assert summary["total"] == 6
    assert summary["train"] == summary["valid"] == 3
    assert summary["cation"] == summary["anion"] == summary["neutral"] == 2
    assert summary["augmented"] == 0
    assert summary["excluded_entities"] == 0
    assert not (artifacts / "corpus_index.json").exists()
    assert (artifacts / "train_index.npy").is_file()
    assert (artifacts / "valid_index.npy").is_file()
    assert (artifacts / "shard_manifest.json").is_file()
    assert (artifacts / "descriptor_schema.json").is_file()
    assert (artifacts / "excluded_entities.csv").is_file()
    assert not (artifacts / ".prepare.sqlite").exists()
    assert not (artifacts / ".raw_descriptors.npy").exists()
    assert not (artifacts / "preparation_state.json").exists()
    assert len(list((artifacts / "shards").glob("*.pt"))) == 4
    metadata_path = artifacts / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["format_version"] == CORPUS_FORMAT_VERSION == 2
    assert metadata["kind"] == CORPUS_KIND
    assert set(metadata["source_hashes"]) == {
        "cation.csv",
        "anion.csv",
        "molecule.csv",
    }
    assert "excluded_entities.csv" in metadata["artifact_hashes"]
    assert "augmentation_audit.json" in metadata["artifact_hashes"]
    dataset = PreparedCorpusDataset(artifacts, "train")
    assert len(dataset) == 3
    sample = dataset[0]
    assert set(sample) == {
        "sample_id",
        "role_id",
        "token_ids",
        "atom_categorical",
        "atom_continuous",
        "bond_categorical",
        "bond_index",
        "descriptors",
        "descriptor_valid",
        "fingerprints",
    }
    assert all(value.dtype == torch.uint8 for value in sample["fingerprints"].values())
    vocabulary = SmilesTokenizer.load(artifacts / "tokenizer.json")
    packed = MultimodalPacker(vocabulary)([sample])
    assert all(
        value.dtype == torch.float32 for value in packed.fingerprints.values.values()
    )
    batched_dataset = PreparedCorpusDataset(artifacts, "train")
    expected_ids = [batched_dataset[index]["sample_id"] for index in (0, 0, 1)]
    load_calls: list[str] = []
    real_load = batched_dataset._load_shard

    def record_load(relative_path):
        load_calls.append(relative_path)
        return real_load(relative_path)

    monkeypatch.setattr(batched_dataset, "_load_shard", record_load)
    batched = batched_dataset.__getitems__([0, 0, 1])
    assert [item["sample_id"] for item in batched] == expected_ids
    assert len(load_calls) == len(set(load_calls))
    if torch.cuda.is_available():
        pinned = packed.pin_memory()
        assert pinned.token_ids.is_pinned()
        assert pinned.graphs.atom_categorical.is_pinned()
        assert pinned.fusion_layout.smiles_lengths.is_pinned()
        assert all(value.is_pinned() for value in pinned.fingerprints.values.values())

def test_prepare_excludes_qc_failures_before_descriptor_calculation(
    tmp_path, monkeypatch
):
    stage1 = tmp_path / "stage1"
    augmentation = stage1 / "augmentation"
    artifacts = tmp_path / "artifacts"
    augmentation.mkdir(parents=True)
    originals = {
        "cation": [("[Na+]", 1), ("[K+]", 1), ("C[NH3+]", 1), ("C[NH2+]C", 1)],
        "anion": [("[Cl-]", -1), ("[Br-]", -1), ("[I-]", -1), ("C(=O)[O-]", -1)],
        "molecule": [("CCO", 0), ("O", 0), ("N", 0), ("CC", 0)],
    }
    for name, rows in originals.items():
        _write_role_csv(stage1 / f"{name}.csv", rows)
    _write_role_csv(
        augmentation / "cation.csv",
        [],
    )
    _write_role_csv(
        augmentation / "anion.csv",
        [("CC(C)[CH2][AlH-](<-[CH2](C)C)[S](C)(=O)=O", -1)],
    )
    _write_role_csv(
        augmentation / "molecule.csv",
        [("CCCCCCCC", 0), ("P", 0)],
    )

    real_ipc = features_module._calculate_ipc

    def fake_ipc(mol):
        if Chem.MolToSmiles(mol, canonical=True) == "P":
            return IPC_SQUARE_OVERFLOW_LIMIT * 2
        return real_ipc(mol)

    descriptor_smiles = []

    def fake_descriptors(mol, names):
        descriptor_smiles.append(Chem.MolToSmiles(mol, canonical=True))
        return np.arange(len(names), dtype=np.float64)

    monkeypatch.setattr(features_module, "_calculate_ipc", fake_ipc)
    monkeypatch.setattr(prepare_module, "calculate_descriptors", fake_descriptors)
    summary = prepare_corpus(
        PretrainConfig(
            data=DataConfig(
                stage1_dir=stage1,
                artifacts_dir=artifacts,
                valid_fraction=0.25,
                seed=7,
                max_smiles_tokens=8,
                include_augmentation=True,
                shard_size=4,
            )
        )
    )

    assert summary["total"] == 12
    assert summary["excluded_entities"] == 3
    assert "P" not in descriptor_smiles
    assert "CCCCCCCC" not in descriptor_smiles
    assert not any("AlH" in smiles for smiles in descriptor_smiles)
    with (artifacts / "excluded_entities.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        excluded = list(csv.DictReader(handle))
    reasons = {
        row["canonical_smiles"]: set(row["exclusion_reasons"].split(";"))
        for row in excluded
    }
    assert reasons["P"] == {"ipc_square_overflow"}
    assert reasons["CCCCCCCC"] == {"smiles_overlength"}
    dative_reasons = next(
        value for smiles, value in reasons.items() if "AlH" in smiles
    )
    assert "unsupported_bcut_bond_type" in dative_reasons
    metadata = json.loads((artifacts / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["quality_control"]["excluded"]["total"] == 3
    assert metadata["augmentation_audit"]["anion"]["retained"] == 0
    assert metadata["augmentation_audit"]["neutral"]["retained"] == 0

def test_full_augmentation_ingestion_dedup_leakage_and_audit(tmp_path):
    stage1 = tmp_path / "stage1"
    augmentation = stage1 / "augmentation"
    artifacts = tmp_path / "artifacts"
    augmentation.mkdir(parents=True)
    originals = {
        "cation": [("[Na+]", 1), ("C[NH3+]", 1), ("C[NH2+]C", 1), ("[K+]", 1)],
        "anion": [("[Cl-]", -1), ("[Br-]", -1), ("C(=O)[O-]", -1), ("[I-]", -1)],
        "molecule": [("CCO", 0), ("O", 0), ("N", 0), ("CC", 0)],
    }
    for name, rows in originals.items():
        _write_role_csv(stage1 / f"{name}.csv", rows)
        charge = rows[0][1]
        all_seeds = ";".join(row[0] for row in rows)
        candidate = (
            "C1CC1"
            if name == "molecule"
            else ("C[NH+](C)C" if name == "cation" else "C[S-]")
        )
        _write_role_csv(
            augmentation / f"{name}.csv",
            [
                (candidate, charge, all_seeds),
                (rows[0][0], charge, "unrelated"),
                (candidate, charge, "unrelated"),
                (candidate, charge, "unrelated"),
                ("CCC" if name == "molecule" else ("CC[NH2+]C" if name == "cation" else "CC[S-]"), charge, "unrelated"),
                ("CCCC" if name == "molecule" else ("CCC[NH2+]C" if name == "cation" else "CCC[S-]"), charge, "unrelated"),
            ],
        )

    config = PretrainConfig(
        data=DataConfig(
            stage1_dir=stage1,
            artifacts_dir=artifacts,
            valid_fraction=0.25,
            seed=7,
            include_augmentation=True,
        )
    )
    summary = prepare_corpus(config)
    assert summary["augmented"] == 9
    metadata = json.loads((artifacts / "metadata.json").read_text(encoding="utf-8"))
    for role in ("cation", "anion", "neutral"):
        assert metadata["augmentation_audit"][role] == {
            "included": True,
            "source_rows": 6,
            "excluded_valid_seed": 1,
            "excluded_overlap": 1,
            "excluded_duplicate": 1,
            "eligible": 3,
            "excluded_qc": 0,
            "retained": 3,
        }
    audit = json.loads(
        (artifacts / "augmentation_audit.json").read_text(encoding="utf-8")
    )
    assert audit["roles"] == metadata["augmentation_audit"]
    with (artifacts / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    valid_smiles = {
        row["canonical_smiles"] for row in rows if row["split"] == "valid"
    }
    augmented_seeds = {
        seed
        for row in rows
        if row["is_augmented"] == "1"
        for seed in row["seed_smiles"].split(";")
        if seed
    }
    assert valid_smiles.isdisjoint(augmented_seeds)

def test_prepare_workers_preserve_artifact_semantics(tmp_path):
    stage1 = tmp_path / "stage1"
    augmentation = stage1 / "augmentation"
    augmentation.mkdir(parents=True)
    originals = {
        "cation": [("[Na+]", 1), ("C[NH3+]", 1), ("[K+]", 1), ("C[NH2+]C", 1)],
        "anion": [("[Cl-]", -1), ("C(=O)[O-]", -1), ("[Br-]", -1), ("C[S-]", -1)],
        "molecule": [("CCO", 0), ("O", 0), ("N", 0), ("CC", 0)],
    }
    additions = {
        "cation": [("CC[NH2+]C", 1, "unrelated")],
        "anion": [("CC[S-]", -1, "unrelated")],
        "molecule": [("CCC", 0, "unrelated")],
    }
    for role, rows in originals.items():
        _write_role_csv(stage1 / f"{role}.csv", rows)
        _write_role_csv(augmentation / f"{role}.csv", additions[role])

    artifact_dirs = []
    for workers in (1, 4):
        artifacts = tmp_path / f"artifacts_{workers}"
        prepare_corpus(
            PretrainConfig(
                data=DataConfig(
                    stage1_dir=stage1,
                    artifacts_dir=artifacts,
                    valid_fraction=0.25,
                    seed=11,
                    include_augmentation=True,
                    shard_size=3,
                ),
                preparation=PreparationConfig(
                    workers=workers,
                    catalog_batch_size=2,
                    qc_batch_size=2,
                    tokenizer_batch_size=2,
                    descriptor_batch_size=2,
                ),
            )
        )
        artifact_dirs.append(artifacts)

    left, right = artifact_dirs
    for filename in (
        "tokenizer.json",
        "descriptor_schema.json",
        "descriptor_scaler.json",
        "manifest.csv",
        "excluded_entities.csv",
        "augmentation_audit.json",
    ):
        assert (left / filename).read_bytes() == (right / filename).read_bytes()
    np.testing.assert_array_equal(
        np.load(left / "train_index.npy"), np.load(right / "train_index.npy")
    )
    np.testing.assert_array_equal(
        np.load(left / "valid_index.npy"), np.load(right / "valid_index.npy")
    )
    for split in ("train", "valid"):
        left_dataset = PreparedCorpusDataset(left, split)
        right_dataset = PreparedCorpusDataset(right, split)
        assert len(left_dataset) == len(right_dataset)
        for index in range(len(left_dataset)):
            left_sample = left_dataset[index]
            right_sample = right_dataset[index]
            assert left_sample.keys() == right_sample.keys()
            for key in left_sample:
                if isinstance(left_sample[key], dict):
                    assert left_sample[key].keys() == right_sample[key].keys()
                    for name in left_sample[key]:
                        assert np.array_equal(
                            left_sample[key][name].numpy(),
                            right_sample[key][name].numpy(),
                        )
                elif hasattr(left_sample[key], "numpy"):
                    assert np.array_equal(
                        left_sample[key].numpy(), right_sample[key].numpy()
                    )
                else:
                    assert left_sample[key] == right_sample[key]

def test_stage1_shard_fingerprint_uint8_preserves_every_bit():
    fingerprint = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    sample = {
        "sample_id": "neutral_00000001",
        "role_id": 2,
        "token_ids": torch.tensor([2, 3]),
        "atom_categorical": torch.zeros((1, 1), dtype=torch.long),
        "atom_continuous": torch.zeros((1, 1)),
        "bond_categorical": torch.zeros((0, 1), dtype=torch.long),
        "bond_index": torch.zeros((2, 0), dtype=torch.long),
        "descriptors": torch.zeros(1),
        "descriptor_valid": torch.ones(1, dtype=torch.bool),
        "fingerprints": {"morgan": fingerprint},
        "canonical_smiles": "CC",
        "sources": ("test",),
        "split": "train",
        "is_augmented": False,
        "seed_smiles": (),
    }
    compact = _stage1_shard_sample(sample)
    assert "canonical_smiles" not in compact
    stored = compact["fingerprints"]["morgan"]
    assert stored.dtype == torch.uint8
    assert torch.equal(stored.float(), fingerprint)

def test_ais_round_trip_and_vocabulary_save_load(tmp_path):
    import atomInSmiles

    smiles = "NCC(=O)O"
    encoded = " ".join(ais_tokenize(smiles))
    decoded = atomInSmiles.decode(encoded)
    assert Chem.MolToSmiles(Chem.MolFromSmiles(decoded)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(smiles)
    )

    vocabulary = SmilesTokenizer.fit([smiles, "CCO"], backend="ais")
    path = tmp_path / "tokenizer.json"
    vocabulary.save(path)
    loaded = SmilesTokenizer.load(path)
    assert loaded.tokens == vocabulary.tokens
    assert loaded.encode(smiles, max_length=32) == vocabulary.encode(
        smiles, max_length=32
    )

def test_smiles_masking_uses_bert_replacement_distribution():
    vocabulary = SmilesTokenizer.fit(["CCO", "NCC"], backend="ais")
    token_ids = np.full(10_002, vocabulary.token_to_id[ais_tokenize("CCO")[0]])
    token_ids[0] = vocabulary.cls_id
    token_ids[-1] = vocabulary.sep_id
    import torch

    token_ids_tensor = torch.from_numpy(token_ids).long()
    positions = torch.arange(1, 10_001)
    corrupted, labels = mask_smiles_tokens(
        token_ids_tensor,
        positions,
        ratio=1.0,
        vocabulary=vocabulary,
        generator=torch.Generator().manual_seed(11),
        drop_entire_modality=False,
    )
    assert (labels != -100).sum().item() == 10_000
    assert labels[0].item() == labels[-1].item() == -100
    mask_fraction = (corrupted[positions] == vocabulary.mask_id).float().mean().item()
    assert 0.77 < mask_fraction < 0.83

def test_descriptor_schema_clean_pruned_groups_and_save_load(tmp_path):
    values = np.asarray(
        [
            [1.0, 2.0, 1.0, np.nan, 4.0, 1.01],
            [2.0, 3.0, 2.0, np.nan, 4.0, 2.01],
            [3.0, 4.0, 3.0, np.nan, 4.0, 3.01],
            [4.0, 5.0, 4.0, np.nan, 4.0, 4.01],
        ]
    )
    names = ("MolWt", "Chi0", "duplicate", "missing", "constant", "Chi1")
    clean = DescriptorSchema.fit(values, names, "clean", 8)
    assert clean.selected_names == ("MolWt", "Chi0", "Chi1")
    assert clean.removal_reasons["missing"] == "all_non_finite"
    assert clean.removal_reasons["constant"] == "zero_variance"
    assert clean.removal_reasons["duplicate"] == "duplicate_of:MolWt"
    assert len(clean.group_indices) == 8
    assert clean.semantic_mapping_version == "rdkit-217-v1"
    assert len(clean.raw_semantic_groups) == len(names)

    pruned = DescriptorSchema.fit(values, names, "pruned", 12, 0.98)
    assert pruned.selected_dim == 1
    assert len(pruned.correlation_clusters) == 1
    assert pruned.cluster_representatives == ("MolWt",)
    path = tmp_path / "schema.json"
    pruned.save(path)
    loaded = DescriptorSchema.load(path, expected_raw_names=names)
    assert loaded == pruned

# --- Scientific model behavior ---

def test_all_five_modalities_use_element_role_weights_and_component_means(
    tiny_config,
    tiny_samples,
) -> None:
    vocabulary, samples = tiny_samples
    batch = MultimodalCollator(
        vocabulary, tiny_config.masking, seed=tiny_config.data.seed
    )(samples)
    model = MultimodalPretrainModel(tiny_config, vocabulary)
    output = model(batch)
    role_weights = torch.tensor(tiny_config.loss.role_weights)

    expected: dict[str, list[tuple[torch.Tensor, ...]]] = {}
    smiles_mask = batch.masks.smiles_labels != -100
    token_roles = batch.roles[:, None].expand_as(smiles_mask)
    expected["smiles"] = [
        _weighted_component(
            F.cross_entropy(
                output.logits["smiles"][smiles_mask],
                batch.masks.smiles_labels[smiles_mask],
                reduction="none",
            ),
            token_roles[smiles_mask],
            role_weights,
        )
    ]

    atom_mask = batch.masks.atom_mask
    atom_roles = batch.roles[batch.graphs.atom_batch][atom_mask]
    expected["atom"] = [
        _weighted_component(
            F.cross_entropy(
                output.logits["atom"][name][atom_mask],
                batch.graphs.atom_categorical[atom_mask, column],
                reduction="none",
            ),
            atom_roles,
            role_weights,
        )
        for column, name in enumerate(ATOM_FEATURE_NAMES)
    ]

    bond_mask = batch.masks.bond_mask
    bond_roles = batch.roles[batch.graphs.bond_batch][bond_mask]
    expected["bond"] = [
        _weighted_component(
            F.cross_entropy(
                output.logits["bond"][name][bond_mask],
                batch.graphs.bond_categorical[bond_mask, column],
                reduction="none",
            ),
            bond_roles,
            role_weights,
        )
        for column, name in enumerate(BOND_FEATURE_NAMES)
    ]

    descriptor_mask = batch.masks.descriptor_loss_mask
    descriptor_roles = batch.roles[:, None].expand_as(descriptor_mask)
    expected["descriptor"] = [
        _weighted_component(
            F.smooth_l1_loss(
                output.logits["descriptor"][descriptor_mask],
                batch.descriptors[descriptor_mask],
                reduction="none",
            ),
            descriptor_roles[descriptor_mask],
            role_weights,
        )
    ]

    expected["fingerprint"] = []
    for family, logits in output.logits["fingerprint"].items():
        loss_mask = batch.masks.fingerprint_loss_mask[family]
        fingerprint_roles = batch.roles[:, None].expand_as(loss_mask)
        expected["fingerprint"].append(
            _weighted_component(
                F.binary_cross_entropy_with_logits(
                    logits[loss_mask],
                    batch.fingerprints.values[family][loss_mask],
                    reduction="none",
                ),
                fingerprint_roles[loss_mask],
                role_weights,
            )
        )

    for modality, components in expected.items():
        statistics = output.loss_statistics[modality]
        assert torch.allclose(
            statistics.numerators,
            torch.stack([component[0] for component in components]),
        )
        assert torch.equal(
            statistics.denominators,
            torch.stack([component[1] for component in components]),
        )
        assert torch.allclose(output.losses[modality], statistics.mean())

def test_end_to_end_forward_backward_has_five_losses_and_shared_gradients(
    tiny_config,
    tiny_samples,
):
    vocabulary, samples = tiny_samples
    batch = MultimodalCollator(
        vocabulary, tiny_config.masking, seed=tiny_config.data.seed
    )(samples)
    model = MultimodalPretrainModel(tiny_config, vocabulary)
    output = model(batch)
    assert set(output.losses) == {
        "smiles",
        "descriptor",
        "atom",
        "bond",
        "fingerprint",
    }
    assert all(torch.isfinite(loss) for loss in output.losses.values())
    assert output.logits["smiles"].shape[:2] == batch.token_ids.shape
    assert output.logits["descriptor"].shape == (3, 217)
    assert output.logits["fingerprint"]["morgan"].shape == (3, 2048)
    assert output.logits["fingerprint"]["maccs"].shape == (3, 167)
    assert output.fused_cls.shape == (3, tiny_config.model.d_model)

    output.loss.backward()
    required_parameters = [
        model.smiles_encoder.token_embedding.weight,
        model.graph_encoder.atom_mask_feature,
        model.graph_encoder.bond_mask_feature,
        model.descriptor_encoder.group_encoders[4].input_projection[0].weight,
        model.fingerprint_encoder.chunk_encoder[0].weight,
        model.fusion.modality_embedding.weight,
        model.fusion.role_embedding.weight,
        model.smiles_head.bias,
        model.atom_heads["atomic_number"].weight,
        model.bond_heads["bond_type"].weight,
        model.descriptor_heads[4].weight,
        model.fingerprint_heads["morgan"].weight,
    ]
    assert all(parameter.grad is not None for parameter in required_parameters)
