from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem

from stage1.config import DataConfig, PreparationConfig, PretrainConfig
import stage1.features as features_module
import stage1.prepare as prepare_module
from stage1.data import (
    CORPUS_FORMAT_VERSION,
    CORPUS_KIND,
    PreparedCorpusDataset,
)
from stage1.features import IPC_SQUARE_OVERFLOW_LIMIT, inspect_entity_qc
from stage1.masking import MultimodalPacker
from stage1.prepare import (
    _descriptor_batch,
    _csv_data_row_count,
    _entity_qc_batch,
    _ordered_batch_map,
    _stage1_shard_sample,
    preparation_source_paths,
    prepare_corpus,
)
from stage1.descriptors import rdkit_descriptor_names
from common.data_identity import write_data_identity


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


def test_prepare_progress_counts_rows_and_only_enabled_sources(tmp_path):
    stage1 = tmp_path / "stage1"
    stage1.mkdir()
    path = stage1 / "cation.csv"
    _write_role_csv(path, [("CC", 0), ("C\nC", 0)])

    assert _csv_data_row_count(path) == 2
    config = DataConfig(
        stage1_dir=stage1,
        include_augmentation=True,
    )
    assert preparation_source_paths(config) == [
        stage1 / "cation.csv",
        stage1 / "anion.csv",
        stage1 / "molecule.csv",
        stage1 / "augmentation" / "cation.csv",
        stage1 / "augmentation" / "anion.csv",
        stage1 / "augmentation" / "molecule.csv",
    ]


def test_prepare_uses_new_original_sources_and_sharded_artifacts(tmp_path):
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
    with pytest.raises(ValueError, match="Legacy corpus.pt"):
        legacy = artifacts / "corpus.pt"
        legacy.touch()
        PreparedCorpusDataset(legacy)
    for unsupported_version in (1, 3):
        metadata["format_version"] = unsupported_version
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(ValueError, match="create corpus v2"):
            PreparedCorpusDataset(artifacts)
    metadata["format_version"] = CORPUS_FORMAT_VERSION
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    shard = artifacts / dataset.shards[0]["path"]
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="Shard hash mismatch"):
        PreparedCorpusDataset(artifacts, "train")[0]


def test_entity_qc_records_multiple_reasons_and_keeps_isolated_hydrogen(monkeypatch):
    record = {
        "canonical_smiles": "CC(C)[CH2][AlH-](<-[CH2](C)C)[S](C)(=O)=O",
        "role": "anion",
        "split": "train",
        "sources": ("augmentation/anion",),
        "is_augmented": True,
    }
    monkeypatch.setattr(
        features_module,
        "_calculate_ipc",
        lambda mol: IPC_SQUARE_OVERFLOW_LIMIT * 2,
    )
    inspected = inspect_entity_qc(record)
    assert inspected.reasons == [
        "unsupported_bcut_bond_type",
        "ipc_square_overflow",
    ]
    assert inspected.unsupported_bond_types == ("DATIVE",)

    monkeypatch.setattr(features_module, "_calculate_ipc", lambda mol: float("nan"))
    nonfinite = inspect_entity_qc({**record, "canonical_smiles": "CC"})
    assert nonfinite.reasons == ["ipc_nonfinite"]

    monkeypatch.setattr(features_module, "_calculate_ipc", lambda mol: 0.0)
    quadruple = inspect_entity_qc(
        {**record, "canonical_smiles": "[Cr]$[Cr]"}
    )
    assert quadruple.reasons == ["unsupported_bcut_bond_type"]
    assert quadruple.unsupported_bond_types == ("QUADRUPLE",)

    hydrogen = inspect_entity_qc({**record, "canonical_smiles": "[HH]"})
    assert hydrogen.reasons == []


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


def test_descriptor_workers_are_bitwise_equal_and_qc_errors_have_context():
    task = ([(0, "CCO"), (1, "C[NH3+]")], rdkit_descriptor_names())
    serial_rows, serial_values = _descriptor_batch(task)
    parallel_rows, parallel_values = next(
        _ordered_batch_map(_descriptor_batch, [task], workers=2)
    )
    assert serial_rows == parallel_rows
    np.testing.assert_array_equal(serial_values, parallel_values)
    with pytest.raises(RuntimeError, match=r"entity_qc record id=7 smiles='invalid'"):
        _entity_qc_batch([(7, "invalid")])


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


def test_prepare_reuses_source_identity_and_writes_performance(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    stage1 = repository / "data/stage1"
    artifacts = repository / "outputs/prepare/artifacts"
    performance_path = repository / "outputs/prepare/performance.json"
    stage1.mkdir(parents=True)
    for role, rows in {
        "cation": [("[Na+]", 1), ("C[NH3+]", 1)],
        "anion": [("[Cl-]", -1), ("C(=O)[O-]", -1)],
        "molecule": [("CCO", 0), ("O", 0)],
    }.items():
        _write_role_csv(stage1 / f"{role}.csv", rows)
    sources = [stage1 / name for name in ("cation.csv", "anion.csv", "molecule.csv")]
    identity = write_data_identity(repository, "stage1", sources)
    real_sha256 = prepare_module.sha256_file

    def reject_source_rehash(path):
        if Path(path) in sources:
            raise AssertionError("source file was hashed twice")
        return real_sha256(path)

    monkeypatch.setattr(prepare_module, "sha256_file", reject_source_rehash)
    monkeypatch.setattr(
        prepare_module,
        "_csv_data_row_count",
        lambda path: (_ for _ in ()).throw(AssertionError("source rows counted twice")),
    )
    data_config = DataConfig(
        stage1_dir=stage1,
        artifacts_dir=artifacts,
        valid_fraction=0.5,
        shard_size=2,
    )
    prepare_corpus(
        PretrainConfig(
            data=data_config,
            preparation=PreparationConfig(descriptor_batch_size=2),
        ),
        source_identity=identity,
        performance_path=performance_path,
        input_identity_elapsed_seconds=1.25,
    )
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    assert set(performance["phases"]) == {
        "input_identity",
        "catalog",
        "entity_qc",
        "tokenizer",
        "descriptors",
        "descriptor_fit",
        "shards",
        "publication",
    }
    assert performance["preparation"]["descriptor_batch_size"] == 2
    assert performance["phases"]["input_identity"]["processed"] == 6
    assert performance["total_elapsed_seconds"] >= 1.25
    metadata = json.loads((artifacts / "metadata.json").read_text())
    assert "performance.json" not in metadata["artifact_hashes"]
    prepare_corpus(
        PretrainConfig(
            data=data_config,
            preparation=PreparationConfig(workers=2, descriptor_batch_size=3),
        ),
        source_identity=identity,
        performance_path=performance_path,
    )
    reused = json.loads(performance_path.read_text(encoding="utf-8"))
    assert reused["preparation"]["workers"] == 2
    assert reused["preparation"]["descriptor_batch_size"] == 3
    assert all(
        phase["reused"]
        for name, phase in reused["phases"].items()
        if name != "input_identity"
    )
    invalid_identity = {**identity, "files": identity["files"][:-1]}
    with pytest.raises(ValueError, match="file set does not match"):
        prepare_corpus(
            PretrainConfig(data=data_config), source_identity=invalid_identity
        )


import numpy as np
import pytest
from rdkit import Chem

from stage1.descriptors import DescriptorSchema, DescriptorStandardizer
from stage1.masking import mask_smiles_tokens
from stage1.tokenizer import SmilesTokenizer, ais_tokenize


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
    with pytest.raises(ValueError, match="exceeding"):
        loaded.encode(smiles, max_length=3)


def test_ais_min_frequency_filters_counts_and_preserves_frequency_order():
    counts = Counter({"rare": 1, "common-b": 3, "common-a": 3, "twice": 2})
    expected = {
        1: ("common-a", "common-b", "twice", "rare"),
        2: ("common-a", "common-b", "twice"),
        3: ("common-a", "common-b"),
    }
    for minimum, learned in expected.items():
        tokenizer = SmilesTokenizer.fit_ais_counts(
            counts, vocab_size=32, min_frequency=minimum
        )
        assert tokenizer.tokens[5:] == learned


def test_token_to_id_is_cached():
    tokenizer = SmilesTokenizer.fit(["CCO"], backend="ais")
    assert tokenizer.token_to_id is tokenizer.token_to_id


def test_token_count_includes_special_tokens_and_matches_encode_boundary():
    tokenizer = SmilesTokenizer.fit(["C" * 255], backend="ais")
    assert tokenizer.token_count("C" * 254) == 256
    assert len(tokenizer.encode("C" * 254, max_length=256)) == 256
    assert tokenizer.token_count("C" * 255) == 257
    with pytest.raises(ValueError, match="257 tokens"):
        tokenizer.encode("C" * 255, max_length=256)


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


def test_descriptor_standardizer_is_finite_aware_and_round_trips(tmp_path):
    values = np.asarray(
        [
            [1.0, 4.0, np.nan],
            [3.0, 4.0, 9.0],
            [100.0, 4.0, 11.0],
        ]
    )
    standardizer = DescriptorStandardizer.fit(values[:2], ("a", "b", "c"))
    assert standardizer.means.tolist() == [2.0, 4.0, 9.0]
    assert standardizer.scales.tolist() == [1.0, 1.0, 1.0]
    assert standardizer.finite_counts.tolist() == [2, 2, 1]

    transformed, valid = standardizer.transform(values)
    assert transformed[0, 2] == 0.0
    assert not valid[0, 2]
    assert transformed[2, 0] == 98.0

    path = tmp_path / "scaler.json"
    standardizer.save(path)
    loaded = DescriptorStandardizer.load(path, expected_names=("a", "b", "c"))
    np.testing.assert_allclose(loaded.means, standardizer.means)
    with pytest.raises(ValueError, match="names/order"):
        DescriptorStandardizer.load(path, expected_names=("b", "a", "c"))

    no_training_value = DescriptorStandardizer.fit(
        np.asarray([[1.0, np.nan], [3.0, np.nan]]), ("a", "missing")
    )
    transformed, valid = no_training_value.transform(np.asarray([[2.0, 8.0]]))
    assert transformed[0, 1] == 0.0
    assert not valid[0, 1]


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
    with pytest.raises(ValueError, match="names/order"):
        DescriptorSchema.load(path, expected_raw_names=names[::-1])


@pytest.mark.parametrize("backend", ["bpe", "spe", "ape"])
def test_data_driven_tokenizers_share_budget_and_round_trip_artifact(
    backend, tmp_path
):
    if backend == "bpe":
        pytest.importorskip("tokenizers")
    elif backend == "spe":
        pytest.importorskip("SmilesPE")
    else:
        pytest.importorskip("apetokenizer")
    corpus = ["CCO", "CCN", "CCC", "C=O"]
    tokenizer = SmilesTokenizer.fit(
        corpus, backend=backend, vocab_size=32, min_frequency=2
    )
    encoded = tokenizer.encode("CCO", max_length=32)
    assert encoded[0] == tokenizer.cls_id
    assert encoded[-1] == tokenizer.sep_id
    assert len(tokenizer.tokens) <= 32
    path = tmp_path / f"{backend}.json"
    tokenizer.save(path)
    loaded = SmilesTokenizer.load(path)
    assert loaded.encode("CCO", max_length=32) == encoded
    assert loaded.backend == backend
    assert loaded.backend_version == tokenizer.backend_version
