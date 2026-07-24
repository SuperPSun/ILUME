from __future__ import annotations

import csv
import json
from collections import Counter

import numpy as np
import pytest
from rdkit import Chem

from ilume_pretrain.config import DataConfig, PretrainConfig
from ilume_pretrain import data as data_module
from ilume_pretrain.data import (
    CORPUS_FORMAT_VERSION,
    IPC_SQUARE_OVERFLOW_LIMIT,
    PreparedCorpusDataset,
    _inspect_entity_qc,
    prepare_corpus,
)
from ilume_pretrain.sampler import (
    RoleBalancedSampler,
    minimum_samples_for_coverage,
)


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
    assert (artifacts / "corpus_index.json").is_file()
    assert (artifacts / "descriptor_schema.json").is_file()
    assert (artifacts / "excluded_entities.csv").is_file()
    assert len(list((artifacts / "shards").glob("*.pt"))) == 6
    metadata_path = artifacts / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["format_version"] == CORPUS_FORMAT_VERSION == 3
    assert "excluded_entities.csv" in metadata["artifact_hashes"]
    dataset = PreparedCorpusDataset(artifacts, "train")
    assert len(dataset) == 3
    with pytest.raises(ValueError, match="Legacy corpus.pt"):
        legacy = artifacts / "corpus.pt"
        legacy.touch()
        PreparedCorpusDataset(legacy)
    metadata["format_version"] = 2
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported corpus artifact format"):
        PreparedCorpusDataset(artifacts)
    metadata["format_version"] = CORPUS_FORMAT_VERSION
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    shard = artifacts / dataset.entries[0]["shard"]
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="Shard hash mismatch"):
        dataset[0]


def test_entity_qc_records_multiple_reasons_and_keeps_isolated_hydrogen(monkeypatch):
    record = {
        "canonical_smiles": "CC(C)[CH2][AlH-](<-[CH2](C)C)[S](C)(=O)=O",
        "role": "anion",
        "split": "train",
        "sources": ("augmentation/anion",),
        "is_augmented": True,
    }
    monkeypatch.setattr(
        data_module,
        "_calculate_ipc",
        lambda mol: IPC_SQUARE_OVERFLOW_LIMIT * 2,
    )
    inspected = _inspect_entity_qc(record)
    assert inspected.reasons == [
        "unsupported_bcut_bond_type",
        "ipc_square_overflow",
    ]
    assert inspected.unsupported_bond_types == ("DATIVE",)

    monkeypatch.setattr(data_module, "_calculate_ipc", lambda mol: float("nan"))
    nonfinite = _inspect_entity_qc({**record, "canonical_smiles": "CC"})
    assert nonfinite.reasons == ["ipc_nonfinite"]

    monkeypatch.setattr(data_module, "_calculate_ipc", lambda mol: 0.0)
    quadruple = _inspect_entity_qc(
        {**record, "canonical_smiles": "[Cr]$[Cr]"}
    )
    assert quadruple.reasons == ["unsupported_bcut_bond_type"]
    assert quadruple.unsupported_bond_types == ("QUADRUPLE",)

    hydrogen = _inspect_entity_qc({**record, "canonical_smiles": "[HH]"})
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
        augmentation / "anion.csv",
        [("CC(C)[CH2][AlH-](<-[CH2](C)C)[S](C)(=O)=O", -1)],
    )
    _write_role_csv(
        augmentation / "molecule.csv",
        [("CCCCCCCC", 0), ("P", 0)],
    )

    real_ipc = data_module._calculate_ipc

    def fake_ipc(mol):
        if Chem.MolToSmiles(mol, canonical=True) == "P":
            return IPC_SQUARE_OVERFLOW_LIMIT * 2
        return real_ipc(mol)

    descriptor_smiles = []

    def fake_descriptors(mol, names):
        descriptor_smiles.append(Chem.MolToSmiles(mol, canonical=True))
        return np.arange(len(names), dtype=np.float64)

    monkeypatch.setattr(data_module, "_calculate_ipc", fake_ipc)
    monkeypatch.setattr(data_module, "calculate_descriptors", fake_descriptors)
    summary = prepare_corpus(
        PretrainConfig(
            data=DataConfig(
                stage1_dir=stage1,
                artifacts_dir=artifacts,
                valid_fraction=0.25,
                seed=7,
                max_smiles_tokens=8,
                augmentation={"cation": 0, "anion": "all", "neutral": "all"},
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
    assert metadata["augmentation_audit"]["anion"]["selected_after_qc"] == 0
    assert metadata["augmentation_audit"]["neutral"]["selected_after_qc"] == 0


def test_augmentation_ratio_and_validation_seed_descendant_isolation(tmp_path):
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
        # First row is guaranteed to be excluded because every original is a seed;
        # the remaining rows are eligible and the 1x cap selects train_count rows.
        all_seeds = ";".join(row[0] for row in rows)
        _write_role_csv(
            augmentation / f"{name}.csv",
            [
                ("C1CC1" if name == "molecule" else ("C[NH+](C)C" if name == "cation" else "C[S-]"), charge, all_seeds),
                ("CCC" if name == "molecule" else ("CC[NH2+]C" if name == "cation" else "CC[S-]"), charge, "unrelated"),
                ("CCCC" if name == "molecule" else ("CCC[NH2+]C" if name == "cation" else "CCC[S-]"), charge, "unrelated"),
                ("CCCCC" if name == "molecule" else ("CCCC[NH2+]C" if name == "cation" else "CCCC[S-]"), charge, "unrelated"),
            ],
        )

    config = PretrainConfig(
        data=DataConfig(
            stage1_dir=stage1,
            artifacts_dir=artifacts,
            valid_fraction=0.25,
            seed=7,
            augmentation={"cation": 1.0, "anion": 1.0, "neutral": 1.0},
        )
    )
    summary = prepare_corpus(config)
    assert summary["augmented"] == 9
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


def test_role_balanced_sampler_uses_45_45_10_and_checks_coverage():
    role_ids = [0] * 2 + [1] * 3 + [2] * 4
    first = list(RoleBalancedSampler(role_ids, num_samples=20, seed=9))
    second = list(RoleBalancedSampler(role_ids, num_samples=20, seed=9))
    counts = Counter(role_ids[index] for index in first)
    assert counts == {0: 9, 1: 9, 2: 2}
    assert first == second
    assert minimum_samples_for_coverage((2, 3, 4)) == 40
    with pytest.raises(ValueError, match="cannot cover"):
        RoleBalancedSampler(
            role_ids,
            num_samples=20,
            seed=9,
            require_full_coverage=True,
        )


def test_sampler_resume_offset_replays_remaining_sequence():
    role_ids = [0] * 4 + [1] * 4 + [2] * 2
    sampler = RoleBalancedSampler(role_ids, num_samples=40, seed=5)
    complete = list(sampler)
    sampler.set_start_offset(13)
    assert list(sampler) == complete[13:]
    restored = RoleBalancedSampler(role_ids, num_samples=40, seed=5)
    restored.load_state_dict(sampler.state_dict())
    assert list(restored) == complete[13:]


def test_sampler_advances_each_role_one_shard_at_a_time():
    role_ids = [0] * 4 + [1] * 4 + [2] * 4
    shard_ids = [
        "cation_a",
        "cation_a",
        "cation_b",
        "cation_b",
        "anion_a",
        "anion_a",
        "anion_b",
        "anion_b",
        "neutral_a",
        "neutral_a",
        "neutral_b",
        "neutral_b",
    ]
    sampler = RoleBalancedSampler(
        role_ids,
        num_samples=40,
        seed=4,
        require_full_coverage=True,
        shard_ids=shard_ids,
    )
    selected = list(sampler)
    for role in range(3):
        role_draws = [index for index in selected if role_ids[index] == role]
        first_cycle = role_draws[:4]
        assert len(set(first_cycle)) == 4
        shard_sequence = [shard_ids[index] for index in first_cycle]
        changes = sum(
            left != right
            for left, right in zip(shard_sequence, shard_sequence[1:])
        )
        assert changes == 1
