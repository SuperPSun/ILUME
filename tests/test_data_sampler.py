from __future__ import annotations

import csv
from collections import Counter

import pytest

from ilume_pretrain.config import DataConfig, PretrainConfig
from ilume_pretrain.data import PreparedCorpusDataset, prepare_corpus
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
    assert (artifacts / "corpus_index.json").is_file()
    assert (artifacts / "descriptor_schema.json").is_file()
    assert len(list((artifacts / "shards").glob("*.pt"))) == 6
    dataset = PreparedCorpusDataset(artifacts, "train")
    assert len(dataset) == 3
    with pytest.raises(ValueError, match="Legacy corpus.pt"):
        legacy = artifacts / "corpus.pt"
        legacy.touch()
        PreparedCorpusDataset(legacy)
    shard = artifacts / dataset.entries[0]["shard"]
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="Shard hash mismatch"):
        dataset[0]


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
