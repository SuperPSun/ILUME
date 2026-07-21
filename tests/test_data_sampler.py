from __future__ import annotations

import csv
from collections import Counter

from ilume_pretrain.config import DataConfig
from ilume_pretrain.data import PreparedCorpusDataset, prepare_corpus
from ilume_pretrain.sampler import RoleBalancedSampler


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
        for smiles, charge in rows:
            writer.writerow([smiles, charge, "test", "", "", "", ""])


def test_prepare_uses_selected_sources_deduplicates_and_ignores_il(tmp_path):
    stage1 = tmp_path / "stage1"
    artifacts = tmp_path / "artifacts"
    stage1.mkdir()
    _write_role_csv(stage1 / "cation.csv", [("[Na+]", 1), ("C[NH3+]", 1)])
    _write_role_csv(stage1 / "anion.csv", [("[Cl-]", -1), ("C(=O)[O-]", -1)])
    _write_role_csv(stage1 / "molecule.csv", [("CCO", 0), ("O", 0)])
    _write_role_csv(stage1 / "solute.csv", [("CCO", 0), ("N", 0)])
    _write_role_csv(stage1 / "solvent.csv", [("O", 0)])
    (stage1 / "IL.csv").write_text(
        "cation,anion\n[K+],[Br-]\n", encoding="utf-8"
    )

    summary = prepare_corpus(
        DataConfig(
            stage1_dir=stage1,
            artifacts_dir=artifacts,
            valid_fraction=0.5,
            seed=3,
        )
    )
    assert summary == {
        "total": 7,
        "train": 3,
        "valid": 4,
        "cation": 2,
        "anion": 2,
        "neutral": 3,
    }
    with (artifacts / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert "[K+]" not in {row["canonical_smiles"] for row in rows}
    ethanol = next(row for row in rows if row["canonical_smiles"] == "CCO")
    assert ethanol["sources"] == "molecule;solute"
    assert (artifacts / "tokenizer.json").is_file()
    assert (artifacts / "descriptor_scaler.json").is_file()
    assert PreparedCorpusDataset(artifacts / "corpus.pt", "train").__len__() == 3


def test_role_balanced_sampler_has_exact_quota_and_is_reproducible():
    role_ids = [0] * 2 + [1] * 3 + [2] * 4
    first = list(RoleBalancedSampler(role_ids, num_samples=20, seed=9))
    second = list(RoleBalancedSampler(role_ids, num_samples=20, seed=9))
    counts = Counter(role_ids[index] for index in first)
    assert counts == {0: 10, 1: 8, 2: 2}
    assert first == second
