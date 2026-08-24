from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from common.io import sha256_file
from stage2.atom_evaluation import (
    PARTIAL_CHARGE_PREDICTION_FIELDS,
    build_partial_charge_benchmark,
    public_partial_charge_score,
    score_partial_charge_predictions,
    write_partial_charge_predictions,
)
from stage2.atom_targets import PARTIAL_CHARGE_MAPPING_CONTRACT


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


def test_partial_charge_benchmark_mapping_subsets_and_identity(tmp_path: Path) -> None:
    test, manifest = _benchmark_files(tmp_path)
    benchmark = build_partial_charge_benchmark(test, manifest, _stats())
    assert len(benchmark.molecules) == 4
    assert [row.mol_id for row in benchmark.evaluated] == [
        "unique", "ambiguous", "fallback"
    ]
    assert benchmark.molecules[-1].exclusion_reason == "invalid_mol2"
    assert benchmark.evaluated[0].mapping.mapping_status == "unique"
    assert benchmark.evaluated[1].mapping.mapping_status == "ambiguous"
    assert benchmark.evaluated[2].mapping.bond_match_mode == "connectivity_only"
    assert benchmark.comparison_identity["payload"]["sources"][
        "simulation/partial_atomic_charge:mapping_contract"
    ] == PARTIAL_CHARGE_MAPPING_CONTRACT["hash"]
    changed = build_partial_charge_benchmark(test, manifest, _stats(3.0))
    assert changed.comparison_identity["hash"] != benchmark.comparison_identity["hash"]


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


def test_partial_charge_prediction_csv_is_json_and_public_safe(tmp_path: Path) -> None:
    test, manifest = _benchmark_files(tmp_path)
    benchmark = build_partial_charge_benchmark(test, manifest, _stats())
    predictions = {
        molecule.mol_id: molecule.target_charges
        for molecule in benchmark.evaluated
    }
    score = score_partial_charge_predictions(benchmark, predictions)
    path = tmp_path / "predictions.csv"
    output = write_partial_charge_predictions(path, benchmark, score)
    assert output["rows"] == 4
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == PARTIAL_CHARGE_PREDICTION_FIELDS
    for row in rows:
        for field in (
            "atom_indices", "elements", "target_charges",
            "predicted_charges", "absolute_errors", "unparsed_bond_types",
        ):
            assert isinstance(json.loads(row[field]), list)
        assert str(tmp_path) not in row["exclusion_reason"]
    excluded = rows[-1]
    assert excluded["evaluation_status"] == "excluded_mapping"
    assert json.loads(excluded["target_charges"]) == []
    public = public_partial_charge_score(score)
    assert "_predictions" not in public
    assert "charge_conservation" not in json.dumps(public)
