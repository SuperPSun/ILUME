from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from rdkit import Chem

from common.identity import semantic_identity
from common.io import sha256_file
from common.reporting import comparison_identity, write_prediction_csv
from .atom_targets import (
    PARTIAL_CHARGE_MAPPING_CONTRACT,
    AtomMappingResult,
    load_structure_manifest,
    load_verify_parse_and_map,
)


PARTIAL_CHARGE_TASK = "simulation/partial_atomic_charge"
PARTIAL_CHARGE_UNIT = (
    f"{PARTIAL_CHARGE_TASK}::molecule_macro_normalized_mae"
)
PARTIAL_CHARGE_SUBSETS = (
    "all_mapped",
    "unique",
    "ambiguous",
    "typed",
    "connectivity_only",
)
PARTIAL_CHARGE_PREDICTION_FIELDS = (
    "source_row",
    "mol_id",
    "canonical_smiles",
    "role",
    "formal_charge",
    "evaluation_status",
    "exclusion_reason",
    "atom_count",
    "atom_indices",
    "elements",
    "target_charges",
    "predicted_charges",
    "absolute_errors",
    "molecule_mae",
    "mapping_status",
    "mapping_count_lower_bound",
    "selected_mapping_rank",
    "bond_match_mode",
    "unparsed_bond_types",
    "bond_fallback_reason",
)


@dataclass(frozen=True)
class PartialChargeMolecule:
    source_row: int
    mol_id: str
    canonical_smiles: str
    role: str
    formal_charge: int
    elements: tuple[str, ...]
    target_charges: tuple[float, ...]
    mapping: AtomMappingResult | None
    exclusion_reason: str = ""

    @property
    def mapped(self) -> bool:
        return self.mapping is not None


@dataclass(frozen=True)
class PartialChargeBenchmark:
    molecules: tuple[PartialChargeMolecule, ...]
    target_mean: float
    target_scale: float
    mapping_audit_hash: str
    evaluated_subset_hash: str
    comparison_identity: dict[str, Any]

    @property
    def evaluated(self) -> tuple[PartialChargeMolecule, ...]:
        return tuple(molecule for molecule in self.molecules if molecule.mapped)


def _canonical(raw: str, context: str) -> tuple[str, Chem.Mol]:
    molecule = Chem.MolFromSmiles((raw or "").strip())
    if molecule is None:
        raise ValueError(f"Invalid partial-charge SMILES: {context}")
    canonical = Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    )
    normalized = Chem.MolFromSmiles(canonical)
    if normalized is None:
        raise AssertionError("Canonical partial-charge SMILES failed RDKit parsing")
    return canonical, normalized


def _role_and_charge(molecule: Chem.Mol) -> tuple[str, int]:
    charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    return ("cation" if charge > 0 else "anion" if charge < 0 else "neutral"), charge


def _mapping_failure(error: BaseException) -> str:
    message = str(error)
    if isinstance(error, FileNotFoundError):
        return "missing_structure"
    if "size mismatch" in message:
        return "structure_size_mismatch"
    if "hash mismatch" in message:
        return "structure_hash_mismatch"
    if "No graph isomorphism" in message:
        return "no_graph_isomorphism"
    if "MOL2" in message or "UTF-8" in message:
        return "invalid_mol2"
    return "mapping_error"


def _audit_payload(molecule: PartialChargeMolecule) -> dict[str, Any]:
    mapping = molecule.mapping
    return {
        "source_row": molecule.source_row,
        "mol_id": molecule.mol_id,
        "canonical_smiles": molecule.canonical_smiles,
        "role": molecule.role,
        "formal_charge": molecule.formal_charge,
        "status": "mapped" if mapping is not None else "excluded",
        "exclusion_reason": molecule.exclusion_reason,
        "mapping_status": "" if mapping is None else mapping.mapping_status,
        "mapping_count_lower_bound": (
            0 if mapping is None else mapping.mapping_count_lower_bound
        ),
        "selected_mapping_rank": (
            0 if mapping is None else mapping.selected_mapping_rank
        ),
        "bond_match_mode": "" if mapping is None else mapping.bond_match_mode,
        "unparsed_bond_types": (
            [] if mapping is None else list(mapping.unparsed_bond_types)
        ),
        "bond_fallback_reason": (
            "" if mapping is None else mapping.bond_fallback_reason
        ),
    }


def _evaluated_payload(molecule: PartialChargeMolecule) -> dict[str, Any]:
    if molecule.mapping is None:
        raise ValueError("Excluded partial-charge molecule is not evaluated")
    return {
        **_audit_payload(molecule),
        "elements": list(molecule.elements),
        "target_charges": list(molecule.target_charges),
    }


def build_partial_charge_benchmark(
    test_path: str | Path,
    manifest_path: str | Path,
    target_stats: Mapping[str, Any],
) -> PartialChargeBenchmark:
    test_source = Path(test_path)
    manifest_source = Path(manifest_path)
    if target_stats.get("weighting") != "molecule_equal":
        raise ValueError("Partial-charge scaler must use molecule_equal weighting")
    try:
        target_mean = float(target_stats["mean"])
        target_scale = float(target_stats["scale"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Malformed partial-charge target scaler") from error
    if not math.isfinite(target_mean) or not math.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("Partial-charge target scaler must be finite and positive")

    manifest = load_structure_manifest(manifest_source)
    molecules: list[PartialChargeMolecule] = []
    expected_fields = (
        "mol_id", "SMILES", "role", "formal_charge", "source_list"
    )
    with test_source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError(
                f"Unexpected partial-charge test columns: {reader.fieldnames}"
            )
        for source_row, raw in enumerate(reader, start=2):
            mol_id = raw["mol_id"].strip()
            if not mol_id:
                raise ValueError(f"Empty partial-charge mol_id at source_row={source_row}")
            canonical, molecule = _canonical(raw["SMILES"], f"source_row={source_row}")
            inferred_role, inferred_charge = _role_and_charge(molecule)
            try:
                declared_charge = int(raw["formal_charge"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid partial-charge formal_charge at source_row={source_row}"
                ) from error
            role = raw["role"].strip()
            if role != inferred_role or declared_charge != inferred_charge:
                raise ValueError(
                    "Partial-charge role/formal_charge mismatch at "
                    f"source_row={source_row}"
                )
            entry = manifest.get(mol_id)
            if entry is None:
                molecules.append(
                    PartialChargeMolecule(
                        source_row, mol_id, canonical, role, declared_charge,
                        (), (), None, "missing_manifest_entry",
                    )
                )
                continue
            try:
                mapping = load_verify_parse_and_map(entry, canonical)
            except (OSError, RuntimeError, ValueError) as error:
                molecules.append(
                    PartialChargeMolecule(
                        source_row, mol_id, canonical, role, declared_charge,
                        (), (), None, _mapping_failure(error),
                    )
                )
                continue
            elements = tuple(atom.GetSymbol() for atom in molecule.GetAtoms())
            if len(elements) != len(mapping.charges):
                raise ValueError(
                    f"Partial-charge mapped atom count mismatch: {mol_id}"
                )
            molecules.append(
                PartialChargeMolecule(
                    source_row, mol_id, canonical, role, declared_charge,
                    elements, tuple(mapping.charges), mapping,
                )
            )
    if not molecules:
        raise ValueError("Partial-charge test split is empty")
    mol_ids = [molecule.mol_id for molecule in molecules]
    if len(mol_ids) != len(set(mol_ids)):
        raise ValueError("Partial-charge test mol_id values must be unique")
    evaluated = [molecule for molecule in molecules if molecule.mapped]
    if not evaluated:
        raise ValueError("Partial-charge test split has no mapped molecules")

    audit_hash = semantic_identity(
        "stage2.partial-charge-mapping-audit.v1",
        {"rows": [_audit_payload(molecule) for molecule in molecules]},
    )["hash"]
    evaluated_hash = semantic_identity(
        "stage2.partial-charge-evaluated-subset.v1",
        {"molecules": [_evaluated_payload(molecule) for molecule in evaluated]},
    )["hash"]
    comparison = comparison_identity(
        "stage2_partial_charge",
        split="test",
        expected=(PARTIAL_CHARGE_UNIT,),
        sources={
            f"{PARTIAL_CHARGE_TASK}:test": sha256_file(test_source),
            f"{PARTIAL_CHARGE_TASK}:structure_manifest": sha256_file(
                manifest_source
            ),
            f"{PARTIAL_CHARGE_TASK}:mapping_contract": (
                PARTIAL_CHARGE_MAPPING_CONTRACT["hash"]
            ),
            f"{PARTIAL_CHARGE_TASK}:mapping_audit": audit_hash,
            f"{PARTIAL_CHARGE_TASK}:evaluated_subset": evaluated_hash,
        },
        normalization={
            PARTIAL_CHARGE_UNIT: {
                "scale": target_scale,
                "weighting": "molecule_equal",
            }
        },
    )
    return PartialChargeBenchmark(
        tuple(molecules), target_mean, target_scale, audit_hash,
        evaluated_hash, comparison,
    )


def _subset_metrics(
    molecules: Sequence[PartialChargeMolecule],
    predictions: Mapping[str, np.ndarray],
    scale: float,
) -> dict[str, Any]:
    if not molecules:
        return {
            "reason": "no_samples",
            "molecule_count": 0,
            "atom_count": 0,
            "molecule_macro_mae": None,
            "molecule_macro_normalized_mae": None,
            "atom_micro_mae": None,
            "atom_micro_rmse": None,
            "atom_micro_r2": None,
            "atom_micro_r2_reason": "no_samples",
        }
    molecule_mae: list[float] = []
    targets: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    for molecule in molecules:
        target = np.asarray(molecule.target_charges, dtype=np.float64)
        prediction = predictions[molecule.mol_id]
        molecule_mae.append(float(np.abs(prediction - target).mean()))
        targets.append(target)
        predicted.append(prediction)
    target_values = np.concatenate(targets)
    prediction_values = np.concatenate(predicted)
    delta = prediction_values - target_values
    denominator = float(np.square(target_values - target_values.mean()).sum())
    raw_macro = float(np.mean(molecule_mae))
    return {
        "reason": None,
        "molecule_count": len(molecules),
        "atom_count": len(target_values),
        "molecule_macro_mae": raw_macro,
        "molecule_macro_normalized_mae": raw_macro / scale,
        "atom_micro_mae": float(np.abs(delta).mean()),
        "atom_micro_rmse": float(np.sqrt(np.square(delta).mean())),
        "atom_micro_r2": (
            None
            if denominator == 0
            else 1.0 - float(np.square(delta).sum()) / denominator
        ),
        "atom_micro_r2_reason": (
            "constant_target" if denominator == 0 else None
        ),
    }


def score_partial_charge_predictions(
    benchmark: PartialChargeBenchmark,
    predictions: Mapping[str, Sequence[float] | np.ndarray],
) -> dict[str, Any]:
    expected = {molecule.mol_id: molecule for molecule in benchmark.evaluated}
    issues: list[str] = []
    materialized: dict[str, np.ndarray] = {}
    missing: list[str] = []
    invalid: list[str] = []
    extra = sorted(set(predictions) - set(expected))
    if extra:
        issues.append(f"extra_predictions={len(extra)}")
    for mol_id, molecule in expected.items():
        if mol_id not in predictions:
            missing.append(mol_id)
            continue
        try:
            values = np.asarray(predictions[mol_id], dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            invalid.append(mol_id)
            continue
        if (
            len(values) != len(molecule.target_charges)
            or not np.isfinite(values).all()
        ):
            invalid.append(mol_id)
            continue
        materialized[mol_id] = values
    if missing:
        issues.append(f"missing_predictions={len(missing)}")
    if invalid:
        issues.append(f"invalid_predictions={len(invalid)}")
    complete = not issues

    subsets: dict[str, Any] = {}
    if complete:
        selectors = {
            "all_mapped": lambda molecule: True,
            "unique": lambda molecule: molecule.mapping.mapping_status == "unique",
            "ambiguous": lambda molecule: molecule.mapping.mapping_status == "ambiguous",
            "typed": lambda molecule: molecule.mapping.bond_match_mode == "typed",
            "connectivity_only": lambda molecule: (
                molecule.mapping.bond_match_mode == "connectivity_only"
            ),
        }
        for name in PARTIAL_CHARGE_SUBSETS:
            selected = [
                molecule for molecule in benchmark.evaluated
                if selectors[name](molecule)
            ]
            subsets[name] = _subset_metrics(
                selected, materialized, benchmark.target_scale
            )
    else:
        subsets = {
            name: {
                **_subset_metrics((), {}, benchmark.target_scale),
                "reason": "incomplete_predictions",
                "atom_micro_r2_reason": "incomplete_predictions",
            }
            for name in PARTIAL_CHARGE_SUBSETS
        }
    coverage = {
        "test_molecule_count": len(benchmark.molecules),
        "mapped_molecule_count": len(expected),
        "excluded_mapping_count": len(benchmark.molecules) - len(expected),
        "prediction_molecule_count": len(materialized),
        "missing_prediction_count": len(missing),
        "invalid_prediction_count": len(invalid),
        "extra_prediction_count": len(extra),
        "mapping_audit_hash": benchmark.mapping_audit_hash,
        "evaluated_subset_hash": benchmark.evaluated_subset_hash,
        "issues": issues,
        "missing_mol_ids": missing,
        "invalid_mol_ids": invalid,
        "extra_mol_ids": extra,
    }
    all_mapped = subsets.get("all_mapped")
    return {
        "target_level": "atom",
        "capability": "supported",
        "status": "complete" if complete else "incomplete",
        "primary": (
            None
            if not complete or all_mapped is None
            else {
                "molecule_macro_mae": all_mapped["molecule_macro_mae"],
                "molecule_macro_normalized_mae": all_mapped[
                    "molecule_macro_normalized_mae"
                ],
            }
        ),
        "atom_micro": (
            None
            if not complete or all_mapped is None
            else {
                "count": all_mapped["atom_count"],
                "mae": all_mapped["atom_micro_mae"],
                "rmse": all_mapped["atom_micro_rmse"],
                "r2": all_mapped["atom_micro_r2"],
                "r2_reason": all_mapped["atom_micro_r2_reason"],
            }
        ),
        "subsets": subsets,
        "coverage": coverage,
        "_predictions": materialized,
    }


def _json_array(values: Sequence[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def write_partial_charge_predictions(
    path: str | Path,
    benchmark: PartialChargeBenchmark,
    score: Mapping[str, Any],
) -> dict[str, Any]:
    predictions = score.get("_predictions", {})
    coverage = score.get("coverage", {})
    missing = set(coverage.get("missing_mol_ids", ()))
    invalid = set(coverage.get("invalid_mol_ids", ()))
    rows: list[dict[str, Any]] = []
    for molecule in benchmark.molecules:
        mapping = molecule.mapping
        prediction = predictions.get(molecule.mol_id)
        if mapping is None:
            status = "excluded_mapping"
        elif molecule.mol_id in missing:
            status = "missing_prediction"
        elif molecule.mol_id in invalid:
            status = "invalid_prediction"
        elif prediction is None:
            raise ValueError("Partial-charge score lacks prediction coverage context")
        else:
            status = "evaluated"
        target = np.asarray(molecule.target_charges, dtype=np.float64)
        absolute = (
            np.abs(prediction - target) if prediction is not None else np.asarray([])
        )
        rows.append(
            {
                "source_row": molecule.source_row,
                "mol_id": molecule.mol_id,
                "canonical_smiles": molecule.canonical_smiles,
                "role": molecule.role,
                "formal_charge": molecule.formal_charge,
                "evaluation_status": status,
                "exclusion_reason": molecule.exclusion_reason,
                "atom_count": len(molecule.elements),
                "atom_indices": _json_array(range(len(molecule.elements))),
                "elements": _json_array(molecule.elements),
                "target_charges": _json_array(molecule.target_charges),
                "predicted_charges": _json_array(
                    [] if prediction is None else prediction.tolist()
                ),
                "absolute_errors": _json_array(absolute.tolist()),
                "molecule_mae": (
                    None if prediction is None else float(absolute.mean())
                ),
                "mapping_status": "" if mapping is None else mapping.mapping_status,
                "mapping_count_lower_bound": (
                    0 if mapping is None else mapping.mapping_count_lower_bound
                ),
                "selected_mapping_rank": (
                    0 if mapping is None else mapping.selected_mapping_rank
                ),
                "bond_match_mode": "" if mapping is None else mapping.bond_match_mode,
                "unparsed_bond_types": _json_array(
                    [] if mapping is None else mapping.unparsed_bond_types
                ),
                "bond_fallback_reason": (
                    "" if mapping is None else mapping.bond_fallback_reason
                ),
            }
        )
    manifest = write_prediction_csv(path, rows, PARTIAL_CHARGE_PREDICTION_FIELDS)
    manifest["task"] = PARTIAL_CHARGE_TASK
    return manifest


def public_partial_charge_score(score: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in score.items() if not key.startswith("_")}


__all__ = [
    "PARTIAL_CHARGE_PREDICTION_FIELDS",
    "PARTIAL_CHARGE_SUBSETS",
    "PARTIAL_CHARGE_TASK",
    "PARTIAL_CHARGE_UNIT",
    "PartialChargeBenchmark",
    "PartialChargeMolecule",
    "build_partial_charge_benchmark",
    "public_partial_charge_score",
    "score_partial_charge_predictions",
    "write_partial_charge_predictions",
]
