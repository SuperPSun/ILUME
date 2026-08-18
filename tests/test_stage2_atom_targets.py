from __future__ import annotations

from pathlib import Path

import pytest

from common.io import sha256_file
from stage2.atom_targets import (
    StructureManifestEntry, map_partial_charges, parse_mol2,
    verify_structure,
)


def _write_mol2(path: Path, atoms, bonds, *, declared_atoms: int | None = None) -> None:
    lines = [
        "@<TRIPOS>MOLECULE", "MOL",
        f"{len(atoms) if declared_atoms is None else declared_atoms} {len(bonds)} 1 0 0",
        "SMALL", "resp", "@<TRIPOS>ATOM",
    ]
    for index, (name, atom_type, charge) in enumerate(atoms, start=1):
        lines.append(f"{index} {name} 0 0 0 {atom_type} 1 MOL {charge}")
    lines.append("@<TRIPOS>BOND")
    for index, (first, second, kind) in enumerate(bonds, start=1):
        lines.append(f"{index} {first} {second} {kind}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_typed_mapping_explicit_h_and_deterministic_automorphism(tmp_path: Path) -> None:
    path = tmp_path / "typed.mol2"
    _write_mol2(
        path,
        [("C1", "c3", 0.2), ("C2", "c3", -0.2), ("H1", "h1", 0.0)],
        [(1, 2, "1"), (1, 3, "du")],
    )
    result = map_partial_charges("CC", parse_mol2(path))
    assert result.bond_match_mode == "typed"
    assert result.unparsed_bond_types == ()
    assert result.mapping_status == "ambiguous"
    assert result.mapping_count_lower_bound == 2
    assert result.charges == pytest.approx((0.2, -0.2))


def test_unknown_bond_is_auditable_connectivity_fallback(tmp_path: Path) -> None:
    path = tmp_path / "fallback.mol2"
    _write_mol2(path, [("C1", "c3", 0.1), ("O1", "os", -0.1)], [(1, 2, "du")])
    result = map_partial_charges("CO", parse_mol2(path))
    assert result.bond_match_mode == "connectivity_only"
    assert result.unparsed_bond_types == ("du",)
    assert result.bond_fallback_reason == "unparsed_bond_type"


def test_mol2_counts_and_manifest_hash_are_strict(tmp_path: Path) -> None:
    path = tmp_path / "bad-count.mol2"
    _write_mol2(path, [("C1", "c3", 0.0)], [], declared_atoms=2)
    with pytest.raises(ValueError, match="declared atom/bond counts"):
        parse_mol2(path)
    good = tmp_path / "good.mol2"
    _write_mol2(good, [("C1", "c3", 0.0)], [])
    verify_structure(StructureManifestEntry("ok", good, good.stat().st_size, sha256_file(good)))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_structure(StructureManifestEntry("bad", good, good.stat().st_size, "0" * 64))
