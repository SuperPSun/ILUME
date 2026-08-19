from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterator

from rdkit import Chem

from common.io import sha256_file


_BOND_TYPES = {
    "1": "single",
    "2": "double",
    "3": "triple",
    "ar": "aromatic",
    "am": "single",
}


@dataclass(frozen=True)
class Mol2Atom:
    atom_id: int
    element: str
    partial_charge: float


@dataclass(frozen=True)
class Mol2Bond:
    first: int
    second: int
    raw_type: str
    normalized_type: str | None


@dataclass(frozen=True)
class Mol2Graph:
    atoms: tuple[Mol2Atom, ...]
    bonds: tuple[Mol2Bond, ...]


@dataclass(frozen=True)
class AtomMappingResult:
    charges: tuple[float, ...]
    structure_atom_count: int
    mapping_status: str
    mapping_count_lower_bound: int
    selected_mapping_rank: int
    bond_match_mode: str
    unparsed_bond_types: tuple[str, ...]
    bond_fallback_reason: str


@dataclass(frozen=True)
class StructureManifestEntry:
    mol_id: str
    path: Path
    size_bytes: int
    sha256: str


def _element(atom_name: str, atom_type: str) -> str:
    lowered = atom_type.lower()
    special = {"cl": "Cl", "br": "Br", "si": "Si"}
    for prefix, symbol in special.items():
        if lowered.startswith(prefix):
            return symbol
    first = lowered[:1]
    simple = {"c": "C", "n": "N", "o": "O", "s": "S", "p": "P", "f": "F", "i": "I", "h": "H", "b": "B"}
    if first in simple:
        return simple[first]
    letters = "".join(character for character in atom_name if character.isalpha())
    for width in (2, 1):
        candidate = letters[:width].title()
        if candidate and Chem.GetPeriodicTable().GetAtomicNumber(candidate) > 0:
            return candidate
    raise ValueError(f"Unsupported MOL2 element: {atom_name}/{atom_type}")


def parse_mol2_text(text: str, *, source: str = "<memory>") -> Mol2Graph:
    atoms: list[Mol2Atom] = []
    bonds: list[Mol2Bond] = []
    section = ""
    seen_sections: set[str] = set()
    molecule_rows: list[tuple[int, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("@<TRIPOS>"):
            section = line.removeprefix("@<TRIPOS>").strip().upper()
            seen_sections.add(section)
            continue
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if section == "MOLECULE":
            molecule_rows.append((line_number, stripped))
        elif section == "ATOM":
            if len(fields) < 9:
                raise ValueError(f"Invalid MOL2 atom row at {source}:{line_number}")
            try:
                atom_id = int(fields[0])
                charge = float(fields[8])
            except ValueError as error:
                raise ValueError(f"Invalid MOL2 atom value at {source}:{line_number}") from error
            if not math.isfinite(charge):
                raise ValueError(f"Non-finite MOL2 partial charge at {source}:{line_number}")
            atoms.append(Mol2Atom(atom_id, _element(fields[1], fields[5]), charge))
        elif section == "BOND":
            if len(fields) < 4:
                raise ValueError(f"Invalid MOL2 bond row at {source}:{line_number}")
            try:
                first, second = int(fields[1]), int(fields[2])
            except ValueError as error:
                raise ValueError(f"Invalid MOL2 bond index at {source}:{line_number}") from error
            raw_type = fields[3].lower()
            bonds.append(Mol2Bond(first, second, raw_type, _BOND_TYPES.get(raw_type)))
    if not {"MOLECULE", "ATOM", "BOND"}.issubset(seen_sections):
        raise ValueError(f"MOL2 must contain MOLECULE, ATOM, and BOND sections: {source}")
    if len(molecule_rows) < 2:
        raise ValueError(f"MOL2 MOLECULE section is missing atom/bond counts: {source}")
    try:
        declared_atoms, declared_bonds = map(int, molecule_rows[1][1].split()[:2])
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid MOL2 atom/bond counts: {source}:{molecule_rows[1][0]}") from error
    if declared_atoms != len(atoms) or declared_bonds != len(bonds):
        raise ValueError(f"MOL2 declared atom/bond counts do not match parsed rows: {source}")
    ids = [atom.atom_id for atom in atoms]
    if not atoms or len(ids) != len(set(ids)):
        raise ValueError(f"MOL2 atom IDs must be non-empty and unique: {source}")
    known = set(ids)
    if any(bond.first not in known or bond.second not in known or bond.first == bond.second for bond in bonds):
        raise ValueError(f"MOL2 contains an invalid bond reference: {source}")
    return Mol2Graph(tuple(atoms), tuple(bonds))


def parse_mol2(path: str | Path) -> Mol2Graph:
    source = Path(path)
    return parse_mol2_text(source.read_text(encoding="utf-8"), source=str(source))


def load_structure_manifest(path: str | Path) -> dict[str, StructureManifestEntry]:
    manifest_path = Path(path)
    entries: dict[str, StructureManifestEntry] = {}
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        expected = ("mol_id", "relative_path", "format", "size_bytes", "sha256", "referenced_by_charge")
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError("Unexpected partial-charge structure manifest columns")
        for row in reader:
            mol_id = row["mol_id"].strip()
            relative = Path(row["relative_path"])
            if not mol_id or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe partial-charge manifest entry")
            structure_path = (manifest_path.parent / relative).resolve()
            if not structure_path.is_relative_to(manifest_path.parent.resolve()):
                raise ValueError("Partial-charge structure path escapes manifest root")
            if row["format"].strip().lower() != "mol2":
                raise ValueError("Object v3 partial-charge resources must be MOL2")
            if mol_id in entries:
                raise ValueError(f"Duplicate mol_id in structure manifest: {mol_id}")
            entries[mol_id] = StructureManifestEntry(
                mol_id=mol_id,
                path=structure_path,
                size_bytes=int(row["size_bytes"]),
                sha256=row["sha256"].strip(),
            )
    return entries


def verify_structure(entry: StructureManifestEntry) -> None:
    if not entry.path.is_file():
        raise FileNotFoundError(entry.path)
    if entry.path.stat().st_size != entry.size_bytes:
        raise ValueError(f"MOL2 size mismatch: {entry.mol_id}")
    if sha256_file(entry.path) != entry.sha256:
        raise ValueError(f"MOL2 hash mismatch: {entry.mol_id}")


def _rdkit_bond_type(bond: Chem.Bond) -> str | None:
    if bond.GetIsAromatic():
        return "aromatic"
    value = bond.GetBondTypeAsDouble()
    return {1.0: "single", 2.0: "double", 3.0: "triple"}.get(value)


def _mapping_candidates(
    model: Chem.Mol,
    structure: Mol2Graph,
    *,
    typed: bool,
) -> Iterator[tuple[int, ...]]:
    model_count = model.GetNumAtoms()
    structure_ids = [atom.atom_id for atom in structure.atoms]
    structure_index = {atom_id: index for index, atom_id in enumerate(structure_ids)}
    structure_elements = [atom.element for atom in structure.atoms]
    structure_adjacency: list[dict[int, str | None]] = [dict() for _ in structure.atoms]
    for bond in structure.bonds:
        first = structure_index[bond.first]
        second = structure_index[bond.second]
        structure_adjacency[first][second] = bond.normalized_type
        structure_adjacency[second][first] = bond.normalized_type
    model_adjacency: list[dict[int, str | None]] = [dict() for _ in range(model_count)]
    for bond in model.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        normalized = _rdkit_bond_type(bond)
        model_adjacency[first][second] = normalized
        model_adjacency[second][first] = normalized
    model_elements = [atom.GetSymbol() for atom in model.GetAtoms()]
    if "H" not in model_elements:
        usable = [index for index, element in enumerate(structure_elements) if element != "H"]
    else:
        usable = list(range(len(structure.atoms)))
    if len(usable) != model_count:
        return
    candidate_sets = {
        model_index: [
            structure_index_value
            for structure_index_value in usable
            if structure_elements[structure_index_value] == model_elements[model_index]
            and sum(neighbor in usable for neighbor in structure_adjacency[structure_index_value]) == len(model_adjacency[model_index])
        ]
        for model_index in range(model_count)
    }
    order = tuple(range(model_count))
    assignment = [-1] * model_count
    used: set[int] = set()

    def search(position: int) -> Iterator[tuple[int, ...]]:
        if position == len(order):
            yield tuple(assignment)
            return
        model_index = order[position]
        for candidate in candidate_sets[model_index]:
            if candidate in used:
                continue
            valid = True
            for other_model, other_structure in enumerate(assignment):
                if other_structure < 0:
                    continue
                model_has_bond = other_model in model_adjacency[model_index]
                structure_has_bond = other_structure in structure_adjacency[candidate]
                if model_has_bond != structure_has_bond:
                    valid = False
                    break
                if typed and model_has_bond and (
                    model_adjacency[model_index][other_model]
                    != structure_adjacency[candidate][other_structure]
                ):
                    valid = False
                    break
            if not valid:
                continue
            assignment[model_index] = candidate
            used.add(candidate)
            yield from search(position + 1)
            used.remove(candidate)
            assignment[model_index] = -1

    yield from search(0)


def map_partial_charges(canonical_smiles: str, structure: Mol2Graph) -> AtomMappingResult:
    model = Chem.MolFromSmiles(canonical_smiles)
    if model is None:
        raise ValueError("Invalid canonical SMILES for partial-charge mapping")
    model_has_hydrogen = any(atom.GetSymbol() == "H" for atom in model.GetAtoms())
    relevant_ids = {
        atom.atom_id for atom in structure.atoms
        if model_has_hydrogen or atom.element != "H"
    }
    unparsed = tuple(sorted({
        bond.raw_type for bond in structure.bonds
        if bond.first in relevant_ids and bond.second in relevant_ids
        and bond.normalized_type is None
    }))
    model_unparsed = any(_rdkit_bond_type(bond) is None for bond in model.GetBonds())
    typed = not unparsed and not model_unparsed
    mappings = list(islice(_mapping_candidates(model, structure, typed=typed), 2))
    fallback_reason = ""
    if typed and not mappings:
        typed = False
        fallback_reason = "typed_isomorphism_failed"
        mappings = list(islice(_mapping_candidates(model, structure, typed=False), 2))
    if not mappings:
        raise ValueError("No graph isomorphism between Stage 1 and MOL2 atoms")
    selected = mappings[0]
    charges = tuple(structure.atoms[index].partial_charge for index in selected)
    return AtomMappingResult(
        charges=charges,
        structure_atom_count=len(structure.atoms),
        mapping_status="ambiguous" if len(mappings) > 1 else "unique",
        mapping_count_lower_bound=min(len(mappings), 2),
        selected_mapping_rank=1,
        bond_match_mode="typed" if typed else "connectivity_only",
        unparsed_bond_types=unparsed,
        bond_fallback_reason=(
            fallback_reason
            or ("unparsed_bond_type" if unparsed else "unsupported_model_bond")
            if not typed
            else ""
        ),
    )


def load_verify_parse_and_map(
    entry: StructureManifestEntry, canonical_smiles: str,
) -> AtomMappingResult:
    payload = entry.path.read_bytes()
    if len(payload) != entry.size_bytes:
        raise ValueError(f"MOL2 size mismatch: {entry.mol_id}")
    if hashlib.sha256(payload).hexdigest() != entry.sha256:
        raise ValueError(f"MOL2 hash mismatch: {entry.mol_id}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"MOL2 is not valid UTF-8: {entry.mol_id}") from error
    return map_partial_charges(
        canonical_smiles, parse_mol2_text(text, source=str(entry.path))
    )


__all__ = [
    "AtomMappingResult", "Mol2Graph", "StructureManifestEntry",
    "load_structure_manifest", "load_verify_parse_and_map", "map_partial_charges",
    "parse_mol2", "parse_mol2_text", "verify_structure",
]
