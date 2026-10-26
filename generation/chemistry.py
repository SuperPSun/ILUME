"""Reviewed resources and a canonical, reversible fragment-growth DAG."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import csv
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import BRICS

from common.io import sha256_file
from common.training import canonical_json_sha256


@lru_cache(maxsize=100000)
def canonical(smiles: str, *, ports: bool = False) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f"Expected one connected molecular species: {smiles}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.GetNumRadicalElectrons() or (atom.GetAtomicNum() == 0 and not ports):
            raise ValueError("Radicals and unapproved attachment ports are unsupported")
        if atom.GetAtomicNum() == 0 and (atom.GetIsotope() <= 0 or atom.GetDegree() != 1):
            raise ValueError("Attachment ports require a positive isotope type and one bond")
        if atom.GetAtomicNum() == 0 and (atom.GetFormalCharge() != 0 or atom.GetBonds()[0].GetBondType() != Chem.BondType.SINGLE):
            raise ValueError("Attachment ports must be neutral and single-bonded")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=100000)
def heavy_atoms(smiles: str) -> int:
    return sum(atom.GetAtomicNum() > 1 for atom in Chem.MolFromSmiles(smiles).GetAtoms())


@lru_cache(maxsize=100000)
def charge(smiles: str) -> int:
    return Chem.GetFormalCharge(Chem.MolFromSmiles(smiles))


class ChemicalLibrary:
    def __init__(self, payload: dict, *, require_approved: bool = True):
        self.payload = payload
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported generation resource format")
        if require_approved and (payload.get("status") != "approved" or not all(
                payload.get("review", {}).get(key) for key in ("chemistry", "synthesis"))):
            raise ValueError("Generation requires chemically and synthetically reviewed resources")
        self.anions, self.cores = {}, {}
        for name, destination, sign in (("anions", self.anions, -1), ("cores", self.cores, 1)):
            for entry in payload[name]:
                smiles = canonical(entry["smiles"], ports=name == "cores")
                if entry["id"] in destination or charge(smiles) * sign <= 0:
                    raise ValueError("Duplicate resource ID or incorrect ion charge")
                if not entry.get("sources") or not entry.get("family"):
                    raise ValueError("Resources require provenance and family")
                destination[entry["id"]] = {**entry, "smiles": smiles}
            if not destination:
                raise ValueError(f"Empty generation resource group: {name}")
        if len({entry["smiles"] for entry in self.anions.values()}) != len(self.anions):
            raise ValueError("Anion IDs must refer to distinct canonical structures")
        self.fragments = []
        ids = set()
        for entry in payload["fragments"]:
            mol = Chem.MolFromSmiles(entry["smiles"])
            if mol is None:
                raise ValueError("Invalid fragment")
            entrances = [a for a in mol.GetAtoms() if a.GetAtomMapNum() == 1]
            if len(entrances) != 1 or entrances[0].GetAtomicNum() != 0:
                raise ValueError("Each fragment requires exactly one mapped entrance port (:1)")
            if entry["id"] in ids or not entry.get("sources"):
                raise ValueError("Fragments require unique IDs and provenance")
            ids.add(entry["id"])
            canonical(entry["smiles"], ports=True)
            if heavy_atoms(entry["smiles"]) < 1 or charge(entry["smiles"]) != 0:
                raise ValueError("Growth fragments must add heavy atoms and preserve net charge")
            port = entrances[0]
            if port.GetBonds()[0].GetBondType() != Chem.BondType.SINGLE:
                raise ValueError("Fragment entrance bonds must be single")
            entrance, neighbor, port_type = port.GetIdx(), port.GetNeighbors()[0].GetIdx(), port.GetIsotope()
            body = Chem.RWMol(mol)
            body.RemoveAtom(entrance)
            neighbor -= neighbor > entrance
            for atom in body.GetAtoms():
                atom.SetAtomMapNum(0)
            self.fragments.append({"id": entry["id"], "mol": mol, "entrance": entrance,
                "neighbor": neighbor, "type": port_type, "body": body.GetMol()})
        self.completion_cost = {}
        for _ in range(len(self.fragments) + 1):
            changed = False
            for fragment in self.fragments:
                exits = [a.GetIsotope() for a in fragment["body"].GetAtoms() if a.GetAtomicNum() == 0]
                if all(port in self.completion_cost for port in exits):
                    cost = heavy_atoms(Chem.MolToSmiles(fragment["mol"])) + sum(self.completion_cost[p] for p in exits)
                    if cost < self.completion_cost.get(fragment["type"], float("inf")):
                        self.completion_cost[fragment["type"]] = cost
                        changed = True
            if not changed:
                break
        for entry in self.cores.values():
            if any(a.GetIsotope() not in self.completion_cost for a in Chem.MolFromSmiles(entry["smiles"]).GetAtoms()
                    if a.GetAtomicNum() == 0):
                raise ValueError("Approved core has an attachment type without a completion rule")
        self.identity = canonical_json_sha256(payload)

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text()))


def extract_resources(cation_csv: str | Path, anion_csv: str | Path) -> dict:
    """Extract unapproved candidates; human review must create a separate resource version."""
    result = {"format_version": 1, "status": "draft", "review": {"chemistry": "", "synthesis": ""},
        "anions": [], "cores": [], "fragments": [], "rejected": [],
        "pretraining_scope": "source_corpus_only"}
    cores, fragments = {}, {}
    for kind, path in (("cation", Path(cation_csv)), ("anion", Path(anion_csv))):
        source = {"file": path.name, "sha256": sha256_file(path)}
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for number, row in enumerate(csv.DictReader(handle), 2):
                try:
                    smiles = canonical(row["SMILES"])
                except ValueError as error:
                    result["rejected"].append({"file": path.name, "row": number, "reason": str(error)})
                    continue
                provenance = {**source, "row": number}
                if charge(smiles) * (1 if kind == "cation" else -1) <= 0:
                    result["rejected"].append({"file": path.name, "row": number, "reason": "source_ion_role_mismatch"})
                    continue
                if kind == "anion":
                    result["anions"].append({"smiles": smiles, "family": "unreviewed",
                        "sources": [provenance], "pretraining_seen": True})
                    continue
                cores[smiles] = provenance
                broken = BRICS.BreakBRICSBonds(Chem.MolFromSmiles(smiles))
                for piece in Chem.GetMolFrags(broken, asMols=True):
                    if Chem.GetFormalCharge(piece) > 0:
                        cores[canonical(Chem.MolToSmiles(piece), ports=True)] = provenance
                    elif Chem.GetFormalCharge(piece) == 0:
                        for atom in piece.GetAtoms():
                            if atom.GetAtomicNum() == 0:
                                copy = Chem.Mol(piece)
                                copy.GetAtomWithIdx(atom.GetIdx()).SetAtomMapNum(1)
                                fragments[Chem.MolToSmiles(copy)] = provenance
    for smiles, source in sorted(cores.items()):
        result["cores"].append({"smiles": smiles, "family": "unreviewed", "sources": [source],
            "pretraining_seen": not any(a.GetAtomicNum() == 0 for a in Chem.MolFromSmiles(smiles).GetAtoms())})
    result["fragments"] = [{"smiles": s, "sources": [source]} for s, source in sorted(fragments.items())]
    for name in ("anions", "cores", "fragments"):
        unique = {entry["smiles"]: entry for entry in result[name]}
        result[name] = [{"id": f"{name}_{i:05d}", **entry} for i, entry in enumerate(unique.values())]
    return result


@dataclass(frozen=True, order=True)
class State:
    anion: str = ""
    cation: str = ""
    terminal: bool = False


@dataclass(frozen=True)
class Constraints:
    max_atoms: int = 60
    min_atoms: int = 1
    core_families: tuple[str, ...] = ()
    required_smarts: tuple[str, ...] = ()
    forbidden_smarts: tuple[str, ...] = ()

    def __post_init__(self):
        if self.min_atoms < 1 or self.max_atoms < self.min_atoms:
            raise ValueError("Invalid cation atom limits")
        for pattern in self.required_smarts + self.forbidden_smarts:
            if Chem.MolFromSmarts(pattern) is None:
                raise ValueError(f"Invalid structure constraint: {pattern}")


class FragmentEnvironment:
    def __init__(self, library: ChemicalLibrary, max_atoms: int = 60):
        self.library, self.max_atoms = library, max_atoms

    def _allowed(self, smiles: str, constraints: Constraints) -> bool:
        mol = Chem.MolFromSmiles(smiles)
        completion = sum(self.library.completion_cost.get(a.GetIsotope(), float("inf"))
            for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
        return heavy_atoms(smiles) + completion <= constraints.max_atoms and not any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(p)) for p in constraints.forbidden_smarts)

    def _cores(self, constraints: Constraints):
        return [entry for entry in self.library.cores.values() if
            (not constraints.core_families or entry["family"] in constraints.core_families)
            and self._allowed(entry["smiles"], constraints)]

    def _join(self, smiles: str, port_index: int, fragment: dict) -> str | None:
        mol, piece = Chem.MolFromSmiles(smiles), fragment["mol"]
        port = mol.GetAtomWithIdx(port_index)
        if port.GetIsotope() != fragment["type"] or port.GetBonds()[0].GetBondType() != Chem.BondType.SINGLE:
            return None
        offset = mol.GetNumAtoms()
        joined = Chem.RWMol(Chem.CombineMols(mol, piece))
        neighbor = piece.GetAtomWithIdx(fragment["entrance"]).GetNeighbors()[0].GetIdx()
        joined.AddBond(port.GetNeighbors()[0].GetIdx(), offset + neighbor, Chem.BondType.SINGLE)
        for index in sorted((port_index, offset + fragment["entrance"]), reverse=True):
            joined.RemoveAtom(index)
        try:
            Chem.SanitizeMol(joined)
            return canonical(Chem.MolToSmiles(joined), ports=True)
        except ValueError:
            return None

    @lru_cache(maxsize=100000)
    def children(self, state: State, constraints: Constraints, fixed_anion: str = "") -> tuple[State, ...]:
        if constraints.max_atoms > self.max_atoms:
            raise ValueError("Request exceeds the trained environment atom limit")
        if fixed_anion and fixed_anion not in self.library.anions:
            raise ValueError("Specified anion is outside the approved library")
        if state.terminal:
            return ()
        if not state.anion:
            return tuple(State(anion) for anion in sorted(self.library.anions)
                if not fixed_anion or anion == fixed_anion)
        if state.anion not in self.library.anions or (fixed_anion and state.anion != fixed_anion):
            return ()
        if not state.cation:
            return tuple(sorted({State(state.anion, e["smiles"]) for e in self._cores(constraints)}))
        mol = Chem.MolFromSmiles(state.cation)
        ports = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        if not ports:
            if heavy_atoms(state.cation) >= constraints.min_atoms and all(
                    mol.HasSubstructMatch(Chem.MolFromSmarts(p)) for p in constraints.required_smarts):
                return (State(state.anion, state.cation, True),)
            return ()
        result = set()
        for port in ports:
            for fragment in self.library.fragments:
                joined = self._join(state.cation, port, fragment)
                if joined and self._allowed(joined, constraints):
                    result.add(State(state.anion, joined))
        return tuple(sorted(result))

    @lru_cache(maxsize=100000)
    def _cation_parents(self, smiles: str, constraints: Constraints) -> tuple[str, ...]:
        mol = Chem.MolFromSmiles(smiles)
        result = set()
        for fragment in self.library.fragments:
            body = fragment["body"]
            # Preserve attachment orientation; symmetric matches can expose distinct parents.
            matches = mol.GetSubstructMatches(body, uniquify=False, maxMatches=1000000, useChirality=True)
            if len(matches) == 1000000:
                raise ValueError("Reverse chemistry match limit reached; reduce the resource space")
            for match in matches:
                selected = set(match)
                entry = match[fragment["neighbor"]]
                external = [(a.GetIdx(), n.GetIdx(), mol.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx()))
                    for a in (mol.GetAtomWithIdx(i) for i in match)
                    for n in a.GetNeighbors() if n.GetIdx() not in selected]
                if len(external) != 1 or external[0][0] != entry or external[0][2].GetBondType() != Chem.BondType.SINGLE:
                    continue
                outside = external[0][1]
                parent = Chem.RWMol(mol)
                dummy = Chem.Atom(0)
                dummy.SetIsotope(fragment["type"])
                index = parent.AddAtom(dummy)
                parent.AddBond(outside, index, Chem.BondType.SINGLE)
                for index in sorted(selected, reverse=True):
                    parent.RemoveAtom(index)
                try:
                    Chem.SanitizeMol(parent)
                    candidate = canonical(Chem.MolToSmiles(parent), ports=True)
                except ValueError:
                    continue
                if self._allowed(candidate, constraints) and self._reachable(candidate, constraints):
                    # Verify atom/query matching against the actual forward chemistry.
                    if State("a", smiles) in self._growth_children(candidate, constraints):
                        result.add(candidate)
        return tuple(sorted(result))

    def _growth_children(self, smiles: str, constraints: Constraints) -> set[State]:
        result = set()
        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                for fragment in self.library.fragments:
                    joined = self._join(smiles, atom.GetIdx(), fragment)
                    if joined and self._allowed(joined, constraints):
                        result.add(State("a", joined))
        return result

    @lru_cache(maxsize=100000)
    def _reachable(self, smiles: str, constraints: Constraints) -> bool:
        return any(e["smiles"] == smiles for e in self._cores(constraints)) or bool(self._cation_parents(smiles, constraints))

    def parents(self, state: State, constraints: Constraints, fixed_anion: str = "") -> tuple[State, ...]:
        if not state.anion:
            return ()
        if not state.cation:
            return (State(),) if state in self.children(State(), constraints, fixed_anion) else ()
        if state.terminal:
            parent = State(state.anion, state.cation)
            return (parent,) if state in self.children(parent, constraints, fixed_anion) else ()
        result = {State(state.anion, p) for p in self._cation_parents(state.cation, constraints)}
        if any(e["smiles"] == state.cation for e in self._cores(constraints)):
            result.add(State(state.anion))
        return tuple(sorted(result))

    def describe(self, state: State) -> dict:
        anion = self.library.anions[state.anion]
        return {"cation": state.cation, "anion": anion["smiles"], "anion_id": state.anion,
            "anion_family": anion["family"], "cation_charge": charge(state.cation),
            "anion_charge": charge(anion["smiles"])}

    def trajectory_record(self, states: list[State]) -> list[dict]:
        return [asdict(s) for s in states]
