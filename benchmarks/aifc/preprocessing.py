from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from rdkit import Chem


ATOM_VOCAB = (
    "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl", "Br", "I",
    "In", "As", "Ga", "Fe", "Au", "Re", "Sb",
)
ATOM_FEATURE_DIM = 46
BOND_FEATURE_DIM = 12
FRAGMENT_SHA256 = "ccc5cfefe87ebf3e1d98762eb1c58170bc00c9f7cfd12b6c9ce23fa957ccdf95"
FRAGMENT_COMMIT = "23973da709212d1e13fd4b8e0968c8109f78fa75"
FRAGMENT_BLOB = "90d8bafc2eb07835854e183df48a637d97b8c09b"


def _one_hot_unknown(value: Any, allowed: Sequence[Any]) -> list[float]:
    resolved = value if value in allowed else allowed[-1]
    return [float(resolved == candidate) for candidate in allowed]


def atom_features(atom: Chem.Atom) -> list[float]:
    chirality = (
        _one_hot_unknown(str(atom.GetProp("_CIPCode")), ("R", "S"))
        if atom.HasProp("_CIPCode")
        else [0.0, 0.0]
    )
    values = (
        _one_hot_unknown(atom.GetSymbol(), ATOM_VOCAB)
        + _one_hot_unknown(atom.GetDegree(), (0, 1, 2, 3, 4, 5))
        + _one_hot_unknown(atom.GetTotalNumHs(), (0, 1, 2, 3, 4))
        + _one_hot_unknown(atom.GetImplicitValence(), (0, 1, 2, 3, 4, 5))
        + _one_hot_unknown(
            str(atom.GetHybridization()), ("SP", "SP2", "SP3", "SP3D", "SP3D2")
        )
        + [float(atom.GetIsAromatic())]
        + [float(atom.HasProp("_ChiralityPossible"))]
        + chirality
        + [float(atom.GetFormalCharge())]
    )
    if len(values) != ATOM_FEATURE_DIM:
        raise RuntimeError("AIFC atom feature width changed")
    return values


def bond_features(bond: Chem.Bond) -> list[float]:
    bond_type = bond.GetBondType()
    stereo = bond.GetStereo()
    return [
        float(bond_type == Chem.rdchem.BondType.SINGLE),
        float(bond_type == Chem.rdchem.BondType.DOUBLE),
        float(bond_type == Chem.rdchem.BondType.TRIPLE),
        float(bond_type == Chem.rdchem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        float(stereo == Chem.rdchem.BondStereo.STEREONONE),
        float(stereo == Chem.rdchem.BondStereo.STEREOANY),
        float(stereo == Chem.rdchem.BondStereo.STEREOZ),
        float(stereo == Chem.rdchem.BondStereo.STEREOE),
        float(stereo == Chem.rdchem.BondStereo.STEREOCIS),
        float(stereo == Chem.rdchem.BondStereo.STEREOTRANS),
    ]


@dataclass(frozen=True)
class FragmentScheme:
    names: tuple[str, ...]
    patterns: tuple[Chem.Mol, ...]
    priorities: tuple[int, ...]
    sha256: str
    order_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "FragmentScheme":
        source = Path(path)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != FRAGMENT_SHA256:
            raise ValueError("AIFC fragment dictionary hash differs from the pinned official blob")
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != ("First-Order Group", "SMARTs", "Priority"):
                raise ValueError("AIFC fragment dictionary columns changed")
            rows = list(reader)
        if len(rows) != 100 or len({row["First-Order Group"] for row in rows}) != 100:
            raise ValueError("AIFC fragment dictionary must contain 100 unique entries")
        order = np.argsort(np.asarray([row["Priority"] for row in rows], dtype=object))
        sorted_rows = [rows[int(index)] for index in order]
        patterns = tuple(Chem.MolFromSmarts(row["SMARTs"]) for row in sorted_rows)
        if any(pattern is None for pattern in patterns):
            raise ValueError("AIFC fragment dictionary contains an invalid SMARTS")
        names = tuple(row["First-Order Group"] for row in sorted_rows)
        order_hash = hashlib.sha256("\n".join(names).encode()).hexdigest()
        return cls(
            names=names,
            patterns=patterns,  # type: ignore[arg-type]
            priorities=tuple(int(row["Priority"]) for row in sorted_rows),
            sha256=digest,
            order_sha256=order_hash,
        )


@dataclass(frozen=True)
class AIFCGraph:
    fragment_nodes: torch.Tensor
    fragment_edge_index: torch.Tensor
    fragment_edges: torch.Tensor
    fragment_batch: torch.Tensor
    motif_nodes: torch.Tensor
    motif_edge_index: torch.Tensor
    motif_edges: torch.Tensor
    motif_batch: torch.Tensor
    atom_count: int
    unknown_atoms: int
    fragment_names: tuple[str, ...]

    def to(self, device: torch.device | str) -> "AIFCGraph":
        return AIFCGraph(
            fragment_nodes=self.fragment_nodes.to(device),
            fragment_edge_index=self.fragment_edge_index.to(device),
            fragment_edges=self.fragment_edges.to(device),
            fragment_batch=self.fragment_batch.to(device),
            motif_nodes=self.motif_nodes.to(device),
            motif_edge_index=self.motif_edge_index.to(device),
            motif_edges=self.motif_edges.to(device),
            motif_batch=self.motif_batch.to(device),
            atom_count=self.atom_count,
            unknown_atoms=self.unknown_atoms,
            fragment_names=self.fragment_names,
        )


def canonicalize_view(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"AIFC could not parse SMILES: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _official_fragment_assignments(
    molecule: Chem.Mol, scheme: FragmentScheme
) -> tuple[list[tuple[int, ...]], list[str], int]:
    assignments: list[tuple[int, ...]] = []
    names: list[str] = []
    claimed: set[int] = set()
    for name, pattern in zip(scheme.names, scheme.patterns, strict=True):
        matches = list(molecule.GetSubstructMatches(pattern))
        if matches:
            for index, item in enumerate(matches):
                item_set = set(item)
                other_matches = matches[:index] + matches[index + 1 :]
                if not item_set.isdisjoint(set(sum(other_matches, ()))):
                    matches = other_matches
            for match in matches:
                atoms = set(match)
                if claimed.isdisjoint(atoms):
                    assignments.append(tuple(sorted(atoms)))
                    names.append(name)
                    claimed.update(atoms)
    unknown = sorted(set(range(molecule.GetNumAtoms())) - claimed)
    assignments.extend((atom,) for atom in unknown)
    names.extend("unknown" for _ in unknown)
    if not assignments or set(sum(assignments, ())) != set(range(molecule.GetNumAtoms())):
        raise RuntimeError("AIFC fragmentation did not assign every atom exactly once")
    return assignments, names, len(unknown)


def smiles_to_aifc_graph(smiles: str, scheme: FragmentScheme) -> AIFCGraph:
    canonical = canonicalize_view(smiles)
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:
        raise ValueError(f"AIFC could not parse canonical SMILES: {canonical}")
    assignments, names, unknown_atoms = _official_fragment_assignments(molecule, scheme)
    atom_to_fragment = {
        atom: fragment
        for fragment, atoms in enumerate(assignments)
        for atom in atoms
    }
    atom_values = [atom_features(atom) for atom in molecule.GetAtoms()]
    fragment_nodes: list[list[float]] = []
    fragment_batch: list[int] = []
    local_atom: dict[int, int] = {}
    for fragment, atoms in enumerate(assignments):
        for atom in atoms:
            local_atom[atom] = len(fragment_nodes)
            fragment_nodes.append(atom_values[atom])
            fragment_batch.append(fragment)

    fragment_src: list[int] = []
    fragment_dst: list[int] = []
    fragment_edges: list[list[float]] = []
    motif_edges_by_pair: dict[tuple[int, int], list[float]] = {}
    for bond in molecule.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = bond_features(bond)
        begin_fragment, end_fragment = atom_to_fragment[begin], atom_to_fragment[end]
        if begin_fragment == end_fragment:
            fragment_src.extend((local_atom[begin], local_atom[end]))
            fragment_dst.extend((local_atom[end], local_atom[begin]))
            fragment_edges.extend((feature, feature))
        else:
            motif_edges_by_pair.setdefault((begin_fragment, end_fragment), feature)
            motif_edges_by_pair.setdefault((end_fragment, begin_fragment), feature)

    motif_nodes = torch.zeros((len(assignments), len(scheme.names)), dtype=torch.float32)
    name_to_index = {name: index for index, name in enumerate(scheme.names)}
    for index, name in enumerate(names):
        if name != "unknown":
            motif_nodes[index, name_to_index[name]] = 1.0
    motif_pairs = list(motif_edges_by_pair)
    return AIFCGraph(
        fragment_nodes=torch.tensor(fragment_nodes, dtype=torch.float32),
        fragment_edge_index=torch.tensor(
            [fragment_src, fragment_dst], dtype=torch.long
        ).reshape(2, -1),
        fragment_edges=torch.tensor(fragment_edges, dtype=torch.float32).reshape(-1, BOND_FEATURE_DIM),
        fragment_batch=torch.tensor(fragment_batch, dtype=torch.long),
        motif_nodes=motif_nodes,
        motif_edge_index=torch.tensor(motif_pairs, dtype=torch.long).reshape(-1, 2).T,
        motif_edges=torch.tensor(
            [motif_edges_by_pair[pair] for pair in motif_pairs], dtype=torch.float32
        ).reshape(-1, BOND_FEATURE_DIM),
        motif_batch=torch.zeros(len(assignments), dtype=torch.long),
        atom_count=molecule.GetNumAtoms(),
        unknown_atoms=unknown_atoms,
        fragment_names=tuple(names),
    )


def _offset_edges(edge_index: torch.Tensor, offset: int) -> torch.Tensor:
    return edge_index + offset if edge_index.numel() else edge_index


def batch_aifc_graphs(graphs: Sequence[AIFCGraph]) -> AIFCGraph:
    if not graphs:
        raise ValueError("Cannot batch an empty AIFC graph sequence")
    fragment_node_offset = 0
    motif_offset = 0
    fragment_edges: list[torch.Tensor] = []
    motif_edges: list[torch.Tensor] = []
    fragment_batches: list[torch.Tensor] = []
    motif_batches: list[torch.Tensor] = []
    for graph_index, graph in enumerate(graphs):
        fragment_edges.append(_offset_edges(graph.fragment_edge_index, fragment_node_offset))
        motif_edges.append(_offset_edges(graph.motif_edge_index, motif_offset))
        fragment_batches.append(graph.fragment_batch + motif_offset)
        motif_batches.append(torch.full_like(graph.motif_batch, graph_index))
        fragment_node_offset += len(graph.fragment_nodes)
        motif_offset += len(graph.motif_nodes)
    return AIFCGraph(
        fragment_nodes=torch.cat([graph.fragment_nodes for graph in graphs]),
        fragment_edge_index=torch.cat(fragment_edges, dim=1),
        fragment_edges=torch.cat([graph.fragment_edges for graph in graphs]),
        fragment_batch=torch.cat(fragment_batches),
        motif_nodes=torch.cat([graph.motif_nodes for graph in graphs]),
        motif_edge_index=torch.cat(motif_edges, dim=1),
        motif_edges=torch.cat([graph.motif_edges for graph in graphs]),
        motif_batch=torch.cat(motif_batches),
        atom_count=sum(graph.atom_count for graph in graphs),
        unknown_atoms=sum(graph.unknown_atoms for graph in graphs),
        fragment_names=tuple(name for graph in graphs for name in graph.fragment_names),
    )


def graph_audit(graphs: Sequence[AIFCGraph]) -> dict[str, Any]:
    atoms = sum(graph.atom_count for graph in graphs)
    unknown = sum(graph.unknown_atoms for graph in graphs)
    return {
        "molecules": len(graphs),
        "atoms": atoms,
        "fragments": sum(len(graph.fragment_names) for graph in graphs),
        "unknown_atoms": unknown,
        "unknown_atom_ratio": unknown / atoms if atoms else 0.0,
        "molecules_with_unknown": sum(graph.unknown_atoms > 0 for graph in graphs),
        "molecule_unknown_ratio": (
            sum(graph.unknown_atoms > 0 for graph in graphs) / len(graphs) if graphs else 0.0
        ),
        "fragmentation_failures": 0,
    }


__all__ = [
    "AIFCGraph",
    "ATOM_FEATURE_DIM",
    "BOND_FEATURE_DIM",
    "FRAGMENT_BLOB",
    "FRAGMENT_COMMIT",
    "FRAGMENT_SHA256",
    "FragmentScheme",
    "batch_aifc_graphs",
    "canonicalize_view",
    "graph_audit",
    "smiles_to_aifc_graph",
]
