"""Checkpoint-compatible AIonopedia molecular graph preprocessing.

Adapted from AIonopedia/AIonopedia-public at commit
17e2f550f91eadcdec39f467c0443f5446d9713c (MIT license).  The apparently
ineffective donor/acceptor membership test is intentionally preserved because
it is part of the released checkpoint's input contract.
"""

from __future__ import annotations

from typing import Any

import torch
from rdkit import Chem
from torch_geometric.data import Data


NODE_FEATURE_DIMENSION = 35
EDGE_FEATURE_DIMENSION = 11
ATOM_SYMBOL_BUCKETS = (
    "C", "H", "O", "N", "S", "Cl", "F", "Br", "P", "I", "B", "Sn",
    "Se", "Si", "Hg", "Ge", "Te_or_other",
)
HYBRIDIZATION_BUCKETS = (
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
)
BOND_TYPE_BUCKETS = (
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
)
BOND_STEREO_BUCKETS = (
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
)


def _one_hot(value: Any, allowed: tuple[Any, ...]) -> list[bool]:
    if value not in allowed:
        value = allowed[-1]
    return [value == candidate for candidate in allowed]


def _atom_features(
    atom: Chem.Atom,
    atom_index: int,
    ring_info: Chem.RingInfo,
    donor_matches: list[tuple[int, ...]],
    acceptor_matches: list[tuple[int, ...]],
) -> torch.Tensor:
    symbol = atom.GetSymbol()
    symbol = symbol if symbol in ATOM_SYMBOL_BUCKETS[:-1] else ATOM_SYMBOL_BUCKETS[-1]
    values: list[bool | int] = [symbol == item for item in ATOM_SYMBOL_BUCKETS]
    values.append(atom.GetDegree())
    values.extend(_one_hot(atom.GetHybridization(), HYBRIDIZATION_BUCKETS))
    values.extend(
        [
            atom.GetImplicitValence(),
            atom.GetIsAromatic(),
            *(ring_info.IsAtomInRingOfSize(atom_index, size) for size in range(3, 9)),
            atom_index in donor_matches,
            atom_index in acceptor_matches,
            atom.GetFormalCharge(),
        ]
    )
    return torch.tensor(values, dtype=torch.float32)


def _bond_features(bond: Chem.Bond) -> torch.Tensor:
    return torch.tensor(
        _one_hot(bond.GetBondType(), BOND_TYPE_BUCKETS)
        + _one_hot(bond.GetStereo(), BOND_STEREO_BUCKETS)
        + [bond.GetIsConjugated(), bond.IsInRing()],
        dtype=torch.float32,
    )


def smiles_to_graph(smiles: str) -> Data:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"AIonopedia could not parse a non-empty molecule: {smiles!r}")
    Chem.AssignStereochemistry(molecule)
    donor_patterns = (
        "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]",
        "[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]",
    )
    acceptor_patterns = (
        "[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]",
        "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
        "$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,"
        "$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]",
    )
    donor_matches = list(
        set(
            match
            for pattern in donor_patterns
            for match in molecule.GetSubstructMatches(Chem.MolFromSmarts(pattern))
        )
    )
    acceptor_matches = list(
        set(
            match
            for pattern in acceptor_patterns
            for match in molecule.GetSubstructMatches(Chem.MolFromSmarts(pattern))
        )
    )
    ring_info = molecule.GetRingInfo()
    nodes = []
    for index, atom in enumerate(molecule.GetAtoms()):
        nodes.append(
            torch.cat(
                (
                    torch.tensor([atom.GetAtomicNum()], dtype=torch.float32),
                    _atom_features(atom, index, ring_info, donor_matches, acceptor_matches),
                )
            )
        )
    edges: list[list[int]] = []
    edge_features: list[torch.Tensor] = []
    for bond in molecule.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = _bond_features(bond)
        edges.extend(([begin, end], [end, begin]))
        edge_features.extend((features, features))
    graph = Data(
        x=torch.stack(nodes),
        edge_index=(
            torch.tensor(edges, dtype=torch.int64).t().contiguous()
            if edges
            else torch.empty((2, 0), dtype=torch.int64)
        ),
        edge_attr=(
            torch.stack(edge_features)
            if edge_features
            else torch.empty((0, EDGE_FEATURE_DIMENSION), dtype=torch.float32)
        ),
    )
    validate_graph(graph)
    return graph


def empty_graph() -> Data:
    return Data(
        x=torch.empty((0, NODE_FEATURE_DIMENSION), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.int64),
        edge_attr=torch.empty((0, EDGE_FEATURE_DIMENSION), dtype=torch.float32),
    )


def validate_graph(graph: Data) -> None:
    if graph.x.shape[1:] != (NODE_FEATURE_DIMENSION,) or graph.x.dtype != torch.float32:
        raise ValueError("AIonopedia graph requires float32 35D atom features")
    if graph.edge_index.shape[0:1] != (2,) or graph.edge_index.dtype != torch.int64:
        raise ValueError("AIonopedia graph requires int64 COO edge indices")
    if graph.edge_attr.shape[1:] != (EDGE_FEATURE_DIMENSION,) or graph.edge_attr.dtype != torch.float32:
        raise ValueError("AIonopedia graph requires float32 11D edge features")
    if graph.edge_index.shape[1] != graph.edge_attr.shape[0]:
        raise ValueError("AIonopedia graph edge index and feature counts differ")


__all__ = [
    "EDGE_FEATURE_DIMENSION",
    "NODE_FEATURE_DIMENSION",
    "empty_graph",
    "smiles_to_graph",
    "validate_graph",
]
