from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from rdkit import Chem


ATOM_FEATURE_NAMES = (
    "atomic_number",
    "degree",
    "formal_charge",
    "chirality",
    "total_hydrogens",
    "hybridization",
    "aromatic",
)
ATOM_CARDINALITIES = (119, 7, 6, 5, 6, 9, 2)
BOND_FEATURE_NAMES = ("bond_type", "stereo", "conjugated", "in_ring")
BOND_CARDINALITIES = (5, 7, 2, 2)

_CHIRALITY = {
    Chem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.ChiralType.CHI_OTHER: 3,
}
_HYBRIDIZATION = {
    Chem.HybridizationType.UNSPECIFIED: 0,
    Chem.HybridizationType.S: 1,
    Chem.HybridizationType.SP: 2,
    Chem.HybridizationType.SP2: 3,
    Chem.HybridizationType.SP3: 4,
    Chem.HybridizationType.SP2D: 5,
    Chem.HybridizationType.SP3D: 6,
    Chem.HybridizationType.SP3D2: 7,
    Chem.HybridizationType.OTHER: 8,
}
_BOND_TYPES = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}
_BOND_STEREO = {
    Chem.BondStereo.STEREONONE: 0,
    Chem.BondStereo.STEREOANY: 1,
    Chem.BondStereo.STEREOZ: 2,
    Chem.BondStereo.STEREOE: 3,
    Chem.BondStereo.STEREOCIS: 4,
    Chem.BondStereo.STEREOTRANS: 5,
}


def _choice(value: int, allowed: Sequence[int]) -> int:
    try:
        return allowed.index(value)
    except ValueError:
        return len(allowed)


@dataclass(frozen=True)
class GraphRecord:
    atom_categorical: torch.Tensor
    atom_continuous: torch.Tensor
    bond_categorical: torch.Tensor
    bond_index: torch.Tensor


@dataclass(frozen=True)
class PackedGraph:
    atom_categorical: torch.Tensor
    atom_continuous: torch.Tensor
    bond_categorical: torch.Tensor
    bond_index: torch.Tensor
    directed_edge_index: torch.Tensor
    reverse_edge_index: torch.Tensor
    directed_to_bond: torch.Tensor
    atom_batch: torch.Tensor
    bond_batch: torch.Tensor
    atom_scopes: tuple[tuple[int, int], ...]
    bond_scopes: tuple[tuple[int, int], ...]

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "PackedGraph":
        return PackedGraph(
            atom_categorical=self.atom_categorical.to(device, non_blocking=non_blocking),
            atom_continuous=self.atom_continuous.to(device, non_blocking=non_blocking),
            bond_categorical=self.bond_categorical.to(device, non_blocking=non_blocking),
            bond_index=self.bond_index.to(device, non_blocking=non_blocking),
            directed_edge_index=self.directed_edge_index.to(device, non_blocking=non_blocking),
            reverse_edge_index=self.reverse_edge_index.to(device, non_blocking=non_blocking),
            directed_to_bond=self.directed_to_bond.to(device, non_blocking=non_blocking),
            atom_batch=self.atom_batch.to(device, non_blocking=non_blocking),
            bond_batch=self.bond_batch.to(device, non_blocking=non_blocking),
            atom_scopes=self.atom_scopes,
            bond_scopes=self.bond_scopes,
        )

    def pin_memory(self) -> "PackedGraph":
        return PackedGraph(
            atom_categorical=self.atom_categorical.pin_memory(),
            atom_continuous=self.atom_continuous.pin_memory(),
            bond_categorical=self.bond_categorical.pin_memory(),
            bond_index=self.bond_index.pin_memory(),
            directed_edge_index=self.directed_edge_index.pin_memory(),
            reverse_edge_index=self.reverse_edge_index.pin_memory(),
            directed_to_bond=self.directed_to_bond.pin_memory(),
            atom_batch=self.atom_batch.pin_memory(),
            bond_batch=self.bond_batch.pin_memory(),
            atom_scopes=self.atom_scopes,
            bond_scopes=self.bond_scopes,
        )


def featurize_mol(mol: Chem.Mol) -> GraphRecord:
    atom_rows: list[list[int]] = []
    atom_mass: list[list[float]] = []
    for atom in mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        atom_rows.append(
            [
                atomic_number - 1 if 1 <= atomic_number <= 118 else 118,
                _choice(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5]),
                _choice(atom.GetFormalCharge(), [-2, -1, 0, 1, 2]),
                _CHIRALITY.get(atom.GetChiralTag(), 4),
                _choice(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]),
                _HYBRIDIZATION.get(atom.GetHybridization(), 8),
                int(atom.GetIsAromatic()),
            ]
        )
        atom_mass.append([atom.GetMass() * 0.01])

    bond_rows: list[list[int]] = []
    endpoints: list[list[int]] = []
    for bond in mol.GetBonds():
        bond_rows.append(
            [
                _BOND_TYPES.get(bond.GetBondType(), 4),
                _BOND_STEREO.get(bond.GetStereo(), 6),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
            ]
        )
        endpoints.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    return GraphRecord(
        atom_categorical=torch.tensor(atom_rows, dtype=torch.long),
        atom_continuous=torch.tensor(atom_mass, dtype=torch.float32),
        bond_categorical=torch.tensor(
            bond_rows, dtype=torch.long
        ).reshape(-1, len(BOND_FEATURE_NAMES)),
        bond_index=torch.tensor(endpoints, dtype=torch.long).reshape(-1, 2).T,
    )


def pack_graphs(graphs: Sequence[GraphRecord]) -> PackedGraph:
    atom_categorical: list[torch.Tensor] = []
    atom_continuous: list[torch.Tensor] = []
    bond_categorical: list[torch.Tensor] = []
    bond_indices: list[torch.Tensor] = []
    atom_batch: list[torch.Tensor] = []
    bond_batch: list[torch.Tensor] = []
    atom_scopes: list[tuple[int, int]] = []
    bond_scopes: list[tuple[int, int]] = []
    atom_offset = 0
    bond_offset = 0

    for batch_index, graph in enumerate(graphs):
        atom_count = graph.atom_categorical.shape[0]
        bond_count = graph.bond_categorical.shape[0]
        atom_categorical.append(graph.atom_categorical)
        atom_continuous.append(graph.atom_continuous)
        bond_categorical.append(graph.bond_categorical)
        bond_indices.append(graph.bond_index + atom_offset)
        atom_batch.append(torch.full((atom_count,), batch_index, dtype=torch.long))
        bond_batch.append(torch.full((bond_count,), batch_index, dtype=torch.long))
        atom_scopes.append((atom_offset, atom_count))
        bond_scopes.append((bond_offset, bond_count))
        atom_offset += atom_count
        bond_offset += bond_count

    all_bonds = torch.cat(bond_indices, dim=1)
    bond_count = all_bonds.shape[1]
    if bond_count:
        directed = torch.empty((2, bond_count * 2), dtype=torch.long)
        directed[:, 0::2] = all_bonds
        directed[0, 1::2] = all_bonds[1]
        directed[1, 1::2] = all_bonds[0]
        reverse = torch.arange(bond_count * 2, dtype=torch.long) ^ 1
        directed_to_bond = torch.arange(bond_count, dtype=torch.long).repeat_interleave(2)
    else:
        directed = torch.empty((2, 0), dtype=torch.long)
        reverse = torch.empty((0,), dtype=torch.long)
        directed_to_bond = torch.empty((0,), dtype=torch.long)

    return PackedGraph(
        atom_categorical=torch.cat(atom_categorical, dim=0),
        atom_continuous=torch.cat(atom_continuous, dim=0),
        bond_categorical=torch.cat(bond_categorical, dim=0),
        bond_index=all_bonds,
        directed_edge_index=directed,
        reverse_edge_index=reverse,
        directed_to_bond=directed_to_bond,
        atom_batch=torch.cat(atom_batch, dim=0),
        bond_batch=torch.cat(bond_batch, dim=0),
        atom_scopes=tuple(atom_scopes),
        bond_scopes=tuple(bond_scopes),
    )
