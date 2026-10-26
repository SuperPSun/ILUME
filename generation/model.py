"""Ion-pair-aware graph policy for one fixed design request."""
from __future__ import annotations

from functools import lru_cache

from rdkit import Chem
import torch
from torch import nn


@lru_cache(maxsize=100000)
def graph_tensors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    atomic = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    features = torch.tensor([[a.GetFormalCharge() / 4, float(a.GetIsAromatic()),
        a.GetDegree() / 4, a.GetIsotope() / 16, int(a.GetChiralTag()) / 4,
        float(a.GetNoImplicit())] for a in mol.GetAtoms()], dtype=torch.float32)
    edges, bonds = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend(((i, j), (j, i)))
        bonds.extend(([bond.GetBondTypeAsDouble(), int(bond.GetStereo()) / 6, float(bond.GetIsConjugated())],) * 2)
    return atomic, features, torch.tensor(edges, dtype=torch.long).reshape(-1, 2), torch.tensor(bonds, dtype=torch.float32).reshape(-1, 3)


class GraphEncoder(nn.Module):
    def __init__(self, width: int, depth: int):
        super().__init__()
        self.atoms = nn.Embedding(119, width)
        self.features = nn.Linear(6, width)
        self.messages = nn.ModuleList(nn.Linear(width + 3, width) for _ in range(depth))
        self.updates = nn.ModuleList(nn.Linear(2 * width, width) for _ in range(depth))

    def forward(self, smiles: str):
        device = self.atoms.weight.device
        if not smiles:
            return self.atoms.weight.new_zeros(self.atoms.embedding_dim)
        atomic, features, edges, bonds = (x.to(device) for x in graph_tensors(smiles))
        values = self.atoms(atomic) + self.features(features)
        for message, update in zip(self.messages, self.updates, strict=True):
            aggregated = torch.zeros_like(values)
            if len(edges):
                aggregated.index_add_(0, edges[:, 1], torch.nn.functional.silu(
                    message(torch.cat((values[edges[:, 0]], bonds), -1))))
            values = values + torch.nn.functional.silu(update(torch.cat((values, aggregated), -1)))
        return values.mean(0)


class DedicatedGFN(nn.Module):
    def __init__(self, library, *, width: int = 64, depth: int = 3,
                 anion_conditioned: bool = True):
        super().__init__()
        if width < 4 or depth < 1:
            raise ValueError("Invalid generator model size")
        self.library = library
        self.graph = GraphEncoder(width, depth)
        self.policy = nn.Sequential(nn.Linear(6 * width + 4, width), nn.SiLU(), nn.Linear(width, 1))
        self.log_z = nn.Parameter(torch.zeros(()))
        self.anion_conditioned = anion_conditioned

    def log_probabilities(self, state, children):
        cation = self.graph(state.cation)
        current_anion = self.graph(self.library.anions[state.anion]["smiles"]) if state.anion else self.graph("")
        if not self.anion_conditioned:
            current_anion = self.graph("")
        logits = []
        for child in children:
            following = self.graph(child.cation)
            anion = (self.graph(self.library.anions[child.anion]["smiles"]) if not state.anion else current_anion)
            features = torch.cat((cation, following, current_anion, anion,
                cation * current_anion, following * anion,
                cation.new_tensor([float(not state.anion), float(bool(state.anion) and not state.cation),
                    float(bool(state.cation)), float(child.terminal)])))
            logits.append(self.policy(features).squeeze(-1))
        return torch.log_softmax(torch.stack(logits), 0)
