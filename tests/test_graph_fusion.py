from __future__ import annotations

import torch
from rdkit import Chem

from ilume_pretrain.encoders import DirectedMessagePassingEncoder
from ilume_pretrain.fusion import FusionTransformer
from ilume_pretrain.graph import featurize_mol, pack_graphs


def test_small_molecule_and_directed_reverse_edges():
    sodium = featurize_mol(Chem.MolFromSmiles("[Na+]"))
    ethane = featurize_mol(Chem.MolFromSmiles("CC"))
    assert sodium.atom_categorical.shape == (1, 7)
    assert sodium.bond_categorical.shape == (0, 4)

    packed = pack_graphs([sodium, ethane])
    assert packed.bond_index.shape == (2, 1)
    assert packed.directed_edge_index.shape == (2, 2)
    assert packed.reverse_edge_index.tolist() == [1, 0]
    assert torch.equal(
        packed.directed_edge_index[:, 0].flip(0),
        packed.directed_edge_index[:, 1],
    )


def test_dpmpnn_mask_vectors_preserve_topology_and_receive_gradients():
    graph = pack_graphs([featurize_mol(Chem.MolFromSmiles("CCO"))])
    original_edges = graph.directed_edge_index.clone()
    encoder = DirectedMessagePassingEncoder(d_model=16, depth=3, dropout=0.0)
    atom_mask = torch.tensor([True, False, False])
    bond_mask = torch.tensor([True, False])
    atom_tokens, bond_tokens = encoder(graph, atom_mask, bond_mask)
    (atom_tokens.sum() + bond_tokens.sum()).backward()

    assert torch.equal(graph.directed_edge_index, original_edges)
    assert encoder.atom_mask_feature.grad is not None
    assert encoder.bond_mask_feature.grad is not None


def test_fusion_layout_tracks_variable_smiles_atoms_bonds_and_descriptor():
    graphs = pack_graphs(
        [
            featurize_mol(Chem.MolFromSmiles("[Na+]")),
            featurize_mol(Chem.MolFromSmiles("CCO")),
        ]
    )
    d_model = 8
    smiles = torch.randn(2, 5, d_model)
    smiles_padding = torch.tensor(
        [[False, False, False, True, True], [False, False, False, False, False]]
    )
    atoms = torch.randn(graphs.atom_categorical.shape[0], d_model)
    bonds = torch.randn(graphs.bond_categorical.shape[0], d_model)
    descriptors = torch.randn(2, 8, d_model)
    fingerprints = torch.randn(2, 18, d_model)
    fusion = FusionTransformer(
        d_model=d_model,
        n_heads=2,
        num_layers=1,
        feedforward_dim=16,
        dropout=0.0,
    )
    fused, layout = fusion(
        smiles,
        smiles_padding,
        atoms,
        bonds,
        descriptors,
        fingerprints,
        graphs,
        torch.tensor([0, 2]),
    )

    assert layout.sequence_lengths.tolist() == [31, 37]
    assert layout.smiles_indices[0].tolist() == [1, 2, 3, -1, -1]
    assert layout.atom_indices.tolist() == [4, 6, 7, 8]
    assert layout.bond_indices.tolist() == [9, 10]
    assert layout.descriptor_indices[0].tolist() == list(range(5, 13))
    assert layout.descriptor_indices[1].tolist() == list(range(11, 19))
    assert layout.fingerprint_indices[0].tolist() == list(range(13, 31))
    assert layout.fingerprint_indices[1].tolist() == list(range(19, 37))
    assert fused.shape == (2, 37, d_model)
