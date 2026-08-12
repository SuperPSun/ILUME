from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from stage1.config import MaskingConfig
from stage1.graph import ATOM_FEATURE_NAMES, BOND_FEATURE_NAMES
from stage1.masking import (
    MultimodalCollator,
    MultimodalMasker,
    MultimodalPacker,
    curriculum_dropout_probability,
)
from stage1.model import MultimodalPretrainModel, _weighted_component


def test_role_weight_reduction_is_exact() -> None:
    losses = torch.tensor([1.0, 3.0, 5.0])
    roles = torch.tensor([0, 1, 2])
    weights = torch.tensor([2.0, 2.0, 1.0])
    numerator, denominator, role_numerators, role_denominators = (
        _weighted_component(losses, roles, weights)
    )
    assert numerator.item() == 13.0
    assert denominator.item() == 5.0
    assert (numerator / denominator).item() == pytest.approx(2.6)
    assert role_numerators.tolist() == [2.0, 6.0, 5.0]
    assert role_denominators.tolist() == [2.0, 2.0, 1.0]


def test_all_five_modalities_use_element_role_weights_and_component_means(
    tiny_config,
    tiny_samples,
) -> None:
    vocabulary, samples = tiny_samples
    batch = MultimodalCollator(
        vocabulary, tiny_config.masking, seed=tiny_config.data.seed
    )(samples)
    model = MultimodalPretrainModel(tiny_config, vocabulary)
    output = model(batch)
    role_weights = torch.tensor(tiny_config.loss.role_weights)

    expected: dict[str, list[tuple[torch.Tensor, ...]]] = {}
    smiles_mask = batch.masks.smiles_labels != -100
    token_roles = batch.roles[:, None].expand_as(smiles_mask)
    expected["smiles"] = [
        _weighted_component(
            F.cross_entropy(
                output.logits["smiles"][smiles_mask],
                batch.masks.smiles_labels[smiles_mask],
                reduction="none",
            ),
            token_roles[smiles_mask],
            role_weights,
        )
    ]

    atom_mask = batch.masks.atom_mask
    atom_roles = batch.roles[batch.graphs.atom_batch][atom_mask]
    expected["atom"] = [
        _weighted_component(
            F.cross_entropy(
                output.logits["atom"][name][atom_mask],
                batch.graphs.atom_categorical[atom_mask, column],
                reduction="none",
            ),
            atom_roles,
            role_weights,
        )
        for column, name in enumerate(ATOM_FEATURE_NAMES)
    ]

    bond_mask = batch.masks.bond_mask
    bond_roles = batch.roles[batch.graphs.bond_batch][bond_mask]
    expected["bond"] = [
        _weighted_component(
            F.cross_entropy(
                output.logits["bond"][name][bond_mask],
                batch.graphs.bond_categorical[bond_mask, column],
                reduction="none",
            ),
            bond_roles,
            role_weights,
        )
        for column, name in enumerate(BOND_FEATURE_NAMES)
    ]

    descriptor_mask = batch.masks.descriptor_loss_mask
    descriptor_roles = batch.roles[:, None].expand_as(descriptor_mask)
    expected["descriptor"] = [
        _weighted_component(
            F.smooth_l1_loss(
                output.logits["descriptor"][descriptor_mask],
                batch.descriptors[descriptor_mask],
                reduction="none",
            ),
            descriptor_roles[descriptor_mask],
            role_weights,
        )
    ]

    expected["fingerprint"] = []
    for family, logits in output.logits["fingerprint"].items():
        loss_mask = batch.masks.fingerprint_loss_mask[family]
        fingerprint_roles = batch.roles[:, None].expand_as(loss_mask)
        expected["fingerprint"].append(
            _weighted_component(
                F.binary_cross_entropy_with_logits(
                    logits[loss_mask],
                    batch.fingerprints.values[family][loss_mask],
                    reduction="none",
                ),
                fingerprint_roles[loss_mask],
                role_weights,
            )
        )

    for modality, components in expected.items():
        statistics = output.loss_statistics[modality]
        assert torch.allclose(
            statistics.numerators,
            torch.stack([component[0] for component in components]),
        )
        assert torch.equal(
            statistics.denominators,
            torch.stack([component[1] for component in components]),
        )
        assert torch.allclose(output.losses[modality], statistics.mean())


def test_modality_dropout_never_drops_all_and_supervises_dropped_content(
    tiny_samples,
):
    vocabulary, samples = tiny_samples
    config = MaskingConfig(
        smiles_ratio=0.0,
        atom_ratio=0.0,
        bond_ratio=0.0,
        descriptor_ratio=0.0,
        fingerprint_ratio=0.0,
        smiles_dropout=1.0,
        graph_dropout=1.0,
        descriptor_dropout=1.0,
        fingerprint_dropout=1.0,
    )
    batch = MultimodalCollator(vocabulary, config, seed=4)(samples)
    assert torch.equal(
        batch.masks.modality_dropped.sum(dim=1),
        torch.full((3,), 3),
    )
    for row, dropped in enumerate(batch.masks.modality_dropped):
        if dropped[0]:
            length = (~batch.token_padding_mask[row]).sum().item()
            assert (batch.masks.smiles_labels[row, 1 : length - 1] != -100).all()
        if dropped[1]:
            atom_start, atom_count = batch.graphs.atom_scopes[row]
            assert batch.masks.atom_mask[atom_start : atom_start + atom_count].all()
        if dropped[2]:
            assert torch.equal(
                batch.masks.descriptor_loss_mask[row], batch.descriptor_valid[row]
            )
        if dropped[3]:
            for family, valid in batch.fingerprints.valid.items():
                assert torch.equal(
                    batch.masks.fingerprint_loss_mask[family][row], valid[row]
                )


def test_asymmetric_masking_boosts_one_available_modality_and_is_reproducible(
    tiny_samples,
):
    vocabulary, all_samples = tiny_samples
    samples = all_samples[1:]
    config = MaskingConfig(
        smiles_ratio=0.0,
        atom_ratio=0.0,
        bond_ratio=0.0,
        descriptor_ratio=0.0,
        fingerprint_ratio=0.0,
        smiles_dropout=0.0,
        graph_dropout=0.0,
        descriptor_dropout=0.0,
        fingerprint_dropout=0.0,
        asymmetric_enabled=True,
        asymmetric_probability=1.0,
        asymmetric_ratio=1.0,
    )
    first = MultimodalCollator(vocabulary, config, seed=12)(samples)
    second = MultimodalCollator(vocabulary, config, seed=12)(samples)
    assert torch.equal(first.token_ids, second.token_ids)
    assert torch.equal(first.masks.atom_mask, second.masks.atom_mask)
    assert torch.equal(
        first.masks.descriptor_loss_mask, second.masks.descriptor_loss_mask
    )

    for row in range(len(samples)):
        length = (~first.token_padding_mask[row]).sum().item()
        smiles_full = bool(
            (first.masks.smiles_labels[row, 1 : length - 1] != -100).all()
        )
        atom_start, atom_count = first.graphs.atom_scopes[row]
        graph_full = bool(
            first.masks.atom_mask[atom_start : atom_start + atom_count].all()
        )
        descriptor_full = torch.equal(
            first.masks.descriptor_loss_mask[row], first.descriptor_valid[row]
        )
        fingerprint_full = all(
            torch.equal(
                first.masks.fingerprint_loss_mask[family][row], valid[row]
            )
            for family, valid in first.fingerprints.valid.items()
        )
        assert sum((smiles_full, graph_full, descriptor_full, fingerprint_full)) == 1


def test_curriculum_knots_and_single_atom_dynamic_masking(tiny_samples):
    assert curriculum_dropout_probability(0.0) == 0.0
    assert curriculum_dropout_probability(0.10) == 0.0
    assert curriculum_dropout_probability(0.60) == pytest.approx(0.05)
    assert curriculum_dropout_probability(1.0) == pytest.approx(0.10)

    vocabulary, samples = tiny_samples
    config = MaskingConfig(
        smiles_ratio=0.0,
        atom_ratio=1.0,
        bond_ratio=1.0,
        descriptor_ratio=0.0,
        fingerprint_ratio=0.0,
        smiles_dropout=0.0,
        graph_dropout=0.0,
        descriptor_dropout=0.0,
        fingerprint_dropout=0.0,
    )
    packed = MultimodalPacker(vocabulary)(samples)
    batch = MultimodalMasker(vocabulary, config, seed=3).apply(packed, 0, 1)
    atom_start, atom_count = batch.graphs.atom_scopes[0]
    assert atom_count == 1
    assert not batch.masks.atom_mask[atom_start]
    second_start, second_count = batch.graphs.atom_scopes[1]
    assert batch.masks.atom_mask[second_start : second_start + second_count].all()


def test_end_to_end_forward_backward_has_five_losses_and_shared_gradients(
    tiny_config,
    tiny_samples,
):
    vocabulary, samples = tiny_samples
    batch = MultimodalCollator(
        vocabulary, tiny_config.masking, seed=tiny_config.data.seed
    )(samples)
    model = MultimodalPretrainModel(tiny_config, vocabulary)
    output = model(batch)
    assert set(output.losses) == {
        "smiles",
        "descriptor",
        "atom",
        "bond",
        "fingerprint",
    }
    assert all(torch.isfinite(loss) for loss in output.losses.values())
    assert output.logits["smiles"].shape[:2] == batch.token_ids.shape
    assert output.logits["descriptor"].shape == (3, 217)
    assert output.logits["fingerprint"]["morgan"].shape == (3, 2048)
    assert output.logits["fingerprint"]["maccs"].shape == (3, 167)
    assert output.fused_cls.shape == (3, tiny_config.model.d_model)

    output.loss.backward()
    required_parameters = [
        model.smiles_encoder.token_embedding.weight,
        model.graph_encoder.atom_mask_feature,
        model.graph_encoder.bond_mask_feature,
        model.descriptor_encoder.group_encoders[4].input_projection[0].weight,
        model.fingerprint_encoder.chunk_encoder[0].weight,
        model.fusion.modality_embedding.weight,
        model.fusion.role_embedding.weight,
        model.smiles_head.bias,
        model.atom_heads["atomic_number"].weight,
        model.bond_heads["bond_type"].weight,
        model.descriptor_heads[4].weight,
        model.fingerprint_heads["morgan"].weight,
    ]
    assert all(parameter.grad is not None for parameter in required_parameters)


def test_encode_uses_complete_unmasked_modalities_and_preserves_forward(
    tiny_config,
    tiny_samples,
):
    vocabulary, samples = tiny_samples
    packed = MultimodalPacker(vocabulary)(samples)
    zero_masking = MaskingConfig(
        smiles_ratio=0.0,
        atom_ratio=0.0,
        bond_ratio=0.0,
        descriptor_ratio=0.0,
        fingerprint_ratio=0.0,
        smiles_dropout=0.0,
        graph_dropout=0.0,
        descriptor_dropout=0.0,
        fingerprint_dropout=0.0,
    )
    masked = MultimodalMasker(vocabulary, zero_masking, seed=1).apply(
        packed, 0, 1
    )
    model = MultimodalPretrainModel(tiny_config, vocabulary).eval()

    encoded = model.encode(packed)
    reconstructed = model(masked)

    assert encoded.shape == (len(samples), tiny_config.model.d_model)
    assert torch.equal(encoded, reconstructed.fused_cls)
    with pytest.raises(ValueError, match="unmasked batch"):
        model.encode(masked)


def test_zero_mask_configuration_returns_backward_safe_zero_loss(
    tiny_config,
    tiny_samples,
):
    vocabulary, samples = tiny_samples
    zero_masking = MaskingConfig(
        smiles_ratio=0.0,
        atom_ratio=0.0,
        bond_ratio=0.0,
        descriptor_ratio=0.0,
        fingerprint_ratio=0.0,
        smiles_dropout=0.0,
        graph_dropout=0.0,
        descriptor_dropout=0.0,
        fingerprint_dropout=0.0,
    )
    config = replace(tiny_config, masking=zero_masking)
    batch = MultimodalCollator(vocabulary, zero_masking, seed=1)(samples)
    output = MultimodalPretrainModel(config, vocabulary)(batch)
    assert output.loss.item() == 0.0
    output.loss.backward()

import torch
from rdkit import Chem

from stage1.encoders import DirectedMessagePassingEncoder
from stage1.fusion import FusionTransformer
from stage1.graph import featurize_mol, pack_graphs


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
