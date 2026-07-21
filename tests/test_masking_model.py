from __future__ import annotations

from dataclasses import replace

import torch

from ilume_pretrain.config import MaskingConfig
from ilume_pretrain.masking import MultimodalCollator
from ilume_pretrain.model import MultimodalPretrainModel


def test_modality_dropout_never_drops_all_and_supervises_dropped_content(
    tiny_samples,
):
    vocabulary, samples = tiny_samples
    config = MaskingConfig(
        smiles_ratio=0.0,
        atom_ratio=0.0,
        bond_ratio=0.0,
        descriptor_ratio=0.0,
        smiles_dropout=1.0,
        graph_dropout=1.0,
        descriptor_dropout=1.0,
    )
    batch = MultimodalCollator(vocabulary, config, seed=4)(samples)
    assert torch.equal(
        batch.masks.modality_dropped.sum(dim=1),
        torch.full((3,), 2),
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
                batch.masks.descriptor_loss_mask[row],
                batch.descriptor_valid[row],
            )
    assert batch.graphs.bond_scopes[0][1] == 0


def test_asymmetric_masking_boosts_exactly_one_modality_and_is_reproducible(
    tiny_samples,
):
    vocabulary, samples = tiny_samples
    config = MaskingConfig(
        smiles_ratio=0.0,
        atom_ratio=0.0,
        bond_ratio=0.0,
        descriptor_ratio=0.0,
        smiles_dropout=0.0,
        graph_dropout=0.0,
        descriptor_dropout=0.0,
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
        assert sum((smiles_full, graph_full, descriptor_full)) == 1


def test_end_to_end_forward_backward_has_all_losses_and_shared_gradients(
    tiny_config,
    tiny_samples,
):
    vocabulary, samples = tiny_samples
    batch = MultimodalCollator(
        vocabulary, tiny_config.masking, seed=tiny_config.data.seed
    )(samples)
    model = MultimodalPretrainModel(tiny_config, vocabulary)
    output = model(batch)
    assert set(output.losses) == {"smiles", "descriptor", "atom", "bond"}
    assert all(torch.isfinite(loss) for loss in output.losses.values())
    assert output.logits["smiles"].shape[:2] == batch.token_ids.shape
    assert output.logits["descriptor"].shape == (3, 217)
    assert output.fused_cls.shape == (3, tiny_config.model.d_model)

    output.loss.backward()
    required_parameters = [
        model.smiles_encoder.token_embedding.weight,
        model.graph_encoder.atom_mask_feature,
        model.graph_encoder.bond_mask_feature,
        model.descriptor_encoder.input_projection[0].weight,
        model.fusion.modality_embedding.weight,
        model.smiles_head.bias,
        model.atom_heads["atomic_number"].weight,
        model.bond_heads["bond_type"].weight,
        model.descriptor_head.weight,
    ]
    assert all(parameter.grad is not None for parameter in required_parameters)
    assert not any("role" in name for name, _ in model.named_parameters())


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
        smiles_dropout=0.0,
        graph_dropout=0.0,
        descriptor_dropout=0.0,
    )
    config = replace(tiny_config, masking=zero_masking)
    batch = MultimodalCollator(vocabulary, zero_masking, seed=1)(samples)
    output = MultimodalPretrainModel(config, vocabulary)(batch)
    assert output.loss.item() == 0.0
    output.loss.backward()
