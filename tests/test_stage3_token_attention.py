"""Behavioral contracts for late entity fusion and physical condition attention."""
from dataclasses import replace

import pytest
import torch

from common.condition_coordinates import physical_conditions
from common.entity_inputs import EntityInputs
from stage3.config import Stage3ModelConfig
from stage3.data import ResolvedTaskSpec, sanitize_task
from stage3.model import Stage3SparseModel, GLOBAL, group_owner, private_owner
from stage3.token_attention import entity_sequence


def spec(task="experiment/test", *, partner=False, organic=False, columns=("temperature_K",)):
    return ResolvedTaskSpec(task, "target", (), columns, "solute_solvent" if organic else "il", "", "system", 1,
        "thermophysical", "interaction" if partner else "none",
        ("solute",) if organic else ("cation", "anion"), ("solvent",) if organic else (("solute",) if partner else ()),
        True, 1., 1, {}, "formal_charge_v1")


def model_for(task_spec, experts=1, partner_injection="l1"):
    # Tiny width uses the exact production modules, without allocating 1024D optimizers.
    config = Stage3ModelConfig(global_experts=experts, group_experts=experts, private_experts=1,
        dropout=0., expert_architecture="entity_token_attention_v1", attention_heads=2,
        partner_injection=partner_injection)
    model = Stage3SparseModel(config, {task_spec.task_id: task_spec}, 16,
        initialization_seed=42, entity_inputs=True, entity_input_dim=1235)
    model.bind_condition_statistics({task_spec.task_id: {"conditions": {
        name: {"mean": 300., "scale": 20.} for name in task_spec.condition_columns}}})
    return model


def ions():
    return EntityInputs.from_slots(torch.randn(2, 2, 1235), torch.tensor([[0, 1], [0, 1]]))


def molecule(role=2):
    return EntityInputs.from_slots(torch.randn(2, 1, 1235), torch.full((2, 1), role))


def test_physical_coordinates_have_shared_meaning_and_strict_types():
    first, mask = physical_conditions(torch.tensor([[1.]]), ("temperature_K",),
                                     {"temperature_K": {"mean": 300., "scale": 20.}})
    second, _ = physical_conditions(torch.tensor([[2.]]), ("temperature_K",),
                                   {"temperature_K": {"mean": 280., "scale": 20.}})
    torch.testing.assert_close(first, second)
    assert first[0, 0] == pytest.approx(.32) and mask.tolist() == [[True, False, False, False]]
    zero, mask = physical_conditions(torch.zeros(1, 1), ("pressure_kPa",),
                                    {"pressure_kPa": {"mean": 0., "scale": 1.}})
    assert zero.sum() == 0 and mask[0, 1]
    with pytest.raises(ValueError, match="Unsupported"):
        physical_conditions(torch.zeros(1, 1), ("phase",), {})
    with pytest.raises(ValueError, match="statistics"):
        physical_conditions(torch.zeros(1, 1), ("temperature_K",), {})
    with pytest.raises(ValueError, match="statistics"):
        physical_conditions(torch.zeros(1, 1), ("temperature_K",), {"temperature_K": {"mean": 300., "scale": 0.}})


def test_tokens_preserve_solvent_first_order_masks_and_no_early_partner_mixing():
    task = spec(partner=True, organic=True)
    model = model_for(task, experts=2).eval()
    solute, solvent = molecule(0), molecule(1)
    values, roles, mask, _ = entity_sequence(solute, solvent, task, 1235)
    torch.testing.assert_close(values[:, 0], solvent.values[:, 0])
    torch.testing.assert_close(values[:, 2], solute.values[:, 0])
    assert roles.tolist() == [[1, 0, 0], [1, 0, 0]]
    assert mask.tolist() == [[True, False, True]] * 2
    first = model(task.task_id, solute, torch.zeros(2, 1), partner_embedding=solvent)
    altered = EntityInputs(solute.values * 2, solute.roles, solute.mask)
    second = model(task.task_id, altered, torch.zeros(2, 1), partner_embedding=solvent)
    torch.testing.assert_close(first.diagnostics['z_global'][:, 0], second.diagnostics['z_global'][:, 0])
    torch.testing.assert_close(first.diagnostics['z_group'][:, 0], second.diagnostics['z_group'][:, 0])
    assert not torch.equal(first.diagnostics['z_global'][:, 2], second.diagnostics['z_global'][:, 2])
    for name in ('l2_global_candidates', 'l2_group_candidates', 'l2_private_candidates'):
        assert not torch.equal(first.diagnostics[name][:, :, 0], second.diagnostics[name][:, :, 0])
    # Slot identity is retained through L2 and the final concatenation head.
    assert first.diagnostics['l2_global_candidates'].shape == (2, 2, 3, 16)
    assert model.towers[sanitize_task(task.task_id)].layers[0].in_features == 51
    assert not model.interactions and not model.condition_films
    with pytest.raises(ValueError, match="partner"):
        model(task.task_id, solute, torch.zeros(2, 1))


def test_only_l2_sees_conditions_and_all_attention_owners_can_learn():
    task = spec()
    model = model_for(task).eval()
    primary = ions()
    cold = model(task.task_id, primary, torch.zeros(2, 1))
    hot = model(task.task_id, primary, torch.full((2, 1), 10.))
    for name in ['z_global', 'z_group', 'l1_global_gate', 'l1_group_gate', 'task_gate']:
        torch.testing.assert_close(cold.diagnostics[name], hot.diagnostics[name], rtol=0, atol=0)
    for name in ['l2_global_candidates', 'l2_group_candidates', 'l2_private_candidates']:
        assert not torch.equal(cold.diagnostics[name], hot.diagnostics[name])
        assert torch.count_nonzero(cold.diagnostics[name][:, :, 2]) == 0
    assert not torch.equal(cold.predictions, hot.predictions)
    changed_values = primary.values.clone(); changed_values[:, 0] *= 2
    changed = model(task.task_id, EntityInputs(changed_values, primary.roles, primary.mask), torch.zeros(2, 1))
    # Cation changes propagate into the anion token without collapsing the two slots.
    assert not torch.equal(cold.diagnostics['z_global'][:, 1], changed.diagnostics['z_global'][:, 1])
    (hot.predictions * torch.tensor([1., -2.])).sum().backward()
    for prefix in ['l1_global_experts', 'l1_group_experts', 'l2_global_experts', 'l2_group_experts',
                   'private_experts', 'global_conditions', 'group_conditions', 'private_conditions']:
        gradients = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients), prefix
    # Expert axis is combined, entity axis is preserved until the tower.
    candidates = torch.cat([hot.diagnostics[k] for k in
                           ['l2_global_candidates', 'l2_group_candidates', 'l2_private_candidates']], 1)
    mixed = (hot.diagnostics['task_gate'][:, :, None, None] * candidates).sum(1)
    expected = model.task_normalizations[sanitize_task(task.task_id)](hot.diagnostics['z_group'] + mixed)
    expected[:, 2] = 0
    assert mixed.shape == (2, 3, 16)
    mask = torch.tensor([[1., 1., 0.]] * 2)
    torch.testing.assert_close(hot.predictions, model.towers[sanitize_task(task.task_id)](
        torch.cat([expected.flatten(1), mask], -1)))


def test_single_charged_entity_padding_and_owner_lifetimes():
    from stage3.home import validate_transfer_architecture

    task = replace(spec(columns=()), primary_slots=("molecule",))
    model = model_for(task).eval()
    primary = molecule(0)
    initial = model(task.task_id, primary, torch.empty(2, 0))
    padded = primary.values.clone(); padded[:, 1] = 500.
    again = model(task.task_id, EntityInputs(padded, primary.roles, primary.mask), torch.empty(2, 0))
    torch.testing.assert_close(initial.predictions, again.predictions, rtol=0, atol=0)
    assert not model.private_conditions[sanitize_task(task.task_id)].projectors
    for owners in [{GLOBAL, group_owner(task.meta_group), private_owner(task.task_id)},
                   {group_owner(task.meta_group), private_owner(task.task_id)}, {private_owner(task.task_id)}]:
        model.set_trainable_owners(owners)
        assert all(p.requires_grad == (owner in owners) for p, owner in model.parameter_ownership().items())
    with pytest.raises(ValueError, match="normalized"):
        model(task.task_id, primary, torch.zeros(2, 1))
    with pytest.raises(ValueError, match="ordered"):
        EntityInputs.from_slots(torch.randn(2, 2, 1235), torch.tensor([[1, 0], [1, 0]]))
    historical = replace(model.model_config, expert_architecture="mlp", attention_heads=8)
    with pytest.raises(ValueError, match="architecture mismatch"):
        validate_transfer_architecture(historical, model.model_config, {}, {})
    assert Stage3ModelConfig().to_dict().get('expert_architecture') is None


@pytest.mark.parametrize("organic", (False, True))
def test_partner_enters_group_l2_only(organic):
    task = spec(partner=True, organic=organic)
    model = model_for(task, experts=2, partner_injection="group_l2").eval()
    previous = model_for(task, experts=2)
    assert model.state_dict().keys() == previous.state_dict().keys()
    assert all(torch.equal(value, previous.state_dict()[key]) for key, value in model.state_dict().items())
    with torch.no_grad():
        model.l1_group_normalizations[task.meta_group].bias.fill_(0.2)
    primary, partner = (molecule(0), molecule(1)) if organic else (ions(), molecule())
    third = primary if organic else partner
    values = third.values.detach().clone().requires_grad_(True)
    third = EntityInputs(values, third.roles, third.mask)
    if organic:
        primary = third
    else:
        partner = third
    first = model(task.task_id, primary, torch.zeros(2, 1), partner_embedding=partner)
    altered = EntityInputs(values * 2, third.roles, third.mask)
    second = model(task.task_id, altered if organic else primary, torch.zeros(2, 1),
                   partner_embedding=partner if organic else altered)
    for name in ('z_global', 'z_group', 'l1_global_gate', 'l1_group_gate', 'task_gate',
                 'l2_global_candidates', 'l2_private_candidates'):
        torch.testing.assert_close(first.diagnostics[name], second.diagnostics[name], rtol=0, atol=0)
    for name in ('z_global', 'z_group', 'l2_global_candidates', 'l2_private_candidates'):
        assert torch.count_nonzero(first.diagnostics[name][..., 2, :]) == 0
    assert not torch.equal(first.diagnostics['l2_group_candidates'][:, :, 0], second.diagnostics['l2_group_candidates'][:, :, 0])
    assert not torch.equal(first.predictions, second.predictions)
    for name in ('l2_global_candidates', 'l2_private_candidates'):
        gradient = torch.autograd.grad(first.diagnostics[name].sum(), values, retain_graph=True, allow_unused=True)[0]
        assert gradient is None or torch.count_nonzero(gradient) == 0
    gradient = torch.autograd.grad((first.predictions * torch.tensor([1., -2.])).sum(), values)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_group_l2_partner_configuration_and_identity_boundaries():
    from stage3.home import validate_transfer_architecture
    from stage3.token_attention import attention_contract

    original = Stage3ModelConfig(expert_architecture="entity_token_attention_v1")
    changed = replace(original, partner_injection="group_l2")
    assert 'partner_injection' not in original.to_dict()
    assert 'partner_injection' not in Stage3ModelConfig().to_dict()
    assert changed.to_dict()['partner_injection'] == 'group_l2'
    assert attention_contract(original) != attention_contract(changed)
    for source, target in ((original, changed), (changed, original)):
        with pytest.raises(ValueError, match="architecture mismatch"):
            validate_transfer_architecture(source, target, {}, {})
    with pytest.raises(ValueError, match="partner_injection"):
        replace(original, partner_injection="global_l1")
    with pytest.raises(ValueError, match="token attention"):
        Stage3ModelConfig(partner_injection="group_l2")
    task = replace(spec(columns=()), primary_slots=("molecule",))
    model = model_for(task, partner_injection="group_l2").eval()
    initial = model(task.task_id, molecule(0), torch.empty(2, 0))
    assert torch.isfinite(initial.predictions).all()
    for name in ('l2_global_candidates', 'l2_group_candidates', 'l2_private_candidates'):
        assert torch.count_nonzero(initial.diagnostics[name][..., 1:, :]) == 0
