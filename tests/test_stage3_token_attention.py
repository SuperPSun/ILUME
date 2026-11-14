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


def model_for(task_spec, experts=1, partner_injection="l1", split_entity_encoders=False, split_system_experts=False):
    # Tiny width uses the exact production modules, without allocating 1024D optimizers.
    config = Stage3ModelConfig(global_experts=experts, group_experts=experts, private_experts=1,
        dropout=0., expert_architecture="entity_token_attention_v1", attention_heads=2,
        partner_injection=partner_injection, split_entity_encoders=split_entity_encoders,
        split_system_experts=split_system_experts)
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


@pytest.mark.parametrize("mode, split", [("l1", False), ("group_l2", False),
    ("group_private_l2", False), ("all_l2", False), ("group_l2_encoded", False), ("group_l2_encoded", True),
    ("private_l2", False)])
def test_single_charged_entity_padding_and_owner_lifetimes(tmp_path, mode, split):
    from stage2.home_contract import transferable_state, load_transferable_state, state_hash
    from stage3.home import validate_transfer_architecture

    task = replace(spec(columns=()), primary_slots=("molecule",))
    model = model_for(task, partner_injection=mode, split_entity_encoders=split).eval()
    primary = molecule(0)
    initial = model(task.task_id, primary, torch.empty(2, 0))
    padded = primary.values.clone(); padded[:, 1] = 500.
    again = model(task.task_id, EntityInputs(padded, primary.roles, primary.mask), torch.empty(2, 0))
    torch.testing.assert_close(initial.predictions, again.predictions, rtol=0, atol=0)
    shared = {name: value + .01 for name, value in transferable_state(model).items()}
    untouched = {name: value.clone() for name, value in model.state_dict().items() if name not in shared}
    assert set(load_transferable_state(model, shared, state_hash(shared))) == set(shared)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in {**shared, **untouched}.items())
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = model_for(task, partner_injection=mode, split_entity_encoders=split).eval()
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(reloaded(task.task_id, primary, torch.empty(2, 0)).predictions,
                               model(task.task_id, primary, torch.empty(2, 0)).predictions, rtol=0, atol=0)
    assert not model.private_conditions[sanitize_task(task.task_id)].projectors
    for owners in [{GLOBAL, group_owner(task.meta_group), private_owner(task.task_id)},
                   {group_owner(task.meta_group), private_owner(task.task_id)}, {private_owner(task.task_id)}]:
        model.set_trainable_owners(owners)
        assert all(p.requires_grad == (owner in owners) for p, owner in model.parameter_ownership().items())
    with pytest.raises(ValueError, match="normalized"):
        model(task.task_id, primary, torch.zeros(2, 1))
    with pytest.raises(ValueError, match="ordered"):
        EntityInputs.from_slots(torch.randn(2, 2, 1235), torch.tensor([[1, 0], [1, 0]]))
    historical = replace(model.model_config, expert_architecture="mlp", attention_heads=8,
                         partner_injection="l1", split_entity_encoders=False)
    with pytest.raises(ValueError, match="architecture mismatch"):
        validate_transfer_architecture(historical, model.model_config, {}, {})
    assert Stage3ModelConfig().to_dict().get('expert_architecture') is None


@pytest.mark.parametrize("organic", (False, True))
@pytest.mark.parametrize("mode, split, receivers", [
    ("group_l2", False, {"group"}),
    ("group_private_l2", False, {"group", "private"}),
    ("all_l2", False, {"global", "group", "private"}),
    ("group_l2_encoded", False, {"group"}),
    ("group_l2_encoded", True, {"group"}),
    ("private_l2", False, {"private"}),
])
def test_partner_enters_declared_l2_branches_only(organic, mode, split, receivers, split_system_experts=False):
    task = spec(partner=True, organic=organic)
    model = model_for(task, experts=2, partner_injection=mode, split_entity_encoders=split,
                      split_system_experts=split_system_experts).eval()
    previous = model_for(task, experts=2)
    if not split and not split_system_experts:
        assert model.state_dict().keys() == previous.state_dict().keys()
        assert all(torch.equal(value, previous.state_dict()[key]) for key, value in model.state_dict().items())
    with torch.no_grad():
        from stage3.token_attention import select_system, system_branch
        select_system(model.l1_group_normalizations[task.meta_group],
                      system_branch(task) if split_system_experts else None).bias.fill_(0.2)
    primary, partner = (molecule(0), molecule(1)) if organic else (ions(), molecule())
    third = primary if organic else partner
    values = third.values.detach().clone().requires_grad_(True)
    third = EntityInputs(values, third.roles, third.mask)
    if organic:
        primary = third
    else:
        partner = third
    first = model(task.task_id, primary, torch.zeros(2, 1), partner_embedding=partner)
    if mode == "private_l2":
        paired = model_for(task, experts=2, partner_injection="group_private_l2").eval()
        paired.load_state_dict(model.state_dict(), strict=True)
        reference = paired(task.task_id, primary, torch.zeros(2, 1), partner_embedding=partner)
        torch.testing.assert_close(first.diagnostics['l2_private_candidates'],
                                   reference.diagnostics['l2_private_candidates'], rtol=0, atol=0)
    altered = EntityInputs(values * 2, third.roles, third.mask)
    second = model(task.task_id, altered if organic else primary, torch.zeros(2, 1),
                   partner_embedding=partner if organic else altered)
    unchanged = ['z_global', 'z_group', 'l1_global_gate', 'l1_group_gate', 'task_gate']
    unchanged += [f'l2_{owner}_candidates' for owner in ('global', 'group', 'private') if owner not in receivers]
    for name in unchanged:
        torch.testing.assert_close(first.diagnostics[name], second.diagnostics[name], rtol=0, atol=0)
    for name in ['z_global', 'z_group', *[f'l2_{owner}_candidates' for owner in ('global', 'group', 'private') if owner not in receivers]]:
        assert torch.count_nonzero(first.diagnostics[name][..., 2, :]) == 0
    for owner in receivers:
        name = f'l2_{owner}_candidates'
        assert not torch.equal(first.diagnostics[name][:, :, 0], second.diagnostics[name][:, :, 0])
    assert not torch.equal(first.predictions, second.predictions)
    for name in [f'l2_{owner}_candidates' for owner in ('global', 'group', 'private') if owner not in receivers]:
        gradient = torch.autograd.grad(first.diagnostics[name].sum(), values, retain_graph=True, allow_unused=True)[0]
        assert gradient is None or torch.count_nonzero(gradient) == 0
    gradient = torch.autograd.grad((first.predictions * torch.tensor([1., -2.])).sum(), values)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_typed_encoders_clone_initialization_route_roles_and_freeze():
    task = spec(partner=True, organic=True)
    shared = model_for(task, experts=2, partner_injection="group_l2_encoded").eval()
    typed = model_for(task, experts=2, partner_injection="group_l2_encoded", split_entity_encoders=True).eval()
    reference = shared.state_dict()
    for name, value in typed.state_dict().items():
        original = name.replace('.ionic.', '.').replace('.neutral.', '.')
        assert torch.equal(value, reference[original]), name
    for experts in [typed.l1_global_experts, *typed.l1_group_experts.values()]:
        for expert in experts:
            for ionic, neutral in zip(expert.ionic.parameters(), expert.neutral.parameters(), strict=True):
                assert ionic.data_ptr() != neutral.data_ptr()
    primary, partner = molecule(), molecule()
    args = (task.task_id, primary, torch.zeros(2, 1))
    torch.testing.assert_close(shared(*args, partner_embedding=partner).predictions,
                               typed(*args, partner_embedding=partner).predictions)
    ionic = {n: p.detach().clone() for n, p in typed.named_parameters() if '.ionic.' in n}
    neutral = {n: p.detach().clone() for n, p in typed.named_parameters() if '.neutral.' in n}
    optimizer = torch.optim.AdamW(typed.parameters(), lr=1e-3, weight_decay=.1)
    (typed(*args, partner_embedding=partner).predictions * torch.tensor([1., -2.])).sum().backward()
    assert all(p.grad is None for n, p in typed.named_parameters() if n in ionic)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in typed.named_parameters() if n in neutral)
    optimizer.step()
    assert all(torch.equal(p, ionic[n]) for n, p in typed.named_parameters() if n in ionic)
    assert any(not torch.equal(p, neutral[n]) for n, p in typed.named_parameters() if n in neutral)
    # Mixed charge roles route by the entity, not the organic task or slot number.
    typed.zero_grad(set_to_none=True)
    charged = molecule(0)
    mixed_primary = EntityInputs(charged.values, torch.tensor([[0, 0], [2, 0]]), charged.mask)
    output = typed(task.task_id, mixed_primary, torch.zeros(2, 1), partner_embedding=partner)
    output.predictions.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in typed.named_parameters() if '.ionic.' in n)
    typed.set_trainable_owners({private_owner(task.task_id)})
    frozen = {n: p.detach().clone() for n, p in typed.named_parameters() if n in ionic or n in neutral}
    typed.zero_grad(set_to_none=True)
    typed(*args, partner_embedding=partner).predictions.sum().backward()
    optimizer.step()
    assert all(torch.equal(p, frozen[n]) for n, p in typed.named_parameters() if n in frozen)


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
    with pytest.raises(ValueError, match="independently encoded"):
        replace(changed, split_entity_encoders=True)
    with pytest.raises(ValueError, match="boolean"):
        replace(changed, split_entity_encoders=1)
    with pytest.raises(ValueError, match="boolean"):
        replace(changed, split_system_experts=1)
    with pytest.raises(ValueError, match="GROUP/PRIVATE"):
        replace(changed, split_system_experts=True)
    variants = [changed, replace(changed, partner_injection="group_private_l2"),
                replace(changed, partner_injection="all_l2"),
                replace(changed, partner_injection="private_l2"),
                replace(changed, partner_injection="group_l2_encoded"),
                replace(changed, partner_injection="group_l2_encoded", split_entity_encoders=True)]
    variants.append(replace(changed, partner_injection="group_private_l2", split_system_experts=True))
    assert len({str(attention_contract(value)) for value in variants}) == len(variants)
    assert 'split_entity_encoders' not in original.to_dict()
    assert 'split_system_experts' not in original.to_dict()
    for source in variants:
        for target in variants:
            if source != target:
                with pytest.raises(ValueError, match="architecture mismatch"):
                    validate_transfer_architecture(source, target, {}, {})
    task = replace(spec(columns=()), primary_slots=("molecule",))
    model = model_for(task, partner_injection="group_l2").eval()
    initial = model(task.task_id, molecule(0), torch.empty(2, 0))
    assert torch.isfinite(initial.predictions).all()
    for name in ('l2_global_candidates', 'l2_group_candidates', 'l2_private_candidates'):
        assert torch.count_nonzero(initial.diagnostics[name][..., 1:, :]) == 0


def test_partner_research_configs_preserve_paired_data_and_training_recipe():
    from pathlib import Path
    from stage2.home_config import load_home_recipe
    from stage3.config import load_stage3_config
    from ablations.entity_home.config import load_research_config, PARTNER_VARIANTS

    stage2 = load_home_recipe('configs/v4/stage2/group_l2_partner.yaml')
    stage3 = load_stage3_config('configs/v4/stage3/group_l2_partner.yaml')
    for variant, (identity, mode, split) in PARTNER_VARIANTS.items():
        root = Path('configs/ablations/partner_injection_3407') / identity
        source = load_home_recipe(root / 'stage2.yaml')
        target = load_research_config(root / 'stage3.yaml')
        expected = replace(stage2.stage3.model, partner_injection=mode, split_entity_encoders=split,
                           split_system_experts=identity == 'S2')
        assert source == replace(stage2, stage3=replace(stage2.stage3, model=expected))
        assert target.model == source.stage3.model == expected
        assert target.ablation.variant == variant and target.ablation.model_id != 'ilume'
        assert target.data == stage3.data and target.training == stage3.training
        assert target.groups == stage3.groups and target.tasks == stage3.tasks
        assert target.initialization == replace(stage3.initialization,
            stage2_final=Path(f'outputs/ablations/partner_injection_3407/{identity}/stage2/train/stage2_final.pt'))
        assert load_research_config(root / 'stage3.yaml').to_dict()['ablation']['variant'] == variant
        with pytest.raises(ValueError, match="declared variant"):
            replace(target, model=stage3.model).validate()


def test_system_experts_isolate_tasks_and_transfer_only_matching_pretraining(tmp_path):
    from stage2.home_contract import transferable_state, load_transferable_state, state_hash
    from stage3.token_attention import unpretrained_system_parameters, system_expert_audit

    config = Stage3ModelConfig(expert_architecture='entity_token_attention_v1', attention_heads=2,
        global_experts=1, group_experts=2, dropout=0., partner_injection='group_private_l2',
        split_system_experts=True)
    ionic = spec('experiment/il_partner', partner=True)
    ionic = replace(ionic, meta_group='solvation')
    organic = replace(spec('experiment/organic', partner=True, organic=True), meta_group='solvation')
    hydration = replace(spec('experiment/hydration'), primary_slots=('solute',), meta_group='solvation')
    density = spec('simulation/density')
    transfer = replace(organic, task_id='simulation/transfer_organic')

    def build(specs, seed):
        model = Stage3SparseModel(config, {s.task_id: s for s in specs}, 16,
            initialization_seed=seed, entity_inputs=True, entity_input_dim=1235)
        model.bind_condition_statistics({s.task_id: {'conditions': {
            'temperature_K': {'mean': 300., 'scale': 20.}}} for s in specs})
        return model

    source = build((density, transfer), 42)
    target = build((ionic, organic, hydration, density, transfer), 71)
    before = {n: p.clone() for n, p in target.state_dict().items()}
    # Two independent parameter copies begin identically without shared storage.
    for module in (source.l1_global_experts[0], source.l2_global_experts[0]):
        for left, right in zip(module['il'].parameters(), module['non_il'].parameters(), strict=True):
            assert torch.equal(left, right) and left.data_ptr() != right.data_ptr()
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    for task, primary, partner in ((density, ions(), None), (transfer, molecule(), molecule())):
        optimizer.zero_grad(set_to_none=True)
        source(task.task_id, primary, torch.zeros(2, 1), partner_embedding=partner).predictions.sum().backward()
        optimizer.step()
    shared = transferable_state(source)
    missing = unpretrained_system_parameters(target)
    assert missing and not set(missing) & set(shared)
    assert set(load_transferable_state(target, shared, state_hash(shared))) == set(shared)
    assert all(torch.equal(target.state_dict()[n], value) for n, value in shared.items())
    assert all(torch.equal(target.state_dict()[n], before[n]) for n in missing)
    incomplete = dict(shared); incomplete.pop(next(iter(shared)))
    with pytest.raises(ValueError, match='incomplete'):
        load_transferable_state(target, incomplete, state_hash(incomplete))
    extra = {**shared, missing[0]: before[missing[0]]}
    with pytest.raises(ValueError, match='extra'):
        load_transferable_state(target, extra, state_hash(extra))
    with pytest.raises(ValueError, match='hash'):
        load_transferable_state(target, shared, 'corrupt')
    audit = system_expert_audit(target)
    assert audit['task_systems'][ionic.task_id] == 'il'
    assert audit['task_systems'][hydration.task_id] == 'non_il'
    assert audit['random_il_solvation_parameters'] == list(missing)
    from stage3.gradient_assembly import assemble_owner_gradients
    left = target.l1_global_experts[0]['il'].projection.weight
    right = target.l1_global_experts[0]['non_il'].projection.weight
    assembled = assemble_owner_gradients(target, {
        ionic.task_id: {left: torch.ones_like(left)}, organic.task_id: {right: torch.ones_like(right)}},
        target.task_specs, {}, task_batch_sizes={ionic.task_id: 3, organic.task_id: 1})
    # Parameter banks are separate; the original GLOBAL sample denominator stays shared.
    torch.testing.assert_close(assembled.gradients[left], torch.full_like(left, .75))
    torch.testing.assert_close(assembled.gradients[right], torch.full_like(right, .25))
    optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3, weight_decay=.1)
    for task, primary, partner, active, inactive in (
        (ionic, ions(), molecule(), 'il', 'non_il'),
        (organic, molecule(0), molecule(), 'non_il', 'il'),
    ):
        target.train(); optimizer.zero_grad(set_to_none=True)
        untouched = {n: p.detach().clone() for n, p in target.named_parameters() if f'.{inactive}.' in n}
        output = target(task.task_id, primary, torch.zeros(2, 1), partner_embedding=partner)
        (output.predictions * torch.tensor([1., -2.])).sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in target.named_parameters() if f'.{active}.' in n)
        assert all(p.grad is None for n, p in target.named_parameters() if n in untouched)
        optimizer.step()
        assert all(torch.equal(p, untouched[n]) for n, p in target.named_parameters() if n in untouched)
    target.zero_grad(set_to_none=True)
    target.set_trainable_owners({private_owner(ionic.task_id)})
    frozen = {n: p.detach().clone() for n, p in target.named_parameters() if '.il.' in n or '.non_il.' in n}
    target(ionic.task_id, ions(), torch.zeros(2, 1), partner_embedding=molecule()).predictions.sum().backward()
    optimizer.step()
    assert all(torch.equal(p, frozen[n]) for n, p in target.named_parameters() if n in frozen)
    target.eval()
    single = target(hydration.task_id, molecule(0), torch.zeros(2, 1))
    assert torch.isfinite(single.predictions).all()
    assert torch.count_nonzero(single.diagnostics['z_group'][:, 1:]) == 0
    checkpoint = tmp_path / 'system.pt'
    torch.save(target.state_dict(), checkpoint)
    reloaded = build((ionic, organic, hydration, density, transfer), 101).eval()
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    target.requires_grad_(False)
    reloaded.requires_grad_(False)
    primary, partner = ions(), molecule()
    with torch.inference_mode():
        torch.testing.assert_close(target(ionic.task_id, primary, torch.zeros(2, 1), partner_embedding=partner).predictions,
                                   reloaded(ionic.task_id, primary, torch.zeros(2, 1), partner_embedding=partner).predictions,
                                   rtol=0, atol=0)
    for organic_task in (False, True):
        test_partner_enters_declared_l2_branches_only(organic_task, 'group_private_l2', False,
                                                     {'group', 'private'}, split_system_experts=True)
