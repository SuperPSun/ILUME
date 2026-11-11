"""Entity-preserving experts; only L2 receives physical condition tokens."""
import torch
from torch import nn

from common.condition_coordinates import CONDITION_CONTRACT, CONDITION_SCALES, physical_conditions
from common.entity_inputs import EntityInputs
from common.training import seeded_initialization
from .data import sanitize_task

TOKEN_ARCHITECTURE = "entity_token_attention_v1"
TOKEN_CONTRACT = {
    "architecture_kind": TOKEN_ARCHITECTURE,
    "slots": ["primary_or_solvent", "anion_if_il", "solute_or_partner"],
    "slot_contract": "solvent_first_three_entity_tokens_v1",
    "condition_contract": CONDITION_CONTRACT,
    "l1": "within_object_attention_without_conditions",
    "l1_gate": "object_local_structural_route_v1",
    "l2": "all_entity_condition_self_attention",
    "fusion": "expert_weighted_per_slot_then_concat_mask_v1",
    "group_residual": "attention_token_residual_only",
    "residual": "per_entity_attention_and_l2_residual",
    "token_count": 3,
    "task_gate": "structural_three_slot_concat_v1",
}


def token_model(config):
    return config.expert_architecture == TOKEN_ARCHITECTURE


def attention_contract(config, entity_dim=None):
    contract = {**TOKEN_CONTRACT, "heads": config.attention_heads, "layers": config.attention_layers,
                **({"packed_dim": 3 * entity_dim + 12} if entity_dim is not None else {})}
    if config.partner_injection == "group_l2":
        contract.update(partner_injection="group_l2", l1="primary_only_attention_without_conditions",
                        l1_gate="primary_only_structural_route_v1",
                        l2="group_partner_attention_global_private_primary_only_v1",
                        partner_projection="paired_group_l1_projection_role_position_norm_v1")
    return contract


class TokenStack(nn.Module):
    def __init__(self, width, hidden, heads, layers, dropout, activation):
        super().__init__()
        act = nn.SiLU() if activation == "silu" else nn.GELU()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, hidden, dropout,
                activation=act, batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.normalization = nn.LayerNorm(width)

    def forward(self, values, mask):
        values = values.masked_fill(~mask[..., None], 0)
        for block in self.blocks:
            values = block(values, src_key_padding_mask=~mask)
            values = values.masked_fill(~mask[..., None], 0)
        return self.normalization(values).masked_fill(~mask[..., None], 0)


class EntityAttentionExpert(nn.Module):
    def __init__(self, entity_dim, width, *, hidden, heads, layers, dropout, activation):
        super().__init__()
        self.projection = nn.Linear(entity_dim, width)
        self.roles = nn.Embedding(3, width)
        self.positions = nn.Embedding(3, width)
        self.stack = TokenStack(width, hidden, heads, layers, dropout, activation)
        nn.init.normal_(self.roles.weight, std=0.02)
        nn.init.normal_(self.positions.weight, std=0.02)

    def partner_token(self, values, roles):
        return self.projection(values[:, 2:]) + self.roles(roles[:, 2:]) + self.positions.weight[None, 2:]

    def forward(self, values, roles, mask):
        tokens = self.projection(values) + self.roles(roles) + self.positions.weight[None]
        # Primary (slots 0/1) and partner (slot 2) remain separate in L1.
        primary = self.stack(tokens[:, :2], mask[:, :2])
        # A dummy valid token avoids all-masked attention for an absent partner.
        partner = self.stack(tokens[:, 2:], torch.ones_like(mask[:, 2:]))
        return torch.cat((primary, partner), 1).masked_fill(~mask[..., None], 0)


class ConditionTokens(nn.Module):
    def __init__(self, columns, width):
        super().__init__()
        self.columns = tuple(columns)
        self.projectors = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, width)) for name in columns})
        self.types = nn.ParameterDict({name: nn.Parameter(torch.empty(width)) for name in columns})
        for parameter in self.types.values():
            nn.init.normal_(parameter, std=0.02)
        self.width = width

    def forward(self, values, mask):
        result = []
        for i, name in enumerate(CONDITION_SCALES):
            if name in self.projectors:
                value = self.projectors[name](values[:, i:i+1]) + self.types[name]
            else:
                if mask[:, i].any():
                    raise ValueError(f"Condition encoder lacks declared type: {name}")
                value = values.new_zeros((len(values), self.width))
            result.append(value)
        return torch.stack(result, 1).masked_fill(~mask[..., None], 0)


class ConditionalAttentionExpert(nn.Module):
    def __init__(self, width, **kwargs):
        super().__init__()
        self.stack = TokenStack(width, **kwargs)

    def forward(self, entities, mask, conditions, condition_mask):
        tokens = torch.cat((entities, conditions), 1)
        active = torch.cat((mask, condition_mask), 1)
        return self.stack(tokens, active)[:, :3].masked_fill(~mask[..., None], 0)


class ConcatenationTower(nn.Module):
    def __init__(self, width, hidden, dropout, activation):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3 * width + 3, hidden),
            nn.SiLU() if activation == "silu" else nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, values):
        return self.layers(values).squeeze(-1)


def initialize_token_modules(model, initialization_seed):
    # Imports remain within Stage3 and avoid a module-import cycle.
    from .model import GLOBAL, group_owner, private_owner
    config, width = model.model_config, model.d_model
    if width % config.attention_heads:
        raise ValueError("Token attention width must be divisible by head count")
    model.condition_statistics = {}
    model.global_conditions = nn.ModuleDict()
    model.group_conditions = nn.ModuleDict()
    model.private_conditions = nn.ModuleDict()
    model.l1_group_experts = nn.ModuleDict()
    model.l1_group_gates = nn.ModuleDict()
    model.l1_group_normalizations = nn.ModuleDict()
    model.l2_group_experts = nn.ModuleDict()
    model.interactions = nn.ModuleDict()
    model.private_experts = nn.ModuleDict()
    model.task_gates = nn.ModuleDict()
    model.condition_films = nn.ModuleDict()
    model.task_normalizations = nn.ModuleDict()
    model.towers = nn.ModuleDict()
    packed = 3 * model.entity_input_contract["entity_dim"] + 12

    def arguments(ratio, dropout):
        return dict(hidden=max(1, round(width * ratio)), heads=config.attention_heads,
                    layers=config.attention_layers, dropout=dropout, activation=config.activation)

    with seeded_initialization(initialization_seed, "GLOBAL"):
        args = arguments(config.expert_hidden_ratio, config.dropout)
        model.l1_global_experts = nn.ModuleList([EntityAttentionExpert(
            model.entity_input_contract["entity_dim"], width, **args) for _ in range(config.global_experts)])
        model.l2_global_experts = nn.ModuleList([ConditionalAttentionExpert(width, **args)
                                               for _ in range(config.global_experts)])
        model.l1_global_gate = nn.Linear(packed, config.global_experts) if config.global_experts else None
        if config.global_experts:
            model.global_conditions["tokens"] = ConditionTokens(CONDITION_SCALES, width)
        owned = [model.l1_global_experts, model.l2_global_experts, model.global_conditions]
        if model.l1_global_gate is not None:
            owned.append(model.l1_global_gate)
        model._own_modules(GLOBAL, *owned)
    for group in model.groups:
        gc = model.group_configs.get(group)
        count = config.group_experts if gc is None or gc.experts is None else gc.experts
        ratio = config.expert_hidden_ratio if gc is None or gc.expert_hidden_ratio is None else gc.expert_hidden_ratio
        with seeded_initialization(initialization_seed, f"GROUP:{group}"):
            args = arguments(ratio, config.dropout)
            model.l1_group_experts[group] = nn.ModuleList([EntityAttentionExpert(
                model.entity_input_contract["entity_dim"], width, **args) for _ in range(count)])
            model.l1_group_gates[group] = nn.Linear(packed, count)
            model.l1_group_normalizations[group] = nn.LayerNorm(width)
            model.l2_group_experts[group] = nn.ModuleList([ConditionalAttentionExpert(width, **args) for _ in range(count)])
            model.group_conditions[group] = ConditionTokens(CONDITION_SCALES, width)
            model._own_modules(group_owner(group), model.l1_group_experts[group],
                model.l1_group_gates[group], model.l1_group_normalizations[group],
                model.l2_group_experts[group], model.group_conditions[group])
    for task, spec in model.task_specs.items():
        if not spec.enabled:
            continue
        key = sanitize_task(task)
        recipe = model.task_private_recipes.get(task)
        overrides = model.task_configs[task].model_overrides if task in model.task_configs else {}
        count = int(overrides.get("private_experts", config.private_experts))
        ratio = recipe.private_hidden_ratio if recipe else overrides.get("private_hidden_ratio", config.expert_hidden_ratio)
        tower = recipe.tower_hidden_ratio if recipe else overrides.get("tower_hidden_ratio", config.tower_hidden_ratio)
        dropout = recipe.private_dropout if recipe else overrides.get("private_dropout", config.dropout)
        group = model.group_configs.get(spec.meta_group)
        group_count = config.group_experts if group is None or group.experts is None else group.experts
        with seeded_initialization(initialization_seed, f"PRIVATE:{task}"):
            model.private_experts[key] = nn.ModuleList([
                ConditionalAttentionExpert(width, **arguments(ratio, dropout)) for _ in range(count)])
            model.task_gates[key] = nn.Linear(6 * width + 3, config.global_experts + group_count + count)
            model.task_normalizations[key] = nn.LayerNorm(width) if config.l2_residual else nn.Identity()
            model.towers[key] = ConcatenationTower(width, max(1, round(width * tower)), dropout, config.activation)
            model.private_conditions[key] = ConditionTokens(spec.condition_columns, width)
            model._own_modules(private_owner(task), model.private_experts[key], model.task_gates[key],
                model.task_normalizations[key], model.towers[key], model.private_conditions[key])
    model._validate_ownership()


def entity_sequence(primary, partner, spec, entity_dim):
    if not isinstance(primary, EntityInputs):
        raise ValueError("Token HoME requires frozen entity inputs")
    primary.pack(entity_dim)
    if not torch.equal(primary.mask[:, 1], torch.full_like(primary.mask[:, 1], len(spec.primary_slots) == 2)):
        raise ValueError("Token HoME primary entity count differs from task topology")
    if (partner is not None) != (spec.partner_mode == "interaction"):
        raise ValueError("Token HoME partner topology mismatch")
    if partner is not None:
        partner.pack(entity_dim)
        if partner.mask[:, 1].any():
            raise ValueError("Token HoME partner must be a single entity")
    # Existing prepared objects remain unchanged; the new model owns slot reordering.
    if partner is not None and spec.primary_slots == ("solute",):
        first, extra = partner, primary
    else:
        first, extra = primary, partner
    extra_values = first.values.new_zeros((len(first.values), 1, entity_dim)) if extra is None else extra.values[:, :1]
    extra_roles = first.roles.new_zeros((len(first.values), 1)) if extra is None else extra.roles[:, :1]
    extra_mask = torch.zeros_like(first.mask[:, :1]) if extra is None else extra.mask[:, :1]
    mask = torch.cat((first.mask, extra_mask), 1)
    values = torch.cat((first.values, extra_values), 1).masked_fill(~mask[..., None], 0)
    roles = torch.cat((first.roles, extra_roles), 1)
    # Route each object independently too; a system-wide gate would let a partner
    # change the primary tokens before L2 even with object-local attention.
    object_mask = mask[:, None] & mask.new_tensor([[True, True, False], [False, False, True]])[None]
    packed = torch.cat((values[:, None].masked_fill(~object_mask[..., None], 0).flatten(2),
        torch.nn.functional.one_hot(roles, 3)[:, None].to(values.dtype)
            .masked_fill(~object_mask[..., None], 0).flatten(2),
        object_mask.to(values.dtype)), -1)
    return values, roles, mask, packed


def l1_token_mixture(model, group, values, roles, mask, packed):
    experts = model.l1_global_experts if group is None else model.l1_group_experts[group]
    gate = model.l1_global_gate if group is None else model.l1_group_gates[group]
    if gate is None:
        return values.new_zeros((len(values), 3, model.d_model)), values.new_empty((len(values), 3, 0))
    object_weights = gate(packed).softmax(-1)
    weights = object_weights[:, [0, 0, 1]].masked_fill(~mask[..., None], 0)
    outputs = torch.stack([expert(values, roles, mask) for expert in experts], 1)
    return (weights.transpose(1, 2)[..., None] * outputs).sum(1), weights


def forward_tokens(model, task, primary, conditions, partner):
    from .model import Stage3ForwardOutput
    spec = model.task_specs.get(task)
    if spec is None or not spec.enabled:
        raise ValueError(f"Inactive Stage3 token task: {task}")
    values, roles, mask, packed = entity_sequence(primary, partner, spec, model.entity_input_contract["entity_dim"])
    key, group = sanitize_task(task), spec.meta_group
    if task not in model.condition_statistics:
        raise ValueError(f"Missing token HoME condition statistics: {task}")
    physical, present = physical_conditions(conditions, spec.condition_columns, model.condition_statistics[task])

    group_partner = model.model_config.partner_injection == "group_l2"
    l1_values, l1_roles, l1_mask = values, roles, mask
    if group_partner:
        l1_mask = mask.clone()
        l1_mask[:, 2] = False
        l1_values = values.masked_fill(~l1_mask[..., None], 0)
        l1_roles = roles.masked_fill(~l1_mask, 0)
        packed = packed.clone()
        packed[:, 1] = 0
    global_values, global_weights = model.l1_token_values(None, l1_values, l1_roles, l1_mask, packed)
    local, group_weights = model.l1_token_values(group, l1_values, l1_roles, l1_mask, packed)
    local = model.l1_group_normalizations[group](local).masked_fill(~mask[..., None], 0)
    if group_partner:
        local = local.masked_fill(~l1_mask[..., None], 0)

    def candidates(experts, entities, encoder, *, with_partner=False):
        if not experts:
            return entities.new_empty((len(entities), 0, 3, model.d_model))
        condition_tokens = encoder(physical, present)
        active = mask if with_partner or not group_partner else l1_mask
        outputs = []
        for index, expert in enumerate(experts):
            current = entities
            if with_partner:
                partner_tokens = model.l1_group_experts[group][index].partner_token(values, roles)
                partner_tokens = model.l1_group_normalizations[group](partner_tokens)
                current = torch.cat((entities[:, :2], partner_tokens), 1).masked_fill(~mask[..., None], 0)
            outputs.append(expert(current, active, condition_tokens, present))
        return torch.stack(outputs, 1) if outputs else entities.new_empty((len(entities), 0, 3, model.d_model))

    global_outputs = (candidates(model.l2_global_experts, global_values, model.global_conditions["tokens"])
                      if model.l2_global_experts else local.new_empty((len(local), 0, 3, model.d_model)))
    group_outputs = candidates(model.l2_group_experts[group], local, model.group_conditions[group], with_partner=group_partner)
    private_outputs = candidates(model.private_experts[key], local, model.private_conditions[key])
    weights = model.task_gate_weights(key, torch.cat((global_values.flatten(1), local.flatten(1), mask.to(local.dtype)), -1))
    all_outputs = torch.cat((global_outputs, group_outputs, private_outputs), 1)
    mixed = (weights[:, :, None, None] * all_outputs).sum(1)
    representation = model.task_normalizations[key](local + mixed) if model.model_config.l2_residual else mixed
    representation = representation.masked_fill(~mask[..., None], 0)
    predictions = model.towers[key](torch.cat((representation.flatten(1), mask.to(representation.dtype)), -1))
    return Stage3ForwardOutput(predictions, {
        "z_global": global_values, "z_group": local, "l1_global_gate": global_weights,
        "l1_group_gate": group_weights, "task_gate": weights,
        "l2_global_candidates": global_outputs, "l2_group_candidates": group_outputs,
        "l2_private_candidates": private_outputs,
    })
