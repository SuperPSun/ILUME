"""Single-view ablations register public Stage1 extensions at entry points; Stage packages do not import them."""
from dataclasses import dataclass, fields

import torch
from torch import nn

from stage1.config import PretrainConfig, config_from_dict, register_config_extension
from stage1.dual_view import DualViewEncoder, DualViewPretrainModel, LearnedEntityEncoding, ResidualFusion
from stage1.encoders import DirectedMessagePassingEncoder, SmilesEncoder
from stage1.graph import ATOM_CARDINALITIES, ATOM_FEATURE_NAMES, BOND_CARDINALITIES, BOND_FEATURE_NAMES
from stage1.model import ReconstructionTrunk, TiedMLMHead


@dataclass(frozen=True)
class Stage1Ablation:
    variant: str


@dataclass(frozen=True)
class SingleViewConfig(PretrainConfig):
    ablation: Stage1Ablation | None = None

    @property
    def model_factory(self):
        return build_single_view

    @property
    def encoder_checkpoint_kind(self):
        return "ilume_stage1_single_view_encoder_v4"

    @property
    def training_checkpoint_kind(self):
        return "ilume_stage1_single_view_pretraining_v4"

    def validate(self):
        super().validate()
        if not self.is_dual_view or self.ablation is None or self.ablation.variant not in {"no_smiles", "no_graph"}:
            raise ValueError("Single-view ablation requires no_smiles or no_graph on the v4 architecture")
        removed = ("smiles", "alignment") if self.ablation.variant == "no_smiles" else ("atom", "bond", "partial_charge", "alignment")
        if any(getattr(self.loss, "lambda_" + name) != 0 for name in removed):
            raise ValueError("Single-view ablation must disable every dependent supervision")
        if any(getattr(self.loss, "lambda_" + name) <= 0 for name in ("descriptor", "unimol", "electronic")):
            raise ValueError("Single-view ablation retains all molecule-level auxiliary supervision")


def single_view_config_from_dict(raw):
    raw = dict(raw)
    spec = Stage1Ablation(**raw.pop("ablation"))
    ordinary = config_from_dict(raw)
    config = SingleViewConfig(**{field.name: getattr(ordinary, field.name) for field in fields(PretrainConfig)}, ablation=spec)
    config.validate()
    return config


class SingleViewEncoder(DualViewEncoder):
    def __init__(self, config, vocabulary, descriptor_schema=None):
        nn.Module.__init__(self)
        self.config, self.descriptor_schema = config, descriptor_schema
        recipe = config.model
        self.token_dim = recipe.d_model
        self.atom_dim = 0 if config.ablation.variant == "no_graph" else recipe.d_model
        self.entity_dim = self.learned_dim = 2 * recipe.d_model
        self.representation_kind = "single_view_" + config.ablation.variant + "_v4"
        if config.ablation.variant == "no_graph":
            self.smiles_encoder = SmilesEncoder(
                len(vocabulary.tokens), config.data.max_smiles_tokens,
                recipe.d_model, recipe.n_heads, recipe.smiles_layers,
                recipe.feedforward_dim, recipe.dropout, recipe.gradient_checkpointing,
            )
        else:
            self.graph_encoder = DirectedMessagePassingEncoder(
                recipe.d_model, recipe.graph_depth, recipe.dropout,
                message_mode=recipe.graph_message_mode, feedforward_dim=recipe.feedforward_dim,
            )
        self.fusion = ResidualFusion(self.entity_dim)

    def views(self, batch):
        zero = self.fusion.normalization.weight.new_zeros((len(batch.roles), self.token_dim))
        if self.config.ablation.variant == "no_graph":
            smiles = self.smiles_encoder(batch.token_ids, batch.token_padding_mask)
            empty = zero.new_empty((0, self.atom_dim))
            return smiles, empty, empty, smiles[:, 0], zero
        masks = batch.masks
        atoms, bonds = self.graph_encoder(
            batch.graphs,
            torch.zeros_like(batch.graphs.atom_batch, dtype=torch.bool) if masks is None else masks.atom_mask,
            torch.zeros_like(batch.graphs.bond_batch, dtype=torch.bool) if masks is None else masks.bond_mask,
        )
        pooled = atoms.new_zeros(zero.shape).index_add(0, batch.graphs.atom_batch, atoms)
        counts = torch.bincount(batch.graphs.atom_batch, minlength=len(batch.roles)).clamp_min(1)
        return zero.new_empty((len(batch.roles), 0, self.token_dim)), atoms, bonds, zero, pooled / counts[:, None]

    def encode_entity(self, batch):
        encoded = super().encode_entity(batch)
        if self.config.ablation.variant == "no_graph":
            return LearnedEntityEncoding(encoded.entity_embedding, encoded.atom_states.new_empty((len(batch.graphs.atom_batch), 0)), batch.graphs.atom_batch)
        return encoded


class SingleViewPretrainModel(SingleViewEncoder, DualViewPretrainModel):
    def __init__(self, config, vocabulary, descriptor_schema=None):
        SingleViewEncoder.__init__(self, config, vocabulary, descriptor_schema)
        width = self.token_dim
        if config.ablation.variant == "no_graph":
            self.smiles_head = TiedMLMHead(width, self.smiles_encoder.token_embedding)
        else:
            self.atom_trunk = ReconstructionTrunk(width, config.model.dropout, config.model.graph_head)
            self.bond_trunk = ReconstructionTrunk(width, config.model.dropout, config.model.graph_head)
            self.atom_heads = nn.ModuleDict({name: nn.Linear(width, size) for name, size in zip(ATOM_FEATURE_NAMES, ATOM_CARDINALITIES, strict=True)})
            self.bond_heads = nn.ModuleDict({name: nn.Linear(width, size) for name, size in zip(BOND_FEATURE_NAMES, BOND_CARDINALITIES, strict=True)})
            if config.loss.lambda_partial_charge > 0:
                self.partial_charge_head = nn.Linear(self.atom_dim, 1)
        self.descriptor_decoder = nn.Linear(self.entity_dim, 217)
        self.unimol_projector = nn.Linear(self.entity_dim, 768)
        self.electronic_head = nn.Linear(self.entity_dim, 13)

    def alignment_loss(self, h_s, h_g):
        return None

    def reconstruction_logits(self, smiles, atoms, bonds):
        if self.config.ablation.variant == "no_graph":
            return {"smiles": self.smiles_head(smiles)}
        atom_features, bond_features = self.atom_trunk(atoms), self.bond_trunk(bonds)
        return {
            "atom": {name: head(atom_features) for name, head in self.atom_heads.items()},
            "bond": {name: head(bond_features) for name, head in self.bond_heads.items()},
        }


def build_single_view(config, vocabulary, descriptor_schema=None, *, encoder_only=False):
    return (SingleViewEncoder if encoder_only else SingleViewPretrainModel)(config, vocabulary, descriptor_schema)


register_config_extension("ablation", single_view_config_from_dict)
