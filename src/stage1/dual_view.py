from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .encoders import DirectedMessagePassingEncoder, SmilesEncoder
from .graph import ATOM_CARDINALITIES, ATOM_FEATURE_NAMES, BOND_CARDINALITIES, BOND_FEATURE_NAMES
from .model import LossStatistics, PretrainOutput, ReconstructionTrunk, TiedMLMHead


@dataclass(frozen=True)
class LearnedEntityEncoding:
    entity_embedding: torch.Tensor
    atom_states: torch.Tensor
    atom_batch: torch.Tensor


class ResidualFusion(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.normalization = nn.LayerNorm(width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.normalization(values + self.layers(values))


class DualViewEncoder(nn.Module):
    """Structure-only encoder. Descriptors and role IDs never enter this forward."""

    def __init__(self, config, vocabulary, descriptor_schema=None) -> None:
        super().__init__()
        self.config = config
        self.descriptor_schema = descriptor_schema
        recipe = config.model
        self.token_dim = self.atom_dim = recipe.d_model
        self.entity_dim = self.learned_dim = 2 * recipe.d_model
        self.representation_kind = "dual_view_learned_v4"
        self.smiles_encoder = SmilesEncoder(
            len(vocabulary.tokens), config.data.max_smiles_tokens,
            recipe.d_model, recipe.n_heads, recipe.smiles_layers,
            recipe.feedforward_dim, recipe.dropout, recipe.gradient_checkpointing,
        )
        self.graph_encoder = DirectedMessagePassingEncoder(
            recipe.d_model, recipe.graph_depth, recipe.dropout,
            message_mode=recipe.graph_message_mode, feedforward_dim=recipe.feedforward_dim,
        )
        self.fusion = ResidualFusion(self.entity_dim)

    def views(self, batch):
        masks = batch.masks
        atoms, bonds = self.graph_encoder(
            batch.graphs,
            torch.zeros_like(batch.graphs.atom_batch, dtype=torch.bool) if masks is None else masks.atom_mask,
            torch.zeros_like(batch.graphs.bond_batch, dtype=torch.bool) if masks is None else masks.bond_mask,
        )
        smiles = self.smiles_encoder(batch.token_ids, batch.token_padding_mask)
        pooled = atoms.new_zeros((len(batch.roles), self.atom_dim)).index_add(0, batch.graphs.atom_batch, atoms)
        counts = torch.bincount(batch.graphs.atom_batch, minlength=len(batch.roles)).clamp_min(1)
        return smiles, atoms, bonds, smiles[:, 0], pooled / counts[:, None]

    def encode_entity(self, batch) -> LearnedEntityEncoding:
        if batch.masks is not None:
            raise ValueError("Dual-view encoding requires an unmasked batch")
        _, atoms, _, smiles, graph = self.views(batch)
        return LearnedEntityEncoding(self.fusion(torch.cat((smiles, graph), -1)), atoms, batch.graphs.atom_batch)

    def encode(self, batch):
        return self.encode_entity(batch).entity_embedding


def molecule_loss_statistics(values, valid, roles, role_weights) -> LossStatistics:
    """One normalized scalar per molecule; weight roles, never sampling."""
    weights = role_weights[roles] * valid.float()
    numerator = (values.float() * weights).sum()
    denominator = weights.sum()
    return LossStatistics(
        numerator.reshape(1), denominator.reshape(1),
        torch.stack([(values.float() * weights * (roles == role)).sum() for role in range(3)])[None],
        torch.stack([(weights * (roles == role)).sum() for role in range(3)])[None],
    )


def reduce_molecule_elements(values, mask, molecule_ids, batch_size):
    # Mean over observed targets/features/elements within each molecule first.
    values = values.float()
    if values.ndim > 1:
        values = (values * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        mask = mask.any(-1)
    sums = values.new_zeros(batch_size).index_add(0, molecule_ids, values * mask)
    counts = values.new_zeros(batch_size).index_add(0, molecule_ids, mask.float())
    return sums / counts.clamp_min(1), counts > 0


def alignment_mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))


class DualViewPretrainModel(DualViewEncoder):
    def __init__(self, config, vocabulary, descriptor_schema=None):
        super().__init__(config, vocabulary, descriptor_schema)
        width = self.token_dim
        self.smiles_head = TiedMLMHead(width, self.smiles_encoder.token_embedding)
        self.atom_trunk = ReconstructionTrunk(width, config.model.dropout, config.model.graph_head)
        self.bond_trunk = ReconstructionTrunk(width, config.model.dropout, config.model.graph_head)
        self.atom_heads = nn.ModuleDict({name: nn.Linear(width, size) for name, size in zip(ATOM_FEATURE_NAMES, ATOM_CARDINALITIES, strict=True)})
        self.bond_heads = nn.ModuleDict({name: nn.Linear(width, size) for name, size in zip(BOND_FEATURE_NAMES, BOND_CARDINALITIES, strict=True)})
        self.smiles_projector = alignment_mlp(width, width, 256)
        self.graph_projector = alignment_mlp(width, width, 256)
        self.smiles_predictor = alignment_mlp(256, 128, 256)
        self.graph_predictor = alignment_mlp(256, 128, 256)
        self.descriptor_decoder = nn.Linear(self.entity_dim, 217)
        self.unimol_projector = nn.Linear(self.entity_dim, 768)
        self.electronic_head = nn.Linear(self.entity_dim, 13)
        if config.loss.lambda_partial_charge > 0:
            self.partial_charge_head = nn.Linear(self.atom_dim, 1)

    def forward(self, batch) -> PretrainOutput:
        if batch.masks is None:
            raise ValueError("Dual-view training requires masked inputs")
        smiles, atoms, bonds, h_s, h_g = self.views(batch)
        dropped = batch.masks.modality_dropped[:, :2]
        z = self.fusion(torch.cat((h_s.masked_fill(dropped[:, :1], 0), h_g.masked_fill(dropped[:, 1:2], 0)), -1))
        p_s, p_g = self.smiles_projector(h_s), self.graph_projector(h_g)
        alignment = -F.cosine_similarity(self.smiles_predictor(p_s).float(), p_g.detach().float()) - F.cosine_similarity(self.graph_predictor(p_g).float(), p_s.detach().float())
        atom_features, bond_features = self.atom_trunk(atoms), self.bond_trunk(bonds)
        logits = {
            "smiles": self.smiles_head(smiles),
            "atom": {name: head(atom_features) for name, head in self.atom_heads.items()},
            "bond": {name: head(bond_features) for name, head in self.bond_heads.items()},
            "descriptor": self.descriptor_decoder(z),
            "unimol": self.unimol_projector(z),
            "electronic": self.electronic_head(z),
        }
        size = len(batch.roles)
        ids = torch.arange(size, device=z.device)
        weights = torch.as_tensor(self.config.loss.role_weights, device=z.device)
        statistics = {}

        def record(name, values, mask, molecule_ids):
            reduced, valid = reduce_molecule_elements(values, mask, molecule_ids, size)
            statistics[name] = molecule_loss_statistics(reduced, valid, batch.roles, weights)

        labels = batch.masks.smiles_labels
        record("smiles", F.cross_entropy(logits["smiles"].transpose(1, 2).float(), labels, reduction="none").flatten(), (labels != -100).flatten(), ids[:, None].expand_as(labels).flatten())
        for name, targets, mask, feature_names, molecule_ids in (
            ("atom", batch.graphs.atom_categorical, batch.masks.atom_mask, ATOM_FEATURE_NAMES, batch.graphs.atom_batch),
            ("bond", batch.graphs.bond_categorical, batch.masks.bond_mask, BOND_FEATURE_NAMES, batch.graphs.bond_batch),
        ):
            losses = torch.stack([F.cross_entropy(logits[name][feature].float(), targets[:, column], reduction="none") for column, feature in enumerate(feature_names)], -1)
            record(name, losses, mask[:, None].expand_as(losses), molecule_ids)
        record("descriptor", F.mse_loss(logits["descriptor"].float(), batch.descriptors.float(), reduction="none"), batch.descriptor_valid, ids)
        statistics["alignment"] = molecule_loss_statistics(alignment, torch.ones(size, device=z.device, dtype=torch.bool), batch.roles, weights)
        aux = batch.auxiliary_targets
        if aux is None:
            raise ValueError("Dual-view training requires audited auxiliary targets")
        record("unimol", 1 - F.cosine_similarity(logits["unimol"].float(), aux["unimol"].float()), aux["unimol_valid"], ids)
        record("electronic", F.smooth_l1_loss(logits["electronic"].float(), aux["electronic"].float(), reduction="none"), aux["electronic_valid"], ids)
        if self.config.loss.lambda_partial_charge > 0:
            logits["partial_charge"] = self.partial_charge_head(atoms).squeeze(-1)
            values = F.smooth_l1_loss(logits["partial_charge"][aux["charge_atom_indices"]].float(), aux["partial_charge"].float(), reduction="none")
            roles = aux["charge_observation_roles"]
            reduced, valid = reduce_molecule_elements(values, aux["partial_charge_valid"], aux["charge_observation_ids"], len(roles))
            statistics["partial_charge"] = molecule_loss_statistics(reduced, valid, roles, weights)
        losses = {name: stats.mean() for name, stats in statistics.items()}
        coefficients = {"smiles": self.config.loss.lambda_smiles, "atom": self.config.loss.lambda_atom, "bond": self.config.loss.lambda_bond, "descriptor": self.config.loss.lambda_descriptor, "alignment": self.config.loss.lambda_alignment, "unimol": self.config.loss.lambda_unimol, "electronic": self.config.loss.lambda_electronic}
        if self.config.loss.lambda_partial_charge > 0:
            coefficients["partial_charge"] = self.config.loss.lambda_partial_charge
        return PretrainOutput(sum(coefficients[name] * value for name, value in losses.items()), losses, statistics, logits, z)
