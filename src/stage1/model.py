from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from common.io import sha256_file
from .config import (
    STAGE1_CHECKPOINT_KIND,
    STAGE1_CHECKPOINT_VERSION,
    PretrainConfig,
    config_from_dict,
)
from .data import MultimodalBatch, PreparedCorpusDataset
from .descriptors import DescriptorSchema
from .encoders import (
    DescriptorEncoder,
    DirectedMessagePassingEncoder,
    FingerprintEncoder,
    SmilesEncoder,
)
from .fingerprints import fingerprint_families
from .fusion import (
    FusionLayout,
    FusionTransformer,
    gather_graph_tokens,
    gather_group_tokens,
    gather_smiles,
)
from .graph import (
    ATOM_CARDINALITIES,
    ATOM_FEATURE_NAMES,
    BOND_CARDINALITIES,
    BOND_FEATURE_NAMES,
)
from .tokenizer import SmilesTokenizer


@dataclass(frozen=True)
class PretrainOutput:
    loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    loss_statistics: dict[str, "LossStatistics"]
    logits: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    fused_cls: torch.Tensor


@dataclass(frozen=True)
class EncodedEntityStates:
    entity_cls: torch.Tensor
    atom_states: torch.Tensor
    atom_batch: torch.Tensor


@dataclass(frozen=True)
class LossStatistics:
    numerators: torch.Tensor
    denominators: torch.Tensor
    role_numerators: torch.Tensor
    role_denominators: torch.Tensor

    def mean(self) -> torch.Tensor:
        return (
            self.numerators
            / torch.where(
                self.denominators > 0,
                self.denominators,
                torch.ones_like(self.denominators),
            )
        ).mean()

    def role_mean(self, role: int) -> torch.Tensor:
        denominators = self.role_denominators[:, role]
        return (
            self.role_numerators[:, role]
            / torch.where(
                denominators > 0,
                denominators,
                torch.ones_like(denominators),
            )
        ).mean()


@dataclass(frozen=True)
class LoadedStage1Model:
    model: "MultimodalPretrainModel"
    config: PretrainConfig
    vocabulary: SmilesTokenizer
    artifact_hash: str


class TiedMLMHead(nn.Module):
    def __init__(self, d_model: int, embedding: nn.Embedding) -> None:
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.embedding = embedding
        self.bias = nn.Parameter(torch.zeros(embedding.num_embeddings))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(self.transform(hidden), self.embedding.weight, self.bias)


class ReconstructionTrunk(nn.Module):
    def __init__(self, d_model: int, dropout: float, kind: str) -> None:
        super().__init__()
        self.kind = kind
        if kind == "mlp":
            self.norm = nn.LayerNorm(d_model)
            self.layers = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout),
            )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.kind == "linear":
            return hidden
        return hidden + self.layers(self.norm(hidden))


def _weighted_component(
    losses: torch.Tensor,
    roles: torch.Tensor,
    role_weights: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    losses = losses.float()
    weights = role_weights[roles]
    if mask is not None:
        weights = weights * mask.to(weights.dtype)
    numerator = (losses * weights).sum()
    denominator = weights.sum()
    role_numerators = torch.stack(
        [
            (losses * weights * (roles == role)).sum()
            for role in range(3)
        ]
    )
    role_denominators = torch.stack(
        [(weights * (roles == role)).sum() for role in range(3)]
    )
    return numerator, denominator, role_numerators, role_denominators


def _statistics(
    components: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    zero_reference: torch.Tensor,
    component_count: int,
) -> LossStatistics:
    if not components:
        zero = zero_reference.float().sum() * 0.0
        return LossStatistics(
            numerators=zero.expand(component_count),
            denominators=zero.new_zeros(component_count),
            role_numerators=zero.new_zeros((component_count, 3)),
            role_denominators=zero.new_zeros((component_count, 3)),
        )
    return LossStatistics(
        numerators=torch.stack([item[0] for item in components]),
        denominators=torch.stack([item[1] for item in components]),
        role_numerators=torch.stack([item[2] for item in components]),
        role_denominators=torch.stack([item[3] for item in components]),
    )


def _masked_multitask_cross_entropy(
    logits: dict[str, torch.Tensor],
    targets: torch.Tensor,
    mask: torch.Tensor,
    feature_names: tuple[str, ...],
    element_roles: torch.Tensor,
    role_weights: torch.Tensor,
    zero_reference: torch.Tensor,
) -> LossStatistics:
    components = [
        _weighted_component(
            F.cross_entropy(
                logits[name], targets[:, column], reduction="none"
            ),
            element_roles,
            role_weights,
            mask,
        )
        for column, name in enumerate(feature_names)
    ]
    return _statistics(components, zero_reference, len(feature_names))


class MultimodalPretrainModel(nn.Module):
    def __init__(
        self,
        config: PretrainConfig,
        vocabulary: SmilesTokenizer,
        descriptor_schema: DescriptorSchema | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.descriptor_schema = descriptor_schema or DescriptorSchema.full(
            config.data.descriptor_dim, config.descriptor.token_count
        )
        model_config = config.model
        self.smiles_encoder = SmilesEncoder(
            vocab_size=len(vocabulary.tokens),
            max_length=config.data.max_smiles_tokens,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            num_layers=model_config.smiles_layers,
            feedforward_dim=model_config.feedforward_dim,
            dropout=model_config.dropout,
            gradient_checkpointing=model_config.gradient_checkpointing,
        )
        self.graph_encoder = DirectedMessagePassingEncoder(
            d_model=model_config.d_model,
            depth=model_config.graph_depth,
            dropout=model_config.dropout,
        )
        self.descriptor_encoder = DescriptorEncoder(
            schema=self.descriptor_schema,
            hidden_dim=model_config.descriptor_hidden_dim,
            blocks=model_config.descriptor_blocks,
            d_model=model_config.d_model,
            dropout=model_config.dropout,
        )
        self.fingerprint_families = fingerprint_families(config.fingerprint.kind)
        fingerprint_dimensions = {
            "morgan": config.fingerprint.morgan_bits,
            "maccs": config.fingerprint.maccs_bits,
        }
        self.fingerprint_encoder = FingerprintEncoder(
            families=self.fingerprint_families,
            dimensions=fingerprint_dimensions,
            chunk_size=config.fingerprint.chunk_size,
            hidden_dim=model_config.descriptor_hidden_dim,
            d_model=model_config.d_model,
            dropout=model_config.dropout,
        )
        self.fusion = FusionTransformer(
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            num_layers=model_config.fusion_layers,
            feedforward_dim=model_config.feedforward_dim,
            dropout=model_config.dropout,
            role_embedding=model_config.role_embedding,
            gradient_checkpointing=model_config.gradient_checkpointing,
        )
        self.smiles_head = TiedMLMHead(
            model_config.d_model, self.smiles_encoder.token_embedding
        )
        self.atom_trunk = ReconstructionTrunk(
            model_config.d_model, model_config.dropout, model_config.graph_head
        )
        self.bond_trunk = ReconstructionTrunk(
            model_config.d_model, model_config.dropout, model_config.graph_head
        )
        self.atom_heads = nn.ModuleDict(
            {
                name: nn.Linear(model_config.d_model, cardinality)
                for name, cardinality in zip(
                    ATOM_FEATURE_NAMES, ATOM_CARDINALITIES, strict=True
                )
            }
        )
        self.bond_heads = nn.ModuleDict(
            {
                name: nn.Linear(model_config.d_model, cardinality)
                for name, cardinality in zip(
                    BOND_FEATURE_NAMES, BOND_CARDINALITIES, strict=True
                )
            }
        )
        self.descriptor_heads = nn.ModuleList(
            [
                nn.Linear(model_config.d_model, len(indices))
                if indices
                else nn.Identity()
                for indices in self.descriptor_schema.group_indices
            ]
        )
        self.fingerprint_heads = nn.ModuleDict(
            {
                family: nn.Linear(model_config.d_model, config.fingerprint.chunk_size)
                for family in self.fingerprint_families
            }
        )

    def _descriptor_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        result = hidden.new_zeros(
            (hidden.shape[0], self.descriptor_schema.selected_dim)
        )
        for group_index, (indices, head) in enumerate(
            zip(
                self.descriptor_schema.group_indices,
                self.descriptor_heads,
                strict=True,
            )
        ):
            if not indices:
                continue
            logits = head(hidden[:, group_index])
            index = torch.tensor(indices, dtype=torch.long, device=hidden.device)
            result[:, index] = logits.to(result.dtype)
        return result

    def _fingerprint_logits(
        self,
        hidden: torch.Tensor,
        family_slices: dict[str, slice],
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for family in self.fingerprint_families:
            chunk_hidden = hidden[:, family_slices[family]]
            chunk_logits = self.fingerprint_heads[family](chunk_hidden)
            dimension = self.config.fingerprint.morgan_bits if family == "morgan" else self.config.fingerprint.maccs_bits
            result[family] = chunk_logits.flatten(1)[:, :dimension]
        return result

    def _encode_fused(
        self,
        batch: MultimodalBatch,
        *,
        atom_mask: torch.Tensor,
        bond_mask: torch.Tensor,
        descriptor_indicator: torch.Tensor,
        fingerprint_indicator: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, FusionLayout, dict[str, slice]]:
        smiles_tokens = self.smiles_encoder(
            batch.token_ids, batch.token_padding_mask
        )
        atom_tokens, bond_tokens = self.graph_encoder(
            batch.graphs,
            atom_mask,
            bond_mask,
        )
        descriptor_tokens = self.descriptor_encoder(
            batch.descriptors, descriptor_indicator
        )
        fingerprint_tokens, family_slices = self.fingerprint_encoder(
            batch.fingerprints,
            fingerprint_indicator,
            descriptor_tokens,
        )
        fused, layout = self.fusion(
            smiles_tokens,
            batch.token_padding_mask,
            atom_tokens,
            bond_tokens,
            descriptor_tokens,
            fingerprint_tokens,
            batch.graphs,
            batch.roles,
            batch.fusion_layout,
        )
        return fused, layout, family_slices

    def _encode_unmasked_fused(
        self, batch: MultimodalBatch
    ) -> tuple[torch.Tensor, FusionLayout]:
        if batch.masks is not None:
            raise ValueError("MultimodalPretrainModel.encode expects an unmasked batch")
        fused, layout, _ = self._encode_fused(
            batch,
            atom_mask=torch.zeros(
                batch.graphs.atom_categorical.shape[0],
                dtype=torch.bool,
                device=batch.graphs.atom_categorical.device,
            ),
            bond_mask=torch.zeros(
                batch.graphs.bond_categorical.shape[0],
                dtype=torch.bool,
                device=batch.graphs.bond_categorical.device,
            ),
            descriptor_indicator=~batch.descriptor_valid,
            fingerprint_indicator={
                name: ~valid for name, valid in batch.fingerprints.valid.items()
            },
        )
        return fused, layout

    def encode_states(self, batch: MultimodalBatch) -> EncodedEntityStates:
        """Encode complete modalities into entity and fused atom states."""
        fused, layout = self._encode_unmasked_fused(batch)
        return EncodedEntityStates(
            entity_cls=fused[:, 0],
            atom_states=gather_graph_tokens(
                fused, layout.atom_indices, batch.graphs.atom_batch
            ),
            atom_batch=batch.graphs.atom_batch,
        )

    def encode(self, batch: MultimodalBatch) -> torch.Tensor:
        """Encode complete, uncorrupted modalities into the fusion CLS state."""
        fused, _ = self._encode_unmasked_fused(batch)
        return fused[:, 0]

    def forward(self, batch: MultimodalBatch) -> PretrainOutput:
        if batch.masks is None:
            raise ValueError("MultimodalPretrainModel requires a dynamically masked batch")
        fused, layout, family_slices = self._encode_fused(
            batch,
            atom_mask=batch.masks.atom_mask,
            bond_mask=batch.masks.bond_mask,
            descriptor_indicator=batch.masks.descriptor_indicator,
            fingerprint_indicator=batch.masks.fingerprint_indicator,
        )

        fused_smiles = gather_smiles(fused, layout)
        fused_atoms = self.atom_trunk(
            gather_graph_tokens(
                fused, layout.atom_indices, batch.graphs.atom_batch
            )
        )
        fused_bonds = self.bond_trunk(
            gather_graph_tokens(
                fused, layout.bond_indices, batch.graphs.bond_batch
            )
        )
        fused_descriptors = gather_group_tokens(fused, layout.descriptor_indices)
        fused_fingerprints = gather_group_tokens(fused, layout.fingerprint_indices)

        smiles_logits = self.smiles_head(fused_smiles)
        atom_logits = {
            name: head(fused_atoms) for name, head in self.atom_heads.items()
        }
        bond_logits = {
            name: head(fused_bonds) for name, head in self.bond_heads.items()
        }
        descriptor_logits = self._descriptor_logits(fused_descriptors)
        fingerprint_logits = self._fingerprint_logits(
            fused_fingerprints, family_slices
        )

        role_weights = torch.as_tensor(
            self.config.loss.role_weights,
            dtype=torch.float32,
            device=fused.device,
        )

        smiles_mask = batch.masks.smiles_labels != -100
        token_roles = batch.roles[:, None].expand_as(smiles_mask)
        smiles_components = [
            _weighted_component(
                F.cross_entropy(
                    smiles_logits.transpose(1, 2),
                    batch.masks.smiles_labels,
                    reduction="none",
                ),
                token_roles,
                role_weights,
                smiles_mask,
            )
        ]
        smiles_statistics = _statistics(smiles_components, fused_smiles, 1)
        atom_statistics = _masked_multitask_cross_entropy(
            atom_logits,
            batch.graphs.atom_categorical,
            batch.masks.atom_mask,
            ATOM_FEATURE_NAMES,
            batch.roles[batch.graphs.atom_batch],
            role_weights,
            fused_atoms,
        )
        bond_statistics = _masked_multitask_cross_entropy(
            bond_logits,
            batch.graphs.bond_categorical,
            batch.masks.bond_mask,
            BOND_FEATURE_NAMES,
            batch.roles[batch.graphs.bond_batch],
            role_weights,
            fused_bonds,
        )
        descriptor_roles = batch.roles[:, None].expand_as(
            batch.masks.descriptor_loss_mask
        )
        descriptor_components = [
            _weighted_component(
                F.smooth_l1_loss(
                    descriptor_logits,
                    batch.descriptors,
                    reduction="none",
                ),
                descriptor_roles,
                role_weights,
                batch.masks.descriptor_loss_mask,
            )
        ]
        descriptor_statistics = _statistics(
            descriptor_components, descriptor_logits, 1
        )

        fingerprint_components = []
        for family, logits in fingerprint_logits.items():
            loss_mask = batch.masks.fingerprint_loss_mask[family]
            fingerprint_roles = batch.roles[:, None].expand_as(loss_mask)
            fingerprint_components.append(
                _weighted_component(
                    F.binary_cross_entropy_with_logits(
                        logits,
                        batch.fingerprints.values[family],
                        reduction="none",
                    ),
                    fingerprint_roles,
                    role_weights,
                    loss_mask,
                )
            )
        fingerprint_statistics = _statistics(
            fingerprint_components, fused, len(self.fingerprint_families)
        )

        loss_statistics = {
            "smiles": smiles_statistics,
            "descriptor": descriptor_statistics,
            "atom": atom_statistics,
            "bond": bond_statistics,
            "fingerprint": fingerprint_statistics,
        }
        losses = {name: stats.mean() for name, stats in loss_statistics.items()}
        weights = self.config.loss
        total = (
            weights.lambda_smiles * losses["smiles"]
            + weights.lambda_descriptor * losses["descriptor"]
            + weights.lambda_atom * losses["atom"]
            + weights.lambda_bond * losses["bond"]
            + weights.lambda_fingerprint * losses["fingerprint"]
        )
        return PretrainOutput(
            loss=total,
            losses=losses,
            loss_statistics=loss_statistics,
            logits={
                "smiles": smiles_logits,
                "atom": atom_logits,
                "bond": bond_logits,
                "descriptor": descriptor_logits,
                "fingerprint": fingerprint_logits,
            },
            fused_cls=fused[:, 0],
        )


def load_stage1_model(
    checkpoint_path: str | Path,
    artifact_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
    backbone_dropout: float | None = None,
) -> LoadedStage1Model:
    checkpoint_path = Path(checkpoint_path)
    artifact_dir = Path(artifact_dir)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        checkpoint.get("kind") != STAGE1_CHECKPOINT_KIND
        or checkpoint.get("format_version") != STAGE1_CHECKPOINT_VERSION
    ):
        raise ValueError("Stage 2 requires a Stage 1 pretraining checkpoint v2")
    config = config_from_dict(checkpoint["config"])
    artifact_hash = sha256_file(artifact_dir / "metadata.json")
    if backbone_dropout is not None:
        config = replace(
            config,
            model=replace(config.model, dropout=backbone_dropout),
        )
    dataset = PreparedCorpusDataset(
        artifact_dir,
        "train",
        config.data.shard_cache_size,
    )
    vocabulary = SmilesTokenizer.load(artifact_dir / "tokenizer.json")
    model = MultimodalPretrainModel(
        config,
        vocabulary,
        dataset.descriptor_schema,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device)
    return LoadedStage1Model(
        model=model,
        config=config,
        vocabulary=vocabulary,
        artifact_hash=artifact_hash,
    )
