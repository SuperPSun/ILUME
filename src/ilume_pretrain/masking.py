from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from .config import MaskingConfig
from .data import MaskPlan, MultimodalBatch, graph_record_from_sample
from .fingerprints import FingerprintBatch
from .graph import pack_graphs
from .tokenizer import SPECIAL_TOKENS, SmilesTokenizer


SMILES_MODALITY = 0
GRAPH_MODALITY = 1
DESCRIPTOR_MODALITY = 2
FINGERPRINT_MODALITY = 3
MODALITY_COUNT = 4


def _sample_positions(
    positions: torch.Tensor,
    ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    count = positions.numel()
    if count == 0 or ratio <= 0.0:
        return positions[:0]
    selected_count = min(count, max(1, math.ceil(count * ratio)))
    permutation = torch.randperm(count, generator=generator)
    return positions[permutation[:selected_count]]


def curriculum_dropout_probability(progress: float) -> float:
    progress = min(1.0, max(0.0, progress))
    if progress <= 0.10:
        return 0.0
    if progress <= 0.60:
        return (progress - 0.10) / 0.50 * 0.05
    return 0.05 + (progress - 0.60) / 0.40 * 0.05


def _dropout_probabilities(config: MaskingConfig, progress: float) -> torch.Tensor:
    maxima = torch.tensor(
        [
            config.smiles_dropout,
            config.graph_dropout,
            config.descriptor_dropout,
            config.fingerprint_dropout,
        ],
        dtype=torch.float32,
    )
    if config.dropout_schedule == "off":
        return torch.zeros_like(maxima)
    if config.dropout_schedule == "static":
        return maxima
    scheduled = curriculum_dropout_probability(progress)
    return maxima * (scheduled / 0.10)


def sample_modality_dropout(
    batch_size: int,
    config: MaskingConfig,
    generator: torch.Generator,
    progress: float = 1.0,
    fingerprint_active: bool = True,
) -> torch.Tensor:
    probabilities = _dropout_probabilities(config, progress)
    active = [SMILES_MODALITY, GRAPH_MODALITY, DESCRIPTOR_MODALITY]
    if fingerprint_active:
        active.append(FINGERPRINT_MODALITY)
    dropped = torch.ones((batch_size, MODALITY_COUNT), dtype=torch.bool)
    draws = torch.rand((batch_size, len(active)), generator=generator)
    dropped[:, active] = draws < probabilities[active]
    for row in range(batch_size):
        if bool(dropped[row, active].all()):
            keep_index = int(
                torch.randint(0, len(active), (1,), generator=generator).item()
            )
            dropped[row, active[keep_index]] = False
    return dropped


def mask_smiles_tokens(
    token_ids: torch.Tensor,
    content_positions: torch.Tensor,
    ratio: float,
    vocabulary: SmilesTokenizer,
    generator: torch.Generator,
    drop_entire_modality: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    corrupted = token_ids.clone()
    labels = torch.full_like(token_ids, -100)
    selected = (
        content_positions
        if drop_entire_modality
        else _sample_positions(content_positions, ratio, generator)
    )
    if not selected.numel():
        return corrupted, labels
    labels[selected] = token_ids[selected]
    if drop_entire_modality:
        corrupted[selected] = vocabulary.mask_id
        return corrupted, labels

    decisions = torch.rand(selected.numel(), generator=generator)
    replace_with_mask = selected[decisions < 0.8]
    replace_with_random = selected[(decisions >= 0.8) & (decisions < 0.9)]
    corrupted[replace_with_mask] = vocabulary.mask_id
    if replace_with_random.numel():
        first_regular_token = len(SPECIAL_TOKENS)
        if len(vocabulary.tokens) > first_regular_token:
            random_ids = torch.randint(
                first_regular_token,
                len(vocabulary.tokens),
                (replace_with_random.numel(),),
                generator=generator,
            )
        else:
            random_ids = torch.full_like(replace_with_random, vocabulary.unk_id)
        corrupted[replace_with_random] = random_ids
    return corrupted, labels


@dataclass(frozen=True)
class MultimodalPacker:
    vocabulary: SmilesTokenizer

    def __call__(self, samples: Sequence[dict[str, Any]]) -> MultimodalBatch:
        if not samples:
            raise ValueError("Cannot collate an empty sample list")
        batch_size = len(samples)
        max_length = max(sample["token_ids"].numel() for sample in samples)
        token_ids = torch.full(
            (batch_size, max_length), self.vocabulary.pad_id, dtype=torch.long
        )
        token_padding_mask = torch.ones((batch_size, max_length), dtype=torch.bool)
        for row, sample in enumerate(samples):
            length = sample["token_ids"].numel()
            token_ids[row, :length] = sample["token_ids"]
            token_padding_mask[row, :length] = False

        graphs = pack_graphs([graph_record_from_sample(sample) for sample in samples])
        descriptors = torch.stack([sample["descriptors"] for sample in samples])
        descriptor_valid = torch.stack(
            [sample["descriptor_valid"] for sample in samples]
        ).bool()
        families = sorted(
            set().union(*(sample.get("fingerprints", {}).keys() for sample in samples))
        )
        fingerprint_values: dict[str, torch.Tensor] = {}
        fingerprint_valid: dict[str, torch.Tensor] = {}
        for family in families:
            dimension = next(
                sample["fingerprints"][family].numel()
                for sample in samples
                if family in sample.get("fingerprints", {})
            )
            rows: list[torch.Tensor] = []
            valid_rows: list[torch.Tensor] = []
            for sample in samples:
                if family in sample.get("fingerprints", {}):
                    rows.append(sample["fingerprints"][family].float())
                    valid_rows.append(torch.ones(dimension, dtype=torch.bool))
                else:
                    rows.append(torch.zeros(dimension, dtype=torch.float32))
                    valid_rows.append(torch.zeros(dimension, dtype=torch.bool))
            fingerprint_values[family] = torch.stack(rows)
            fingerprint_valid[family] = torch.stack(valid_rows)

        return MultimodalBatch(
            token_ids=token_ids,
            token_padding_mask=token_padding_mask,
            graphs=graphs,
            descriptors=descriptors,
            descriptor_valid=descriptor_valid,
            fingerprints=FingerprintBatch(fingerprint_values, fingerprint_valid),
            roles=torch.tensor(
                [sample["role_id"] for sample in samples], dtype=torch.long
            ),
            sample_ids=tuple(sample["sample_id"] for sample in samples),
            masks=None,
        )


@dataclass(frozen=True)
class MultimodalMasker:
    vocabulary: SmilesTokenizer
    config: MaskingConfig
    seed: int = 42

    def apply(
        self,
        batch: MultimodalBatch,
        global_step: int = 0,
        total_steps: int = 1,
        *,
        evaluation: bool = False,
    ) -> MultimodalBatch:
        if batch.masks is not None:
            raise ValueError("MultimodalMasker expects an unmasked packed batch")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        generator_seed = self.seed + (0 if evaluation else global_step)
        generator = torch.Generator().manual_seed(generator_seed)
        progress = min(1.0, max(0.0, global_step / total_steps))
        batch_size = batch.token_ids.shape[0]
        fingerprint_active = bool(batch.fingerprints.values)
        if evaluation:
            dropout_config = replace(
                self.config,
                dropout_schedule="off",
                asymmetric_enabled=False,
            )
        else:
            dropout_config = self.config
        dropped = sample_modality_dropout(
            batch_size,
            dropout_config,
            generator,
            progress=progress,
            fingerprint_active=fingerprint_active,
        )

        ratios = {
            "smiles": [self.config.smiles_ratio] * batch_size,
            "atom": [self.config.atom_ratio] * batch_size,
            "bond": [self.config.bond_ratio] * batch_size,
            "descriptor": [self.config.descriptor_ratio] * batch_size,
            "fingerprint": [self.config.fingerprint_ratio] * batch_size,
        }
        if dropout_config.asymmetric_enabled:
            active_modalities = [SMILES_MODALITY, GRAPH_MODALITY, DESCRIPTOR_MODALITY]
            if fingerprint_active:
                active_modalities.append(FINGERPRINT_MODALITY)
            for row in range(batch_size):
                if torch.rand((), generator=generator).item() >= self.config.asymmetric_probability:
                    continue
                available = [modality for modality in active_modalities if not dropped[row, modality]]
                selected = available[
                    int(torch.randint(0, len(available), (1,), generator=generator).item())
                ]
                if selected == SMILES_MODALITY:
                    ratios["smiles"][row] = self.config.asymmetric_ratio
                elif selected == GRAPH_MODALITY:
                    ratios["atom"][row] = self.config.asymmetric_ratio
                    ratios["bond"][row] = self.config.asymmetric_ratio
                elif selected == DESCRIPTOR_MODALITY:
                    ratios["descriptor"][row] = self.config.asymmetric_ratio
                else:
                    ratios["fingerprint"][row] = self.config.asymmetric_ratio

        token_ids = batch.token_ids.clone()
        smiles_labels = torch.full_like(token_ids, -100)
        for row in range(batch_size):
            length = int((~batch.token_padding_mask[row]).sum().item())
            positions = torch.arange(1, max(1, length - 1), dtype=torch.long)
            corrupted, labels = mask_smiles_tokens(
                token_ids[row],
                positions,
                ratios["smiles"][row],
                self.vocabulary,
                generator,
                bool(dropped[row, SMILES_MODALITY]),
            )
            token_ids[row] = corrupted
            smiles_labels[row] = labels

        atom_mask = torch.zeros(batch.graphs.atom_categorical.shape[0], dtype=torch.bool)
        bond_mask = torch.zeros(batch.graphs.bond_categorical.shape[0], dtype=torch.bool)
        for row, ((atom_start, atom_count), (bond_start, bond_count)) in enumerate(
            zip(batch.graphs.atom_scopes, batch.graphs.bond_scopes, strict=True)
        ):
            atom_positions = torch.arange(atom_start, atom_start + atom_count)
            bond_positions = torch.arange(bond_start, bond_start + bond_count)
            if dropped[row, GRAPH_MODALITY]:
                selected_atoms = atom_positions
                selected_bonds = bond_positions
            else:
                selected_atoms = (
                    atom_positions[:0]
                    if atom_count <= 1
                    else _sample_positions(atom_positions, ratios["atom"][row], generator)
                )
                selected_bonds = _sample_positions(
                    bond_positions, ratios["bond"][row], generator
                )
            atom_mask[selected_atoms] = True
            bond_mask[selected_bonds] = True

        descriptor_indicator = ~batch.descriptor_valid
        descriptor_loss_mask = torch.zeros_like(batch.descriptor_valid)
        for row in range(batch_size):
            valid_positions = torch.where(batch.descriptor_valid[row])[0]
            selected = (
                valid_positions
                if dropped[row, DESCRIPTOR_MODALITY]
                else _sample_positions(
                    valid_positions, ratios["descriptor"][row], generator
                )
            )
            descriptor_indicator[row, selected] = True
            descriptor_loss_mask[row, selected] = True

        fingerprint_indicator: dict[str, torch.Tensor] = {}
        fingerprint_loss_mask: dict[str, torch.Tensor] = {}
        for family, valid in batch.fingerprints.valid.items():
            indicator = ~valid
            loss_mask = torch.zeros_like(valid)
            for row in range(batch_size):
                valid_positions = torch.where(valid[row])[0]
                selected = (
                    valid_positions
                    if dropped[row, FINGERPRINT_MODALITY]
                    else _sample_positions(
                        valid_positions, ratios["fingerprint"][row], generator
                    )
                )
                indicator[row, selected] = True
                loss_mask[row, selected] = True
            fingerprint_indicator[family] = indicator
            fingerprint_loss_mask[family] = loss_mask

        return replace(
            batch,
            token_ids=token_ids,
            masks=MaskPlan(
                smiles_labels=smiles_labels,
                atom_mask=atom_mask,
                bond_mask=bond_mask,
                descriptor_indicator=descriptor_indicator,
                descriptor_loss_mask=descriptor_loss_mask,
                fingerprint_indicator=fingerprint_indicator,
                fingerprint_loss_mask=fingerprint_loss_mask,
                modality_dropped=dropped,
            ),
        )


@dataclass
class MultimodalCollator:
    vocabulary: SmilesTokenizer
    config: MaskingConfig
    seed: int = 42

    def __post_init__(self) -> None:
        self.packer = MultimodalPacker(self.vocabulary)
        self.masker = MultimodalMasker(self.vocabulary, self.config, self.seed)
        self.call_index = 0

    def __call__(self, samples: Sequence[dict[str, Any]]) -> MultimodalBatch:
        packed = self.packer(samples)
        masked = self.masker.apply(packed, self.call_index, total_steps=1)
        self.call_index += 1
        return masked
