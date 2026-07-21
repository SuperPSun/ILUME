from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .config import MaskingConfig
from .data import (
    MaskPlan,
    MultimodalBatch,
    graph_record_from_sample,
)
from .graph import pack_graphs
from .tokenizer import AISVocabulary, SPECIAL_TOKENS


SMILES_MODALITY = 0
GRAPH_MODALITY = 1
DESCRIPTOR_MODALITY = 2


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


def sample_modality_dropout(
    batch_size: int,
    config: MaskingConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    probabilities = torch.tensor(
        [
            config.smiles_dropout,
            config.graph_dropout,
            config.descriptor_dropout,
        ],
        dtype=torch.float32,
    )
    dropped = torch.rand((batch_size, 3), generator=generator) < probabilities
    for row in range(batch_size):
        if bool(dropped[row].all()):
            keep = int(torch.randint(0, 3, (1,), generator=generator).item())
            dropped[row, keep] = False
    return dropped


def mask_smiles_tokens(
    token_ids: torch.Tensor,
    content_positions: torch.Tensor,
    ratio: float,
    vocabulary: AISVocabulary,
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


@dataclass
class MultimodalCollator:
    vocabulary: AISVocabulary
    config: MaskingConfig
    seed: int = 42

    def __post_init__(self) -> None:
        self.generator = torch.Generator().manual_seed(self.seed)

    def __call__(self, samples: Sequence[dict[str, Any]]) -> MultimodalBatch:
        if not samples:
            raise ValueError("Cannot collate an empty sample list")
        batch_size = len(samples)
        max_length = max(sample["token_ids"].numel() for sample in samples)
        token_ids = torch.full(
            (batch_size, max_length),
            self.vocabulary.pad_id,
            dtype=torch.long,
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
        roles = torch.tensor([sample["role_id"] for sample in samples], dtype=torch.long)
        dropped = sample_modality_dropout(batch_size, self.config, self.generator)

        smiles_ratios = [self.config.smiles_ratio] * batch_size
        atom_ratios = [self.config.atom_ratio] * batch_size
        bond_ratios = [self.config.bond_ratio] * batch_size
        descriptor_ratios = [self.config.descriptor_ratio] * batch_size
        if self.config.asymmetric_enabled:
            for row in range(batch_size):
                trigger = torch.rand((), generator=self.generator).item()
                if trigger >= self.config.asymmetric_probability:
                    continue
                available = torch.where(~dropped[row])[0]
                selected = available[
                    torch.randint(
                        0, available.numel(), (1,), generator=self.generator
                    )
                ].item()
                if selected == SMILES_MODALITY:
                    smiles_ratios[row] = self.config.asymmetric_ratio
                elif selected == GRAPH_MODALITY:
                    atom_ratios[row] = self.config.asymmetric_ratio
                    bond_ratios[row] = self.config.asymmetric_ratio
                else:
                    descriptor_ratios[row] = self.config.asymmetric_ratio

        smiles_labels = torch.full_like(token_ids, -100)
        for row, sample in enumerate(samples):
            length = sample["token_ids"].numel()
            positions = torch.arange(1, max(1, length - 1), dtype=torch.long)
            corrupted, labels = mask_smiles_tokens(
                token_ids[row],
                positions,
                smiles_ratios[row],
                self.vocabulary,
                self.generator,
                bool(dropped[row, SMILES_MODALITY]),
            )
            token_ids[row] = corrupted
            smiles_labels[row] = labels

        atom_mask = torch.zeros(graphs.atom_categorical.shape[0], dtype=torch.bool)
        bond_mask = torch.zeros(graphs.bond_categorical.shape[0], dtype=torch.bool)
        for row, ((atom_start, atom_count), (bond_start, bond_count)) in enumerate(
            zip(graphs.atom_scopes, graphs.bond_scopes, strict=True)
        ):
            atom_positions = torch.arange(
                atom_start, atom_start + atom_count, dtype=torch.long
            )
            bond_positions = torch.arange(
                bond_start, bond_start + bond_count, dtype=torch.long
            )
            if dropped[row, GRAPH_MODALITY]:
                selected_atoms = atom_positions
                selected_bonds = bond_positions
            else:
                selected_atoms = _sample_positions(
                    atom_positions, atom_ratios[row], self.generator
                )
                selected_bonds = _sample_positions(
                    bond_positions, bond_ratios[row], self.generator
                )
            atom_mask[selected_atoms] = True
            bond_mask[selected_bonds] = True

        descriptor_indicator = ~descriptor_valid
        descriptor_loss_mask = torch.zeros_like(descriptor_valid)
        for row in range(batch_size):
            valid_positions = torch.where(descriptor_valid[row])[0]
            if dropped[row, DESCRIPTOR_MODALITY]:
                selected = valid_positions
            else:
                selected = _sample_positions(
                    valid_positions, descriptor_ratios[row], self.generator
                )
            descriptor_indicator[row, selected] = True
            descriptor_loss_mask[row, selected] = True

        return MultimodalBatch(
            token_ids=token_ids,
            token_padding_mask=token_padding_mask,
            graphs=graphs,
            descriptors=descriptors,
            descriptor_valid=descriptor_valid,
            roles=roles,
            sample_ids=tuple(sample["sample_id"] for sample in samples),
            masks=MaskPlan(
                smiles_labels=smiles_labels,
                atom_mask=atom_mask,
                bond_mask=bond_mask,
                descriptor_indicator=descriptor_indicator,
                descriptor_loss_mask=descriptor_loss_mask,
                modality_dropped=dropped,
            ),
        )
