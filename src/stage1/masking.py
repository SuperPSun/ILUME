from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from .config import MaskingConfig
from .data import (
    BatchFusionLayout,
    MaskPlan,
    MultimodalBatch,
    graph_record_from_sample,
)
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
    probabilities = _dropout_probabilities(config, progress).to(generator.device)
    active = [SMILES_MODALITY, GRAPH_MODALITY, DESCRIPTOR_MODALITY]
    if fingerprint_active:
        active.append(FINGERPRINT_MODALITY)
    device = generator.device
    dropped = torch.ones(
        (batch_size, MODALITY_COUNT), dtype=torch.bool, device=device
    )
    draws = torch.rand(
        (batch_size, len(active)), generator=generator, device=device
    )
    dropped[:, active] = draws < probabilities[active]
    all_dropped = dropped[:, active].all(dim=1)
    keep = torch.randint(
        0, len(active), (batch_size,), generator=generator, device=device
    )
    rows = torch.arange(batch_size, device=device)[all_dropped]
    active_tensor = torch.tensor(active, dtype=torch.long, device=device)
    dropped[rows, active_tensor[keep[all_dropped]]] = False
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
        smiles_lengths = (~token_padding_mask).sum(dim=1)
        atom_counts = torch.tensor(
            [count for _, count in graphs.atom_scopes], dtype=torch.long
        )
        bond_counts = torch.tensor(
            [count for _, count in graphs.bond_scopes], dtype=torch.long
        )
        atom_local_indices = torch.cat(
            [torch.arange(count) for count in atom_counts.tolist()]
        )
        bond_local_indices = torch.cat(
            [torch.arange(count) for count in bond_counts.tolist()]
        )
        core_lengths = 1 + smiles_lengths + atom_counts + bond_counts
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
            fusion_layout=BatchFusionLayout(
                smiles_lengths=smiles_lengths,
                atom_counts=atom_counts,
                bond_counts=bond_counts,
                atom_local_indices=atom_local_indices,
                bond_local_indices=bond_local_indices,
                max_core_length=int(core_lengths.max()),
                max_atom_count=int(atom_counts.max()),
                max_bond_count=int(bond_counts.max()),
            ),
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
        generator = torch.Generator(device=batch.token_ids.device).manual_seed(
            generator_seed
        )
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

        ratios = torch.tensor(
            [
                self.config.smiles_ratio,
                self.config.atom_ratio,
                self.config.bond_ratio,
                self.config.descriptor_ratio,
                self.config.fingerprint_ratio,
            ],
            dtype=torch.float32,
            device=batch.token_ids.device,
        )[None, :].expand(batch_size, -1).clone()
        if dropout_config.asymmetric_enabled:
            active_modalities = [SMILES_MODALITY, GRAPH_MODALITY, DESCRIPTOR_MODALITY]
            if fingerprint_active:
                active_modalities.append(FINGERPRINT_MODALITY)
            candidates = torch.tensor(
                active_modalities, dtype=torch.long, device=batch.token_ids.device
            )
            available = ~dropped[:, candidates]
            choice_scores = torch.rand(
                available.shape, generator=generator, device=batch.token_ids.device
            ).masked_fill(~available, -1.0)
            selected = candidates[choice_scores.argmax(dim=1)]
            asymmetric = torch.rand(
                batch_size, generator=generator, device=batch.token_ids.device
            ) < self.config.asymmetric_probability
            boosted = torch.full(
                (batch_size,),
                self.config.asymmetric_ratio,
                device=batch.token_ids.device,
            )
            for modality, columns in (
                (SMILES_MODALITY, (0,)),
                (GRAPH_MODALITY, (1, 2)),
                (DESCRIPTOR_MODALITY, (3,)),
                (FINGERPRINT_MODALITY, (4,)),
            ):
                rows = asymmetric & (selected == modality)
                for column in columns:
                    ratios[:, column] = torch.where(
                        rows, boosted, ratios[:, column]
                    )

        token_ids = batch.token_ids.clone()
        smiles_labels = torch.full_like(token_ids, -100)
        smiles_columns = torch.arange(token_ids.shape[1], device=token_ids.device)
        smiles_eligible = (
            (smiles_columns[None, :] >= 1)
            & (smiles_columns[None, :] < batch.fusion_layout.smiles_lengths[:, None] - 1)
        )
        smiles_mask = _select_padded(
            smiles_eligible,
            ratios[:, 0],
            dropped[:, SMILES_MODALITY],
            generator,
        )
        smiles_labels[smiles_mask] = token_ids[smiles_mask]
        decisions = torch.rand(
            token_ids.shape, generator=generator, device=token_ids.device
        )
        replace_with_mask = smiles_mask & (
            dropped[:, SMILES_MODALITY, None] | (decisions < 0.8)
        )
        replace_with_random = (
            smiles_mask
            & ~dropped[:, SMILES_MODALITY, None]
            & (decisions >= 0.8)
            & (decisions < 0.9)
        )
        token_ids[replace_with_mask] = self.vocabulary.mask_id
        first_regular_token = len(SPECIAL_TOKENS)
        if len(self.vocabulary.tokens) > first_regular_token:
            random_ids = torch.randint(
                first_regular_token,
                len(self.vocabulary.tokens),
                token_ids.shape,
                generator=generator,
                device=token_ids.device,
            )
        else:
            random_ids = torch.full_like(token_ids, self.vocabulary.unk_id)
        token_ids[replace_with_random] = random_ids[replace_with_random]

        atom_grid = torch.arange(
            batch.fusion_layout.max_atom_count, device=token_ids.device
        )
        atom_all = atom_grid[None, :] < batch.fusion_layout.atom_counts[:, None]
        atom_eligible = atom_all & (batch.fusion_layout.atom_counts[:, None] > 1)
        atom_padded = _select_padded(
            atom_eligible,
            ratios[:, 1],
            torch.zeros_like(dropped[:, GRAPH_MODALITY]),
            generator,
        )
        atom_padded |= dropped[:, GRAPH_MODALITY, None] & atom_all
        atom_mask = atom_padded[
            batch.graphs.atom_batch, batch.fusion_layout.atom_local_indices
        ]

        bond_grid = torch.arange(
            batch.fusion_layout.max_bond_count, device=token_ids.device
        )
        bond_eligible = bond_grid[None, :] < batch.fusion_layout.bond_counts[:, None]
        bond_padded = _select_padded(
            bond_eligible,
            ratios[:, 2],
            dropped[:, GRAPH_MODALITY],
            generator,
        )
        bond_mask = bond_padded[
            batch.graphs.bond_batch, batch.fusion_layout.bond_local_indices
        ]

        descriptor_indicator = ~batch.descriptor_valid
        descriptor_loss_mask = _select_padded(
            batch.descriptor_valid,
            ratios[:, 3],
            dropped[:, DESCRIPTOR_MODALITY],
            generator,
        )
        descriptor_indicator |= descriptor_loss_mask

        fingerprint_indicator: dict[str, torch.Tensor] = {}
        fingerprint_loss_mask: dict[str, torch.Tensor] = {}
        for family, valid in batch.fingerprints.valid.items():
            loss_mask = _select_padded(
                valid,
                ratios[:, 4],
                dropped[:, FINGERPRINT_MODALITY],
                generator,
            )
            indicator = ~valid | loss_mask
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


def _select_padded(
    eligible: torch.Tensor,
    ratios: torch.Tensor,
    drop_entire: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    counts = eligible.sum(dim=1)
    selected_counts = torch.ceil(counts * ratios).to(torch.long)
    selected_counts = torch.where(
        (counts > 0) & (ratios > 0), selected_counts.clamp_min(1), 0
    )
    selected_counts = torch.where(drop_entire, counts, selected_counts)
    scores = torch.rand(
        eligible.shape, generator=generator, device=eligible.device
    ).masked_fill(~eligible, 2.0)
    order = scores.argsort(dim=1)
    ranks = torch.empty_like(order)
    ranks.scatter_(
        1,
        order,
        torch.arange(eligible.shape[1], device=eligible.device)[None, :].expand_as(
            order
        ),
    )
    return eligible & (ranks < selected_counts[:, None])


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
