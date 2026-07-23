from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from rdkit import Chem

from ilume_pretrain.config import (
    DescriptorConfig,
    FingerprintConfig,
    MaskingConfig,
    PretrainConfig,
)
from ilume_pretrain.fingerprints import calculate_fingerprints
from ilume_pretrain.graph import featurize_mol
from ilume_pretrain.tokenizer import AISVocabulary


@pytest.fixture
def tiny_config() -> PretrainConfig:
    config = PretrainConfig()
    return replace(
        config,
        masking=MaskingConfig(
            smiles_ratio=0.5,
            atom_ratio=0.5,
            bond_ratio=0.5,
            descriptor_ratio=0.5,
            fingerprint_ratio=0.25,
            smiles_dropout=0.0,
            graph_dropout=0.0,
            descriptor_dropout=0.0,
            fingerprint_dropout=0.0,
        ),
        fingerprint=FingerprintConfig(kind="both"),
        descriptor=DescriptorConfig(mode="full", token_count=8),
        model=replace(
            config.model,
            d_model=32,
            n_heads=4,
            smiles_layers=1,
            graph_depth=2,
            descriptor_hidden_dim=64,
            descriptor_blocks=1,
            fusion_layers=1,
            feedforward_dim=64,
            dropout=0.0,
        ),
    )


@pytest.fixture
def tiny_samples():
    smiles_values = ["[Na+]", "CC", "CCO"]
    vocabulary = AISVocabulary.fit(smiles_values)
    generator = torch.Generator().manual_seed(7)
    samples = []
    for index, smiles in enumerate(smiles_values):
        graph = featurize_mol(Chem.MolFromSmiles(smiles))
        values = torch.randn(217, generator=generator)
        valid = torch.ones(217, dtype=torch.bool)
        if index == 0:
            valid[-2:] = False
            values[-2:] = 0.0
        samples.append(
            {
                "sample_id": f"sample_{index}",
                "role_id": index,
                "token_ids": torch.tensor(
                    vocabulary.encode(smiles, max_length=384), dtype=torch.long
                ),
                "atom_categorical": graph.atom_categorical,
                "atom_continuous": graph.atom_continuous,
                "bond_categorical": graph.bond_categorical,
                "bond_index": graph.bond_index,
                "descriptors": values,
                "descriptor_valid": valid,
                "fingerprints": {
                    name: torch.from_numpy(value)
                    for name, value in calculate_fingerprints(
                        Chem.MolFromSmiles(smiles),
                        tiny_config_fingerprint(),
                    ).items()
                },
            }
        )
    return vocabulary, samples


def tiny_config_fingerprint() -> FingerprintConfig:
    return FingerprintConfig(kind="both")
