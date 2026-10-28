from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from rdkit import Chem

from stage1.config import (
    DescriptorConfig,
    FingerprintConfig,
    MaskingConfig,
    PretrainConfig,
)
from stage1.fingerprints import calculate_fingerprints
from stage1.graph import featurize_mol
from stage1.tokenizer import SmilesTokenizer


@pytest.fixture
def chemberta_snapshot_factory(tmp_path, monkeypatch):
    """Small genuine RoBERTa checkpoints with explicit test-only source hashes."""
    import json
    import importlib.util
    if importlib.util.find_spec("transformers") is None:
        return None
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import RobertaConfig, RobertaModel
    from ablations.entity_home import chemberta
    from common.frozen_encoder import FrozenEncoderConfig
    from common.io import sha256_file

    def create(variant):
        root = tmp_path / variant
        root.mkdir()
        vocabulary = {f"[unused{i}]": i for i in range(32)}
        vocabulary.update({"[PAD]": 0, "[UNK]": 11, "[CLS]": 12, "[SEP]": 13,
                           "C": 15, "CC": 16, "[Na+]": 17, "[Cl-]": 18})
        vocabulary = {k: v for k, v in vocabulary.items() if not (k.startswith("[unused") and v in {0, 11, 12, 13, 15, 16, 17, 18})}
        tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 12), ("[SEP]", 13)])
        tokenizer.save(str(root / "tokenizer.json"))
        config = RobertaConfig(hidden_size=384, num_hidden_layers=1, num_attention_heads=12,
                               intermediate_size=32, vocab_size=32, max_position_embeddings=515,
                               pad_token_id=1, hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
        (root / "config.json").write_text(json.dumps(config.to_dict()))
        torch.manual_seed(13 if variant.endswith("mtr") else 17)
        model = RobertaModel(config, add_pooling_layer=False)
        torch.save({"roberta." + k: v for k, v in model.state_dict().items()}, root / "pytorch_model.bin")
        monkeypatch.setitem(chemberta.FILES, variant, {name: sha256_file(root / name) for name in ("config.json", "tokenizer.json", "pytorch_model.bin")})
        repo_id, revision = chemberta.SOURCES[variant]
        return FrozenEncoderConfig("chemberta2", variant, repo_id, revision, str(root))
    return create


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
    vocabulary = SmilesTokenizer.fit(smiles_values, backend="ais")
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
                    vocabulary.encode(smiles, max_length=256), dtype=torch.long
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
