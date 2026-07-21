from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from ilume_pretrain.descriptors import DescriptorStandardizer
from ilume_pretrain.masking import mask_smiles_tokens
from ilume_pretrain.tokenizer import AISVocabulary, ais_tokenize


def test_ais_round_trip_and_vocabulary_save_load(tmp_path):
    import atomInSmiles

    smiles = "NCC(=O)O"
    encoded = " ".join(ais_tokenize(smiles))
    decoded = atomInSmiles.decode(encoded)
    assert Chem.MolToSmiles(Chem.MolFromSmiles(decoded)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(smiles)
    )

    vocabulary = AISVocabulary.fit([smiles, "CCO"])
    path = tmp_path / "tokenizer.json"
    vocabulary.save(path)
    loaded = AISVocabulary.load(path)
    assert loaded.tokens == vocabulary.tokens
    assert loaded.encode(smiles, max_length=32) == vocabulary.encode(
        smiles, max_length=32
    )
    with pytest.raises(ValueError, match="exceeding"):
        loaded.encode(smiles, max_length=3)


def test_smiles_masking_uses_bert_replacement_distribution():
    vocabulary = AISVocabulary.fit(["CCO", "NCC"])
    token_ids = np.full(10_002, vocabulary.token_to_id[ais_tokenize("CCO")[0]])
    token_ids[0] = vocabulary.cls_id
    token_ids[-1] = vocabulary.sep_id
    import torch

    token_ids_tensor = torch.from_numpy(token_ids).long()
    positions = torch.arange(1, 10_001)
    corrupted, labels = mask_smiles_tokens(
        token_ids_tensor,
        positions,
        ratio=1.0,
        vocabulary=vocabulary,
        generator=torch.Generator().manual_seed(11),
        drop_entire_modality=False,
    )
    assert (labels != -100).sum().item() == 10_000
    assert labels[0].item() == labels[-1].item() == -100
    mask_fraction = (corrupted[positions] == vocabulary.mask_id).float().mean().item()
    assert 0.77 < mask_fraction < 0.83


def test_descriptor_standardizer_is_finite_aware_and_round_trips(tmp_path):
    values = np.asarray(
        [
            [1.0, 4.0, np.nan],
            [3.0, 4.0, 9.0],
            [100.0, 4.0, 11.0],
        ]
    )
    standardizer = DescriptorStandardizer.fit(values[:2], ("a", "b", "c"))
    assert standardizer.means.tolist() == [2.0, 4.0, 9.0]
    assert standardizer.scales.tolist() == [1.0, 1.0, 1.0]
    assert standardizer.finite_counts.tolist() == [2, 2, 1]

    transformed, valid = standardizer.transform(values)
    assert transformed[0, 2] == 0.0
    assert not valid[0, 2]
    assert transformed[2, 0] == 98.0

    path = tmp_path / "scaler.json"
    standardizer.save(path)
    loaded = DescriptorStandardizer.load(path, expected_names=("a", "b", "c"))
    np.testing.assert_allclose(loaded.means, standardizer.means)
    with pytest.raises(ValueError, match="names/order"):
        DescriptorStandardizer.load(path, expected_names=("b", "a", "c"))

    no_training_value = DescriptorStandardizer.fit(
        np.asarray([[1.0, np.nan], [3.0, np.nan]]), ("a", "missing")
    )
    transformed, valid = no_training_value.transform(np.asarray([[2.0, 8.0]]))
    assert transformed[0, 1] == 0.0
    assert not valid[0, 1]
