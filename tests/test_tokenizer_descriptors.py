from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from ilume_pretrain.descriptors import DescriptorSchema, DescriptorStandardizer
from ilume_pretrain.masking import mask_smiles_tokens
from ilume_pretrain.tokenizer import AISVocabulary, SmilesTokenizer, ais_tokenize


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


def test_token_count_includes_special_tokens_and_matches_encode_boundary():
    tokenizer = AISVocabulary.fit(["C" * 255])
    assert tokenizer.token_count("C" * 254) == 256
    assert len(tokenizer.encode("C" * 254, max_length=256)) == 256
    assert tokenizer.token_count("C" * 255) == 257
    with pytest.raises(ValueError, match="257 tokens"):
        tokenizer.encode("C" * 255, max_length=256)


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


def test_descriptor_schema_clean_pruned_groups_and_save_load(tmp_path):
    values = np.asarray(
        [
            [1.0, 2.0, 1.0, np.nan, 4.0, 1.01],
            [2.0, 3.0, 2.0, np.nan, 4.0, 2.01],
            [3.0, 4.0, 3.0, np.nan, 4.0, 3.01],
            [4.0, 5.0, 4.0, np.nan, 4.0, 4.01],
        ]
    )
    names = ("MolWt", "Chi0", "duplicate", "missing", "constant", "Chi1")
    clean = DescriptorSchema.fit(values, names, "clean", 8)
    assert clean.selected_names == ("MolWt", "Chi0", "Chi1")
    assert clean.removal_reasons["missing"] == "all_non_finite"
    assert clean.removal_reasons["constant"] == "zero_variance"
    assert clean.removal_reasons["duplicate"] == "duplicate_of:MolWt"
    assert len(clean.group_indices) == 8
    assert clean.semantic_mapping_version == "rdkit-217-v1"
    assert len(clean.raw_semantic_groups) == len(names)

    pruned = DescriptorSchema.fit(values, names, "pruned", 12, 0.98)
    assert pruned.selected_dim == 1
    assert len(pruned.correlation_clusters) == 1
    assert pruned.cluster_representatives == ("MolWt",)
    path = tmp_path / "schema.json"
    pruned.save(path)
    loaded = DescriptorSchema.load(path, expected_raw_names=names)
    assert loaded == pruned
    with pytest.raises(ValueError, match="names/order"):
        DescriptorSchema.load(path, expected_raw_names=names[::-1])


@pytest.mark.parametrize("backend", ["bpe", "spe", "ape"])
def test_data_driven_tokenizers_share_budget_and_round_trip_artifact(
    backend, tmp_path
):
    if backend == "bpe":
        pytest.importorskip("tokenizers")
    elif backend == "spe":
        pytest.importorskip("SmilesPE")
    else:
        pytest.importorskip("apetokenizer")
    corpus = ["CCO", "CCN", "CCC", "C=O"]
    tokenizer = SmilesTokenizer.fit(
        corpus, backend=backend, vocab_size=32, min_frequency=2
    )
    encoded = tokenizer.encode("CCO", max_length=32)
    assert encoded[0] == tokenizer.cls_id
    assert encoded[-1] == tokenizer.sep_id
    assert len(tokenizer.tokens) <= 32
    path = tmp_path / f"{backend}.json"
    tokenizer.save(path)
    loaded = SmilesTokenizer.load(path)
    assert loaded.encode("CCO", max_length=32) == encoded
    assert loaded.backend == backend
    assert loaded.backend_version == tokenizer.backend_version
