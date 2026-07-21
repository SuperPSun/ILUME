from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")


def _ais_module():
    try:
        import atomInSmiles
    except ImportError as exc:
        raise RuntimeError(
            "Atom-in-SMILES support requires atomInSmiles==1.0.2. "
            "Install the project dependencies first."
        ) from exc
    return atomInSmiles


def ais_tokenize(smiles: str) -> list[str]:
    encoded = _ais_module().encode(smiles)
    return encoded.split()


@dataclass(frozen=True)
class AISVocabulary:
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.tokens[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError("AIS vocabulary must start with the required special tokens")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("AIS vocabulary contains duplicate tokens")

    @classmethod
    def fit(cls, smiles_values: Iterable[str]) -> "AISVocabulary":
        counts: Counter[str] = Counter()
        for smiles in smiles_values:
            counts.update(ais_tokenize(smiles))
        learned = sorted(counts, key=lambda token: (-counts[token], token))
        learned = [token for token in learned if token not in SPECIAL_TOKENS]
        return cls(tuple(SPECIAL_TOKENS) + tuple(learned))

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def cls_id(self) -> int:
        return 2

    @property
    def sep_id(self) -> int:
        return 3

    @property
    def mask_id(self) -> int:
        return 4

    def encode(self, smiles: str, max_length: int) -> list[int]:
        ids_by_token = self.token_to_id
        content = [ids_by_token.get(token, self.unk_id) for token in ais_tokenize(smiles)]
        ids = [self.cls_id, *content, self.sep_id]
        if len(ids) > max_length:
            raise ValueError(
                f"AIS sequence has {len(ids)} tokens, exceeding max_length={max_length}: "
                f"{smiles}"
            )
        return ids

    def save(self, path: str | Path) -> None:
        payload = {
            "format_version": 1,
            "backend": "atomInSmiles",
            "tokens": list(self.tokens),
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "AISVocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported tokenizer artifact format")
        return cls(tuple(payload["tokens"]))
