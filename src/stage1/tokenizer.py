from __future__ import annotations

import io
import importlib.metadata
import json
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
APE_COMMIT = "ff1b3cc00476a8d017d7d54e925681a04475d47f"


def tokenizer_backend_version(backend: str) -> str:
    if backend == "ape":
        return APE_COMMIT
    distribution = {
        "ais": "atomInSmiles",
        "bpe": "tokenizers",
        "spe": "SmilesPE",
    }.get(backend)
    if distribution is None:
        raise ValueError(f"Unsupported tokenizer backend: {backend}")
    return importlib.metadata.version(distribution)


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
    return _ais_module().encode(smiles).split()


def _require_backend(name: str, install_hint: str):
    try:
        return __import__(name, fromlist=["*"])
    except ImportError as exc:
        raise RuntimeError(install_hint) from exc


@dataclass(frozen=True)
class SmilesTokenizer:
    tokens: tuple[str, ...]
    backend: str = "ais"
    state: str | None = None
    vocabulary_budget: int | None = None
    min_frequency: int | None = None
    backend_version: str | None = None

    def __post_init__(self) -> None:
        if self.tokens[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError("Tokenizer vocabulary must start with required special tokens")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Tokenizer vocabulary contains duplicate tokens")
        if self.backend not in {"ais", "ape", "bpe", "spe"}:
            raise ValueError(f"Unsupported tokenizer backend: {self.backend}")

    @classmethod
    def fit(
        cls,
        smiles_values: Iterable[str],
        backend: str = "ais",
        vocab_size: int = 2048,
        min_frequency: int = 2,
    ) -> "SmilesTokenizer":
        iterator = iter(smiles_values)
        try:
            first = next(iterator)
        except StopIteration as error:
            raise ValueError("Cannot fit a tokenizer on an empty corpus") from error
        corpus = chain((first,), iterator)
        if backend == "ais":
            counts: Counter[str] = Counter()
            for smiles in corpus:
                counts.update(ais_tokenize(smiles))
            learned = sorted(counts, key=lambda token: (-counts[token], token))
            learned = [token for token in learned if token not in SPECIAL_TOKENS]
            learned = learned[: max(0, vocab_size - len(SPECIAL_TOKENS))]
            return cls(
                tuple(SPECIAL_TOKENS) + tuple(learned),
                backend="ais",
                vocabulary_budget=vocab_size,
                min_frequency=min_frequency,
                backend_version=tokenizer_backend_version("ais"),
            )
        if backend == "bpe":
            _require_backend(
                "tokenizers",
                "BPE support requires the tokenizers extra: pip install -e '.[tokenizers]'",
            )
            from tokenizers import Tokenizer, models, pre_tokenizers, trainers

            tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=list(SPECIAL_TOKENS),
                show_progress=False,
            )
            tokenizer.train_from_iterator(corpus, trainer=trainer)
            vocabulary = tokenizer.get_vocab()
            tokens = tuple(
                token for token, _ in sorted(vocabulary.items(), key=lambda item: item[1])
            )
            return cls(
                tokens=tokens,
                backend="bpe",
                state=tokenizer.to_str(),
                vocabulary_budget=vocab_size,
                min_frequency=min_frequency,
                backend_version=tokenizer_backend_version("bpe"),
            )
        if backend == "spe":
            corpus = list(corpus)
            _require_backend(
                "SmilesPE",
                "SPE support requires the tokenizers extra: pip install -e '.[tokenizers]'",
            )
            from SmilesPE import learner
            from SmilesPE.pretokenizer import atomwise_tokenizer

            atom_tokens = sorted({token for value in corpus for token in atomwise_tokenizer(value)})
            merge_budget = max(0, vocab_size - len(SPECIAL_TOKENS) - len(atom_tokens))
            output = io.StringIO()
            learner.learn_SPE(
                corpus,
                output,
                num_symbols=merge_budget,
                min_frequency=min_frequency,
                verbose=False,
            )
            codes = output.getvalue()
            merged = ["".join(line.split()) for line in codes.splitlines() if line.strip()]
            learned = list(dict.fromkeys([*atom_tokens, *merged]))
            return cls(
                tokens=tuple(SPECIAL_TOKENS) + tuple(learned),
                backend="spe",
                state=codes,
                vocabulary_budget=vocab_size,
                min_frequency=min_frequency,
                backend_version=tokenizer_backend_version("spe"),
            )
        if backend == "ape":
            corpus = list(corpus)
            _require_backend(
                "apetokenizer",
                "APE support requires the pinned tokenizer extra: pip install -e '.[tokenizers]'",
            )
            from apetokenizer.ape_tokenizer import APETokenizer

            tokenizer = APETokenizer(
                pad_token="[PAD]",
                bos_token="[CLS]",
                eos_token="[SEP]",
                unk_token="[UNK]",
                mask_token="[MASK]",
            )
            sequences = [tokenizer.pre_tokenize(value) for value in corpus]
            token_counts = Counter(token for sequence in sequences for token in sequence)
            learned = sorted(token_counts, key=lambda token: (-token_counts[token], token))
            learned_set = set(learned)
            target = max(0, vocab_size - len(SPECIAL_TOKENS))
            while len(learned) < target:
                pair_counts: Counter[tuple[str, str]] = Counter(
                    pair
                    for sequence in sequences
                    for pair in zip(sequence, sequence[1:])
                )
                if not pair_counts:
                    break
                pair, frequency = max(
                    pair_counts.items(), key=lambda item: (item[1], item[0])
                )
                if frequency < min_frequency:
                    break
                merged = "".join(pair)
                new_sequences: list[list[str]] = []
                for sequence in sequences:
                    result: list[str] = []
                    cursor = 0
                    while cursor < len(sequence):
                        if (
                            cursor + 1 < len(sequence)
                            and (sequence[cursor], sequence[cursor + 1]) == pair
                        ):
                            result.append(merged)
                            cursor += 2
                        else:
                            result.append(sequence[cursor])
                            cursor += 1
                    new_sequences.append(result)
                sequences = new_sequences
                if merged not in learned_set:
                    learned.append(merged)
                    learned_set.add(merged)
                else:
                    # No new vocabulary item means this merge cannot make progress.
                    break
            return cls(
                tokens=tuple(SPECIAL_TOKENS) + tuple(learned[:target]),
                backend="ape",
                vocabulary_budget=vocab_size,
                min_frequency=min_frequency,
                backend_version=tokenizer_backend_version("ape"),
            )
        raise ValueError(f"Unsupported tokenizer backend: {backend}")

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

    def _tokenize(self, smiles: str) -> list[str]:
        if self.backend == "ais":
            return ais_tokenize(smiles)
        if self.backend == "bpe":
            from tokenizers import Tokenizer

            if self.state is None:
                raise ValueError("BPE tokenizer artifact is missing its serialized state")
            tokenizer = Tokenizer.from_str(self.state)
            return tokenizer.encode(smiles, add_special_tokens=False).tokens
        if self.backend == "spe":
            from SmilesPE.tokenizer import SPE_Tokenizer

            tokenizer = SPE_Tokenizer(io.StringIO(self.state or ""))
            tokenized = tokenizer.tokenize(smiles)
            return tokenized.split() if tokenized else []
        # The official APE encoder uses longest-vocabulary matching; keeping it here
        # makes the saved artifact self-contained and independent of upstream internals.
        learned = sorted(self.tokens[len(SPECIAL_TOKENS) :], key=lambda token: (-len(token), token))
        result: list[str] = []
        cursor = 0
        while cursor < len(smiles):
            match = next((token for token in learned if smiles.startswith(token, cursor)), None)
            if match is None:
                result.append("[UNK]")
                cursor += 1
            else:
                result.append(match)
                cursor += len(match)
        return result

    def encode(self, smiles: str, max_length: int) -> list[int]:
        ids_by_token = self.token_to_id
        content = [ids_by_token.get(token, self.unk_id) for token in self._tokenize(smiles)]
        ids = [self.cls_id, *content, self.sep_id]
        if len(ids) > max_length:
            raise ValueError(
                f"{self.backend.upper()} sequence has {len(ids)} tokens, "
                f"exceeding max_length={max_length}: {smiles}"
            )
        return ids

    def token_count(self, smiles: str) -> int:
        return len(self._tokenize(smiles)) + 2

    def save(self, path: str | Path) -> None:
        payload = {
            "format_version": 2,
            "backend": self.backend,
            "tokens": list(self.tokens),
            "state": self.state,
            "vocabulary_budget": self.vocabulary_budget,
            "actual_vocabulary_size": len(self.tokens),
            "min_frequency": self.min_frequency,
            "backend_version": self.backend_version,
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "SmilesTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") == 1:
            return cls(tuple(payload["tokens"]), backend="ais")
        if payload.get("format_version") != 2:
            raise ValueError("Unsupported tokenizer artifact format")
        return cls(
            tokens=tuple(payload["tokens"]),
            backend=payload["backend"],
            state=payload.get("state"),
            vocabulary_budget=payload.get("vocabulary_budget"),
            min_frequency=payload.get("min_frequency"),
            backend_version=payload.get("backend_version"),
        )


class AISVocabulary(SmilesTokenizer):
    @classmethod
    def fit(cls, smiles_values: Iterable[str]) -> "AISVocabulary":
        tokenizer = SmilesTokenizer.fit(smiles_values, backend="ais")
        return cls(
            tokens=tokenizer.tokens,
            backend="ais",
            vocabulary_budget=tokenizer.vocabulary_budget,
            min_frequency=tokenizer.min_frequency,
            backend_version=tokenizer.backend_version,
        )

    @classmethod
    def load(cls, path: str | Path) -> "AISVocabulary":
        tokenizer = SmilesTokenizer.load(path)
        return cls(
            tokens=tokenizer.tokens,
            backend=tokenizer.backend,
            state=tokenizer.state,
            vocabulary_budget=tokenizer.vocabulary_budget,
            min_frequency=tokenizer.min_frequency,
            backend_version=tokenizer.backend_version,
        )
