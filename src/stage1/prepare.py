from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import random
import re
import sqlite3
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import torch
from rdkit import Chem, rdBase

from common.io import atomic_json, sha256_file
from common.identity import IDENTITY_CONTRACT_VERSION, semantic_identity
from common.progress import ProgressReporter
from common.training import canonical_json_sha256
from .config import DataConfig, PretrainConfig
from .data import (
    CORPUS_FORMAT_VERSION,
    GLOBAL_RDKIT_CORPUS_FORMAT_VERSION,
    CORPUS_KIND,
    CORPUS_SHARD_KIND,
    INDEX_DTYPE,
    PreparedCorpusDataset,
)
from .descriptors import (
    DescriptorSchema,
    DescriptorStandardizer,
    calculate_descriptors,
    rdkit_descriptor_names,
)
from .features import (
    IPC_SQUARE_OVERFLOW_LIMIT,
    ROLE_TO_ID,
    build_entity_sample,
    inspect_entity_qc,
)
from .identity import (
    build_stage1_corpus_identity,
    build_stage1_feature_identity,
    build_stage1_sampler_layout_identity,
    feature_generation_contract,
)
from .tokenizer import SmilesTokenizer, ais_tokenize


ROLE_SOURCE_FILES = {
    "cation": "cation.csv",
    "anion": "anion.csv",
    "neutral": "molecule.csv",
}
_EMPTY_JSON = "[]"


def _batches(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _ordered_batch_map(
    worker: Callable[[Any], Any],
    batches: Iterable[Any],
    workers: int,
) -> Iterator[Any]:
    if workers == 1:
        for batch in batches:
            yield worker(batch)
        return
    iterator = iter(batches)
    pending = deque()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        for _ in range(2 * workers):
            try:
                pending.append(executor.submit(worker, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(worker, next(iterator)))
            except StopIteration:
                pass


def _performance_phase(
    processed: int, elapsed_seconds: float, reused: bool
) -> dict[str, float | int | bool]:
    return {
        "processed": processed,
        "elapsed_seconds": elapsed_seconds,
        "items_per_second": (
            processed / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        ),
        "reused": reused,
    }


def _write_performance(
    path: Path | None,
    config: PretrainConfig,
    phases: dict[str, dict[str, float | int | bool]],
    total_elapsed_seconds: float,
) -> None:
    if path is None:
        return
    atomic_json(
        path,
        {
            "preparation": config.to_dict()["preparation"],
            "phases": phases,
            "total_elapsed_seconds": total_elapsed_seconds,
        },
    )


def _canonicalize_augmentation_batch(
    batch: list[tuple[int, str, tuple[str, ...], str]],
) -> list[tuple[int, str, tuple[str, ...]]]:
    results = []
    for row_number, smiles, seed_values, context in batch:
        try:
            canonical = _canonicalize(smiles, f"{context}:{row_number}")
            seeds = []
            for seed in seed_values:
                try:
                    seeds.append(_canonicalize(seed, f"{context}:{row_number} seed"))
                except ValueError:
                    seeds.append(seed)
            results.append((row_number, canonical, tuple(seeds)))
        except BaseException as error:
            raise RuntimeError(
                f"catalog record row={row_number} smiles={smiles!r}: "
                f"{type(error).__name__}: {error}"
            ) from error
    return results


def _entity_qc_batch(
    batch: list[tuple[int, str]],
) -> list[tuple[int, tuple[str, ...], tuple[str, ...], float]]:
    results = []
    for record_id, canonical_smiles in batch:
        try:
            inspected = inspect_entity_qc({"canonical_smiles": canonical_smiles})
        except BaseException as error:
            raise RuntimeError(
                f"entity_qc record id={record_id} smiles={canonical_smiles!r}: "
                f"{type(error).__name__}: {error}"
            ) from error
        results.append(
            (
                record_id,
                tuple(inspected.reasons),
                inspected.unsupported_bond_types,
                inspected.ipc,
            )
        )
    return results


def _ais_batch(
    task: tuple[list[tuple[int, str, str]], int],
) -> tuple[list[tuple[int, int, bool]], Counter[str]]:
    batch, max_smiles_tokens = task
    updates = []
    counts: Counter[str] = Counter()
    for record_id, canonical_smiles, split in batch:
        try:
            tokens = ais_tokenize(canonical_smiles)
        except BaseException as error:
            raise RuntimeError(
                f"tokenizer record id={record_id} smiles={canonical_smiles!r}: "
                f"{type(error).__name__}: {error}"
            ) from error
        token_count = len(tokens) + 2
        overlength = token_count > max_smiles_tokens
        if split == "train" and not overlength:
            counts.update(tokens)
        updates.append((record_id, token_count, overlength))
    return updates, counts


def _descriptor_batch(
    task: tuple[list[tuple[int, str]], tuple[str, ...]],
) -> tuple[list[int], np.ndarray]:
    batch, raw_names = task
    rows = []
    values = []
    for descriptor_row, canonical_smiles in batch:
        try:
            mol = Chem.MolFromSmiles(canonical_smiles)
            if mol is None:
                raise RuntimeError("Canonical SMILES unexpectedly failed RDKit parsing")
            value = calculate_descriptors(mol, raw_names)
        except BaseException as error:
            raise RuntimeError(
                f"descriptors row={descriptor_row} smiles={canonical_smiles!r}: "
                f"{type(error).__name__}: {error}"
            ) from error
        rows.append(descriptor_row)
        values.append(value)
    return rows, np.asarray(values, dtype=np.float64)


def preparation_source_paths(config: PretrainConfig | DataConfig) -> list[Path]:
    config = config.data if isinstance(config, PretrainConfig) else config
    paths = [config.stage1_dir / name for name in ROLE_SOURCE_FILES.values()]
    if config.include_augmentation:
        paths.extend(
            config.stage1_dir / "augmentation" / name
            for name in ROLE_SOURCE_FILES.values()
        )
    return paths


def _csv_data_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def _source_measurements(
    data_config: DataConfig,
    source_paths: list[Path],
    source_identity: dict[str, object] | None,
) -> tuple[dict[str, dict[str, Any]], int]:
    if source_identity is None:
        measurements = {
            str(path.relative_to(data_config.stage1_dir)): {
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
                "rows": _csv_data_row_count(path),
            }
            for path in source_paths
        }
        return measurements, sum(int(item["rows"]) for item in measurements.values())
    if source_identity.get("schema_version") != 2:
        raise ValueError("Unsupported Stage 1 source identity schema")
    if source_identity.get("stage") != "stage1":
        raise ValueError("Stage 1 source identity has the wrong stage")
    locator = source_identity.get("locator", {}).get("files", {})
    integrity = source_identity.get("integrity", {}).get("files", {})
    semantic_sources = (
        source_identity.get("semantic", {})
        .get("identities", {})
        .get("source", {})
        .get("payload", {})
        .get("sources", {})
    )
    if (
        not isinstance(locator, dict)
        or not isinstance(integrity, dict)
        or not isinstance(semantic_sources, dict)
        or set(locator) != set(integrity)
        or set(locator) != set(semantic_sources)
        or len(locator) != len(source_paths)
    ):
        raise ValueError("Stage 1 source identity file set does not match the config")
    remaining = set(locator)
    measurements: dict[str, dict[str, Any]] = {}
    source_row_count = 0
    relatives = sorted(
        (path.relative_to(data_config.stage1_dir) for path in source_paths),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for relative in relatives:
        matches = [logical_id for logical_id in remaining if tuple(
            Path(str(locator[logical_id])).parts[-len(relative.parts) :]
        ) == relative.parts]
        if len(matches) != 1:
            raise ValueError(
                f"Stage 1 source identity does not uniquely identify {relative}"
            )
        logical_id = matches[0]
        item = semantic_sources[logical_id]
        digest = item.get("sha256")
        rows = item.get("rows")
        size = item.get("size")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(rows, int)
            or rows < 0
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValueError(f"Invalid Stage 1 source identity entry for {relative}")
        measurements[relative.as_posix()] = {
            "sha256": digest,
            "size": size,
            "rows": rows,
        }
        source_row_count += rows
        remaining.remove(logical_id)
    if remaining:
        raise ValueError("Stage 1 source identity contains unexpected files")
    return measurements, source_row_count


def _canonicalize(smiles: str, context: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES in {context}: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def _seed_values(raw: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in re.split(r"[;|]", raw) if value.strip())


def _record_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "role": row["role"],
        "role_id": int(row["role_id"]),
        "canonical_smiles": row["canonical_smiles"],
        "sources": tuple(json.loads(row["sources"])),
        "split": row["split"],
        "is_augmented": bool(row["is_augmented"]),
        "seed_smiles": tuple(json.loads(row["seed_smiles"])),
        "sample_id": row["sample_id"],
    }


def _connect_catalog(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def _create_catalog(connection: sqlite3.Connection, signature: str) -> None:
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE records (
            id INTEGER PRIMARY KEY,
            role TEXT NOT NULL,
            role_id INTEGER NOT NULL,
            canonical_smiles TEXT NOT NULL,
            sources TEXT NOT NULL,
            split TEXT NOT NULL,
            is_augmented INTEGER NOT NULL,
            seed_smiles TEXT NOT NULL,
            mix_key BLOB NOT NULL,
            reasons TEXT NOT NULL DEFAULT '[]',
            unsupported_bond_types TEXT NOT NULL DEFAULT '[]',
            ipc TEXT,
            token_count INTEGER,
            sample_id TEXT,
            descriptor_row INTEGER,
            UNIQUE(role, canonical_smiles)
        );
        """
    )
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES('signature', ?)",
        (json.dumps(signature),),
    )
    connection.commit()


def _create_catalog_indexes(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE INDEX IF NOT EXISTS records_split ON records(split);
        CREATE INDEX IF NOT EXISTS records_mix
            ON records(split, mix_key, canonical_smiles);
        """
    )
    connection.commit()


def _catalog_metadata(connection: sqlite3.Connection, key: str) -> Any:
    row = connection.execute(
        "SELECT value FROM metadata WHERE key = ?", (key,)
    ).fetchone()
    return None if row is None else json.loads(row["value"])


def _set_catalog_metadata(
    connection: sqlite3.Connection, key: str, value: Any
) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
        (key, json.dumps(value, sort_keys=True)),
    )
    connection.commit()


def _mix_key(seed: int, split: str, canonical_smiles: str) -> bytes:
    return hashlib.sha256(
        f"{seed}\0{split}\0{canonical_smiles}".encode()
    ).digest()


def _load_originals(
    connection: sqlite3.Connection,
    config: DataConfig,
    progress: Any,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    originals: dict[str, set[str]] = {}
    validation: dict[str, set[str]] = {}
    for role_index, (role, filename) in enumerate(ROLE_SOURCE_FILES.items()):
        path = config.stage1_dir / filename
        canonical_to_sources: dict[str, set[str]] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                canonical = _canonicalize(
                    (row.get("SMILES") or "").strip(), f"{path}:{row_number}"
                )
                canonical_to_sources.setdefault(canonical, set()).add(path.stem)
                progress.update(1)
        items = sorted(canonical_to_sources.items())
        random.Random(config.seed + role_index).shuffle(items)
        if config.max_samples_per_role is not None:
            items = items[: config.max_samples_per_role]
        if len(items) < 2:
            raise ValueError(f"Role {role} needs at least two original entities")
        valid_count = min(
            max(1, round(len(items) * config.valid_fraction)), len(items) - 1
        )
        valid_smiles = {smiles for smiles, _ in items[:valid_count]}
        originals[role] = {smiles for smiles, _ in items}
        validation[role] = valid_smiles
        connection.executemany(
            """
            INSERT INTO records(
                role, role_id, canonical_smiles, sources, split,
                is_augmented, seed_smiles, mix_key
            ) VALUES(?, ?, ?, ?, ?, 0, '[]', ?)
            """,
            [
                (
                    role,
                    ROLE_TO_ID[role],
                    smiles,
                    json.dumps(sorted(sources)),
                    "valid" if smiles in valid_smiles else "train",
                    _mix_key(
                        config.seed,
                        "valid" if smiles in valid_smiles else "train",
                        smiles,
                    ),
                )
                for smiles, sources in items
            ],
        )
        connection.commit()
    return originals, validation


def _empty_augmentation_stats(included: bool) -> dict[str, Any]:
    return {
        "included": included,
        "source_rows": 0,
        "excluded_valid_seed": 0,
        "excluded_overlap": 0,
        "excluded_duplicate": 0,
        "eligible": 0,
        "excluded_qc": 0,
        "retained": 0,
    }


def _load_augmentation(
    connection: sqlite3.Connection,
    config: PretrainConfig,
    originals: dict[str, set[str]],
    validation: dict[str, set[str]],
    progress: Any,
) -> dict[str, dict[str, Any]]:
    data_config = config.data
    audit: dict[str, dict[str, Any]] = {}
    for role, filename in ROLE_SOURCE_FILES.items():
        stats = _empty_augmentation_stats(data_config.include_augmentation)
        audit[role] = stats
        if not data_config.include_augmentation:
            continue
        path = data_config.stage1_dir / "augmentation" / filename
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)

            def inputs() -> Iterator[tuple[int, str, tuple[str, ...], str]]:
                for row_number, row in enumerate(reader, start=2):
                    yield (
                        row_number,
                        (row.get("SMILES") or "").strip(),
                        _seed_values(
                            (row.get("seed_smiles_list") or "").strip()
                        ),
                        str(path),
                    )

            batches = _batches(inputs(), config.preparation.catalog_batch_size)
            for result_batch in _ordered_batch_map(
                _canonicalize_augmentation_batch,
                batches,
                config.preparation.workers,
            ):
                insert_rows = []
                for row_number, canonical, seeds_tuple in result_batch:
                    del row_number
                    seeds = list(seeds_tuple)
                    stats["source_rows"] += 1
                    if any(seed in validation[role] for seed in seeds):
                        stats["excluded_valid_seed"] += 1
                    elif canonical in originals[role]:
                        stats["excluded_overlap"] += 1
                    else:
                        insert_rows.append(
                            (
                                role,
                                ROLE_TO_ID[role],
                                canonical,
                                json.dumps([f"augmentation/{path.stem}"]),
                                json.dumps(sorted(set(seeds))),
                                _mix_key(data_config.seed, "train", canonical),
                            )
                        )
                before = connection.total_changes
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO records(
                        role, role_id, canonical_smiles, sources, split,
                        is_augmented, seed_smiles, mix_key
                    ) VALUES(?, ?, ?, ?, 'train', 1, ?, ?)
                    """,
                    insert_rows,
                )
                inserted = connection.total_changes - before
                stats["eligible"] += inserted
                stats["excluded_duplicate"] += len(insert_rows) - inserted
                connection.commit()
                progress.update(len(result_batch))
        connection.commit()
    return audit


def _build_catalog(
    connection: sqlite3.Connection,
    config: PretrainConfig,
    source_row_count: int,
    reporter: ProgressReporter,
) -> dict[str, dict[str, Any]]:
    with reporter.bar(
        total=source_row_count, desc="Load/canonicalize", unit="row"
    ) as progress:
        originals, validation = _load_originals(connection, config.data, progress)
        audit = _load_augmentation(
            connection, config, originals, validation, progress
        )
    _create_catalog_indexes(connection)
    _set_catalog_metadata(connection, "augmentation_audit", audit)
    _set_catalog_metadata(connection, "phase", "catalog")
    return audit


def _retained_smiles(
    connection: sqlite3.Connection, split: str | None = None
) -> Iterator[str]:
    where = "reasons = '[]'"
    parameters: tuple[Any, ...] = ()
    if split is not None:
        where += " AND split = ?"
        parameters = (split,)
    for row in connection.execute(
        f"SELECT canonical_smiles FROM records WHERE {where} ORDER BY id",
        parameters,
    ):
        yield row["canonical_smiles"]


def _validate_role_splits(connection: sqlite3.Connection) -> None:
    missing = [
        f"{role}/{split}"
        for role in ROLE_TO_ID
        for split in ("train", "valid")
        if connection.execute(
            """
            SELECT 1 FROM records
            WHERE role = ? AND split = ? AND reasons = '[]' LIMIT 1
            """,
            (role, split),
        ).fetchone()
        is None
    ]
    if missing:
        raise ValueError(
            "Quality control removed every entity from: " + ", ".join(missing)
        )


def _run_qc_and_tokenizer(
    connection: sqlite3.Connection,
    config: PretrainConfig,
    reporter: ProgressReporter,
) -> tuple[SmilesTokenizer, dict[str, dict[str, float | int | bool]]]:
    total = int(connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])
    entity_started = time.perf_counter()
    connection.execute(
        """
        UPDATE records SET reasons = '[]', unsupported_bond_types = '[]',
                           ipc = NULL, token_count = NULL,
                           sample_id = NULL, descriptor_row = NULL
        """
    )
    with reporter.bar(total=total, desc="Entity QC", unit="entity") as progress:
        updates: list[tuple[str, str, str, int]] = []
        inputs = (
            (int(row["id"]), row["canonical_smiles"])
            for row in connection.execute(
                "SELECT id, canonical_smiles FROM records ORDER BY id"
            )
        )
        for result_batch in _ordered_batch_map(
            _entity_qc_batch,
            _batches(inputs, config.preparation.qc_batch_size),
            config.preparation.workers,
        ):
            updates.extend(
                (
                    json.dumps(reasons),
                    json.dumps(unsupported),
                    repr(ipc),
                    record_id,
                )
                for record_id, reasons, unsupported, ipc in result_batch
            )
            if len(updates) >= 10000:
                connection.executemany(
                    """
                    UPDATE records SET reasons = ?, unsupported_bond_types = ?,
                                       ipc = ? WHERE id = ?
                    """,
                    updates,
                )
                connection.commit()
                updates.clear()
            progress.update(len(result_batch))
        if updates:
            connection.executemany(
                """
                UPDATE records SET reasons = ?, unsupported_bond_types = ?,
                                   ipc = ? WHERE id = ?
                """,
                updates,
            )
            connection.commit()

    entity_elapsed = time.perf_counter() - entity_started
    tokenizer_started = time.perf_counter()
    if config.tokenizer.backend == "ais":
        _validate_role_splits(connection)
        retained = int(
            connection.execute(
                "SELECT COUNT(*) FROM records WHERE reasons = '[]'"
            ).fetchone()[0]
        )
        counts: Counter[str] = Counter()
        updates = []
        with reporter.bar(
            total=retained,
            desc="AIS tokenization/QC",
            unit="entity",
        ) as progress:
            inputs = (
                (int(row["id"]), row["canonical_smiles"], row["split"])
                for row in connection.execute(
                    """
                    SELECT id, canonical_smiles, split FROM records
                    WHERE reasons = '[]' ORDER BY id
                    """
                )
            )
            tasks = (
                (batch, config.data.max_smiles_tokens)
                for batch in _batches(
                    inputs, config.preparation.tokenizer_batch_size
                )
            )
            for result_batch, local_counts in _ordered_batch_map(
                _ais_batch, tasks, config.preparation.workers
            ):
                counts.update(local_counts)
                updates.extend(
                    (
                        token_count,
                        json.dumps(["smiles_overlength"] if overlength else []),
                        record_id,
                    )
                    for record_id, token_count, overlength in result_batch
                )
                if len(updates) >= 10000:
                    connection.executemany(
                        "UPDATE records SET token_count = ?, reasons = ? WHERE id = ?",
                        updates,
                    )
                    connection.commit()
                    updates.clear()
                progress.update(len(result_batch))
            if updates:
                connection.executemany(
                    "UPDATE records SET token_count = ?, reasons = ? WHERE id = ?",
                    updates,
                )
                connection.commit()
        tokenizer = SmilesTokenizer.fit_ais_counts(
            counts,
            vocab_size=config.tokenizer.vocab_size,
            min_frequency=config.tokenizer.min_frequency,
        )
    else:
        pass_index = 0
        while True:
            pass_index += 1
            _validate_role_splits(connection)
            with reporter.status(f"Tokenizer fit pass {pass_index}"):
                tokenizer = SmilesTokenizer.fit(
                    _retained_smiles(connection, "train"),
                    backend=config.tokenizer.backend,
                    vocab_size=config.tokenizer.vocab_size,
                    min_frequency=config.tokenizer.min_frequency,
                )
            retained = int(
                connection.execute(
                    "SELECT COUNT(*) FROM records WHERE reasons = '[]'"
                ).fetchone()[0]
            )
            newly_excluded = 0
            updates = []
            with reporter.bar(
                total=retained,
                desc=f"Token length QC pass {pass_index}",
                unit="entity",
            ) as progress:
                for row in connection.execute(
                    """
                    SELECT id, canonical_smiles, reasons FROM records
                    WHERE reasons = '[]' ORDER BY id
                    """
                ):
                    token_count = tokenizer.token_count(row["canonical_smiles"])
                    reasons = json.loads(row["reasons"])
                    if token_count > config.data.max_smiles_tokens:
                        reasons.append("smiles_overlength")
                        newly_excluded += 1
                    updates.append((token_count, json.dumps(reasons), int(row["id"])))
                    if len(updates) >= 10000:
                        connection.executemany(
                            "UPDATE records SET token_count = ?, reasons = ? WHERE id = ?",
                            updates,
                        )
                        connection.commit()
                        updates.clear()
                    progress.update(1)
                if updates:
                    connection.executemany(
                        "UPDATE records SET token_count = ?, reasons = ? WHERE id = ?",
                        updates,
                    )
                    connection.commit()
            if newly_excluded == 0:
                break

    _validate_role_splits(connection)
    for role in ROLE_TO_ID:
        sample_id_updates: list[tuple[str, int]] = []
        for number, row in enumerate(
            connection.execute(
                """
                SELECT id FROM records WHERE role = ? AND reasons = '[]'
                ORDER BY CASE split WHEN 'valid' THEN 0 ELSE 1 END,
                         is_augmented, canonical_smiles
                """,
                (role,),
            ),
            start=1,
        ):
            sample_id_updates.append((f"{role}_{number:08d}", int(row["id"])))
            if len(sample_id_updates) >= 10000:
                connection.executemany(
                    "UPDATE records SET sample_id = ? WHERE id = ?",
                    sample_id_updates,
                )
                sample_id_updates.clear()
        if sample_id_updates:
            connection.executemany(
                "UPDATE records SET sample_id = ? WHERE id = ?",
                sample_id_updates,
            )
    descriptor_updates: list[tuple[int, int]] = []
    for descriptor_row, row in enumerate(
        connection.execute(
            """
            SELECT id FROM records WHERE reasons = '[]'
            ORDER BY CASE split WHEN 'train' THEN 0 ELSE 1 END,
                     role_id, sample_id
            """
        )
    ):
        descriptor_updates.append((descriptor_row, int(row["id"])))
        if len(descriptor_updates) >= 10000:
            connection.executemany(
                "UPDATE records SET descriptor_row = ? WHERE id = ?",
                descriptor_updates,
            )
            descriptor_updates.clear()
    if descriptor_updates:
        connection.executemany(
            "UPDATE records SET descriptor_row = ? WHERE id = ?",
            descriptor_updates,
        )
    connection.commit()
    connection.execute(
        "CREATE INDEX IF NOT EXISTS records_token_count "
        "ON records(reasons, token_count)"
    )
    connection.commit()
    return tokenizer, {
        "entity_qc": _performance_phase(total, entity_elapsed, False),
        "tokenizer": _performance_phase(
            retained, time.perf_counter() - tokenizer_started, False
        ),
    }


def _quality_control_summary(
    connection: sqlite3.Connection,
    tokenizer: SmilesTokenizer,
    max_smiles_tokens: int,
) -> dict[str, Any]:
    total = int(connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])
    excluded_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM records WHERE reasons != '[]'"
        ).fetchone()[0]
    )
    role_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for row in connection.execute(
        "SELECT role, split, sources, reasons FROM records WHERE reasons != '[]'"
    ):
        role_counts[row["role"]] += 1
        split_counts[row["split"]] += 1
        source_counts.update(json.loads(row["sources"]))
        reason_counts.update(json.loads(row["reasons"]))

    def summarized(counts: Counter[str]) -> dict[str, Any]:
        return {
            value: {
                "count": count,
                "percent_of_pre_filter": 100.0 * count / total,
            }
            for value, count in sorted(counts.items())
        }

    return {
        "pre_filter_total": total,
        "post_filter_total": total - excluded_count,
        "thresholds": {
            "bcut_supported_bond_types": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
            "ipc_square_overflow_abs": IPC_SQUARE_OVERFLOW_LIMIT,
            "tokenizer_backend": tokenizer.backend,
            "max_smiles_tokens": max_smiles_tokens,
        },
        "excluded": {
            "total": excluded_count,
            "percent_of_pre_filter": 100.0 * excluded_count / total,
            "by_role": summarized(role_counts),
            "by_split": summarized(split_counts),
            "by_source": summarized(source_counts),
            "by_reason": summarized(reason_counts),
        },
    }


def _write_excluded_entities(
    path: Path,
    connection: sqlite3.Connection,
    tokenizer: SmilesTokenizer,
    max_smiles_tokens: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "canonical_smiles",
                "role",
                "split",
                "is_augmented",
                "sources",
                "exclusion_reasons",
                "unsupported_bond_types",
                "ipc",
                "tokenizer_backend",
                "token_count",
                "max_smiles_tokens",
            ],
        )
        writer.writeheader()
        for row in connection.execute(
            "SELECT * FROM records WHERE reasons != '[]' ORDER BY id"
        ):
            writer.writerow(
                {
                    "canonical_smiles": row["canonical_smiles"],
                    "role": row["role"],
                    "split": row["split"],
                    "is_augmented": row["is_augmented"],
                    "sources": ";".join(json.loads(row["sources"])),
                    "exclusion_reasons": ";".join(json.loads(row["reasons"])),
                    "unsupported_bond_types": ";".join(
                        json.loads(row["unsupported_bond_types"])
                    ),
                    "ipc": row["ipc"],
                    "tokenizer_backend": tokenizer.backend,
                    "token_count": row["token_count"] or "",
                    "max_smiles_tokens": max_smiles_tokens,
                }
            )
    temporary.replace(path)


def _write_manifest(path: Path, connection: sqlite3.Connection) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "role",
                "canonical_smiles",
                "split",
                "sources",
                "is_augmented",
                "seed_smiles",
            ],
        )
        writer.writeheader()
        for row in connection.execute(
            """
            SELECT * FROM records WHERE reasons = '[]'
            ORDER BY role_id,
                     CASE split WHEN 'valid' THEN 0 ELSE 1 END,
                     is_augmented, canonical_smiles
            """
        ):
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "role": row["role"],
                    "canonical_smiles": row["canonical_smiles"],
                    "split": row["split"],
                    "sources": ";".join(json.loads(row["sources"])),
                    "is_augmented": row["is_augmented"],
                    "seed_smiles": ";".join(json.loads(row["seed_smiles"])),
                }
            )
    temporary.replace(path)


def _descriptor_matrix(
    connection: sqlite3.Connection,
    raw_names: tuple[str, ...],
    output_dir: Path,
    signature: str,
    reporter: ProgressReporter,
    config: PretrainConfig,
) -> tuple[np.memmap, dict[str, float | int | bool]]:
    started = time.perf_counter()
    total = int(
        connection.execute(
            "SELECT COUNT(*) FROM records WHERE reasons = '[]'"
        ).fetchone()[0]
    )
    path = output_dir / ".raw_descriptors.npy"
    state_path = output_dir / "preparation_state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {}
    )
    completed = int(state.get("descriptor_completed", 0))
    reuse = (
        state.get("preparation_signature") == signature
        and path.is_file()
        and 0 <= completed <= total
    )
    if reuse:
        matrix = np.load(path, mmap_mode="r+")
        if matrix.shape != (total, len(raw_names)):
            reuse = False
    if not reuse:
        matrix = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float64, shape=(total, len(raw_names))
        )
        completed = 0
    initial_completed = completed
    with reporter.bar(
        total=total,
        initial=completed,
        desc="Descriptors",
        unit="entity",
    ) as progress:
        inputs = (
            (int(row["descriptor_row"]), row["canonical_smiles"])
            for row in connection.execute(
                """
                SELECT canonical_smiles, descriptor_row FROM records
                WHERE reasons = '[]' AND descriptor_row >= ?
                ORDER BY descriptor_row
                """,
                (completed,),
            )
        )
        tasks = (
            (batch, raw_names)
            for batch in _batches(inputs, config.preparation.descriptor_batch_size)
        )
        durable_completed = completed
        for descriptor_rows, values in _ordered_batch_map(
            _descriptor_batch, tasks, config.preparation.workers
        ):
            matrix[descriptor_rows] = values
            completed = descriptor_rows[-1] + 1
            progress.update(len(descriptor_rows))
            if completed - durable_completed >= 10000 or completed == total:
                matrix.flush()
                atomic_json(
                    state_path,
                    {
                        "format_version": 1,
                        "kind": CORPUS_KIND,
                        "preparation_signature": signature,
                        "phase": "descriptors",
                        "descriptor_completed": completed,
                    },
                )
                durable_completed = completed
                reporter.emit_json(
                    {
                        "event": "prepare_descriptors",
                        "completed": completed,
                        "total": total,
                    }
                )
    matrix.flush()
    return matrix, _performance_phase(
        total - initial_completed,
        time.perf_counter() - started,
        initial_completed == total,
    )


def _stage1_shard_sample(sample: dict[str, Any]) -> dict[str, Any]:
    result = {
        "sample_id": sample["sample_id"],
        "role_id": sample["role_id"],
        "token_ids": sample["token_ids"],
        "atom_categorical": sample["atom_categorical"],
        "atom_continuous": sample["atom_continuous"],
        "bond_categorical": sample["bond_categorical"],
        "bond_index": sample["bond_index"],
        "descriptors": sample["descriptors"],
        "descriptor_valid": sample["descriptor_valid"],
    }
    if "fingerprints" in sample:
        result["fingerprints"] = {
            name: value.to(torch.uint8)
            for name, value in sample["fingerprints"].items()
        }
    return result


def _write_shards(
    connection: sqlite3.Connection,
    raw_matrix: np.ndarray,
    schema: DescriptorSchema,
    standardizer: DescriptorStandardizer,
    tokenizer: SmilesTokenizer,
    config: PretrainConfig,
    output_dir: Path,
    signature: str,
    reporter: ProgressReporter,
) -> tuple[list[dict[str, Any]], int, dict[str, float | int | bool]]:
    corpus_format_version = (
        GLOBAL_RDKIT_CORPUS_FORMAT_VERSION
        if config.is_global_rdkit
        else CORPUS_FORMAT_VERSION
    )
    started = time.perf_counter()
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        split: int(
            connection.execute(
                "SELECT COUNT(*) FROM records WHERE reasons = '[]' AND split = ?",
                (split,),
            ).fetchone()[0]
        )
        for split in ("train", "valid")
    }
    total_shards = sum(math.ceil(count / config.data.shard_size) for count in counts.values())
    shard_manifest: list[dict[str, Any]] = []
    unk_count = 0
    active_paths: set[Path] = set()
    all_reused = True
    with reporter.bar(total=total_shards, desc="Shards", unit="shard") as progress:
        for split in ("train", "valid"):
            index_path = output_dir / f"{split}_index.npy"
            temporary_index_path = output_dir / f".{split}_index.npy.tmp"
            compact = np.lib.format.open_memmap(
                temporary_index_path,
                mode="w+",
                dtype=INDEX_DTYPE,
                shape=(counts[split],),
            )
            cursor = connection.execute(
                """
                SELECT * FROM records
                WHERE reasons = '[]' AND split = ?
                ORDER BY mix_key, canonical_smiles
                """,
                (split,),
            )
            logical_offset = 0
            shard_number = 0
            while True:
                rows = cursor.fetchmany(config.data.shard_size)
                if not rows:
                    break
                filename = f"{split}_{shard_number:05d}.pt"
                path = shard_dir / filename
                active_paths.add(path)
                records = [_record_from_row(row) for row in rows]
                expected_ids = [record["sample_id"] for record in records]
                samples: list[dict[str, Any]] | None = None
                if path.is_file():
                    existing = torch.load(path, map_location="cpu", weights_only=False)
                    if (
                        existing.get("kind") == CORPUS_SHARD_KIND
                        and existing.get("format_version") == corpus_format_version
                        and existing.get("preparation_signature") == signature
                        and [item["sample_id"] for item in existing.get("samples", [])]
                        == expected_ids
                    ):
                        samples = existing["samples"]
                reused = samples is not None
                all_reused = all_reused and reused
                if samples is None:
                    samples = [
                        _stage1_shard_sample(
                            build_entity_sample(
                                record,
                                raw_matrix[int(row["descriptor_row"])],
                                schema,
                                standardizer,
                                tokenizer,
                                config,
                            )
                        )
                        for row, record in zip(rows, records, strict=True)
                    ]
                    temporary = path.with_suffix(".pt.tmp")
                    torch.save(
                        {
                            "kind": CORPUS_SHARD_KIND,
                            "format_version": corpus_format_version,
                            "preparation_signature": signature,
                            "samples": samples,
                        },
                        temporary,
                    )
                    temporary.replace(path)
                shard_id = len(shard_manifest)
                shard_manifest.append(
                    {
                        "path": f"shards/{filename}",
                        "split": split,
                        "count": len(samples),
                        "sha256": sha256_file(path),
                    }
                )
                for offset, (row, sample) in enumerate(
                    zip(rows, samples, strict=True)
                ):
                    compact[logical_offset + offset] = (
                        shard_id,
                        offset,
                        int(row["role_id"]),
                    )
                    unk_count += int(
                        (sample["token_ids"] == tokenizer.unk_id).sum().item()
                    )
                logical_offset += len(samples)
                reporter.emit_json(
                    {
                        "event": "prepare_shard",
                        "shard": filename,
                        "samples": len(samples),
                        "reused": reused,
                    }
                )
                progress.update(1)
                shard_number += 1
            compact.flush()
            del compact
            temporary_index_path.replace(index_path)
    for stale in shard_dir.glob("*.pt"):
        if stale not in active_paths:
            stale.unlink()
    for temporary in shard_dir.glob("*.pt.tmp"):
        temporary.unlink()
    atomic_json(
        output_dir / "shard_manifest.json",
        {
            "kind": CORPUS_KIND,
            "format_version": corpus_format_version,
            "shards": shard_manifest,
        },
    )
    return (
        shard_manifest,
        unk_count,
        _performance_phase(
            sum(counts.values()), time.perf_counter() - started, all_reused
        ),
    )


def _tokenizer_statistics(
    connection: sqlite3.Connection, unk_count: int
) -> dict[str, float | int]:
    row = connection.execute(
        """
        SELECT COUNT(*) AS count, MIN(token_count) AS minimum,
               MAX(token_count) AS maximum, AVG(token_count) AS mean
        FROM records WHERE reasons = '[]'
        """
    ).fetchone()
    count = int(row["count"])

    def percentile(percent: float) -> float:
        position = (count - 1) * percent / 100.0
        lower = math.floor(position)
        upper = math.ceil(position)
        values = [
            int(item["token_count"])
            for item in connection.execute(
                """
                SELECT token_count FROM records WHERE reasons = '[]'
                ORDER BY token_count LIMIT ? OFFSET ?
                """,
                (upper - lower + 1, lower),
            )
        ]
        if lower == upper:
            return float(values[0])
        return values[0] + (values[1] - values[0]) * (position - lower)

    return {
        "min_length": int(row["minimum"]),
        "max_length": int(row["maximum"]),
        "mean_length": float(row["mean"]),
        "p50_length": percentile(50),
        "p90_length": percentile(90),
        "p95_length": percentile(95),
        "p99_length": percentile(99),
        "unk_count": unk_count,
    }


def _preparation_signature(
    corpus_identity: dict[str, Any], shard_size: int
) -> str:
    return canonical_json_sha256(
        {
            "signature_version": 2,
            "corpus_identity": corpus_identity["hash"],
            "shard_size": shard_size,
        }
    )


def prepare_corpus(
    config: PretrainConfig | DataConfig,
    *,
    source_identity: dict[str, object] | None = None,
    performance_path: Path | None = None,
    input_identity_elapsed_seconds: float = 0.0,
) -> dict[str, int]:
    prepare_started = time.perf_counter()
    phases: dict[str, dict[str, float | int | bool]] = {}
    if isinstance(config, DataConfig):
        config = PretrainConfig(data=config)
    config.validate()
    data_config = config.data
    corpus_format_version = (
        GLOBAL_RDKIT_CORPUS_FORMAT_VERSION
        if config.is_global_rdkit
        else CORPUS_FORMAT_VERSION
    )
    output_dir = data_config.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter()
    raw_names = rdkit_descriptor_names()
    if len(raw_names) != data_config.descriptor_dim:
        raise ValueError(
            f"Configured descriptor_dim={data_config.descriptor_dim}, but this RDKit "
            f"provides {len(raw_names)} descriptors"
        )

    source_paths = preparation_source_paths(data_config)
    identity_started = time.perf_counter()
    with reporter.status("Hash/count input files"):
        source_measurements, source_row_count = _source_measurements(
            data_config, source_paths, source_identity
        )
    source_hashes = {
        relative: str(item["sha256"])
        for relative, item in source_measurements.items()
    }
    identity_elapsed = (
        input_identity_elapsed_seconds
        if source_identity is not None
        else time.perf_counter() - identity_started
    )
    phases["input_identity"] = _performance_phase(
        source_row_count, identity_elapsed, False
    )
    if source_identity is None:
        locator = {
            relative: relative for relative in sorted(source_measurements)
        }
        source_identity = {
            "schema_version": 2,
            "identity_contract_version": IDENTITY_CONTRACT_VERSION,
            "stage": "stage1",
            "locator": {"files": locator},
            "semantic": {
                "identities": {
                    "source": semantic_identity(
                        "stage1.source-data",
                        {
                            "stage": "stage1",
                            "sources": source_measurements,
                        },
                    )
                }
            },
            "integrity": {
                "files": {
                    relative: {
                        "sha256": item["sha256"],
                        "size": item["size"],
                    }
                    for relative, item in sorted(source_measurements.items())
                }
            },
            "provenance": {},
        }
    corpus_identity = build_stage1_corpus_identity(config, source_identity)

    def flush_performance() -> None:
        _write_performance(
            performance_path,
            config,
            phases,
            time.perf_counter() - prepare_started
            + (input_identity_elapsed_seconds if source_identity is not None else 0.0),
        )

    flush_performance()
    signature = _preparation_signature(corpus_identity, data_config.shard_size)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("identity_contract_version") != IDENTITY_CONTRACT_VERSION:
            raise ValueError(
                "Stage 1 corpus predates identity contract v1; archive it and regenerate"
            )
        if (
            existing.get("kind") == CORPUS_KIND
            and existing.get("format_version") == corpus_format_version
            and existing.get("preparation_signature") == signature
        ):
            reused_started = time.perf_counter()
            PreparedCorpusDataset(output_dir, "train", data_config.shard_cache_size)
            PreparedCorpusDataset(output_dir, "valid", data_config.shard_cache_size)
            summary = {
                key: int(value) for key, value in existing["summary"].items()
            }
            total = summary["total"]
            for name in (
                "catalog",
                "entity_qc",
                "tokenizer",
                "descriptors",
                "descriptor_fit",
                "shards",
            ):
                phases[name] = _performance_phase(total, 0.0, True)
            phases["publication"] = _performance_phase(
                total, time.perf_counter() - reused_started, True
            )
            flush_performance()
            return summary
        metadata_path.unlink()

    catalog_started = time.perf_counter()
    catalog_path = output_dir / ".prepare.sqlite"
    reuse_catalog = False
    if catalog_path.is_file():
        connection = _connect_catalog(catalog_path)
        try:
            reuse_catalog = (
                _catalog_metadata(connection, "signature") == signature
                and _catalog_metadata(connection, "phase") in {"catalog", "qc"}
                and isinstance(
                    _catalog_metadata(connection, "augmentation_audit"), dict
                )
            )
        except sqlite3.DatabaseError:
            reuse_catalog = False
        if not reuse_catalog:
            connection.close()
    if not reuse_catalog:
        for suffix in ("", "-wal", "-shm"):
            Path(str(catalog_path) + suffix).unlink(missing_ok=True)
        connection = _connect_catalog(catalog_path)
        _create_catalog(connection, signature)
        augmentation_audit = _build_catalog(
            connection, config, source_row_count, reporter
        )
    else:
        augmentation_audit = _catalog_metadata(connection, "augmentation_audit")
    catalog_count = int(
        connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    )
    phases["catalog"] = _performance_phase(
        source_row_count if not reuse_catalog else catalog_count,
        time.perf_counter() - catalog_started,
        reuse_catalog,
    )
    flush_performance()

    phase = _catalog_metadata(connection, "phase")
    tokenizer_path = output_dir / "tokenizer.json"
    if phase == "catalog" or not tokenizer_path.is_file():
        tokenizer, qc_phases = _run_qc_and_tokenizer(connection, config, reporter)
        phases.update(qc_phases)
        tokenizer.save(tokenizer_path)
        _set_catalog_metadata(connection, "phase", "qc")
    else:
        tokenizer = SmilesTokenizer.load(tokenizer_path)
        phases["entity_qc"] = _performance_phase(catalog_count, 0.0, True)
        phases["tokenizer"] = _performance_phase(catalog_count, 0.0, True)
    flush_performance()

    quality_control = _quality_control_summary(
        connection, tokenizer, data_config.max_smiles_tokens
    )
    _write_excluded_entities(
        output_dir / "excluded_entities.csv",
        connection,
        tokenizer,
        data_config.max_smiles_tokens,
    )
    for role, stats in augmentation_audit.items():
        retained = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM records
                WHERE role = ? AND is_augmented = 1 AND reasons = '[]'
                """,
                (role,),
            ).fetchone()[0]
        )
        stats["retained"] = retained
        stats["excluded_qc"] = int(stats["eligible"]) - retained
    atomic_json(
        output_dir / "augmentation_audit.json",
        {
            "kind": CORPUS_KIND,
            "format_version": corpus_format_version,
            "include_augmentation": data_config.include_augmentation,
            "roles": augmentation_audit,
        },
    )

    raw_matrix, descriptor_performance = _descriptor_matrix(
        connection, raw_names, output_dir, signature, reporter, config
    )
    phases["descriptors"] = descriptor_performance
    flush_performance()
    train_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM records WHERE reasons = '[]' AND split = 'train'"
        ).fetchone()[0]
    )
    descriptor_fit_started = time.perf_counter()
    with reporter.status("Descriptor schema/scaler"):
        training_matrix = raw_matrix[:train_count]
        schema = DescriptorSchema.fit(
            training_matrix,
            raw_names,
            mode=config.descriptor.mode,
            token_count=config.descriptor.token_count,
            correlation_threshold=config.descriptor.correlation_threshold,
        )
        standardizer = DescriptorStandardizer.fit_columns(
            training_matrix,
            schema.retained_indices,
            schema.selected_names,
        )
        fitted = standardizer.finite_counts > 0
        invalid = fitted & (
            ~np.isfinite(standardizer.means)
            | ~np.isfinite(standardizer.scales)
            | (standardizer.scales <= 0.0)
        )
        if invalid.any():
            names = [
                standardizer.names[index]
                for index in np.flatnonzero(invalid).tolist()
            ]
            raise ValueError(
                "Non-finite descriptor standardization statistics: "
                + ", ".join(names)
            )
        schema.save(output_dir / "descriptor_schema.json")
        standardizer.save(output_dir / "descriptor_scaler.json")
    phases["descriptor_fit"] = _performance_phase(
        train_count, time.perf_counter() - descriptor_fit_started, False
    )
    flush_performance()

    _, unk_count, shard_performance = _write_shards(
        connection,
        raw_matrix,
        schema,
        standardizer,
        tokenizer,
        config,
        output_dir,
        signature,
        reporter,
    )
    phases["shards"] = shard_performance
    flush_performance()
    publication_started = time.perf_counter()
    _write_manifest(output_dir / "manifest.csv", connection)
    summary = {
        "total": int(
            connection.execute(
                "SELECT COUNT(*) FROM records WHERE reasons = '[]'"
            ).fetchone()[0]
        ),
        "train": train_count,
        "valid": int(
            connection.execute(
                "SELECT COUNT(*) FROM records WHERE reasons = '[]' AND split = 'valid'"
            ).fetchone()[0]
        ),
        **{
            role: int(
                connection.execute(
                    "SELECT COUNT(*) FROM records WHERE reasons = '[]' AND role = ?",
                    (role,),
                ).fetchone()[0]
            )
            for role in ROLE_TO_ID
        },
        "augmented": int(
            connection.execute(
                """
                SELECT COUNT(*) FROM records
                WHERE reasons = '[]' AND is_augmented = 1
                """
            ).fetchone()[0]
        ),
        "excluded_entities": quality_control["excluded"]["total"],
        "descriptor_dim": schema.selected_dim,
    }
    artifact_files = (
        "tokenizer.json",
        "descriptor_schema.json",
        "descriptor_scaler.json",
        "train_index.npy",
        "valid_index.npy",
        "shard_manifest.json",
        "manifest.csv",
        "excluded_entities.csv",
        "augmentation_audit.json",
    )
    shard_manifest = json.loads(
        (output_dir / "shard_manifest.json").read_text(encoding="utf-8")
    )["shards"]
    sampler_layout_identity = build_stage1_sampler_layout_identity(
        shard_manifest, shard_size=data_config.shard_size
    )
    metadata = {
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "kind": CORPUS_KIND,
        "format_version": corpus_format_version,
        "preparation_signature": signature,
        "rdkit_version": rdBase.rdkitVersion,
        "atom_in_smiles_version": importlib.metadata.version("atomInSmiles"),
        "descriptor_raw_names": list(raw_names),
        "descriptor_names": list(schema.selected_names),
        "descriptor_dim": schema.selected_dim,
        "descriptor_mode": schema.mode,
        "descriptor_token_count": schema.token_count,
        "tokenizer_backend": tokenizer.backend,
        "tokenizer_backend_version": tokenizer.backend_version,
        "tokenizer_budget": config.tokenizer.vocab_size,
        "tokenizer_actual_size": len(tokenizer.tokens),
        "max_smiles_tokens": data_config.max_smiles_tokens,
        "tokenizer_statistics": _tokenizer_statistics(connection, unk_count),
        "feature_generation_contract": feature_generation_contract(config),
        "role_source_files": ROLE_SOURCE_FILES,
        "ignored_stage1_files": [
            "simulation_mol.csv",
            "solute.csv",
            "solvent.csv",
            "IL.csv",
        ],
        "include_augmentation": data_config.include_augmentation,
        "augmentation_audit": augmentation_audit,
        "quality_control": quality_control,
        "source_hashes": source_hashes,
        "seed": data_config.seed,
        "valid_fraction": data_config.valid_fraction,
        "shard_hashes": {
            item["path"]: item["sha256"]
            for item in shard_manifest
        },
        "summary": summary,
        "artifact_hashes": {
            filename: sha256_file(output_dir / filename)
            for filename in artifact_files
        },
    }
    if not config.is_global_rdkit:
        metadata.update(
            {
                "fingerprint_kind": config.fingerprint.kind,
                "fingerprint_contract": config.to_dict()["fingerprint"],
            }
        )
    feature_identity = build_stage1_feature_identity(output_dir, metadata)
    all_integrity = {
        **{
            filename: {
                "sha256": metadata["artifact_hashes"][filename],
                "size": (output_dir / filename).stat().st_size,
            }
            for filename in artifact_files
        },
        **{
            item["path"]: {
                "sha256": item["sha256"],
                "size": (output_dir / item["path"]).stat().st_size,
            }
            for item in shard_manifest
        },
    }
    metadata.update(
        {
            "locator": {"files": {name: name for name in all_integrity}},
            "semantic": {
                "identities": {
                    "corpus": corpus_identity,
                    "sampler_layout": sampler_layout_identity,
                    "feature": feature_identity,
                }
            },
            "integrity": {"files": all_integrity},
            "provenance": {
                "rdkit_version": metadata["rdkit_version"],
                "atom_in_smiles_version": metadata["atom_in_smiles_version"],
            },
        }
    )
    connection.close()
    for suffix in ("", "-wal", "-shm"):
        Path(str(catalog_path) + suffix).unlink(missing_ok=True)
    (output_dir / ".raw_descriptors.npy").unlink(missing_ok=True)
    (output_dir / "corpus_index.json").unlink(missing_ok=True)
    (output_dir / "preparation_state.json").unlink(missing_ok=True)
    atomic_json(metadata_path, metadata)
    phases["publication"] = _performance_phase(
        summary["total"], time.perf_counter() - publication_started, False
    )
    flush_performance()
    return summary
