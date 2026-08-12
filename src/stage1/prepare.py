from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from rdkit import Chem, rdBase

from common.io import atomic_json, sha256_file
from common.progress import ProgressReporter
from common.training import canonical_json_sha256
from .config import DataConfig, PretrainConfig
from .data import (
    CORPUS_FORMAT_VERSION,
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
from .tokenizer import SmilesTokenizer


ROLE_SOURCE_FILES = {
    "cation": "cation.csv",
    "anion": "anion.csv",
    "neutral": "molecule.csv",
}
_EMPTY_JSON = "[]"


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
        CREATE INDEX records_split ON records(split);
        CREATE INDEX records_mix ON records(split, mix_key, canonical_smiles);
        """
    )
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES('signature', ?)",
        (json.dumps(signature),),
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
    config: DataConfig,
    originals: dict[str, set[str]],
    validation: dict[str, set[str]],
    progress: Any,
) -> dict[str, dict[str, Any]]:
    audit: dict[str, dict[str, Any]] = {}
    for role, filename in ROLE_SOURCE_FILES.items():
        stats = _empty_augmentation_stats(config.include_augmentation)
        audit[role] = stats
        if not config.include_augmentation:
            continue
        path = config.stage1_dir / "augmentation" / filename
        pending = 0
        with path.open(newline="", encoding="utf-8") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                stats["source_rows"] += 1
                smiles = (row.get("SMILES") or "").strip()
                canonical = _canonicalize(smiles, f"{path}:{row_number}")
                seeds: list[str] = []
                for seed in _seed_values(
                    (row.get("seed_smiles_list") or "").strip()
                ):
                    try:
                        seeds.append(_canonicalize(seed, f"{path}:{row_number} seed"))
                    except ValueError:
                        seeds.append(seed)
                if any(seed in validation[role] for seed in seeds):
                    stats["excluded_valid_seed"] += 1
                elif canonical in originals[role]:
                    stats["excluded_overlap"] += 1
                else:
                    try:
                        connection.execute(
                            """
                            INSERT INTO records(
                                role, role_id, canonical_smiles, sources, split,
                                is_augmented, seed_smiles, mix_key
                            ) VALUES(?, ?, ?, ?, 'train', 1, ?, ?)
                            """,
                            (
                                role,
                                ROLE_TO_ID[role],
                                canonical,
                                json.dumps([f"augmentation/{path.stem}"]),
                                json.dumps(sorted(set(seeds))),
                                _mix_key(config.seed, "train", canonical),
                            ),
                        )
                        stats["eligible"] += 1
                        pending += 1
                    except sqlite3.IntegrityError:
                        stats["excluded_duplicate"] += 1
                if pending >= 10000:
                    connection.commit()
                    pending = 0
                progress.update(1)
        connection.commit()
    return audit


def _build_catalog(
    connection: sqlite3.Connection,
    config: DataConfig,
    source_row_count: int,
    reporter: ProgressReporter,
) -> dict[str, dict[str, Any]]:
    with reporter.bar(
        total=source_row_count, desc="Load/canonicalize", unit="row"
    ) as progress:
        originals, validation = _load_originals(connection, config, progress)
        audit = _load_augmentation(
            connection, config, originals, validation, progress
        )
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
) -> SmilesTokenizer:
    total = int(connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])
    connection.execute(
        """
        UPDATE records SET reasons = '[]', unsupported_bond_types = '[]',
                           ipc = NULL, token_count = NULL,
                           sample_id = NULL, descriptor_row = NULL
        """
    )
    with reporter.bar(total=total, desc="Entity QC", unit="entity") as progress:
        updates: list[tuple[str, str, str, int]] = []
        for row in connection.execute("SELECT * FROM records ORDER BY id"):
            inspected = inspect_entity_qc(_record_from_row(row))
            updates.append(
                (
                    json.dumps(inspected.reasons),
                    json.dumps(inspected.unsupported_bond_types),
                    repr(inspected.ipc),
                    int(row["id"]),
                )
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
            progress.update(1)
        if updates:
            connection.executemany(
                """
                UPDATE records SET reasons = ?, unsupported_bond_types = ?,
                                   ipc = ? WHERE id = ?
                """,
                updates,
            )
            connection.commit()

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
    return tokenizer


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
) -> np.memmap:
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
    with reporter.bar(
        total=total,
        initial=completed,
        desc="Descriptors",
        unit="entity",
    ) as progress:
        for row in connection.execute(
            """
            SELECT canonical_smiles, descriptor_row FROM records
            WHERE reasons = '[]' AND descriptor_row >= ?
            ORDER BY descriptor_row
            """,
            (completed,),
        ):
            mol = Chem.MolFromSmiles(row["canonical_smiles"])
            if mol is None:
                raise RuntimeError("Canonical SMILES unexpectedly failed RDKit parsing")
            descriptor_row = int(row["descriptor_row"])
            matrix[descriptor_row] = calculate_descriptors(mol, raw_names)
            completed = descriptor_row + 1
            progress.update(1)
            if completed % 10000 == 0 or completed == total:
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
                reporter.emit_json(
                    {
                        "event": "prepare_descriptors",
                        "completed": completed,
                        "total": total,
                    }
                )
    matrix.flush()
    return matrix


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
) -> tuple[list[dict[str, Any]], int]:
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
                        and existing.get("format_version") == CORPUS_FORMAT_VERSION
                        and existing.get("preparation_signature") == signature
                        and [item["sample_id"] for item in existing.get("samples", [])]
                        == expected_ids
                    ):
                        samples = existing["samples"]
                reused = samples is not None
                if samples is None:
                    samples = [
                        build_entity_sample(
                            record,
                            raw_matrix[int(row["descriptor_row"])],
                            schema,
                            standardizer,
                            tokenizer,
                            config,
                        )
                        for row, record in zip(rows, records, strict=True)
                    ]
                    temporary = path.with_suffix(".pt.tmp")
                    torch.save(
                        {
                            "kind": CORPUS_SHARD_KIND,
                            "format_version": CORPUS_FORMAT_VERSION,
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
            "format_version": CORPUS_FORMAT_VERSION,
            "shards": shard_manifest,
        },
    )
    return shard_manifest, unk_count


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
    config: PretrainConfig, source_hashes: dict[str, str]
) -> str:
    payload = config.to_dict()
    payload["data"].pop("artifacts_dir", None)
    payload["data"].pop("shard_cache_size", None)
    return canonical_json_sha256(
        {
            "signature_version": 1,
            "kind": CORPUS_KIND,
            "format_version": CORPUS_FORMAT_VERSION,
            "rdkit_version": rdBase.rdkitVersion,
            "data": payload["data"],
            "tokenizer": payload["tokenizer"],
            "descriptor": payload["descriptor"],
            "fingerprint": payload["fingerprint"],
            "source_hashes": source_hashes,
        }
    )


def prepare_corpus(config: PretrainConfig | DataConfig) -> dict[str, int]:
    if isinstance(config, DataConfig):
        config = PretrainConfig(data=config)
    config.validate()
    data_config = config.data
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
    with reporter.status("Hash/count input files"):
        source_hashes = {
            str(path.relative_to(data_config.stage1_dir)): sha256_file(path)
            for path in source_paths
        }
        source_row_count = sum(_csv_data_row_count(path) for path in source_paths)
    signature = _preparation_signature(config, source_hashes)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            existing.get("kind") == CORPUS_KIND
            and existing.get("format_version") == CORPUS_FORMAT_VERSION
            and existing.get("preparation_signature") == signature
        ):
            PreparedCorpusDataset(output_dir, "train", data_config.shard_cache_size)
            PreparedCorpusDataset(output_dir, "valid", data_config.shard_cache_size)
            return {key: int(value) for key, value in existing["summary"].items()}
        metadata_path.unlink()

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
            connection, data_config, source_row_count, reporter
        )
    else:
        augmentation_audit = _catalog_metadata(connection, "augmentation_audit")

    phase = _catalog_metadata(connection, "phase")
    tokenizer_path = output_dir / "tokenizer.json"
    if phase == "catalog" or not tokenizer_path.is_file():
        tokenizer = _run_qc_and_tokenizer(connection, config, reporter)
        tokenizer.save(tokenizer_path)
        _set_catalog_metadata(connection, "phase", "qc")
    else:
        tokenizer = SmilesTokenizer.load(tokenizer_path)

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
            "format_version": CORPUS_FORMAT_VERSION,
            "include_augmentation": data_config.include_augmentation,
            "roles": augmentation_audit,
        },
    )

    raw_matrix = _descriptor_matrix(
        connection, raw_names, output_dir, signature, reporter
    )
    train_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM records WHERE reasons = '[]' AND split = 'train'"
        ).fetchone()[0]
    )
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

    _, unk_count = _write_shards(
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
    metadata = {
        "kind": CORPUS_KIND,
        "format_version": CORPUS_FORMAT_VERSION,
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
        "fingerprint_kind": config.fingerprint.kind,
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
    connection.close()
    for suffix in ("", "-wal", "-shm"):
        Path(str(catalog_path) + suffix).unlink(missing_ok=True)
    (output_dir / ".raw_descriptors.npy").unlink(missing_ok=True)
    (output_dir / "corpus_index.json").unlink(missing_ok=True)
    (output_dir / "preparation_state.json").unlink(missing_ok=True)
    atomic_json(metadata_path, metadata)
    return summary
