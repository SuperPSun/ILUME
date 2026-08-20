from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Iterable, Mapping

from .identity import IDENTITY_CONTRACT_VERSION, semantic_identity
from .io import atomic_json, sha256_file


def _row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def _source_repository_state(repository_root: Path) -> tuple[str | None, bool | None]:
    source = repository_root.parent / "ILUME-Data"
    if not (source / ".git").exists():
        return None, None
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return commit, dirty


def write_data_identity(
    repository_root: Path,
    stage: str,
    paths: Iterable[Path] | Mapping[str, Path],
) -> dict[str, object]:
    if isinstance(paths, Mapping):
        entries = [(str(logical_id), path.resolve()) for logical_id, path in paths.items()]
    else:
        entries = []
        seen: set[Path] = set()
        for path in paths:
            resolved = path.resolve()
            if resolved not in seen:
                entries.append((f"source_{len(entries):05d}", resolved))
                seen.add(resolved)
    locator: dict[str, str] = {}
    integrity: dict[str, dict[str, object]] = {}
    semantic_sources: dict[str, dict[str, object]] = {}
    for logical_id, path in entries:
        try:
            relative = path.relative_to(repository_root.resolve()).as_posix()
        except ValueError as error:
            raise ValueError(f"Data input must be inside the repository: {path}") from error
        digest = sha256_file(path)
        size = path.stat().st_size
        rows = _row_count(path)
        locator[logical_id] = relative
        integrity[logical_id] = {"sha256": digest, "size": size}
        semantic_sources[logical_id] = {
            "sha256": digest,
            "size": size,
            "rows": rows,
        }
    commit, dirty = _source_repository_state(repository_root)
    payload: dict[str, object] = {
        "schema_version": 2,
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "stage": stage,
        "locator": {"files": locator},
        "semantic": {
            "identities": {
                "source": semantic_identity(
                    f"{stage}.source-data",
                    {"stage": stage, "sources": semantic_sources},
                )
            }
        },
        "integrity": {"files": integrity},
        "provenance": {
            "source_repository_commit": commit,
            "source_repository_dirty": dirty,
        },
    }
    output = repository_root / "data" / stage / "metadata.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    return payload
