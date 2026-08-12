from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path
from typing import Iterable

from .io import atomic_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    paths: Iterable[Path],
) -> dict[str, object]:
    unique_paths = sorted({path.resolve() for path in paths})
    files: list[dict[str, object]] = []
    for path in unique_paths:
        try:
            relative = path.relative_to(repository_root.resolve()).as_posix()
        except ValueError as error:
            raise ValueError(f"Data input must be inside the repository: {path}") from error
        files.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "size": path.stat().st_size,
                "rows": _row_count(path),
            }
        )
    commit, dirty = _source_repository_state(repository_root)
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": stage,
        "source_repository_commit": commit,
        "source_repository_dirty": dirty,
        "files": files,
    }
    output = repository_root / "data" / stage / "metadata.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    return payload

