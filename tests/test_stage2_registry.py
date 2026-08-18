from __future__ import annotations

import csv
from pathlib import Path

from stage2.registry import load_stage2_registry


FIELDS = (
    "catalog_schema_version", "stage", "task_id", "task_kind",
    "target_level", "source_file", "target_columns", "identity_columns",
    "condition_columns", "system_type", "simulation_method",
    "materialized_path", "label_source", "resource_manifest",
)


def _write_catalog(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(task_id: str, *, stage: int = 2) -> dict[str, object]:
    return {
        "catalog_schema_version": 1,
        "stage": stage,
        "task_id": task_id,
        "task_kind": "object_property",
        "target_level": "object",
        "source_file": f"simulation/{task_id.rsplit('/', 1)[-1]}.csv",
        "target_columns": "value",
        "identity_columns": "SMILES",
        "condition_columns": "",
        "system_type": "molecule",
        "simulation_method": "test",
        "materialized_path": f"stage2/{task_id.rsplit('/', 1)[-1]}",
        "label_source": "materialized_csv",
        "resource_manifest": "",
    }


def test_registry_accepts_new_semantic_task_without_python_whitelist(tmp_path: Path) -> None:
    path = tmp_path / "task_catalog.csv"
    _write_catalog(path, [_row("simulation/new_property")])
    registry = load_stage2_registry(path)
    assert registry.task_ids == ("simulation/new_property",)
    assert registry.tasks[0].topology == "single_entity"


def test_registry_hash_ignores_non_stage2_catalog_rows(tmp_path: Path) -> None:
    path = tmp_path / "task_catalog.csv"
    stage2 = _row("simulation/property")
    _write_catalog(path, [stage2])
    before = load_stage2_registry(path)
    _write_catalog(path, [stage2, _row("experiment/unrelated", stage=3)])
    after = load_stage2_registry(path)
    assert after.registry_hash == before.registry_hash
    assert after.catalog_sha256 != before.catalog_sha256
