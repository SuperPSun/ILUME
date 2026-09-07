from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.identity import semantic_identity
from common.outputs import open_run_directory, repository_path
from stage3.capacity import summarize_capacity_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize ILUME capacity probe, robustness, or comparison runs."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = repository_path(args.manifest)
    import yaml

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    result = summarize_capacity_manifest(manifest_path)

    def semantic_payload(value):
        if isinstance(value, dict):
            return {
                key: semantic_payload(item)
                for key, item in value.items()
                if key not in {"fold_runs", "run_paths", "run_root"}
            }
        if isinstance(value, list):
            return [semantic_payload(item) for item in value]
        return value

    identity = semantic_identity(
        "stage3.capacity-report",
        {"contract_version": 1, "report": semantic_payload(result)},
    )
    run = open_run_directory(
        stage="stage3",
        operation="capacity_report",
        config_path=args.manifest,
        config_payload=manifest,
        semantic_identity=identity,
        output=args.output,
        seed=42,
        details={"capacity_report_kind": result["kind"]},
    )
    try:
        run.complete(result)
    except BaseException:
        run.fail()
        raise
    print(f"Capacity report complete: {args.output}")


if __name__ == "__main__":
    main()
