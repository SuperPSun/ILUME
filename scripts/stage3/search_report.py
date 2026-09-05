from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a validation-only report for the ILUME v2 Stage 3 A/B/C search."
    )
    parser.add_argument(
        "--search-output", required=True, help="shared repository-relative v2 search output root"
    )
    parser.add_argument(
        "--output",
        help="new repository-relative report directory; defaults to SEARCH_OUTPUT/search_report",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    from common.identity import semantic_identity
    from common.outputs import open_run_directory, repository_path, repository_relative
    from stage3.search_report import build_search_report, write_search_report

    search_root = repository_path(args.search_output)
    output = (
        repository_path(args.output)
        if args.output is not None
        else search_root / "search_report"
    )
    payload = build_search_report(search_root)
    config_payload = {
        "schema_version": 1,
        "search_output": repository_relative(search_root),
        "source_sha256": payload["source_sha256"],
        "metric": payload["metric"],
        "task_order": payload["task_order"],
    }
    identity = semantic_identity(
        "stage3.v2-search-report",
        {"contract_version": 1, "report": config_payload},
    )
    run = open_run_directory(
        stage="stage3",
        operation="v2_search_report",
        config_path="configs/v2/stage3/search.yaml",
        config_payload=config_payload,
        semantic_identity=identity,
        output=repository_relative(output),
        seed=42,
        details={"search_output": repository_relative(search_root)},
    )
    try:
        write_search_report(payload, run.root)
        run.complete(payload)
    except BaseException:
        run.fail()
        raise
    print(f"Stage 3 search report complete: {repository_relative(run.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
