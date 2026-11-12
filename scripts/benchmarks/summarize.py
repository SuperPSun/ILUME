from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.summary import publish_summary
from common.outputs import repository_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic ILUME and baseline result leaderboards."
    )
    parser.add_argument(
        "--input", nargs="+", metavar="PATH",
        help="One or more repository-relative directories to scan recursively.",
    )
    parser.add_argument(
        "--include", nargs="+", metavar="PATH",
        help=(
            "Optional exact repository-relative directory prefixes to include; "
            "each must be inside an input directory."
        ),
    )
    parser.add_argument(
        "--exclude", nargs="+", metavar="PATH",
        help="Optional repository-relative directories inside an input to exclude recursively.",
    )
    parser.add_argument("--output", default="summary")
    parser.add_argument("--baseline-config", help="Assemble the complete audited 22-task baseline report from a dedicated YAML.")
    parser.add_argument("--ablation-analysis-output", help="Write a separate v4 ablation comparison; neither output may already exist.")
    parser.add_argument("--ablation-bootstrap", action="store_true", help="Add fixed-seed 10000-repeat paired system bootstrap intervals.")
    parser.add_argument("--ablation-authority-config", default="configs/v4/stage3/base.yaml", help="Paired Stage3 config for ablation groups, task sizes and bootstrap sources.")
    parser.add_argument("--ablation-base-run", help="Exact comparable run ID to use as the ablation reference, such as I4 for S1.")
    args = parser.parse_args()
    if args.baseline_config:
        if args.input or args.include or args.exclude or args.ablation_analysis_output or args.ablation_bootstrap or args.ablation_base_run:
            parser.error("--baseline-config cannot be combined with input filters or ablation analysis")
        from benchmarks.common.baseline_report import publish_baseline_report
        publish_baseline_report(repository_path(args.baseline_config), repository_path(args.output), ROOT)
        return
    if not args.input:
        parser.error("--input is required unless --baseline-config is provided")
    if args.ablation_bootstrap and not args.ablation_analysis_output:
        parser.error("--ablation-bootstrap requires --ablation-analysis-output")
    if args.ablation_base_run and not args.ablation_analysis_output:
        parser.error("--ablation-base-run requires --ablation-analysis-output")
    input_roots = [repository_path(path) for path in args.input]
    include_roots = [repository_path(path) for path in (args.include or ())]
    exclude_roots = [repository_path(path) for path in (args.exclude or ())]
    output = repository_path(args.output)
    analysis_output = repository_path(args.ablation_analysis_output) if args.ablation_analysis_output else None
    if analysis_output is not None and (output.exists() or analysis_output.exists()):
        raise FileExistsError("New ablation reports cannot overwrite existing output")
    payload = publish_summary(
        input_roots, output, ROOT,
        include_roots=include_roots, exclude_roots=exclude_roots,
    )
    if analysis_output is not None:
        from ablations.entity_home.report import write_analysis
        write_analysis(payload, repository_path(args.ablation_authority_config), analysis_output,
                       base_run=args.ablation_base_run, input_roots=input_roots, repository_root=ROOT, bootstrap=args.ablation_bootstrap)


if __name__ == "__main__":
    main()
