from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stage1.descriptor_selection import load_selection_config, run_selection, selection_source


def main():
    parser = argparse.ArgumentParser(description="Audit RDKit descriptors on Stage1 train without changing models or artifacts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="New independent audit directory; existing outputs are rejected.")
    args = parser.parse_args()
    from common.outputs import open_run_directory

    config = load_selection_config(args.config)
    metadata, identity = selection_source(config)
    run = open_run_directory(stage="stage1", operation="descriptor_selection",
        config_path=args.config, config_payload=config.to_dict(), semantic_identity=identity,
        output=args.output, seed=config.seed,
        data_metadata=str(config.artifacts_dir / "metadata.json"))
    try:
        result = run_selection(config, run.artifacts, metadata, identity)
        run.complete(result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    except BaseException:
        run.fail()
        raise


if __name__ == "__main__":
    main()
