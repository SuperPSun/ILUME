from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.outputs import open_run_directory, repository_path
from stage1.config import load_config
from stage1.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 1.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_config(args.config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = config.training.device == "cuda" or (
        config.training.device == "auto" and torch.cuda.is_available()
    )
    if use_cuda:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    if world_size > 1:
        if use_cuda:
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    rank = dist.get_rank() if dist.is_initialized() else 0
    run = None
    try:
        open_error: str | None = None
        if rank == 0:
            try:
                run = open_run_directory(
                    stage="stage1",
                    operation="train",
                    config_path=args.config,
                    config_payload=config.to_dict(),
                    output=args.output,
                    seed=config.data.seed,
                    resume=args.resume,
                    ignored_config_sections={"preparation"},
                    ignored_config_fields={"training.compile"},
                    details={
                        "world_size": world_size,
                        "amp_dtype": config.training.amp_dtype,
                        "compile": config.training.compile,
                    },
                )
            except BaseException as error:
                open_error = f"{type(error).__name__}: {error}"
        if dist.is_initialized():
            payload = [open_error]
            dist.broadcast_object_list(payload, src=0)
            open_error = payload[0]
        if open_error is not None:
            raise RuntimeError(open_error)
        output_dir = run.root if run is not None else repository_path(args.output)
        rows = run_training(
            config, output_dir=output_dir,
            resume_from=repository_path(args.resume) if args.resume else None,
            attempt_id=run.metadata["attempt_id"] if run is not None else "worker",
        )
        if run is not None:
            run.complete(rows[-1] if rows else {"status": "already_complete"})
    except BaseException:
        if run is not None:
            run.fail()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
