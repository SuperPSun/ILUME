from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ablations.stage3_full_finetune.evaluate import evaluate_finetuned
from ablations.stage3_full_finetune.representation import load_config, prepare_features
from ablations.stage3_full_finetune.train import run_finetuning
from common.io import atomic_json
from stage3.config import configure_process_runtime, validate_stage3_folds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated Stage 3 Stage1/ObjectEncoder fine-tuning ablation."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Build immutable molecular input features.")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--output", required=True)
    train = commands.add_parser("train", help="Train selected folds with live Phase 1 encoders.")
    train.add_argument("--config", required=True)
    train.add_argument("--feature-dir", required=True)
    train.add_argument("--fold", type=int, nargs="+", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Evaluate final fine-tuned folds.")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--feature-dir", required=True)
    evaluate.add_argument("--checkpoint-dir", required=True)
    evaluate.add_argument("--split", choices=("valid", "test"), required=True)
    evaluate.add_argument("--fold", type=int, nargs="+")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--historical-base-root", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config, recipe = load_config(args.config)
    configure_process_runtime(config)
    if args.command == "prepare":
        prepare_features(config, args.output)
    elif args.command == "train":
        folds = validate_stage3_folds(args.fold)
        root = Path(args.output)
        if root.exists() and not args.resume:
            raise FileExistsError(f"Fine-tuning output root already exists: {root}")
        if args.resume and not root.is_dir():
            raise FileNotFoundError(f"Fine-tuning resume root is missing: {root}")
        for fold in folds:
            run_finetuning(
                config, recipe, fold=fold, feature_dir=args.feature_dir,
                output_dir=root / f"fold{fold}", resume=args.resume,
            )
    else:
        if args.split == "valid":
            folds = validate_stage3_folds(args.fold or ())
        elif args.fold is not None:
            raise ValueError("Test evaluation always uses all five folds; omit --fold")
        else:
            folds = (None,)
        root = Path(args.output)
        if root.exists():
            raise FileExistsError(f"Fine-tuning evaluation output already exists: {root}")
        root.mkdir(parents=True)
        for fold in folds:
            destination = root / (f"fold{fold}" if fold is not None else "test")
            destination.mkdir()
            result = evaluate_finetuned(
                config, recipe, feature_dir=args.feature_dir,
                checkpoint_dir=args.checkpoint_dir,
                split=args.split, fold=fold,
                predictions_dir=destination / "predictions",
                historical_base_root=args.historical_base_root,
            )
            atomic_json(destination / "summary.json", result)


if __name__ == "__main__":
    main()
