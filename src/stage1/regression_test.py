"""锁定后独立后训练终测；不拟合统计、不训练参数。"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from common.io import atomic_json, sha256_file
from common.reporting import write_prediction_csv
from common.training import resolve_device
from .config import config_from_dict
from .partial_charge import load_charge_test_rows, charge_test_source_contract
from .regression import (encode_regression_molecules, filter_regression_inputs, load_scalar_regression_rows,
                         load_regression_head, load_regression_rows, evaluate_regression_head)
from .features import load_stage1_feature_inputs

TEST_TASKS = ("HOMO_eV", "LUMO_eV", "partial_atomic_charge")


def evaluate_locked_head(config, checkpoint, task_root, output):
    head, manifest = load_regression_head(task_root, checkpoint)
    task = manifest["task"]
    if task not in TEST_TASKS:
        raise ValueError("Only HOMO, LUMO and partial charge have Stage1 test sets")
    source = config_from_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["config"])
    if task == "partial_atomic_charge":
        recipe = replace(source, auxiliary=replace(source.auxiliary, simulation_dir=config.simulation_dir,
                         partial_charge_manifest=config.partial_charge_manifest))
        mapped, audit = load_charge_test_rows(recipe)
        rows = [row for key in sorted(mapped) for row in mapped[key]]
        source_contract = charge_test_source_contract(recipe)
        atomic_json(Path(output) / "mapping_audit.json", audit)
    else:
        rows = load_scalar_regression_rows(config, task, "test")
        source_contract = {"csv_sha256": sha256_file(config.simulation_dir / ("homo" if task == "HOMO_eV" else "lumo") / "test.csv")}
    dev = {row["canonical_smiles"] for split in ("train", "valid")
           for row in load_regression_rows(config, source, task, split)}
    if dev & {row["canonical_smiles"] for row in rows}:
        raise ValueError("Stage1 test canonical overlap with development rows")
    _, vocabulary, _, _, _ = load_stage1_feature_inputs(checkpoint, config.stage1_artifacts)
    data, audit = filter_regression_inputs({task: {"test": rows}}, vocabulary, source.data.max_smiles_tokens)
    rows = data[task]["test"]
    if not rows:
        raise ValueError("Stage1 test has no encodable observations")
    bank = encode_regression_molecules(config, checkpoint, {row["canonical_smiles"]: row for row in rows})
    device = resolve_device(config.device)
    head = head.to(device).requires_grad_(False).eval()
    scaler, atom = manifest["scaler"], task == "partial_atomic_charge"
    metrics = evaluate_regression_head(head, rows, bank, atom, scaler, device, config.batch_size)
    predictions = []
    with torch.no_grad():
        for observation, row in enumerate(rows):
            inputs = bank[row["canonical_smiles"]]["atoms" if atom else "entity"].to(device)
            values = (head(inputs).squeeze(-1).float() * scaler["scale"] + scaler["mean"]).cpu().tolist()
            for index, (target, value) in enumerate(zip(row["targets"], values, strict=True)):
                predictions.append({"observation": observation, "canonical_smiles": row["canonical_smiles"],
                    "mol_id": row.get("mol_id", ""), "atom": index if atom else "", "target": float(target), "prediction": value})
    file = write_prediction_csv(Path(output) / "predictions.csv", predictions, tuple(predictions[0]))
    file["path"] = "predictions.csv"
    return {"task": task, "split": "test", "selection": "locked-fixed-final", "metrics": metrics,
            "head_sha256": manifest["artifact_sha256"], "source": source_contract, "filter": audit, "predictions": file}
