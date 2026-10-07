"""Independent final-epoch regressors on frozen, unmasked Stage1 representations."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from rdkit import Chem

from common.identity import semantic_identity, tensor_state_hash, validate_semantic_identity
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.progress import ProgressReporter
from common.training import resolve_device
from .auxiliary import ELECTRONIC_COLUMNS, ELECTRONIC_SOURCES
from .config import STAGE1_CHECKPOINT_KIND, config_from_dict
from .features import ROLE_TO_ID, build_entity_sample, load_stage1_feature_inputs
from .identity import encoding_state_hash
from .masking import MultimodalPacker
from .model import load_stage1_model
from .partial_charge import PartialChargeCache, charge_source_contract, load_charge_rows
from .predictors import PredictorConfig, build_predictor, predictor_from_dict, predictor_rng


REGRESSION_TASKS = (*ELECTRONIC_COLUMNS, "partial_atomic_charge")
REGRESSION_KIND = "ilume_stage1_frozen_regression_head_v4"


@dataclass(frozen=True)
class RegressionConfig:
    stage1_artifacts: Path = Path("outputs/v4/stage1/base/prepare/artifacts")
    simulation_dir: Path = Path("data/stage2")
    partial_charge_manifest: Path = Path("data/stage1/properties/partial_atomic_charge/charge_20260514/structure_manifest.csv")
    partial_charge_cache: Path = Path("outputs/v4/stage1/base/partial_charge/artifacts")
    epochs: int = 10
    learning_rate: float = 1.e-4
    batch_size: int = 128
    weight_decay: float = .01
    betas: tuple[float, float] = (.9, .999)
    eps: float = 1.e-8
    max_grad_norm: float = 1.
    amp_dtype: str = "bf16"
    device: str = "auto"
    seed: int = 42
    role_weights: tuple[float, float, float] = (2., 2., 1.)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    predictor_overrides: dict[str, PredictorConfig] = field(default_factory=dict)

    def validate(self):
        self.predictor.validate()
        if set(self.predictor_overrides) - set(REGRESSION_TASKS):
            raise ValueError("Unknown predictor override target")
        for recipe in self.predictor_overrides.values():
            recipe.validate()
        for name in ("epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"regression.{name} must be a positive integer")
        for name in ("learning_rate", "eps", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"regression.{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("regression.weight_decay must be finite and nonnegative")
        if len(self.betas) != 2 or any(not 0 <= v < 1 for v in self.betas):
            raise ValueError("regression.betas must contain two values in [0,1)")
        if len(self.role_weights) != 3 or any(not math.isfinite(v) or v <= 0 for v in self.role_weights):
            raise ValueError("regression.role_weights must contain three positive values")
        if self.amp_dtype not in {"bf16", "fp16", "none"}:
            raise ValueError("Unsupported regression AMP precision")

    def to_dict(self):
        result = {k: str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v
                  for k, v in asdict(self).items() if k not in {"predictor", "predictor_overrides"}}
        if self.predictor != PredictorConfig():
            result["predictor"] = self.predictor.to_dict()
        if self.predictor_overrides:
            result["predictor_overrides"] = {task: recipe.to_dict() for task, recipe in self.predictor_overrides.items()}
        return result

    def resolved_predictor(self, task):
        if task not in REGRESSION_TASKS:
            raise ValueError("Unknown predictor target")
        return self.predictor_overrides.get(task, self.predictor)


def load_regression_config(path):
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict) or set(raw) - set(RegressionConfig.__dataclass_fields__):
        raise ValueError("Unknown regression configuration fields")
    for name in ("stage1_artifacts", "simulation_dir", "partial_charge_manifest", "partial_charge_cache"):
        if name in raw:
            raw[name] = Path(raw[name])
    for name in ("betas", "role_weights"):
        if name in raw:
            raw[name] = tuple(raw[name])
    if "predictor" in raw:
        raw["predictor"] = predictor_from_dict(raw["predictor"])
    if "predictor_overrides" in raw:
        if not isinstance(raw["predictor_overrides"], dict):
            raise ValueError("predictor_overrides must be a target-to-predictor mapping")
        raw["predictor_overrides"] = {task: predictor_from_dict(value) for task, value in raw["predictor_overrides"].items()}
    config = RegressionConfig(**raw)
    config.validate()
    return config


def regression_tasks(tasks=None):
    result = tuple(tasks) if tasks is not None else REGRESSION_TASKS
    if not result or len(set(result)) != len(result) or set(result) - set(REGRESSION_TASKS):
        raise ValueError("Regression tasks must be nonempty, unique, known targets")
    return tuple(task for task in REGRESSION_TASKS if task in result)


def regression_identity(config, checkpoint_path, tasks=None):
    config.validate()
    tasks = regression_tasks(tasks)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = config_from_dict(payload["config"])
    if (payload.get("kind") != STAGE1_CHECKPOINT_KIND or payload.get("format_version") != 4
        or not source.is_dual_view or payload.get("completed_epoch") != source.training.epochs):
        raise ValueError("Regression requires a complete final v4 pretraining checkpoint with heads")
    validate_semantic_identity(payload["training_identity"])
    scientific = payload["training_identity"]["payload"]
    if scientific["model"] != source.to_dict()["model"] or scientific["loss"] != source.to_dict()["loss"]:
        raise ValueError("Regression checkpoint config differs from its training identity")
    if payload.get("global_step") != source.training.epochs * payload.get("steps_per_epoch", -1):
        raise ValueError("Regression checkpoint is not a complete final epoch boundary")
    if "partial_atomic_charge" in tasks and "partial_charge_head.weight" not in payload["model"]:
        raise ValueError("Source checkpoint has no partial-charge head; select existing electronic tasks or retrain Stage1")
    data = {}
    source = replace(source, auxiliary=replace(source.auxiliary, simulation_dir=config.simulation_dir,
                     partial_charge_manifest=config.partial_charge_manifest))
    for task in tasks:
        if task == "partial_atomic_charge":
            data[task] = {split: charge_source_contract(source, split) for split in ("train", "valid")}
        else:
            directory = next(name for name, columns, _ in ELECTRONIC_SOURCES if task in columns)
            data[task] = {split: sha256_file(config.simulation_dir / directory / f"{split}.csv")
                          for split in ("train", "valid")}
    recipe = config.to_dict()
    recipe.pop("predictor", None)
    recipe.pop("predictor_overrides", None)
    if any(config.resolved_predictor(task).type != "linear" for task in tasks):
        recipe["predictors"] = {task: config.resolved_predictor(task).to_dict() for task in tasks}
    for key in ("stage1_artifacts", "simulation_dir", "partial_charge_manifest", "partial_charge_cache", "device"):
        recipe.pop(key)
    metadata = json.loads((config.stage1_artifacts / "metadata.json").read_text())
    return semantic_identity("stage1.frozen-regression.training.v4", {
        "base_checkpoint_sha256": sha256_file(checkpoint_path),
        "base_training_identity": payload["training_identity"]["hash"],
        "feature_identity": metadata["semantic"]["identities"]["feature"]["hash"],
        "data": data, "tasks": list(tasks), "recipe": recipe,
        "selection": "fixed-final-epoch", "inputs": "unmasked-frozen-learned1024-or-atom512",
    })


def load_regression_rows(config, source_config, task, split):
    if split not in {"train", "valid"}:
        raise ValueError("Post-training accepts train/valid only, never test")
    if task == "partial_atomic_charge":
        recipe = replace(source_config, auxiliary=replace(source_config.auxiliary,
                         simulation_dir=config.simulation_dir, partial_charge_manifest=config.partial_charge_manifest))
        rows, _ = load_charge_rows(recipe, split)
        return [row for key in sorted(rows) for row in rows[key]]
    directory = next(name for name, columns, _ in ELECTRONIC_SOURCES if task in columns)
    rows = {}
    with (config.simulation_dir / directory / f"{split}.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"SMILES", task} <= set(reader.fieldnames or ()):
            raise ValueError(f"Regression source columns missing: {task}")
        for row in reader:
            value = float(row[task]) if row[task].strip() else float("nan")
            if not np.isfinite(value):
                continue
            molecule = Chem.MolFromSmiles(row["SMILES"])
            if molecule is None:
                raise ValueError(f"Invalid regression SMILES: {directory}/{split}")
            key = Chem.MolToSmiles(molecule, canonical=True)
            charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
            role = "cation" if charge > 0 else "anion" if charge < 0 else "neutral"
            if key in rows and rows[key]["targets"][0] != value:
                raise ValueError(f"Conflicting regression labels: {task}/{key}")
            rows[key] = {"canonical_smiles": key, "role_id": ROLE_TO_ID[role],
                         "targets": np.asarray([value], dtype=np.float64)}
    return [rows[key] for key in sorted(rows)]


def initialize_regression_head(state, task, old_scaler, new_scaler, device, predictor=None, seed=42):
    predictor = predictor if predictor is not None else PredictorConfig()
    if task == "partial_atomic_charge":
        weight, bias = state["partial_charge_head.weight"], state["partial_charge_head.bias"]
    else:
        index = ELECTRONIC_COLUMNS.index(task)
        weight, bias = state["electronic_head.weight"][index:index + 1], state["electronic_head.bias"][index:index + 1]
    with predictor_rng(predictor, seed, device):
        head = build_predictor(predictor, weight.shape[1]).to(device)
    if predictor.type != "linear":
        return head
    with torch.no_grad():
        head.weight.copy_(weight.to(device) * (old_scaler["scale"] / new_scaler["scale"]))
        head.bias.copy_((bias.to(device) * old_scaler["scale"] + old_scaler["mean"] - new_scaler["mean"]) / new_scaler["scale"])
    return head


def regression_batch(rows, bank, atom_level, scaler, device):
    inputs = [bank[row["canonical_smiles"]]["atoms" if atom_level else "entity"] for row in rows]
    counts = torch.tensor([len(value) for value in inputs], device=device)
    molecule_ids = torch.repeat_interleave(torch.arange(len(rows), device=device), counts)
    targets = torch.as_tensor(np.concatenate([row["targets"] for row in rows]), dtype=torch.float32, device=device)
    return torch.cat(inputs).to(device), (targets - scaler["mean"]) / scaler["scale"], molecule_ids, counts


def regression_loss(prediction, targets, molecule_ids, counts, roles, role_weights):
    values = F.smooth_l1_loss(prediction.float(), targets, reduction="none")
    means = values.new_zeros(len(counts)).index_add(0, molecule_ids, values) / counts
    weights = torch.as_tensor(role_weights, device=values.device)[roles]
    return (means * weights).sum() / weights.sum()


def evaluate_regression_head(head, rows, bank, atom_level, scaler, device, batch_size):
    if not rows:
        return {"mae": None, "rmse": None, "molecules": 0}
    mae, mse = 0., 0.
    head.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            inputs, targets, ids, counts = regression_batch(batch, bank, atom_level, scaler, device)
            difference = (head(inputs).squeeze(-1).float() - targets) * scaler["scale"]
            mae += (difference.new_zeros(len(batch)).index_add(0, ids, difference.abs()) / counts).sum().item()
            mse += (difference.new_zeros(len(batch)).index_add(0, ids, difference.square()) / counts).sum().item()
    return {"mae": mae / len(rows), "rmse": math.sqrt(mse / len(rows)), "molecules": len(rows)}


def run_regression(config, checkpoint_path, output_dir, tasks=None):
    tasks = regression_tasks(tasks)
    identity = regression_identity(config, checkpoint_path, tasks)
    root = Path(output_dir)
    if (root / "representations.pt").exists() or (root / "tasks").exists():
        raise FileExistsError("Regression output already contains artifacts; use a new output directory")
    root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    loaded = load_stage1_model(checkpoint_path, config.stage1_artifacts, device=device)
    encoder = loaded.model.requires_grad_(False).eval()
    encoder_hash = encoding_state_hash(encoder)
    source_config, vocabulary, schema, standardizer, _ = load_stage1_feature_inputs(checkpoint_path, config.stage1_artifacts)
    electronic_scaler = json.loads((config.stage1_artifacts / "electronic_scaler.json").read_text())
    charge_scaler = None
    if "partial_atomic_charge" in tasks:
        charge_recipe = replace(source_config, auxiliary=replace(source_config.auxiliary,
                               simulation_dir=config.simulation_dir, partial_charge_manifest=config.partial_charge_manifest,
                               partial_charge_cache=config.partial_charge_cache))
        metadata = json.loads((config.stage1_artifacts / "metadata.json").read_text())
        cache = PartialChargeCache(charge_recipe, metadata)
        if cache.manifest["identity"]["hash"] != source["training_identity"]["payload"]["training"].get("partial_charge_identity"):
            raise ValueError("Regression partial-charge scaler differs from source training identity")
        charge_scaler = cache.scaler
    data, molecules = {}, {}
    for task in tasks:
        data[task] = {split: load_regression_rows(config, source_config, task, split) for split in ("train", "valid")}
        if not data[task]["train"]:
            raise ValueError(f"Regression task has no training labels: {task}")
        if {r["canonical_smiles"] for r in data[task]["train"]} & {r["canonical_smiles"] for r in data[task]["valid"]}:
            raise ValueError(f"Regression train/valid canonical overlap: {task}")
        for split in ("train", "valid"):
            for row in data[task][split]:
                molecules[row["canonical_smiles"]] = row
    bank, keys = {}, sorted(molecules)
    reporter = ProgressReporter()
    packer = MultimodalPacker(vocabulary)
    amp_enabled = device.type == "cuda" and config.amp_dtype != "none"
    amp_dtype = torch.float16 if config.amp_dtype == "fp16" else torch.bfloat16
    # Encode once at full precision; inference inputs are clean and heads alone train.
    with reporter.bar(total=len(keys), desc="Frozen Stage1 regression representations", unit="molecule") as progress, torch.no_grad():
        for start in range(0, len(keys), config.batch_size):
            selected = keys[start:start + config.batch_size]
            samples = [build_entity_sample({"canonical_smiles": key, "sample_id": key,
                        "role_id": molecules[key]["role_id"]}, np.zeros(217), schema, standardizer, vocabulary, source_config) for key in selected]
            encoded = encoder.encode_entity(packer(samples).to(device))
            offset = 0
            for index, key in enumerate(selected):
                count = len(samples[index]["atom_categorical"])
                bank[key] = {"entity": encoded.entity_embedding[index:index + 1].float().cpu(),
                             "atoms": encoded.atom_states[offset:offset + count].float().cpu()}
                offset += count
            progress.update(len(selected))
    bank_hash = tensor_state_hash("stage1.regression.frozen-representations.v4", bank)
    atomic_torch_save(root / "representations.pt", {"identity": identity, "encoder_hash": encoder_hash,
                                                   "state_hash": bank_hash, "representations": bank})
    atomic_json(root / "representation_manifest.json", {"identity": identity, "encoder_hash": encoder_hash,
                "state_hash": bank_hash, "artifact_sha256": sha256_file(root / "representations.pt")})
    results = {}
    for task in tasks:
        train, valid = data[task]["train"], data[task]["valid"]
        atom_level = task == "partial_atomic_charge"
        targets = np.concatenate([r["targets"] for r in train])
        scaler = {"mean": float(targets.mean()), "scale": float(targets.std()) or 1.}
        if any(not math.isfinite(value) for value in scaler.values()):
            raise ValueError(f"Non-finite regression train statistics: {task}")
        if atom_level:
            old_scaler = charge_scaler
        else:
            index = ELECTRONIC_COLUMNS.index(task)
            old_scaler = {"mean": electronic_scaler["mean"][index], "scale": electronic_scaler["scale"][index]}
        seed = config.seed + REGRESSION_TASKS.index(task)
        predictor = config.resolved_predictor(task)
        input_dim = source["model"]["partial_charge_head.weight" if atom_level else "electronic_head.weight"].shape[1]
        with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []):
            head = initialize_regression_head(source["model"], task, old_scaler, scaler, device, predictor, seed)
        task_root = root / "tasks" / task
        task_root.mkdir(parents=True)
        initial = evaluate_regression_head(head, valid, bank, atom_level, scaler, device, config.batch_size)
        optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
                                     betas=config.betas, eps=config.eps)
        grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and config.amp_dtype == "fp16")
        updates = 0
        with predictor_rng(predictor, seed, device), reporter.bar(total=config.epochs, desc=f"Regression {task}", unit="epoch") as progress:
            for epoch in range(1, config.epochs + 1):
                head.train()
                order = np.random.default_rng(seed + epoch).permutation(len(train))
                total_loss, total_weight = 0., 0.
                for start in range(0, len(order), config.batch_size):
                    rows = [train[i] for i in order[start:start + config.batch_size]]
                    inputs, labels, ids, counts = regression_batch(rows, bank, atom_level, scaler, device)
                    roles = torch.tensor([r["role_id"] for r in rows], device=device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                        loss = regression_loss(head(inputs).squeeze(-1), labels, ids, counts, roles, config.role_weights)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(head.parameters(), config.max_grad_norm)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    updates += 1
                    batch_weight = sum(config.role_weights[r["role_id"]] for r in rows)
                    total_loss += loss.item() * batch_weight
                    total_weight += batch_weight
                metrics = {"task": task, "epoch": epoch, "updates": updates,
                           "learning_rate": config.learning_rate, "train_loss": total_loss / total_weight,
                           "validation": evaluate_regression_head(head, valid, bank, atom_level, scaler, device, config.batch_size)}
                with (task_root / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                reporter.emit_json(metrics)
                progress.update(1)
        state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
        state_hash = tensor_state_hash("stage1.regression.head.v4", state)
        manifest = {"kind": REGRESSION_KIND, "format_version": 2, "identity": identity,
                    "task": task, "input_dim": input_dim, "scaler": scaler,
                    "predictor": predictor.to_dict(), "parameter_count": sum(p.numel() for p in head.parameters()),
                    "initialization": "pretrained-affine" if predictor.type == "linear" else "task-seeded-random",
                    "base_checkpoint_sha256": identity["payload"]["base_checkpoint_sha256"],
                    "base_encoder_hash": encoder_hash, "base_training_identity": source["training_identity"],
                    "fixed_final_epoch": config.epochs, "updates": updates, "seed": seed,
                    "state_hash": state_hash, "initial_validation": initial, "validation": metrics["validation"]}
        atomic_torch_save(task_root / "regression_head.pt", {**manifest, "model": state})
        manifest["artifact_sha256"] = sha256_file(task_root / "regression_head.pt")
        atomic_json(task_root / "regression_head.json", manifest)
        results[task] = manifest
    if encoding_state_hash(encoder) != encoder_hash or any(p.grad is not None or p.requires_grad for p in encoder.parameters()):
        raise RuntimeError("Frozen regression modified the Stage1 encoder")
    if regression_identity(config, checkpoint_path, tasks) != identity:
        raise ValueError("Regression source changed during training")
    summary = {"kind": "ilume_stage1_regression_summary_v4", "identity": identity,
               "encoder_hash": encoder_hash, "representation_hash": bank_hash, "tasks": results}
    atomic_json(root / "summary.json", summary)
    return summary


def load_regression_head(task_root, checkpoint_path):
    root = Path(task_root)
    manifest = json.loads((root / "regression_head.json").read_text())
    if manifest.get("kind") != REGRESSION_KIND or manifest.get("format_version") not in {1, 2}:
        raise ValueError("Unsupported regression-head artifact")
    validate_semantic_identity(manifest["identity"])
    validate_semantic_identity(manifest["base_training_identity"])
    if (manifest["identity"]["payload"]["base_checkpoint_sha256"] != manifest["base_checkpoint_sha256"]
        or manifest["identity"]["payload"]["base_training_identity"] != manifest["base_training_identity"]["hash"]
        or manifest["task"] not in manifest["identity"]["payload"]["tasks"]):
        raise ValueError("Regression-head source/target identity mismatch")
    if sha256_file(checkpoint_path) != manifest["base_checkpoint_sha256"]:
        raise ValueError("Regression head belongs to a different Stage1 checkpoint")
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_key = "partial_charge_head.weight" if manifest["task"] == "partial_atomic_charge" else "electronic_head.weight"
    if manifest["input_dim"] != source["model"][source_key].shape[1]:
        raise ValueError("Regression-head input dimension differs from its frozen source")
    if sha256_file(root / "regression_head.pt") != manifest["artifact_sha256"]:
        raise ValueError("Regression-head artifact hash mismatch")
    payload = torch.load(root / "regression_head.pt", map_location="cpu", weights_only=False)
    if {key: value for key, value in payload.items() if key != "model"} != {key: value for key, value in manifest.items() if key != "artifact_sha256"}:
        raise ValueError("Regression-head manifest/payload mismatch")
    if tensor_state_hash("stage1.regression.head.v4", payload["model"]) != manifest["state_hash"]:
        raise ValueError("Regression-head state hash mismatch")
    predictor = predictor_from_dict(manifest["predictor"]) if manifest["format_version"] == 2 else PredictorConfig()
    resolved = manifest["identity"]["payload"]["recipe"].get("predictors", {})
    if predictor.to_dict() != resolved.get(manifest["task"], {"type": "linear"}):
        raise ValueError("Regression-head predictor differs from its training identity")
    head = build_predictor(predictor, manifest["input_dim"])
    if manifest["format_version"] == 2:
        if manifest["parameter_count"] != sum(p.numel() for p in head.parameters()):
            raise ValueError("Regression-head parameter count mismatch")
        expected = "pretrained-affine" if predictor.type == "linear" else "task-seeded-random"
        if manifest["initialization"] != expected:
            raise ValueError("Regression-head initialization contract mismatch")
    head.load_state_dict(payload["model"], strict=True)
    return head.requires_grad_(False).eval(), manifest
