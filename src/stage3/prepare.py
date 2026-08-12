from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from rdkit import Chem, rdBase

from stage1.data import ROLE_TO_ID, _build_sample, _inspect_entity_qc
from stage1.descriptors import calculate_descriptors, rdkit_descriptor_names
from stage1.masking import MultimodalPacker
from common.progress import ProgressReporter
from stage2.config import _stage2_config_from_checkpoint_dict
from stage2.data import _load_pretrain_inputs
from stage2.model import Stage2AlignmentModel, load_stage1_model, sha256_file
from stage2.prepare import resolve_device
from .config import Stage3Config
from .data import (
    STAGE3_ARTIFACT_VERSION,
    TASK_REGISTRY,
    build_task_payload,
    collect_entity_keys_with_audit,
    fit_fold_scalers,
    sanitize_task,
    signature,
    source_hashes,
)


STAGE2_CHECKPOINT_VERSION = 2
STAGE2_CHECKPOINT_KIND = "ilume_stage2_alignment"


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _config_hash(raw: dict[str, Any]) -> str:
    payload = _stage2_config_from_checkpoint_dict(raw).to_dict()
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_frozen_stage2(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[Stage2AlignmentModel, dict[str, Any], str]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != STAGE2_CHECKPOINT_VERSION or checkpoint.get("kind") != STAGE2_CHECKPOINT_KIND:
        raise ValueError("Stage 3 requires a Stage 2 v2 alignment checkpoint")
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("Stage 2 checkpoint configuration is missing")
    stage2_config = _stage2_config_from_checkpoint_dict(raw_config)
    loaded = load_stage1_model(
        stage2_config.initialization.checkpoint,
        stage2_config.data.pretrain_artifacts_dir,
        device=device,
        backbone_dropout=0.0,
    )
    model = Stage2AlignmentModel(loaded.model, head_dropout=stage2_config.model.head_dropout)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    provenance = {
        "stage2_checkpoint": str(checkpoint_path),
        "stage2_config_hash": _config_hash(raw_config),
        "stage2_data_metadata_hash": sha256_file(
            stage2_config.data.artifacts_dir / "metadata.json"
        ),
        "stage1_checkpoint": str(stage2_config.initialization.checkpoint),
        "stage1_artifact_hash": loaded.artifact_hash,
        "tokenizer_hash": sha256_file(stage2_config.data.pretrain_artifacts_dir / "tokenizer.json"),
        "descriptor_schema_hash": sha256_file(stage2_config.data.pretrain_artifacts_dir / "descriptor_schema.json"),
        "descriptor_scaler_hash": sha256_file(stage2_config.data.pretrain_artifacts_dir / "descriptor_scaler.json"),
    }
    del checkpoint
    return model, provenance, str(stage2_config.initialization.checkpoint)


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _stage2_pretrain_inputs(raw_stage2_config: dict[str, Any]):
    config = _stage2_config_from_checkpoint_dict(raw_stage2_config)
    return config, _load_pretrain_inputs(config)


def _encode_entities(
    config: Stage3Config,
    domain: str,
    tasks: Sequence[str],
    stage2_model: Stage2AlignmentModel,
    raw_stage2_config: dict[str, Any],
    device: torch.device,
    reporter: ProgressReporter,
) -> tuple[list[dict[str, Any]], torch.Tensor, list[dict[str, Any]]]:
    stage2_config, inputs = _stage2_pretrain_inputs(raw_stage2_config)
    pretrain_config, vocabulary, schema, standardizer, _ = inputs
    keys, parse_exclusions = collect_entity_keys_with_audit(config, tasks)
    exclusions = list(parse_exclusions)
    entries: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    raw_names = rdkit_descriptor_names()
    for role, canonical in keys:
        record = {
            "sample_id": f"stage3_entity_{len(entries):08d}",
            "role": role,
            "role_id": ROLE_TO_ID[role],
            "canonical_smiles": canonical,
            "sources": ("stage3",),
            "split": "stage3",
            "is_augmented": False,
            "seed_smiles": (),
        }
        qc = _inspect_entity_qc(record)
        token_count = vocabulary.token_count(canonical)
        if token_count > pretrain_config.data.max_smiles_tokens:
            qc.reasons.append("smiles_overlength")
        sample = None
        detail = ""
        if not qc.reasons:
            try:
                molecule = Chem.MolFromSmiles(canonical)
                if molecule is None:
                    raise ValueError("RDKit parsing failed")
                sample = _build_sample(
                    record,
                    calculate_descriptors(molecule, raw_names),
                    schema,
                    standardizer,
                    vocabulary,
                    pretrain_config,
                )
            except (RuntimeError, ValueError, OverflowError) as error:
                qc.reasons.append("feature_error")
                detail = str(error)
        if sample is None:
            exclusions.append({
                "task": "*", "fold": "*", "source_row": "*", "column": role,
                "smiles": canonical, "reason": ";".join(qc.reasons) + (f":{detail}" if detail else ""),
            })
            continue
        entries.append({"entity_id": len(entries), "role": role, "role_id": ROLE_TO_ID[role], "canonical_smiles": canonical})
        samples.append(sample)
    if not samples:
        raise ValueError("Stage 3 retained no encodable entities")
    embeddings = torch.empty((len(samples), stage2_model.backbone.config.model.d_model), dtype=torch.float32)
    packer = MultimodalPacker(vocabulary)
    with torch.no_grad(), reporter.bar(
        total=len(samples),
        desc=f"Stage 3 {domain} frozen entity CLS",
        unit="entity",
    ) as progress:
        for start in range(0, len(samples), config.data.entity_batch_size):
            end = min(len(samples), start + config.data.entity_batch_size)
            batch = packer(samples[start:end]).to(device)
            encoded = stage2_model.backbone.encode(batch).float().cpu()
            if not torch.isfinite(encoded).all():
                raise RuntimeError(f"Non-finite Stage 3 entity CLS in rows {start}:{end}")
            embeddings[start:end] = encoded
            progress.update(end - start)
    del stage2_config
    return entries, embeddings, exclusions


def _pair_cache(
    config: Stage3Config,
    tasks: Sequence[str],
    stage2_model: Stage2AlignmentModel,
    entries: list[dict[str, Any]],
    entity_embeddings: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    entity_ids = {(row["role"], row["canonical_smiles"]): row["entity_id"] for row in entries}
    il_keys: set[tuple[int, int]] = set()
    neutral_keys: set[tuple[int, int]] = set()
    for fold in range(1, 6):
        scalers = fit_fold_scalers(config, fold, tasks)
        for task in tasks:
            for split in ("train", "valid", "test"):
                payload, _ = build_task_payload(config, task, fold, split, entity_ids, scalers)
                ids = payload["entity_ids"]
                if TASK_REGISTRY[task].topology in {"il", "il_solute"}:
                    il_keys.update((int(row[0]), int(row[1])) for row in ids)
                elif TASK_REGISTRY[task].topology == "neutral_pair":
                    neutral_keys.update((int(row[0]), int(row[1])) for row in ids)

    def encode(keys: set[tuple[int, int]], encoder: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        ordered = sorted(keys)
        key_tensor = torch.tensor(ordered, dtype=torch.long).reshape(len(ordered), 2)
        values = torch.empty((len(ordered), entity_embeddings.shape[1]), dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, len(ordered), config.data.entity_batch_size):
                end = min(len(ordered), start + config.data.entity_batch_size)
                pair = key_tensor[start:end]
                values[start:end] = encoder(
                    entity_embeddings[pair[:, 0]].to(device),
                    entity_embeddings[pair[:, 1]].to(device),
                ).float().cpu()
        return key_tensor, values

    il_key_tensor, il_embeddings = encode(il_keys, stage2_model.il_pair_encoder)
    neutral_key_tensor, neutral_embeddings = encode(neutral_keys, stage2_model.transfer_pair_encoder)
    return {
        "format_version": STAGE3_ARTIFACT_VERSION,
        "entity_embeddings": entity_embeddings,
        "il_pair_keys": il_key_tensor,
        "il_pair_embeddings": il_embeddings,
        "neutral_pair_keys": neutral_key_tensor,
        "neutral_pair_embeddings": neutral_embeddings,
    }


def _exposure_audit(task_payloads: dict[tuple[int, str, str], dict[str, Any]], tasks: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        pairs: dict[tuple[str, str], set[tuple[int, int]]] = {}
        for task in tasks:
            for split in ("train", "valid"):
                payload = task_payloads[(fold, task, split)]
                if TASK_REGISTRY[task].topology in {"il", "il_solute"}:
                    pairs[(task, split)] = {tuple(map(int, value[:2])) for value in payload["entity_ids"].tolist()}
                else:
                    pairs[(task, split)] = set()
            overlap = pairs[(task, "train")] & pairs[(task, "valid")]
            if overlap:
                raise ValueError(f"Stage 3 train/valid IL-pair leakage in {task}/fold{fold}: {len(overlap)}")
        for valid_task in tasks:
            valid_pairs = pairs[(valid_task, "valid")]
            if not valid_pairs:
                continue
            for train_task in tasks:
                if train_task == valid_task:
                    continue
                count = len(valid_pairs & pairs[(train_task, "train")])
                if count:
                    rows.append({"fold": fold, "valid_task": valid_task, "train_task": train_task, "exposed_il_pairs": count})
    return rows


def _prepare_domain(
    config: Stage3Config,
    domain: str,
    stage2_model: Stage2AlignmentModel,
    raw_stage2_config: dict[str, Any],
    provenance: dict[str, Any],
    device: torch.device,
    reporter: ProgressReporter,
) -> dict[str, Any]:
    tasks = config.tasks_for_domain(domain)
    output_dir = config.data.artifacts_dir / domain
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes(config, tasks)
    data_signature = signature(
        {
            "domain": domain,
            "tasks": tasks,
            "sources": hashes,
            "provenance": provenance,
        }
    )
    metadata_path = output_dir / "metadata.json"
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("format_version") == STAGE3_ARTIFACT_VERSION and existing.get("data_signature") == data_signature:
            for relative, expected in existing.get("artifact_hashes", {}).items():
                if sha256_file(output_dir / relative) != expected:
                    raise ValueError(f"Stage 3 artifact hash mismatch: {relative}")
            return existing
    entries, entity_embeddings, excluded_entities = _encode_entities(
        config,
        domain,
        tasks,
        stage2_model,
        raw_stage2_config,
        device,
        reporter,
    )
    entity_ids = {(row["role"], row["canonical_smiles"]): row["entity_id"] for row in entries}
    cache = _pair_cache(
        config,
        tasks,
        stage2_model,
        entries,
        entity_embeddings,
        device,
    )
    _atomic_json(output_dir / "entity_index.json", {"format_version": STAGE3_ARTIFACT_VERSION, "entries": entries})
    _atomic_torch_save(output_dir / "frozen_embeddings.pt", cache)
    _write_csv(output_dir / "excluded_entities.csv", ("task", "fold", "source_row", "column", "smiles", "reason"), excluded_entities)
    task_payloads: dict[tuple[int, str, str], dict[str, Any]] = {}
    excluded_rows: list[dict[str, Any]] = []
    artifact_files = ["entity_index.json", "frozen_embeddings.pt", "excluded_entities.csv"]
    scaler_payload: dict[str, Any] = {}
    for fold in range(1, 6):
        fold_dir = output_dir / "folds" / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        scalers = fit_fold_scalers(config, fold, tasks)
        scaler_payload[f"fold{fold}"] = scalers
        for task in tasks:
            for split in ("train", "valid", "test"):
                payload, exclusions = build_task_payload(config, task, fold, split, entity_ids, scalers)
                task_payloads[(fold, task, split)] = payload
                excluded_rows.extend(exclusions)
                relative = f"folds/fold{fold}/{sanitize_task(task)}_{split}.pt"
                _atomic_torch_save(output_dir / relative, payload)
                artifact_files.append(relative)
    _atomic_json(output_dir / "scalers.json", scaler_payload)
    artifact_files.append("scalers.json")
    _write_csv(output_dir / "excluded_rows.csv", ("task", "fold", "source_row", "reason"), excluded_rows)
    artifact_files.append("excluded_rows.csv")
    exposure_rows = (
        _exposure_audit(task_payloads, tuple(tasks))
        if domain == "il21"
        else []
    )
    _write_csv(output_dir / "cross_task_exposure.csv", ("fold", "valid_task", "train_task", "exposed_il_pairs"), exposure_rows)
    artifact_files.append("cross_task_exposure.csv")
    metadata = {
        "format_version": STAGE3_ARTIFACT_VERSION,
        "kind": "ilume_stage3_data",
        "domain": domain,
        "tasks": list(tasks),
        "task_registry": {task: TASK_REGISTRY[task].__dict__ for task in tasks},
        "embedding_dim": int(entity_embeddings.shape[1]),
        "entity_count": len(entries),
        "source_hashes": hashes,
        "data_signature": data_signature,
        "provenance": provenance,
        "rdkit_version": rdBase.rdkitVersion,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "artifact_hashes": {relative: sha256_file(output_dir / relative) for relative in artifact_files},
        "summary": {"excluded_entities": len(excluded_entities), "excluded_rows": len(excluded_rows), "cross_task_exposures": len(exposure_rows)},
    }
    metadata = json.loads(json.dumps(metadata, ensure_ascii=False))
    _atomic_json(metadata_path, metadata)
    reporter.emit_json(
        {
            "event": "stage3_prepare_domain_complete",
            "domain": domain,
            **metadata["summary"],
        }
    )
    return metadata


def prepare_stage3(
    config: Stage3Config,
    *,
    reporter: ProgressReporter | None = None,
) -> dict[str, dict[str, Any]]:
    config.validate()
    reporter = reporter or ProgressReporter()
    device = resolve_device(config.training.device)
    stage2_checkpoint = torch.load(
        config.initialization.stage2_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    raw_stage2_config = stage2_checkpoint.get("config")
    del stage2_checkpoint
    stage2_model, provenance, _ = load_frozen_stage2(
        config.initialization.stage2_checkpoint,
        device=device,
    )
    results = {
        domain: _prepare_domain(
            config,
            domain,
            stage2_model,
            raw_stage2_config,
            provenance,
            device,
            reporter,
        )
        for domain in config.active_domains
    }
    reporter.emit_json(
        {
            "event": "stage3_prepare_complete",
            "domains": list(config.active_domains),
        }
    )
    return results


def load_frozen_embeddings(
    config: Stage3Config,
    domain: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tasks = config.tasks_for_domain(domain)
    root = config.data.artifacts_dir / domain
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != STAGE3_ARTIFACT_VERSION or metadata.get("kind") != "ilume_stage3_data":
        raise ValueError("Unsupported Stage 3 artifact format")
    if metadata.get("domain") != domain or tuple(metadata.get("tasks", ())) != tasks:
        raise ValueError("Stage 3 artifact domain/task registry mismatch")
    if metadata.get("rdkit_version") != rdBase.rdkitVersion:
        raise ValueError("Stage 3 artifact RDKit version mismatch")
    if metadata.get("source_hashes") != source_hashes(config, tasks):
        raise ValueError("Stage 3 source data hash mismatch")
    if metadata.get("provenance", {}).get("stage2_checkpoint") != str(
        config.initialization.stage2_checkpoint
    ):
        raise ValueError("Stage 3 frozen cache Stage 2 checkpoint path mismatch")
    path = root / "frozen_embeddings.pt"
    if metadata["artifact_hashes"].get("frozen_embeddings.pt") != sha256_file(path):
        raise ValueError("Stage 3 frozen embedding hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload, metadata
