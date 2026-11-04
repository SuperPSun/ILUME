"""Standalone train-only descriptor selection without changing Stage artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import partial
from itertools import islice
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem

from .descriptors import DescriptorSchema, RING_COUNT_NAMES, calculate_descriptors, mask_descriptor_values, rdkit_descriptor_names


@dataclass(frozen=True)
class DescriptorSelectionConfig:
    artifacts_dir: Path
    max_molecules: int | None = None
    seed: int = 42
    max_invalid_fraction: float = .01
    max_std_to_iqr: float = 100.
    max_abs_to_p99: float = 1.e6
    correlation_threshold: float = .98
    min_pair_samples: int = 100
    workers: int = 16
    batch_size: int = 1024
    audit_mode: str = "legacy_selection"
    max_single_variance_fraction: float | None = None

    def validate(self):
        for key in ("workers", "batch_size", "min_pair_samples"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.min_pair_samples < 2:
            raise ValueError("min_pair_samples must be at least 2")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.max_molecules is not None and (isinstance(self.max_molecules, bool)
            or not isinstance(self.max_molecules, int) or self.max_molecules < 1):
            raise ValueError("max_molecules must be null or a positive integer")
        for key in ("max_invalid_fraction", "max_std_to_iqr", "max_abs_to_p99", "correlation_threshold"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite and numeric")
        if not 0 <= self.max_invalid_fraction < 1 or not 0 < self.correlation_threshold < 1:
            raise ValueError("Invalid missing fraction or correlation threshold")
        if self.max_std_to_iqr <= 1 or self.max_abs_to_p99 <= 1:
            raise ValueError("Heavy-tail ratios must exceed 1")
        if not isinstance(self.audit_mode, str) or self.audit_mode not in {"legacy_selection", "prepared_review"}:
            raise ValueError("Unknown descriptor audit_mode")
        fraction = self.max_single_variance_fraction
        if fraction is not None and (isinstance(fraction, bool) or not isinstance(fraction, (float, int))
                                     or not math.isfinite(fraction) or not 0 < fraction <= 1):
            raise ValueError("max_single_variance_fraction must be in (0, 1]")

    def to_dict(self):
        result = {**asdict(self), "artifacts_dir": str(self.artifacts_dir)}
        if self.audit_mode == "legacy_selection":
            result.pop("audit_mode")
        if self.max_single_variance_fraction is None:
            result.pop("max_single_variance_fraction")
        return result


def load_selection_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - set(DescriptorSelectionConfig.__dataclass_fields__):
        raise ValueError("Unknown descriptor selection fields")
    config = DescriptorSelectionConfig(**{**raw, "artifacts_dir": Path(raw["artifacts_dir"])})
    config.validate()
    return config


def selection_source(config):
    from common.identity import semantic_identity, validate_semantic_identity
    from common.io import sha256_file
    from .identity import metadata_identity, validate_feature_generation_runtime

    config.validate()
    root = config.artifacts_dir
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("kind") != "ilume_stage1_corpus" or metadata.get("format_version") != 4:
        raise ValueError("Descriptor audit requires a v4 prepared Stage1 corpus")
    validate_feature_generation_runtime(metadata)
    if list(rdkit_descriptor_names()) != metadata["descriptor_raw_names"]:
        raise ValueError("Descriptor names/order differ from prepared authority")
    manifest_hash = sha256_file(root / "manifest.csv")
    if manifest_hash != metadata["artifact_hashes"]["manifest.csv"]:
        raise ValueError("Stage1 manifest hash mismatch")
    corpus = metadata_identity(metadata, "corpus", context="Stage1 descriptor audit")
    validate_semantic_identity(corpus)
    recipe = config.to_dict()
    for key in ("artifacts_dir", "workers", "batch_size"):
        recipe.pop(key)
    preprocessing = {}
    if config.audit_mode == "prepared_review":
        schema_path = root / "descriptor_schema.json"
        if sha256_file(schema_path) != metadata["artifact_hashes"].get("descriptor_schema.json"):
            raise ValueError("Descriptor schema SHA mismatch")
        schema = DescriptorSchema.load(schema_path, expected_raw_names=rdkit_descriptor_names())
        if schema.value_policy == "raw":
            raise ValueError("Prepared descriptor review requires the new value policy; rerun prepare")
        if schema.value_policy != corpus["payload"]["descriptor"].get("value_policy", "raw"):
            raise ValueError("Prepared descriptor schema/corpus value policy mismatch")
        preprocessing = {"prepared_schema_sha256": sha256_file(schema_path), "value_policy": schema.value_policy}
    return metadata, semantic_identity("stage1.descriptor-selection.audit.v1", {
        "corpus_identity": corpus["hash"], "manifest_sha256": manifest_hash,
        "metadata_sha256": sha256_file(root / "metadata.json"),
        "descriptor_names": metadata["descriptor_raw_names"], "recipe": recipe,
        "fit_split": "train", "raw_values": "recomputed-float64",
        "sampling": "sorted-PCG64-choice-train-indices", "numpy_version": np.__version__,
        "correlation": "pairwise-complete-pearson-greedy-kept-representatives",
        **preprocessing,
    })


def _descriptor_row(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid canonical SMILES in prepared manifest")
    return calculate_descriptors(molecule, rdkit_descriptor_names())


def _descriptor_policy_row(smiles, *, value_policy):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid canonical SMILES in prepared manifest")
    names = rdkit_descriptor_names()
    raw = calculate_descriptors(molecule, names)
    return raw, mask_descriptor_values(raw, names, molecule, value_policy)


def collect_raw_descriptors(config, metadata, destination, reporter, schema=None):
    """Stream train rows through bounded CPU batches; never invert float32 features."""
    population = int(metadata["summary"]["train"])
    size = min(population, config.max_molecules or population)
    if size < 2:
        raise ValueError("Descriptor selection needs at least two training molecules")
    selected = None if size == population else set(np.random.default_rng(config.seed).choice(
        population, size=size, replace=False).tolist())
    matrix = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float64,
                                        shape=(size, len(metadata["descriptor_raw_names"])))
    processed = (None if schema is None else np.lib.format.open_memmap(
        destination.with_name("processed_descriptors.npy"), mode="w+", dtype=np.float64,
        shape=(size, schema.selected_dim)))
    worker = _descriptor_row if schema is None else partial(_descriptor_policy_row, value_policy=schema.value_policy)
    sample_hash = hashlib.sha256()
    pool = (ProcessPoolExecutor(max_workers=config.workers,
            mp_context=multiprocessing.get_context("spawn")) if config.workers > 1 else nullcontext(None))
    written, seen, pending = 0, 0, []
    with pool as executor, reporter.bar(total=size, desc="Raw train RDKit descriptors", unit="molecule") as progress:
        def flush():
            nonlocal written
            values = list(map(worker, pending)) if executor is None else list(
                executor.map(worker, pending, chunksize=16))
            if schema is None:
                matrix[written:written + len(values)] = values
            else:
                raw, masked = zip(*values, strict=True)
                matrix[written:written + len(values)] = raw
                processed[written:written + len(values)] = schema.select(np.asarray(masked))
            written += len(values)
            progress.update(len(values))
            reporter.emit_json({"event": "descriptor_selection", "completed": written, "total": size})
            pending.clear()

        with (config.artifacts_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["split"] == "valid":
                    continue
                if row["split"] != "train":
                    raise ValueError("Unknown split in Stage1 manifest")
                take = selected is None or seen in selected
                seen += 1
                if not take:
                    continue
                sample_hash.update((row["sample_id"] + "\0" + row["canonical_smiles"] + "\n").encode())
                pending.append(row["canonical_smiles"])
                if len(pending) == config.batch_size:
                    flush()
            if pending:
                flush()
    if seen != population or written != size:
        raise ValueError("Manifest training population differs from prepared metadata")
    matrix.flush()
    if processed is not None:
        processed.flush()
    return matrix, sample_hash.hexdigest()


def select_descriptor_columns(values, names, config):
    """Fit only the supplied train matrix; zero IQR alone never removes sparse columns."""
    config.validate()
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("Invalid descriptor matrix/names")
    statistics, reasons, duplicates, reviews = [], {}, {}, {}
    review_mode = config.audit_mode == "prepared_review"
    for index, name in enumerate(names):
        column = np.asarray(values[:, index])
        valid = np.isfinite(column)
        finite = column[valid]
        row = {"name": name, "finite_count": len(finite),
               "invalid_fraction": (len(column) - len(finite)) / len(column)}
        statistics.append(row)
        if not len(finite):
            reasons[name] = "all_non_finite"
            continue
        max_abs = float(np.max(np.abs(finite)))
        row["max_abs"] = max_abs
        if max_abs > np.sqrt(np.finfo(np.float64).max):
            reasons[name] = "unsafe_float64_square"
            continue
        q = np.quantile(finite, [.01, .25, .5, .75, .99])
        row.update(zip(("p01", "p25", "p50", "p75", "p99"), q.tolist()))
        row["min"], row["max"] = float(finite.min()), float(finite.max())
        scaled = finite / (max_abs or 1.)
        row["mean"], row["std"] = float(scaled.mean() * max_abs), float(scaled.std() * max_abs)
        iqr = q[3] - q[1]
        abs_p99 = float(np.quantile(np.abs(finite), .99))
        with np.errstate(over="ignore", divide="ignore"):
            std_ratio = float(row["std"] / iqr) if iqr > 0 else None
            peak_ratio = float(max_abs / abs_p99) if abs_p99 > 0 else None
        row["std_to_iqr"] = std_ratio if std_ratio is not None and math.isfinite(std_ratio) else None
        row["max_abs_to_p99"] = peak_ratio if peak_ratio is not None and math.isfinite(peak_ratio) else None
        row["zero_iqr"] = bool(iqr == 0)
        single_fraction = (float(max((scaled.min() - scaled.mean()) ** 2,
                                     (scaled.max() - scaled.mean()) ** 2)
                                 / (len(finite) * scaled.var())) if row["std"] > 0 else None)
        if config.max_single_variance_fraction is not None:
            row["max_single_variance_fraction"] = single_fraction
        flags = []
        if ((std_ratio is not None and std_ratio > config.max_std_to_iqr)
                or (peak_ratio is not None and peak_ratio > config.max_abs_to_p99)):
            flags.append("heavy_tail")
        if (config.max_single_variance_fraction is not None and single_fraction is not None
                and single_fraction > config.max_single_variance_fraction):
            flags.append("single_observation_dominance")
        if row["invalid_fraction"] > config.max_invalid_fraction:
            flags.append("excess_non_finite")
        if review_mode:
            reviews[name] = flags
        if not review_mode and row["invalid_fraction"] > config.max_invalid_fraction:
            reasons[name] = "excess_non_finite"
        elif row["min"] == row["max"] or row["std"] == 0:
            reasons[name] = "zero_variance"
        elif not review_mode and flags:
            reasons[name] = "heavy_tail"
        else:
            digest = hashlib.sha256(valid.tobytes() + finite.tobytes()).hexdigest()
            if digest in duplicates:
                reasons[name] = f"duplicate_of:{duplicates[digest]}"
            else:
                duplicates[digest] = name

    candidates = [i for i, name in enumerate(names) if name not in reasons]
    count = len(candidates)
    paired = np.zeros((count, count))
    sums, squares, products = (np.zeros_like(paired) for _ in range(3))
    means = np.asarray([statistics[i]["mean"] for i in candidates])
    scales = np.asarray([statistics[i]["std"] for i in candidates])
    for start in range(0, len(values), 8192):
        chunk = np.asarray(values[start:start + 8192, candidates])
        valid = np.isfinite(chunk)
        normalized = (np.where(valid, chunk, means) - means) / scales
        observed = valid.astype(np.float64)
        paired += observed.T @ observed
        sums += normalized.T @ observed
        squares += (normalized ** 2).T @ observed
        products += normalized.T @ normalized
    denominator = np.maximum(paired, 1.)
    variance = np.maximum(squares - sums ** 2 / denominator, 0.)
    covariance = products - sums * sums.T / denominator
    scale = np.sqrt(variance) * np.sqrt(variance.T)
    correlation = np.divide(covariance, scale, out=np.zeros_like(scale), where=scale > 0)
    correlation = np.clip(correlation, -1., 1.)
    # Every pruned column must directly match a final retained representative.
    kept, pairs = [], []
    order = sorted(range(count), key=lambda p: (-statistics[candidates[p]]["finite_count"], candidates[p]))
    for position in order:
        for representative in kept:
            if (paired[position, representative] >= config.min_pair_samples
                and abs(correlation[position, representative]) > config.correlation_threshold):
                name, target = names[candidates[position]], names[candidates[representative]]
                if review_mode:
                    reviews[name].append(f"correlated_with:{target}")
                else:
                    reasons[name] = f"correlated_with:{target}"
                pairs.append({("descriptor" if review_mode else "removed"): name, "retained": target,
                              "pearson": float(correlation[position, representative]),
                              "paired_samples": int(paired[position, representative])})
                break
        else:
            kept.append(position)
    for row in statistics:
        if review_mode:
            row["review_flags"] = ";".join(reviews.get(row["name"], []))
        row["decision"] = ("drop" if row["name"] in reasons else "review"
                           if review_mode and reviews.get(row["name"]) else "keep")
        row["reason"] = reasons.get(row["name"], "retained")
    retained = [i for i, name in enumerate(names) if name not in reasons]
    return {"raw_names": list(names), "retained_indices": retained,
            "retained_names": [names[i] for i in retained], "removal_reasons": reasons,
            **({"review_flags": {name: flags for name, flags in reviews.items() if flags}} if review_mode else {}),
            "statistics": statistics, "correlated_pairs": pairs}


def descriptor_group_audit(config, metadata, raw, values, schema, result):
    names = schema.selected_names
    stats = {row["name"]: row for row in result["statistics"]}
    means = np.asarray([stats[name].get("mean", 0.) for name in names])
    scales = np.asarray([stats[name].get("std", 1.) or 1. for name in names])
    usable = np.asarray([stats[name].get("std", 0.) > 0 for name in names])
    population, size = int(metadata["summary"]["train"]), len(values)
    selected = (None if size == population else set(np.random.default_rng(config.seed).choice(
        population, size=size, replace=False).tolist()))

    def members():
        seen = 0
        with (config.artifacts_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["split"] != "train":
                    continue
                take = selected is None or seen in selected
                seen += 1
                if take:
                    yield row

    focus = {"Phi", *RING_COUNT_NAMES}
    focus.update(name for name, flags in result["review_flags"].items()
                 if any(flag in {"heavy_tail", "single_observation_dominance"} for flag in flags))
    focus = [name for name in names if name in focus]
    extremes = {name: [] for name in focus}
    groups = {}
    total_squares = np.zeros(len(names))
    iterator = members()
    phi = names.index("Phi") if "Phi" in names else None
    for start in range(0, size, 8192):
        chunk = np.asarray(values[start:start + 8192])
        rows = list(islice(iterator, len(chunk)))
        if len(rows) != len(chunk):
            raise ValueError("Descriptor review manifest/tensor row mismatch")
        valid = np.isfinite(chunk)
        z = (np.where(valid & usable, chunk, means) - means) / scales
        total_squares += np.square(z).sum(axis=0)
        labels = np.asarray([f"{row['role']}|augmented={row['is_augmented']}" for row in rows])
        masked_phi = (np.isfinite(raw[start:start + len(chunk), schema.raw_names.index("Phi")]) & ~valid[:, phi]
                      if phi is not None else np.zeros(len(chunk), dtype=bool))
        for label in sorted(set(labels)):
            mask = labels == label
            group = groups.setdefault(label, {"molecules": 0, "phi_inapplicable": 0,
                "finite_count": np.zeros(len(names), dtype=np.int64), "squares": np.zeros(len(names))})
            group["molecules"] += int(mask.sum())
            group["phi_inapplicable"] += int(masked_phi[mask].sum())
            group["finite_count"] += valid[mask].sum(axis=0)
            group["squares"] += np.square(z[mask]).sum(axis=0)
        for name in focus:
            column = names.index(name)
            positions = np.flatnonzero(valid[:, column])
            positions = positions[np.argsort(-np.abs(z[positions, column]), kind="stable")[:20]]
            for pos in positions:
                row = rows[pos]
                extremes[name].append({"fit_index": start + int(pos), "sample_id": row["sample_id"],
                    "canonical_smiles": row["canonical_smiles"], "role": row["role"],
                    "is_augmented": row["is_augmented"], "seed_smiles": row.get("seed_smiles", ""),
                    "value": float(chunk[pos, column]), "z": float(z[pos, column])})
            extremes[name] = sorted(extremes[name], key=lambda row: (-abs(row["z"]), row["fit_index"]))[:20]
    for group in groups.values():
        group["descriptors"] = {name: {"finite_count": int(group["finite_count"][i]),
            "invalid_count": group["molecules"] - int(group["finite_count"][i]),
            "squared_deviation_share": float(group["squares"][i] / total_squares[i]) if total_squares[i] > 0 else None}
            for i, name in enumerate(names)}
        del group["finite_count"], group["squares"]
    seed_groups = {}
    for samples in extremes.values():
        for row in samples:
            key = row["seed_smiles"] or row["canonical_smiles"]
            seed_groups.setdefault(key, set()).add(row["sample_id"])
    return {"fit_molecules": size, "groups": groups, "extreme_samples": extremes,
            "extreme_seed_groups": [{"seed_smiles_or_canonical": key, "sample_ids": sorted(ids), "records": len(ids)}
                for key, ids in sorted(seed_groups.items(), key=lambda item: (-len(item[1]), item[0]))],
            "seed_scope": "seed_smiles strings on top20 records per reviewed descriptor; not a corpus-wide family estimate"}


def run_selection(config, output_dir, metadata, identity):
    from common.io import atomic_json, sha256_file
    from common.progress import ProgressReporter

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    reporter = ProgressReporter()
    schema = (DescriptorSchema.load(config.artifacts_dir / "descriptor_schema.json",
                                  expected_raw_names=metadata["descriptor_raw_names"])
              if config.audit_mode == "prepared_review" else None)
    matrix, sample_hash = collect_raw_descriptors(config, metadata, output / "raw_descriptors.npy", reporter, schema)
    audited = matrix if schema is None else np.load(output / "processed_descriptors.npy", mmap_mode="r")
    names = metadata["descriptor_raw_names"] if schema is None else schema.selected_names
    with reporter.status("Descriptor distributions and correlations"):
        result = select_descriptor_columns(audited, names, config)
    extra_files = []
    if schema is not None:
        with reporter.status("Descriptor role/augmentation and extreme-structure review"):
            grouped = descriptor_group_audit(config, metadata, matrix, audited, schema, result)
        atomic_json(output / "descriptor_groups.json", grouped)
        extra_files = ["processed_descriptors.npy", "descriptor_groups.json"]
        result["audited_names"] = list(names)
        result["value_policy"] = schema.value_policy
        result["raw_names"] = list(schema.raw_names)
        result["retained_indices"] = [schema.raw_names.index(name) for name in result["retained_names"]]
        result["removal_reasons"] = {**schema.removal_reasons, **result["removal_reasons"]}
    columns = sorted({key for row in result["statistics"] for key in row})
    with (output / "descriptor_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result.pop("statistics"))
    payload = {"kind": "ilume_stage1_descriptor_selection_audit", "format_version": 1 if schema is None else 2,
               "identity": identity, "fit_split": "train", "population": metadata["summary"]["train"],
               "fit_molecules": len(matrix), "fit_sample_hash": sample_hash,
               "sampled": len(matrix) < metadata["summary"]["train"],
               "raw_matrix_sha256": sha256_file(output / "raw_descriptors.npy"), **result}
    atomic_json(output / "descriptor_selection.json", payload)
    if selection_source(config)[1] != identity:
        raise ValueError("Descriptor source changed during audit")
    hashes = {name: sha256_file(output / name) for name in (
        "descriptor_statistics.csv", "descriptor_selection.json")}
    hashes["raw_descriptors.npy"] = payload["raw_matrix_sha256"]
    for name in extra_files:
        hashes[name] = sha256_file(output / name)
    return {"identity": identity, "population": payload["population"],
            "fit_molecules": len(matrix), "sampled": payload["sampled"],
            "raw_dim": len(result["raw_names"]), "retained_dim": len(result["retained_names"]),
            "retained_names": result["retained_names"], "removal_reasons": result["removal_reasons"],
            **({"review_flags": result["review_flags"], "value_policy": schema.value_policy,
                "audited_dim": schema.selected_dim} if schema is not None else {}),
            "artifact_hashes": hashes}
