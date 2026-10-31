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
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem

from .descriptors import calculate_descriptors, rdkit_descriptor_names


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

    def to_dict(self):
        return {**asdict(self), "artifacts_dir": str(self.artifacts_dir)}


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
    return metadata, semantic_identity("stage1.descriptor-selection.audit.v1", {
        "corpus_identity": corpus["hash"], "manifest_sha256": manifest_hash,
        "metadata_sha256": sha256_file(root / "metadata.json"),
        "descriptor_names": metadata["descriptor_raw_names"], "recipe": recipe,
        "fit_split": "train", "raw_values": "recomputed-float64",
        "sampling": "sorted-PCG64-choice-train-indices", "numpy_version": np.__version__,
        "correlation": "pairwise-complete-pearson-greedy-kept-representatives",
    })


def _descriptor_row(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid canonical SMILES in prepared manifest")
    return calculate_descriptors(molecule, rdkit_descriptor_names())


def collect_raw_descriptors(config, metadata, destination, reporter):
    """Stream train rows through bounded CPU batches; never invert float32 features."""
    population = int(metadata["summary"]["train"])
    size = min(population, config.max_molecules or population)
    if size < 2:
        raise ValueError("Descriptor selection needs at least two training molecules")
    selected = None if size == population else set(np.random.default_rng(config.seed).choice(
        population, size=size, replace=False).tolist())
    matrix = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float64,
                                        shape=(size, len(metadata["descriptor_raw_names"])))
    sample_hash = hashlib.sha256()
    pool = (ProcessPoolExecutor(max_workers=config.workers,
            mp_context=multiprocessing.get_context("spawn")) if config.workers > 1 else nullcontext(None))
    written, seen, pending = 0, 0, []
    with pool as executor, reporter.bar(total=size, desc="Raw train RDKit descriptors", unit="molecule") as progress:
        def flush():
            nonlocal written
            values = list(map(_descriptor_row, pending)) if executor is None else list(
                executor.map(_descriptor_row, pending, chunksize=16))
            matrix[written:written + len(values)] = values
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
    return matrix, sample_hash.hexdigest()


def select_descriptor_columns(values, names, config):
    """Fit only the supplied train matrix; zero IQR alone never removes sparse columns."""
    config.validate()
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("Invalid descriptor matrix/names")
    statistics, reasons, duplicates = [], {}, {}
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
        if row["invalid_fraction"] > config.max_invalid_fraction:
            reasons[name] = "excess_non_finite"
        elif row["min"] == row["max"] or row["std"] == 0:
            reasons[name] = "zero_variance"
        elif ((std_ratio is not None and std_ratio > config.max_std_to_iqr)
              or (peak_ratio is not None and peak_ratio > config.max_abs_to_p99)):
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
                reasons[name] = f"correlated_with:{target}"
                pairs.append({"removed": name, "retained": target,
                              "pearson": float(correlation[position, representative]),
                              "paired_samples": int(paired[position, representative])})
                break
        else:
            kept.append(position)
    for row in statistics:
        row["decision"] = "drop" if row["name"] in reasons else "keep"
        row["reason"] = reasons.get(row["name"], "retained")
    retained = [i for i, name in enumerate(names) if name not in reasons]
    return {"raw_names": list(names), "retained_indices": retained,
            "retained_names": [names[i] for i in retained], "removal_reasons": reasons,
            "statistics": statistics, "correlated_pairs": pairs}


def run_selection(config, output_dir, metadata, identity):
    from common.io import atomic_json, sha256_file
    from common.progress import ProgressReporter

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    reporter = ProgressReporter()
    matrix, sample_hash = collect_raw_descriptors(config, metadata, output / "raw_descriptors.npy", reporter)
    with reporter.status("Descriptor distributions and correlations"):
        result = select_descriptor_columns(matrix, metadata["descriptor_raw_names"], config)
    columns = sorted({key for row in result["statistics"] for key in row})
    with (output / "descriptor_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result.pop("statistics"))
    payload = {"kind": "ilume_stage1_descriptor_selection_audit", "format_version": 1,
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
    return {"identity": identity, "population": payload["population"],
            "fit_molecules": len(matrix), "sampled": payload["sampled"],
            "raw_dim": len(result["raw_names"]), "retained_dim": len(result["retained_names"]),
            "retained_names": result["retained_names"], "removal_reasons": result["removal_reasons"],
            "artifact_hashes": hashes}
