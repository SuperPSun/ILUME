from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import sqlite3
import time
from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from common.identity import require_compatible_identity, semantic_identity
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.progress import ProgressReporter


ELECTRONIC_COLUMNS = ("HOMO_eV", "LUMO_eV", "ESP_max", "ESP_min", "ESP_std", "ESP_pos_frac", "Dipole", "Quadrupole", "q_max", "q_min", "q_std", "q_pos_frac", "gap_eV")
ELECTRONIC_SOURCES = (("homo", ELECTRONIC_COLUMNS[:1], "PBE/TZVP"), ("lumo", ELECTRONIC_COLUMNS[1:2], "PBE/TZVP"), ("simulated_qm_elec_hf", ELECTRONIC_COLUMNS[2:], "HF"))


def electronic_source_contract(config):
    return {task: {"split": "train", "sha256": sha256_file(config.auxiliary.simulation_dir / task / "train.csv"), "columns": list(columns), "method": method} for task, columns, method in ELECTRONIC_SOURCES}


def load_electronic_labels(config):
    labels = {}
    for task, columns, _ in ELECTRONIC_SOURCES:
        with (config.auxiliary.simulation_dir / task / "train.csv").open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"SMILES", *columns} <= set(reader.fieldnames or ()):
                raise ValueError(f"Electronic train columns missing: {task}")
            for row in reader:
                mol = Chem.MolFromSmiles(row["SMILES"])
                if mol is None:
                    raise ValueError("Invalid electronic train SMILES")
                key = Chem.MolToSmiles(mol, canonical=True)
                values = labels.setdefault(key, np.full(13, np.nan))
                for column in columns:
                    raw = row[column]
                    value = float(raw) if raw.strip() else float("nan")
                    index = ELECTRONIC_COLUMNS.index(column)
                    if np.isfinite(value):
                        if np.isfinite(values[index]) and values[index] != value:
                            raise ValueError(f"Conflicting electronic labels: {key}/{column}")
                        values[index] = value
    return labels


def fit_electronic_targets(config, connection, output_dir):
    labels = load_electronic_labels(config)
    observed = set()
    for row in connection.execute("SELECT canonical_smiles FROM records WHERE split='train' AND reasons='[]'"):
        if row[0] in labels:
            observed.add(row[0])
    values = np.stack([labels[key] for key in sorted(observed)]) if observed else np.empty((0, 13))
    counts = np.isfinite(values).sum(0)
    means = np.divide(np.nansum(values, 0), counts, out=np.zeros(13), where=counts > 0)
    scales = np.sqrt(np.divide(np.nansum((values - means) ** 2, 0), counts, out=np.ones(13), where=counts > 0))
    scales[scales == 0] = 1
    stats = {"columns": list(ELECTRONIC_COLUMNS), "sources": electronic_source_contract(config), "mean": means.tolist(), "scale": scales.tolist(), "count": counts.tolist()}
    atomic_json(Path(output_dir) / "electronic_scaler.json", stats)
    return {key: {"electronic": torch.from_numpy(np.where(np.isfinite(raw), (raw - means) / scales, 0).astype(np.float32)), "electronic_valid": torch.from_numpy(np.isfinite(raw) & (counts > 0))} for key, raw in labels.items()}


def empty_auxiliary_targets():
    return {"electronic": torch.zeros(13), "electronic_valid": torch.zeros(13, dtype=torch.bool), "unimol": torch.zeros(768), "unimol_valid": torch.tensor(False)}


def generate_conformer(smiles, seed):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(hashlib.sha256(f"{seed}:{smiles}".encode()).hexdigest()[:8], 16) % 2147483647
    if AllChem.EmbedMolecule(molecule, parameters) != 0:
        raise ValueError("conformer_failed")
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        AllChem.MMFFOptimizeMolecule(molecule)
    elif AllChem.UFFHasAllMoleculeParams(molecule):
        AllChem.UFFOptimizeMolecule(molecule)
    else:
        raise ValueError("forcefield_parameters_unavailable")
    if not np.isfinite(molecule.GetConformer().GetPositions()).all():
        raise ValueError("nonfinite_coordinates")
    return molecule


def _teacher_input(smiles, seed, featurize):
    try:
        molecule = generate_conformer(smiles, seed)
        if featurize:
            from unimol_tools.data.conformer import mol2unimolv2
            return mol2unimolv2(molecule, max_atoms=Chem.RemoveHs(molecule).GetNumAtoms(), remove_hs=True), None
        return molecule, None
    except (ValueError, RuntimeError, AssertionError, KeyError, IndexError) as error:
        return None, f"{type(error).__name__}: {error}"


def _teacher_inputs(keys, seed, featurize, executor, prefetch):
    if executor is None:
        for key in keys:
            yield _teacher_input(key, seed, featurize)
        return
    # Ordered, bounded work: CPU workers prepare the next batch during GPU inference.
    pending = deque()
    iterator = iter(keys)
    for key in iterator:
        pending.append(executor.submit(_teacher_input, key, seed, featurize))
        if len(pending) == prefetch:
            break
    while pending:
        future = pending.popleft()
        key = next(iterator, None)
        if key is not None:
            pending.append(executor.submit(_teacher_input, key, seed, featurize))
        yield future.result()


class UniMolTeacher:
    """Offline-only adapter; refuses implicit downloads and atom cropping."""

    def __init__(self, config, device):
        checkpoint = config.auxiliary.teacher_checkpoint
        if not checkpoint.is_file() or tuple(checkpoint.parts[-3:]) != ("modelzoo", "84M", "checkpoint.pt"):
            raise ValueError("Supply the local Uni-Mol2 modelzoo/84M/checkpoint.pt; automatic download is disabled")
        if importlib.metadata.version("unimol_tools") != config.auxiliary.teacher_version:
            raise ValueError("Uni-Mol tools version mismatch; use the isolated pinned teacher environment")
        from unimol_tools.models import unimolv2
        from unimol_tools.data.conformer import mol2unimolv2
        if Path(unimolv2.MODEL_CONFIG_V2["weight"]["84m"]) != Path("modelzoo/84M/checkpoint.pt"):
            raise ValueError("Uni-Mol2 weight layout mismatch; automatic download is disabled")
        unimolv2.WEIGHT_DIR = str(checkpoint.parents[2])
        class OfflineTeacher(unimolv2.UniMolV2Model):
            def load_pretrained_weights(self, path, strict=False):
                # Local trusted source only; avoid the upstream torch.load default
                # changing with PyTorch versions. Prediction heads are not teachers.
                state = torch.load(path, map_location="cpu", weights_only=False)
                state = state.get("model", state.get("model_state_dict", state))
                state = {key: value for key, value in state.items() if not key.startswith(("classification_head.", "classification_heads."))}
                missing, _ = self.load_state_dict(state, strict=False)
                if any(not key.startswith(("classification_head.", "classification_heads.")) for key in missing):
                    raise ValueError("Uni-Mol2 pretrained encoder tensor set is incomplete")
        self.model = OfflineTeacher(model_size="84m").to(device).eval()
        self.model.requires_grad_(False)
        self.featurize = mol2unimolv2
        self.device = torch.device(device)

    @torch.inference_mode()
    def __call__(self, molecules):
        features = [self.featurize(mol, max_atoms=Chem.RemoveHs(mol).GetNumAtoms(), remove_hs=True) for mol in molecules]
        return self.predict_features(features)

    @torch.inference_mode()
    def predict_features(self, features):
        batch, _ = self.model.batch_collate_fn([(item, 0) for item in features])
        result = self.model(**{key: value.to(self.device) for key, value in batch.items()}, return_repr=True)["cls_repr"].float().cpu()
        if result.shape != (len(features), 768) or not torch.isfinite(result).all():
            raise ValueError("Invalid Uni-Mol2 teacher output")
        return result


def teacher_recipe(config, corpus_metadata):
    return semantic_identity("stage1.unimol2-cache.v4", {
        "corpus_identity": corpus_metadata["semantic"]["identities"]["corpus"]["hash"],
        "teacher": "unimol2-84m-cls-768", "version": config.auxiliary.teacher_version,
        "checkpoint_sha256": sha256_file(config.auxiliary.teacher_checkpoint),
        "seed": config.data.seed, "rdkit_version": rdBase.rdkitVersion,
        "conformer": "ETKDGv3-MMFF-else-UFF-no-2d-no-cropping-v1",
        "shard_size": config.auxiliary.teacher_shard_size,
    })


def prepare_teacher_cache(config, *, device="cpu", batch_size=32, workers=1, limit=None, output=None, teacher=None):
    if workers < 1:
        raise ValueError("Teacher --workers must be positive")
    if not config.is_dual_view or batch_size < 1 or (limit is not None and (limit < 1 or output is None)):
        raise ValueError("Teacher preparation requires dual_view_v4; audits need a separate --output and positive --limit")
    root = Path(output) if output is not None else config.auxiliary.teacher_cache
    if limit is not None and root.resolve() == config.auxiliary.teacher_cache.resolve():
        raise ValueError("Audit output must not overwrite the formal teacher cache")
    metadata = json.loads((config.data.artifacts_dir / "metadata.json").read_text())
    from .identity import validate_feature_generation_runtime
    if metadata.get("format_version") != 4 or metadata.get("feature_generation_contract", {}).get("architecture") != "dual_view_v4":
        raise ValueError("Teacher cache requires the v4 prepared corpus")
    validate_feature_generation_runtime(metadata)
    identity = teacher_recipe(config, metadata)
    root.mkdir(parents=True, exist_ok=True)
    recipe_path = root / "recipe.json"
    recipe = {"identity": identity, "audit_limit": limit}
    if recipe_path.exists():
        if json.loads(recipe_path.read_text()) != recipe:
            raise ValueError("Teacher cache resume recipe mismatch")
    else:
        atomic_json(recipe_path, recipe)
    if (root / "manifest.json").exists():
        cache = TeacherCache(root, metadata, validate_all=True)
        require_compatible_identity(identity, cache.manifest["identity"], context="Uni-Mol cache recipe")
        return cache.manifest
    connection = sqlite3.connect(root / "index.sqlite")
    connection.execute("CREATE TABLE IF NOT EXISTS structures(smiles TEXT PRIMARY KEY, shard INTEGER, offset INTEGER)")
    with (config.data.artifacts_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        connection.executemany("INSERT OR IGNORE INTO structures(smiles) VALUES (?)", ((row["canonical_smiles"],) for row in reader))
    connection.commit()
    total = connection.execute("SELECT COUNT(*) FROM structures").fetchone()[0]
    count = min(total, limit) if limit is not None else total
    teacher = teacher if teacher is not None else UniMolTeacher(config, device)
    shards = []
    success = 0
    started = time.perf_counter()
    reporter = ProgressReporter()
    processed = 0
    last_log = started
    def log_progress(committed):
        nonlocal last_log
        last_log = time.perf_counter()
        elapsed = last_log - started
        if not reporter.interactive:
            print(json.dumps({"event": "teacher_progress", "processed": processed, "total": count,
                              "committed": committed, "valid": success, "failed": processed - success,
                              "elapsed_seconds": elapsed, "molecules_per_second": processed / max(elapsed, 1e-9),
                              "workers": workers, "batch_size": batch_size}, sort_keys=True), flush=True)
    cursor = connection.execute("SELECT smiles FROM structures ORDER BY smiles LIMIT ?", (count,))
    # Never fork a CUDA-initialized process. Only the parent owns the teacher model.
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=torch.set_num_threads, initargs=(1,)) if workers > 1 else nullcontext(None)
    log_progress(0)
    with pool as executor, reporter.bar(total=count, desc="Uni-Mol2 teacher", unit="molecule") as progress:
        while keys := [row[0] for row in cursor.fetchmany(config.auxiliary.teacher_shard_size)]:
            shard_id = len(shards)
            filename = f"shard_{shard_id:06d}.pt"
            path = root / filename
            sidecar = path.with_suffix(".json")
            if path.exists():
                record = json.loads(sidecar.read_text())
                if record.get("sha256") != sha256_file(path):
                    raise ValueError("Teacher shard hash mismatch")
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload["identity_hash"] != identity["hash"] or payload["smiles"] != keys:
                    raise ValueError("Teacher shard identity/order mismatch")
                processed += len(keys)
                success += record["valid"]
                progress.update(len(keys))
            else:
                vectors = torch.zeros((len(keys), 768))
                valid = torch.zeros(len(keys), dtype=torch.bool)
                failures = {}
                inputs = _teacher_inputs(keys, config.data.seed, isinstance(teacher, UniMolTeacher), executor,
                                         prefetch=max(2 * batch_size, 2 * workers))
                for begin in range(0, len(keys), batch_size):
                    molecules, positions = [], []
                    for offset in range(begin, min(begin + batch_size, len(keys))):
                        molecule, failure = next(inputs)
                        if failure is None:
                            molecules.append(molecule)
                            positions.append(offset)
                        else:
                            failures[str(offset)] = failure
                    if positions:
                        values = teacher.predict_features(molecules) if isinstance(teacher, UniMolTeacher) else teacher(molecules)
                        if values.shape != (len(positions), 768) or not torch.isfinite(values).all():
                            raise ValueError("Teacher returned invalid representations")
                        vectors[positions] = values
                        valid[positions] = True
                    completed = min(batch_size, len(keys) - begin)
                    processed += completed
                    success += len(positions)
                    progress.update(completed)
                    if time.perf_counter() - last_log >= 30:
                        log_progress(sum(shard["count"] for shard in shards))
                payload = {"identity_hash": identity["hash"], "smiles": keys, "embeddings": vectors, "valid": valid, "failures": failures}
                atomic_torch_save(path, payload)
                record = {"path": filename, "sha256": sha256_file(path), "count": len(keys), "valid": int(valid.sum())}
                atomic_json(sidecar, record)
            connection.executemany("UPDATE structures SET shard=?, offset=? WHERE smiles=?", ((shard_id, offset, key) for offset, key in enumerate(keys)))
            connection.commit()
            shards.append(record)
            log_progress(processed)
    connection.close()
    elapsed = time.perf_counter() - started
    manifest = {"kind": "ilume_stage1_unimol2_cache_v4", "identity": identity, "complete": limit is None, "attempted": count, "unique_structures": total, "valid": success, "failed": count - success, "elapsed_seconds": elapsed, "molecules_per_second": count / max(elapsed, 1e-9), "embedding_bytes": count * 768 * 4, "index_sha256": sha256_file(root / "index.sqlite"), "shards": shards}
    atomic_json(root / "manifest.json", manifest)
    return manifest


class TeacherCache:
    def __init__(self, root, corpus_metadata, *, require_complete=False, validate_all=False):
        from common.identity import validate_semantic_identity

        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        manifest = self.manifest
        validate_semantic_identity(manifest["identity"])
        if (manifest.get("kind") != "ilume_stage1_unimol2_cache_v4"
            or manifest["identity"]["payload"]["corpus_identity"] != corpus_metadata["semantic"]["identities"]["corpus"]["hash"]
            or (require_complete and (not manifest["complete"] or manifest["attempted"] != manifest["unique_structures"]))
            or manifest["index_sha256"] != sha256_file(self.root / "index.sqlite")):
            raise ValueError("Teacher cache corpus/coverage/index mismatch")
        self.cache = OrderedDict()
        self.verified = set()
        if validate_all:
            for shard, record in enumerate(manifest["shards"]):
                if sha256_file(self.root / record["path"]) != record["sha256"]:
                    raise ValueError("Teacher shard integrity mismatch")
                self.verified.add(shard)
        self.connection = None
        self.process = None

    def __getstate__(self):
        return {**self.__dict__, "cache": OrderedDict(), "connection": None, "process": None}

    def attach(self, sample):
        # Each loader process owns a read-only connection and memory-mapped shards.
        if self.process != os.getpid():
            self.connection = sqlite3.connect(f"file:{self.root / 'index.sqlite'}?mode=ro", uri=True)
            self.process = os.getpid()
        row = self.connection.execute("SELECT shard, offset FROM structures WHERE smiles=?", (sample["canonical_smiles"],)).fetchone()
        if row is None or row[0] is None:
            raise ValueError("Teacher cache is missing a corpus molecule")
        shard, offset = row
        if shard not in self.cache:
            record = self.manifest["shards"][shard]
            path = self.root / record["path"]
            if shard not in self.verified and sha256_file(path) != record["sha256"]:
                raise ValueError("Teacher shard integrity mismatch")
            self.verified.add(shard)
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            if payload["identity_hash"] != self.manifest["identity"]["hash"]:
                raise ValueError("Teacher shard identity mismatch")
            self.cache[shard] = payload
            if len(self.cache) > 128:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        payload = self.cache[shard]
        if payload["smiles"][offset] != sample["canonical_smiles"]:
            raise ValueError("Teacher molecule order mismatch")
        targets = {**sample["auxiliary_targets"], "unimol": payload["embeddings"][offset], "unimol_valid": payload["valid"][offset]}
        return {**sample, "auxiliary_targets": targets}
