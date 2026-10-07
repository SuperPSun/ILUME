"""Train-only atom supervision attached without modifying the Stage1 corpus."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from common.atom_targets import (
    PARTIAL_CHARGE_MAPPING_CONTRACT, NoAtomMappingError, load_structure_manifest,
    load_verify_parse_and_map, verify_structure,
)
from common.identity import require_compatible_identity, semantic_identity, tensor_state_hash
from common.io import atomic_json, atomic_torch_save, sha256_file
from common.progress import ProgressReporter
from .data import PreparedCorpusDataset
from .features import ROLE_TO_ID


PARTIAL_CHARGE_KIND = "ilume_stage1_partial_charge_targets_v4"


def charge_source_contract(config, split="train"):
    if split not in {"train", "valid"}:
        raise ValueError("Partial-charge supervision accepts train/valid only")
    path = config.auxiliary.simulation_dir / "partial_atomic_charge" / f"{split}.csv"
    manifest = config.auxiliary.partial_charge_manifest
    entries = load_structure_manifest(manifest)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"SMILES", "mol_id"} <= set(reader.fieldnames or ()):
            raise ValueError("Partial-charge CSV requires SMILES and mol_id")
        ids = sorted({row["mol_id"].strip() for row in reader})
    structures = {}
    for mol_id in ids:
        if mol_id not in entries:
            raise ValueError(f"Partial-charge structure missing: {mol_id}")
        entry = entries[mol_id]
        verify_structure(entry)
        structures[mol_id] = {"sha256": entry.sha256, "size_bytes": entry.size_bytes}
    return {"split": split, "csv_sha256": sha256_file(path),
            "manifest_sha256": sha256_file(manifest), "structures": structures,
            "mapping": PARTIAL_CHARGE_MAPPING_CONTRACT["hash"],
            "observations": "all-source-rows-independent-v2",
            "mol2_atom_section_alias": "MOLM",
            "mol2_bond_section_alias": "MOLD",
            "unmapped_policy": "skip-no-isomorphism-audit-v1"}


def load_charge_rows(config, split="train"):
    """Retain mapped observations; audit and skip only absent graph isomorphism."""
    if split not in {"train", "valid"}:
        raise ValueError("Partial-charge supervision accepts train/valid only")
    entries = load_structure_manifest(config.auxiliary.partial_charge_manifest)
    path = config.auxiliary.simulation_dir / "partial_atomic_charge" / f"{split}.csv"
    rows, audit = {}, []
    reporter = ProgressReporter()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"SMILES", "mol_id"} <= set(reader.fieldnames or ()):
            raise ValueError("Partial-charge CSV requires SMILES and mol_id")
        with reporter.bar(total=len(entries), desc=f"Partial-charge {split} mapping", unit="molecule") as progress:
            for row in reader:
                molecule = Chem.MolFromSmiles(row["SMILES"])
                if molecule is None:
                    raise ValueError(f"Invalid partial-charge SMILES: {row['mol_id']}")
                key = Chem.MolToSmiles(molecule, canonical=True)
                charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
                role = "cation" if charge > 0 else "anion" if charge < 0 else "neutral"
                if row.get("role", role) != role or int(row.get("formal_charge", charge)) != charge:
                    raise ValueError(f"Partial-charge role/formal-charge mismatch: {row['mol_id']}")
                mol_id = row["mol_id"].strip()
                if mol_id not in entries:
                    raise ValueError(f"Partial-charge structure missing: {mol_id}")
                entry = entries[mol_id]
                provenance = {"canonical_smiles": key, "mol_id": mol_id, "split": split,
                              "csv_line": reader.line_num, "structure_file": entry.path.name,
                              "structure_sha256": entry.sha256}
                try:
                    result = load_verify_parse_and_map(entry, key)
                except NoAtomMappingError as error:
                    audit.append({**provenance, "status": "skipped",
                                  "reason": "no_graph_isomorphism", "error": str(error)})
                    progress.update(1)
                    continue
                values = np.asarray(result.charges, dtype=np.float64)
                rows.setdefault(key, []).append({"canonical_smiles": key, "mol_id": mol_id,
                                                 "role_id": ROLE_TO_ID[role], "targets": values})
                audit.append({**provenance, "status": "mapped",
                              "atom_count": len(values), "mapping_status": result.mapping_status,
                              "bond_match_mode": result.bond_match_mode,
                              "mapping_count_lower_bound": result.mapping_count_lower_bound})
                progress.update(1)
    reporter.emit_json({"event": "partial_charge_mapping", "split": split,
                        "rows": len(audit), "unique_molecules": len(rows),
                        "mapped_rows": sum(map(len, rows.values())),
                        "skipped_rows": sum(row["status"] == "skipped" for row in audit)})
    return rows, audit


def partial_charge_recipe(config, metadata):
    return semantic_identity("stage1.partial-charge.recipe.v4", {
        "corpus_identity": metadata["semantic"]["identities"]["corpus"]["hash"],
        "manifest_sha256": metadata["artifact_hashes"]["manifest.csv"],
        "source": charge_source_contract(config),
        "join": "canonical-exact-no-seed-propagation",
        "observations": "all-source-rows-independent-v2",
        "statistics": "matched-stage1-train-atoms-population-std",
    })


def prepare_partial_charge(config, output_dir=None):
    if not config.is_dual_view or config.loss.lambda_partial_charge <= 0:
        raise ValueError("Partial-charge preparation requires enabled dual_view_v4 supervision")
    dataset = PreparedCorpusDataset(config.data.artifacts_dir, "train")
    recipe = partial_charge_recipe(config, dataset.metadata)
    root = Path(output_dir) if output_dir is not None else config.auxiliary.partial_charge_cache
    if root.exists() and any(root.iterdir()):
        cached = PartialChargeCache(config, dataset.metadata, root=root)
        return cached.manifest
    rows, audit = load_charge_rows(config)
    matched, training = set(), set()
    with (config.data.artifacts_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = row["canonical_smiles"]
            if key in rows:
                matched.add(key)
                if row["split"] == "train":
                    training.add(key)
    atoms = np.concatenate([row["targets"] for key in sorted(training) for row in rows[key]]) if training else np.empty(0)
    mean, scale = (float(atoms.mean()), float(atoms.std())) if len(atoms) else (0., 1.)
    scale = scale or 1.
    targets = {key: torch.from_numpy(np.stack([((row["targets"] - mean) / scale).astype(np.float32) for row in rows[key]]))
               for key in sorted(matched)}
    scaler = {"mean": mean, "scale": scale, "atom_count": len(atoms),
              "molecule_count": len(training), "observation_count": sum(len(rows[key]) for key in training)}
    state_hash = tensor_state_hash("stage1.partial-charge.targets.v4", {"targets": targets, "scaler": scaler})
    identity = semantic_identity("stage1.partial-charge.materialization.v4", {
        "recipe": recipe["hash"], "state_hash": state_hash})
    root.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(root / "targets.pt", {"kind": PARTIAL_CHARGE_KIND, "format_version": 2,
                                          "targets": targets, "scaler": scaler, "identity": identity})
    atomic_json(root / "mapping_audit.json", audit)
    manifest = {"kind": PARTIAL_CHARGE_KIND, "format_version": 2, "recipe": recipe,
                "identity": identity, "state_hash": state_hash, "scaler": scaler,
                "matched_molecules": len(matched), "source_molecules": len(rows),
                "source_observations": sum(map(len, rows.values())),
                "attempted_observations": len(audit),
                "skipped_observations": sum(row["status"] == "skipped" for row in audit),
                "artifact_hashes": {name: sha256_file(root / name) for name in ("targets.pt", "mapping_audit.json")}}
    atomic_json(root / "metadata.json", manifest)
    return manifest


class PartialChargeCache:
    def __init__(self, config, metadata, *, root=None):
        root = Path(root) if root is not None else config.auxiliary.partial_charge_cache
        self.manifest = json.loads((root / "metadata.json").read_text())
        if self.manifest.get("kind") != PARTIAL_CHARGE_KIND or self.manifest.get("format_version") != 2:
            raise ValueError("Unsupported Stage1 partial-charge sidecar")
        require_compatible_identity(partial_charge_recipe(config, metadata), self.manifest["recipe"],
                                    context="Stage1 partial-charge source/corpus")
        for name in ("targets.pt", "mapping_audit.json"):
            if sha256_file(root / name) != self.manifest["artifact_hashes"][name]:
                raise ValueError(f"Partial-charge artifact hash mismatch: {name}")
        payload = torch.load(root / "targets.pt", map_location="cpu", weights_only=False)
        self.targets, self.scaler = payload["targets"], payload["scaler"]
        actual = tensor_state_hash("stage1.partial-charge.targets.v4", {"targets": self.targets, "scaler": self.scaler})
        if actual != self.manifest["state_hash"] or self.scaler != self.manifest["scaler"]:
            raise ValueError("Partial-charge tensor/scaler hash mismatch")
        identity = semantic_identity("stage1.partial-charge.materialization.v4", {
            "recipe": self.manifest["recipe"]["hash"], "state_hash": actual})
        require_compatible_identity(identity, self.manifest["identity"], context="Partial-charge sidecar")
        require_compatible_identity(identity, payload["identity"], context="Partial-charge tensor payload")
        if payload.get("kind") != PARTIAL_CHARGE_KIND or payload.get("format_version") != 2:
            raise ValueError("Unsupported partial-charge tensor payload")

    def attach(self, sample):
        count = len(sample["atom_categorical"])
        values = self.targets.get(sample["canonical_smiles"])
        valid = values is not None and self.scaler["atom_count"] > 0
        if values is not None and (values.ndim != 2 or values.shape[1] != count):
            raise ValueError(f"Partial-charge atom count mismatch: {sample['sample_id']}")
        return {**sample, "auxiliary_targets": {**sample["auxiliary_targets"],
                "partial_charge": values if valid else torch.zeros(1, count),
                "partial_charge_valid": torch.full(values.shape if valid else (1, count), valid, dtype=torch.bool)}}
