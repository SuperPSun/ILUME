"""Deterministic 2D dual-graph topology and corpus-bound, resumable sidecars."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json
import multiprocessing

import torch

from common.identity import require_compatible_identity, semantic_identity
from common.io import atomic_json, atomic_torch_save, sha256_file


DGT_CONTRACT = {
    "version": "dgt-2d-spd-ring-v1",
    "reference_commit": "2f5a4c3ef2e376d60238f481c36a87eeb5a581e8",
    "shortest_path_cutoff": 8, "unreachable_or_distant": 9,
    "chordless_ring_max_size": 18,
    "attention": "channelwise-pair-score-and-value",
    "feature_fusion": "incident-sum", "normalization": "LayerNorm",
    "activation": "GELU", "readout": "atom-bond-mean-linear",
}
TOPOLOGY_KIND = "ilume_stage1_dgt_topology"


def _pair_codes(neighbors):
    count = len(neighbors)
    distances = torch.full((count, count), 9, dtype=torch.uint8)
    rings = torch.zeros((count, count), dtype=torch.uint8)
    for source in range(count):
        distances[source, source] = 0
        visited = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            if visited[node] == 8:
                continue
            for nxt in sorted(neighbors[node]):
                if nxt not in visited:
                    visited[nxt] = visited[node] + 1
                    distances[source, nxt] = visited[nxt]
                    queue.append(nxt)

    # Enumerate induced cycles once, starting at their smallest vertex.
    def extend(path, start):
        for nxt in sorted(neighbors[path[-1]]):
            if nxt <= start or nxt in path:
                continue
            contacts = neighbors[nxt].intersection(path[:-1])
            if contacts == {start}:
                if len(path) + 1 <= 18 and path[1] < nxt:
                    cycle = path + [nxt]
                    for left in cycle:
                        for right in cycle:
                            if left != right:
                                rings[left, right] = 1 if right in neighbors[left] else 2
            elif not contacts and len(path) < 17:
                extend(path + [nxt], start)

    for start in range(count):
        extend([start], start)
    return distances, rings


def build_dgt_topology(atom_count: int, bond_index: torch.Tensor):
    """Use connectivity only; categorical labels, roles and coordinates are absent."""
    edges = bond_index.cpu().T.tolist()
    neighbors = [set() for _ in range(atom_count)]
    incident = [[] for _ in range(atom_count)]
    for bond, (left, right) in enumerate(edges):
        neighbors[left].add(right)
        neighbors[right].add(left)
        incident[left].append(bond)
        incident[right].append(bond)
    line_neighbors = [set() for _ in edges]
    line_edges, shared_atoms = [], []
    for atom, bonds in enumerate(incident):
        for left in bonds:
            for right in bonds:
                if left != right:
                    line_neighbors[left].add(right)
                    line_edges.append((left, right))
                    shared_atoms.append(atom)
    atom_spd, atom_ring = _pair_codes(neighbors)
    bond_spd, bond_ring = _pair_codes(line_neighbors)
    return {
        "atom_spd": atom_spd, "atom_ring": atom_ring,
        "bond_spd": bond_spd, "bond_ring": bond_ring,
        "line_index": torch.tensor(line_edges, dtype=torch.int32).reshape(-1, 2).T.contiguous(),
        "shared_atom": torch.tensor(shared_atoms, dtype=torch.int32),
    }


def _topology_arrays(atom_count, bond_index):
    # NumPy IPC avoids one shared-memory file descriptor per tensor storage.
    topology = build_dgt_topology(atom_count, torch.from_numpy(bond_index))
    return {name: value.numpy() for name, value in topology.items()}


def graph_structure_hash(samples):
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(json.dumps([sample["sample_id"], len(sample["atom_categorical"])],
                                 separators=(",", ":")).encode())
        digest.update(sample["bond_index"].contiguous().numpy().tobytes())
    return digest.hexdigest()


def topology_recipe(config, metadata):
    from .identity import metadata_identity, validate_feature_generation_runtime
    if config.model.graph_encoder_type != "dgt":
        raise ValueError("DGT topology preparation requires graph_encoder_type=dgt")
    validate_feature_generation_runtime(metadata)
    corpus = metadata_identity(metadata, "corpus", context="DGT corpus")
    if corpus["payload"]["descriptor"] != config.to_dict()["descriptor"]:
        raise ValueError("DGT corpus descriptor recipe differs; rerun Stage1 prepare")
    return semantic_identity("stage1.dgt-topology.v1", {
        "corpus_identity": corpus["hash"],
        "sampler_layout_identity": metadata_identity(metadata, "sampler_layout", context="DGT corpus")["hash"],
        "source_shards": metadata["shard_hashes"], "algorithm": DGT_CONTRACT,
    })


def prepare_dgt_topology(config, output):
    from .data import PreparedCorpusDataset
    from tqdm.auto import tqdm
    dataset = PreparedCorpusDataset(config.data.artifacts_dir, "all", 1)
    recipe = topology_recipe(config, dataset.metadata)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    progress_path = root / "progress.json"
    if manifest_path.exists():
        return DGTTopologyCache(config, dataset.metadata, root=root, validate_all=True).manifest
    progress = {"identity": recipe, "shards": {}}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        require_compatible_identity(recipe, progress["identity"], context="DGT topology resume")
    (root / "shards").mkdir(exist_ok=True)
    workers = config.preparation.workers
    pool_context = (ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
                    if workers > 1 else nullcontext(None))
    with pool_context as pool:
        for index, shard in enumerate(tqdm(dataset.shards, desc="DGT topology shards")):
            source = shard["path"]
            if source in progress["shards"]:
                item = progress["shards"][source]
                if sha256_file(root / item["path"]) != item["sha256"]:
                    raise ValueError(f"DGT topology shard hash mismatch: {source}")
                continue
            target = root / "shards" / f"topology_{index:06d}.pt"
            if target.exists():
                raise ValueError(f"Uncommitted DGT topology shard exists: {target}; use a new output")
            samples = dataset._load_shard(source)
            counts = [len(row["atom_categorical"]) for row in samples]
            bonds = [row["bond_index"] for row in samples]
            records = ([{name: torch.from_numpy(value) for name, value in arrays.items()}
                        for arrays in pool.map(_topology_arrays, counts,
                                               (edge.numpy() for edge in bonds), chunksize=32)] if pool is not None
                       else [build_dgt_topology(count, edge) for count, edge in zip(counts, bonds, strict=True)])
            payload = {"kind": TOPOLOGY_KIND, "format_version": 1, "identity": recipe,
                       "sample_ids": [row["sample_id"] for row in samples],
                       "graph_hash": graph_structure_hash(samples), "records": records}
            atomic_torch_save(target, payload)
            progress["shards"][source] = {"path": str(target.relative_to(root)),
                                          "sha256": sha256_file(target), "count": len(samples),
                                          "graph_hash": payload["graph_hash"]}
            atomic_json(progress_path, progress)
    manifest = {"kind": TOPOLOGY_KIND, "format_version": 1, "complete": True, **progress}
    atomic_json(manifest_path, manifest)
    return manifest


class DGTTopologyCache:
    def __init__(self, config, metadata, *, root=None, validate_all=False):
        self.root = Path(root) if root is not None else config.auxiliary.graph_topology_cache
        if self.root is None or not (self.root / "manifest.json").is_file():
            raise ValueError("DGT topology cache missing; run stage1/prepare.py --dgt-topology-only")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if (self.manifest.get("kind") != TOPOLOGY_KIND or self.manifest.get("format_version") != 1
                or self.manifest.get("complete") is not True):
            raise ValueError("Incomplete or unsupported DGT topology cache")
        require_compatible_identity(topology_recipe(config, metadata), self.manifest["identity"], context="DGT topology source")
        if set(self.manifest["shards"]) != set(metadata["shard_hashes"]):
            raise ValueError("DGT topology shard set mismatch")
        for source, item in self.manifest["shards"].items():
            path = self.root / item["path"]
            if not path.is_file() or (validate_all and sha256_file(path) != item["sha256"]):
                raise ValueError(f"DGT topology shard missing or corrupt: {source}")

    @property
    def materialization_identity(self):
        return semantic_identity("stage1.dgt-topology.materialization.v1", {
            "recipe": self.manifest["identity"]["hash"], "shards": self.manifest["shards"],
        })

    def attach_shard(self, source, samples):
        item = self.manifest["shards"][source]
        path = self.root / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"DGT topology shard hash mismatch: {source}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require_compatible_identity(self.manifest["identity"], payload["identity"], context="DGT topology shard")
        if (payload.get("kind") != TOPOLOGY_KIND or payload.get("format_version") != 1
                or payload["sample_ids"] != [row["sample_id"] for row in samples]
                or len(payload["records"]) != len(samples) or item["count"] != len(samples)
                or payload["graph_hash"] != item["graph_hash"]
                or payload["graph_hash"] != graph_structure_hash(samples)):
            raise ValueError("DGT topology sample order or structure mismatch")
        return [{**row, "dgt_topology": topology} for row, topology in zip(samples, payload["records"], strict=True)]
