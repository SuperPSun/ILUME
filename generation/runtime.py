"""Preparation, training and candidate evaluation for one frozen design request."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import torch
import yaml

from common.identity import semantic_identity
from common.io import atomic_json, sha256_file
from common.outputs import open_run_directory, repository_path
from common.training import canonical_json_sha256
from stage3.inference import FrozenPropertyPredictor
from .chemistry import ChemicalLibrary, FragmentEnvironment, extract_resources
from .model import DedicatedGFN
from .reward import ApplicabilityDomain, RewardOracle
from .sampling import sample_candidates
from .tasks import DesignRequest
from .training import load_checkpoint, train


PROTOCOL_KIND = "ilume_dedicated_gfn_protocol_v1"


def prepare_protocol(predictor, request: DesignRequest) -> dict:
    references = {}
    for task in request.task_ids:
        cap = predictor.capabilities[task]
        fields = ("cation", "anion", *cap.condition_columns, *(("solute",) if cap.requires_solute else ()))
        rows = {tuple(str(row[field]) for field in fields) for row in predictor.reference_rows(task)}
        if not rows:
            raise ValueError(f"No development reference rows: {task}")
        references[task] = [dict(zip(fields, values, strict=True)) for values in sorted(rows)]
    payload = {"kind": PROTOCOL_KIND, "format_version": 2, "predictor_identity": predictor.identity,
        "request": request.to_dict(),
        "capabilities": [row for row in predictor.capability_table() if row["task_id"] in request.task_ids],
        "references": references, "test_labels_used": False}
    return {"identity": canonical_json_sha256(payload), "payload": payload}


def build_components(config: dict, *, device: str, predictor=None):
    library = ChemicalLibrary.load(repository_path(config["resources"]["library_path"]))
    bundle = json.loads(repository_path(config["resources"]["protocol_path"]).read_text())
    protocol = bundle["payload"]
    if (bundle["identity"] != canonical_json_sha256(protocol) or protocol.get("format_version") != 2
            or protocol.get("kind") != PROTOCOL_KIND or protocol.get("test_labels_used") is not False):
        raise ValueError("Dedicated generation protocol hash or kind mismatch; prepare a new protocol")
    predictor = predictor or FrozenPropertyPredictor(repository_path(config["predictor"]["config_path"]),
        repository_path(config["predictor"]["checkpoint_dir"]), device=device)
    request = DesignRequest.from_dict(config["request"], predictor.capabilities)
    if canonical_json_sha256(request.to_dict()) != canonical_json_sha256(protocol["request"]):
        raise ValueError("Fixed design request differs from the prepared protocol; prepare and train a new model")
    capabilities = [row for row in predictor.capability_table() if row["task_id"] in request.task_ids]
    if (predictor.identity != protocol["predictor_identity"] or
            canonical_json_sha256(capabilities) != canonical_json_sha256(protocol["capabilities"])):
        raise ValueError("Protocol differs from the current frozen Predictor")
    if set(protocol["references"]) != set(request.task_ids):
        raise ValueError("Dedicated protocol reference tasks differ from the fixed request")
    if request.fixed_anion and request.fixed_anion not in library.anions:
        raise ValueError("Specified anion is outside the approved library")
    environment = FragmentEnvironment(library, request.constraints.max_atoms)
    model = DedicatedGFN(library, **config["model"]).to(device)
    identity = semantic_identity("generation.dedicated-gfn.v1", {"library": library.identity,
        "protocol": bundle["identity"], "predictor": predictor.identity["hash"],
        "request": request.to_dict(), "model": config["model"], "reward": config["reward"],
        "training": config["training"], "seed": config["seed"]})
    return library, bundle, predictor, request, environment, model, identity


def validate_settings(config: dict):
    if set(config) != {"seed", "predictor", "resources", "request", "model", "reward", "training", "sampling"}:
        raise ValueError("Dedicated config requires one inline request; universal task/environment fields are unsupported")
    settings = config["training"]
    if set(settings) != {"steps", "batch_size", "replay_batch_size", "replay_capacity", "learning_rate",
            "max_grad_norm", "checkpoint_interval"}:
        raise ValueError("Dedicated training settings do not support task sampling or reward relabeling")
    positive = ("steps", "batch_size", "replay_capacity", "checkpoint_interval")
    if any(not isinstance(settings[key], int) or settings[key] < 1 for key in positive):
        raise ValueError("Training counts must be positive integers")
    if not isinstance(settings["replay_batch_size"], int) or settings["replay_batch_size"] < 0:
        raise ValueError("Invalid replay batch size")
    if any(not math.isfinite(settings[key]) or settings[key] <= 0 for key in ("learning_rate", "max_grad_norm")):
        raise ValueError("Training rates must be positive")
    if config["seed"] < 0:
        raise ValueError("Seed must be nonnegative")


def main(operation: str, argv=None):
    parser = argparse.ArgumentParser(description=f"Dedicated ILUME GFlowNet {operation} for one fixed request.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    if operation == "prepare":
        parser.add_argument("--kind", required=True, choices=("resources", "protocol"))
    if operation == "train":
        parser.add_argument("--resume", help="Read a complete same-request checkpoint; write to a new output.")
    if operation in {"sample", "evaluate"}:
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--random-policy", action="store_true")
        parser.add_argument("--independent-pairing", action="store_true")
    args = parser.parse_args(argv)
    if repository_path(args.output).exists():
        raise FileExistsError("Generation output already exists")
    config_path = repository_path(args.config)
    config = yaml.safe_load(config_path.read_text())
    validate_settings(config)
    torch.manual_seed(config["seed"])
    rng = random.Random(config["seed"])
    inputs = {"config": sha256_file(config_path)}
    if operation == "prepare":
        if args.kind == "resources":
            inputs.update({name: sha256_file(repository_path(config["resources"][name]))
                for name in ("cation_path", "anion_path")})
        else:
            predictor = FrozenPropertyPredictor(repository_path(config["predictor"]["config_path"]),
                repository_path(config["predictor"]["checkpoint_dir"]), device=args.device)
            request = DesignRequest.from_dict(config["request"], predictor.capabilities)
            inputs.update(predictor=predictor.identity, request=request.to_dict())
        identity = semantic_identity("generation.dedicated-prepare.v1", {"kind": args.kind, "inputs": inputs})
    else:
        library, bundle, predictor, request, environment, model, identity = build_components(config, device=args.device)
        inputs["predictor_execution"] = getattr(predictor, "execution_identity", predictor.identity)
        if operation in {"sample", "evaluate"}:
            inputs["checkpoint"] = sha256_file(repository_path(args.checkpoint))
        elif args.resume:
            inputs["resume"] = sha256_file(repository_path(args.resume))
    operation_identity = semantic_identity("generation.dedicated-operation.v1", {"model": identity,
        "operation": operation, "inputs": inputs, "random_policy": getattr(args, "random_policy", False),
        "independent_pairing": getattr(args, "independent_pairing", False),
        "device_type": torch.device(args.device).type})
    run = open_run_directory(stage="generation", operation=operation, config_path=args.config,
        config_payload=config, semantic_identity=operation_identity, output=args.output, seed=config["seed"],
        details={"generation_identity": identity})
    started = time.monotonic()
    try:
        if operation == "prepare":
            if args.kind == "resources":
                payload = extract_resources(repository_path(config["resources"]["cation_path"]),
                    repository_path(config["resources"]["anion_path"]))
                atomic_json(run.root / "resources.draft.json", payload)
                summary = {"status": "awaiting_resource_review", "counts": {name: len(payload[name])
                    for name in ("anions", "cores", "fragments")}}
            else:
                bundle = prepare_protocol(predictor, request)
                atomic_json(run.root / "protocol.json", bundle)
                summary = {"properties": len(request.objectives), "request": request.to_dict(),
                    "protocol_identity": bundle["identity"]}
        else:
            atomic_json(run.root / "resources.json", library.payload)
            atomic_json(run.root / "protocol.json", bundle)
            atomic_json(run.root / "request.json", request.to_dict())
            domain = ApplicabilityDomain(bundle["payload"]["references"], predictor.capabilities)
            oracle = RewardOracle(predictor, domain, **config["reward"], cache_path=run.root / "prediction_cache.jsonl")
            if operation == "train":
                history = train(model, environment, oracle, request, rng, config["training"], run.root, identity,
                    resume=repository_path(args.resume) if args.resume else None)
                summary = {"completed_steps": len(history), "new_property_queries": oracle.new_queries,
                    "final_loss": history[-1]["loss"], "request": request.to_dict(), "generation_identity": identity}
            else:
                load_checkpoint(repository_path(args.checkpoint), model, identity)
                selected, metrics = sample_candidates(model, environment, oracle, request, rng,
                    **config["sampling"], raw_path=run.root / "raw_samples_0000.jsonl",
                    random_policy=args.random_policy, independent_pairing=args.independent_pairing)
                atomic_json(run.root / "candidates_0000.json", selected)
                summary = {"request": request.to_dict(), **metrics, "generation_identity": identity,
                    "reliability_interpretation": "heuristic_risk_screening",
                    "policy": "random" if args.random_policy else "gfn", "independent_pairing": args.independent_pairing}
        summary["elapsed_seconds"] = time.monotonic() - started
        run.complete(summary)
    except BaseException:
        run.fail()
        raise
