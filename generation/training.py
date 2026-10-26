"""Fixed-request trajectory balance with identity-bound full-state checkpoints."""
from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
import random

import torch

from common.identity import tensor_state_hash
from common.io import atomic_json, atomic_torch_save, sha256_file
from .chemistry import State

CHECKPOINT_KIND = "ilume_dedicated_gfn_v1"


@torch.no_grad()
def rollout(model, environment, request, rng: random.Random, *, random_policy: bool = False):
    states = [State()]
    for _ in range(request.constraints.max_atoms + 4):
        state = states[-1]
        children = environment.children(state, request.constraints, request.fixed_anion)
        if not children:
            return None
        if random_policy:
            child = rng.choice(children)
        else:
            probabilities = model.log_probabilities(state, children).exp().cpu().tolist()
            child = rng.choices(children, weights=probabilities, k=1)[0]
        states.append(child)
        if child.terminal:
            return states
    raise RuntimeError("Fragment environment failed its acyclic trajectory bound")


def trajectory_balance(model, environment, request, states, log_reward):
    if states[0] != State() or not states[-1].terminal or not math.isfinite(log_reward):
        raise ValueError("TB requires a complete root-to-terminal trajectory and finite reward")
    residual = model.log_z - log_reward
    for state, child in zip(states, states[1:]):
        children = environment.children(state, request.constraints, request.fixed_anion)
        parents = environment.parents(child, request.constraints, request.fixed_anion)
        if child not in children or state not in parents:
            raise ValueError("Trajectory does not belong to the fixed-request chemistry DAG")
        residual = residual + model.log_probabilities(state, children)[children.index(child)] + math.log(len(parents))
    return residual.square()


def save_checkpoint(path: Path, model, optimizer, identity: dict, step: int, replay, rng, history, oracle):
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError("Generation checkpoints cannot be overwritten")
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    payload = {"kind": CHECKPOINT_KIND, "identity": identity, "step": step,
        "model": state, "model_state_hash": tensor_state_hash("generation.dedicated-model.v1", state),
        "optimizer": optimizer.state_dict(), "replay": list(replay), "random_state": rng.getstate(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()
            if next(model.parameters()).device.type == "cuda" else None,
        "device_type": next(model.parameters()).device.type,
        "history": history, "prediction_cache": oracle.cache, "property_queries": oracle.new_queries,
        "prediction_execution": getattr(oracle.predictor, "execution_identity", oracle.predictor.identity)}
    atomic_torch_save(path, payload)
    atomic_json(path.with_suffix(".json"), {"kind": payload["kind"], "identity": identity,
        "step": step, "artifact_sha256": sha256_file(path), "model_state_hash": payload["model_state_hash"]})


def load_checkpoint(path: Path, model, expected_identity: dict):
    manifest = json.loads(path.with_suffix(".json").read_text())
    if manifest.get("kind") != CHECKPOINT_KIND:
        raise ValueError("Dedicated GFN cannot load universal or historical checkpoint kinds")
    if manifest["artifact_sha256"] != sha256_file(path) or manifest["identity"] != expected_identity:
        raise ValueError("Generator checkpoint artifact or source identity mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["kind"] != CHECKPOINT_KIND or payload["identity"] != expected_identity or payload["step"] != manifest["step"]:
        raise ValueError("Generator checkpoint contract mismatch")
    digest = tensor_state_hash("generation.dedicated-model.v1", payload["model"])
    if digest != payload["model_state_hash"] or digest != manifest["model_state_hash"]:
        raise ValueError("Generator checkpoint tensor hash mismatch")
    model.load_state_dict(payload["model"], strict=True)
    return payload


def train(model, environment, oracle, request, rng: random.Random, settings: dict, output: Path, identity: dict,
          *, resume: Path | None = None):
    if identity.get("payload", {}).get("request") != request.to_dict():
        raise ValueError("Training request differs from the dedicated model identity")
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    replay = deque(maxlen=settings["replay_capacity"])
    history, first_step = [], 0
    if resume:
        previous = load_checkpoint(resume, model, identity)
        if previous["device_type"] != next(model.parameters()).device.type:
            raise ValueError("Training resume requires the original device type")
        if previous["prediction_execution"] != getattr(oracle.predictor, "execution_identity", oracle.predictor.identity):
            raise ValueError("Training resume Predictor execution identity mismatch")
        optimizer.load_state_dict(previous["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(next(model.parameters()).device)
        replay.extend(previous["replay"])
        rng.setstate(previous["random_state"])
        torch.set_rng_state(previous["torch_rng"])
        if previous["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
        history, first_step = previous["history"], previous["step"]
        oracle.cache = previous["prediction_cache"]
        oracle.new_queries = previous["property_queries"]
        if oracle.cache_path:
            with oracle.cache_path.open("x", encoding="utf-8") as cache:
                for item in oracle.cache.values():
                    cache.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
    model.train()
    if first_step > settings["steps"]:
        raise ValueError("Resume checkpoint exceeds the frozen training budget")
    with (output / "metrics.jsonl").open("x", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        for step in range(first_step + 1, settings["steps"] + 1):
            losses, failed = [], 0
            for _ in range(settings["batch_size"]):
                states = rollout(model, environment, request, rng)
                if states is None:
                    failed += 1
                    continue
                replay.append(states)
                candidate = environment.describe(states[-1])
                reward = oracle.evaluate(candidate["cation"], candidate["anion"], request)
                losses.append(trajectory_balance(model, environment, request, states, reward["log_reward"]))
            for _ in range(settings["replay_batch_size"]):
                if not replay:
                    break
                states = rng.choice(replay)
                candidate = environment.describe(states[-1])
                reward = oracle.evaluate(candidate["cation"], candidate["anion"], request)
                losses.append(trajectory_balance(model, environment, request, states, reward["log_reward"]))
            if not losses:
                raise ValueError("No complete legal trajectories; review resources and constraints")
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite dedicated TB loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            row = {"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm),
                "failed_trajectories": failed, "new_property_queries": oracle.new_queries, "replay_size": len(replay)}
            history.append(row)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            if step % settings["checkpoint_interval"] == 0:
                save_checkpoint(output / f"checkpoint_step_{step:07d}.pt", model, optimizer, identity, step, replay, rng, history, oracle)
    save_checkpoint(output / "gfn_final.pt", model, optimizer, identity, settings["steps"], replay, rng, history, oracle)
    return history
