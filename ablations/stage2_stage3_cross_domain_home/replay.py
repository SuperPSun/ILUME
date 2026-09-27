from __future__ import annotations

from typing import Any, Mapping

import torch
from torch.nn import functional as F

from common.identity import tensor_state_hash
from common.training import canonical_json_sha256
from stage2.model import molecule_equal_smooth_l1_loss
from stage3.data import stable_seed


def replay_events(total_updates: int, every: int = 4, initial_weight: float = 0.1):
    return [(update, initial_weight * (1 - (update - 1) / max(1, total_updates - 1)))
            for update in range(every, total_updates + 1, every)
            if update < total_updates]


def simulation_loss(task, predictions, dataset, indices, full_indices, *, atom_values=None,
                    atom_mask=None, atom_samples=None):
    if task == "simulation/partial_atomic_charge":
        return molecule_equal_smooth_l1_loss(predictions, atom_values, atom_mask, atom_samples,
                                             len(indices)) * (len(indices) / len(full_indices))
    targets = dataset.targets[indices].to(predictions.device)
    mask = dataset.target_mask[indices].to(predictions.device)
    if task == "simulation/simulated_qm_elec_hf":
        counts = dataset.target_mask[full_indices].sum(dim=0).to(predictions.device)
        valid = counts > 0
        values = F.smooth_l1_loss(predictions, targets, reduction="none") * mask
        return ((values.sum(dim=0) / counts.clamp_min(1)) * valid).sum() / valid.sum()
    if not bool(mask.all()):
        raise ValueError("Replay ordinary task has missing targets")
    return F.smooth_l1_loss(predictions, targets, reduction="sum") / dataset.targets[full_indices].numel()


class SimulationReplay:
    def __init__(self, datasets, cache, *, seed, fold, total_updates, weights,
                 batch_size=256, microbatch_size=128):
        self.datasets, self.cache = datasets, cache
        self.tasks = tuple(sorted(datasets))
        if len(self.tasks) != 9:
            raise ValueError("Simulation replay requires nine source tasks")
        self.seed = stable_seed(seed, fold, "simulation_replay")
        self.events = replay_events(total_updates)
        self.by_update = {update: (self.tasks[index % len(self.tasks)], weight)
                          for index, (update, weight) in enumerate(self.events)}
        self.batch_size, self.microbatch_size = batch_size, microbatch_size
        mean_weight = sum(weights.values()) / len(weights)
        self.weights = {task: weights[task] / mean_weight for task in self.tasks}
        self.positions = {task: {"cycle": 0, "cursor": 0} for task in self.tasks}
        self.consumed = 0
        self.rng = None
        self.begin_epoch()

    def expected_updates(self, through_update):
        return {f"SIM_PRIVATE:{task}": sum(1 for update, (selected, _) in self.by_update.items()
                                          if update <= through_update and selected == task)
                for task in self.tasks}

    def begin_epoch(self):
        self.records = []

    def diagnostics(self):
        state = self.state_dict()
        return {"batches": list(self.records), "total_replay_updates": self.consumed,
                "tasks": {task: {"updates": sum(item["task"] == task for item in self.records),
                                 "samples": sum(item["samples"] for item in self.records if item["task"] == task)}
                          for task in self.tasks},
                "cursor_hash": state["cursor_hash"], "rng_hash": state["rng_hash"]}

    def next_indices(self, task):
        state = self.positions[task]
        count = len(self.datasets[task])
        if count < 1:
            raise ValueError("Replay train task is empty")
        if state["cursor"] == count:
            state["cycle"] += 1
            state["cursor"] = 0
        generator = torch.Generator().manual_seed(stable_seed(self.seed, task, state["cycle"]))
        permutation = torch.randperm(count, generator=generator)
        start = state["cursor"]
        state["cursor"] = min(count, start + self.batch_size)
        return permutation[start:state["cursor"]]

    def _predict(self, model, task, dataset, indices, device):
        ids = dataset.entity_indices[indices]
        slots = self.cache["slots"][ids].to(device)
        roles = self.cache["roles"][ids].to(device)
        spec = dataset.spec
        conditions = dataset.conditions[indices].to(device)
        home = model.simulation_home
        if spec.target_level == "atom":
            objects = model.stage2_object_encoder(slots, roles)
            atoms, values, masks, samples = [], [], [], []
            for row, artifact_index in enumerate(indices.tolist()):
                entity_id = int(ids[row, 0])
                atom_states = self.cache["atoms"][entity_id].to(device)
                start, end = dataset.atom_target_offsets[artifact_index:artifact_index + 2].tolist()
                if end - start != len(atom_states):
                    raise ValueError("Replay atom cache/target order mismatch")
                atoms.append(model.simulation_atom_adapter(torch.cat((atom_states, objects[row:row + 1].expand(len(atom_states), -1)), dim=-1)))
                values.append(dataset.atom_target_values[start:end])
                masks.append(dataset.atom_target_mask[start:end])
                samples.append(torch.full((len(atom_states),), row, dtype=torch.long, device=device))
            embedding = torch.cat(atoms)
            predictions = home(task, embedding, embedding.new_empty((len(embedding), 0))).predictions
            return predictions, dict(atom_values=torch.cat(values).to(device),
                                     atom_mask=torch.cat(masks).to(device), atom_samples=torch.cat(samples))
        if spec.topology == "interaction":
            primary = model.stage2_object_encoder(slots[:, :1], roles[:, :1])
            partner = model.stage2_object_encoder(slots[:, 1:], roles[:, 1:])
        else:
            primary, partner = model.stage2_object_encoder(slots, roles), None
        count = len(spec.target_columns)
        tasks = (task,) if count == 1 else tuple(f"{task}::target_{i}" for i in range(count))
        return torch.stack([home(item, primary, conditions, partner_embedding=partner).predictions
                            for item in tasks], dim=-1), {}

    def add_gradients(self, model, assembled, update):
        if update not in self.by_update:
            return
        task, weight = self.by_update[update]
        indices = self.next_indices(task)
        dataset = self.datasets[task]
        device = next(model.parameters()).device
        parameters = tuple(parameter for parameter, owner in model.parameter_ownership().items()
                           if parameter.requires_grad and (owner.scope in {"ENCODER_STAGE2", "SIM_GLOBAL", "SIM_GROUP"}
                               or owner.label == f"SIM_PRIVATE:{task}"))
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        loss_value = 0.0
        with torch.random.fork_rng(devices=devices):
            if self.rng is None:
                torch.set_rng_state(torch.Generator().manual_seed(self.seed).get_state())
                if devices:
                    torch.cuda.set_rng_state(torch.Generator(device=device).manual_seed(self.seed).get_state(), devices[0])
            else:
                torch.set_rng_state(self.rng["cpu"])
                if devices:
                    torch.cuda.set_rng_state(self.rng["cuda"], devices[0])
            for part in indices.split(self.microbatch_size):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    prediction, atom = self._predict(model, task, dataset, part, device)
                    loss = simulation_loss(task, prediction, dataset, part, indices, **atom)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite simulation replay loss")
                gradients = torch.autograd.grad(loss * self.weights[task] * weight, parameters, allow_unused=True)
                for parameter, gradient in zip(parameters, gradients, strict=True):
                    if gradient is not None:
                        value = gradient.detach().float()
                        assembled[parameter] = assembled.get(parameter, torch.zeros_like(value)) + value
                loss_value += float(loss.detach().float().cpu())
            self.rng = {"cpu": torch.get_rng_state()}
            if devices:
                self.rng["cuda"] = torch.cuda.get_rng_state(devices[0])
        self.consumed += 1
        self.records.append({"task": task, "update": update, "samples": len(indices),
                             "weight": weight, "loss": loss_value, "task_coefficient": self.weights[task]})

    def state_dict(self):
        payload = {"positions": {task: dict(state) for task, state in self.positions.items()},
                   "consumed": self.consumed, "rng": self.rng}
        payload["cursor_hash"] = canonical_json_sha256({key: payload[key] for key in ("positions", "consumed")})
        payload["rng_hash"] = tensor_state_hash("cross-domain.replay-rng", self.rng or {})
        return payload

    def load_state_dict(self, state: Mapping[str, Any], through_update):
        expected_count = sum(update <= through_update for update, _ in self.events)
        if not isinstance(state, Mapping) or state.get("consumed") != expected_count:
            raise ValueError("Replay update count mismatch")
        expected_positions = {task: {"cycle": 0, "cursor": 0} for task in self.tasks}
        for update, _ in self.events:
            if update > through_update:
                break
            task = self.by_update[update][0]
            item = expected_positions[task]
            count = len(self.datasets[task])
            if item["cursor"] == count:
                item["cycle"] += 1
                item["cursor"] = 0
            item["cursor"] = min(count, item["cursor"] + self.batch_size)
        if (state.get("positions") != expected_positions
            or state.get("cursor_hash") != canonical_json_sha256({key: state[key] for key in ("positions", "consumed")})
            or state.get("rng_hash") != tensor_state_hash("cross-domain.replay-rng", state.get("rng") or {})
            or (expected_count > 0 and state.get("rng") is None)):
            raise ValueError("Replay cursor or RNG integrity mismatch")
        self.positions, self.consumed, self.rng = expected_positions, expected_count, state["rng"]
