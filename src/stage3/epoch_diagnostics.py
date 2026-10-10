from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .model import GLOBAL, Stage3SparseModel, group_owner


@dataclass
class _MeanMax:
    count: int = 0
    total: float = 0.0
    maximum: float | None = None

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def result(self) -> dict[str, float | None]:
        return {"mean": self.total / self.count if self.count else None, "max": self.maximum}


@dataclass
class _Contribution:
    samples: int = 0
    weight_sum: float = 0.0

    def result(self, samples: int, steps: int) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "sample_fraction": self.samples / samples if samples else None,
            "weight_sum": self.weight_sum,
            "weight_mean": self.weight_sum / steps if steps else None,
        }


class _SharedSamples:
    def __init__(self, tasks: Mapping[str, str], *, groups: Sequence[str] = ()):
        self.steps = self.samples = 0
        self.tasks = {task: _Contribution() for task in tasks}
        self.groups = {group: _Contribution() for group in groups}
        self.task_groups = tasks

    def add(self, sizes: Mapping[str, int]) -> None:
        total = sum(sizes.values())
        if not total:
            return
        self.steps += 1
        self.samples += total
        group_sizes: dict[str, int] = {}
        for task, size in sizes.items():
            self.tasks[task].samples += size
            self.tasks[task].weight_sum += size / total
            group = self.task_groups[task]
            group_sizes[group] = group_sizes.get(group, 0) + size
        for group, size in group_sizes.items():
            if group in self.groups:
                self.groups[group].samples += size
                self.groups[group].weight_sum += size / total

    def result(self) -> dict[str, Any]:
        result = {
            "steps": self.steps, "samples": self.samples,
            "tasks": {task: value.result(self.samples, self.steps) for task, value in self.tasks.items()},
        }
        if self.groups:
            result["groups"] = {group: value.result(self.samples, self.steps) for group, value in self.groups.items()}
        return result


class EpochDiagnostics:
    """Accumulate existing CPU norms and batch lengths without touching tensors."""

    def __init__(self, model: Stage3SparseModel, tasks: Sequence[str], max_grad_norm: float):
        self.steps = 0
        self.max_grad_norm = max_grad_norm
        self.trainable: dict[str, bool] = {}
        for parameter, owner in model.parameter_ownership().items():
            self.trainable[owner.label] = self.trainable.get(owner.label, False) or parameter.requires_grad
        self.owner_pre = {owner: _MeanMax() for owner in sorted(self.trainable)}
        self.owner_post = {owner: _MeanMax() for owner in self.owner_pre}
        self.clip_counts = dict.fromkeys(self.owner_pre, 0)
        self.total_pre, self.total_post = _MeanMax(), _MeanMax()
        self.task_norms = {task: _MeanMax() for task in tasks}
        self.task_samples = dict.fromkeys(tasks, 0)
        self.task_steps = dict.fromkeys(tasks, 0)
        private_only = set(getattr(model, "simulation_tasks", ()))
        self.task_groups = {task: model.task_specs[task].meta_group for task in tasks if task not in private_only}
        self.shared = {GLOBAL.label: _SharedSamples(self.task_groups, groups=model.groups)}
        for group in model.groups:
            self.shared[group_owner(group).label] = _SharedSamples({
                task: value for task, value in self.task_groups.items() if value == group
            })

    def record(
        self, sizes: Mapping[str, int], task_norms: Mapping[str, float],
        clip_values: tuple[float, float, dict[str, float], dict[str, float]],
        *, shared: bool,
    ) -> None:
        self.steps += 1
        pre, post, owner_pre, owner_post = clip_values
        self.total_pre.add(pre)
        self.total_post.add(post)
        for owner, value in owner_pre.items():
            self.owner_pre[owner].add(value)
            self.owner_post[owner].add(owner_post[owner])
            # Match clip_grad_norm_'s coefficient, including its denominator epsilon.
            self.clip_counts[owner] += int(self.max_grad_norm > 0 and value + 1e-6 > self.max_grad_norm)
        for task, size in sizes.items():
            self.task_samples[task] += size
            self.task_steps[task] += 1
        for task, value in task_norms.items():
            self.task_norms[task].add(value)
        if shared:
            experimental = {task: size for task, size in sizes.items() if task in self.task_groups}
            for owner, values in self.shared.items():
                if self.trainable[owner]:
                    values.add({task: size for task, size in experimental.items() if task in values.tasks})

    def result(self) -> dict[str, Any]:
        return {
            "diagnostics_version": 2,
            "epoch_gradient_stats": {
                "steps": self.steps,
                "clip_enabled": self.max_grad_norm > 0,
                "max_grad_norm": self.max_grad_norm,
                "total": {"pre_norm": self.total_pre.result(), "post_norm": self.total_post.result()},
                "owners": {
                    owner: {
                        "trainable_steps": self.steps if self.trainable[owner] else 0,
                        "gradient_steps": pre.count,
                        "pre_norm": pre.result(), "post_norm": self.owner_post[owner].result(),
                        "clip_count": self.clip_counts[owner],
                        "clip_rate": self.clip_counts[owner] / pre.count if pre.count else None,
                    } for owner, pre in self.owner_pre.items()
                },
                "tasks": {
                    task: {"samples": self.task_samples[task], "steps": self.task_steps[task],
                           "gradient_norm": norms.result()} for task, norms in self.task_norms.items()
                },
            },
            "shared_sample_contributions": {owner: value.result() for owner, value in self.shared.items()},
        }
