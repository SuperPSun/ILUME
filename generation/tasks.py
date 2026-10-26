"""Validated, fixed design requests for dedicated GFlowNet training."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

from .chemistry import Constraints, canonical


@dataclass(frozen=True)
class Objective:
    task: str
    direction: str
    unit: str
    threshold: float | None = None
    weight: float = 1.0


@dataclass
class DesignRequest:
    objectives: tuple[Objective, ...]
    conditions: dict[str, dict[str, float]]
    solutes: dict[str, str] = field(default_factory=dict)
    constraints: Constraints = field(default_factory=Constraints)
    fixed_anion: str = ""

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(o.task for o in self.objectives))

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["objectives"] = list(payload["objectives"])
        payload["constraints"] = {key: list(value) if isinstance(value, tuple) else value
            for key, value in payload["constraints"].items()}
        return payload

    @classmethod
    def from_dict(cls, payload: dict, capabilities: dict):
        if set(payload) - {"objectives", "conditions", "solutes", "constraints", "fixed_anion"}:
            raise ValueError("Unknown design request fields")
        objectives = tuple(Objective(**{**o,
            "threshold": None if o.get("threshold") is None else float(o["threshold"]),
            "weight": float(o.get("weight", 1.0))}) for o in payload["objectives"])
        tasks = {o.task for o in objectives}
        if not tasks or len(tasks) != len(objectives):
            raise ValueError("Requests require unique objectives")
        if not math.isfinite(sum(o.weight for o in objectives)):
            raise ValueError("Non-finite total preference weight")
        conditions = payload.get("conditions", {})
        solutes = payload.get("solutes", {})
        if set(conditions) != tasks or set(solutes) - tasks:
            raise ValueError("Request conditions and objectives differ")
        for objective in objectives:
            if objective.task not in capabilities or objective.direction not in {"maximize", "minimize"}:
                raise ValueError("Unknown property or optimization direction")
            cap = capabilities[objective.task]
            if not math.isfinite(objective.weight) or objective.weight <= 0 or objective.unit != cap.unit:
                raise ValueError("Objective requires a positive weight and explicit matching unit")
            if objective.threshold is not None:
                cap.to_native(float(objective.threshold), objective.unit)
            current = conditions[objective.task]
            if set(current) != set(cap.condition_columns) or any(not math.isfinite(float(v)) for v in current.values()):
                raise ValueError("Missing, extra or non-finite operating conditions")
            if cap.requires_solute != (objective.task in solutes):
                raise ValueError("Missing or unexpected solute")
        raw = payload.get("constraints", {})
        constraints = Constraints(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})
        return cls(tuple(sorted(objectives, key=lambda o: o.task)),
            {t: {k: float(v) for k, v in values.items()} for t, values in conditions.items()},
            {t: canonical(s) for t, s in solutes.items()}, constraints, payload.get("fixed_anion", ""))
