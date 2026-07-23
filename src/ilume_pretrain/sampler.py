from __future__ import annotations

import math
import random
import hashlib
from collections.abc import Iterator, Sequence
from typing import Any

from torch.utils.data import Sampler


DEFAULT_ROLE_PROBABILITIES = (0.45, 0.45, 0.10)


def allocate_role_quotas(
    num_samples: int,
    role_probabilities: Sequence[float] = DEFAULT_ROLE_PROBABILITIES,
) -> tuple[int, int, int]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    exact = [num_samples * probability for probability in role_probabilities]
    quotas = [math.floor(value) for value in exact]
    remainder_count = num_samples - sum(quotas)
    order = sorted(
        range(3),
        key=lambda role: (exact[role] - quotas[role], -role),
        reverse=True,
    )
    for role in order[:remainder_count]:
        quotas[role] += 1
    return tuple(quotas)  # type: ignore[return-value]


def minimum_samples_for_coverage(
    role_counts: Sequence[int],
    role_probabilities: Sequence[float] = DEFAULT_ROLE_PROBABILITIES,
) -> int:
    if len(role_counts) != 3 or len(role_probabilities) != 3:
        raise ValueError("Exactly three roles are required")
    minimum = max(
        math.ceil(count / probability)
        for count, probability in zip(role_counts, role_probabilities, strict=True)
        if count
    )
    while any(
        quota < count
        for quota, count in zip(
            allocate_role_quotas(minimum, role_probabilities),
            role_counts,
            strict=True,
        )
    ):
        minimum += 1
    return minimum


class RoleBalancedSampler(Sampler[int]):
    """Samples exact global role quotas with deterministic within-role cycles."""

    def __init__(
        self,
        role_ids: Sequence[int],
        num_samples: int,
        role_probabilities: Sequence[float] = DEFAULT_ROLE_PROBABILITIES,
        seed: int = 42,
        require_full_coverage: bool = False,
        shard_ids: Sequence[str] | None = None,
    ) -> None:
        if len(role_probabilities) != 3:
            raise ValueError("Exactly three role probabilities are required")
        if any(probability < 0 for probability in role_probabilities):
            raise ValueError("Role probabilities cannot be negative")
        if not math.isclose(sum(role_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("Role probabilities must sum to 1")
        self.indices_by_role = {
            role: [index for index, value in enumerate(role_ids) if value == role]
            for role in range(3)
        }
        if shard_ids is not None and len(shard_ids) != len(role_ids):
            raise ValueError("shard_ids must align with role_ids")
        effective_shards = (
            list(shard_ids)
            if shard_ids is not None
            else ["__all__"] * len(role_ids)
        )
        self.indices_by_role_shard: dict[int, dict[str, list[int]]] = {
            role: {} for role in range(3)
        }
        for index, (role, shard) in enumerate(
            zip(role_ids, effective_shards, strict=True)
        ):
            self.indices_by_role_shard[int(role)].setdefault(str(shard), []).append(
                index
            )
        serialized_shards = "\n".join(effective_shards).encode()
        self.shard_layout_hash = hashlib.sha256(serialized_shards).hexdigest()
        missing = [role for role, indices in self.indices_by_role.items() if not indices]
        if missing:
            raise ValueError(f"Dataset has no samples for roles: {missing}")
        self.num_samples = num_samples
        self.role_probabilities = tuple(role_probabilities)
        self.seed = seed
        self.epoch = 0
        self.start_offset = 0
        if require_full_coverage:
            self.validate_coverage()

    def __len__(self) -> int:
        return self.num_samples - self.start_offset

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start_offset(self, start_offset: int) -> None:
        if not 0 <= start_offset <= self.num_samples:
            raise ValueError("Sampler start_offset is outside its sample budget")
        self.start_offset = start_offset

    def quotas(self) -> tuple[int, int, int]:
        return allocate_role_quotas(self.num_samples, self.role_probabilities)

    def validate_coverage(self) -> None:
        counts = tuple(len(self.indices_by_role[role]) for role in range(3))
        quotas = self.quotas()
        if any(quota < count for quota, count in zip(quotas, counts, strict=True)):
            required = minimum_samples_for_coverage(counts, self.role_probabilities)
            raise ValueError(
                "Training sample budget cannot cover every selected entity once: "
                f"role_counts={counts}, quotas={quotas}, minimum_total_draws={required}"
            )

    def state_dict(self, start_offset: int | None = None) -> dict[str, Any]:
        quotas = self.quotas()
        return {
            "format_version": 2,
            "epoch": self.epoch,
            "start_offset": self.start_offset if start_offset is None else start_offset,
            "num_samples": self.num_samples,
            "seed": self.seed,
            "shard_layout_hash": self.shard_layout_hash,
            "quotas": list(quotas),
            "role_cycle_counts": [
                math.ceil(quotas[role] / len(self.indices_by_role[role]))
                for role in range(3)
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("format_version", 0)) != 2:
            raise ValueError("Unsupported sampler state format")
        if int(state.get("num_samples", self.num_samples)) != self.num_samples:
            raise ValueError("Sampler state sample budget does not match")
        if int(state.get("seed", self.seed)) != self.seed:
            raise ValueError("Sampler state seed does not match")
        if state.get("shard_layout_hash", self.shard_layout_hash) != self.shard_layout_hash:
            raise ValueError("Sampler state shard layout does not match")
        self.set_epoch(int(state["epoch"]))
        self.set_start_offset(int(state["start_offset"]))

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        selected_by_role: dict[int, list[int]] = {}
        for role, quota in enumerate(self.quotas()):
            shards = self.indices_by_role_shard[role]
            remaining = quota
            role_selected: list[int] = []
            while remaining:
                shard_order = sorted(shards)
                rng.shuffle(shard_order)
                for shard in shard_order:
                    local = shards[shard].copy()
                    rng.shuffle(local)
                    take = min(remaining, len(local))
                    role_selected.extend(local[:take])
                    remaining -= take
                    if remaining == 0:
                        break
            selected_by_role[role] = role_selected
        role_order = [
            role
            for role, quota in enumerate(self.quotas())
            for _ in range(quota)
        ]
        rng.shuffle(role_order)
        cursors = [0, 0, 0]
        selected: list[int] = []
        for role in role_order:
            selected.append(selected_by_role[role][cursors[role]])
            cursors[role] += 1
        return iter(selected[self.start_offset :])
