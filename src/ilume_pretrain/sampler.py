from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class RoleBalancedSampler(Sampler[int]):
    """Samples an exact role quota for a requested sample budget."""

    def __init__(
        self,
        role_ids: Sequence[int],
        num_samples: int,
        role_probabilities: Sequence[float] = (0.5, 0.4, 0.1),
        seed: int = 42,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if len(role_probabilities) != 3:
            raise ValueError("Exactly three role probabilities are required")
        if any(probability < 0 for probability in role_probabilities):
            raise ValueError("Role probabilities cannot be negative")
        total = sum(role_probabilities)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("Role probabilities must sum to 1")
        self.indices_by_role = {
            role: [index for index, value in enumerate(role_ids) if value == role]
            for role in range(3)
        }
        missing = [role for role, indices in self.indices_by_role.items() if not indices]
        if missing:
            raise ValueError(f"Dataset has no samples for roles: {missing}")
        self.num_samples = num_samples
        self.role_probabilities = tuple(role_probabilities)
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _quotas(self) -> list[int]:
        exact = [
            self.num_samples * probability
            for probability in self.role_probabilities
        ]
        quotas = [math.floor(value) for value in exact]
        remainder_count = self.num_samples - sum(quotas)
        order = sorted(
            range(3),
            key=lambda role: (exact[role] - quotas[role], -role),
            reverse=True,
        )
        for role in order[:remainder_count]:
            quotas[role] += 1
        return quotas

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        selected: list[int] = []
        for role, quota in enumerate(self._quotas()):
            pool = self.indices_by_role[role]
            remaining = quota
            while remaining:
                cycle = pool.copy()
                rng.shuffle(cycle)
                take = min(remaining, len(cycle))
                selected.extend(cycle[:take])
                remaining -= take
        rng.shuffle(selected)
        return iter(selected)
