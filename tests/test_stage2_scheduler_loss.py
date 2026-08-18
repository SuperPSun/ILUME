from __future__ import annotations

import pytest
import torch

from stage2.data import epoch_batch_schedule
from stage2.model import (
    masked_target_macro_smooth_l1_loss, molecule_equal_smooth_l1_loss,
)
from stage2.train import task_compensation_scale


class _SizedDataset:
    def __init__(self, rows: int) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return self.rows


def test_round_robin_is_complete_deterministic_and_does_not_cycle() -> None:
    datasets = {"a": _SizedDataset(5), "b": _SizedDataset(3), "c": _SizedDataset(1)}
    first = epoch_batch_schedule(datasets, 2, seed=17, epoch=3)  # type: ignore[arg-type]
    second = epoch_batch_schedule(datasets, 2, seed=17, epoch=3)  # type: ignore[arg-type]
    assert [(item.task, item.indices.tolist()) for item in first] == [
        (item.task, item.indices.tolist()) for item in second
    ]
    for task, dataset in datasets.items():
        observed = torch.cat([item.indices for item in first if item.task == task]).tolist()
        assert sorted(observed) == list(range(len(dataset)))
    assert [item.task for item in first].count("c") == 1


def test_loss_reductions_and_teacher_independence() -> None:
    predictions = torch.tensor([[2.0, 2.0], [0.0, 2.0]])
    target = torch.zeros_like(predictions)
    macro = masked_target_macro_smooth_l1_loss(
        predictions, target, torch.tensor([[True, True], [False, True]])
    )
    assert macro.item() == pytest.approx(1.5)
    molecule = molecule_equal_smooth_l1_loss(
        torch.tensor([2.0, 2.0, 2.0]), torch.zeros(3),
        torch.ones(3, dtype=torch.bool), torch.tensor([0, 1, 1]), 2,
    )
    assert molecule.item() == pytest.approx(1.5)
    compensation = task_compensation_scale(0.25, 20, 4, 10)
    assert (compensation * torch.tensor(2.0) + 0.1 * torch.tensor(3.0)).item() == pytest.approx(4.3)
