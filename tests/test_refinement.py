from __future__ import annotations

import pytest

from common.refinement import (
    refinement_cosine_factor,
    refinement_geometry,
    selection_record,
)


@pytest.mark.parametrize(
    ("epochs", "boundary", "refinement"),
    ((5, 4, 1), (10, 8, 2), (20, 16, 4), (50, 40, 10), (100, 80, 20)),
)
def test_refinement_geometry_matches_registered_budgets(
    epochs: int, boundary: int, refinement: int
) -> None:
    assert refinement_geometry(epochs, 0.20) == (boundary, refinement)


def test_refinement_cosine_has_no_warmup_and_honors_floor() -> None:
    assert refinement_cosine_factor(0, 10, 0.05) == pytest.approx(1.0)
    assert refinement_cosine_factor(10, 10, 0.05) == pytest.approx(0.05)
    assert refinement_cosine_factor(5, 10, 0.05) == pytest.approx(0.525)


def test_selection_record_only_marks_strict_improvement() -> None:
    tied = selection_record(
        metric_name="normalized_mae",
        boundary_epoch=8,
        boundary_metric=0.2,
        selected_epoch=8,
        best_metric=0.2,
    )
    improved = selection_record(
        metric_name="normalized_mae",
        boundary_epoch=8,
        boundary_metric=0.2,
        selected_epoch=9,
        best_metric=0.1,
    )
    assert tied["improved"] is False
    assert improved["improved"] is True
