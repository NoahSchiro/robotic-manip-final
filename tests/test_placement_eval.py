"""Placement condition names and fixed etas (no physics)."""

import numpy as np
import pytest

from src.evaluation.placement import (
    Condition,
    farthest_eta,
    nominal_eta,
    parse_condition,
)


def test_conditions_parse() -> None:
    assert parse_condition("true_eta") == Condition("true_eta", None, 0, "true")
    assert parse_condition("none") == Condition("none", None, 0, "estimated")
    assert parse_condition("nominal_eta").eta_source == "nominal"
    assert parse_condition("farthest_eta").eta_source == "farthest"
    assert parse_condition("learned:3") == Condition(
        "learned:3", "learned", 3, "estimated"
    )
    assert parse_condition("long_fast:5").prober == "long_fast"


def test_fixed_etas_are_grid_points_at_the_center_and_far_end() -> None:
    from src.estimation.params import Eta, GridSpec, grid_spans, make_grid
    from src.experiment import reference_object

    grid = make_grid(GridSpec(), reference_object().parts)
    nominal = nominal_eta(grid)
    assert nominal in grid
    assert nominal.mu == pytest.approx(0.5)
    truth = Eta(0.15, grid[0].bx, grid[0].by)
    worst = farthest_eta(grid, truth)
    assert worst in grid
    assert worst.mu == pytest.approx(0.9)
    distance = np.linalg.norm((worst.as_array() - truth.as_array()) / grid_spans(grid))
    assert all(
        np.linalg.norm((g.as_array() - truth.as_array()) / grid_spans(grid))
        <= distance + 1e-12
        for g in grid
    )


@pytest.mark.parametrize("name", ["random", "random:0", "greedy:2", "learned:x", ""])
def test_bad_conditions_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        parse_condition(name)
