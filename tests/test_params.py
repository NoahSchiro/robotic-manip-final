"""Hypothesis-grid helpers."""

import numpy as np
import pytest

from src.estimation.params import Eta, nearest_subcell, subcells

# An L of rod positions (one corner missing) at three friction values.
SITES = [(0.0, 0.0), (0.01, 0.0), (0.0, 0.01)]
GRID = [Eta(mu, bx, by) for mu in (0.2, 0.3, 0.4) for bx, by in SITES]


def test_one_subcell_is_the_grid_itself() -> None:
    cells = subcells(GRID, 1)
    assert np.allclose(cells.points, [g.as_array() for g in GRID])
    assert np.allclose(cells.weights, np.eye(len(GRID)))


def test_subcell_points_tile_each_cell_with_weights_summing_to_one() -> None:
    cells = subcells(GRID, 2)
    assert len(cells.owner) == 8 * len(GRID)
    assert np.bincount(cells.owner).tolist() == [8] * len(GRID)
    assert np.allclose(cells.weights.sum(axis=1), 1.0)
    offsets = (cells.points - np.array([g.as_array() for g in GRID])[cells.owner]) / [
        0.1,
        0.01,
        0.01,
    ]
    assert np.allclose(np.abs(offsets), 0.25)


def test_interior_interpolation_is_exact_for_a_linear_prediction() -> None:
    # Along friction, every cell of the middle layer has both neighbours, so
    # a prediction linear in mu interpolates exactly there.
    cells = subcells(GRID, 2)
    predictions = np.array([g.mu for g in GRID])
    middle = np.isclose(cells.points[:, 0], 0.275) | np.isclose(
        cells.points[:, 0], 0.325
    )
    assert middle.any()
    assert cells.weights[middle] @ predictions == pytest.approx(cells.points[middle, 0])


def test_missing_neighbors_hold_the_prediction_constant() -> None:
    # The cell at (0.01, 0) has no neighbour at larger bx or larger by, so
    # its points beyond the grid take the owning hypothesis's bx and by.
    cells = subcells(GRID, 2)
    corner = 1 + 3  # mu 0.3, site (0.01, 0.0)
    outer = (cells.owner == corner) & (cells.points[:, 1] > 0.01)
    rows = cells.weights[outer]
    for row in rows:
        used = [GRID[i] for i in np.nonzero(row)[0]]
        assert all(g.bx == 0.01 for g in used)


def test_nearest_subcell_lies_in_the_cell_holding_eta() -> None:
    cells = subcells(GRID, 2)
    k = nearest_subcell(cells, GRID, Eta(0.31, 0.002, 0.007))
    assert GRID[cells.owner[k]] == Eta(0.3, 0.0, 0.01)
