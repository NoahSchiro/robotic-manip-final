"""Hypothesis-grid helpers."""

import numpy as np

from src.params import Eta, grid_neighbors


def test_neighbors_are_one_step_along_one_parameter() -> None:
    # An L of rod positions (one corner missing) at two friction values.
    sites = [(0.0, 0.0), (0.01, 0.0), (0.0, 0.01)]
    grid = [Eta(mu, bx, by) for mu in (0.2, 0.3) for bx, by in sites]
    pairs = {tuple(pair) for pair in grid_neighbors(grid).tolist()}
    assert pairs == {(0, 1), (0, 2), (3, 4), (3, 5), (0, 3), (1, 4), (2, 5)}
    assert np.bincount(grid_neighbors(grid).ravel()).tolist() == [3, 2, 2, 3, 2, 2]
