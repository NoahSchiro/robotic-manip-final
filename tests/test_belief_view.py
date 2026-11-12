import numpy as np
import pytest

from src.belief_view import grid_steps, mu_marginal, rod_marginal
from src.object import Object, Rod
from src.params import Eta, GridSpec, make_grid

GRID = [
    Eta(0.2, 0.0, 0.0),
    Eta(0.2, 0.01, 0.0),
    Eta(0.4, 0.0, 0.0),
    Eta(0.4, 0.01, 0.0),
]
PROBS = np.array([0.1, 0.2, 0.3, 0.4])


def test_rod_marginal_sums_over_friction() -> None:
    positions, probs = rod_marginal(GRID, PROBS)
    assert positions.tolist() == [[0.0, 0.0], [0.01, 0.0]]
    assert probs == pytest.approx([0.4, 0.6])


def test_mu_marginal_sums_over_rod_positions() -> None:
    mus, probs = mu_marginal(GRID, PROBS)
    assert mus.tolist() == [0.2, 0.4]
    assert probs == pytest.approx([0.3, 0.7])


def test_marginals_of_the_default_grid_keep_all_probability() -> None:
    obj = Object(np.random.default_rng(0), rod=Rod(0.05, 0.0, 0.2))
    grid = make_grid(GridSpec(), obj.parts)
    probs = np.random.default_rng(1).dirichlet(np.ones(len(grid)))
    positions, rod_probs = rod_marginal(grid, probs)
    mus, mu_probs = mu_marginal(grid, probs)
    assert len(positions) * len(mus) == len(grid)
    assert rod_probs.sum() == pytest.approx(1.0)
    assert mu_probs.sum() == pytest.approx(1.0)


def test_grid_steps_are_each_parameters_spacing() -> None:
    assert grid_steps(GRID) == pytest.approx([0.2, 0.01, np.inf])
