from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest

from src.estimation.dynamics import DynamicsModel
from src.estimation.estimator import (
    GridBayesFilter,
    gaussian_log_likelihood,
    normalized_log_probs,
    squared_mahalanobis,
    student_t_log_likelihood,
)
from src.estimation.params import Eta
from src.world.observation import PushObservation

GRID = [Eta(0.2, 0.0, 0.0), Eta(0.4, 0.0, 0.0), Eta(0.6, 0.0, 0.0)]
# A one-unit error in x or y is two standard deviations.
R = np.diag([0.25, 0.25, 1.0, 1.0, 1.0, 1.0])
MEASURED = np.zeros(6)


class FixedPredictions(DynamicsModel):
    """A prediction model whose hypotheses predict x errors set by the test."""

    def __init__(self, x_errors: Sequence[float]) -> None:
        self.x_errors = x_errors

    def predict(self, etas: Sequence[Eta], obs: PushObservation) -> np.ndarray:
        predicted = np.zeros((len(etas), 6))
        predicted[:, 0] = -np.asarray(self.x_errors)
        return predicted


class FakeObservation:
    """The only part of a PushObservation the filter reads."""

    def measured(self) -> np.ndarray:
        return MEASURED


OBS = cast(PushObservation, FakeObservation())


def make_filter(x_errors: Sequence[float], nu: float | None = 3.0) -> GridBayesFilter:
    """A filter on the grid points alone, so its updates are plain Bayes rule."""
    return GridBayesFilter(GRID, FixedPredictions(x_errors), R, nu, n_subcells=1)


def test_squared_mahalanobis_scales_by_the_covariance() -> None:
    errors = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    assert squared_mahalanobis(errors, np.linalg.inv(R)) == pytest.approx([4.0, 8.0])


def test_normalized_log_probs_sum_to_one_without_underflowing() -> None:
    log_probs = normalized_log_probs(np.array([-1e5, -1e5 - 1.0, -1e5 - 2.0]))
    assert np.exp(log_probs).sum() == pytest.approx(1.0)
    assert log_probs[0] - log_probs[1] == pytest.approx(1.0)


def test_student_t_matches_its_formula() -> None:
    q = np.array([0.0, 2.0, 50.0])
    expected = -0.5 * (3.0 + 6) * np.log(1.0 + q / 3.0)
    assert student_t_log_likelihood(q, 3.0, 6) == pytest.approx(expected)


def test_student_t_approaches_the_gaussian_as_nu_grows() -> None:
    q = np.array([0.0, 1.0, 4.0])
    assert student_t_log_likelihood(q, 1e7, 6) == pytest.approx(
        gaussian_log_likelihood(q), abs=1e-4
    )


def test_student_t_penalizes_a_huge_error_far_less_than_the_gaussian() -> None:
    q = np.array([1e4])
    assert student_t_log_likelihood(q, 3.0, 6)[0] > -50.0
    assert gaussian_log_likelihood(q)[0] == -5000.0


def test_belief_starts_uniform() -> None:
    belief = make_filter([0.0, 0.0, 0.0])
    assert belief.probs == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert belief.entropy() == pytest.approx(np.log(3))


def test_update_favors_the_hypothesis_predicting_best() -> None:
    belief = make_filter([1.0, 0.0, 0.5])
    belief.update(OBS)
    assert belief.map_eta() == GRID[1]
    assert belief.probs.sum() == pytest.approx(1.0)
    assert belief.probs[1] > belief.probs[2] > belief.probs[0]
    assert belief.entropy() < np.log(3)


def test_updates_match_the_bayes_rule_product() -> None:
    belief = make_filter([1.0, 0.0, 0.5], nu=None)
    belief.update(OBS)
    belief.update(OBS)
    # Each update multiplies in exp(-q / 2), with q = 4 x^2 under R.
    expected = np.exp(2 * -0.5 * 4.0 * np.array([1.0, 0.0, 0.5]) ** 2)
    assert belief.probs == pytest.approx(expected / expected.sum())


def test_equal_predictions_leave_the_belief_alone() -> None:
    belief = make_filter([0.3, 0.3, 0.3])
    belief.update(OBS)
    assert belief.probs == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_one_outlier_push_cannot_eliminate_a_hypothesis_under_student_t() -> None:
    errors = [100.0, 0.0, 0.0]
    student_t = make_filter(errors)
    student_t.update(OBS)
    gaussian = make_filter(errors, nu=None)
    gaussian.update(OBS)
    assert student_t.probs[0] > 1e-30
    assert gaussian.probs[0] < 1e-300


def test_estimation_error_is_the_map_minus_the_truth() -> None:
    belief = make_filter([1.0, 0.0, 1.0])
    belief.update(OBS)
    assert belief.estimation_error(Eta(0.5, 0.01, -0.02)) == pytest.approx(
        [-0.1, -0.01, 0.02]
    )


@pytest.mark.parametrize(
    ("grid", "covariance", "nu", "error"),
    [
        ([], R, 3.0, ValueError),
        (GRID, R, 0.0, ValueError),
        (GRID, np.eye(5), 3.0, ValueError),
        (GRID, -R, 3.0, np.linalg.LinAlgError),
    ],
)
def test_bad_arguments_are_rejected(
    grid: list[Eta], covariance: np.ndarray, nu: float, error: type[Exception]
) -> None:
    with pytest.raises(error):
        GridBayesFilter(grid, FixedPredictions([]), covariance, nu)


def test_one_subcell_is_the_grid_itself() -> None:
    belief = make_filter([1.0, 0.0, 0.5])
    belief.update(OBS)
    product = np.exp(student_t_log_likelihood(np.array([4.0, 0.0, 1.0]), 3.0, 6))
    assert belief.probs == pytest.approx(product / product.sum())


def test_subcell_belief_sums_the_posterior_over_interpolated_points() -> None:
    grid = [Eta(mu, 0.0, 0.0) for mu in (0.2, 0.4, 0.6, 0.8, 1.0)]
    errors = [-1.5, -0.5, 0.5, 1.5, 2.5]
    belief = GridBayesFilter(grid, FixedPredictions(errors), R, 3.0, 2)
    belief.update(OBS)
    belief.update(OBS)
    # Interior points sit a quarter step from their hypothesis, so their x
    # errors move by 0.25; at the ends the prediction is held constant.
    point_errors = np.array(
        [[-1.5, -1.25], [-0.75, -0.25], [0.25, 0.75], [1.25, 1.75], [2.25, 2.5]]
    )
    q = point_errors**2 / 0.25
    likelihood = np.exp(2 * student_t_log_likelihood(q.ravel(), 3.0, 6))
    expected = likelihood.reshape(5, 2).sum(axis=1)
    assert belief.probs == pytest.approx(expected / expected.sum())


def test_too_few_subcells_are_rejected() -> None:
    with pytest.raises(ValueError, match="n_subcells"):
        GridBayesFilter(GRID, FixedPredictions([]), R, 3.0, 0)


class YawPredictions(DynamicsModel):
    """Hypotheses predicting yaws on either side of +-pi."""

    def predict(self, etas: Sequence[Eta], obs: PushObservation) -> np.ndarray:
        predicted = np.zeros((len(etas), 6))
        predicted[:, 2] = [np.pi - 0.02, -np.pi + 0.02, -np.pi + 0.06]
        return predicted


def test_subcells_interpolate_yaw_errors_across_pi() -> None:
    class AtPi:
        def measured(self) -> np.ndarray:
            measured = np.zeros(6)
            measured[2] = np.pi
            return measured

    belief = GridBayesFilter(GRID, YawPredictions(), R, None, 2)
    belief.update(cast(PushObservation, AtPi()))
    # The first two hypotheses straddle the measured yaw; interpolating raw
    # yaws would put their in-between points near 0 rad, far from it.
    assert belief.probs[2] < belief.probs[0]
    assert belief.probs[0] == pytest.approx(belief.probs[1], rel=0.05)
