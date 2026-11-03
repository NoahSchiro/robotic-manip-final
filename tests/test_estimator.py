from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest

from src.dynamics import DynamicsModel
from src.estimator import (
    GridBayesFilter,
    gaussian_log_likelihood,
    normalized_log_probs,
    squared_mahalanobis,
    student_t_log_likelihood,
)
from src.observation import PushObservation
from src.params import Eta

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
    return GridBayesFilter(GRID, FixedPredictions(x_errors), R, nu)


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
