"""Grid Bayes filter over eta = (mu, b).

After each push t, every hypothesis i is reweighted by the likelihood of
the measured outcome D_t:

    p_(t+1, i) = p_(t, i) P(D_t | eta_i) / sum_j p_(t, j) P(D_t | eta_j)

where e_(t,i) is the measured minus predicted measurement (x, y, yaw at
the push window's end; vx, vy, wz at the push segment's end; see
observation.py), yaw wrapped, and q = e^T R^-1 e.

The likelihood is a multivariate Student-t,

    P(D_t | eta_i) ~ (1 + q / nu)^(-(nu + 6) / 2),

with R as its scale matrix, rather than the proposal's Gaussian
exp(-q / 2) (nu=None).  Under a Gaussian, a single badly mispredicted push
costs the correct hypothesis hundreds of nats and a wrong one takes over
with near certainty; the t's log-penalty grows only logarithmically in q,
so no single push can eliminate a hypothesis.  Measuring velocity at the
push segment's end removed most outliers, though rare ones remain, so the t
is a safety net: in an offline replay of 19 held-out trials it beat the
Gaussian on MAPs within one grid step of the truth (100% against 89%) and on
confidently wrong final beliefs (0% against 5%), a margin of one or two
trials.  Live runs since end within one grid step in 17 or 18 of 19, with up
to one confidently wrong.

Probabilities are kept as logs so long push sequences don't underflow.  A
wait (no push) yields no observation, so it makes no update.
"""

from collections.abc import Sequence
from typing import cast

import numpy as np

from .dynamics import DynamicsModel
from .observation import PushObservation, planar_error
from .params import Eta

# Prediction-error covariances of the measurement (x, y, yaw, vx, vy, wz),
# in m, rad, m/s, rad/s, one per prediction model.  Rerun main.py calibrate
# and paste its output here after changing the prediction model, the
# measurement, the grid, the push space, or the arm maneuver.
#
# CALIBRATED_R covers the Drake plant replays of measured tip paths.
# Calibrated by main.py calibrate: 192 truth pushes, seed 20000,
# truth timestep 0.0001, model timestep 0.002, settle time 0.3 s.
CALIBRATED_R = np.array(
    [
        [8.958e-07, -5.951e-08, -1.755e-06, 1.599e-06, -2.785e-07, 9.758e-07],
        [-5.951e-08, 9.205e-07, 3.652e-06, -1.225e-08, 1.608e-06, 3.516e-06],
        [-1.755e-06, 3.652e-06, 4.465e-04, 3.415e-07, 1.008e-05, 7.358e-04],
        [1.599e-06, -1.225e-08, 3.415e-07, 5.333e-06, -2.943e-07, 1.496e-05],
        [-2.785e-07, 1.608e-06, 1.008e-05, -2.943e-07, 5.096e-06, 1.859e-05],
        [9.758e-07, 3.516e-06, 7.358e-04, 1.496e-05, 1.859e-05, 1.912e-03],
    ]
)

# CALIBRATED_R_SURROGATE covers the trained surrogate predicting from
# measured tip paths.  Its errors carry the arm's force- and
# gravity-dependent tracking deviation from the commanded path, which the
# constructed training paths do not reproduce — the env's estimator noise
# is scaled to this world so the policy trains against deployment-honest
# belief degradation.
CALIBRATED_R_SURROGATE = np.array(
    [
        [3.892e-05, 1.022e-05, 5.268e-05, 1.437e-04, 3.888e-05, 4.768e-04],
        [1.022e-05, 4.819e-05, 7.738e-06, 1.075e-05, 1.839e-04, -7.429e-04],
        [5.268e-05, 7.738e-06, 3.673e-03, -1.501e-04, 2.191e-04, 7.709e-03],
        [1.437e-04, 1.075e-05, -1.501e-04, 1.252e-03, 1.263e-04, 9.018e-05],
        [3.888e-05, 1.839e-04, 2.191e-04, 1.263e-04, 1.329e-03, -2.172e-03],
        [4.768e-04, -7.429e-04, 7.709e-03, 9.018e-05, -2.172e-03, 1.941e-01],
    ]
)


def squared_mahalanobis(errors: np.ndarray, R_inv: np.ndarray) -> np.ndarray:
    """Get q = e^T R^-1 e for each row of the (N, d) `errors`."""
    return cast(np.ndarray, np.einsum("ni,ij,nj->n", errors, R_inv, errors))


def gaussian_log_likelihood(q: np.ndarray) -> np.ndarray:
    """Get the Gaussian log-likelihood of squared distances `q`, up to a
    constant."""
    return -0.5 * q


def student_t_log_likelihood(q: np.ndarray, nu: float, dim: int) -> np.ndarray:
    """Get the `dim`-dimensional Student-t log-likelihood of squared distances
    `q`, up to a constant."""
    return -0.5 * (nu + dim) * np.log1p(q / nu)


def normalized_log_probs(log_weights: np.ndarray) -> np.ndarray:
    """Scale unnormalized `log_weights` so their probabilities sum to 1."""
    # Shifting by the max before exponentiating keeps the sum from underflowing.
    peak = log_weights.max()
    return cast(
        np.ndarray, log_weights - (peak + np.log(np.exp(log_weights - peak).sum()))
    )


class GridBayesFilter:
    """Belief over a fixed hypothesis grid, updated push by push."""

    def __init__(
        self,
        grid: Sequence[Eta],
        model: DynamicsModel,
        R: np.ndarray,
        nu: float | None,
    ) -> None:
        """Start uniform over `grid`.  `R` is the prediction-error covariance;
        `nu` is the Student-t likelihood's degrees of freedom, and None uses
        the Gaussian likelihood instead.

        Raises ValueError when `grid` is empty, `nu` is not positive or `R` is
        not 6x6, and LinAlgError when `R` is not positive-definite.
        """
        if not grid:
            raise ValueError("the hypothesis grid is empty")
        if nu is not None and nu <= 0.0:
            raise ValueError(f"nu must be positive, got {nu}")
        if R.shape != (6, 6):
            raise ValueError(f"R must be 6x6, got {R.shape}")
        # Cholesky fails unless R is symmetric positive-definite.
        np.linalg.cholesky(R)
        self.grid = list(grid)
        self.model = model
        self.nu = nu
        self.R_inv = np.linalg.inv(R)
        self.log_probs = np.full(len(self.grid), -np.log(len(self.grid)))

    @property
    def probs(self) -> np.ndarray:
        """Get the belief's probabilities, one per hypothesis."""
        return np.exp(self.log_probs)

    def update(self, obs: PushObservation) -> None:
        """Reweight every hypothesis by its likelihood of `obs`."""
        self.update_from_predictions(obs.measured(), self.model.predict(self.grid, obs))

    def update_from_predictions(
        self, measured: np.ndarray, predicted: np.ndarray
    ) -> None:
        """Reweight every hypothesis by the likelihood of `measured` against
        each row of the (N, 6) `predicted`.  Split from update() so an offline
        replay can score several likelihoods against one prediction pass."""
        errors = planar_error(measured, predicted)
        q = squared_mahalanobis(errors, self.R_inv)
        log_likelihood = (
            gaussian_log_likelihood(q)
            if self.nu is None
            else student_t_log_likelihood(q, self.nu, errors.shape[1])
        )
        self.log_probs = normalized_log_probs(self.log_probs + log_likelihood)

    def map_eta(self) -> Eta:
        """Get the maximum-probability hypothesis (what MPC plans with)."""
        return self.grid[int(np.argmax(self.log_probs))]

    def mean_eta(self) -> Eta:
        """Get the belief's mean parameters.  The mean rod position need not
        lie inside the outline, so this is for reporting, not for building
        an object."""
        mu, bx, by = self.probs @ np.array([eta.as_array() for eta in self.grid])
        return Eta(float(mu), float(bx), float(by))

    def entropy(self) -> float:
        """Get the belief's Shannon entropy H(p) = -sum p log p (nats), the
        learned prober's reward signal."""
        p = self.probs
        nonzero = p > 0.0
        return float(-(p[nonzero] * self.log_probs[nonzero]).sum())

    def estimation_error(self, eta_true: Eta) -> np.ndarray:
        """Get the MAP hypothesis minus `eta_true`, as (mu, bx, by)."""
        return cast(np.ndarray, self.map_eta().as_array() - eta_true.as_array())
