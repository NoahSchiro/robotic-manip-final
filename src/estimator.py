"""Grid Bayes filter over eta = (mu, b).

After each push t, every hypothesis i is reweighted by the likelihood of
the measured outcome D_t:

    p_(t+1, i) = p_(t, i) P(D_t | eta_i) / sum_j p_(t, j) P(D_t | eta_j)

where e_(t,i) is the measured minus predicted measurement (x, y, yaw at
the push window's end; vx, vy, wz at the push segment's end; see
observation.py), yaw wrapped, and q = e^T R^-1 e.

The likelihood is a multivariate Student-t by default,

    P(D_t | eta_i) ~ (1 + q / nu)^(-(nu + 6) / 2),

with R as its scale matrix, rather than the proposal's Gaussian
exp(-q / 2) (nu=None).  Under a Gaussian, a single badly mispredicted push
costs the correct hypothesis hundreds of nats and a wrong one takes over
with near certainty; the t's log-penalty grows only logarithmically in q,
so no single push can eliminate a hypothesis.  Measuring velocity at the
push segment's end removed the outliers seen so far, so the t is a safety
net: in an offline replay of 19 held-out trials it still took MAPs within
one grid step of the truth from 89% to 100% and confidently wrong final
beliefs from 5% to 0%.

Probabilities are kept as logs so long push sequences don't underflow.  A
wait (no push) yields no observation, so it makes no update.
"""

from collections.abc import Sequence
from typing import cast

import numpy as np

from .dynamics import DynamicsModel
from .observation import PushObservation, planar_error
from .params import Eta

# Prediction-error covariance R of the measurement (x, y, yaw, vx, vy, wz),
# in m, rad, m/s, rad/s.  It describes the prediction model's errors, so
# rerun scripts/calibrate_R.py and paste its output here after changing the
# prediction model, the measurement, or the grid.
# Calibrated by scripts/calibrate_R.py: 48 truth pushes, seed 0,
# truth timestep 0.0001, model timestep 0.002, settle time 0.1 s.
CALIBRATED_R = np.array(
    [
        [2.278e-07, -7.229e-08, 1.755e-06, 5.373e-07, -1.048e-07, 4.540e-06],
        [-7.229e-08, 2.519e-07, -2.344e-06, -4.940e-07, 4.773e-07, -2.719e-06],
        [1.755e-06, -2.344e-06, 1.299e-04, 6.646e-06, -1.864e-06, 2.958e-04],
        [5.373e-07, -4.940e-07, 6.646e-06, 6.955e-06, 6.318e-07, -4.723e-06],
        [-1.048e-07, 4.773e-07, -1.864e-06, 6.318e-07, 2.253e-06, -1.957e-06],
        [4.540e-06, -2.719e-06, 2.958e-04, -4.723e-06, -1.957e-06, 9.148e-04],
    ]
)

# Student-t degrees of freedom: heavy tails, chosen by the offline replay.
DEFAULT_NU = 3.0


class GridBayesFilter:
    """Belief over a fixed hypothesis grid, updated push by push."""

    def __init__(
        self,
        grid: Sequence[Eta],
        model: DynamicsModel,
        R: np.ndarray = CALIBRATED_R,
        nu: float | None = DEFAULT_NU,
    ) -> None:
        """Start uniform over `grid`.  `R` defaults to the calibrated
        covariance; `nu` is the Student-t likelihood's degrees of freedom,
        and None uses the Gaussian likelihood instead."""
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
        # The latest update's per-hypothesis errors, for diagnostics.
        self.last_errors: np.ndarray | None = None

    @property
    def probs(self) -> np.ndarray:
        """Get the belief's probabilities, one per hypothesis."""
        return np.exp(self.log_probs)

    def update(self, obs: PushObservation) -> None:
        """Reweight every hypothesis by its likelihood of `obs`."""
        predicted = self.model.predict(self.grid, obs)
        errors = planar_error(obs.measured(), predicted)
        q = np.einsum("ni,ij,nj->n", errors, self.R_inv, errors)
        if self.nu is None:
            log_likelihood = -0.5 * q
        else:
            dim = errors.shape[1]
            log_likelihood = -0.5 * (self.nu + dim) * np.log1p(q / self.nu)
        log_probs = self.log_probs + log_likelihood
        # Log-sum-exp normalization: shift by the max before exponentiating.
        peak = log_probs.max()
        self.log_probs = log_probs - (peak + np.log(np.exp(log_probs - peak).sum()))
        self.last_errors = errors

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
