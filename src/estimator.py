"""Grid Bayes filter over eta = (mu, b).

After each push t, every hypothesis i is reweighted by the likelihood of
the measured outcome D_t:

    p_(t+1, i) = p_(t, i) P(D_t | eta_i) / sum_j p_(t, j) P(D_t | eta_j)
    P(D_t | eta_i) ~ exp(-1/2 e_(t,i)^T R^-1 e_(t,i))

where e_(t,i) is the measured minus predicted final planar state (x, y,
yaw, vx, vy, wz) of the push window, yaw wrapped.  Probabilities are kept
as logs so long push sequences don't underflow.  A wait (no push) yields no
observation, so it makes no update.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np

from .dynamics import DynamicsModel
from .observation import PushObservation, planar_error
from .params import Eta

# Fallback prediction-error standard deviations for (x, y, yaw, vx, vy, wz),
# in m, rad, m/s, rad/s, for use before calibrate_R.py has run.
DEFAULT_R_STD = np.array([1e-3, 1e-3, 1e-2, 2e-3, 2e-3, 2e-2])


def default_R() -> np.ndarray:
    """Get the uncalibrated diagonal prediction-error covariance."""
    return np.diag(DEFAULT_R_STD**2)


def load_R(path: Path) -> np.ndarray:
    """Load a calibrated covariance, falling back to default_R() when
    `path` doesn't exist."""
    return np.load(path) if path.exists() else default_R()


class GridBayesFilter:
    """Belief over a fixed hypothesis grid, updated push by push."""

    def __init__(
        self, grid: Sequence[Eta], model: DynamicsModel, R: np.ndarray
    ) -> None:
        if not grid:
            raise ValueError("the hypothesis grid is empty")
        if R.shape != (6, 6):
            raise ValueError(f"R must be 6x6, got {R.shape}")
        # Cholesky fails unless R is symmetric positive-definite.
        np.linalg.cholesky(R)
        self.grid = list(grid)
        self.model = model
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
        errors = planar_error(obs.end.planar(), predicted)
        log_likelihood = -0.5 * np.einsum("ni,ij,nj->n", errors, self.R_inv, errors)
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
