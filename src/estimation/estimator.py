"""Grid Bayes filter over eta = (mu, b): one probability per hypothesis cell,
reweighted after each push by the likelihood of its residual e = measured minus
predicted (yaw wrapped), with q = e^T R^-1 e. The likelihood is Student-t with
nu degrees of freedom, or Gaussian when nu is None.

Log-probabilities avoid underflow over long push sequences. A wait (no push)
yields no observation and no update.
"""

from collections.abc import Sequence
from typing import cast

import numpy as np

from src.estimation.dynamics import DynamicsModel
from src.estimation.params import Eta, Subcells, subcells
from src.world.observation import PushObservation, planar_error

# Prediction-error covariances of the measurement (x, y, yaw, vx, vy, wz),
# in m, rad, m/s, rad/s, one per prediction model.  Rerun main.py calibrate
# and paste its output here after changing the prediction model, the
# measurement, the grid, the push space, or the arm maneuver.
#
# CALIBRATED_R covers the Drake plant replays of measured tip paths.
# Calibrated by main.py calibrate: 192 truth pushes, seed 20000,
# truth timestep 0.0001, model timestep 0.002, settle time 0.3 s,
# 2^3 sub-cells.
CALIBRATED_R = np.array(
    [
        [4.612e-07, -2.929e-08, -2.460e-06, 6.580e-07, -6.996e-08, -2.244e-06],
        [-2.929e-08, 3.184e-07, 1.337e-06, 1.109e-07, 4.611e-07, 2.295e-06],
        [-2.460e-06, 1.337e-06, 1.900e-04, -1.671e-06, 3.805e-06, 2.832e-04],
        [6.580e-07, 1.109e-07, -1.671e-06, 2.321e-06, 6.012e-07, 4.894e-06],
        [-6.996e-08, 4.611e-07, 3.805e-06, 6.012e-07, 2.022e-06, 1.491e-05],
        [-2.244e-06, 2.295e-06, 2.832e-04, 4.894e-06, 1.491e-05, 9.611e-04],
    ]
)

# Belief points per axis in each grid cell (see params.subcells); CALIBRATED_R
# is calibrated for this value.
SUBCELLS = 2

# CALIBRATED_R_SURROGATE covers the trained surrogate's predictions from
# measured tip paths. Its errors include arm tracking deviation, which the
# training paths' synthetic deviations reproduce only in part. The PPO
# environment's estimator noise is scaled to this covariance.
# Calibrated by main.py calibrate --predictor surrogate: 192 truth pushes,
# seed 20000, truth timestep 0.0001, settle time 0.3 s, 2^3 sub-cells,
# ensemble trained on 140k samples.
CALIBRATED_R_SURROGATE = np.array(
    [
        [4.245e-06, -4.095e-07, -2.228e-05, 1.442e-05, -1.958e-06, -4.233e-05],
        [-4.095e-07, 2.913e-06, 7.439e-06, -6.945e-07, 1.477e-05, -5.209e-05],
        [-2.228e-05, 7.439e-06, 1.120e-03, -1.403e-05, 1.612e-04, 3.136e-03],
        [1.442e-05, -6.945e-07, -1.403e-05, 3.773e-04, -2.504e-05, 1.779e-05],
        [-1.958e-06, 1.477e-05, 1.612e-04, -2.504e-05, 4.697e-04, -9.305e-05],
        [-4.233e-05, -5.209e-05, 3.136e-03, 1.779e-05, -9.305e-05, 1.018e-01],
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
        n_subcells: int = SUBCELLS,
    ) -> None:
        """Start uniform over `grid`.  `R` is the prediction-error covariance;
        `nu` is the Student-t likelihood's degrees of freedom, and None uses
        the Gaussian likelihood instead.  `n_subcells` is the number of belief
        points per axis inside each grid cell (1: the grid points alone).

        Raises ValueError when `grid` is empty, `nu` is not positive,
        `n_subcells` is below 1 or `R` is not 6x6, and LinAlgError when `R` is
        not positive-definite.
        """
        if not grid:
            raise ValueError("the hypothesis grid is empty")
        if nu is not None and nu <= 0.0:
            raise ValueError(f"nu must be positive, got {nu}")
        if n_subcells < 1:
            raise ValueError(f"n_subcells must be at least 1, got {n_subcells}")
        if R.shape != (6, 6):
            raise ValueError(f"R must be 6x6, got {R.shape}")
        np.linalg.cholesky(R)
        self.grid = list(grid)
        self.model = model
        self.nu = nu
        self.R_inv = np.linalg.inv(R)
        self.cells: Subcells = subcells(self.grid, n_subcells)
        n_points = len(self.cells.owner)
        self.point_log_probs = np.full(n_points, -np.log(n_points))

    @property
    def probs(self) -> np.ndarray:
        """Get the belief's probabilities, one per grid hypothesis (cell)."""
        return np.bincount(
            self.cells.owner,
            weights=np.exp(self.point_log_probs),
            minlength=len(self.grid),
        )

    @property
    def log_probs(self) -> np.ndarray:
        """Get the log of `probs`."""
        with np.errstate(divide="ignore"):
            return np.log(self.probs)

    def update(self, obs: PushObservation) -> None:
        """Reweight every hypothesis by its likelihood of `obs`."""
        self.update_from_predictions(obs.measured(), self.model.predict(self.grid, obs))

    def update_from_predictions(
        self, measured: np.ndarray, predicted: np.ndarray
    ) -> None:
        """Reweight the belief by the likelihood of `measured` given the (N, 6)
        `predicted` measurement of each grid hypothesis, with errors interpolated
        to the sub-cell points. Split from update() so replay can reuse one
        prediction pass."""
        # Interpolate wrapped errors, not predictions: blending yaws across +-pi
        # would land near 0.
        errors = self.cells.weights @ planar_error(measured, predicted)
        q = squared_mahalanobis(errors, self.R_inv)
        log_likelihood = (
            gaussian_log_likelihood(q)
            if self.nu is None
            else student_t_log_likelihood(q, self.nu, errors.shape[1])
        )
        self.point_log_probs = normalized_log_probs(
            self.point_log_probs + log_likelihood
        )

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
        """Get the belief's Shannon entropy H(p) = -sum p log p (nats)."""
        p = self.probs
        nonzero = p > 0.0
        return float(-(p[nonzero] * self.log_probs[nonzero]).sum())

    def estimation_error(self, eta_true: Eta) -> np.ndarray:
        """Get the MAP hypothesis minus `eta_true`, as (mu, bx, by)."""
        return cast(np.ndarray, self.map_eta().as_array() - eta_true.as_array())
