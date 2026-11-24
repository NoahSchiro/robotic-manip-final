"""Model-bias helpers (no physics)."""

import numpy as np

from scripts.model_bias import band_mask, best_mu, mismatch

R = np.diag([1e-6, 1e-6, 1e-4, 1e-5, 1e-5, 1e-3])


def test_mismatch_is_mahalanobis_with_wrapped_yaw() -> None:
    measured = np.zeros((2, 6))
    predicted = np.zeros((2, 6))
    predicted[0, 0] = 2e-3
    measured[1, 2], predicted[1, 2] = np.pi - 0.005, -np.pi + 0.005
    assert np.allclose(mismatch(measured, predicted, R), [2.0, 1.0])


def test_best_mu_picks_the_sweep_value_matching_every_push() -> None:
    mus = np.array([0.2, 0.5, 0.8])
    measured = np.zeros((3, 6))
    measured[:, 0] = 0.5e-2
    sweep = np.zeros((3, 3, 6))
    sweep[:, :, 0] = mus * 1e-2
    assert best_mu(measured, sweep, mus, R) == 0.5


def test_band_mask_is_half_open() -> None:
    mu = np.array([0.1, 0.35, 0.6, 0.9])
    assert band_mask(mu, (0.35, 0.6)).tolist() == [False, True, False, False]
