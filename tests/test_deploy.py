"""Deployed policy inputs: the observed position shift (no model, no physics)."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.policy import env

N_PUSHES = 4


class FakeBelief:
    probs = np.full(5, 0.2)

    def entropy(self) -> float:
        return 1.0

    def map_eta(self) -> SimpleNamespace:
        return SimpleNamespace(as_array=lambda: np.array([0.3, 0.0, 0.0]))

    def mean_eta(self) -> SimpleNamespace:
        return SimpleNamespace(as_array=lambda: np.array([0.5, 0.0, 0.0]))


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, np.ndarray]:
    seen: dict[str, np.ndarray] = {}

    def mask(pose: np.ndarray, _config: object) -> np.ndarray:
        seen["mask_pose"] = pose.copy()
        return np.ones(N_PUSHES, dtype=bool)

    def decide(_model: object, obs: np.ndarray, _mask: np.ndarray) -> int:
        seen["obs"] = obs
        return 0

    monkeypatch.setattr(env, "reachability_mask", mask)
    monkeypatch.setattr(env, "policy_decision", decide)
    return seen


def select(
    pose: np.ndarray, measurement: np.ndarray, pushes: int, shift: object
) -> None:
    env.select_push_action(
        model=None,
        belief=FakeBelief(),
        pose=pose,
        last_measurement=measurement,
        last_action=np.zeros(8),
        pushes=pushes,
        config=SimpleNamespace(n_actions=N_PUSHES + 1),
        excluded=np.zeros(N_PUSHES, dtype=bool),
        observed_shift=shift,
    )


def test_policy_sees_shifted_positions_and_the_mask_the_true_pose(
    captured: dict[str, np.ndarray],
) -> None:
    pose = np.array([0.40, 0.30, 1.0])
    measurement = np.array([0.41, 0.31, 1.1, 0.02, 0.0, 0.1])
    shift = np.array([0.15, -0.30])
    select(pose, measurement, pushes=2, shift=shift)
    obs = captured["obs"]
    np.testing.assert_allclose(captured["mask_pose"], pose)
    np.testing.assert_allclose(obs[:3], [0.55, 0.0, 1.0])
    measured = obs[3 + 5 + 2 + 3 + 3 :][:6]
    np.testing.assert_allclose(measured, [0.56, 0.01, 1.1, 0.02, 0.0, 0.1], atol=1e-6)


def test_measurement_stays_zero_before_the_first_push(
    captured: dict[str, np.ndarray],
) -> None:
    select(
        np.array([0.40, 0.30, 1.0]), np.zeros(6), pushes=0, shift=np.array([0.1, 0.1])
    )
    measured = captured["obs"][3 + 5 + 2 + 3 + 3 :][:6]
    np.testing.assert_array_equal(measured, np.zeros(6))


def test_no_shift_leaves_the_observation_unchanged(
    captured: dict[str, np.ndarray],
) -> None:
    pose = np.array([0.40, 0.30, 1.0])
    select(pose, np.zeros(6), pushes=0, shift=None)
    np.testing.assert_allclose(captured["obs"][:3], pose)
