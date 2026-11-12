"""record_trial regressions with a fake truth sim and selector (no physics)."""

from types import SimpleNamespace

import numpy as np
import pytest

from src import trials
from src.probing import episode_conditions

SITE_POSITIONS = np.array([[0.0, 0.0, 0.015], [0.0, 0.02, 0.015]])
SITE_NORMALS = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
CONFIG = SimpleNamespace(site_positions=SITE_POSITIONS, site_normals=SITE_NORMALS)


class FakeTruthSim:
    def __init__(self, *_args: object) -> None:
        pass

    def plan_push(self, _obj: object, _action: object) -> object:
        return object()

    def record_push(self, obj: object, action: object, _trajectory: object) -> object:
        x, y, yaw = obj.pose
        return SimpleNamespace(
            action=action,
            final_pose=lambda: (x + 0.01, y, yaw),
            measured=lambda: np.zeros(6),
        )


def make_setup(pushes: int) -> trials.RecordSetup:
    return trials.RecordSetup(
        rod_mass=0.2,
        pushes=pushes,
        truth=SimpleNamespace(),
        probing=CONFIG,
    )


def test_record_trial_returns_requested_pushes_and_episode_eta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trials, "TruthSim", FakeTruthSim)
    monkeypatch.setattr(
        trials, "random_selector", lambda *_: lambda *_args: (0, 0.05, 0.2)
    )
    trial = trials.record_trial(5, make_setup(pushes=3))
    eta, _, _ = episode_conditions(5)
    assert trial.eta == eta
    assert len(trial.observations) == 3


def test_record_trial_ends_early_when_no_push_is_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trials, "TruthSim", FakeTruthSim)
    monkeypatch.setattr(trials, "random_selector", lambda *_: lambda *_args: None)
    trial = trials.record_trial(5, make_setup(pushes=3))
    assert trial.observations == []
