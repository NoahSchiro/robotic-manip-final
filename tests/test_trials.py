"""record_trial regressions with a fake truth sim and selector (no physics)."""

from types import SimpleNamespace

import numpy as np
import pytest

from src import trials
from src.action import Action
from src.geometry import SurfaceNormal
from src.probing import episode_conditions

PUSH = Action(
    contact=SurfaceNormal(
        position=np.array([0.0, 0.0, 0.015]), normal=np.array([1.0, 0.0, 0.0])
    ),
    velocity=0.05,
    time=0.2,
)


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
    return trials.make_record_setup(0.2, pushes, SimpleNamespace())


def test_record_trial_returns_requested_pushes_and_episode_eta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trials, "TruthSim", FakeTruthSim)
    monkeypatch.setattr(trials, "random_selector", lambda *_: lambda *_args: PUSH)
    trial = trials.record_trial(5, make_setup(pushes=3))
    eta, _, _ = episode_conditions(5)
    assert trial.eta == eta
    assert len(trial.observations) == 3


def test_record_trial_ends_early_when_no_push_is_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trials, "TruthSim", FakeTruthSim)
    monkeypatch.setattr(
        trials, "random_selector", lambda *_: lambda *_args: "unreachable"
    )
    trial = trials.record_trial(5, make_setup(pushes=3))
    assert trial.observations == []
