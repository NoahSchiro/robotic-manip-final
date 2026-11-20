"""Replay's per-trial records, with a fake prediction model (no physics)."""

from types import SimpleNamespace

import numpy as np

from src.action import Action
from src.estimator import CALIBRATED_R
from src.geometry import SurfaceNormal
from src.params import Eta
from src.replay import build_report, replay_trial
from src.trials import TrialData

GRID = [Eta(0.2, 0.0, 0.0), Eta(0.8, 0.0, 0.0)]
TRUTH = Eta(0.25, 0.0, 0.0)


class FakeModel:
    """Predicts x = mu for every hypothesis; the measurement is at x = 0.25."""

    def predict(self, etas: list[Eta], _obs: object) -> np.ndarray:
        predictions = np.zeros((len(etas), 6))
        predictions[:, 0] = [eta.mu * 1e-2 for eta in etas]
        return predictions


def observation() -> SimpleNamespace:
    action = Action(
        contact=SurfaceNormal(
            position=np.array([0.0, 0.0, 0.015]), normal=np.array([1.0, 0.0, 0.0])
        ),
        velocity=0.1,
        time=0.4,
    )
    measured = np.zeros(6)
    measured[0] = TRUTH.mu * 1e-2
    return SimpleNamespace(action=action, measured=lambda: measured)


def test_replay_scores_truth_distance_and_lists_trials() -> None:
    trial = TrialData(eta=TRUTH, observations=[observation(), observation()])
    track = replay_trial(trial, GRID, FakeModel(), CALIBRATED_R)
    assert track.true_eta_distance == [0.0, 0.0]
    report = build_report([track], GRID, {"seed": 1}, dropped_empty=0)
    (record,) = report.trials
    assert record["eta"] == TRUTH.as_array().tolist()
    pushes = record["pushes"]
    assert isinstance(pushes, list) and len(pushes) == 2
    assert pushes[0]["distance_m"] == np.float64(0.1 * 0.4)
    student_t = record["likelihoods"]["student_t"]  # type: ignore[index]
    assert student_t["map_eta"] == GRID[0].as_array().tolist()
    assert len(student_t["entropy_per_step"]) == 3
