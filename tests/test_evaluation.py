"""Runner regressions that do not require a physics rollout."""

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from src import evaluation, probing
from src.params import Eta


class FakeBelief:
    def __init__(self, grid: list[Eta], *_args: object, **_kwargs: object) -> None:
        self.grid = grid

    def map_eta(self) -> Eta:
        return self.grid[0]

    def entropy(self) -> float:
        return 1.0

    def estimation_error(self, eta: Eta) -> np.ndarray:
        return self.map_eta().as_array() - eta.as_array()

    def update(self, _observation: object) -> None:
        pass


class FakeTruthSim:
    planned_poses: ClassVar[list[tuple[float, float, float]]] = []

    def __init__(self, *_args: object) -> None:
        pass

    def plan_push(self, obj: object, _action: object) -> object:
        self.planned_poses.append(obj.pose)
        return object()

    def record_push(self, obj: object, action: object, _trajectory: object) -> object:
        x, y, yaw = obj.pose
        return SimpleNamespace(
            action=action,
            final_pose=lambda: (x + 0.01, y, yaw),
            measured=lambda: np.zeros(6),
        )


@pytest.fixture
def drake_setup(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr(evaluation, "GridBayesFilter", FakeBelief)
    monkeypatch.setattr(evaluation, "TruthSim", FakeTruthSim)
    monkeypatch.setattr(
        evaluation,
        "episode_conditions",
        lambda *_: (Eta(0.5, 0, 0), np.array([0.55, 0, 0]), np.random.default_rng(0)),
    )
    monkeypatch.setattr(
        evaluation,
        "random_selector",
        lambda *_: lambda *_args: (0, 0.05, 0.2),
    )
    monkeypatch.setattr(probing, "out_of_bounds", lambda *_: False)
    FakeTruthSim.planned_poses = []
    return SimpleNamespace(
        grid=[Eta(0.1, 0, 0), Eta(0.9, 0, 0)],
        max_pushes=3,
        entropy_weight=0.1,
        step_cost=0.05,
        out_of_bounds_penalty=1.0,
        site_positions=np.array([[0.0, 0.0, 0.015]]),
        site_normals=np.array([[1.0, 0.0, 0.0]]),
    )


def run_drake(config: SimpleNamespace, budget: int) -> evaluation.EpisodeRecord:
    return evaluation._drake_episode(
        None,
        config,
        SimpleNamespace(),
        object(),
        seed=7,
        forced_pushes=budget,
        policy=False,
        r_matrix=np.eye(6),
    )


def test_zero_budget_does_not_plan_or_execute(drake_setup: SimpleNamespace) -> None:
    record = run_drake(drake_setup, 0)
    assert record.pushes == 0
    assert record.reason == "matched_budget"
    assert record.reward == pytest.approx(-0.5)
    assert FakeTruthSim.planned_poses == []


def test_next_push_uses_action_end_pose(drake_setup: SimpleNamespace) -> None:
    record = run_drake(drake_setup, 2)
    assert record.pushes == 2
    assert FakeTruthSim.planned_poses == [
        (0.55, 0.0, 0.0),
        (0.56, 0.0, 0.0),
    ]


def test_immediate_policy_stop_receives_terminal_reward(
    drake_setup: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        evaluation,
        "policy_selector",
        lambda *_: lambda *_args: None,
    )
    record = evaluation._drake_episode(
        object(),
        drake_setup,
        SimpleNamespace(),
        object(),
        seed=7,
        forced_pushes=None,
        policy=True,
        r_matrix=np.eye(6),
    )
    assert record.pushes == 0
    assert record.reason == "stop"
    assert record.reward == pytest.approx(-0.5)
    assert FakeTruthSim.planned_poses == []


def test_surrogate_stop_is_processed_by_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEnv:
        def __init__(self, _config: object) -> None:
            pass

        def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
            return np.zeros(1), {}

        def action_masks(self) -> np.ndarray:
            return np.array([False, True])

        def step(
            self, action: int
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
            assert action == 1
            return (
                np.zeros(1),
                -0.4,
                True,
                False,
                {
                    "pushes": 0,
                    "estimation_error": np.zeros(3),
                    "entropy": 1.0,
                    "reason": "stop",
                },
            )

    monkeypatch.setattr(evaluation, "ProbingEnv", FakeEnv)
    model = SimpleNamespace(predict=lambda *_args, **_kwargs: (1, None))
    config = SimpleNamespace(n_actions=2, grid=[Eta(0.1, 0, 0), Eta(0.9, 0, 0)])
    record = evaluation._surrogate_episode(model, config, seed=7)
    assert record.pushes == 0
    assert record.reward == -0.4
    assert record.reason == "stop"
