"""Placement episode regressions with fake truth sim, belief and controller (no
physics).  The fake push moves the block by velocity * time along +x, and the
fake controller predicts exactly that, costing the block's x distance to goal."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.action import Action
from src.episode import MpcSelector, run_episode
from src.geometry import SurfaceNormal
from src.mpc import Candidate, GoalTolerance
from src.probing import ProbingState
from src.push_space import Face, PushSpace

SITE_POSITION = np.array([0.0, 0.0, 0.015])
SITE_NORMAL = np.array([1.0, 0.0, 0.0])
START = (0.55, 0.0, 0.0)
TOLERANCE = GoalTolerance(position=0.005, yaw=0.1)
SPACE = PushSpace(
    faces=(
        Face(
            start=np.array([0.0, -0.05]),
            end=np.array([0.0, 0.05]),
            normal=np.array([1.0, 0.0, 0.0]),
            height=0.015,
            start_inset=0.012,
            end_inset=0.012,
        ),
    ),
    speed_range=(0.03, 0.12),
    duration_range=(0.1, 0.5),
)


def push(velocity: float, time: float) -> Action:
    return Action(
        contact=SurfaceNormal(position=SITE_POSITION, normal=SITE_NORMAL),
        velocity=velocity,
        time=time,
    )


def displacement(action: Action) -> float:
    return action.velocity * action.time


class FakeBelief:
    def __init__(self) -> None:
        self.updates = 0
        self.probs = np.array([0.5, 0.5])

    def entropy(self) -> float:
        return 1.0

    def update(self, _observation: object) -> None:
        self.updates += 1

    def map_eta(self) -> SimpleNamespace:
        return SimpleNamespace(as_array=lambda: (0.3, 0.0, 0.0))


class FakeTruthSim:
    def __init__(self, ik_failures: dict[tuple[float, float], int] | None = None):
        self.ik_failures = dict(ik_failures or {})
        self.clock = 0.0
        self.plan_calls = 0
        self.context = SimpleNamespace(get_time=lambda: self.clock)

    def plan_push(self, _obj: object, action: Action) -> object:
        self.plan_calls += 1
        key = (action.velocity, action.time)
        if self.ik_failures.get(key, 0) > 0:
            self.ik_failures[key] -= 1
            raise RuntimeError("arm cannot reach this push")
        return object()

    def record_push(self, obj: object, action: Action, _trajectory: object) -> object:
        self.clock += action.time
        x, y, yaw = obj.pose
        final = (x + displacement(action), y, yaw)
        return SimpleNamespace(
            action=action,
            final_pose=lambda: final,
            measured=lambda: np.zeros(6),
        )


class FakeController:
    def __init__(self, actions: list[Action]) -> None:
        self.actions = list(actions)
        self.plan_calls = 0

    def cost(self, obj: object) -> float:
        return abs(obj.pose[0] - obj.goal[0])

    def plan(self, obj: object, _eta: object) -> list[Candidate]:
        self.plan_calls += 1
        x, y, yaw = obj.pose
        candidates = []
        for action in self.actions:
            predicted = np.array([x + displacement(action), y, yaw])
            candidates.append(
                Candidate(
                    action=action,
                    predicted=predicted,
                    cost=abs(predicted[0] - obj.goal[0]),
                )
            )
        return sorted(candidates, key=lambda c: c.cost)


def run_fake_episode(
    truth: FakeTruthSim,
    controller: FakeController,
    *,
    start: tuple[float, float, float] = START,
    goal: tuple[float, float, float] = (0.57, 0.0, 0.0),
    belief: FakeBelief | None = None,
    probe_pushes: int = 0,
    max_pushes: int = 5,
):
    return run_episode(
        action_rng=np.random.default_rng(0),
        obj=SimpleNamespace(pose=start, goal=goal),
        truth_sim=truth,
        meshcat=None,
        dashboard=None,
        controller=controller,
        belief=belief,
        plan_eta=lambda: None,
        push_space=SPACE,
        probe_pushes=probe_pushes,
        max_pushes=max_pushes,
        tolerance=TOLERANCE,
    )


def test_reaching_goal_stops_at_goal_and_updates_belief_per_push() -> None:
    truth = FakeTruthSim()
    belief = FakeBelief()
    result = run_fake_episode(
        truth,
        FakeController([push(0.05, 0.2)]),
        belief=belief,
    )
    assert result.stop_reason == "at goal"
    assert result.success
    assert result.probe_pushes == 0
    assert result.mpc_pushes == 2
    assert result.position_error < TOLERANCE.position
    assert belief.updates == 2


def test_no_improving_candidate_is_stuck() -> None:
    truth = FakeTruthSim()
    controller = FakeController([push(0.05, 0.2)])
    result = run_fake_episode(truth, controller, goal=(0.50, 0.0, 0.0))
    assert result.stop_reason == "stuck"
    assert not result.success
    assert result.mpc_pushes == 0
    assert controller.plan_calls == 1
    assert truth.plan_calls == 0


def test_improving_candidates_all_failing_ik_is_unreachable() -> None:
    truth = FakeTruthSim(
        ik_failures={(0.05, 0.2): 10**6, (0.1, 0.2): 10**6},
    )
    controller = FakeController([push(0.05, 0.2), push(0.1, 0.2), push(0.1, 0.4)])
    result = run_fake_episode(truth, controller)
    assert result.stop_reason == "unreachable"
    assert result.mpc_pushes == 0
    assert truth.plan_calls == 2
    assert controller.plan_calls == 1


def test_selector_plans_once_per_decision_across_ik_retries() -> None:
    truth = FakeTruthSim(ik_failures={(0.1, 0.2): 1})
    controller = FakeController([push(0.1, 0.2), push(0.05, 0.2)])
    result = run_fake_episode(truth, controller)
    assert result.stop_reason == "at goal"
    assert result.mpc_pushes == 2
    assert controller.plan_calls == 2
    assert truth.plan_calls == 3


def test_probes_and_placement_share_one_push_budget() -> None:
    truth = FakeTruthSim()
    result = run_fake_episode(
        truth,
        FakeController([push(0.05, 0.2)]),
        goal=(0.75, 0.0, 0.0),
        probe_pushes=2,
        max_pushes=3,
    )
    assert result.stop_reason == "push budget"
    assert result.probe_pushes == 2
    assert result.mpc_pushes == 1
    assert result.probe_pushes + result.mpc_pushes <= 3
    assert not result.success


def test_probes_past_the_budget_leave_no_placement_pushes() -> None:
    truth = FakeTruthSim()
    controller = FakeController([push(0.05, 0.2)])
    result = run_fake_episode(
        truth,
        controller,
        probe_pushes=5,
        max_pushes=3,
    )
    assert result.stop_reason == "push budget"
    assert result.probe_pushes == 3
    assert result.mpc_pushes == 0
    assert controller.plan_calls == 0


def test_block_at_goal_after_probes_is_at_goal_not_budget() -> None:
    truth = FakeTruthSim()
    controller = FakeController([push(0.05, 0.2)])
    result = run_fake_episode(
        truth,
        controller,
        start=(0.55, 0.0, 0.0),
        goal=(0.55, 0.0, 0.0),
        probe_pushes=0,
        max_pushes=0,
    )
    assert result.stop_reason == "at goal"
    assert result.success
    assert controller.plan_calls == 0


def test_selector_reports_the_latest_decision() -> None:
    truth = FakeTruthSim()
    controller = FakeController([push(0.05, 0.2)])
    chosen_predictions: list[float] = []

    def on_choose(_state: ProbingState, candidate: Candidate) -> None:
        chosen_predictions.append(float(candidate.predicted[0]))

    selector = MpcSelector(controller, lambda: None, TOLERANCE, on_choose=on_choose)
    state = ProbingState(
        truth_sim=truth,
        obj=SimpleNamespace(pose=START, goal=(0.57, 0.0, 0.0)),
        belief=None,
    )
    action = selector(state, [])
    assert isinstance(action, Action)
    assert selector.chosen is not None
    assert selector.cost_before == pytest.approx(0.02)
    assert selector.chosen.cost < selector.cost_before
    assert chosen_predictions == [pytest.approx(0.56)]
    assert selector.planning_seconds >= selector.plan_seconds >= 0.0
