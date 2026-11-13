"""Probing loop regressions with fake truth sim and belief (no physics)."""

from types import SimpleNamespace

import numpy as np
import pytest

from src import probing
from src.action import Action
from src.geometry import SurfaceNormal
from src.probing import ProbingState, episode_conditions, random_selector, run_probing
from src.push_space import Face, PushSpace

SITE_POSITIONS = np.array([[0.0, 0.0, 0.015], [0.0, 0.02, 0.015]])
SITE_NORMALS = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
START = (0.55, 0.0, 0.0)


def site_action(site: int, speed: float = 0.05, contact_time: float = 0.2) -> Action:
    return Action(
        contact=SurfaceNormal(position=SITE_POSITIONS[site], normal=SITE_NORMALS[site]),
        velocity=speed,
        time=contact_time,
    )


def site_index(action: Action) -> int:
    matches = np.flatnonzero(np.all(SITE_POSITIONS == action.contact.position, axis=1))
    return int(matches[0])


class FakeBelief:
    def entropy(self) -> float:
        return 1.0

    def update(self, _observation: object) -> None:
        pass


class FakeTruthSim:
    def __init__(
        self, step: float = 0.01, failures_left: dict[int, int] | None = None
    ) -> None:
        self.step = step
        self.failures_left = dict(failures_left or {})
        self.plan_calls = 0
        self.planned_poses: list[tuple[float, float, float]] = []
        self.planned_sites: list[int] = []

    def plan_push(self, obj: object, action: Action) -> object:
        self.plan_calls += 1
        site = site_index(action)
        if self.failures_left.get(site, 0) > 0:
            self.failures_left[site] -= 1
            raise RuntimeError("arm cannot reach this site")
        self.planned_poses.append(obj.pose)
        self.planned_sites.append(site)
        return object()

    def record_push(self, obj: object, action: Action, _trajectory: object) -> object:
        x, y, yaw = obj.pose
        return SimpleNamespace(
            action=action,
            final_pose=lambda: (x + self.step, y, yaw),
            measured=lambda: np.arange(6, dtype=float),
        )


def make_state(truth: FakeTruthSim) -> ProbingState:
    return ProbingState(
        truth_sim=truth,
        obj=SimpleNamespace(pose=START),
        belief=FakeBelief(),
    )


def fixed_selector(site: int):
    return lambda _state, _failed: site_action(site)


def test_zero_budget_executes_nothing() -> None:
    truth = FakeTruthSim()
    calls: list[int] = []

    def selector(_state: ProbingState, _failed: list[Action]) -> str:
        calls.append(1)
        return "stop"

    state = make_state(truth)
    outcome = run_probing(state, selector, budget=0)
    assert outcome.reason == "budget"
    assert outcome.records == ()
    assert calls == []
    assert truth.plan_calls == 0
    assert state.pushes == 0


def test_next_push_plans_from_action_end_pose() -> None:
    truth = FakeTruthSim()
    state = make_state(truth)
    outcome = run_probing(state, fixed_selector(0), budget=2)
    assert outcome.reason == "budget"
    assert len(outcome.records) == 2
    assert truth.planned_poses == [
        pytest.approx((0.55, 0.0, 0.0)),
        pytest.approx((0.56, 0.0, 0.0)),
    ]


def test_ik_failure_retries_then_succeeds_and_resets_failed_sites() -> None:
    truth = FakeTruthSim(failures_left={0: 1})
    seen_failed: list[list[int]] = []

    def selector(_state: ProbingState, failed: list[Action]) -> Action:
        seen_failed.append([site_index(action) for action in failed])
        return site_action(1 if any(site_index(a) == 0 for a in failed) else 0)

    outcome = run_probing(make_state(truth), selector, budget=2)
    assert outcome.reason == "budget"
    assert [site_index(record.observation.action) for record in outcome.records] == [
        1,
        0,
    ]
    assert truth.planned_sites == [1, 0]
    assert seen_failed == [[], [0], []]


def test_ik_retries_are_bounded() -> None:
    truth = FakeTruthSim(failures_left={0: 1000})
    outcome = run_probing(make_state(truth), fixed_selector(0), budget=5)
    assert outcome.reason == "ik_exhausted"
    assert outcome.records == ()
    assert truth.plan_calls == probing.MAX_IK_RETRIES


def test_out_of_bounds_stops_after_the_push() -> None:
    truth = FakeTruthSim(step=5.0)
    selections: list[int] = []

    def selector(_state: ProbingState, _failed: list[Action]) -> Action:
        selections.append(1)
        return site_action(0)

    outcome = run_probing(make_state(truth), selector, budget=5)
    assert outcome.reason == "out_of_bounds"
    assert len(outcome.records) == 1
    assert len(selections) == 1


def test_state_and_pose_reflect_final_push() -> None:
    truth = FakeTruthSim()
    state = make_state(truth)
    callbacks: list[int] = []

    def on_push(current: ProbingState, record: probing.PushRecord) -> None:
        assert current is state
        callbacks.append(current.pushes)
        assert site_index(record.observation.action) == 0

    outcome = run_probing(state, fixed_selector(0), budget=3, on_push=on_push)
    assert outcome.reason == "budget"
    assert callbacks == [1, 2, 3]
    assert state.pushes == 3
    assert state.obj.pose == pytest.approx((0.58, 0.0, 0.0))
    np.testing.assert_allclose(state.last_measurement, np.arange(6, dtype=float))
    np.testing.assert_allclose(
        state.last_action, [0.0, 0.0, 0.015, 1.0, 0.0, 0.0, 0.05, 0.2]
    )


def test_selector_stop_reason_is_returned() -> None:
    truth = FakeTruthSim()
    outcome = run_probing(
        make_state(truth),
        lambda _state, _failed: "unreachable",
        budget=5,
    )
    assert outcome.reason == "unreachable"
    assert outcome.records == ()
    assert truth.plan_calls == 0


def test_run_probing_without_belief_executes_pushes_and_reports_no_entropy() -> None:
    truth = FakeTruthSim()
    state = ProbingState(truth_sim=truth, obj=SimpleNamespace(pose=START), belief=None)
    outcome = run_probing(state, fixed_selector(0), budget=2)
    assert outcome.reason == "budget"
    assert state.pushes == 2
    assert [record.entropy_before for record in outcome.records] == [None, None]
    assert [record.entropy_after for record in outcome.records] == [None, None]
    assert [record.update_seconds for record in outcome.records] == [0.0, 0.0]
    assert state.obj.pose == pytest.approx((0.57, 0.0, 0.0))


def test_random_selector_draws_only_reachable_pushes() -> None:
    space = PushSpace(
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
    rng = np.random.default_rng(3)
    selector = random_selector(space, rng)
    state = ProbingState(
        truth_sim=FakeTruthSim(),
        obj=SimpleNamespace(pose=START),
        belief=None,
    )
    picked = selector(state, [])
    assert isinstance(picked, Action)
    far = ProbingState(
        truth_sim=FakeTruthSim(),
        obj=SimpleNamespace(pose=(2.0, 0.0, 0.0)),
        belief=None,
    )
    assert selector(far, []) == "unreachable"


def test_episode_conditions_are_deterministic_per_seed() -> None:
    eta_a, pose_a, _ = episode_conditions(11)
    eta_b, pose_b, _ = episode_conditions(11)
    assert eta_a == eta_b
    np.testing.assert_array_equal(pose_a, pose_b)
    eta_c, _, _ = episode_conditions(12)
    assert eta_c != eta_a


def test_action_stream_is_independent_of_condition_stream() -> None:
    _, _, action_rng = episode_conditions(11)
    draws = action_rng.integers(2**32, size=4)
    condition_stream = np.random.default_rng(11).integers(2**32, size=4)
    assert not np.array_equal(draws, condition_stream)
    fresh = np.random.default_rng(np.random.SeedSequence([11, 1]))
    np.testing.assert_array_equal(draws, fresh.integers(2**32, size=4))
