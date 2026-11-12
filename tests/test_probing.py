"""Probing loop regressions with fake truth sim and belief (no physics)."""

from types import SimpleNamespace

import numpy as np
import pytest

from src import probing
from src.probing import ProbingState, episode_conditions, run_probing

SITE_POSITIONS = np.array([[0.0, 0.0, 0.015], [0.0, 0.02, 0.015]])
SITE_NORMALS = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
CONFIG = SimpleNamespace(site_positions=SITE_POSITIONS, site_normals=SITE_NORMALS)
START = (0.55, 0.0, 0.0)


def site_index(action: object) -> int:
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

    def plan_push(self, obj: object, action: object) -> object:
        self.plan_calls += 1
        site = site_index(action)
        if self.failures_left.get(site, 0) > 0:
            self.failures_left[site] -= 1
            raise RuntimeError("arm cannot reach this site")
        self.planned_poses.append(obj.pose)
        self.planned_sites.append(site)
        return object()

    def record_push(self, obj: object, action: object, _trajectory: object) -> object:
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
    return lambda _state, _failed: (site, 0.05, 0.2)


def test_zero_budget_executes_nothing() -> None:
    truth = FakeTruthSim()
    calls: list[int] = []

    def selector(_state: ProbingState, _failed: set[int]) -> None:
        calls.append(1)

    state = make_state(truth)
    outcome = run_probing(state, CONFIG, selector, budget=0, none_reason="stop")
    assert outcome.reason == "budget"
    assert outcome.records == ()
    assert calls == []
    assert truth.plan_calls == 0
    assert state.pushes == 0


def test_next_push_plans_from_action_end_pose() -> None:
    truth = FakeTruthSim()
    state = make_state(truth)
    outcome = run_probing(
        state, CONFIG, fixed_selector(0), budget=2, none_reason="stop"
    )
    assert outcome.reason == "budget"
    assert len(outcome.records) == 2
    assert truth.planned_poses == [
        pytest.approx((0.55, 0.0, 0.0)),
        pytest.approx((0.56, 0.0, 0.0)),
    ]


def test_ik_failure_retries_then_succeeds_and_resets_failed_sites() -> None:
    truth = FakeTruthSim(failures_left={0: 1})
    seen_failed: list[set[int]] = []

    def selector(_state: ProbingState, failed: set[int]) -> tuple[int, float, float]:
        seen_failed.append(set(failed))
        return (1 if 0 in failed else 0), 0.05, 0.2

    outcome = run_probing(
        make_state(truth), CONFIG, selector, budget=2, none_reason="stop"
    )
    assert outcome.reason == "budget"
    assert [record.site for record in outcome.records] == [1, 0]
    assert truth.planned_sites == [1, 0]
    assert seen_failed == [set(), {0}, set()]


def test_ik_retries_are_bounded() -> None:
    truth = FakeTruthSim(failures_left={0: 1000})
    outcome = run_probing(
        make_state(truth),
        CONFIG,
        fixed_selector(0),
        budget=5,
        none_reason="stop",
    )
    assert outcome.reason == "ik_exhausted"
    assert outcome.records == ()
    assert truth.plan_calls == probing.MAX_IK_RETRIES


def test_out_of_bounds_stops_after_the_push() -> None:
    truth = FakeTruthSim(step=5.0)
    selections: list[int] = []

    def selector(_state: ProbingState, _failed: set[int]) -> tuple[int, float, float]:
        selections.append(1)
        return 0, 0.05, 0.2

    outcome = run_probing(
        make_state(truth), CONFIG, selector, budget=5, none_reason="stop"
    )
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
        assert record.site == 0

    outcome = run_probing(
        state,
        CONFIG,
        fixed_selector(0),
        budget=3,
        none_reason="stop",
        on_push=on_push,
    )
    assert outcome.reason == "budget"
    assert callbacks == [1, 2, 3]
    assert state.pushes == 3
    assert state.obj.pose == pytest.approx((0.58, 0.0, 0.0))
    np.testing.assert_allclose(state.last_measurement, np.arange(6, dtype=float))
    np.testing.assert_allclose(
        state.last_action, [0.0, 0.0, 0.015, 1.0, 0.0, 0.0, 0.05, 0.2]
    )


def test_none_selection_returns_none_reason() -> None:
    truth = FakeTruthSim()
    outcome = run_probing(
        make_state(truth),
        CONFIG,
        lambda _state, _failed: None,
        budget=5,
        none_reason="unreachable",
    )
    assert outcome.reason == "unreachable"
    assert outcome.records == ()
    assert truth.plan_calls == 0


def test_run_probing_without_belief_executes_pushes_and_reports_no_entropy() -> None:
    truth = FakeTruthSim()
    state = ProbingState(truth_sim=truth, obj=SimpleNamespace(pose=START), belief=None)
    outcome = run_probing(
        state, CONFIG, fixed_selector(0), budget=2, none_reason="stop"
    )
    assert outcome.reason == "budget"
    assert state.pushes == 2
    assert [record.entropy_before for record in outcome.records] == [None, None]
    assert [record.entropy_after for record in outcome.records] == [None, None]
    assert [record.update_seconds for record in outcome.records] == [0.0, 0.0]
    assert state.obj.pose == pytest.approx((0.57, 0.0, 0.0))


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
