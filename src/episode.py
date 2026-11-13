"""One placement episode: optional random probing, then the one-push MPC
toward the goal, on one live ProbingState and one push budget.

With no probe pushes this is the no-probing baseline: the MPC plans with
the MAP eta from the first push, and the estimator learns from the
placement pushes themselves.  Probe pushes count against the episode's
push budget, so probing has to earn back the pushes it spends.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydrake.geometry import Meshcat

from .action import Action
from .belief_view import BeliefDashboard, show_belief
from .estimator import GridBayesFilter
from .mpc import Candidate, GoalTolerance, MpcController, at_goal, goal_error
from .object import Object
from .observation import PushObservation
from .params import Eta
from .probing import (
    ProbingOutcome,
    ProbingState,
    PushRecord,
    random_selector,
    run_probing,
)
from .push_space import PushSpace
from .recording import TruthSim
from .scene import show_object_goal, show_prediction

StopReason = Literal[
    "at goal",
    "push budget",
    "stuck",
    "unreachable",
    "ik_exhausted",
    "out_of_bounds",
]

_PLACEMENT_STOP_REASONS: dict[str, StopReason] = {
    "at goal": "at goal",
    "budget": "push budget",
    "stuck": "stuck",
    "unreachable": "unreachable",
    "ik_exhausted": "ik_exhausted",
    "out_of_bounds": "out_of_bounds",
}


@dataclass(frozen=True)
class EpisodeResult:
    """How an episode ended."""

    success: bool
    stop_reason: StopReason
    probe_pushes: int
    mpc_pushes: int
    position_error: float  # m, final distance to the goal
    yaw_error: float  # rad, final absolute yaw error
    sim_time: float  # s on the truth sim's clock, probing included
    planning_time: float  # s of wall time in MPC planning
    estimation_time: float  # s of wall time in estimator updates


def push_line(number: int, label: str, obs: PushObservation, extra: str) -> str:
    """Render what push `number` was, ending with `extra`."""
    return (
        f"push {number} ({label}): v={obs.action.velocity:.2f} m/s for "
        f"{obs.action.time:.1f} s{extra}"
    )


def belief_line(belief: GridBayesFilter, update_seconds: float) -> str:
    """Render what `belief` now holds."""
    mu, bx, by = belief.map_eta().as_array()
    return (
        f"  update {update_seconds:.1f} s, MAP mu={mu:.2f} "
        f"b=({bx * 100:+.1f}, {by * 100:+.1f}) cm, max p "
        f"{belief.probs.max():.2f}"
    )


def goal_line(position_error: float, yaw_error: float) -> str:
    """Render how far the object is from its goal."""
    return (
        f"  goal error {position_error * 100:.1f} cm, {np.rad2deg(yaw_error):.1f} deg"
    )


def _same_push(a: Action, b: Action) -> bool:
    """Whether `a` and `b` command the same push: contact position, speed and time."""
    return bool(
        np.array_equal(a.contact.position, b.contact.position)
        and a.velocity == b.velocity
        and a.time == b.time
    )


class MpcSelector:
    """Chooses placement pushes by MPC on the shared probing state.

    A decision is the span between two executed pushes, identified by
    `state.pushes`.  Each decision is planned once and its ranked candidates are
    kept, so re-selections after an arm-plan failure only scan them again.  The
    latest decision's planning time, pre-push cost and chosen candidate are kept
    as attributes for reporting; `on_choose` runs when a candidate is chosen,
    before its arm plan and push.
    """

    def __init__(
        self,
        controller: MpcController,
        plan_eta: Callable[[], Eta],
        tolerance: GoalTolerance,
        on_choose: Callable[[ProbingState, Candidate], None] | None = None,
    ) -> None:
        self.controller = controller
        self.plan_eta = plan_eta
        self.tolerance = tolerance
        self.on_choose = on_choose
        self.planning_seconds = 0.0  # s of wall time, summed over decisions
        self.plan_seconds = 0.0  # s of wall time, latest decision
        self.cost_before = 0.0  # pose cost before the latest decision's push
        self.chosen: Candidate | None = None  # latest decision's candidate
        self._decision_pushes: int | None = None
        self._candidates: list[Candidate] = []

    def __call__(self, state: ProbingState, failed: list[Action]) -> Action | str:
        if at_goal(state.obj.pose, state.obj.goal, self.tolerance):
            return "at goal"
        if self._decision_pushes != state.pushes:
            self._plan(state)
        improving = [c for c in self._candidates if c.cost < self.cost_before]
        if not improving:
            return "stuck"
        for candidate in improving:
            if any(_same_push(candidate.action, f) for f in failed):
                continue
            self.chosen = candidate
            if self.on_choose is not None:
                self.on_choose(state, candidate)
            return candidate.action
        return "unreachable"

    def _plan(self, state: ProbingState) -> None:
        print(
            f"planning push {state.pushes + 1} over "
            f"{len(self.controller.actions)} candidates...",
            flush=True,
        )
        started = time.perf_counter()
        self._candidates = self.controller.plan(state.obj, self.plan_eta())
        self.plan_seconds = time.perf_counter() - started
        self.planning_seconds += self.plan_seconds
        self.cost_before = self.controller.cost(state.obj)
        self.chosen = None
        self._decision_pushes = state.pushes


def placement_stop_reason(reason: str) -> StopReason:
    """Map a probing outcome's reason to the episode's stop reason."""
    return _PLACEMENT_STOP_REASONS[reason]


def report_push(
    state: ProbingState,
    record: PushRecord,
    label: str,
    extra: str,
    meshcat: Meshcat | None,
    dashboard: BeliefDashboard | None,
) -> None:
    """Print push `record` as the state now stands, and draw the belief when
    there is one."""
    lines = [push_line(state.pushes, label, record.observation, extra)]
    if state.belief is not None:
        lines.append(belief_line(state.belief, record.update_seconds))
        show_belief(state.pushes, state.belief, state.obj, meshcat, dashboard)
    lines.append(goal_line(*goal_error(state.obj.pose, state.obj.goal)))
    print("\n".join(lines))


def run_episode(
    action_rng: np.random.Generator,
    obj: Object,
    truth_sim: TruthSim,
    meshcat: Meshcat | None,
    dashboard: BeliefDashboard | None,
    controller: MpcController,
    belief: GridBayesFilter | None,
    plan_eta: Callable[[], Eta],
    push_space: PushSpace,
    probe_pushes: int,
    max_pushes: int,
    tolerance: GoalTolerance,
) -> EpisodeResult:
    """Push `obj` from its current pose toward its goal in `truth_sim`, which
    must hold the block where `obj.pose` says, moving `obj.pose` along.  The
    first `probe_pushes` are random draws from `push_space`; the rest come
    from `controller`, which asks `plan_eta` for its eta before each plan.
    Both phases run on one ProbingState and share the `max_pushes` budget.
    `belief` (None skips estimation) is updated after every push and drawn
    in `meshcat` and `dashboard` when they're open."""
    if meshcat is not None:
        show_object_goal(meshcat, obj)
    if belief is not None:
        show_belief(0, belief, obj, meshcat, dashboard)
    started = truth_sim.context.get_time()
    state = ProbingState(truth_sim=truth_sim, obj=obj, belief=belief)

    def report_probe(current: ProbingState, record: PushRecord) -> None:
        report_push(current, record, "probe", "", meshcat, dashboard)

    probe_outcome = run_probing(
        state,
        random_selector(push_space, action_rng),
        min(probe_pushes, max_pushes),
        on_push=report_probe,
    )
    if probe_outcome.reason != "budget":
        print(f"probing stopped early ({probe_outcome.reason})")

    def show_choice(current: ProbingState, candidate: Candidate) -> None:
        if meshcat is not None:
            show_prediction(meshcat, current.obj, candidate.predicted)

    selector = MpcSelector(controller, plan_eta, tolerance, on_choose=show_choice)

    def report_mpc(current: ProbingState, record: PushRecord) -> None:
        chosen = selector.chosen
        assert chosen is not None, "a placement push has a chosen candidate"
        report_push(
            current,
            record,
            "mpc",
            f", plan {selector.plan_seconds:.1f} s, cost {selector.cost_before:.3f}"
            f" -> predicted {chosen.cost:.3f}",
            meshcat,
            dashboard,
        )

    # The goal check comes before the budget check, so a probe phase that
    # uses up the budget while leaving the block at the goal still succeeds.
    if at_goal(obj.pose, obj.goal, tolerance):
        placement_outcome = ProbingOutcome("at goal", ())
    elif probe_outcome.reason == "out_of_bounds":
        placement_outcome = ProbingOutcome("out_of_bounds", ())
    else:
        placement_outcome = run_probing(state, selector, max_pushes, on_push=report_mpc)

    stop_reason = placement_stop_reason(placement_outcome.reason)
    position_error, yaw_error = goal_error(obj.pose, obj.goal)
    return EpisodeResult(
        success=stop_reason == "at goal",
        stop_reason=stop_reason,
        probe_pushes=len(probe_outcome.records),
        mpc_pushes=len(placement_outcome.records),
        position_error=position_error,
        yaw_error=yaw_error,
        sim_time=truth_sim.context.get_time() - started,
        planning_time=selector.planning_seconds,
        estimation_time=sum(
            record.update_seconds
            for record in (*probe_outcome.records, *placement_outcome.records)
        ),
    )
