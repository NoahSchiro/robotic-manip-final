"""One placement episode: optional random probing, then the one-push MPC
toward the goal, updating the estimator after every push.

With no probe pushes this is the no-probing baseline: the MPC plans with
the MAP eta from the first push, and the estimator learns from the
placement pushes themselves.  Probe pushes count against the episode's
push budget, so probing has to earn back the pushes it spends.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from pydrake.all import PiecewisePolynomial
from pydrake.geometry import Meshcat

from .action import Action
from .estimator import GridBayesFilter
from .geometry import SurfaceNormal
from .iiwa import Pusher, build_iiwa_pusher_plant, make_push_trajectory, maneuver_time
from .mpc import Candidate, GoalTolerance, MpcController, at_goal, goal_error
from .object import Object
from .observation import PushObservation
from .params import Eta
from .recording import TruthSimConfig, record_push, record_random_push
from .scene import show_object_goal, show_prediction, show_scene

StopReason = Literal["at goal", "push budget", "stuck", "unreachable"]


@dataclass(frozen=True)
class EpisodeResult:
    """How an episode ended."""

    success: bool
    stop_reason: StopReason
    probe_pushes: int
    mpc_pushes: int
    position_error: float  # m, final distance to the goal
    yaw_error: float  # rad, final absolute yaw error
    sim_time: float  # s of push trajectories, probing included
    planning_time: float  # s of wall time in MPC planning
    estimation_time: float  # s of wall time in estimator updates


@dataclass(frozen=True)
class PushRunner:
    """How an episode's pushes run in the truth sim."""

    pusher: Pusher
    truth: TruthSimConfig
    meshcat: Meshcat | None
    realtime_rate: float


@dataclass(frozen=True)
class Tally:
    """What an episode has spent so far."""

    pushes: int = 0
    probes: int = 0
    sim_time: float = 0.0  # s of push trajectories
    planning_time: float = 0.0  # s of wall time
    estimation_time: float = 0.0  # s of wall time


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


def apply_push(
    obs: PushObservation,
    label: str,
    extra: str,
    obj: Object,
    belief: GridBayesFilter | None,
    tally: Tally,
) -> Tally:
    """Update `belief` from an executed push, moving `obj` to where it left the
    object, printing the outcome, returning `tally` with the push counted."""
    number = tally.pushes + 1
    lines = [push_line(number, label, obs, extra)]
    update_seconds = 0.0
    if belief is not None:
        started = time.time()
        belief.update(obs)
        update_seconds = time.time() - started
        lines.append(belief_line(belief, update_seconds))
    sim_time = maneuver_time(obj, obs.action)
    obj.pose = obs.final_pose()
    lines.append(goal_line(*goal_error(obj.pose, obj.goal)))
    print("\n".join(lines))
    return replace(
        tally,
        pushes=number,
        sim_time=tally.sim_time + sim_time,
        estimation_time=tally.estimation_time + update_seconds,
    )


def run_probes(
    rng: np.random.Generator,
    obj: Object,
    runner: PushRunner,
    belief: GridBayesFilter | None,
    probe_sites: list[SurfaceNormal],
    count: int,
) -> Tally:
    """Run up to `count` random pushes from `probe_sites`, stopping early when
    none is reachable."""
    tally = Tally()
    for _ in range(count):
        if runner.meshcat is not None:
            show_object_goal(runner.meshcat, obj)
        try:
            obs = record_random_push(
                rng,
                obj,
                runner.pusher,
                probe_sites,
                runner.truth,
                meshcat=runner.meshcat,
                realtime_rate=runner.realtime_rate,
            )
        except RuntimeError:
            print("no reachable probe push; starting placement")
            break
        tally = apply_push(
            obs, "probe", "", obj, belief, replace(tally, probes=tally.probes + 1)
        )
    return tally


def first_reachable_improvement(
    candidates: Sequence[Candidate],
    cost_now: float,
    plan_push: Callable[[Action], PiecewisePolynomial],
) -> tuple[Candidate, PiecewisePolynomial] | None:
    """Get the first of the cheapest-first `candidates` that beats `cost_now`
    and that `plan_push` can plan, with its plan, or None without one.
    `plan_push` raises RuntimeError for a push out of the arm's reach."""
    for candidate in candidates:
        if candidate.cost >= cost_now:
            return None
        try:
            return candidate, plan_push(candidate.action)
        except RuntimeError:
            continue  # out of the arm's reach; try the next best
    return None


def no_push_reason(candidates: Sequence[Candidate], cost_now: float) -> StopReason:
    """Get why no push was chosen from the cheapest-first `candidates`."""
    return "stuck" if candidates[0].cost >= cost_now else "unreachable"


def run_placement(
    obj: Object,
    runner: PushRunner,
    controller: MpcController,
    belief: GridBayesFilter | None,
    plan_eta: Callable[[], Eta],
    max_pushes: int,
    tolerance: GoalTolerance,
    tally: Tally,
) -> tuple[StopReason, Tally]:
    """Push `obj` toward its goal with the MPC, replanning after every push,
    returning why it stopped with `tally` brought up to date."""
    ik_plant = build_iiwa_pusher_plant(runner.pusher)
    while True:
        if at_goal(obj.pose, obj.goal, tolerance):
            return "at goal", tally
        if tally.pushes >= max_pushes:
            return "push budget", tally
        print(
            f"planning push {tally.pushes + 1} over {len(controller.actions)} "
            "candidates...",
            flush=True,
        )
        started = time.time()
        candidates = controller.plan(obj, plan_eta())
        plan_time = time.time() - started
        tally = replace(tally, planning_time=tally.planning_time + plan_time)
        cost_now = controller.cost(obj)
        choice = first_reachable_improvement(
            candidates,
            cost_now,
            lambda action: make_push_trajectory(ik_plant, runner.pusher, obj, action),
        )
        if choice is None:
            return no_push_reason(candidates, cost_now), tally
        chosen, q_traj = choice
        if runner.meshcat is not None:
            show_object_goal(runner.meshcat, obj)
            show_prediction(runner.meshcat, obj, chosen.predicted)
        obs = record_push(
            obj,
            runner.pusher,
            chosen.action,
            q_traj,
            runner.truth,
            meshcat=runner.meshcat,
            realtime_rate=runner.realtime_rate,
        )
        tally = apply_push(
            obs,
            "mpc",
            f", plan {plan_time:.1f} s, cost {cost_now:.3f} -> predicted "
            f"{chosen.cost:.3f}",
            obj,
            belief,
            tally,
        )


def run_episode(
    rng: np.random.Generator,
    obj: Object,
    runner: PushRunner,
    controller: MpcController,
    belief: GridBayesFilter | None,
    plan_eta: Callable[[], Eta],
    probe_sites: list[SurfaceNormal],
    probe_pushes: int,
    max_pushes: int,
    tolerance: GoalTolerance,
) -> EpisodeResult:
    """Push `obj` from its current pose toward its goal in the truth sim,
    moving `obj.pose` along.  The first `probe_pushes` are random draws from
    `probe_sites`; the rest come from `controller`, which asks `plan_eta` for
    its eta before each plan.  `belief` (None skips estimation) is updated
    after every push."""
    if runner.meshcat is not None:
        show_scene(runner.meshcat, obj, runner.pusher, runner.truth.time_step)
        show_object_goal(runner.meshcat, obj)
    probed = run_probes(
        rng, obj, runner, belief, probe_sites, min(probe_pushes, max_pushes)
    )
    stop_reason, tally = run_placement(
        obj, runner, controller, belief, plan_eta, max_pushes, tolerance, probed
    )
    position_error, yaw_error = goal_error(obj.pose, obj.goal)
    return EpisodeResult(
        success=stop_reason == "at goal",
        stop_reason=stop_reason,
        probe_pushes=tally.probes,
        mpc_pushes=tally.pushes - tally.probes,
        position_error=position_error,
        yaw_error=yaw_error,
        sim_time=tally.sim_time,
        planning_time=tally.planning_time,
        estimation_time=tally.estimation_time,
    )
