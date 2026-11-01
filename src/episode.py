"""One placement episode: optional random probing, then the one-push MPC
toward the goal, updating the estimator after every push.

With no probe pushes this is the no-probing baseline: the MPC plans with
the MAP eta from the first push, and the estimator learns from the
placement pushes themselves.  Probe pushes count against the episode's
push budget, so probing has to earn back the pushes it spends.
"""

import time
from dataclasses import dataclass

import numpy as np
from pydrake.geometry import Meshcat

from .estimator import GridBayesFilter
from .geometry import SurfaceNormal
from .iiwa import Pusher, build_iiwa_pusher_plant, make_push_trajectory
from .mpc import MpcController, at_goal, goal_error
from .object import Object
from .observation import (
    PushObservation,
    maneuver_time,
    record_push,
    record_random_push,
)
from .params import Eta
from .scene import show_object_goal, show_prediction, show_scene

# Pushes per episode, probing included.
MAX_PUSHES = 20


@dataclass(frozen=True)
class EpisodeResult:
    """How an episode ended."""

    success: bool
    stop_reason: str  # "at goal", "push budget", "stuck", or "unreachable"
    probe_pushes: int
    mpc_pushes: int
    position_error: float  # m, final distance to the goal
    yaw_error: float  # rad, final absolute yaw error
    sim_time: float  # s of push trajectories, probing included
    planning_time: float  # s of wall time in MPC planning
    estimation_time: float  # s of wall time in estimator updates


def run_episode(
    rng: np.random.Generator,
    obj: Object,
    pusher: Pusher,
    controller: MpcController,
    belief: GridBayesFilter | None,
    probe_sites: list[SurfaceNormal],
    time_step: float,
    settle_time: float = 0.1,
    probe_pushes: int = 0,
    max_pushes: int = MAX_PUSHES,
    known_eta: Eta | None = None,
    meshcat: Meshcat | None = None,
    realtime_rate: float = 0.0,
) -> EpisodeResult:
    """Push `obj` from its current pose toward its goal in the truth sim,
    moving `obj.pose` along.  The first `probe_pushes` are random draws
    from `probe_sites`; the rest come from `controller`, planning with
    `known_eta` if given, else with `belief`'s MAP.  `belief` (None skips
    estimation) is updated after every push."""
    if belief is None and known_eta is None:
        raise ValueError("planning needs a belief or a known eta")
    ik_plant = build_iiwa_pusher_plant(pusher)
    pushes = probes = 0
    sim_time = planning_time = estimation_time = 0.0

    def finish(obs: PushObservation, label: str, extra: str) -> None:
        """Account for an executed push, update the belief, and move the
        object to where the push left it."""
        nonlocal pushes, sim_time, estimation_time
        pushes += 1
        sim_time += maneuver_time(obj, obs.action)
        line = (
            f"push {pushes} ({label}): v={obs.action.velocity:.2f} m/s for "
            f"{obs.action.time:.1f} s{extra}"
        )
        if belief is not None:
            started = time.time()
            belief.update(obs)
            estimation_time += time.time() - started
            mu, bx, by = belief.map_eta().as_array()
            line += (
                f"\n  update {time.time() - started:.1f} s, MAP mu={mu:.2f} "
                f"b=({bx * 100:+.1f}, {by * 100:+.1f}) cm, max p "
                f"{belief.probs.max():.2f}"
            )
        x, y, yaw = obs.final.planar()[:3]
        obj.pose = (float(x), float(y), float(yaw))
        position_error, yaw_error = goal_error(obj)
        print(
            f"{line}\n  goal error {position_error * 100:.1f} cm, "
            f"{np.rad2deg(yaw_error):.1f} deg"
        )

    if meshcat is not None:
        show_scene(meshcat, obj, pusher, time_step)
        show_object_goal(meshcat, obj)

    for _ in range(min(probe_pushes, max_pushes)):
        if meshcat is not None:
            show_object_goal(meshcat, obj)
        try:
            obs = record_random_push(
                rng,
                obj,
                pusher,
                probe_sites,
                time_step,
                settle_time,
                meshcat=meshcat,
                realtime_rate=realtime_rate,
            )
        except RuntimeError:
            print("no reachable probe push; starting placement")
            break
        probes += 1
        finish(obs, "probe", "")

    while True:
        if at_goal(obj):
            stop_reason = "at goal"
            break
        if pushes >= max_pushes:
            stop_reason = "push budget"
            break
        if known_eta is not None:
            eta = known_eta
        else:
            assert belief is not None
            eta = belief.map_eta()
        print(
            f"planning push {pushes + 1} over {len(controller.actions)} candidates...",
            flush=True,
        )
        started = time.time()
        candidates = controller.plan(obj, eta)
        plan_time = time.time() - started
        planning_time += plan_time
        cost_now = controller.cost(obj)
        # Best first: the first improving push the arm can reach.
        chosen = None
        for candidate in candidates:
            if candidate.cost >= cost_now:
                break
            try:
                q_traj = make_push_trajectory(ik_plant, pusher, obj, candidate.action)
            except RuntimeError:
                continue  # out of the arm's reach; try the next best
            chosen = candidate
            break
        if chosen is None:
            stop_reason = "stuck" if candidates[0].cost >= cost_now else "unreachable"
            break
        if meshcat is not None:
            show_object_goal(meshcat, obj)
            show_prediction(meshcat, obj, chosen.predicted)
        obs = record_push(
            obj,
            pusher,
            chosen.action,
            q_traj,
            time_step,
            settle_time,
            meshcat=meshcat,
            realtime_rate=realtime_rate,
        )
        finish(
            obs,
            "mpc",
            f", plan {plan_time:.1f} s, cost {cost_now:.3f} -> predicted "
            f"{chosen.cost:.3f}",
        )

    position_error, yaw_error = goal_error(obj)
    return EpisodeResult(
        success=stop_reason == "at goal",
        stop_reason=stop_reason,
        probe_pushes=probes,
        mpc_pushes=pushes - probes,
        position_error=position_error,
        yaw_error=yaw_error,
        sim_time=sim_time,
        planning_time=planning_time,
        estimation_time=estimation_time,
    )
