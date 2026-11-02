"""Recording pushes from the truth sim: the arm executes a push and the result
comes back as a PushObservation."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from pydrake.all import (
    Context,
    Diagram,
    MultibodyPlant,
    PiecewisePolynomial,
    Simulator,
)
from pydrake.geometry import Meshcat

from .action import Action, sample_action
from .geometry import SurfaceNormal
from .iiwa import (
    Pusher,
    build_iiwa_pusher_plant,
    hover_tip_pose,
    make_push_trajectory,
    maneuver_time,
    push_maneuver,
)
from .object import Object, Table
from .observation import ObjectState, PushObservation
from .scene import build_diagram, place_object, state_logger

TipSample = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class TruthSimConfig:
    """How the truth sim runs and records a push: plant timestep, measurement
    delay after the push segment, and plant-state logging period (all s)."""

    time_step: float
    settle_time: float
    log_period: float


class PushWindow(NamedTuple):
    """The times bounding what gets recorded, on the trajectory's clock (s)."""

    start: float
    push_end: float
    end: float


def push_window(obj: Object, action: Action, settle_time: float) -> PushWindow:
    """Get the push window: from the tip parked at its standoff point, through
    the push segment's end, to `settle_time` after it."""
    durations = [d for _pose, d in push_maneuver(obj, action, hover_tip_pose(obj))]
    # Segments 0-1 move over and lower to the standoff; segment 2 pushes.
    start = durations[0] + durations[1]
    push_end = start + durations[2]
    return PushWindow(start, push_end, push_end + settle_time)


def read_object_state(
    plant: MultibodyPlant, plant_context: Context, obj: Object
) -> ObjectState:
    """Read the object's free-body state from `plant_context`."""
    body = plant.GetBodyByName(obj.body_box, plant.GetModelInstanceByName(obj.name))
    return ObjectState(
        pose=plant.EvalBodyPoseInWorld(plant_context, body),
        velocity=plant.EvalBodySpatialVelocityInWorld(plant_context, body),
    )


def read_tip(
    plant: MultibodyPlant, plant_context: Context, pusher: Pusher
) -> TipSample:
    """Read the stick tip's world position and translational velocity."""
    stick = plant.GetBodyByName(pusher.body, plant.GetModelInstanceByName(pusher.name))
    X_WS = plant.EvalBodyPoseInWorld(plant_context, stick)
    V_WS = plant.EvalBodySpatialVelocityInWorld(plant_context, stick)
    p_ST_W = X_WS.rotation() @ pusher.tip_point
    return X_WS @ pusher.tip_point, V_WS.Shift(p_ST_W).translational()


def tip_samples_inside_window(
    plant: MultibodyPlant,
    pusher: Pusher,
    log_times: np.ndarray,
    log_states: np.ndarray,
    window: PushWindow,
) -> tuple[np.ndarray, list[TipSample]]:
    """Rebuild tip kinematics from logged plant states strictly inside
    `window`, returning the samples' times alongside them."""
    # A margin keeps logged samples from nearly duplicating the window's
    # endpoints, which would make the replayed spline's breaks degenerate.
    margin = 1e-6
    inside = (log_times > window.start + margin) & (log_times < window.end - margin)
    scratch = plant.CreateDefaultContext()
    samples = []
    for x in log_states[:, inside].T:
        plant.SetPositionsAndVelocities(scratch, x)
        samples.append(read_tip(plant, scratch, pusher))
    return log_times[inside], samples


def observe_push(
    diagram: Diagram,
    plant: MultibodyPlant,
    context: Context,
    obj: Object,
    pusher: Pusher,
    action: Action,
    settle_time: float,
    realtime_rate: float,
) -> PushObservation:
    """Run the truth sim through `action`'s trajectory (or its push window, if
    longer), leaving `context` at the end.

    Raises RuntimeError when `diagram` has no state logger; record_push builds
    one correctly.
    """
    plant_context = plant.GetMyContextFromRoot(context)
    window = push_window(obj, action, settle_time)
    t_final = max(maneuver_time(obj, action), window.end)
    logger = state_logger(diagram)

    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(realtime_rate)
    simulator.AdvanceTo(window.start)
    start = read_object_state(plant, plant_context, obj)
    tip_start = read_tip(plant, plant_context, pusher)
    simulator.AdvanceTo(window.push_end)
    push_end = read_object_state(plant, plant_context, obj)
    simulator.AdvanceTo(window.end)
    end = read_object_state(plant, plant_context, obj)
    tip_end = read_tip(plant, plant_context, pusher)
    simulator.AdvanceTo(t_final)
    final = read_object_state(plant, plant_context, obj)

    log = logger.FindLog(context)
    inside_times, inside_tips = tip_samples_inside_window(
        plant, pusher, log.sample_times(), log.data(), window
    )
    tips = [tip_start, *inside_tips, tip_end]

    return PushObservation(
        action=action,
        start=start,
        push_end=push_end,
        end=end,
        final=final,
        push_end_time=window.push_end - window.start,
        tip_times=np.concatenate(
            [[0.0], inside_times - window.start, [window.end - window.start]]
        ),
        tip_positions=np.column_stack([p for p, _v in tips]),
        tip_velocities=np.column_stack([v for _p, v in tips]),
    )


def record_push(
    obj: Object,
    pusher: Pusher,
    action: Action,
    q_traj: PiecewisePolynomial,
    config: TruthSimConfig,
    meshcat: Meshcat | None = None,
    realtime_rate: float = 0.0,
) -> PushObservation:
    """Run `action`, planned as `q_traj` by make_push_trajectory, in the truth
    sim from `obj`'s current pose.  The table shares the object's friction, as
    in main.py."""
    diagram, plant = build_diagram(
        config.time_step,
        obj,
        Table(obj.friction),
        pusher,
        meshcat=meshcat,
        q_ref=q_traj,
        log_period=config.log_period,
    )
    context = diagram.CreateDefaultContext()
    place_object(plant, plant.GetMyMutableContextFromRoot(context), obj)
    return observe_push(
        diagram, plant, context, obj, pusher, action, config.settle_time, realtime_rate
    )


def draw_reachable_push(
    rng: np.random.Generator,
    obj: Object,
    pusher: Pusher,
    sites: list[SurfaceNormal],
    max_tries: int,
) -> tuple[Action, PiecewisePolynomial]:
    """Draw a random library push the arm can reach from `obj`'s current pose,
    returning it with its planned trajectory.

    Raises RuntimeError when none of `max_tries` draws is reachable.
    """
    ik_plant = build_iiwa_pusher_plant(pusher)
    for _ in range(max_tries):
        action = sample_action(rng, sites)
        try:
            return action, make_push_trajectory(ik_plant, pusher, obj, action)
        except RuntimeError:
            continue  # this contact is out of the arm's reach; draw again
    raise RuntimeError(f"no reachable push in {max_tries} draws")


def record_random_push(
    rng: np.random.Generator,
    obj: Object,
    pusher: Pusher,
    sites: list[SurfaceNormal],
    config: TruthSimConfig,
    meshcat: Meshcat | None = None,
    realtime_rate: float = 0.0,
    max_tries: int = 20,
) -> PushObservation:
    """Record a random reachable library push from `obj`'s current pose.

    Raises RuntimeError when none of `max_tries` draws is reachable.
    """
    action, q_traj = draw_reachable_push(rng, obj, pusher, sites, max_tries)
    return record_push(obj, pusher, action, q_traj, config, meshcat, realtime_rate)
