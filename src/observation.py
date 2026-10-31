"""What one push produces for the estimator: the object's state before and
after, plus the pusher tip's measured path in between, recorded from the
truth sim.

The push window opens when the tip has lowered to its standoff point
(before contact) and closes `settle_time` after the push segment ends, so
the post-release slide, which depends on friction, lands in the
measurement.
"""

from dataclasses import dataclass

import numpy as np
from pydrake.all import (
    Context,
    Diagram,
    MultibodyPlant,
    RigidTransform,
    RollPitchYaw,
    Simulator,
    SpatialVelocity,
    VectorLogSink,
)
from pydrake.geometry import Meshcat

from .action import Action, sample_action
from .geometry import SurfaceNormal
from .iiwa import (
    Pusher,
    build_iiwa_pusher_plant,
    hover_tip_pose,
    make_push_trajectory,
    push_maneuver,
)
from .object import Object, Table
from .scene import STATE_LOGGER, build_diagram, place_object

# Plant-state logging period for recorded pushes (s).
LOG_PERIOD = 1e-3


@dataclass(frozen=True)
class ObjectState:
    """The object's free-body state: world pose and world spatial velocity."""

    pose: RigidTransform
    velocity: SpatialVelocity

    def planar(self) -> np.ndarray:
        """Get the planar state (x, y, yaw, vx, vy, wz)."""
        x, y, _ = self.pose.translation()
        yaw = RollPitchYaw(self.pose.rotation()).yaw_angle()
        vx, vy, _ = self.velocity.translational()
        wz = self.velocity.rotational()[2]
        return np.array([x, y, yaw, vx, vy, wz])


@dataclass(frozen=True)
class PushObservation:
    """One executed push: the object's state at the window's start and end,
    and the tip's world-frame path over the window, times starting at 0."""

    action: Action
    start: ObjectState
    end: ObjectState
    tip_times: np.ndarray  # (n,) s, from the window's start
    tip_positions: np.ndarray  # (3, n) world frame
    tip_velocities: np.ndarray  # (3, n) world frame

    @property
    def duration(self) -> float:
        """Get the window's length (s)."""
        return float(self.tip_times[-1])


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    """Wrap angles into [-pi, pi)."""
    return np.asarray((np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def planar_error(measured: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Get measured minus predicted planar state, with the yaw wrapped."""
    e = np.asarray(measured - predicted, dtype=float)
    e[..., 2] = wrap_angle(e[..., 2])
    return e


def push_window(obj: Object, action: Action, settle_time: float) -> tuple[float, float]:
    """Get the push window (start, end) on the push trajectory's clock: from
    the tip parked at its standoff point to `settle_time` after the push
    segment ends."""
    durations = [d for _pose, d in push_maneuver(obj, action, hover_tip_pose(obj))]
    # Segments 0-1 move over and lower to the standoff; segment 2 pushes.
    start = durations[0] + durations[1]
    return start, start + durations[2] + settle_time


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
) -> tuple[np.ndarray, np.ndarray]:
    """Read the stick tip's world position and translational velocity."""
    stick = plant.GetBodyByName(pusher.body, plant.GetModelInstanceByName(pusher.name))
    X_WS = plant.EvalBodyPoseInWorld(plant_context, stick)
    V_WS = plant.EvalBodySpatialVelocityInWorld(plant_context, stick)
    p_ST_W = X_WS.rotation() @ pusher.tip_point
    return X_WS @ pusher.tip_point, V_WS.Shift(p_ST_W).translational()


def observe_push(
    diagram: Diagram,
    plant: MultibodyPlant,
    context: Context,
    obj: Object,
    pusher: Pusher,
    action: Action,
    settle_time: float = 0.1,
    realtime_rate: float = 0.0,
) -> PushObservation:
    """Run the truth sim through `action`'s push window and record it.

    `diagram` must come from build_diagram with logging on, with its
    reference trajectory planned for `action`.  The sim stops at the
    window's end, leaving `context` there.
    """
    plant_context = plant.GetMyContextFromRoot(context)
    t_start, t_end = push_window(obj, action, settle_time)

    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(realtime_rate)
    simulator.AdvanceTo(t_start)
    start = read_object_state(plant, plant_context, obj)
    tip_start = read_tip(plant, plant_context, pusher)
    simulator.AdvanceTo(t_end)
    end = read_object_state(plant, plant_context, obj)
    tip_end = read_tip(plant, plant_context, pusher)

    # Logged samples strictly inside the window, rebuilt into tip kinematics
    # on a scratch context; the exact endpoints come from the live context.
    logger = diagram.GetSubsystemByName(STATE_LOGGER)
    assert isinstance(logger, VectorLogSink)
    log = logger.FindLog(context)
    times, states = log.sample_times(), log.data()
    # A margin keeps logged samples from nearly duplicating the endpoints,
    # which would make the replayed spline's breaks degenerate.
    margin = 1e-6
    inside = (times > t_start + margin) & (times < t_end - margin)
    scratch = plant.CreateDefaultContext()
    tips = [tip_start]
    for x in states[:, inside].T:
        plant.SetPositionsAndVelocities(scratch, x)
        tips.append(read_tip(plant, scratch, pusher))
    tips.append(tip_end)

    return PushObservation(
        action=action,
        start=start,
        end=end,
        tip_times=np.concatenate([[0.0], times[inside] - t_start, [t_end - t_start]]),
        tip_positions=np.column_stack([p for p, _v in tips]),
        tip_velocities=np.column_stack([v for _p, v in tips]),
    )


def record_random_push(
    rng: np.random.Generator,
    obj: Object,
    pusher: Pusher,
    sites: list[SurfaceNormal],
    time_step: float,
    settle_time: float = 0.1,
    meshcat: Meshcat | None = None,
    realtime_rate: float = 0.0,
    max_tries: int = 20,
) -> PushObservation:
    """Draw a random library push the arm can reach, run it in the truth
    sim from `obj`'s current pose, and record it.  The table shares the
    object's friction, as in main.py."""
    ik_plant = build_iiwa_pusher_plant(pusher)
    for _ in range(max_tries):
        action = sample_action(rng, sites)
        try:
            q_traj = make_push_trajectory(ik_plant, pusher, obj, action)
        except RuntimeError:
            continue  # this contact is out of the arm's reach; draw again
        diagram, plant = build_diagram(
            time_step,
            obj,
            Table(obj.friction),
            pusher,
            meshcat=meshcat,
            q_ref=q_traj,
            log_period=LOG_PERIOD,
        )
        context = diagram.CreateDefaultContext()
        place_object(plant, plant.GetMyMutableContextFromRoot(context), obj)
        return observe_push(
            diagram, plant, context, obj, pusher, action, settle_time, realtime_rate
        )
    raise RuntimeError(f"no reachable push in {max_tries} draws")
