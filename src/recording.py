"""Recording pushes from the truth sim: the arm executes a push and the result
comes back as a PushObservation.

TruthSim is one continuous simulation, built once per run.  Pushes run one
after another on it, each from the state the last left the arm and the
block in; only the arm's commanded trajectory changes in between.  The sim's
clock stops between pushes, so planning and estimating take no sim time.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from pydrake.all import (
    Context,
    MultibodyPlant,
    PiecewisePolynomial,
    Simulator,
)
from pydrake.geometry import Meshcat

from .action import Action, sample_action
from .geometry import SurfaceNormal
from .iiwa import (
    MOVE_TIME,
    Q_HOME,
    Pusher,
    build_iiwa_pusher_plant,
    chain_from_parked,
    hold_trajectory,
    hover_tip_pose,
    make_push_trajectory,
    push_maneuver,
    solve_tip_ik,
)
from .object import Object, Table
from .observation import ObjectState, PushObservation
from .scene import arm_reference, build_diagram, place_object, state_logger

TipSample = tuple[np.ndarray, np.ndarray]


class NoReachablePush(RuntimeError):
    """The arm can't reach any push it tried from the block's current pose."""


@dataclass(frozen=True)
class TruthSimConfig:
    """How the truth sim runs and records a push: plant timestep, measurement
    delay after the push segment, and plant-state logging period (all s)."""

    time_step: float
    settle_time: float
    log_period: float


class PushWindow(NamedTuple):
    """The times bounding what gets recorded, on the sim's clock (s)."""

    start: float
    push_end: float
    end: float


def push_window(
    obj: Object, action: Action, settle_time: float, trajectory_start: float
) -> PushWindow:
    """Get the push window of `action`'s trajectory, begun at `trajectory_start`:
    from the tip parked at its standoff point, through the push segment's end,
    to `settle_time` after it."""
    durations = [d for _pose, d in push_maneuver(obj, action, hover_tip_pose(obj))]
    # Segments 0-1 move over and lower to the standoff; segment 2 pushes.
    start = trajectory_start + durations[0] + durations[1]
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


def starting_posture(
    ik_plant: MultibodyPlant, pusher: Pusher, obj: Object
) -> np.ndarray:
    """Get the arm's starting posture: parked over `obj`, or at home when that's
    out of reach, which planning any push will then report."""
    try:
        return solve_tip_ik(ik_plant, pusher, hover_tip_pose(obj), Q_HOME)
    except RuntimeError:
        return Q_HOME


class TruthSim:
    """One continuous simulation of the arm, table and block, starting with
    `obj` at its current pose.  The table shares the object's friction, as in
    main.py."""

    def __init__(
        self,
        obj: Object,
        pusher: Pusher,
        config: TruthSimConfig,
        meshcat: Meshcat | None = None,
        realtime_rate: float = 0.0,
    ) -> None:
        self.pusher = pusher
        self.settle_time = config.settle_time
        self.ik_plant = build_iiwa_pusher_plant(pusher)
        self.q_parked = starting_posture(self.ik_plant, pusher, obj)
        self.diagram, self.plant = build_diagram(
            config.time_step,
            obj,
            Table(obj.friction),
            pusher,
            meshcat=meshcat,
            q_ref=hold_trajectory(self.q_parked),
            log_period=config.log_period,
        )
        self.reference = arm_reference(self.diagram)
        self.logger = state_logger(self.diagram)
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        place_object(self.plant, self.plant_context, obj)
        self.simulator = Simulator(self.diagram, self.context)
        self.simulator.set_target_realtime_rate(realtime_rate)
        self.simulator.Initialize()

    def plan_push(self, obj: Object, action: Action) -> PiecewisePolynomial:
        """Plan `action` against `obj`'s current pose, for record_push to run.

        Raises RuntimeError when the arm can't reach the push.
        """
        return make_push_trajectory(
            self.ik_plant, self.pusher, obj, action, self.q_parked
        )

    def record_push(
        self, obj: Object, action: Action, q_traj: PiecewisePolynomial
    ) -> PushObservation:
        """Run `action`, planned as `q_traj` by plan_push, leaving the arm parked
        at the trajectory's end and the sim's clock stopped there."""
        now = self.context.get_time()
        commanded = chain_from_parked(self.q_parked, q_traj, now)
        window = push_window(obj, action, self.settle_time, now + MOVE_TIME)
        self.reference.UpdateTrajectory(commanded)
        # Wall time since the last push isn't the sim falling behind: left
        # counted, the pacer would race through it instead of tracking real time.
        self.simulator.ResetStatistics()

        self.simulator.AdvanceTo(window.start)
        start = read_object_state(self.plant, self.plant_context, obj)
        tip_start = read_tip(self.plant, self.plant_context, self.pusher)
        self.simulator.AdvanceTo(window.push_end)
        push_end = read_object_state(self.plant, self.plant_context, obj)
        self.simulator.AdvanceTo(window.end)
        end = read_object_state(self.plant, self.plant_context, obj)
        tip_end = read_tip(self.plant, self.plant_context, self.pusher)
        self.simulator.AdvanceTo(max(commanded.end_time(), window.end))
        final = read_object_state(self.plant, self.plant_context, obj)
        self.q_parked = q_traj.value(q_traj.end_time()).ravel()

        log = self.logger.FindLog(self.context)
        inside_times, inside_tips = tip_samples_inside_window(
            self.plant, self.pusher, log.sample_times(), log.data(), window
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

    def record_random_push(
        self,
        rng: np.random.Generator,
        obj: Object,
        sites: list[SurfaceNormal],
        max_tries: int = 20,
    ) -> PushObservation:
        """Record a random reachable library push from `obj`'s current pose.

        Raises NoReachablePush when none of `max_tries` draws is reachable.
        """
        action, q_traj = self._draw_reachable_push(rng, obj, sites, max_tries)
        return self.record_push(obj, action, q_traj)

    def _draw_reachable_push(
        self,
        rng: np.random.Generator,
        obj: Object,
        sites: list[SurfaceNormal],
        max_tries: int,
    ) -> tuple[Action, PiecewisePolynomial]:
        """Draw a random library push the arm can reach from `obj`'s current
        pose, returning it with its planned trajectory.

        Raises NoReachablePush when none of `max_tries` draws is reachable.
        """
        for _ in range(max_tries):
            action = sample_action(rng, sites)
            try:
                return action, self.plan_push(obj, action)
            except RuntimeError:
                continue  # this contact is out of the arm's reach; draw again
        raise NoReachablePush(f"no reachable push in {max_tries} draws")
