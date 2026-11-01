"""One-push MPC: predict where every candidate push leaves the object under
the estimator's MAP eta, and rank the pushes by cost.

A push's cost is the proposal's pose cost at its predicted outcome,

    V(q) = |r - r_g|^2 / l^2 + lambda_phi wrap(phi - phi_g)^2,

plus an action cost proportional to its maneuver time (moving to the
push, pushing, backing off, and returning).  The caller executes the best
push the arm can reach, updates the estimator, and replans.

The candidates are the push library minus the sites where the stick,
lowered to its standoff point, lands on the block: at the T's inner
corners the standoff sits within a stick radius of the other part.  The
nominal tip path starts there, so the prediction would begin with the
stick inside the block.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .action import Action
from .dynamics import PlanningModel
from .geometry import SurfaceNormal, distance_to_footprint
from .iiwa import STANDOFF, Pusher
from .object import Object
from .observation import maneuver_time, wrap_angle
from .params import Eta

# Pose cost: l is half the T's 14 cm extent, so a 7 cm position error
# costs as much as a 1 rad yaw error.
LENGTH = 0.07  # l (m)
YAW_WEIGHT = 1.0  # lambda_phi (1/rad^2)
# Action cost per second of maneuver time.
TIME_WEIGHT = 0.005
# Success: within these of the goal.
POSITION_TOLERANCE = 0.01  # m
YAW_TOLERANCE = float(np.deg2rad(10.0))  # rad
# Gap the lowered stick must leave to the block at a candidate site (m).
SITE_CLEARANCE = 1e-3


def pose_cost(
    poses: np.ndarray,
    goal: Sequence[float],
    length: float = LENGTH,
    yaw_weight: float = YAW_WEIGHT,
) -> np.ndarray:
    """Get V(q) for planar poses (..., 3) of (x, y, yaw) against `goal`."""
    poses = np.asarray(poses, dtype=float)
    offset = poses[..., :2] - np.asarray(goal[:2])
    yaw_error = wrap_angle(poses[..., 2] - goal[2])
    return np.asarray((offset**2).sum(axis=-1) / length**2 + yaw_weight * yaw_error**2)


def goal_error(obj: Object) -> tuple[float, float]:
    """Get `obj`'s position error (m) and absolute yaw error (rad) to its
    goal."""
    x, y, yaw = obj.pose
    gx, gy, gyaw = obj.goal
    return float(np.hypot(x - gx, y - gy)), float(abs(wrap_angle(yaw - gyaw)))


def at_goal(obj: Object) -> bool:
    """Check whether `obj` is within the success tolerances of its goal."""
    position_error, yaw_error = goal_error(obj)
    return position_error < POSITION_TOLERANCE and yaw_error < YAW_TOLERANCE


def clear_sites(
    obj: Object,
    sites: list[SurfaceNormal],
    pusher: Pusher,
    clearance: float = SITE_CLEARANCE,
) -> list[SurfaceNormal]:
    """Keep the sites where the stick, lowered to its standoff point,
    clears the block by `clearance`."""
    return [
        site
        for site in sites
        if distance_to_footprint(
            obj.parts, *(site.position[:2] - STANDOFF * site.normal[:2])
        )
        > pusher.radius + clearance
    ]


@dataclass(frozen=True)
class Candidate:
    """A scored push: its predicted outcome and total cost."""

    action: Action
    predicted: np.ndarray  # (x, y, yaw) once the stick has risen clear
    cost: float  # pose cost of `predicted` plus the action cost


class MpcController:
    """Ranks candidate pushes by predicted cost under a given eta."""

    def __init__(
        self,
        model: PlanningModel,
        actions: Sequence[Action],
        length: float = LENGTH,
        yaw_weight: float = YAW_WEIGHT,
        time_weight: float = TIME_WEIGHT,
    ) -> None:
        if not actions:
            raise ValueError("no candidate pushes")
        self.model = model
        self.actions = list(actions)
        self.length = length
        self.yaw_weight = yaw_weight
        self.time_weight = time_weight

    def cost(self, obj: Object) -> float:
        """Get the pose cost of `obj` where it is now, pushing nothing."""
        return float(
            pose_cost(np.array(obj.pose), obj.goal, self.length, self.yaw_weight)
        )

    def plan(self, obj: Object, eta: Eta) -> list[Candidate]:
        """Score every candidate push from `obj`'s current pose toward its
        goal under `eta`, cheapest first."""
        predicted = self.model.predict_pushes(eta, obj.pose, self.actions)
        times = np.array([maneuver_time(obj, action) for action in self.actions])
        costs = (
            pose_cost(predicted, obj.goal, self.length, self.yaw_weight)
            + self.time_weight * times
        )
        return [
            Candidate(self.actions[i], predicted[i], float(costs[i]))
            for i in np.argsort(costs, kind="stable")
        ]
