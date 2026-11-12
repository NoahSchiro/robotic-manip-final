"""One-push MPC: predict where every candidate push leaves the object under
the estimator's MAP eta, and rank the pushes by cost.

A push's cost is the proposal's pose cost at its predicted outcome,

    V(q) = |r - r_g|^2 / l^2 + lambda_phi wrap(phi - phi_g)^2,

plus an action cost proportional to its maneuver time (moving to the
push, pushing, backing off, and returning).  The caller executes the best
push the arm can reach, updates the estimator, and replans.

The candidates are the push library minus the sites iiwa.clear_sites
drops, where the stick, lowered to its standoff point, lands on the block:
at the T's inner corners the standoff sits within a stick radius of the
other part.  The nominal tip path starts there, so the prediction would
begin with the stick inside the block.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .action import Action
from .dynamics import PlanningModel, Pose
from .iiwa import maneuver_time
from .object import Object
from .observation import wrap_angle
from .params import Eta


@dataclass(frozen=True)
class PoseCostWeights:
    """The pose cost V(q)'s scales: `length` l (m) and `yaw_weight` lambda_phi
    (1/rad^2)."""

    length: float
    yaw_weight: float


@dataclass(frozen=True)
class GoalTolerance:
    """How close to the goal counts as placed."""

    position: float  # m
    yaw: float  # rad


def pose_cost(
    poses: np.ndarray, goal: Sequence[float], weights: PoseCostWeights
) -> np.ndarray:
    """Get V(q) for planar poses (..., 3) of (x, y, yaw) against `goal`."""
    poses = np.asarray(poses, dtype=float)
    offset = poses[..., :2] - np.asarray(goal[:2])
    yaw_error = wrap_angle(poses[..., 2] - goal[2])
    return np.asarray(
        (offset**2).sum(axis=-1) / weights.length**2 + weights.yaw_weight * yaw_error**2
    )


def goal_error(pose: Pose, goal: Pose) -> tuple[float, float]:
    """Get `pose`'s position error (m) and absolute yaw error (rad) to `goal`."""
    x, y, yaw = pose
    gx, gy, gyaw = goal
    return float(np.hypot(x - gx, y - gy)), float(abs(wrap_angle(yaw - gyaw)))


def at_goal(pose: Pose, goal: Pose, tolerance: GoalTolerance) -> bool:
    """Check whether `pose` is within `tolerance` of `goal`."""
    position_error, yaw_error = goal_error(pose, goal)
    return position_error < tolerance.position and yaw_error < tolerance.yaw


@dataclass(frozen=True)
class Candidate:
    """A scored push: its predicted outcome and total cost."""

    action: Action
    predicted: np.ndarray  # (x, y, yaw) once the stick has risen clear
    cost: float  # pose cost of `predicted` plus the action cost


def rank_pushes(
    actions: Sequence[Action],
    predicted: np.ndarray,
    maneuver_times: np.ndarray,
    goal: Sequence[float],
    pose_weights: PoseCostWeights,
    time_weight: float,
) -> list[Candidate]:
    """Score each of `actions` by its predicted pose cost plus its maneuver
    time's action cost, cheapest first."""
    costs = pose_cost(predicted, goal, pose_weights) + time_weight * maneuver_times
    return [
        Candidate(actions[i], predicted[i], float(costs[i]))
        for i in np.argsort(costs, kind="stable")
    ]


class MpcController:
    """Ranks candidate pushes by predicted cost under a given eta.

    Raises ValueError when `actions` is empty.
    """

    def __init__(
        self,
        model: PlanningModel,
        actions: Sequence[Action],
        pose_weights: PoseCostWeights,
        time_weight: float,
    ) -> None:
        if not actions:
            raise ValueError("no candidate pushes")
        self.model = model
        self.actions = list(actions)
        self.pose_weights = pose_weights
        self.time_weight = time_weight

    def cost(self, obj: Object) -> float:
        """Get the pose cost of `obj` where it is now, pushing nothing."""
        return float(pose_cost(np.array(obj.pose), obj.goal, self.pose_weights))

    def plan(self, obj: Object, eta: Eta) -> list[Candidate]:
        """Score every candidate push from `obj`'s current pose toward its
        goal under `eta`, cheapest first."""
        return rank_pushes(
            self.actions,
            self.model.predict_pushes(eta, obj.pose, self.actions),
            np.array([maneuver_time(obj, action) for action in self.actions]),
            obj.goal,
            self.pose_weights,
            self.time_weight,
        )
