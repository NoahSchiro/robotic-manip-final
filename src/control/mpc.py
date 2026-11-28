"""One-push MPC: predict where each candidate push leaves the object under the
estimator's MAP eta, and rank the pushes by cost.

A push's cost is the pose cost at its predicted outcome,

    V(q) = |r - r_g|^2 / l^2 + lambda_phi wrap(phi - phi_g)^2,

plus a term proportional to its maneuver time. Only pushes inside the arm's
reach band from the current pose (push_space.reachable_mask) are predicted.
A push is ranked only if, from its predicted end pose, at least
MIN_REACHABLE_SHARE of the candidates stay reachable (or no fewer than now,
when the block is already below that share); otherwise the one-push lookahead
drives the block toward the arm base, where it can no longer be pushed. The
caller executes the best push whose arm plan succeeds, then updates the
estimator and replans.

The candidates are a lattice on the push space (push_space.PushSpace.grid).
Contacts are inset from the corners, so the lowered stick clears the other
part of the T.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.estimation.dynamics import PlanningModel, Pose
from src.estimation.params import Eta
from src.world.action import Action
from src.world.iiwa import maneuver_time
from src.world.object import Object
from src.world.observation import wrap_angle
from src.world.push_space import reachable_mask

MIN_REACHABLE_SHARE = 0.8


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
        # Body-frame contact xy and inward normal xy (A, 2); push distance (A,) m.
        self._contact_xy = np.array([a.contact.position[:2] for a in self.actions])
        self._normal_xy = np.array([a.contact.normal[:2] for a in self.actions])
        self._distance = np.array([a.velocity * a.time for a in self.actions])

    def _reachable_mask(self, pose: Pose) -> np.ndarray:
        return reachable_mask(
            np.asarray(pose, dtype=float),
            self._contact_xy,
            self._normal_xy,
            self._distance,
        )

    def reachable(self, pose: Pose) -> list[Action]:
        """The candidates inside the reach band for a block at `pose`."""
        mask = self._reachable_mask(pose)
        return [a for a, ok in zip(self.actions, mask, strict=True) if ok]

    def reachable_share(self, pose: Pose) -> float:
        """The share of candidates inside the reach band at `pose`."""
        return float(self._reachable_mask(pose).mean())

    def cost(self, obj: Object) -> float:
        """Get the pose cost of `obj` where it is now, pushing nothing."""
        return float(pose_cost(np.array(obj.pose), obj.goal, self.pose_weights))

    def plan(self, obj: Object, eta: Eta) -> list[Candidate]:
        """Score the reachable candidate pushes from `obj`'s pose under `eta`,
        cheapest first, keeping those that leave enough candidates reachable
        (see the module docstring). Empty when none qualifies."""
        actions = self.reachable(obj.pose)
        if not actions:
            return []
        predicted = self.model.predict_pushes(eta, obj.pose, actions)
        required = min(MIN_REACHABLE_SHARE, self.reachable_share(obj.pose))
        keep = [
            i
            for i, pose in enumerate(predicted)
            if self.reachable_share(pose) >= required - 1e-9
        ]
        if not keep:
            return []
        return rank_pushes(
            [actions[i] for i in keep],
            predicted[keep],
            np.array([maneuver_time(obj, actions[i]) for i in keep]),
            obj.goal,
            self.pose_weights,
            self.time_weight,
        )
