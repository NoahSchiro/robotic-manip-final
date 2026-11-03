"""What one push produces for the estimator: the object's state before and
after, plus the pusher tip's measured path in between.  Also where the push
leaves the object, for chaining pushes.

The push window opens when the tip has lowered to its standoff point
(before contact) and closes `settle_time` after the push segment ends.

The estimator's measurement is the object's pose at the window's end and
its velocity at the push segment's end.  The late pose carries the
post-release slide, which depends on friction.  The velocity is read while
the stick still pushes steadily: by the window's end the arm is reversing,
and its PD lag makes the stick tap the block, so a velocity read then lands
mid-collision and is noise.  In an offline replay of 19 held-out trials,
this split removed every prediction outlier and gave the best estimates of
the schemes compared.
"""

from dataclasses import dataclass

import numpy as np
from pydrake.all import RigidTransform, RollPitchYaw, SpatialVelocity

from .action import Action


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


def measurement(push_end: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Combine planar states into the estimator's measurement (x, y, yaw,
    vx, vy, wz): the pose from `end`, the velocity from `push_end`."""
    return np.concatenate([end[..., :3], push_end[..., 3:]], axis=-1)


@dataclass(frozen=True)
class PushObservation:
    """One executed push: the object's state at the window's start, at the
    push segment's end, and at the window's end, plus the tip's world-frame
    path over the window, times starting at 0.  `final` is the object's
    state at the push trajectory's end, the arm parked back at its hover:
    where the push leaves it."""

    action: Action
    start: ObjectState
    push_end: ObjectState
    end: ObjectState
    final: ObjectState
    push_end_time: float  # s, from the window's start
    tip_times: np.ndarray  # (n,) s, from the window's start
    tip_positions: np.ndarray  # (3, n) world frame
    tip_velocities: np.ndarray  # (3, n) world frame

    @property
    def duration(self) -> float:
        """Get the window's length (s)."""
        return float(self.tip_times[-1])

    def measured(self) -> np.ndarray:
        """Get the estimator's measurement: pose at the window's end,
        velocity at the push segment's end."""
        return measurement(self.push_end.planar(), self.end.planar())

    def final_pose(self) -> tuple[float, float, float]:
        """Get the object's planar pose (x, y, yaw) where the push leaves it."""
        x, y, yaw = self.final.planar()[:3]
        return float(x), float(y), float(yaw)


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    """Wrap angles into [-pi, pi)."""
    return np.asarray((np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def planar_error(measured: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Get measured minus predicted planar state, with the yaw wrapped."""
    e = np.asarray(measured - predicted, dtype=float)
    e[..., 2] = wrap_angle(e[..., 2])
    return e
