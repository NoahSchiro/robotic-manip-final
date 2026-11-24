"""The experiment's shared setup built from src.constants: simulation
configs, the push lattice, the reference block, the placement tasks, and how
a seed becomes an episode.

Poses are planar (x, y, yaw) in the world frame (m, rad); the arm's base is
the origin.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.constants import (
    FAR_BAND_HALF_ANGLE,
    FAR_BAND_RADII,
    FAR_BOUNDS_HALF_ANGLE,
    FAR_BOUNDS_RADII,
    FAR_GOAL_DISTANCES,
    GOAL_RADIUS,
    LATTICE_DISTANCES,
    LATTICE_SPACING,
    LATTICE_SPEEDS,
    LOG_PERIOD,
    MODEL_TIME_STEP,
    ROD_MASS,
    SETTLE_TIME,
    START_RADIUS,
    TRUTH_TIME_STEP,
    WORKSPACE_CENTER,
    WORKSPACE_RADIUS,
)
from src.estimation.dynamics import PushModelConfig
from src.estimation.params import Eta, GridSpec, sample_eta
from src.world.action import Action
from src.world.object import Object, Rod
from src.world.push_space import PushSpace
from src.world.recording import TruthSimConfig


def truth_sim_config(
    time_step: float = TRUTH_TIME_STEP,
    settle_time: float = SETTLE_TIME,
    log_period: float = LOG_PERIOD,
) -> TruthSimConfig:
    return TruthSimConfig(
        time_step=time_step, settle_time=settle_time, log_period=log_period
    )


def push_model_config(
    time_step: float = MODEL_TIME_STEP, rod_mass: float = ROD_MASS
) -> PushModelConfig:
    return PushModelConfig(time_step=time_step, rod_mass=rod_mass)


def reference_object(rod_mass: float = ROD_MASS) -> Object:
    """The block with its rod at the center: its outline and height are the
    same for every eta, so it serves wherever only the shape matters."""
    return Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, rod_mass))


def lattice(space: PushSpace) -> list[Action]:
    """The push lattice on `space`."""
    return space.grid(LATTICE_SPACING, LATTICE_SPEEDS, LATTICE_DISTANCES)


def disk_point(rng: np.random.Generator, radius: float) -> tuple[float, float]:
    """A uniform point within `radius` of the workspace center."""
    rho = radius * np.sqrt(rng.uniform())
    theta = rng.uniform(-np.pi, np.pi)
    return (
        WORKSPACE_CENTER[0] + rho * np.cos(theta),
        WORKSPACE_CENTER[1] + rho * np.sin(theta),
    )


class Task(Protocol):
    """Where a placement episode's start and goal positions are drawn, and
    where the block center must stay (world frame, m)."""

    def start_xy(self, rng: np.random.Generator) -> tuple[float, float]: ...

    def goal_xy(
        self, rng: np.random.Generator, start_xy: tuple[float, float]
    ) -> tuple[float, float]: ...

    def out_of_bounds(self, xy: np.ndarray) -> bool: ...


@dataclass(frozen=True)
class DiskTask:
    """Start and goal uniform within their radii of the workspace center; the
    block must stay within the workspace radius."""

    def start_xy(self, rng: np.random.Generator) -> tuple[float, float]:
        return disk_point(rng, START_RADIUS)

    def goal_xy(
        self, rng: np.random.Generator, start_xy: tuple[float, float]
    ) -> tuple[float, float]:
        return disk_point(rng, GOAL_RADIUS)

    def out_of_bounds(self, xy: np.ndarray) -> bool:
        return bool(np.linalg.norm(xy - np.array(WORKSPACE_CENTER)) > WORKSPACE_RADIUS)


def _in_sector(xy: np.ndarray, radii: tuple[float, float], half_angle: float) -> bool:
    radius = float(np.linalg.norm(xy))
    angle = float(np.arctan2(xy[1], xy[0]))
    return radii[0] <= radius <= radii[1] and abs(angle) <= half_angle


@dataclass(frozen=True)
class BandTask:
    """Start and goal uniform over the band sector (FAR_BAND_*), the goal
    redrawn until it lies FAR_GOAL_DISTANCES from the start; the block must
    stay in the wider bounds sector (FAR_BOUNDS_*)."""

    def _band_point(self, rng: np.random.Generator) -> tuple[float, float]:
        inner, outer = FAR_BAND_RADII
        radius = np.sqrt(rng.uniform(inner**2, outer**2))  # uniform over area
        angle = rng.uniform(-FAR_BAND_HALF_ANGLE, FAR_BAND_HALF_ANGLE)
        return (float(radius * np.cos(angle)), float(radius * np.sin(angle)))

    def start_xy(self, rng: np.random.Generator) -> tuple[float, float]:
        return self._band_point(rng)

    def goal_xy(
        self, rng: np.random.Generator, start_xy: tuple[float, float]
    ) -> tuple[float, float]:
        low, high = FAR_GOAL_DISTANCES
        while True:
            goal = self._band_point(rng)
            if low <= np.hypot(goal[0] - start_xy[0], goal[1] - start_xy[1]) <= high:
                return goal

    def out_of_bounds(self, xy: np.ndarray) -> bool:
        return not _in_sector(xy, FAR_BOUNDS_RADII, FAR_BOUNDS_HALF_ANGLE)


TASKS: dict[str, Task] = {"near": DiskTask(), "far": BandTask()}
"""The placement tasks by name.  "near" is the one the PPO policy was trained
on; every function below defaults to it."""


def sample_episode_conditions(
    rng: np.random.Generator,
    parts: list[tuple[float, float, float, float]],
    task: Task = TASKS["near"],
) -> tuple[Eta, np.ndarray]:
    """The true eta and start pose from `rng`; the PPO environment's reset and
    the Drake episodes both draw this way, so paired evaluations see identical
    conditions.  Eta is drawn first, so a seed's eta is the same in every
    task."""
    eta = sample_eta(rng, GridSpec(), parts)
    x, y = task.start_xy(rng)
    return eta, np.array([x, y, rng.uniform(-np.pi, np.pi)])


def episode_conditions(
    seed: int, task: Task = TASKS["near"]
) -> tuple[Eta, np.ndarray, np.random.Generator]:
    """The seed's true eta, start pose and action RNG.  The action stream is
    independent of the condition stream, so a selector's draws never shift the
    episode's conditions."""
    eta, start_pose = sample_episode_conditions(
        np.random.default_rng(seed), reference_object().parts, task
    )
    action_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    return eta, start_pose, action_rng


def episode_goal(
    seed: int, start_pose: np.ndarray, task: Task = TASKS["near"]
) -> tuple[float, float, float]:
    """The seed's placement goal for a block starting at `start_pose`, from
    its own stream, so adding a goal never shifts the probing conditions or
    actions."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    x, y = task.goal_xy(rng, (float(start_pose[0]), float(start_pose[1])))
    return (x, y, rng.uniform(-np.pi, np.pi))


def episode_object(
    eta: Eta, start_pose: np.ndarray, rod_mass: float = ROD_MASS
) -> Object:
    """The block for `eta` resting at `start_pose`."""
    obj = eta.to_object(np.random.default_rng(0), rod_mass)
    obj.pose = (float(start_pose[0]), float(start_pose[1]), float(start_pose[2]))
    return obj


def out_of_bounds(pose: np.ndarray, task: Task = TASKS["near"]) -> bool:
    """True when the block center (pose[:2]) leaves `task`'s bounds."""
    return task.out_of_bounds(np.asarray(pose[:2], dtype=float))
