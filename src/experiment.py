"""The experiment's shared setup built from src.constants: simulation
configs, the push lattice, the reference block, and how a seed becomes an
episode.

Poses are planar (x, y, yaw) in the world frame (m, rad).
"""

import numpy as np

from src.constants import (
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


def sample_episode_conditions(
    rng: np.random.Generator, parts: list[tuple[float, float, float, float]]
) -> tuple[Eta, np.ndarray]:
    """The true eta and start pose from `rng`; the PPO environment's reset and
    the Drake episodes both draw this way, so paired evaluations see identical
    conditions."""
    eta = sample_eta(rng, GridSpec(), parts)
    x, y = disk_point(rng, START_RADIUS)
    return eta, np.array([x, y, rng.uniform(-np.pi, np.pi)])


def episode_conditions(seed: int) -> tuple[Eta, np.ndarray, np.random.Generator]:
    """The seed's true eta, start pose and action RNG.  The action stream is
    independent of the condition stream, so a selector's draws never shift the
    episode's conditions."""
    eta, start_pose = sample_episode_conditions(
        np.random.default_rng(seed), reference_object().parts
    )
    action_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    return eta, start_pose, action_rng


def episode_goal(seed: int) -> tuple[float, float, float]:
    """The seed's placement goal, from its own stream, so adding a goal never
    shifts the probing conditions or actions."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    x, y = disk_point(rng, GOAL_RADIUS)
    return (x, y, rng.uniform(-np.pi, np.pi))


def episode_object(
    eta: Eta, start_pose: np.ndarray, rod_mass: float = ROD_MASS
) -> Object:
    """The block for `eta` resting at `start_pose`."""
    obj = eta.to_object(np.random.default_rng(0), rod_mass)
    obj.pose = (float(start_pose[0]), float(start_pose[1]), float(start_pose[2]))
    return obj


def out_of_bounds(pose: np.ndarray) -> bool:
    """True when the block center (pose[:2]) leaves the workspace disk."""
    return bool(
        np.linalg.norm(pose[:2] - np.array(WORKSPACE_CENTER)) > WORKSPACE_RADIUS
    )
