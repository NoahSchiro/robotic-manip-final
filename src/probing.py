"""The Drake-world probing loop shared by evaluation, the interactive run and
the recorders.

Each push is selected from the pruned library by a Selector, planned against
the arm's current state, executed in the truth sim, and folded into the belief
when there is one.  ProbingState is the one authoritative live episode; the
loop leaves it at the true final state so a later phase can continue from it.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from pydrake.all import PiecewisePolynomial
from sb3_contrib import MaskablePPO

from .action import Action
from .env import (
    WORKSPACE_CENTER,
    WORKSPACE_RADIUS,
    ProbingEnvConfig,
    env_parts,
    reachability_mask,
    sample_episode_conditions,
    select_push_action,
    split_action,
)
from .estimator import GridBayesFilter
from .geometry import SurfaceNormal
from .object import Object
from .observation import PushObservation
from .params import Eta
from .recording import TruthSim

MAX_IK_RETRIES = 40


@dataclass
class ProbingState:
    """The live episode.  `obj.pose` is the block's planar pose (x m, y m, yaw
    rad) in the world frame; `belief` is None when no filter runs;
    `last_measurement` is the (6,) estimator measurement of the last push and
    `last_action` the (8,) vector [site position (3), site normal (3), speed
    m/s, contact time s]."""

    truth_sim: TruthSim
    obj: Object
    belief: GridBayesFilter | None
    last_measurement: np.ndarray = field(default_factory=lambda: np.zeros(6))
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(8))
    pushes: int = 0


@dataclass(frozen=True)
class PushRecord:
    """One executed push and the belief change it caused.  The entropies and
    the update time are None or 0.0 when the state has no belief."""

    observation: PushObservation
    site: int  # index into config.site_positions
    entropy_before: float | None  # nats
    entropy_after: float | None  # nats
    update_seconds: float  # wall time of belief.update


@dataclass(frozen=True)
class ProbingOutcome:
    """Why the loop stopped and the pushes it executed, in order.  Reasons:
    "budget", the selector's none_reason, "ik_exhausted", "out_of_bounds"."""

    reason: str
    records: tuple[PushRecord, ...]


Selector = Callable[[ProbingState, set[int]], tuple[int, float, float] | None]
"""Chooses the next push from the state and the IK-failed sites, returning
(site index, speed m/s, contact time s), or None to stop."""


def episode_conditions(seed: int) -> tuple[Eta, np.ndarray, np.random.Generator]:
    """The seed's true eta, start pose (x m, y m, yaw rad) and action RNG.  The
    action stream is independent of the condition stream, so the selector's
    draws never shift the episode's sampled conditions."""
    eta, start_pose = sample_episode_conditions(
        np.random.default_rng(seed), env_parts()
    )
    action_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    return eta, start_pose, action_rng


def episode_object(eta: Eta, start_pose: np.ndarray, rod_mass: float) -> Object:
    """The block for `eta` resting at `start_pose` (x m, y m, yaw rad)."""
    obj = eta.to_object(np.random.default_rng(0), rod_mass)
    obj.pose = (float(start_pose[0]), float(start_pose[1]), float(start_pose[2]))
    return obj


def out_of_bounds(pose: np.ndarray) -> bool:
    """True when the block center (pose[:2], m) leaves the workspace disk."""
    return bool(
        np.linalg.norm(pose[:2] - np.array(WORKSPACE_CENTER)) > WORKSPACE_RADIUS
    )


def _pose(state: ProbingState) -> np.ndarray:
    return np.asarray(state.obj.pose, dtype=float)


def random_selector(config: ProbingEnvConfig, rng: np.random.Generator) -> Selector:
    """Uniform over reachable library pushes whose site has not failed IK;
    None when no push is feasible."""

    def select(
        state: ProbingState, failed_sites: set[int]
    ) -> tuple[int, float, float] | None:
        mask = reachability_mask(_pose(state), config)
        per_site = len(config.speeds) * len(config.times)
        for site in failed_sites:
            mask[site * per_site : (site + 1) * per_site] = False
        feasible = np.flatnonzero(mask)
        if len(feasible) == 0:
            return None
        return split_action(int(rng.choice(feasible)), config)

    return select


def policy_selector(model: MaskablePPO, config: ProbingEnvConfig) -> Selector:
    """The trained policy's masked choice; None when it stops."""

    def select(
        state: ProbingState, failed_sites: set[int]
    ) -> tuple[int, float, float] | None:
        assert state.belief is not None, "the policy chooses from the belief"
        return select_push_action(
            model,
            state.belief,
            _pose(state),
            state.last_measurement,
            state.last_action,
            state.pushes,
            config,
            failed_sites,
        )

    return select


def run_probing(
    state: ProbingState,
    config: ProbingEnvConfig,
    selector: Selector,
    budget: int,
    none_reason: str,
    on_push: Callable[[ProbingState, PushRecord], None] | None = None,
) -> ProbingOutcome:
    """Execute pushes chosen by `selector` until a stop condition, updating
    `state` in place.  Stops before selecting once `budget` pushes have run,
    and returns `none_reason` when the selector returns None.  A push whose
    arm plan fails marks its site and re-selects; more than MAX_IK_RETRIES
    consecutive failures stop the loop."""
    records: list[PushRecord] = []
    failed_sites: set[int] = set()
    attempts = 0
    while True:
        if state.pushes >= budget:
            return ProbingOutcome("budget", tuple(records))
        picked = selector(state, failed_sites)
        if picked is None:
            return ProbingOutcome(none_reason, tuple(records))
        attempts += 1
        if attempts > MAX_IK_RETRIES:
            return ProbingOutcome("ik_exhausted", tuple(records))
        site, speed, contact_time = picked
        action = Action(
            contact=SurfaceNormal(
                position=config.site_positions[site],
                normal=config.site_normals[site],
            ),
            velocity=speed,
            time=contact_time,
        )
        entropy_before = None if state.belief is None else state.belief.entropy()
        try:
            q_traj = state.truth_sim.plan_push(state.obj, action)
        except RuntimeError:
            failed_sites.add(site)  # the arm cannot reach this site from here
            continue
        record = _execute_push(state, config, action, site, entropy_before, q_traj)
        records.append(record)
        attempts = 0
        failed_sites.clear()
        if on_push is not None:
            on_push(state, record)
        if out_of_bounds(_pose(state)):
            return ProbingOutcome("out_of_bounds", tuple(records))


def _execute_push(
    state: ProbingState,
    config: ProbingEnvConfig,
    action: Action,
    site: int,
    entropy_before: float | None,
    q_traj: PiecewisePolynomial,
) -> PushRecord:
    """Run a planned push, move the block to where it ends, and update the
    belief when there is one."""
    observation = state.truth_sim.record_push(state.obj, action, q_traj)
    state.obj.pose = observation.final_pose()
    entropy_after = None
    update_seconds = 0.0
    if state.belief is not None:
        started = time.perf_counter()
        state.belief.update(observation)
        update_seconds = time.perf_counter() - started
        entropy_after = state.belief.entropy()
    state.pushes += 1
    state.last_measurement = observation.measured()
    state.last_action = np.array(
        [
            *config.site_positions[site],
            *config.site_normals[site],
            action.velocity,
            action.time,
        ]
    )
    return PushRecord(
        observation=observation,
        site=site,
        entropy_before=entropy_before,
        entropy_after=entropy_after,
        update_seconds=update_seconds,
    )
