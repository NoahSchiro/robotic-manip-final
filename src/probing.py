"""The Drake-world probing loop shared by evaluation, the interactive run, the
recorders and the MPC's probe phase.

Each push is chosen by a Selector, planned against the arm's current state,
executed in the truth sim, and folded into the belief when there is one.
ProbingState is the one authoritative live episode; the loop leaves it at the
true final state so a later phase can continue from it.
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
    sample_episode_conditions,
    select_push_action,
)
from .estimator import GridBayesFilter
from .geometry import SurfaceNormal
from .object import Object
from .observation import PushObservation
from .params import Eta
from .push_space import PushSpace, reachable
from .recording import TruthSim

MAX_IK_RETRIES = 40
GOAL_RADIUS = 0.1  # m, goal disk around the workspace center, as for starts


@dataclass
class ProbingState:
    """The live episode.  `obj.pose` is the block's planar pose (x m, y m, yaw
    rad) in the world frame; `belief` is None when no filter runs;
    `last_measurement` is the (6,) estimator measurement of the last push and
    `last_action` the (8,) vector [contact position (3), contact normal (3), speed
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
    entropy_before: float | None  # nats
    entropy_after: float | None  # nats
    update_seconds: float  # wall time of belief.update


@dataclass(frozen=True)
class ProbingOutcome:
    """Why the loop stopped and the pushes it executed, in order.  Reasons:
    "budget", the selector's stop reason, "ik_exhausted", "out_of_bounds"."""

    reason: str
    records: tuple[PushRecord, ...]


Selector = Callable[[ProbingState, list[Action]], Action | str]
"""Chooses the next push from the state and the actions whose arm plan failed at
this decision, returning the push, or a string giving the reason to stop."""


def episode_conditions(seed: int) -> tuple[Eta, np.ndarray, np.random.Generator]:
    """The seed's true eta, start pose (x m, y m, yaw rad) and action RNG.  The
    action stream is independent of the condition stream, so the selector's
    draws never shift the episode's sampled conditions."""
    eta, start_pose = sample_episode_conditions(
        np.random.default_rng(seed), env_parts()
    )
    action_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    return eta, start_pose, action_rng


def episode_goal(seed: int) -> tuple[float, float, float]:
    """The seed's placement goal (x m, y m, yaw rad), drawn from its own stream
    so adding a goal never shifts the probing conditions or actions."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    rho = GOAL_RADIUS * np.sqrt(rng.uniform())
    theta = rng.uniform(-np.pi, np.pi)
    return (
        WORKSPACE_CENTER[0] + rho * np.cos(theta),
        WORKSPACE_CENTER[1] + rho * np.sin(theta),
        rng.uniform(-np.pi, np.pi),
    )


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


def random_selector(
    space: PushSpace, rng: np.random.Generator, max_draws: int = 50
) -> Selector:
    """Draws pushes from `space` until one is reachable from the block's pose,
    returning "unreachable" after `max_draws` misses.  Failed pushes are not
    excluded: a re-selection is a fresh draw."""

    def select(state: ProbingState, _failed: list[Action]) -> Action | str:
        pose = _pose(state)
        for _ in range(max_draws):
            action = space.sample(rng)
            if reachable(pose, action):
                return action
        return "unreachable"

    return select


def policy_selector(model: MaskablePPO, config: ProbingEnvConfig) -> Selector:
    """The trained policy's masked choice from the pruned library, returning
    "stop" when it stops.  Library sites whose arm plan failed at this decision
    are masked."""

    def select(state: ProbingState, failed: list[Action]) -> Action | str:
        assert state.belief is not None, "the policy chooses from the belief"
        failed_sites = {
            int(site)
            for action in failed
            for site in np.flatnonzero(
                np.all(config.site_positions == action.contact.position, axis=1)
            )
        }
        chosen = select_push_action(
            model,
            state.belief,
            _pose(state),
            state.last_measurement,
            state.last_action,
            state.pushes,
            config,
            failed_sites,
        )
        if chosen is None:
            return "stop"
        site, speed, contact_time = chosen
        return Action(
            contact=SurfaceNormal(
                position=config.site_positions[site],
                normal=config.site_normals[site],
            ),
            velocity=speed,
            time=contact_time,
        )

    return select


def run_probing(
    state: ProbingState,
    selector: Selector,
    budget: int,
    on_push: Callable[[ProbingState, PushRecord], None] | None = None,
) -> ProbingOutcome:
    """Execute pushes chosen by `selector` until a stop condition, updating
    `state` in place.  Stops before selecting once `budget` pushes have run,
    and returns the selector's reason when it stops.  A push whose arm plan
    fails is re-selected with that push in the failed list; more than
    MAX_IK_RETRIES consecutive failures stop the loop."""
    records: list[PushRecord] = []
    failed: list[Action] = []
    attempts = 0
    while True:
        if state.pushes >= budget:
            return ProbingOutcome("budget", tuple(records))
        picked = selector(state, failed)
        if isinstance(picked, str):
            return ProbingOutcome(picked, tuple(records))
        attempts += 1
        if attempts > MAX_IK_RETRIES:
            return ProbingOutcome("ik_exhausted", tuple(records))
        entropy_before = None if state.belief is None else state.belief.entropy()
        try:
            q_traj = state.truth_sim.plan_push(state.obj, picked)
        except RuntimeError:
            failed.append(picked)  # the arm cannot reach this push from here
            continue
        record = _execute_push(state, picked, entropy_before, q_traj)
        records.append(record)
        attempts = 0
        failed.clear()
        if on_push is not None:
            on_push(state, record)
        if out_of_bounds(_pose(state)):
            return ProbingOutcome("out_of_bounds", tuple(records))


def _execute_push(
    state: ProbingState,
    action: Action,
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
            *action.contact.position,
            *action.contact.normal,
            action.velocity,
            action.time,
        ]
    )
    return PushRecord(
        observation=observation,
        entropy_before=entropy_before,
        entropy_after=entropy_after,
        update_seconds=update_seconds,
    )
