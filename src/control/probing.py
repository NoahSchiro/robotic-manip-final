"""The Drake-world probing loop shared by evaluation, the interactive run, the
recorders and the MPC's probe phase.

Each push is chosen by a Selector, planned against the arm's current state,
executed in the truth sim, and folded into the belief when there is one.
ProbingState is the single authoritative live episode.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from pydrake.all import PiecewisePolynomial

from src.estimation.estimator import GridBayesFilter
from src.experiment import TASKS, Task, out_of_bounds
from src.world.action import Action
from src.world.object import Object
from src.world.observation import PushObservation
from src.world.push_space import PushSpace, reachable
from src.world.recording import TruthSim

MAX_IK_RETRIES = 40
LONG_PUSH = 0.068  # m: lower bound of the long pushes
FAST_PUSH = 0.11  # m/s: lower bound of the fast pushes


@dataclass
class ProbingState:
    """The live episode. `obj.pose` is the block's planar pose (x m, y m, yaw
    rad) in the world frame; `belief` is None when no filter runs.
    `last_measurement` is the (6,) measurement of the last push; `last_action`
    is (8,) [contact position (3), contact normal (3), speed m/s, time s]."""

    truth_sim: TruthSim
    obj: Object
    belief: GridBayesFilter | None
    task: Task = TASKS["near"]
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


def lattice_selector(actions: Sequence[Action], rng: np.random.Generator) -> Selector:
    """Uniform over the reachable pushes among `actions`, skipping pushes
    whose arm plan failed at this decision; "unreachable" when none is
    left."""

    def select(state: ProbingState, failed: list[Action]) -> Action | str:
        pose = _pose(state)
        tried = {id(a) for a in failed}
        options = [a for a in actions if id(a) not in tried and reachable(pose, a)]
        if not options:
            return "unreachable"
        return options[int(rng.integers(len(options)))]

    return select


def long_fast(actions: Sequence[Action]) -> list[Action]:
    """The pushes with distance >= LONG_PUSH (m) and speed >= FAST_PUSH (m/s)."""
    return [
        a
        for a in actions
        if a.velocity * a.time >= LONG_PUSH - 1e-9 and a.velocity >= FAST_PUSH - 1e-9
    ]


def run_probing(
    state: ProbingState,
    selector: Selector,
    budget: int,
    on_push: Callable[[ProbingState, PushRecord], None] | None = None,
) -> ProbingOutcome:
    """Execute pushes chosen by `selector` until a stop condition, updating
    `state` in place. Stops before selecting once `budget` pushes have run.
    A push whose arm plan fails is re-selected with it in the failed list; more
    than MAX_IK_RETRIES consecutive failures stop the loop."""
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
        if out_of_bounds(_pose(state), state.task):
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
