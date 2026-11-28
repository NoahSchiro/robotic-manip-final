"""One placement episode under one probing condition, in the Drake truth sim
with the Drake filter, written as a JSON record.

Conditions, all on the seed's paired true eta, start pose and goal:
- "true_eta": MPC plans with the true eta, no filter (the ceiling);
- "none": no probing; MPC plans with the filter's MAP from the first push;
- "nominal_eta": MPC plans with a fixed uninformed eta (the grid point nearest
  the grid's mean), no filter;
- "farthest_eta": MPC plans with the grid hypothesis farthest from the true
  eta, no filter (the worst fixed model);
- "<prober>:<k>": k probe pushes, then MPC. Prober "learned" is the trained
  policy (stop masked), "random" is uniform over the probing lattice, and
  "long_fast" is uniform over its long, fast subset.

Probe and placement pushes share one budget. The episode stops at the tolerance
given, so run at the tightest tolerance of interest: the recorded goal errors
give success at looser tolerances and smaller budgets afterwards.
"""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from src.constants import MPC_LENGTH, MPC_TIME_WEIGHT, MPC_YAW_WEIGHT, ROD_MASS
from src.control.episode import run_episode
from src.control.mpc import GoalTolerance, MpcController, PoseCostWeights
from src.control.probing import Selector, lattice_selector, long_fast
from src.estimation.dynamics import ParallelPushModel
from src.estimation.estimator import CALIBRATED_R, GridBayesFilter
from src.estimation.params import Eta, grid_spans
from src.experiment import (
    TASKS,
    episode_conditions,
    episode_goal,
    episode_object,
    push_model_config,
    truth_sim_config,
)
from src.policy.deploy import policy_selector
from src.policy.env import make_probing_env_config
from src.provenance import git_state
from src.world.iiwa import Pusher
from src.world.recording import TruthSim

PROBERS = ("learned", "random", "long_fast")


# Where the MPC's eta comes from: the filter's MAP, or a fixed eta.
ETA_SOURCES = {
    "none": "estimated",
    "true_eta": "true",
    "nominal_eta": "nominal",
    "farthest_eta": "farthest",
}


@dataclass(frozen=True)
class Condition:
    """A parsed condition: prober None unless probes run first; `eta_source`
    is "estimated", "true", "nominal" or "farthest"."""

    name: str
    prober: str | None
    probes: int
    eta_source: str


def parse_condition(name: str) -> Condition:
    """Parse "true_eta", "none", "nominal_eta", "farthest_eta", or
    "<prober>:<k>".

    Raises ValueError for anything else."""
    if name in ETA_SOURCES:
        return Condition(name, None, 0, ETA_SOURCES[name])
    prober, _, count = name.partition(":")
    if prober not in PROBERS or not count.isdigit() or int(count) < 1:
        raise ValueError(f"unknown condition {name!r}")
    return Condition(name, prober, int(count), "estimated")


def nominal_eta(grid: list[Eta]) -> Eta:
    """The grid hypothesis nearest the grid's mean (range-normalized), as a
    system without an estimator might assume. A grid point, since the mean rod
    position can fall outside the block."""
    points = np.array([g.as_array() for g in grid])
    offsets = (points - points.mean(axis=0)) / grid_spans(grid)
    return grid[int(np.argmin(np.linalg.norm(offsets, axis=1)))]


def farthest_eta(grid: list[Eta], truth: Eta) -> Eta:
    """The grid hypothesis farthest from `truth` (range-normalized)."""
    points = np.array([g.as_array() for g in grid])
    offsets = (points - truth.as_array()) / grid_spans(grid)
    return grid[int(np.argmax(np.linalg.norm(offsets, axis=1)))]


def normalized_error(estimate: Eta | None, truth: Eta, grid: list[Eta]) -> float | None:
    """The range-normalized distance from `estimate` to `truth`."""
    if estimate is None:
        return None
    return float(
        np.linalg.norm((estimate.as_array() - truth.as_array()) / grid_spans(grid))
    )


def run_placement(
    seed: int,
    condition_name: str,
    task_name: str,
    policy_path: Path,
    workers: int,
    max_pushes: int,
    tolerance: GoalTolerance,
) -> tuple[dict[str, object], str]:
    """Run one placement episode of task `task_name` (a TASKS key); return its
    record (units: m, rad, s) and a one-line summary."""
    condition = parse_condition(condition_name)
    task = TASKS[task_name]
    eta, start_pose, action_rng = episode_conditions(seed, task)
    goal = episode_goal(seed, start_pose, task)
    obj = episode_object(eta, start_pose, ROD_MASS)
    obj.goal = goal
    config = make_probing_env_config(None, family_path=None)
    probe_selector: Selector = lattice_selector(config.actions, action_rng)
    if condition.prober == "learned":
        probe_selector = policy_selector(
            MaskablePPO.load(policy_path), config, min_pushes=condition.probes
        )
    elif condition.prober == "long_fast":
        probe_selector = lattice_selector(long_fast(config.actions), action_rng)

    fixed_eta = {
        "true": eta,
        "nominal": nominal_eta(config.grid),
        "farthest": farthest_eta(config.grid, eta),
    }.get(condition.eta_source)
    started = time.perf_counter()
    with ParallelPushModel(workers, push_model_config()) as model:
        belief: GridBayesFilter | None = None
        plan_eta: Callable[[], Eta]
        if fixed_eta is None:
            belief = GridBayesFilter(config.grid, model, CALIBRATED_R, nu=3.0)
            plan_eta = belief.map_eta
        else:
            planned: Eta = fixed_eta
            plan_eta = lambda: planned
        result = run_episode(
            obj=obj,
            truth_sim=TruthSim(obj, Pusher(), truth_sim_config()),
            meshcat=None,
            dashboard=None,
            controller=MpcController(
                model,
                config.actions,
                PoseCostWeights(MPC_LENGTH, MPC_YAW_WEIGHT),
                MPC_TIME_WEIGHT,
            ),
            belief=belief,
            plan_eta=plan_eta,
            probe_selector=probe_selector,
            probe_pushes=condition.probes,
            max_pushes=max_pushes,
            tolerance=tolerance,
            task=task,
        )
    final_map = belief.map_eta() if belief is not None else None
    wall_time = time.perf_counter() - started
    summary = (
        f"{task_name} {condition.name} seed {seed}: success {result.success} "
        f"({result.stop_reason}), {result.probe_pushes} probes + "
        f"{result.mpc_pushes} MPC pushes, final error "
        f"{result.position_error * 100:.1f} cm / "
        f"{np.rad2deg(result.yaw_error):.1f} deg, {wall_time / 60:.1f} min"
    )
    record = {
        "seed": seed,
        "task": task_name,
        "condition": condition.name,
        "prober": condition.prober,
        "probes": condition.probes,
        "true_eta": eta.as_array().tolist(),
        "eta_source": condition.eta_source,
        # The fixed eta the MPC planned with; None when it used the filter.
        "plan_eta": fixed_eta.as_array().tolist() if fixed_eta is not None else None,
        "plan_eta_error": normalized_error(fixed_eta, eta, config.grid),
        "start_pose": [float(v) for v in start_pose],
        "goal": [float(v) for v in goal],
        **asdict(result),
        "probe_map_eta": (
            result.probe_map_eta.as_array().tolist()
            if result.probe_map_eta is not None
            else None
        ),
        "probe_estimation_error": normalized_error(
            result.probe_map_eta, eta, config.grid
        ),
        "final_estimation_error": normalized_error(final_map, eta, config.grid),
        "wall_time": wall_time,
        "max_pushes": max_pushes,
        "position_tolerance": tolerance.position,
        "yaw_tolerance": tolerance.yaw,
        **git_state(),
    }
    return record, summary
