"""Package E: one placement episode under one probing condition, in the Drake
truth sim with the Drake filter, written as a JSON record.

Conditions, all on the seed's paired true eta, start pose and goal:
- "true_eta": MPC plans with the true parameters, no filter (the ceiling);
- "none": no probing, MPC plans with the filter's MAP from the first push;
- "<prober>:<k>": k probe pushes, then MPC; the prober is "learned" (the
  trained policy choosing pushes, stop masked), "random" (uniform over the
  probing lattice) or "long_fast" (uniform over its long, fast subset).

Probe and placement pushes share one budget.  The episode stops at the
tolerance given, so run it at the tightest tolerance of interest: the
recorded goal error after every push gives success at any looser tolerance
and any smaller budget afterwards (the one-push MPC never looks at either).
"""

import subprocess
import time
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
from src.world.iiwa import Pusher
from src.world.recording import TruthSim

PROBERS = ("learned", "random", "long_fast")


@dataclass(frozen=True)
class Condition:
    """A parsed condition: prober None for "true_eta" and "none"."""

    name: str
    prober: str | None
    probes: int
    true_eta: bool


def parse_condition(name: str) -> Condition:
    """Parse "true_eta", "none", or "<prober>:<k>".

    Raises ValueError for anything else."""
    if name == "true_eta":
        return Condition(name, None, 0, True)
    if name == "none":
        return Condition(name, None, 0, False)
    prober, _, count = name.partition(":")
    if prober not in PROBERS or not count.isdigit() or int(count) < 1:
        raise ValueError(f"unknown condition {name!r}")
    return Condition(name, prober, int(count), False)


def normalized_error(estimate: Eta | None, truth: Eta, grid: list[Eta]) -> float | None:
    """The range-normalized distance from `estimate` to `truth`."""
    if estimate is None:
        return None
    return float(
        np.linalg.norm((estimate.as_array() - truth.as_array()) / grid_spans(grid))
    )


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return "unknown"


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

    started = time.perf_counter()
    with ParallelPushModel(workers, push_model_config()) as model:
        belief = (
            None
            if condition.true_eta
            else GridBayesFilter(config.grid, model, CALIBRATED_R, nu=3.0)
        )
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
            plan_eta=belief.map_eta if belief is not None else lambda: eta,
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
        "commit": git_commit(),
    }
    return record, summary
