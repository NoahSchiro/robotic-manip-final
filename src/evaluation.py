"""Evaluate learned or random probing in the continuous Drake truth sim.

The paired comparison runs the learned policy and random baseline through
the same arm pipeline. The random-only mode needs no learned assets.

Each episode samples the true eta and start pose
(episode_conditions); the learned policy runs first, then the
random-probing baseline reuses the learned policy's push count on the
identical episode, so the comparison isolates push selection.  The same
episodes also run in the surrogate world as the reference, which
quantifies the sim-to-real gap of the surrogate pipeline directly.

Every push runs the arm and a full-grid prediction when Drake is selected.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from sb3_contrib import MaskablePPO

from .dynamics import DynamicsModel, ParallelPushModel, PushModelConfig
from .env import (
    ProbingEnv,
    ProbingEnvConfig,
    make_probing_env_config,
)
from .estimator import CALIBRATED_R, CALIBRATED_R_SURROGATE, GridBayesFilter
from .iiwa import Pusher
from .params import Eta, grid_spans
from .probing import (
    ProbingState,
    PushRecord,
    episode_conditions,
    episode_object,
    policy_selector,
    random_selector,
    run_probing,
)
from .recording import TruthSim, TruthSimConfig
from .surrogate import SurrogatePredictor, load_ensemble

ROD_MASS = 0.2  # kg, as everywhere


@dataclass(frozen=True)
class EpisodeRecord:
    """One episode's outcome in one world."""

    seed: int
    policy: str  # "learned" or "random"
    pushes: int
    reward: float
    error: float  # range-normalized Euclidean MAP error
    friction_error: float  # signed mu error
    rod_x_error: float  # signed body-frame meters
    rod_y_error: float  # signed body-frame meters
    entropy: float
    reason: str


@dataclass(frozen=True)
class WorldSummary:
    """Aggregated episodes for one world and policy."""

    episodes: int
    mean_reward: float
    mean_error: float
    mean_abs_friction_error: float
    mean_abs_rod_x_error: float
    mean_abs_rod_y_error: float
    mean_pushes: float


def _summary(records: list[EpisodeRecord]) -> WorldSummary:
    return WorldSummary(
        episodes=len(records),
        mean_reward=float(np.mean([r.reward for r in records])),
        mean_error=float(np.mean([r.error for r in records])),
        mean_abs_friction_error=float(
            np.mean([abs(r.friction_error) for r in records])
        ),
        mean_abs_rod_x_error=float(np.mean([abs(r.rod_x_error) for r in records])),
        mean_abs_rod_y_error=float(np.mean([abs(r.rod_y_error) for r in records])),
        mean_pushes=float(np.mean([r.pushes for r in records])),
    )


def stop_reward(belief: GridBayesFilter, true_eta: Eta, spans: np.ndarray) -> float:
    error = (belief.map_eta().as_array() - true_eta.as_array()) / spans
    return -float(np.linalg.norm(error))


def _entropy_drop(record: PushRecord) -> float:
    """Nats the push removed from the belief; evaluation episodes always run
    with a belief, so the entropies are present."""
    assert record.entropy_before is not None and record.entropy_after is not None
    return record.entropy_before - record.entropy_after


def _episode_reason(
    outcome_reason: str, forced_pushes: int | None, config: ProbingEnvConfig
) -> str:
    """Report label for the probing loop's stop.  A budget stop is the matched
    baseline's when the forced count is within max_pushes, else max_pushes."""
    if outcome_reason != "budget":
        return outcome_reason
    if forced_pushes is not None and forced_pushes <= config.max_pushes:
        return "matched_budget"
    return "max_pushes"


def _drake_episode(
    model: MaskablePPO | None,
    config: ProbingEnvConfig,
    truth_config: TruthSimConfig,
    filter_model: DynamicsModel,
    seed: int,
    forced_pushes: int | None,
    policy: bool,
    r_matrix: np.ndarray,
) -> EpisodeRecord:
    """One episode in the Drake world.  `policy` runs the trained policy's
    stop decision; otherwise the episode runs `forced_pushes` random pushes
    (the paired baseline)."""
    eta, start_pose, action_rng = episode_conditions(seed)
    obj = episode_object(eta, start_pose, ROD_MASS)
    spans = grid_spans(config.grid)
    belief = GridBayesFilter(config.grid, filter_model, r_matrix, nu=3.0)
    state = ProbingState(
        truth_sim=TruthSim(obj, Pusher(), truth_config),
        obj=obj,
        belief=belief,
    )
    if policy:
        assert model is not None
        selector = policy_selector(model, config)
        none_reason = "stop"
    else:
        selector = random_selector(config, action_rng)
        none_reason = "unreachable"
    budget = (
        config.max_pushes
        if forced_pushes is None
        else min(forced_pushes, config.max_pushes)
    )
    outcome = run_probing(state, config, selector, budget, none_reason)

    total = 0.0
    for record in outcome.records:
        total += config.entropy_weight * _entropy_drop(record) - config.step_cost
    terminal = stop_reward(belief, eta, spans)
    if outcome.reason == "out_of_bounds":
        terminal -= config.out_of_bounds_penalty
    total += terminal

    component_error = belief.estimation_error(eta)
    return EpisodeRecord(
        seed=seed,
        policy="learned" if policy else "random",
        pushes=state.pushes,
        reward=total,
        error=float(np.linalg.norm(component_error / spans)),
        friction_error=float(component_error[0]),
        rod_x_error=float(component_error[1]),
        rod_y_error=float(component_error[2]),
        entropy=belief.entropy(),
        reason=_episode_reason(outcome.reason, forced_pushes, config),
    )


def _surrogate_episode(
    model: MaskablePPO, config: ProbingEnvConfig, seed: int
) -> EpisodeRecord:
    """The same episode in the surrogate world, for the gap measurement."""
    env = ProbingEnv(config)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    terminated = False
    info: dict[str, object] = {}
    while not terminated:
        masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, reward, terminated, _, info = env.step(int(action))
        total += reward
    component_error = cast(np.ndarray, info["estimation_error"])
    return EpisodeRecord(
        seed=seed,
        policy="learned",
        pushes=int(cast(int, info["pushes"])),
        reward=total,
        error=float(np.linalg.norm(component_error / grid_spans(config.grid))),
        friction_error=float(component_error[0]),
        rod_x_error=float(component_error[1]),
        rod_y_error=float(component_error[2]),
        entropy=cast(float, info["entropy"]),
        reason=str(info["reason"]),
    )


def _run_episodes(
    model: MaskablePPO,
    config: ProbingEnvConfig,
    truth_config: TruthSimConfig,
    filter_model: DynamicsModel,
    r_matrix: np.ndarray,
    episodes: int,
    seed: int,
) -> tuple[list[EpisodeRecord], list[EpisodeRecord], list[EpisodeRecord]]:
    """Run the learned episodes, their paired random baselines, and the
    same episodes in the surrogate world (the gap reference)."""
    learned_records: list[EpisodeRecord] = []
    random_records: list[EpisodeRecord] = []
    surrogate_records: list[EpisodeRecord] = []
    for episode in range(episodes):
        episode_seed = seed + episode
        learned = _drake_episode(
            model,
            config,
            truth_config,
            filter_model,
            episode_seed,
            forced_pushes=None,
            policy=True,
            r_matrix=r_matrix,
        )
        learned_records.append(learned)
        baseline = _drake_episode(
            model,
            config,
            truth_config,
            filter_model,
            episode_seed,
            forced_pushes=learned.pushes,
            policy=False,
            r_matrix=r_matrix,
        )
        random_records.append(baseline)
        surrogate_records.append(_surrogate_episode(model, config, episode_seed))
        print(
            f"episode {episode}: drake learned err {learned.error:.3f} "
            f"({learned.pushes} pushes, {learned.reason}) | random "
            f"{baseline.error:.3f} | surrogate "
            f"{surrogate_records[-1].error:.3f}",
            flush=True,
        )
    return learned_records, random_records, surrogate_records


def run_evaluation(
    model_path: Path,
    ensemble_path: Path,
    family_path: Path,
    out_path: Path,
    episodes: int,
    seed: int,
    workers: int,
    predictor: str = "drake",
    random_only: bool = False,
    random_pushes: int | None = None,
) -> str:
    """Run the paired evaluation — the learned policy and the
    random-probing baseline through the identical Drake pipeline — and
    write the report.  `predictor` picks the filter's prediction model:
    the Drake plant (gold-standard accuracy), or the surrogate (online
    speed, calibrated to CALIBRATED_R_SURROGATE)."""
    truth_config = TruthSimConfig(time_step=1e-4, settle_time=0.1, log_period=1e-3)
    started = time.perf_counter()
    if random_only:
        if predictor != "drake":
            raise ValueError("random-only evaluation requires the Drake predictor")
        config = make_probing_env_config(None, family_path=None)
        with ParallelPushModel(
            workers, PushModelConfig(time_step=2e-3, rod_mass=ROD_MASS)
        ) as random_model:
            records = [
                _drake_episode(
                    None,
                    config,
                    truth_config,
                    random_model,
                    seed + episode,
                    forced_pushes=(
                        config.max_pushes if random_pushes is None else random_pushes
                    ),
                    policy=False,
                    r_matrix=CALIBRATED_R,
                )
                for episode in range(episodes)
            ]
        report = {
            "episodes": episodes,
            "seed": seed,
            "predictor": predictor,
            "drake_world": {"random": _summary(records).__dict__},
            "per_episode": [record.__dict__ for record in records],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        summary = _summary(records)
        return (
            f"drake random: reward {summary.mean_reward:+.3f} "
            f"err {summary.mean_error:.3f} ({summary.mean_pushes:.1f} pushes) "
            f"({time.perf_counter() - started:.0f} s)"
        )

    config = make_probing_env_config(ensemble_path, family_path=family_path)
    model = MaskablePPO.load(model_path)
    filter_model: DynamicsModel
    r_matrix: np.ndarray
    if predictor == "surrogate":
        filter_model = SurrogatePredictor(load_ensemble(ensemble_path))
        r_matrix = CALIBRATED_R_SURROGATE
        drake_learned, drake_random, surrogate_learned = _run_episodes(
            model, config, truth_config, filter_model, r_matrix, episodes, seed
        )
    else:
        with ParallelPushModel(
            workers, PushModelConfig(time_step=2e-3, rod_mass=ROD_MASS)
        ) as filter_model:
            drake_learned, drake_random, surrogate_learned = _run_episodes(
                model,
                config,
                truth_config,
                filter_model,
                CALIBRATED_R,
                episodes,
                seed,
            )

    report = {
        "episodes": episodes,
        "seed": seed,
        "predictor": predictor,
        "drake_world": {
            "learned": _summary(drake_learned).__dict__,
            "random": _summary(drake_random).__dict__,
        },
        "surrogate_world": {
            "learned": _summary(surrogate_learned).__dict__,
        },
        "per_episode": [
            {
                "seed": record.seed,
                "drake_learned": record.__dict__,
                "drake_random": baseline.__dict__,
            }
            for record, baseline in zip(drake_learned, drake_random)
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    dl, dr = _summary(drake_learned), _summary(drake_random)
    return (
        f"drake world ({predictor} predictor): learned reward "
        f"{dl.mean_reward:+.3f} err {dl.mean_error:.3f} ({dl.mean_pushes:.1f} pushes) "
        f"vs random {dr.mean_reward:+.3f} err {dr.mean_error:.3f} "
        f"({time.perf_counter() - started:.0f} s)"
    )
