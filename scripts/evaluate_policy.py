"""Score a trained probing policy against random probing in the surrogate world,
and summarize which pushes it chooses.

Each episode seed runs twice from identical conditions (true eta, start pose,
noise stream): the learned policy (env.policy_decision) until it stops, then
uniformly random reachable pushes, as many as the policy used, then stop.
Errors are the final MAP's, per parameter and range-normalized as in the
training reward.

    uv run python -m scripts.evaluate_policy --out data/results/ppo_evaluation
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np

from src import paths
from src.estimation.params import grid_spans
from src.policy.env import (
    ProbingEnv,
    ProbingEnvConfig,
    make_probing_env_config,
    policy_decision,
)

# Picks a push index (or the last index, stop) from an observation and mask.
Chooser = Callable[[np.ndarray, np.ndarray], int]

SPEED_BINS = (0.03, 0.07, 0.11, 0.15)  # m/s edges: slow, medium, fast
DISTANCE_BINS = (0.0, 0.035, 0.068, 0.1)  # m edges: short, medium, long


@dataclass(frozen=True)
class Episode:
    """One episode's outcome: return, pushes, final MAP error (mu, bx m, by m)
    and its range-normalized norm, and the chosen push indices."""

    seed: int
    policy: str
    reward: float
    pushes: int
    error: list[float]
    normalized_error: float
    actions: list[int]


def run_episode(
    env: ProbingEnv,
    config: ProbingEnvConfig,
    seed: int,
    choose: Chooser,
    budget: int | None,
) -> Episode:
    """Run from `seed`, taking `choose`'s actions until it stops or, with a
    `budget`, until that many pushes, then stop."""
    obs, _ = env.reset(seed=seed)
    stop = config.n_actions - 1
    total, done = 0.0, False
    actions: list[int] = []
    info: dict[str, object] = {}
    while not done:
        if budget is not None and len(actions) >= budget:
            action = stop
        else:
            action = choose(obs, env.action_masks())
        obs, reward, done, _, info = env.step(action)
        total += float(reward)
        if action != stop:
            actions.append(action)
    error = cast(np.ndarray, info["estimation_error"])
    return Episode(
        seed=seed,
        policy="",
        reward=total,
        pushes=len(actions),
        error=error.tolist(),
        normalized_error=float(np.linalg.norm(error / grid_spans(config.grid))),
        actions=actions,
    )


def random_chooser(seed: int) -> Chooser:
    """Uniformly random reachable pushes, stopping when none is reachable."""
    rng = np.random.default_rng(seed)

    def choose(_obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask[:-1])
        return int(rng.choice(legal)) if len(legal) else len(mask) - 1

    return choose


def evaluate(
    model_path: Path, ensemble: Path, family: Path, episodes: int, seed: int
) -> tuple[ProbingEnvConfig, list[Episode], list[Episode]]:
    from sb3_contrib import MaskablePPO

    config = make_probing_env_config(ensemble, family_path=family)
    model = MaskablePPO.load(model_path)
    env = ProbingEnv(config)
    learned, random = [], []
    for k in range(episodes):
        episode_seed = seed + k
        mine = run_episode(
            env, config, episode_seed, lambda o, m: policy_decision(model, o, m), None
        )
        theirs = run_episode(
            env, config, episode_seed, random_chooser(episode_seed), mine.pushes
        )
        learned.append(Episode(**{**asdict(mine), "policy": "learned"}))
        random.append(Episode(**{**asdict(theirs), "policy": "random"}))
    return config, learned, random


def shares(values: np.ndarray, edges: tuple[float, ...]) -> list[float]:
    """Fraction of `values` in each bin between consecutive `edges`."""
    counts, _ = np.histogram(values, bins=edges)
    return [float(c) for c in counts / max(counts.sum(), 1)]


def face_of(normal: np.ndarray) -> str:
    """The push direction as a compass label in the block frame."""
    angle = np.degrees(np.arctan2(normal[1], normal[0])) % 360
    return ("+x", "+y", "-x", "-y")[int(((angle + 45) % 360) // 90)]


def summary(
    config: ProbingEnvConfig, learned: list[Episode], random: list[Episode]
) -> dict[str, object]:
    def stats(group: list[Episode]) -> dict[str, object]:
        errors = np.abs(np.array([e.error for e in group]))
        normalized = np.array([e.normalized_error for e in group])
        actions = np.array([a for e in group for a in e.actions], dtype=int)
        speeds = config.speeds[actions]
        distances = config.speeds[actions] * config.times[actions]
        faces = [face_of(config.contact_normals[a]) for a in actions]
        n = len(group)
        return {
            "return_mean": float(np.mean([e.reward for e in group])),
            "normalized_error_mean": float(normalized.mean()),
            "normalized_error_se": float(normalized.std(ddof=1) / np.sqrt(n)),
            "mu_abs_error_mean": float(errors[:, 0].mean()),
            "rod_x_abs_error_cm_mean": float(errors[:, 1].mean() * 100),
            "rod_y_abs_error_cm_mean": float(errors[:, 2].mean() * 100),
            "within_one_step_share": float(
                np.mean(
                    np.all(errors <= np.array([0.1, 0.0165, 0.0165]) + 1e-9, axis=1)
                )
            ),
            "pushes_mean": float(np.mean([e.pushes for e in group])),
            "pushes_histogram": np.bincount(
                [e.pushes for e in group], minlength=config.max_pushes + 1
            ).tolist(),
            "speed_shares": shares(speeds, SPEED_BINS),
            "distance_shares": shares(distances, DISTANCE_BINS),
            "push_direction_shares": {
                f: faces.count(f) / max(len(faces), 1) for f in ("+x", "+y", "-x", "-y")
            },
            "distinct_pushes": len(set(actions.tolist())),
        }

    paired = np.array(
        [
            r.normalized_error - m.normalized_error
            for m, r in zip(learned, random, strict=True)
        ]
    )
    return {
        "episodes": len(learned),
        "learned": stats(learned),
        "random": stats(random),
        "paired_error_advantage_mean": float(paired.mean()),
        "paired_error_advantage_se": float(paired.std(ddof=1) / np.sqrt(len(paired))),
        "learned_better_share": float(np.mean(paired > 0)),
    }


def plot(result: dict[str, object], out: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    learned = cast(dict[str, list[float]], result["learned"])
    random = cast(dict[str, list[float]], result["random"])
    figure = Figure(figsize=(10, 3.6), facecolor="#fcfcfb")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    panels = (
        (
            "speed_shares",
            "push speed",
            ("slow\n3–7 cm/s", "medium\n7–11", "fast\n11–15"),
        ),
        (
            "distance_shares",
            "push distance",
            ("short\n<3.5 cm", "medium\n3.5–6.8", "long\n>6.8"),
        ),
    )
    width = 0.38
    for ax, (key, title, labels) in zip(axes[:2], panels, strict=False):
        x = np.arange(len(labels))
        ax.bar(
            x - width / 2, learned[key], width - 0.03, color="#2a78d6", label="learned"
        )
        ax.bar(
            x + width / 2, random[key], width - 0.03, color="#b8b6ad", label="random"
        )
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_title(f"Share of pushes by {title}", fontsize=10, loc="left")
        ax.set_ylim(0, 1)
    hist_l = np.array(learned["pushes_histogram"], dtype=float)
    x = np.arange(len(hist_l))
    axes[2].bar(x, hist_l / hist_l.sum(), color="#2a78d6", width=0.8)
    axes[2].set_title("Pushes before stopping (learned)", fontsize=10, loc="left")
    axes[2].set_xticks(x)
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color="#e1e0d9", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(out, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=paths.PPO_POLICY)
    parser.add_argument("--ensemble", type=Path, default=paths.SURROGATE_ENSEMBLE)
    parser.add_argument("--path-family", type=Path, default=paths.PATH_FAMILY)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=70000)
    parser.add_argument("--out", type=Path, default=paths.RESULTS / "ppo_evaluation")
    args = parser.parse_args()
    config, learned, random = evaluate(
        args.model, args.ensemble, args.path_family, args.episodes, args.seed
    )
    result = {
        "model": str(args.model),
        "seed": args.seed,
        **summary(config, learned, random),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
    (args.out / "episodes.json").write_text(
        json.dumps([asdict(e) for e in learned + random]) + "\n"
    )
    plot(result, args.out / "behaviour.png")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
