"""Combine per-episode `main.py evaluate` reports (for example from a cluster
array job) into paired statistics.

Each strategy's final MAP error is range-normalized (EpisodeRecord.error);
paired differences compare strategies on identical episodes, so the
standard error reflects push selection rather than episode difficulty.

    uv run python -m scripts.merge_evaluations data/results/transfer/episode_*.json --out data/results/transfer/summary.json
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# One per_episode entry: its seed, and one EpisodeRecord dict per strategy.
PairedEpisode = dict[str, Any]

STRATEGIES = ("drake_learned", "drake_random", "drake_long_fast", "surrogate_learned")
ONE_STEP = np.array([0.1, 0.0165, 0.0165])  # grid step: mu, rod x m, rod y m


def episodes_from(paths: list[Path]) -> list[PairedEpisode]:
    """Every paired episode in the reports, each seed once."""
    by_seed: dict[int, PairedEpisode] = {}
    for path in paths:
        for episode in json.loads(path.read_text())["per_episode"]:
            by_seed[int(episode["seed"])] = episode
    return [by_seed[seed] for seed in sorted(by_seed)]


def strategy_stats(records: list[dict[str, float]]) -> dict[str, float]:
    errors = np.array([r["error"] for r in records])
    signed = np.array(
        [[r["friction_error"], r["rod_x_error"], r["rod_y_error"]] for r in records]
    )
    n = len(records)
    return {
        "episodes": n,
        "error_mean": float(errors.mean()),
        "error_se": float(errors.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        "within_one_step_share": float(
            np.mean(np.all(np.abs(signed) <= ONE_STEP + 1e-9, axis=1))
        ),
        "mu_abs_error_mean": float(np.abs(signed[:, 0]).mean()),
        "rod_x_abs_error_cm_mean": float(np.abs(signed[:, 1]).mean() * 100),
        "rod_y_abs_error_cm_mean": float(np.abs(signed[:, 2]).mean() * 100),
        "pushes_mean": float(np.mean([r["pushes"] for r in records])),
        "reward_mean": float(np.mean([r["reward"] for r in records])),
    }


def paired(episodes: list[PairedEpisode], better: str, worse: str) -> dict[str, float]:
    """`worse`'s error minus `better`'s per episode: positive favours `better`."""
    diff = np.array([e[worse]["error"] - e[better]["error"] for e in episodes])
    return {
        "mean": float(diff.mean()),
        "se": float(diff.std(ddof=1) / np.sqrt(len(diff)))
        if len(diff) > 1
        else float("nan"),
        "better_share": float(np.mean(diff > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    episodes = episodes_from(args.reports)
    present = [s for s in STRATEGIES if all(s in e for e in episodes)]
    strategies = {s: strategy_stats([e[s] for e in episodes]) for s in present}
    comparisons = {
        f"learned_vs_{name}": paired(episodes, "drake_learned", f"drake_{name}")
        for name in ("random", "long_fast")
        if f"drake_{name}" in present
    }
    print(f"{len(episodes)} paired episodes")
    print(
        f"{'strategy':18} {'error':>14} {'within 1 step':>13} {'mu':>6} "
        f"{'rod x cm':>9} {'rod y cm':>9} {'pushes':>7}"
    )
    for name, st in strategies.items():
        print(
            f"{name:18} {st['error_mean']:7.3f} ± {st['error_se']:.3f} "
            f"{st['within_one_step_share']:13.2f} {st['mu_abs_error_mean']:6.3f} "
            f"{st['rod_x_abs_error_cm_mean']:9.2f} {st['rod_y_abs_error_cm_mean']:9.2f} "
            f"{st['pushes_mean']:7.2f}"
        )
    for key, d in comparisons.items():
        print(
            f"{key}: error advantage {d['mean']:+.3f} ± {d['se']:.3f}, "
            f"learned better in {d['better_share']:.0%}"
        )
    if args.out is not None:
        result = {
            "episodes": len(episodes),
            "seeds": [int(e["seed"]) for e in episodes],
            "strategies": strategies,
            **comparisons,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
