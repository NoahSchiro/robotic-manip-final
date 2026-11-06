"""The evaluation report renderer: from an evaluation report json it
produces the paired-results table, the learning-curve plot and the
behavioral summary — the submission-ready artifacts.

    uv run python -u main.py evaluation-report --report data/evaluation_report.json
"""

import json
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO

from .env import ProbingEnv, ProbingEnvConfig, make_probing_env_config


def _plot_curve(curve_path: Path, out: Path) -> None:
    d = np.load(curve_path)
    t = d["timesteps"] / 1000
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(t, d["learned_reward"], label="learned")
    axes[0].plot(t, d["random_reward"], label="random")
    axes[0].set_title("episode return")
    axes[0].legend()
    axes[1].plot(t, d["learned_error"], label="learned")
    axes[1].plot(t, d["random_error"], label="random")
    axes[1].set_title("final MAP error")
    axes[1].legend()
    axes[2].plot(t, d["learned_pushes"])
    axes[2].set_title("pushes per episode")
    for ax in axes:
        ax.set_xlabel("env steps (thousands)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _paired_table(report: dict[str, object]) -> list[str]:
    worlds = cast(dict[str, dict[str, float]], report["drake_world"])
    learned = worlds["learned"]
    random = worlds["random"]
    title = (
        f"## Paired evaluation on the real arm "
        f"({report['episodes']} held-out episodes, {report['predictor']} filter)"
    )
    learned_row = (
        f"| learned probing | {learned['mean_reward']:+.3f} "
        f"| {learned['mean_error']:.3f} | {learned['mean_pushes']:.2f} |"
    )
    random_row = (
        f"| random probing | {random['mean_reward']:+.3f} "
        f"| {random['mean_error']:.3f} | {random['mean_pushes']:.2f} |"
    )
    return [
        "",
        title,
        "",
        "| policy | return | final MAP error | pushes |",
        "|---|---|---|---|",
        learned_row,
        random_row,
    ]


def _behavioral_probe(
    model: MaskablePPO,
    config: ProbingEnvConfig,
    out_dir: Path,
    episodes: int = 24,
    seed: int = 60000,
) -> list[str]:
    """The policy's behavior on fresh episodes: push count, action
    diversity, stop-timing adaptation to belief difficulty."""
    env = ProbingEnv(config)
    pushes_list, errors, action_counts = (
        [],
        [],
        np.zeros(config.n_actions - 1, dtype=int),
    )
    stop_ent, follow_sets = [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        terminated = False
        picks: list[int] = []
        while not terminated:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            if int(action) == config.n_actions - 1:
                break
            action_counts[int(action)] += 1
            picks.append(int(action))
            obs, _, terminated, _, info = env.step(int(action))
        pushes_list.append(int(cast(int, info["pushes"])))
        errors.append(float(np.linalg.norm(cast(np.ndarray, info["estimation_error"]))))
        stop_ent.append(float(cast(float, info["entropy"])))
        follow_sets.append(tuple(picks[1:]))
    pushes_arr, errors_arr = np.array(pushes_list), np.array(errors)

    sites = np.nonzero(action_counts)[0]
    push_line = (
        f"- pushes: mean {pushes_arr.mean():.2f}, histogram "
        f"{np.bincount(pushes_arr).tolist()}"
    )
    error_line = (
        f"- final MAP error: mean {errors_arr.mean():.3f}, "
        f"p50 {np.median(errors_arr):.3f}, max {errors_arr.max():.3f}"
    )
    lines = [
        "",
        "## Behavior on fresh episodes",
        "",
        push_line,
        error_line,
        f"- distinct sites used: {len(sites)}/{config.n_push_actions // (len(config.speeds) * len(config.times))}",
        f"- distinct follow-up sequences: {len(set(follow_sets))}/{episodes}",
        f"- entropy at stop: mean {np.mean(stop_ent):.2f} nats",
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        pushes_arr,
        bins=np.arange(-0.5, config.max_pushes + 1.5).tolist(),
    )
    ax.set_title("pushes per episode (learned policy)")
    ax.set_xlabel("pushes")
    fig.tight_layout()
    fig.savefig(out_dir / "push_histogram.png", dpi=140)
    plt.close(fig)
    return lines


def make_evaluation_report(
    report_path: Path,
    model_path: Path,
    ensemble_path: Path,
    family_path: Path,
    out_dir: Path,
    probe_episodes: int = 24,
) -> str:
    """Render the submission-ready report: the paired table, the learning
    curve, the push histogram and the behavioral summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(report_path.read_text())

    lines = ["# Evaluation report", ""]
    lines += _paired_table(report)

    curve_file = model_path.parent / "learning_curve.npz"
    if curve_file.exists():
        _plot_curve(curve_file, out_dir / "learning_curve.png")
        lines += ["", "![learning curve](learning_curve.png)"]

    config = make_probing_env_config(ensemble_path, family_path=family_path)
    model = MaskablePPO.load(model_path)
    lines += _behavioral_probe(model, config, out_dir, episodes=probe_episodes)

    report_md = out_dir / "evaluation_report.md"
    report_md.write_text("\n".join(lines) + "\n")
    return f"wrote {report_md}"
