"""Top-down maps of where the block center can sit: for each center position,
the share of the 396 lattice pushes the arm can reach, averaged over block yaw.
The first figure marks the near task's workspace and start/goal disks; the
second shows both placement tasks' regions and each seed's start and goal.

    uv run python -m scripts.plot_reach
"""

import argparse
from pathlib import Path

import numpy as np

from src.constants import (
    FAR_BAND_HALF_ANGLE,
    FAR_BAND_RADII,
    FAR_BOUNDS_HALF_ANGLE,
    FAR_BOUNDS_RADII,
    GOAL_RADIUS,
    START_RADIUS,
    WORKSPACE_CENTER,
    WORKSPACE_RADIUS,
)
from src.experiment import TASKS, episode_conditions, episode_goal
from src.policy.env import make_probing_env_config, reachability_mask

TASK_SEEDS = range(95000, 95030)  # the headroom run's seeds


def push_share(xs: np.ndarray, ys: np.ndarray, n_yaws: int = 24) -> np.ndarray:
    """(len(ys), len(xs)) share of lattice pushes reachable for a block centred
    at each world (x, y) m, averaged over evenly spaced yaws."""
    config = make_probing_env_config(None, family_path=None)
    yaws = np.linspace(-np.pi, np.pi, n_yaws, endpoint=False)
    share = np.zeros((len(ys), len(xs)))
    for a, y in enumerate(ys):
        for b, x in enumerate(xs):
            share[a, b] = np.mean(
                [reachability_mask(np.array([x, y, w]), config).mean() for w in yaws]
            )
    return share


def plot(xs: np.ndarray, ys: np.ndarray, share: np.ndarray, out: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    figure = Figure(figsize=(6.4, 5.6), facecolor="#fcfcfb")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    image = ax.pcolormesh(
        xs, ys, share * 100, cmap="Blues", vmin=0, vmax=100, shading="nearest"
    )
    figure.colorbar(image, ax=ax, shrink=0.8, label="lattice pushes reachable (%)")
    ax.add_patch(
        Circle(WORKSPACE_CENTER, WORKSPACE_RADIUS, fill=False, lw=1.6, ec="#eb6834")
    )
    ax.add_patch(
        Circle(
            WORKSPACE_CENTER,
            max(START_RADIUS, GOAL_RADIUS),
            fill=False,
            ls=":",
            lw=1.6,
            ec="#eb6834",
        )
    )
    ax.plot([0], [0], marker="s", color="#0b0b0b", ms=6)
    ax.legend(
        handles=[
            Line2D([], [], color="#eb6834", label="workspace disk"),
            Line2D([], [], ls=":", color="#eb6834", label="start/goal disk"),
            Line2D([], [], marker="s", ls="", color="#0b0b0b", label="arm base"),
        ],
        frameon=False,
        fontsize=8,
        loc="lower left",
    )
    ax.set_title("Allowable block positions", fontsize=10, loc="left")
    ax.set_aspect("equal")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=150)


def plot_tasks(
    xs: np.ndarray, ys: np.ndarray, share: np.ndarray, seeds: range, out: Path
) -> None:
    """One panel per placement task: the reach map in grey, the task's
    start/goal region (solid) and bounds (dashed), and each seed's start
    (dot) with an arrow to its goal."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch, Wedge

    def sector(radii: tuple[float, float], half_angle: float, **style: object) -> Wedge:
        degrees = float(np.rad2deg(half_angle))
        return Wedge(
            (0.0, 0.0),
            radii[1],
            -degrees,
            degrees,
            width=radii[1] - radii[0],
            fill=False,
            **style,
        )

    regions: dict[str, list[Patch]] = {
        "near": [
            Circle(
                WORKSPACE_CENTER,
                max(START_RADIUS, GOAL_RADIUS),
                fill=False,
                ec="#eb6834",
                lw=1.6,
            ),
            Circle(
                WORKSPACE_CENTER,
                WORKSPACE_RADIUS,
                fill=False,
                ec="#eb6834",
                ls="--",
                lw=1.2,
            ),
        ],
        "far": [
            sector(FAR_BAND_RADII, FAR_BAND_HALF_ANGLE, ec="#eb6834", lw=1.6),
            sector(
                FAR_BOUNDS_RADII, FAR_BOUNDS_HALF_ANGLE, ec="#eb6834", ls="--", lw=1.2
            ),
        ],
    }
    figure = Figure(figsize=(11.5, 6.2), facecolor="#fcfcfb")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, len(regions), sharey=True)
    for ax, (name, patches) in zip(axes, regions.items(), strict=True):
        image = ax.pcolormesh(
            xs, ys, share * 100, cmap="Greys", vmin=0, vmax=160, shading="nearest"
        )
        for patch in patches:
            ax.add_patch(patch)
        task = TASKS[name]
        distances = []
        for seed in seeds:
            _, start, _ = episode_conditions(seed, task)
            goal = episode_goal(seed, start, task)
            distances.append(np.hypot(goal[0] - start[0], goal[1] - start[1]) * 100)
            ax.annotate(
                "",
                xy=goal[:2],
                xytext=start[:2],
                arrowprops={"arrowstyle": "->", "color": "#2a78d6", "lw": 0.9},
            )
            ax.plot(*start[:2], "o", ms=3, color="#0b0b0b")
        ax.plot([0], [0], marker="s", color="#0b0b0b", ms=6)
        ax.set_title(
            f"{name} task ({len(seeds)} seeds, start to goal "
            f"{np.mean(distances):.0f} cm on average)",
            fontsize=9,
            loc="left",
        )
        ax.set_aspect("equal")
        ax.set_xlim(-0.1, 0.85)
        ax.set_ylim(-0.8, 0.8)
        ax.set_xlabel("world x (m)")
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("world y (m)")
    bar = figure.colorbar(
        image, ax=list(axes), shrink=0.6, label="lattice pushes reachable (%)"
    )
    bar.ax.set_ylim(0, 100)
    axes[0].legend(
        handles=[
            Line2D([], [], marker="o", ls="", ms=4, color="#0b0b0b", label="start"),
            Line2D([], [], color="#2a78d6", label="to goal"),
            Patch(fill=False, ec="#eb6834", lw=1.6, label="start/goal region"),
            Patch(fill=False, ec="#eb6834", ls="--", lw=1.2, label="bounds"),
            Line2D([], [], marker="s", ls="", color="#0b0b0b", label="arm base"),
        ],
        frameon=False,
        fontsize=8,
        loc="lower left",
    )
    figure.savefig(out, dpi=150, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/figures/reach.png"))
    parser.add_argument(
        "--tasks-out", type=Path, default=Path("docs/figures/placement_tasks.png")
    )
    args = parser.parse_args()
    xs = ys = np.arange(-0.95, 0.951, 0.025)
    share = push_share(xs, ys)
    plot(xs, ys, share, args.out)
    plot_tasks(xs, ys, share, TASK_SEEDS, args.tasks_out)
    print(f"wrote {args.out} and {args.tasks_out}")


if __name__ == "__main__":
    main()
