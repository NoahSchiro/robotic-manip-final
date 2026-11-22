"""Top-down map of where the block can sit: for each block centre position,
the share of the 396 lattice pushes the arm can reach, averaged over block
yaw.  The workspace and start/goal disks are drawn on top.

    uv run python -m scripts.plot_reach
"""

import argparse
from pathlib import Path

import numpy as np

from src.constants import GOAL_RADIUS, START_RADIUS, WORKSPACE_CENTER, WORKSPACE_RADIUS
from src.policy.env import make_probing_env_config, reachability_mask


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/figures/reach.png"))
    args = parser.parse_args()
    xs = ys = np.arange(-0.95, 0.951, 0.025)
    plot(xs, ys, push_share(xs, ys), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
