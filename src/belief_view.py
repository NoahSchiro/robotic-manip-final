"""Live views of the estimator's belief, for watching it work in --estimate
or an MPC episode.

Two views, both refreshed after every push:
- in Meshcat, the belief over the rod's position drawn onto the block's top
  face, next to the red marker that shows where the rod truly is;
- a matplotlib window charting the belief over the rod's position and over
  friction, with the entropy, the most likely hypothesis's probability and
  the MAP's error over the pushes so far.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from pydrake.all import Cylinder, RigidTransform
from pydrake.geometry import Meshcat, Rgba

from .estimator import GridBayesFilter
from .geometry import SurfacePart, boundary_edges
from .object import Object
from .params import Eta

# A grid position holding all the probability gets a disc this big, smaller
# than the true rod's marker so a ring of the marker shows around it.
_DISC_MAX_RADIUS = 0.006
_DISC_MIN_RADIUS = 0.001
_DISC_THICKNESS = 0.0005
_DISC_RGBA = Rgba(1.0, 0.85, 0.1, 0.9)


def rod_marginal(
    grid: Sequence[Eta], probs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sum `probs` over friction: get the grid's distinct rod positions, as
    (N, 2) rows of (bx, by), with each one's probability."""
    positions, index = np.unique(
        [(eta.bx, eta.by) for eta in grid], axis=0, return_inverse=True
    )
    return positions, np.bincount(index.ravel(), weights=probs)


def mu_marginal(
    grid: Sequence[Eta], probs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sum `probs` over rod positions: get the grid's distinct friction values
    with each one's probability."""
    mus, index = np.unique([eta.mu for eta in grid], return_inverse=True)
    return mus, np.bincount(index, weights=probs)


def grid_steps(grid: Sequence[Eta]) -> np.ndarray:
    """Get the grid's spacing in each of (mu, bx, by); a parameter with a
    single value has an infinite step."""
    points = np.array([eta.as_array() for eta in grid])
    steps = [np.diff(np.unique(column)) for column in points.T]
    return np.array([step.min() if step.size else np.inf for step in steps])


def show_rod_belief(meshcat: Meshcat, obj: Object, belief: GridBayesFilter) -> None:
    """Draw the belief over the rod's position on `obj`'s top face: a yellow
    disc at each grid position, its area proportional to the probability
    there.  The discs ride with the block as it is pushed."""
    positions, probs = rod_marginal(belief.grid, belief.probs)
    # Children of the visualizer's path for the block's body move with it.
    path = f"visualizer/{obj.name}/{obj.body_box}/rod_belief"
    z = obj.height + obj.rod_marker_thickness + _DISC_THICKNESS / 2
    for i, ((bx, by), p) in enumerate(zip(positions, probs)):
        radius = max(_DISC_MIN_RADIUS, _DISC_MAX_RADIUS * np.sqrt(p))
        meshcat.SetObject(f"{path}/{i}", Cylinder(radius, _DISC_THICKNESS), _DISC_RGBA)
        meshcat.SetTransform(f"{path}/{i}", RigidTransform(np.array([bx, by, z])))


class BeliefDashboard:
    """A matplotlib window charting the belief after each push."""

    def __init__(
        self, grid: Sequence[Eta], parts: list[SurfacePart], eta_true: Eta
    ) -> None:
        """Open the window for a belief over `grid`, on the object whose
        footprint is `parts`, whose true parameters are `eta_true`."""
        self.eta_true = eta_true
        self.parts = parts
        self.steps = grid_steps(grid)
        self.max_entropy = float(np.log(len(grid)))
        # One row per update: push, entropy, max p, then the MAP's error in
        # grid steps for mu, bx, by.
        self.history: list[tuple[float, ...]] = []
        plt.ion()
        self.fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Estimator belief")
        (self.ax_rod, self.ax_mu), (self.ax_entropy, self.ax_error) = axes
        self.ax_max_p = self.ax_entropy.twinx()
        plt.show(block=False)

    def update(self, push: int, belief: GridBayesFilter) -> None:
        """Redraw every chart for `belief`, the belief after `push` pushes.
        Before the first push the belief is uniform, so it has no MAP."""
        map_eta = belief.map_eta() if push > 0 else None
        error = belief.estimation_error(self.eta_true)
        step_errors = np.abs(error) / self.steps if push > 0 else np.full(3, np.nan)
        self.history.append(
            (push, belief.entropy(), float(belief.probs.max()), *step_errors)
        )
        self._draw_rod(belief, map_eta, error)
        self._draw_mu(belief, map_eta, error)
        self._draw_history()
        self.fig.suptitle(f"Belief after {push} push{'' if push == 1 else 'es'}")
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _draw_rod(
        self, belief: GridBayesFilter, map_eta: Eta | None, error: np.ndarray
    ) -> None:
        """Chart the belief over the rod's position on the T, in cm."""
        ax = self.ax_rod
        ax.clear()
        for (x0, y0), (x1, y1) in boundary_edges(self.parts):
            ax.plot([x0 * 100, x1 * 100], [y0 * 100, y1 * 100], color="0.4")
        positions, probs = rod_marginal(belief.grid, belief.probs)
        ax.scatter(
            positions[:, 0] * 100,
            positions[:, 1] * 100,
            s=6 + 600 * probs,
            c=probs,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            label="belief (area ~ probability)",
        )
        ax.scatter(
            [self.eta_true.bx * 100],
            [self.eta_true.by * 100],
            marker="+",
            s=200,
            color=Object.rod_marker_color,
            label="true rod",
        )
        if map_eta is None:
            title = "Rod position: uniform before any push"
        else:
            ax.scatter(
                [map_eta.bx * 100],
                [map_eta.by * 100],
                marker="o",
                s=260,
                facecolors="none",
                edgecolors="k",
                label="MAP",
            )
            title = (
                f"Rod position: MAP ({map_eta.bx * 100:+.2f}, "
                f"{map_eta.by * 100:+.2f}) cm, error ({error[1] * 100:+.2f}, "
                f"{error[2] * 100:+.2f}) cm"
            )
        ax.set_aspect("equal")
        ax.set_xlabel("rod x (cm, body frame)")
        ax.set_ylabel("rod y (cm, body frame)")
        ax.set_title(title, fontsize=10)
        # The T leaves its upper right corner empty.  Fixed legend marker sizes
        # keep a large belief disc from swelling the legend.
        legend = ax.legend(loc="upper right", fontsize=8)
        for handle in legend.legend_handles:
            if isinstance(handle, PathCollection):
                handle.set_sizes([40])

    def _draw_mu(
        self, belief: GridBayesFilter, map_eta: Eta | None, error: np.ndarray
    ) -> None:
        """Chart the belief over friction."""
        ax = self.ax_mu
        ax.clear()
        mus, probs = mu_marginal(belief.grid, belief.probs)
        map_mu = None if map_eta is None else map_eta.mu
        colors = ["k" if mu == map_mu else "tab:blue" for mu in mus]
        ax.bar(mus, probs, width=0.8 * self.steps[0], color=colors)
        ax.axvline(
            self.eta_true.mu,
            color=Object.rod_marker_color,
            linestyle="--",
            label="true mu",
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("friction mu")
        ax.set_ylabel("probability (summed over rod positions)")
        ax.set_title(
            "Friction: uniform before any push"
            if map_eta is None
            else f"Friction: MAP {map_eta.mu:.3f} (black), error {error[0]:+.3f}",
            fontsize=10,
        )
        ax.legend(loc="upper right", fontsize=8)

    def _draw_history(self) -> None:
        """Chart the entropy, max p and MAP error over the pushes so far."""
        push, entropy, max_p, *errors = np.array(self.history).T
        ax = self.ax_entropy
        ax.clear()
        self.ax_max_p.clear()
        ax.plot(push, entropy, "o-", color="tab:purple", label="entropy")
        ax.axhline(self.max_entropy, color="tab:purple", linestyle=":", linewidth=1)
        ax.set_ylim(0.0, 1.05 * self.max_entropy)
        ax.set_xlabel("push")
        ax.set_ylabel("entropy (nats)", color="tab:purple")
        self.ax_max_p.plot(push, max_p, "s-", color="tab:green", label="max p")
        self.ax_max_p.set_ylim(0.0, 1.05)
        # Clearing a twin axis moves its label back to the left.
        self.ax_max_p.yaxis.set_label_position("right")
        self.ax_max_p.set_ylabel("probability of the MAP", color="tab:green")
        ax.set_title(
            f"Certainty: entropy {entropy[-1]:.3f} (dotted: uniform), "
            f"max p {max_p[-1]:.3f}",
            fontsize=10,
        )

        ax = self.ax_error
        ax.clear()
        labels = [
            f"mu (step {self.steps[0]:.2f})",
            f"rod x (step {self.steps[1] * 100:.2f} cm)",
            f"rod y (step {self.steps[2] * 100:.2f} cm)",
        ]
        for values, label in zip(errors, labels):
            ax.plot(push, values, "o-", label=label)
        ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("push")
        ax.set_ylabel("|MAP - true| (grid steps)")
        ax.set_title("MAP error (dashed: one grid step)", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        self.ax_entropy.set_xticks(push)
        self.ax_error.set_xticks(push)


def show_belief(
    push: int,
    belief: GridBayesFilter,
    obj: Object,
    meshcat: Meshcat | None,
    dashboard: BeliefDashboard | None,
) -> None:
    """Draw `belief`, the belief after `push` pushes, in whichever views are
    open."""
    if meshcat is not None:
        show_rod_belief(meshcat, obj, belief)
    if dashboard is not None:
        dashboard.update(push, belief)
