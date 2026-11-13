"""Record one probing episode's belief and plot how it sharpens push by push.

    uv run python -m scripts.plot_belief record --seed 123 --pushes 3 --out data/belief_seed123.npz
    uv run python -m scripts.plot_belief plot data/belief_seed123.npz --out docs/figures/belief_seed123.png

`record` runs the truth sim and the Drake prediction model.  `plot` reads only
the .npz and uses numpy, matplotlib and the outline geometry in src/geometry.py.
"""

import argparse
import dataclasses
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from src.geometry import SurfacePart, boundary_edges, footprint_center, trace_outline

if TYPE_CHECKING:
    from src.probing import ProbingState, PushRecord

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]

# Palette (docs-facing values from the dataviz reference palette).  The
# sequential ramp is blue steps 100-700; orange is categorical slot 2 and marks
# the push; everything else is text or neutral tokens.
SEQUENTIAL_BLUE = ("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
BAR_BLUE = "#2a78d6"
PUSH_ORANGE = "#eb6834"
SURFACE = "#fcfcfb"
OUTLINE_FILL = "#f0efec"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

M_TO_CM = 100.0
MIN_MARKER_AREA = 36.0  # pt^2 (6 pt across), so the faintest hypothesis stays visible
MAX_MARKER_AREA = 200.0  # pt^2, the most probable rod position
GRID_MARKER_AREA = 12.0  # pt^2, the hollow marker for every hypothesis
CONTACT_ARROW_CM = 3.0
MAP_MARGIN_CM = 5.0


@dataclass(frozen=True)
class BeliefRecord:
    """One episode's belief history.  With P pushes and N hypotheses: the belief
    before any push is probs[0], and probs[k] follows push k.  Rod positions and
    contacts are in the body frame (origin at the footprint's bottom-face
    center, m); `parts` is the outline in the footprint frame (m)."""

    grid: np.ndarray  # (N, 3) hypotheses (mu, bx, by)
    probs: np.ndarray  # (P+1, N) belief after 0..P pushes
    eta_true: np.ndarray  # (3,) (mu, bx, by) that generated the data
    parts: np.ndarray  # (K, 4) outline boxes (size_x, size_y, center_x, center_y)
    poses: np.ndarray  # (P+1, 3) block (x m, y m, yaw rad) in the world frame
    entropy: np.ndarray  # (P+1,) nats
    contact_position: np.ndarray  # (P, 3) contact point at mid-height
    contact_normal: np.ndarray  # (P, 3) inward unit normal of the contact
    speed: np.ndarray  # (P,) push speed along the normal, m/s
    duration: np.ndarray  # (P,) push window length, s
    metadata: dict[str, Any]

    @property
    def pushes(self) -> int:
        return len(self.speed)

    def __post_init__(self) -> None:
        pushes = self.pushes
        expected = {
            "grid": (len(self.grid), 3),
            "probs": (pushes + 1, len(self.grid)),
            "eta_true": (3,),
            "poses": (pushes + 1, 3),
            "entropy": (pushes + 1,),
            "contact_position": (pushes, 3),
            "contact_normal": (pushes, 3),
            "duration": (pushes,),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise ValueError(f"{name} has shape {actual}, expected {shape}")


def save_record(path: Path, record: BeliefRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"schema_version": SCHEMA_VERSION, **record.metadata}
    np.savez_compressed(
        path,
        grid=record.grid,
        probs=record.probs,
        eta_true=record.eta_true,
        parts=record.parts,
        poses=record.poses,
        entropy=record.entropy,
        contact_position=record.contact_position,
        contact_normal=record.contact_normal,
        speed=record.speed,
        duration=record.duration,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )


def load_record(path: Path) -> BeliefRecord:
    with np.load(path) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path} has schema {metadata.get('schema_version')}, "
                f"expected {SCHEMA_VERSION}"
            )
        return BeliefRecord(
            grid=data["grid"],
            probs=data["probs"],
            eta_true=data["eta_true"],
            parts=data["parts"],
            poses=data["poses"],
            entropy=data["entropy"],
            contact_position=data["contact_position"],
            contact_normal=data["contact_normal"],
            speed=data["speed"],
            duration=data["duration"],
            metadata=metadata,
        )


# -- recording (needs Drake; imported lazily so `plot` stays Drake-free) -------


@dataclass
class _Trace:
    """The belief and push history collected while the episode runs."""

    probs: list[np.ndarray]
    entropy: list[float]
    poses: list[np.ndarray]
    contact_position: list[np.ndarray]
    contact_normal: list[np.ndarray]
    speed: list[float]
    duration: list[float]

    def on_push(self, state: ProbingState, push: PushRecord) -> None:
        assert state.belief is not None
        action = push.observation.action
        self.probs.append(state.belief.probs.copy())
        self.entropy.append(state.belief.entropy())
        self.poses.append(np.array(state.obj.pose, dtype=float))
        self.contact_position.append(np.asarray(action.contact.position, dtype=float))
        self.contact_normal.append(np.asarray(action.contact.normal, dtype=float))
        self.speed.append(float(action.velocity))
        self.duration.append(push.observation.duration)


def git_state() -> dict[str, Any]:
    """The checkout's commit and whether it has uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except OSError, subprocess.CalledProcessError:
        return {"commit": "unknown", "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def record_episode(args: argparse.Namespace) -> BeliefRecord:
    """Run one episode through the shared probing loop with random pushes and
    a grid Bayes filter, collecting the belief after each push."""
    from src.dynamics import ParallelPushModel, PushModelConfig
    from src.env import ROD_MASS
    from src.estimator import CALIBRATED_R, GridBayesFilter
    from src.iiwa import Pusher
    from src.params import Eta, GridSpec, make_grid
    from src.probing import (
        ProbingState,
        episode_conditions,
        episode_object,
        random_selector,
        run_probing,
    )
    from src.push_space import make_push_space
    from src.recording import TruthSim, TruthSimConfig

    sampled, start_pose, action_rng = episode_conditions(args.seed)
    eta = Eta(
        mu=args.mu if args.mu is not None else sampled.mu,
        bx=args.rod_x if args.rod_x is not None else sampled.bx,
        by=args.rod_y if args.rod_y is not None else sampled.by,
    )
    obj = episode_object(eta, start_pose, ROD_MASS)
    grid = make_grid(GridSpec(), obj.parts)
    truth_config = TruthSimConfig(time_step=1e-4, settle_time=0.1, log_period=1e-3)
    model_config = PushModelConfig(time_step=2e-3, rod_mass=ROD_MASS)

    with ParallelPushModel(args.workers, model_config) as model:
        belief = GridBayesFilter(grid, model, CALIBRATED_R, nu=3.0)
        trace = _Trace(
            probs=[belief.probs.copy()],
            entropy=[belief.entropy()],
            poses=[np.array(obj.pose, dtype=float)],
            contact_position=[],
            contact_normal=[],
            speed=[],
            duration=[],
        )
        state = ProbingState(
            truth_sim=TruthSim(obj, Pusher(), truth_config), obj=obj, belief=belief
        )
        outcome = run_probing(
            state,
            random_selector(make_push_space(obj.parts, obj.height), action_rng),
            budget=args.pushes,
            on_push=trace.on_push,
        )
    print(f"stopped after {state.pushes} pushes: {outcome.reason}")

    return BeliefRecord(
        grid=np.array([h.as_array() for h in grid], dtype=float),
        probs=np.array(trace.probs, dtype=float),
        eta_true=eta.as_array(),
        parts=np.array(obj.parts, dtype=float),
        poses=np.array(trace.poses, dtype=float),
        entropy=np.array(trace.entropy, dtype=float),
        contact_position=np.array(trace.contact_position, dtype=float).reshape(-1, 3),
        contact_normal=np.array(trace.contact_normal, dtype=float).reshape(-1, 3),
        speed=np.array(trace.speed, dtype=float),
        duration=np.array(trace.duration, dtype=float),
        metadata={
            "seed": args.seed,
            "pushes_requested": args.pushes,
            "pushes_run": state.pushes,
            "stop_reason": outcome.reason,
            "eta_sampled": sampled.as_array().tolist(),
            "eta_overrides": {"mu": args.mu, "rod_x": args.rod_x, "rod_y": args.rod_y},
            "workers": args.workers,
            "rod_mass_kg": ROD_MASS,
            "truth_config": dataclasses.asdict(truth_config),
            "model_config": dataclasses.asdict(model_config),
            "likelihood": {"student_t_nu": 3.0, "R": CALIBRATED_R.tolist()},
            "git": git_state(),
        },
    )


# -- plotting (numpy and matplotlib only) --------------------------------------


def outline_vertices(parts: np.ndarray) -> np.ndarray:
    """(V, 2) vertices in meters of the block outline in the body frame,
    counterclockwise."""
    surface_parts: list[SurfacePart] = [
        (float(sx), float(sy), float(cx), float(cy)) for sx, sy, cx, cy in parts
    ]
    loop = trace_outline(boundary_edges(surface_parts))
    center = footprint_center(surface_parts)
    vertices: np.ndarray = np.array([start for start, _end in loop], dtype=float)
    offset: np.ndarray = vertices - center
    return offset


def marginal_groups(
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group hypotheses by rod position and by friction, so probabilities can
    be summed over the other parameter.  Returns the unique rod positions (M, 2),
    each hypothesis's rod index (N,), the unique friction values (K,) and each
    hypothesis's friction index (N,)."""
    rod_xy, rod_index = np.unique(grid[:, 1:], axis=0, return_inverse=True)
    mu_values, mu_index = np.unique(grid[:, 0], return_inverse=True)
    return rod_xy, rod_index.reshape(-1), mu_values, mu_index.reshape(-1)


def build_figure(record: BeliefRecord) -> Figure:
    steps = record.pushes + 1
    rod_xy, rod_index, mu_values, mu_index = marginal_groups(record.grid)
    rod_marginal = np.stack(
        [np.bincount(rod_index, weights=p, minlength=len(rod_xy)) for p in record.probs]
    )
    mu_marginal = np.stack(
        [
            np.bincount(mu_index, weights=p, minlength=len(mu_values))
            for p in record.probs
        ]
    )
    rod_cm = rod_xy * M_TO_CM
    outline_cm = outline_vertices(record.parts) * M_TO_CM
    rod_max = max(float(rod_marginal.max()), 1e-12)
    mu_top = 1.1 * float(mu_marginal.max())
    cmap = LinearSegmentedColormap.from_list("belief_blue", SEQUENTIAL_BLUE)
    norm = Normalize(vmin=0.0, vmax=rod_max)
    true_mu = float(record.eta_true[0])
    true_bx = float(record.eta_true[1])
    true_by = float(record.eta_true[2])
    true_rod_cm = (true_bx * M_TO_CM, true_by * M_TO_CM)
    lo = outline_cm.min(axis=0) - MAP_MARGIN_CM
    hi = outline_cm.max(axis=0) + MAP_MARGIN_CM
    half = float(max(hi - lo)) / 2
    centre = (lo + hi) / 2
    limits_x = (centre[0] - half, centre[0] + half)
    limits_y = (centre[1] - half, centre[1] + half)
    mu_step = float(np.diff(mu_values).min()) if len(mu_values) > 1 else 0.1

    fig = Figure(
        figsize=(2.9 * steps + 1.6, 7.4), facecolor=SURFACE, layout="constrained"
    )
    grid_spec = fig.add_gridspec(2, steps, height_ratios=(2.6, 1.0))
    rod_axes: list[Axes] = []
    mu_axes: list[Axes] = []
    for k in range(steps):
        rod_ax = fig.add_subplot(
            grid_spec[0, k],
            sharex=rod_axes[0] if rod_axes else None,
            sharey=rod_axes[0] if rod_axes else None,
        )
        rod_axes.append(rod_ax)
        map_index = int(np.argmax(record.probs[k]))
        map_rod_cm = record.grid[map_index, 1:] * M_TO_CM
        mu_error = abs(float(record.grid[map_index, 0]) - true_mu)
        rod_error = float(
            np.hypot(
                record.grid[map_index, 1] - true_bx, record.grid[map_index, 2] - true_by
            )
            * M_TO_CM
        )
        draw_rod_panel(
            rod_ax,
            outline_cm=outline_cm,
            rod_cm=rod_cm,
            rod_p=rod_marginal[k],
            cmap=cmap,
            norm=norm,
            true_rod_cm=true_rod_cm,
            map_rod_cm=map_rod_cm,
            push=None
            if k == 0
            else (
                record.contact_position[k - 1, :2] * M_TO_CM,
                record.contact_normal[k - 1, :2],
            ),
        )
        rod_ax.set_xlim(*limits_x)
        rod_ax.set_ylim(*limits_y)
        rod_ax.set_title(
            step_title(k, record, mu_error, rod_error),
            fontsize=8.5,
            color=INK,
            loc="center",
        )
        if k == 0:
            rod_ax.set_ylabel("body y (cm)", fontsize=8, color=INK_SECONDARY)
        rod_ax.set_xlabel("body x (cm)", fontsize=8, color=INK_SECONDARY)

        mu_ax = fig.add_subplot(grid_spec[1, k], sharey=mu_axes[0] if mu_axes else None)
        mu_axes.append(mu_ax)
        draw_mu_panel(
            mu_ax,
            mu_values=mu_values,
            mu_p=mu_marginal[k],
            bar_width=0.6 * mu_step,
            true_mu=true_mu,
            top=mu_top,
        )
        if k == 0:
            mu_ax.set_ylabel("P(mu)", fontsize=8, color=INK_SECONDARY)
        mu_ax.set_xlabel("friction mu", fontsize=8, color=INK_SECONDARY)

    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=rod_axes,
        shrink=0.85,
        label="P(rod position), summed over mu",
    ).ax.tick_params(labelsize=7.5, colors=INK_SECONDARY)
    fig.suptitle(
        f"Belief over friction and rod position, seed {record.metadata.get('seed')}: "
        f"true mu = {true_mu:.3f}, rod = ({true_bx * M_TO_CM:+.2f}, "
        f"{true_by * M_TO_CM:+.2f}) cm",
        fontsize=11,
        color=INK,
    )
    fig.legend(
        handles=legend_handles(),
        loc="outside lower center",
        ncol=6,
        frameon=False,
        fontsize=8,
        labelcolor=INK_SECONDARY,
    )
    return fig


def step_title(k: int, record: BeliefRecord, mu_error: float, rod_error: float) -> str:
    head = "Prior" if k == 0 else f"Push {k}, v = {record.speed[k - 1]:.2f} m/s"
    return (
        f"{head}\nH = {record.entropy[k]:.2f} nats\n"
        f"MAP error: mu {mu_error:.2f}, rod {rod_error:.1f} cm"
    )


def draw_rod_panel(
    ax: Axes,
    *,
    outline_cm: np.ndarray,
    rod_cm: np.ndarray,
    rod_p: np.ndarray,
    cmap: LinearSegmentedColormap,
    norm: Normalize,
    true_rod_cm: tuple[float, float],
    map_rod_cm: np.ndarray,
    push: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    """Draw one step's rod-position marginal on the block outline.  `push` is
    the contact point (cm) and inward normal of the push that led here."""
    ax.add_patch(
        Polygon(
            outline_cm,
            closed=True,
            facecolor=OUTLINE_FILL,
            edgecolor=INK_SECONDARY,
            linewidth=1.0,
            zorder=1,
        )
    )
    ax.scatter(
        rod_cm[:, 0],
        rod_cm[:, 1],
        s=GRID_MARKER_AREA,
        facecolors="none",
        edgecolors=INK_MUTED,
        linewidths=0.6,
        zorder=2,
    )
    sizes = MIN_MARKER_AREA + (MAX_MARKER_AREA - MIN_MARKER_AREA) * np.clip(
        rod_p / norm.vmax, 0.0, 1.0
    )
    ax.scatter(
        rod_cm[:, 0],
        rod_cm[:, 1],
        s=sizes,
        c=rod_p,
        cmap=cmap,
        norm=norm,
        edgecolors=SURFACE,
        linewidths=0.8,
        zorder=3,
    )
    if push is not None:
        contact_cm, normal = push
        tail = contact_cm - CONTACT_ARROW_CM * normal
        ax.annotate(
            "",
            xy=(float(contact_cm[0]), float(contact_cm[1])),
            xytext=(float(tail[0]), float(tail[1])),
            arrowprops={
                "arrowstyle": "-|>",
                "color": PUSH_ORANGE,
                "linewidth": 1.6,
                "mutation_scale": 11,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=5,
        )
        ax.scatter(
            [contact_cm[0]],
            [contact_cm[1]],
            s=40,
            color=PUSH_ORANGE,
            edgecolors=SURFACE,
            linewidths=0.9,
            zorder=6,
        )
    ax.scatter(
        [map_rod_cm[0]],
        [map_rod_cm[1]],
        marker="D",
        s=56,
        facecolors="none",
        edgecolors=INK,
        linewidths=1.4,
        zorder=6,
    )
    ax.scatter(
        [true_rod_cm[0]],
        [true_rod_cm[1]],
        marker="X",
        s=90,
        color=INK,
        edgecolors=SURFACE,
        linewidths=1.2,
        zorder=7,
    )
    style_axes(ax)
    ax.set_aspect("equal")


def draw_mu_panel(
    ax: Axes,
    *,
    mu_values: np.ndarray,
    mu_p: np.ndarray,
    bar_width: float,
    true_mu: float,
    top: float,
) -> None:
    ax.bar(mu_values, mu_p, width=bar_width, color=BAR_BLUE, zorder=2)
    ax.axvline(true_mu, color=INK, linewidth=1.4, zorder=3)
    ax.set_xlim(float(mu_values.min()) - bar_width, float(mu_values.max()) + bar_width)
    ax.set_ylim(0.0, top)
    ax.set_xticks(mu_values)
    ax.set_xticklabels([f"{mu:.1f}" for mu in mu_values])
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    style_axes(ax)


def style_axes(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(labelsize=7.5, colors=INK_SECONDARY, length=2)


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            marker="X",
            linestyle="",
            color=INK,
            markersize=8,
            label="true rod position",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="",
            markerfacecolor="none",
            markeredgecolor=INK,
            markeredgewidth=1.4,
            markersize=7,
            label="MAP rod position (joint)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="none",
            markeredgecolor=INK_MUTED,
            markersize=4,
            label="hypothesis rod position",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="-",
            color=PUSH_ORANGE,
            markersize=6,
            linewidth=1.6,
            label="push contact and direction",
        ),
        Line2D([], [], color=INK, linewidth=1.4, label="true mu"),
    ]


def plot_record(record: BeliefRecord, out: Path) -> None:
    figure = build_figure(record)
    FigureCanvasAgg(figure)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150, facecolor=SURFACE)


# -- command line ---------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    record = commands.add_parser("record", help="run one episode and save its belief")
    record.add_argument("--seed", type=int, required=True)
    record.add_argument("--pushes", type=int, default=3)
    record.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    record.add_argument("--mu", type=float, help="override the sampled friction")
    record.add_argument("--rod-x", type=float, help="override rod x (m, body frame)")
    record.add_argument("--rod-y", type=float, help="override rod y (m, body frame)")
    record.add_argument("--out", type=Path, default=None)

    plot = commands.add_parser("plot", help="draw a saved belief record")
    plot.add_argument("record", type=Path)
    plot.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "record":
        if args.pushes < 0:
            parser.error("--pushes must be nonnegative")
        if args.workers < 1:
            parser.error("--workers must be at least 1")
        if args.out is None:
            args.out = Path(f"data/belief_seed{args.seed}.npz")
    else:
        if args.out is None:
            args.out = args.record.with_suffix(".png")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "record":
        save_record(args.out, record_episode(args))
    else:
        plot_record(load_record(args.record), args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
