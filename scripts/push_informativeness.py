"""How well a push separates neighbouring hypotheses, by speed at matched push
distance.

For each contact on the MPC grid and each (distance, speed) pair, the Drake
prediction model predicts the estimator measurement along the nominal tip path
under a base hypothesis and under that hypothesis moved one grid step in one
parameter (friction, rod x, or rod y).  The separation is the Mahalanobis
distance between the two predictions under CALIBRATED_R: about 1 or more means
one push noticeably shifts the belief between them.  Planning-only; no truth
simulation.

    uv run python -m scripts.push_informativeness --out data/push_informativeness.json
    # Plot saved results without re-simulating
    uv run python -m scripts.push_informativeness --from-json data/push_informativeness.json --plot docs/figures
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pydrake.all import RigidTransform, RollPitchYaw, SpatialVelocity

from src.action import Action
from src.dynamics import DrakePushModel, PushModelConfig
from src.env import ROD_MASS, SETTLE_TIME
from src.estimator import CALIBRATED_R
from src.geometry import SurfaceNormal
from src.iiwa import transfer_height
from src.object import Object, Rod
from src.observation import ObjectState, measurement, planar_error
from src.params import Eta
from src.push_space import make_push_space
from src.surrogate import nominal_tip_path

MODEL_TIME_STEP = 2e-3  # s, as the estimator's predictions
CONTACT_SPACING = 0.012  # m, the MPC grid's contact spacing
# One grid step per parameter: friction, rod x (m), rod y (m).
GRID_STEPS = {
    "mu": (0.1, 0.0, 0.0),
    "rod_x": (0.0, 0.0165, 0.0),
    "rod_y": (0.0, 0.0, 0.0165),
}
BASES = (Eta(0.2, -0.0495, 0.0), Eta(0.7, -0.0495, 0.0))
SPEEDS = (0.03, 0.06, 0.09, 0.12, 0.15)  # m/s, spanning the push space
DISTANCES = (0.015, 0.045, 0.09)  # m
POSE = RigidTransform(RollPitchYaw(0.0, 0.0, 0.0), np.array([0.55, 0.0, 0.0]))

_model: DrakePushModel | None = None
_height = 0.0


def _start_worker(height: float) -> None:
    global _model, _height
    _model = DrakePushModel(
        PushModelConfig(time_step=MODEL_TIME_STEP, rod_mass=ROD_MASS)
    )
    _height = height


def _predict(task: tuple[Action, tuple[float, float, float]]) -> np.ndarray:
    """The (6,) estimator measurement of `action` under one hypothesis."""
    action, eta = task
    assert _model is not None
    path = nominal_tip_path(POSE, action, SETTLE_TIME, _height)
    start = ObjectState(
        pose=POSE, velocity=SpatialVelocity(np.zeros(3), np.full(3, 1e-9))
    )
    _model.set_eta(Eta(*eta))
    push_end, end = _model.predict_path(
        start,
        path.tip_times,
        path.tip_positions,
        path.tip_velocities,
        [path.push_end_time, path.window_end],
    )
    return measurement(push_end, end)


def separation(base: np.ndarray, moved: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Mahalanobis distance under `r` between paired measurements (..., 6),
    with the yaw difference wrapped."""
    diff = planar_error(moved, base)
    squared = np.einsum("...k,kl,...l->...", diff, np.linalg.inv(r), diff)
    return np.asarray(np.sqrt(squared))


def contacts_on_grid(spacing: float) -> list[SurfaceNormal]:
    """The distinct contact points of the MPC grid."""
    reference = Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, ROD_MASS))
    space = make_push_space(reference.parts, reference.height)
    unique = {
        a.contact.position.tobytes(): a.contact for a in space.grid(spacing, 1, 1)
    }
    return list(unique.values())


@dataclass(frozen=True)
class Separation:
    """Separation statistics over the contacts for one base hypothesis,
    parameter step, push distance (m), and speed (m/s)."""

    base_mu: float
    parameter: str
    distance_m: float
    speed_m_per_s: float
    median: float
    p90: float
    share_above_1: float


def summarize(
    separations: np.ndarray,
    combos: Sequence[tuple[float, float]],
    base_mu: float,
    parameter: str,
) -> list[Separation]:
    """Per (distance, speed) column of `separations` (contacts, combos): the
    median, 90th percentile, and share of contacts above 1."""
    return [
        Separation(
            base_mu=base_mu,
            parameter=parameter,
            distance_m=distance,
            speed_m_per_s=speed,
            median=float(np.median(separations[:, j])),
            p90=float(np.percentile(separations[:, j], 90)),
            share_above_1=float(np.mean(separations[:, j] > 1.0)),
        )
        for j, (distance, speed) in enumerate(combos)
    ]


def feasible_combos(
    duration_range: tuple[float, float],
) -> list[tuple[float, float]]:
    """Every (distance m, speed m/s) of DISTANCES x SPEEDS whose contact time
    distance / speed lies within `duration_range` (s)."""
    low, high = duration_range
    return [
        (d, v) for d in DISTANCES for v in SPEEDS if low - 1e-9 <= d / v <= high + 1e-9
    ]


def run(workers: int) -> list[Separation]:
    contacts = contacts_on_grid(CONTACT_SPACING)
    etas = [tuple(base.as_array()) for base in BASES]
    etas += [
        tuple(np.add(base.as_array(), step))
        for base in BASES
        for step in GRID_STEPS.values()
    ]
    reference = Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, ROD_MASS))
    space = make_push_space(reference.parts, reference.height)
    combos = feasible_combos(space.duration_range)
    tasks = [
        (Action(contact=c, velocity=v, time=d / v), eta)
        for c in contacts
        for d, v in combos
        for eta in etas
    ]
    height = transfer_height(
        Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, ROD_MASS))
    )
    with Pool(workers, initializer=_start_worker, initargs=(height,)) as pool:
        predictions = np.array(pool.map(_predict, tasks, chunksize=8))
    predictions = predictions.reshape(len(contacts), len(combos), len(etas), 6)
    rows: list[Separation] = []
    for b, base in enumerate(BASES):
        for k, parameter in enumerate(GRID_STEPS):
            moved = len(BASES) + b * len(GRID_STEPS) + k
            column = separation(
                predictions[:, :, b], predictions[:, :, moved], CALIBRATED_R
            )
            rows += summarize(column, combos, base.mu, parameter)
    return rows


DISTANCE_COLORS = ("#86b6ef", "#3987e5", "#0d366b")  # short to long pushes
PARAMETER_TITLES = {
    "mu": "Friction: neighbouring hypotheses 0.1 apart",
    "rod_x": "Rod x: neighbouring hypotheses 1.65 cm apart",
    "rod_y": "Rod y: neighbouring hypotheses 1.65 cm apart",
}


def plot(rows: Sequence[Separation], out_dir: Path) -> list[Path]:
    """One figure per parameter, sharing a y scale: median separation against
    push speed, one line per push distance, solid at the low base friction
    and dashed at the high one.  Returns the written paths."""
    top = max(row.median for row in rows) * 1.1
    paths = []
    for parameter, title in PARAMETER_TITLES.items():
        path = out_dir / f"push_informativeness_{parameter}.png"
        plot_parameter([r for r in rows if r.parameter == parameter], title, top, path)
        paths.append(path)
    return paths


def plot_parameter(
    rows: Sequence[Separation], title: str, top: float, out: Path
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    figure = Figure(figsize=(6.4, 4.6), facecolor="#fcfcfb")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    distances = sorted({row.distance_m for row in rows})
    base_mus = sorted({row.base_mu for row in rows})
    styles = dict(zip(base_mus, ("-", "--"), strict=False))
    for distance, color in zip(distances, DISTANCE_COLORS, strict=False):
        for base_mu in base_mus:
            line = sorted(
                (r.speed_m_per_s, r.median)
                for r in rows
                if r.distance_m == distance and r.base_mu == base_mu
            )
            speeds, medians = zip(*line, strict=True)
            ax.plot(
                np.array(speeds) * 100,
                medians,
                styles[base_mu],
                color=color,
                lw=2,
                marker="o",
                ms=6,
                markeredgecolor="#fcfcfb",
            )
    ax.axhline(1.0, color="#898781", lw=1, ls=":")
    ax.set_title(title, fontsize=11, loc="left", color="#0b0b0b")
    ax.set_xlabel("push speed (cm/s)")
    ax.set_ylabel("separation after one push\n(median over contact points, units of R)")
    ax.set_xticks(np.array(SPEEDS) * 100)
    ax.set_xlim(1.5, 16.5)
    ax.set_ylim(0, top)
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color="#e1e0d9", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = (
        [
            Line2D([], [], color=c, lw=2, marker="o", label=f"{d * 100:g} cm push")
            for d, c in zip(distances, DISTANCE_COLORS, strict=False)
        ]
        + [
            Line2D(
                [],
                [],
                color="#52514e",
                lw=2,
                ls=styles[m],
                label=f"true friction {m:g}",
            )
            for m in base_mus
        ]
        + [Line2D([], [], color="#898781", lw=1, ls=":", label="1 = barely noticeable")]
    )
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--out", type=Path, default=Path("data/push_informativeness.json")
    )
    parser.add_argument(
        "--from-json", type=Path, help="reuse saved results instead of simulating"
    )
    parser.add_argument(
        "--plot", type=Path, help="also save one figure per parameter in this folder"
    )
    args = parser.parse_args()
    if args.from_json is not None:
        rows = [Separation(**row) for row in json.loads(args.from_json.read_text())]
    else:
        rows = run(args.workers)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([asdict(row) for row in rows], indent=1) + "\n")
    if args.plot is not None:
        for path in plot(rows, args.plot):
            print(f"wrote {path}")
    for row in rows:
        print(
            f"mu {row.base_mu:.1f} {row.parameter:6} {row.distance_m * 100:4.1f} cm "
            f"at {row.speed_m_per_s * 100:5.1f} cm/s: median {row.median:.2f}, "
            f"share > 1: {row.share_above_1:.2f}"
        )


if __name__ == "__main__":
    main()
