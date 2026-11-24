"""How well a push separates neighbouring hypotheses, by speed at matched push
distance.

For each contact on the MPC grid and each (distance, speed) pair, the Drake
prediction model predicts the estimator measurement along the nominal tip path
under a base hypothesis and under that hypothesis moved one grid step in one
parameter (friction, rod x, or rod y).  The separation is the Mahalanobis
distance between the two predictions under CALIBRATED_R: about 1 or more means
one push noticeably shifts the belief between them.  Planning-only; no truth
simulation.

    uv run python -m scripts.push_informativeness --out data/results/push_informativeness.json
    # Plot saved results without re-simulating
    uv run python -m scripts.push_informativeness --from-json data/results/push_informativeness.json --plot docs/figures
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pydrake.all import RigidTransform, RollPitchYaw, SpatialVelocity

from src import paths
from src.constants import LATTICE_SPACING, SETTLE_TIME
from src.estimation.dynamics import DrakePushModel
from src.estimation.estimator import CALIBRATED_R
from src.estimation.params import Eta
from src.experiment import push_model_config, reference_object
from src.surrogate.model import nominal_tip_path
from src.world.action import Action
from src.world.geometry import SurfaceNormal
from src.world.iiwa import transfer_height
from src.world.observation import ObjectState, measurement, planar_error
from src.world.push_space import make_push_space

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
    _model = DrakePushModel(push_model_config())
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
    reference = reference_object()
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


@dataclass(frozen=True)
class ContactSeparation:
    """The separation of one push: its body-frame contact point and inward
    normal (x, y), push distance (m) and speed (m/s), for one base hypothesis
    and parameter step."""

    base_mu: float
    parameter: str
    distance_m: float
    speed_m_per_s: float
    contact_xy: tuple[float, float]
    normal_xy: tuple[float, float]
    separation: float


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


def run(workers: int) -> tuple[list[Separation], list[ContactSeparation]]:
    """Summaries per (base, parameter, distance, speed), and every push's
    separation."""
    contacts = contacts_on_grid(LATTICE_SPACING)
    etas = [tuple(base.as_array()) for base in BASES]
    etas += [
        tuple(np.add(base.as_array(), step))
        for base in BASES
        for step in GRID_STEPS.values()
    ]
    reference = reference_object()
    space = make_push_space(reference.parts, reference.height)
    combos = feasible_combos(space.duration_range)
    tasks = [
        (Action(contact=c, velocity=v, time=d / v), eta)
        for c in contacts
        for d, v in combos
        for eta in etas
    ]
    height = transfer_height(reference_object())
    with Pool(workers, initializer=_start_worker, initargs=(height,)) as pool:
        predictions = np.array(pool.map(_predict, tasks, chunksize=8))
    predictions = predictions.reshape(len(contacts), len(combos), len(etas), 6)
    rows: list[Separation] = []
    per_contact: list[ContactSeparation] = []
    for b, base in enumerate(BASES):
        for k, parameter in enumerate(GRID_STEPS):
            moved = len(BASES) + b * len(GRID_STEPS) + k
            column = separation(
                predictions[:, :, b], predictions[:, :, moved], CALIBRATED_R
            )
            rows += summarize(column, combos, base.mu, parameter)
            per_contact += [
                ContactSeparation(
                    base_mu=base.mu,
                    parameter=parameter,
                    distance_m=d,
                    speed_m_per_s=v,
                    contact_xy=(float(c.position[0]), float(c.position[1])),
                    normal_xy=(float(c.normal[0]), float(c.normal[1])),
                    separation=float(column[i, j]),
                )
                for i, c in enumerate(contacts)
                for j, (d, v) in enumerate(combos)
            ]
    return rows, per_contact


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
    ax.set_title(title, fontsize=11, loc="left", color="#0b0b0b")
    ax.set_xlabel("push speed (cm/s)")
    ax.set_ylabel("separation s (units of R)")
    ax.set_xticks(np.array(SPEEDS) * 100)
    ax.set_xlim(1.5, 16.5)
    ax.set_ylim(0, top)
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color="#e1e0d9", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [
        Line2D([], [], color=c, lw=2, marker="o", label=f"{d * 100:g} cm push")
        for d, c in zip(distances, DISTANCE_COLORS, strict=False)
    ] + [
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
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150)


MAP_PUSH = (0.09, 0.15)  # m, m/s: the push shown on the contact maps
MAP_BASE_MU = 0.2
MAP_TITLES = {"mu": "Friction", "rod_x": "Rod x", "rod_y": "Rod y"}


def plot_contact_maps(contacts: Sequence[ContactSeparation], out: Path) -> None:
    """Separation of one push type (MAP_PUSH at true friction MAP_BASE_MU) at
    every contact point, one panel per parameter, on the block outline."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Polygon

    from scripts.plot_belief import outline_vertices

    outline = outline_vertices(np.array(reference_object().parts)) * 100
    figure = Figure(figsize=(12, 5.0), facecolor="#fcfcfb")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, len(MAP_TITLES))
    for ax, parameter in zip(axes, MAP_TITLES, strict=True):
        chosen = [
            c
            for c in contacts
            if c.parameter == parameter
            and c.base_mu == MAP_BASE_MU
            and np.isclose(c.distance_m, MAP_PUSH[0])
            and np.isclose(c.speed_m_per_s, MAP_PUSH[1])
        ]
        xy = np.array([c.contact_xy for c in chosen]) * 100
        normals = np.array([c.normal_xy for c in chosen])
        values = np.array([c.separation for c in chosen])
        ax.add_patch(
            Polygon(
                outline, closed=True, facecolor="#f0efec", edgecolor="#52514e", lw=1
            )
        )
        # Arrows start outside the block and point along the push.
        ax.quiver(
            xy[:, 0] - 1.8 * normals[:, 0],
            xy[:, 1] - 1.8 * normals[:, 1],
            normals[:, 0],
            normals[:, 1],
            color="#b8b6ad",
            angles="xy",
            scale_units="xy",
            scale=1 / 1.3,
            width=0.006,
        )
        scatter = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=values,
            cmap="Blues",
            vmin=0,
            vmax=values.max(),
            s=36,
            edgecolors="#52514e",
            linewidths=0.5,
            zorder=3,
        )
        figure.colorbar(
            scatter,
            ax=ax,
            orientation="horizontal",
            shrink=0.8,
            pad=0.16,
            label="separation s (units of R)",
        )
        ax.set_title(MAP_TITLES[parameter], fontsize=11, loc="left")
        ax.set_aspect("equal")
        ax.set_xlim(-10.5, 10.5)
        ax.set_ylim(-10.5, 10.5)
        ax.set_xlabel("body x (cm)")
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("body y (cm)")
    figure.suptitle(
        f"Separation of neighbouring hypotheses by contact point: {MAP_PUSH[0] * 100:g} cm "
        f"push at {MAP_PUSH[1] * 100:g} cm/s, true friction {MAP_BASE_MU:g}",
        fontsize=11,
    )
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150)


def json_rows(tables: dict[str, list[dict[str, object]]]) -> str:
    """JSON with one row per line, so result diffs stay readable."""
    blocks = [
        f'"{name}": [\n' + ",\n".join(json.dumps(row) for row in rows) + "\n]"
        for name, rows in tables.items()
    ]
    return "{\n" + ",\n".join(blocks) + "\n}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--out", type=Path, default=paths.RESULTS / "push_informativeness.json"
    )
    parser.add_argument(
        "--from-json", type=Path, help="reuse saved results instead of simulating"
    )
    parser.add_argument(
        "--plot", type=Path, help="also save one figure per parameter in this folder"
    )
    args = parser.parse_args()
    if args.from_json is not None:
        saved = json.loads(args.from_json.read_text())
        rows = [Separation(**row) for row in saved["summary"]]
        contacts = [
            ContactSeparation(
                **{
                    **c,
                    "contact_xy": tuple(c["contact_xy"]),
                    "normal_xy": tuple(c["normal_xy"]),
                }
            )
            for c in saved["contacts"]
        ]
    else:
        rows, contacts = run(args.workers)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json_rows(
                {
                    "summary": [asdict(row) for row in rows],
                    "contacts": [asdict(c) for c in contacts],
                }
            )
        )
    if args.plot is not None:
        for path in plot(rows, args.plot):
            print(f"wrote {path}")
        contact_map = args.plot / "push_informativeness_contacts.png"
        plot_contact_maps(contacts, contact_map)
        print(f"wrote {contact_map}")
    for row in rows:
        print(
            f"mu {row.base_mu:.1f} {row.parameter:6} {row.distance_m * 100:4.1f} cm "
            f"at {row.speed_m_per_s * 100:5.1f} cm/s: median {row.median:.2f}, "
            f"share > 1: {row.share_above_1:.2f}"
        )


if __name__ == "__main__":
    main()
