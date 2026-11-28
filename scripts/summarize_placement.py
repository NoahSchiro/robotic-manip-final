"""Score placement episode records (main.py placement-eval) after the fact.

Each record holds the distance and yaw to the goal at the start and after every
push, so success is scored at several tolerances and push budgets without
rerunning.  An episode that stopped early keeps its last pose for the remaining
budget.  Conditions are compared on shared seeds.  "true_eta" (the MPC plans
with the true eta) and "none" (the MPC estimates eta from its own pushes) do no
probing, so their paired gap bounds what probing could gain.

    uv run python -m scripts.summarize_placement data/results/headroom/near data/results/headroom/far --out data/results/headroom
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from src.constants import MPC_LENGTH, MPC_YAW_WEIGHT

CONDITION_LABELS = {
    "true_eta": "MPC, η known",
    "none": "MPC, η unknown (estimated)",
    "nominal_eta": "MPC, η fixed at nominal",
    "farthest_eta": "MPC, η fixed at farthest",
    "learned:3": "3 learned probes, then MPC",
    "random:3": "3 random probes, then MPC",
    "long_fast:3": "3 long-fast probes, then MPC",
}

# (position m, yaw deg): the episode tolerance and two looser ones.
TOLERANCES = ((0.005, 5.0), (0.01, 10.0), (0.02, 20.0))
BUDGETS = (2, 3, 5, 10, 20)

Record = dict[str, Any]


def load(directories: list[Path]) -> list[Record]:
    return [
        json.loads(path.read_text())
        for directory in directories
        for path in sorted(directory.glob("*.json"))
    ]


def tolerance_key(position: float, yaw_deg: float) -> str:
    return f"{position * 100:g}cm_{yaw_deg:g}deg"


def pushes_to_reach(record: Record, position: float, yaw_deg: float) -> float:
    """The first push count at which the block is within tolerance (0 if it
    starts there), or inf if it never is."""
    for k, (distance, yaw) in enumerate(record["goal_errors"]):
        if distance <= position and np.rad2deg(yaw) <= yaw_deg:
            return float(k)
    return float("inf")


def final_cost(record: Record) -> float:
    """The MPC pose cost where the episode ended."""
    distance, yaw = record["goal_errors"][-1]
    return float((distance / MPC_LENGTH) ** 2 + MPC_YAW_WEIGHT * yaw**2)


def spread(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def condition_stats(records: list[Record]) -> dict[str, Any]:
    stops: dict[str, int] = defaultdict(int)
    for r in records:
        stops[r["stop_reason"]] += 1
    reached = {
        tolerance_key(p, y): np.array([pushes_to_reach(r, p, y) for r in records])
        for p, y in TOLERANCES
    }
    return {
        "episodes": len(records),
        "stop_reasons": dict(stops),
        "success_by_budget": {
            tol: {str(b): float(np.mean(k <= b)) for b in BUDGETS}
            for tol, k in reached.items()
        },
        # Per episode, in seed order; never reaching it is null.
        "pushes_to_reach": {
            tol: [None if np.isinf(v) else int(v) for v in k]
            for tol, k in reached.items()
        },
        "median_pushes_to_reach": {
            tol: float(np.median(k)) for tol, k in reached.items()
        },
        "final_cost_median": float(np.median([final_cost(r) for r in records])),
        "final_position_error_cm_median": float(
            np.median([r["position_error"] * 100 for r in records])
        ),
        # How far the goal starts: distance (cm) and rotation (deg).
        "start_distance_cm": spread([r["goal_errors"][0][0] * 100 for r in records]),
        "start_rotation_deg": spread(
            [float(np.rad2deg(r["goal_errors"][0][1])) for r in records]
        ),
        "wall_time_min_mean": float(np.mean([r["wall_time"] / 60 for r in records])),
    }


def paired(
    by_seed: dict[str, dict[int, Record]], better: str, worse: str
) -> dict[str, Any]:
    """`worse` minus `better` in pushes to reach each tolerance, over shared
    seeds; episodes that never reach it count as max budget + 1 pushes."""
    seeds = sorted(set(by_seed[better]) & set(by_seed[worse]))
    result: dict[str, Any] = {"seeds": len(seeds)}
    cap = max(BUDGETS) + 1
    for p, y in TOLERANCES:
        tol = tolerance_key(p, y)
        a = np.array(
            [min(pushes_to_reach(by_seed[worse][s], p, y), cap) for s in seeds]
        )
        b = np.array(
            [min(pushes_to_reach(by_seed[better][s], p, y), cap) for s in seeds]
        )
        diff = a - b
        nonzero = diff[diff != 0]
        result[tol] = {
            "mean_extra_pushes": float(diff.mean()),
            "se": float(diff.std(ddof=1) / np.sqrt(len(diff)))
            if len(diff) > 1
            else None,
            "worse_share": float(np.mean(diff > 0)),
            "better_share": float(np.mean(diff < 0)),
            "wilcoxon_p": float(stats.wilcoxon(nonzero).pvalue)
            if len(nonzero) > 0
            else None,
        }
    return result


def by_task(records: list[Record]) -> dict[str, dict[str, dict[int, Record]]]:
    """Records indexed by task, condition and seed."""
    tasks: dict[str, dict[str, dict[int, Record]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in records:
        tasks[r.get("task", "near")][r["condition"]][r["seed"]] = r
    return tasks


def summarize(records: list[Record]) -> dict[str, Any]:
    """Per task: each condition's statistics, and none against true_eta."""
    tasks = by_task(records)
    summary: dict[str, Any] = {}
    for task, by_condition in sorted(tasks.items()):
        summary[task] = {
            "conditions": {
                name: condition_stats([by_seed[k] for k in sorted(by_seed)])
                for name, by_seed in sorted(by_condition.items())
            }
        }
        if "true_eta" in by_condition:
            summary[task]["vs_true_eta"] = {
                name: paired(by_condition, "true_eta", name)
                for name in sorted(by_condition)
                if name != "true_eta"
            }
    return summary


def plot(summary: dict[str, Any], out: Path) -> None:
    """Success rate against push budget: one row per task, one column per
    tolerance, one line per condition."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    # (color, marker, size, hollow) per condition; hollow rings stay visible
    # where markers coincide.
    styles = {
        "true_eta": ("#0b0b0b", "o", 3.5, False),
        "none": ("#2a78d6", "o", 7.0, True),
        "nominal_eta": ("#eb6834", "s", 4.0, False),
        "farthest_eta": ("#b3261e", "^", 4.5, False),
        "learned:3": ("#1f8a4c", "D", 4.0, False),
        "random:3": ("#8a8880", "v", 4.5, False),
        "long_fast:3": ("#7a4fb5", "P", 5.0, False),
    }
    budgets = np.arange(max(BUDGETS) + 1)
    figure = Figure(
        figsize=(3.6 * len(TOLERANCES), 2.9 * len(summary)), facecolor="#fcfcfb"
    )
    FigureCanvasAgg(figure)
    axes = np.atleast_2d(
        figure.subplots(len(summary), len(TOLERANCES), sharex=True, sharey=True)
    )
    for row, (task, task_summary) in zip(axes, summary.items(), strict=True):
        for ax, (position, yaw) in zip(row, TOLERANCES, strict=True):
            tol = tolerance_key(position, yaw)
            # One dot per whole-push budget.
            for name, st in task_summary["conditions"].items():
                reached = np.array(
                    [np.inf if k is None else k for k in st["pushes_to_reach"][tol]]
                )
                rates = [np.mean(reached <= b) * 100 for b in budgets]
                color, marker, size, hollow = styles.get(
                    name, ("#8a8880", "D", 4.0, False)
                )
                ax.plot(
                    budgets,
                    rates,
                    ls="",
                    marker=marker,
                    ms=size,
                    mfc="none" if hollow else color,
                    mec=color,
                    mew=1.3,
                    color=color,
                    label=CONDITION_LABELS.get(name, name),
                )
            ax.set_title(
                f"{task} task, tolerance {position * 100:g} cm / {yaw:g}°",
                fontsize=9.5,
                loc="left",
            )
            ax.set_ylim(0, 102)
            ax.set_xlim(-0.5, max(BUDGETS) + 0.5)
            ax.grid(color="#e1e0d9", lw=0.7)
            ax.set_facecolor("#fcfcfb")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        row[0].set_ylabel("episodes at the goal (%)")
    for ax in axes[-1]:
        ax.set_xlabel("push budget")
    axes[-1, 0].legend(frameon=False, fontsize=8, loc="lower right")
    counts = sorted(
        {
            st["episodes"]
            for task_summary in summary.values()
            for st in task_summary["conditions"].values()
        }
    )
    seeds = (
        f"{counts[0]} paired seeds"
        if len(counts) == 1
        else f"{counts[0]}–{counts[-1]} seeds per condition"
    )
    figure.suptitle(
        f"Episodes at the goal by push budget ({seeds} per task)", fontsize=11
    )
    figure.tight_layout()
    figure.savefig(out, dpi=150)


PAIRED_RANGE = (-3, 6)  # pushes saved shown; the top column also holds larger gaps
# The conditions compared with eta known, and how each row names its eta.
PAIRED_SOURCES = {
    "none": "η estimated",
    "nominal_eta": "η fixed at nominal",
    "farthest_eta": "η fixed at farthest",
}


def plot_paired(tasks: dict[str, dict[str, dict[int, Record]]], out: Path) -> None:
    """Per seed, the pushes saved by knowing eta: the pushes the MPC needs to
    reach each tolerance with another eta source (PAIRED_SOURCES) minus with
    eta known ("true_eta"), as stacked dots.  One row per task and source, one
    column per tolerance; seeds where only one condition, or neither, reaches
    the tolerance get their own columns."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    low, high = PAIRED_RANGE
    only_other, only_known, neither = low - 2, high + 2, high + 4
    colors = {only_other: "#2a78d6", only_known: "#0b0b0b", neither: "#b8b6ad"}
    rows = [
        (task, source, conditions)
        for task, conditions in sorted(tasks.items())
        if "true_eta" in conditions
        for source in PAIRED_SOURCES
        if source in conditions
    ]
    figure = Figure(figsize=(4 * len(TOLERANCES), 2.6 * len(rows)), facecolor="#fcfcfb")
    FigureCanvasAgg(figure)
    axes = np.atleast_2d(
        figure.subplots(len(rows), len(TOLERANCES), sharex=True, sharey=True)
    )
    for row, (task, source, conditions) in zip(axes, rows, strict=True):
        other, known = conditions[source], conditions["true_eta"]
        seeds = sorted(set(other) & set(known))
        for ax, (position, yaw) in zip(row, TOLERANCES, strict=True):
            stacked: dict[int, int] = defaultdict(int)
            for seed in seeds:
                a = pushes_to_reach(other[seed], position, yaw)
                b = pushes_to_reach(known[seed], position, yaw)
                if np.isinf(a) and np.isinf(b):
                    x = neither
                elif np.isinf(a):
                    x = only_known
                elif np.isinf(b):
                    x = only_other
                else:
                    x = int(np.clip(a - b, low, high))
                stacked[x] += 1
                color = colors.get(
                    x, "#0b0b0b" if x > 0 else "#2a78d6" if x < 0 else "#8a8880"
                )
                ax.plot(x, stacked[x], "o", ms=5, color=color)
            ax.axvline(0, color="#e1e0d9", lw=1, zorder=0)
            ax.set_title(
                f"{task} task, {PAIRED_SOURCES[source]}, "
                f"tolerance {position * 100:g} cm / {yaw:g}°",
                fontsize=8.5,
                loc="left",
            )
            ax.set_facecolor("#fcfcfb")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        row[0].set_ylabel(f"seeds (of {len(seeds)})")
    ticks = [only_other, *range(low, high + 1), only_known, neither]
    names = [
        "only\nother η\nreaches",
        *[f"{v:+d}" if v else "0" for v in range(low, high)],
        f"≥{high:+d}",
        "only\nknown η\nreaches",
        "neither",
    ]
    for ax in axes[-1]:
        ax.set_xticks(ticks, names, fontsize=7)
        ax.set_xlim(only_other - 1, neither + 1)
        ax.set_xlabel("pushes saved by knowing η (other − known)", fontsize=8.5)
    figure.tight_layout()
    figure.savefig(out, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", type=Path, nargs="+", help="record folders")
    parser.add_argument(
        "--out",
        type=Path,
        help="folder for summary.json, success.png and paired.png",
    )
    args = parser.parse_args()
    records = load(args.directories)
    summary = summarize(records)
    for task, task_summary in summary.items():
        first = next(iter(task_summary["conditions"].values()))
        d, a = first["start_distance_cm"], first["start_rotation_deg"]
        print(
            f"== {task}: goal starts {d['mean']:.1f} cm away (median {d['median']:.1f}, "
            f"{d['min']:.1f}-{d['max']:.1f}), rotation median {a['median']:.0f} deg "
            f"({a['min']:.0f}-{a['max']:.0f})"
        )
        for name, st in task_summary["conditions"].items():
            rates = st["success_by_budget"]
            print(
                f"  {name:10} n={st['episodes']:3}  "
                + "  ".join(
                    f"{tol}: {rates[tol]['5']:.0%}@5 {rates[tol]['20']:.0%}@20"
                    for tol in rates
                )
                + f"  stops {st['stop_reasons']}"
            )
        for name, comparison in task_summary.get("vs_true_eta", {}).items():
            for tol, d in comparison.items():
                if tol != "seeds":
                    print(
                        f"  {name} vs true_eta, {tol}: "
                        f"{d['mean_extra_pushes']:+.2f} pushes "
                        f"± {d['se'] or float('nan'):.2f}, "
                        f"Wilcoxon p={d['wilcoxon_p'] or float('nan'):.3f}"
                    )
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        plot(summary, args.out / "success.png")
        plot_paired(by_task(records), args.out / "paired.png")
        print(f"wrote summary.json, success.png and paired.png in {args.out}")


if __name__ == "__main__":
    main()
