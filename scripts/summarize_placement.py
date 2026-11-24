"""Score placement episode records (main.py placement-eval) after the fact.

Each record holds the distance and yaw to the goal at the start and after
every push, so success is scored here at several tolerances and push budgets
without rerunning.  An episode that stopped early keeps its last pose for the
remaining budget.  Conditions are compared on the seeds they share; the
paired comparison of "none" against "true_eta" is the most probing could gain.

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


def condition_stats(records: list[Record]) -> dict[str, Any]:
    stops: dict[str, int] = defaultdict(int)
    for r in records:
        stops[r["stop_reason"]] += 1
    reached = {
        f"{p * 100:g}cm_{y:g}deg": np.array([pushes_to_reach(r, p, y) for r in records])
        for p, y in TOLERANCES
    }
    return {
        "episodes": len(records),
        "stop_reasons": dict(stops),
        "success_by_budget": {
            tol: {str(b): float(np.mean(k <= b)) for b in BUDGETS}
            for tol, k in reached.items()
        },
        "median_pushes_to_reach": {
            tol: float(np.median(k)) for tol, k in reached.items()
        },
        "final_cost_median": float(np.median([final_cost(r) for r in records])),
        "final_position_error_cm_median": float(
            np.median([r["position_error"] * 100 for r in records])
        ),
        "start_distance_cm_mean": float(
            np.mean([r["goal_errors"][0][0] * 100 for r in records])
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
        tol = f"{p * 100:g}cm_{y:g}deg"
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


def summarize(records: list[Record]) -> dict[str, Any]:
    """Per task: each condition's statistics, and none against true_eta."""
    tasks: dict[str, dict[str, dict[int, Record]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in records:
        tasks[r.get("task", "near")][r["condition"]][r["seed"]] = r
    summary: dict[str, Any] = {}
    for task, by_condition in sorted(tasks.items()):
        summary[task] = {
            "conditions": {
                name: condition_stats(list(by_seed.values()))
                for name, by_seed in sorted(by_condition.items())
            }
        }
        if "true_eta" in by_condition and "none" in by_condition:
            summary[task]["none_vs_true_eta"] = paired(by_condition, "true_eta", "none")
    return summary


def plot(summary: dict[str, Any], out: Path) -> None:
    """Success rate against push budget, one panel per task, one line per
    condition, solid at the episode tolerance and dashed at the loosest."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    colors = {"true_eta": "#0b0b0b", "none": "#2a78d6"}
    tight = f"{TOLERANCES[0][0] * 100:g}cm_{TOLERANCES[0][1]:g}deg"
    loose = f"{TOLERANCES[-1][0] * 100:g}cm_{TOLERANCES[-1][1]:g}deg"
    figure = Figure(figsize=(5 * len(summary), 3.8), facecolor="#fcfcfb")
    FigureCanvasAgg(figure)
    axes = np.atleast_1d(figure.subplots(1, len(summary), sharey=True))
    for ax, (task, task_summary) in zip(axes, summary.items(), strict=True):
        for name, st in task_summary["conditions"].items():
            color = colors.get(name, "#eb6834")
            for tol, style in ((tight, "-"), (loose, "--")):
                rates = [st["success_by_budget"][tol][str(b)] * 100 for b in BUDGETS]
                label = f"{name} ({tol.replace('_', ', ')})"
                ax.plot(
                    BUDGETS, rates, style, marker="o", ms=4, color=color, label=label
                )
        ax.set_title(f"{task} task ({st['episodes']} seeds)", fontsize=10, loc="left")
        ax.set_xlabel("push budget")
        ax.set_xticks(BUDGETS)
        ax.set_ylim(0, 102)
        ax.grid(axis="y", color="#e1e0d9")
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("episodes at the goal (%)")
    axes[-1].legend(frameon=False, fontsize=7.5, loc="lower right")
    figure.tight_layout()
    figure.savefig(out, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", type=Path, nargs="+", help="record folders")
    parser.add_argument(
        "--out", type=Path, help="folder for summary.json and success.png"
    )
    args = parser.parse_args()
    summary = summarize(load(args.directories))
    for task, task_summary in summary.items():
        print(f"== {task}")
        for name, st in task_summary["conditions"].items():
            rates = st["success_by_budget"]
            print(
                f"  {name:10} n={st['episodes']:3} start {st['start_distance_cm_mean']:5.1f} cm  "
                + "  ".join(
                    f"{tol}: {rates[tol]['5']:.0%}@5 {rates[tol]['20']:.0%}@20"
                    for tol in rates
                )
                + f"  stops {st['stop_reasons']}"
            )
        if "none_vs_true_eta" in task_summary:
            for tol, d in task_summary["none_vs_true_eta"].items():
                if tol != "seeds":
                    print(
                        f"  none vs true_eta, {tol}: {d['mean_extra_pushes']:+.2f} pushes "
                        f"± {d['se'] or float('nan'):.2f}, "
                        f"Wilcoxon p={d['wilcoxon_p'] or float('nan'):.3f}"
                    )
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        plot(summary, args.out / "success.png")
        print(f"wrote {args.out / 'summary.json'} and {args.out / 'success.png'}")


if __name__ == "__main__":
    main()
