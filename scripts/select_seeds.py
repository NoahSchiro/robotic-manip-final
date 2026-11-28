"""Write the first episode seeds, counting up from --start, whose true friction
lies in [--min-mu, --max-mu], one per line, for scripts/cluster/placement.sbatch.

    uv run python -m scripts.select_seeds --max-mu 0.3 --start 96000 --count 30 --out data/results/low_mu/seeds.txt
"""

import argparse
from pathlib import Path

from src.experiment import TASKS, episode_conditions


def select_seeds(
    task: str, start: int, count: int, min_mu: float, max_mu: float
) -> list[int]:
    """The first `count` seeds from `start` whose true mu is within bounds."""
    seeds: list[int] = []
    seed = start
    while len(seeds) < count:
        eta, _, _ = episode_conditions(seed, TASKS[task])
        if min_mu <= eta.mu <= max_mu:
            seeds.append(seed)
        seed += 1
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), default="far")
    parser.add_argument("--start", type=int, default=96000)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--min-mu", type=float, default=0.0)
    parser.add_argument("--max-mu", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seeds = select_seeds(args.task, args.start, args.count, args.min_mu, args.max_mu)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(f"{s}\n" for s in seeds))
    print(f"wrote {len(seeds)} seeds ({seeds[0]}-{seeds[-1]}) to {args.out}")


if __name__ == "__main__":
    main()
