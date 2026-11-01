"""Calibrate the estimator's prediction-error covariance R.

Each sample draws off-grid true parameters and a random reachable push,
runs the truth sim, and predicts the push with the *nearest* grid
hypothesis, so R covers both the prediction model's mismatch and the
grid's discretization.  R is the errors' second moment about zero (the
likelihood assumes zero-mean errors) plus a diagonal floor.

Run from the repo root:

    uv run python -m scripts.calibrate_R --samples 48
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src import ASSET_DIR
from src.dynamics import DrakePushModel
from src.estimator import DEFAULT_R_STD
from src.iiwa import Pusher
from src.object import Object, Rod
from src.observation import planar_error, record_random_push
from src.params import Eta, GridSpec, make_grid, nearest_hypothesis, sample_eta

# Floor on each error's standard deviation, as a fraction of the defaults:
# keeps R invertible where the samples barely vary (e.g. at-rest velocity).
FLOOR_FRACTION = 0.1
LABELS = ("x", "y", "yaw", "vx", "vy", "wz")

# Each worker process's own prediction model, built once by _init_worker.
_model: DrakePushModel | None = None


def _init_worker(model_time_step: float) -> None:
    """Build this worker process's prediction model."""
    global _model
    _model = DrakePushModel(time_step=model_time_step)


def _parts() -> list[tuple[float, float, float, float]]:
    """Get the object's footprint parts (the same for every eta)."""
    return Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, 0.2)).parts


def calibration_sample(
    seed: int,
    grid: list[Eta],
    spec: GridSpec,
    time_step: float,
    push_density: float,
    settle_time: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run one truth push at off-grid parameters, returning its prediction
    errors under the nearest grid hypothesis and under the true parameters
    (or None when no push was reachable)."""
    assert _model is not None, "worker was not initialized"
    rng = np.random.default_rng(seed)
    eta_true = sample_eta(rng, spec, _parts())
    obj = eta_true.to_object(rng)
    try:
        obs = record_random_push(
            rng,
            obj,
            Pusher(),
            obj.surface_normals(push_density),
            time_step,
            settle_time,
        )
    except RuntimeError:
        return None
    nearest, true = _model.predict([nearest_hypothesis(grid, eta_true), eta_true], obs)
    measured = obs.measured()
    return planar_error(measured, nearest), planar_error(measured, true)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--samples", type=int, default=48, help="truth pushes")
    parser.add_argument("--workers", type=int, default=None, help="processes")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    parser.add_argument(
        "--time-step", type=float, default=1e-4, help="truth sim timestep"
    )
    parser.add_argument(
        "--model-time-step",
        type=float,
        default=2e-3,
        help="prediction model timestep",
    )
    parser.add_argument(
        "--push-density",
        type=float,
        default=2.0,
        help="push-contact density: surface normals per cm of object outline",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.1,
        help="measurement delay after each push segment ends (s)",
    )
    parser.add_argument(
        "--out", default=str(ASSET_DIR / "R.npy"), help="where to save R"
    )
    args = parser.parse_args()
    args.push_density *= 100  # count/cm -> count/m
    return args


def main(args: argparse.Namespace) -> None:
    spec = GridSpec()
    grid = make_grid(spec, _parts())
    seeds = np.random.default_rng(args.seed).integers(2**32, size=args.samples)
    started = time.time()
    with ProcessPoolExecutor(
        args.workers, initializer=_init_worker, initargs=(args.model_time_step,)
    ) as pool:
        futures = [
            pool.submit(
                calibration_sample,
                int(seed),
                grid,
                spec,
                args.time_step,
                args.push_density,
                args.settle_time,
            )
            for seed in seeds
        ]
        results = [r for r in (f.result() for f in futures) if r is not None]
    if not results:
        raise SystemExit("no sample found a reachable push")
    nearest = np.array([e for e, _ in results])
    true = np.array([e for _, e in results])
    print(
        f"{len(results)}/{args.samples} samples in {time.time() - started:.0f} s "
        f"({len(grid)} hypotheses)"
    )
    for name, errors in (("nearest grid", nearest), ("true eta", true)):
        rms = np.sqrt((errors**2).mean(axis=0))
        print(
            f"rms error, {name:>12}: "
            + "  ".join(f"{label}={v:.2g}" for label, v in zip(LABELS, rms))
        )

    floor = np.diag((FLOOR_FRACTION * DEFAULT_R_STD) ** 2)
    R = nearest.T @ nearest / len(nearest) + floor
    np.linalg.cholesky(R)  # positive-definite, or this raises
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, R)
    print(
        f"saved R to {out}; std devs: "
        + "  ".join(f"{label}={v:.2g}" for label, v in zip(LABELS, np.sqrt(np.diag(R))))
    )


if __name__ == "__main__":
    main(parse_args())
