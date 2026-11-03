"""Calibrate the estimator's prediction-error covariance R.

Each sample draws off-grid true parameters and a random reachable push,
runs the truth sim, and predicts the push with the *nearest* grid
hypothesis, so R covers both the prediction model's mismatch and the
grid's discretization.  R is the errors' second moment about zero (the
likelihood assumes zero-mean errors) plus a diagonal floor.

R lives in the code as estimator.CALIBRATED_R.  Rerun this after changing
the prediction model, the measurement, or the grid, then paste the
constant it prints into src/estimator.py.  From the repo root:

    uv run python -m scripts.calibrate_R --samples 48
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import cast

import numpy as np

from src.cli import (
    add_shared_options,
    finish_shared_options,
    push_model_config,
    truth_sim_config,
)
from src.dynamics import DrakePushModel, PushModelConfig
from src.geometry import SurfacePart
from src.iiwa import Pusher
from src.object import Object, Rod
from src.observation import planar_error
from src.params import Eta, GridSpec, make_grid, nearest_hypothesis, sample_eta
from src.recording import NoReachablePush, TruthSim, TruthSimConfig

# Floor on each error's standard deviation (x, y, yaw, vx, vy, wz), in m,
# rad, m/s, rad/s: keeps R invertible where the samples barely vary.
FLOOR_STD = np.array([1e-4, 1e-4, 1e-3, 2e-4, 2e-4, 2e-3])
LABELS = ("x", "y", "yaw", "vx", "vy", "wz")


@dataclass(frozen=True)
class SampleSetup:
    """What every calibration sample shares."""

    grid: list[Eta]
    spec: GridSpec
    parts: list[SurfacePart]
    rod_mass: float
    push_density: float
    truth: TruthSimConfig
    model: PushModelConfig


def calibration_sample(
    seed: int, setup: SampleSetup
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run one truth push at off-grid parameters, returning its prediction
    errors under the nearest grid hypothesis and under the true parameters
    (or None when no push was reachable)."""
    rng = np.random.default_rng(seed)
    eta_true = sample_eta(rng, setup.spec, setup.parts)
    obj = eta_true.to_object(rng, setup.rod_mass)
    try:
        obs = TruthSim(obj, Pusher(), setup.truth).record_random_push(
            rng, obj, obj.surface_normals(setup.push_density)
        )
    except NoReachablePush:
        return None
    nearest, true = DrakePushModel(setup.model).predict(
        [nearest_hypothesis(setup.grid, eta_true), eta_true], obs
    )
    measured = obs.measured()
    return planar_error(measured, nearest), planar_error(measured, true)


def second_moment_R(errors: np.ndarray, floor_std: np.ndarray) -> np.ndarray:
    """Get the (N, 6) `errors`' second moment about zero plus a diagonal floor."""
    return cast(np.ndarray, errors.T @ errors / len(errors) + np.diag(floor_std**2))


def rounded_R(R: np.ndarray) -> np.ndarray:
    """Round `R` to the four significant figures format_constant prints.

    Raises LinAlgError when the rounded R is not positive-definite.
    """
    # The rounded constant is what the estimator will use, so check that one.
    rounded = cast(np.ndarray, np.vectorize(lambda v: float(f"{v:.3e}"))(R))
    np.linalg.cholesky(rounded)
    return rounded


def labeled(values: np.ndarray) -> str:
    """Render one value per measurement component."""
    return "  ".join(f"{label}={v:.2g}" for label, v in zip(LABELS, values))


def format_constant(
    R: np.ndarray,
    samples: int,
    seed: int,
    truth: TruthSimConfig,
    model: PushModelConfig,
) -> str:
    """Render R as the CALIBRATED_R source block for src/estimator.py.  Four
    significant figures is well inside the estimate's sampling error."""
    rows = "\n".join(
        "        [" + ", ".join(f"{v:.3e}" for v in row) + "]," for row in R
    )
    return (
        f"# Calibrated by scripts/calibrate_R.py: {samples} truth pushes, seed "
        f"{seed},\n"
        f"# truth timestep {truth.time_step:g}, model timestep "
        f"{model.time_step:g}, settle time {truth.settle_time:g} s.\n"
        f"CALIBRATED_R = np.array(\n    [\n{rows}\n    ]\n)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_shared_options(parser)
    parser.add_argument("--samples", type=int, default=48, help="truth pushes")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    args = parser.parse_args()
    finish_shared_options(args)
    return args


def main(args: argparse.Namespace) -> None:
    spec = GridSpec()
    # Only the footprint is needed, and it's the same for every eta.
    parts = Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, args.rod_mass)).parts
    truth = truth_sim_config(args)
    model = push_model_config(args)
    setup = SampleSetup(
        grid=make_grid(spec, parts),
        spec=spec,
        parts=parts,
        rod_mass=args.rod_mass,
        push_density=args.push_density,
        truth=truth,
        model=model,
    )
    seeds = np.random.default_rng(args.seed).integers(2**32, size=args.samples)
    started = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(calibration_sample, int(s), setup) for s in seeds]
        results = [r for r in (f.result() for f in futures) if r is not None]
    if not results:
        raise SystemExit("no sample found a reachable push")
    nearest = np.array([e for e, _ in results])
    true = np.array([e for _, e in results])
    print(
        f"{len(results)}/{args.samples} samples in {time.time() - started:.0f} s "
        f"({len(setup.grid)} hypotheses)"
    )
    for name, errors in (("nearest grid", nearest), ("true eta", true)):
        rms = np.sqrt((errors**2).mean(axis=0))
        print(f"rms error, {name:>12}: {labeled(rms)}")

    R = second_moment_R(nearest, FLOOR_STD)
    print(f"R std devs: {labeled(np.sqrt(np.diag(R)))}")
    print(
        "\nPaste into src/estimator.py, replacing CALIBRATED_R:\n\n"
        + format_constant(rounded_R(R), len(results), args.seed, truth, model)
    )


if __name__ == "__main__":
    main(parse_args())
