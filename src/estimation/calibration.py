"""Calibrate the estimator's prediction-error covariance R.

Each sample is one episode's first push: off-grid true parameters and a start
pose from episode_conditions, a random push from the push space, run in the
truth sim.  The push is predicted at the belief point nearest the true
parameters (`n_subcells` points per axis in each grid cell, its error
interpolated from the grid's; match estimator.SUBCELLS), so R covers the
prediction model's mismatch, the interpolation, and the remaining discretization.  R
is the errors' second moment about zero (the likelihood assumes zero-mean
errors) plus a diagonal floor.

R lives in the code as estimator.CALIBRATED_R.  Rerun this after changing
the prediction model, the measurement, the grid, or the push space, then
paste the constant it prints into src/estimation/estimator.py:

    uv run python main.py calibrate --samples 192 --seed 20000
"""

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from src.control.probing import ProbingState, random_selector, run_probing
from src.estimation.dynamics import DrakePushModel, DynamicsModel, PushModelConfig
from src.estimation.params import (
    Eta,
    GridSpec,
    Subcells,
    make_grid,
    nearest_subcell,
    subcells,
)
from src.experiment import episode_conditions, episode_object, reference_object
from src.surrogate.model import SurrogatePredictor, load_ensemble
from src.world.iiwa import Pusher
from src.world.observation import planar_error
from src.world.push_space import PushSpace, make_push_space
from src.world.recording import TruthSim, TruthSimConfig

# Floor on each error's standard deviation (x, y, yaw, vx, vy, wz), in m,
# rad, m/s, rad/s: keeps R invertible where the samples barely vary.
FLOOR_STD = np.array([1e-4, 1e-4, 1e-3, 2e-4, 2e-4, 2e-3])
LABELS = ("x", "y", "yaw", "vx", "vy", "wz")


PREDICTORS = ("drake", "surrogate")


@dataclass(frozen=True)
class SampleSetup:
    """What every calibration sample shares: the push space the push is drawn
    from and the hypothesis grid it is predicted on, built once."""

    push_space: PushSpace
    grid: list[Eta]
    cells: Subcells
    n_subcells: int
    rod_mass: float
    truth: TruthSimConfig
    model: PushModelConfig
    predictor: str = "drake"
    ensemble_path: Path | None = None

    def build_predictor(self) -> DynamicsModel:
        """Build the sample's predictor: the Drake plant, or the trained
        surrogate whose errors R should cover."""
        if self.predictor == "surrogate":
            assert self.ensemble_path is not None
            return SurrogatePredictor(load_ensemble(self.ensemble_path))
        return DrakePushModel(self.model)


def make_sample_setup(
    rod_mass: float,
    truth: TruthSimConfig,
    model: PushModelConfig,
    predictor: str = "drake",
    ensemble_path: Path | None = None,
    n_subcells: int = 1,
) -> SampleSetup:
    """Collect what every calibration sample shares."""
    reference = reference_object(rod_mass)
    grid = make_grid(GridSpec(), reference.parts)
    return SampleSetup(
        push_space=make_push_space(reference.parts, reference.height),
        grid=grid,
        cells=subcells(grid, n_subcells),
        n_subcells=n_subcells,
        rod_mass=rod_mass,
        truth=truth,
        model=model,
        predictor=predictor,
        ensemble_path=ensemble_path,
    )


def calibration_sample(
    seed: int, setup: SampleSetup
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run one truth push at the seed's episode conditions, returning its
    prediction errors at the nearest belief point and under the true
    parameters (or None when no push was reachable)."""
    eta_true, start_pose, action_rng = episode_conditions(seed)
    obj = episode_object(eta_true, start_pose, setup.rod_mass)
    state = ProbingState(
        truth_sim=TruthSim(obj, Pusher(), setup.truth), obj=obj, belief=None
    )
    outcome = run_probing(
        state,
        random_selector(setup.push_space, action_rng),
        budget=1,
    )
    if not outcome.records:
        return None
    obs = outcome.records[0].observation
    predictor = setup.build_predictor()
    weights = setup.cells.weights[nearest_subcell(setup.cells, setup.grid, eta_true)]
    (corners,) = np.nonzero(weights)
    predicted = predictor.predict([*(setup.grid[i] for i in corners), eta_true], obs)
    measured = obs.measured()
    errors = planar_error(measured, predicted)
    return weights[corners] @ errors[:-1], errors[-1]


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
    n_subcells: int,
) -> str:
    """Render R as the CALIBRATED_R source block for src/estimator.py.  Four
    significant figures is well inside the estimate's sampling error."""
    rows = "\n".join(
        "        [" + ", ".join(f"{v:.3e}" for v in row) + "]," for row in R
    )
    return (
        f"# Calibrated by main.py calibrate: {samples} truth pushes, seed {seed},\n"
        f"# truth timestep {truth.time_step:g}, model timestep "
        f"{model.time_step:g}, settle time {truth.settle_time:g} s,\n"
        f"# {n_subcells}^3 sub-cells.\n"
        f"CALIBRATED_R = np.array(\n    [\n{rows}\n    ]\n)"
    )


def calibrate(samples: int, seed: int, workers: int, setup: SampleSetup) -> None:
    """Run `samples` calibration pushes on `workers` processes, printing the
    CALIBRATED_R block they give.

    Raises SystemExit when no sample found a reachable push.
    """
    seeds = np.random.default_rng(seed).integers(2**32, size=samples)
    started = time.time()
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(calibration_sample, int(s), setup) for s in seeds]
        results = [r for r in (f.result() for f in futures) if r is not None]
    if not results:
        raise SystemExit("no sample found a reachable push")
    nearest = np.array([e for e, _ in results])
    true = np.array([e for _, e in results])
    print(
        f"{len(results)}/{samples} samples in {time.time() - started:.0f} s "
        f"({len(setup.grid)} hypotheses, {setup.n_subcells}^3 sub-cells)"
    )
    for name, errors in (("nearest point", nearest), ("true eta", true)):
        rms = np.sqrt((errors**2).mean(axis=0))
        print(f"rms error, {name:>12}: {labeled(rms)}")

    R = second_moment_R(nearest, FLOOR_STD)
    print(f"R std devs: {labeled(np.sqrt(np.diag(R)))}")
    print(
        "\nPaste into src/estimator.py, replacing CALIBRATED_R:\n\n"
        + format_constant(
            rounded_R(R),
            len(results),
            seed,
            setup.truth,
            setup.model,
            setup.n_subcells,
        )
    )
