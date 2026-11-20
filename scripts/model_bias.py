"""How far the prediction model is from the truth sim on recorded pushes.

Every push in a recorded trials file is re-predicted with the Drake prediction
model along its measured tip path:

- at the true eta, at the model's timestep and at a fine one (by default the
  truth sim's), to separate timestep error from the rest of the model gap;
- at the nearest grid hypothesis, the error R is calibrated on;
- over a friction sweep with the rod at its true position and at the nearest
  grid position, to find the friction each trial's pushes favour.

Mismatch is the Mahalanobis distance under CALIBRATED_R between the measured
and the predicted measurement.  Writes every prediction to an .npz and prints a
summary by friction band and per trial.

    uv run python -m scripts.model_bias --out data/model_bias.npz
"""

import argparse
import time
from collections.abc import Sequence
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.dynamics import DrakePushModel, PushModelConfig
from src.estimator import (
    CALIBRATED_R,
    squared_mahalanobis,
    student_t_log_likelihood,
)
from src.object import Object, Rod
from src.observation import PushObservation, measurement, planar_error
from src.params import Eta, GridSpec, make_grid, nearest_hypothesis
from src.trials import TrialData, load_dataset

MU_SWEEP = np.round(np.arange(0.05, 1.0001, 0.05), 3)
MU_BANDS = ((0.0, 0.35), (0.35, 0.6), (0.6, 1.0))
NU = 3.0  # the estimator's Student-t degrees of freedom
COMPONENTS = ("x", "y", "yaw", "vx", "vy", "wz")

# (push index, model timestep s, hypothesis as (mu, bx, by))
Task = tuple[int, float, tuple[float, float, float]]

_models: dict[float, DrakePushModel] = {}
_pushes: list[PushObservation] = []


def _start_worker(trials_path: Path, time_steps: tuple[float, ...]) -> None:
    global _pushes
    trials, metadata = load_dataset(trials_path)
    _pushes = [obs for trial in trials for obs in trial.observations]
    for time_step in time_steps:
        _models[time_step] = DrakePushModel(
            PushModelConfig(time_step=time_step, rod_mass=float(metadata["rod_mass"]))
        )


def _predict(task: Task) -> np.ndarray:
    """(12,): the (6,) measurement, then the planar pose (x, y, yaw) at the
    push segment's end and at the window's end."""
    index, time_step, eta = task
    obs = _pushes[index]
    model = _models[time_step]
    model.set_eta(Eta(*eta))
    push_end, end = model.predict_path(
        obs.start,
        obs.tip_times,
        obs.tip_positions,
        obs.tip_velocities,
        [obs.push_end_time, obs.duration],
    )
    return np.concatenate([measurement(push_end, end), push_end[:3], end[:3]])


def mismatch(measured: np.ndarray, predicted: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Mahalanobis distance under `r` between measurements (..., 6), yaw
    wrapped."""
    errors = planar_error(measured, predicted)
    flat = squared_mahalanobis(errors.reshape(-1, 6), np.linalg.inv(r))
    return np.asarray(np.sqrt(flat.reshape(errors.shape[:-1])))


def best_mu(
    measured: np.ndarray, sweep: np.ndarray, mus: np.ndarray, r: np.ndarray
) -> float:
    """The friction in `mus` whose predictions `sweep` (P, len(mus), 6) best
    explain a trial's measurements `measured` (P, 6), by summed Student-t log
    likelihood."""
    return float(mus[int(np.argmax(log_likelihood_curve(measured, sweep, r)))])


def log_likelihood_curve(
    measured: np.ndarray, sweep: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """Summed Student-t log likelihood over the P pushes of `measured` (P, 6),
    per column of `sweep` (P, S, 6): shape (S,)."""
    errors = planar_error(measured[:, None, :], sweep)
    q = squared_mahalanobis(errors.reshape(-1, 6), np.linalg.inv(r))
    log_likelihood = student_t_log_likelihood(q, NU, 6).reshape(errors.shape[:2])
    return np.asarray(log_likelihood.sum(axis=0))


def band_mask(mu: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return np.asarray((mu >= band[0]) & (mu < band[1]))


def tasks_for(
    trials: Sequence[TrialData], grid: list[Eta], model_step: float, fine_step: float
) -> list[Task]:
    pushes = [
        (trial, nearest_hypothesis(grid, trial.eta))
        for trial in trials
        for _ in trial.observations
    ]
    tasks: list[Task] = []
    for i, (trial, near) in enumerate(pushes):
        tasks.append((i, model_step, tuple(trial.eta.as_array())))
        tasks.append((i, fine_step, tuple(trial.eta.as_array())))
        tasks.append((i, model_step, tuple(near.as_array())))
        tasks += [(i, model_step, (mu, trial.eta.bx, trial.eta.by)) for mu in MU_SWEEP]
        tasks += [(i, model_step, (mu, near.bx, near.by)) for mu in MU_SWEEP]
    return tasks


def run(trials_path: Path, fine_step: float, workers: int) -> dict[str, np.ndarray]:
    trials, metadata = load_dataset(trials_path)
    trials = [trial for trial in trials if trial.observations]
    model_step = float(metadata["model_time_step"])
    rod_mass = float(metadata["rod_mass"])
    grid = make_grid(
        GridSpec(), Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, rod_mass)).parts
    )
    tasks = tasks_for(trials, grid, model_step, fine_step)
    with Pool(
        workers,
        initializer=_start_worker,
        initargs=(trials_path, (model_step, fine_step)),
    ) as pool:
        results = np.array(pool.map(_predict, tasks, chunksize=4))
    pushes = [obs for trial in trials for obs in trial.observations]
    n_sweep = len(MU_SWEEP)
    per_push = results.reshape(len(pushes), 3 + 2 * n_sweep, 12)
    return {
        "trial_id": np.array(
            [k for k, trial in enumerate(trials) for _ in trial.observations]
        ),
        "eta": np.array(
            [trial.eta.as_array() for trial in trials for _ in trial.observations]
        ),
        "nearest_eta": np.array(
            [
                nearest_hypothesis(grid, trial.eta).as_array()
                for trial in trials
                for _ in trial.observations
            ]
        ),
        "measured": np.array([obs.measured() for obs in pushes]),
        "truth_start_pose": np.array([obs.start.planar()[:3] for obs in pushes]),
        "truth_push_end_pose": np.array([obs.push_end.planar()[:3] for obs in pushes]),
        "truth_end_pose": np.array([obs.end.planar()[:3] for obs in pushes]),
        "true_eta": per_push[:, 0, :6],
        "true_eta_fine": per_push[:, 1, :6],
        "nearest": per_push[:, 2, :6],
        "true_eta_push_end_pose": per_push[:, 0, 6:9],
        "true_eta_end_pose": per_push[:, 0, 9:12],
        "true_eta_fine_push_end_pose": per_push[:, 1, 6:9],
        "true_eta_fine_end_pose": per_push[:, 1, 9:12],
        "sweep_true_rod": per_push[:, 3 : 3 + n_sweep, :6],
        "sweep_nearest_rod": per_push[:, 3 + n_sweep :, :6],
        "mu_sweep": MU_SWEEP,
        "model_time_step": np.array(model_step),
        "fine_time_step": np.array(fine_step),
        "r": CALIBRATED_R,
    }


def summary(data: dict[str, np.ndarray]) -> str:
    mu = data["eta"][:, 0]
    r = data["r"]
    sd = np.sqrt(np.diag(r))
    model_ms = float(data["model_time_step"]) * 1e3
    fine_ms = float(data["fine_time_step"]) * 1e3
    lines = [f"{len(mu)} pushes; mismatch = Mahalanobis distance under CALIBRATED_R"]
    for band in MU_BANDS:
        m = band_mask(mu, band)
        coarse = mismatch(data["measured"][m], data["true_eta"][m], r)
        fine = mismatch(data["measured"][m], data["true_eta_fine"][m], r)
        near = mismatch(data["measured"][m], data["nearest"][m], r)
        bias = planar_error(data["measured"][m], data["true_eta"][m]).mean(axis=0) / sd
        slide_truth = np.linalg.norm(
            data["truth_end_pose"][m, :2] - data["truth_push_end_pose"][m, :2], axis=1
        )
        slide_model = np.linalg.norm(
            data["true_eta_end_pose"][m, :2] - data["true_eta_push_end_pose"][m, :2],
            axis=1,
        )
        lines += [
            (
                f"mu {band[0]:.2f}-{band[1]:.2f} ({m.sum()} pushes): median mismatch "
                f"true eta {np.median(coarse):.2f} at {model_ms:g} ms, "
                f"{np.median(fine):.2f} at {fine_ms:g} ms; nearest grid {np.median(near):.2f}"
            ),
            "    mean residual / R sd: "
            + ", ".join(f"{n} {b:+.2f}" for n, b in zip(COMPONENTS, bias, strict=True)),
            (
                f"    slide after push end: truth {slide_truth.mean() * 1e3:.2f} mm, "
                f"model {slide_model.mean() * 1e3:.2f} mm"
            ),
        ]
    lines.append("trial  true mu  best mu (true rod)  best mu (nearest-grid rod)")
    mus = data["mu_sweep"]
    for k in np.unique(data["trial_id"]):
        m = data["trial_id"] == k
        at_true = best_mu(data["measured"][m], data["sweep_true_rod"][m], mus, r)
        at_near = best_mu(data["measured"][m], data["sweep_nearest_rod"][m], mus, r)
        lines.append(f"{k:5d}  {mu[m][0]:7.2f}  {at_true:18.2f}  {at_near:26.2f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials", type=Path, default=Path("data/replay_trials_v2.npz")
    )
    parser.add_argument("--fine-time-step", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("data/model_bias.npz"))
    args = parser.parse_args()
    started = time.time()
    data = run(args.trials, args.fine_time_step, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, allow_pickle=False, trials_file=np.array(str(args.trials)), **data
    )
    print(summary(data))
    print(f"wrote {args.out} in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
