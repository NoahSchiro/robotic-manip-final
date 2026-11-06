"""Offline replay of recorded trials through the grid Bayes filter: the
frozen reference the estimator is judged by

    uv run python main.py record-trials   # the dataset (slow, Drake truth)
    uv run python main.py replay          # this report (minutes)

Each recorded trial replays push by push: one prediction pass over every
hypothesis feeds two filters that differ only in likelihood, the Student-t
(nu = 3) the estimator ships with and the proposal's Gaussian.  Metrics per
likelihood, scored against the true parameters:

Trials recorded with zero pushes carry no signal (their MAP is an arbitrary
tie-break), so they are dropped from the metrics and counted in the report.
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .dynamics import DynamicsModel, ParallelPushModel, PushModelConfig
from .estimator import CALIBRATED_R, CALIBRATED_R_SURROGATE, GridBayesFilter
from .object import Object, Rod
from .params import Eta, GridSpec, grid_spans, grid_step, make_grid
from .surrogate import SurrogatePredictor, load_ensemble
from .trials import TrialData, load_dataset

# A final belief this peaked with a wrong MAP counts as confidently wrong.
CONFIDENT_PROB = 0.9

# The likelihoods every replay compares, by name and degrees of freedom
# (None: the Gaussian the proposal specifies).
LIKELIHOODS: dict[str, float | None] = {"student_t": 3.0, "gaussian": None}

# Eta's components, in as_array() order, for labeling per-parameter errors.
ETA_PARAMETERS = ("mu", "bx", "by")


@dataclass(frozen=True)
class StepRecord:
    """The filter's state after one replayed push (or the prior at index 0)."""

    entropy: float
    max_p: float
    map_index: int
    mean_eta: np.ndarray


@dataclass(frozen=True)
class TrialTracks:
    """One trial's per-likelihood step records, prior included."""

    eta: Eta
    steps: dict[str, list[StepRecord]]


@dataclass(frozen=True)
class FinalMetrics:
    """One trial's final-step scores against its true parameters."""

    within_one_grid_step: bool
    map_is_nearest_cell: bool
    confidently_wrong: bool
    map_error_scaled: float
    abs_error_scaled: np.ndarray


@dataclass(frozen=True)
class MetricCount:
    """A pass/fail metric over the scored trials."""

    count: int
    total: int
    rate: float


@dataclass(frozen=True)
class CurvePoint:
    """Mean belief entropy after `push` pushes into a trial."""

    push: int
    mean_entropy: float


@dataclass(frozen=True)
class LikelihoodReport:
    """One likelihood's aggregated final metrics and entropy curve."""

    within_one_grid_step: MetricCount
    map_is_nearest_cell: MetricCount
    confidently_wrong: MetricCount
    map_error_scaled_mean: float
    abs_error_scaled_mean: dict[str, float]
    final_max_p_mean: float
    entropy_curve: list[CurvePoint]


@dataclass(frozen=True)
class ReplayReport:
    """The replay's aggregate over the scored trials, per likelihood."""

    recorded_trials: int
    scored_trials: int
    dropped_empty_trials: int
    mean_pushes: float
    hypotheses: int
    metadata: dict[str, float | int | str]
    likelihoods: dict[str, LikelihoodReport]


def nearest_index(grid: list[Eta], eta: Eta) -> int:
    """Get the index of the grid hypothesis nearest `eta` in scaled units."""
    points = np.array([g.as_array() for g in grid])
    distances = (((points - eta.as_array()) / grid_spans(grid)) ** 2).sum(axis=1)
    return int(np.argmin(distances))


def filter_step(belief: GridBayesFilter) -> StepRecord:
    """Snapshot the filter's current state."""
    return StepRecord(
        entropy=belief.entropy(),
        max_p=float(belief.probs.max()),
        map_index=int(np.argmax(belief.log_probs)),
        mean_eta=belief.mean_eta().as_array(),
    )


def replay_trial(
    trial: TrialData,
    grid: list[Eta],
    model: DynamicsModel,
    r_matrix: np.ndarray,
) -> TrialTracks:
    """Replay one trial's pushes through one filter per likelihood, sharing
    each push's single prediction pass between them."""
    filters = {
        name: GridBayesFilter(grid, model, r_matrix, nu)
        for name, nu in LIKELIHOODS.items()
    }
    steps = {name: [filter_step(belief)] for name, belief in filters.items()}
    for obs in trial.observations:
        predicted = model.predict(grid, obs)
        measured = obs.measured()
        for name, belief in filters.items():
            belief.update_from_predictions(measured, predicted)
            steps[name].append(filter_step(belief))
    return TrialTracks(eta=trial.eta, steps=steps)


def final_metrics(step: StepRecord, grid: list[Eta], eta: Eta) -> FinalMetrics:
    """Score one trial's final step against its true parameters."""
    points = np.array([g.as_array() for g in grid])
    scaled = (points[step.map_index] - eta.as_array()) / grid_spans(grid)
    step_error = np.abs(points[step.map_index] - eta.as_array()) / grid_step(grid)
    within_one_grid_step = bool((step_error <= 1.0 + 1e-9).all())
    return FinalMetrics(
        within_one_grid_step=within_one_grid_step,
        map_is_nearest_cell=step.map_index == nearest_index(grid, eta),
        confidently_wrong=step.max_p >= CONFIDENT_PROB and not within_one_grid_step,
        map_error_scaled=float(np.linalg.norm(scaled)),
        abs_error_scaled=np.abs(scaled),
    )


def count(flags: Iterable[bool]) -> MetricCount:
    """Tally one boolean metric over the scored trials."""
    values = list(flags)
    return MetricCount(
        count=sum(values), total=len(values), rate=float(np.mean(values))
    )


def likelihood_report(
    tracks: Sequence[TrialTracks], grid: list[Eta], name: str
) -> LikelihoodReport:
    """Aggregate one likelihood's final metrics and entropy curve."""
    finals = [final_metrics(track.steps[name][-1], grid, track.eta) for track in tracks]
    longest = max(len(track.steps[name]) for track in tracks)
    curve = [
        CurvePoint(
            push=push,
            mean_entropy=float(
                np.mean(
                    [
                        track.steps[name][push].entropy
                        for track in tracks
                        if len(track.steps[name]) > push
                    ]
                )
            ),
        )
        for push in range(longest)
    ]
    return LikelihoodReport(
        within_one_grid_step=count(f.within_one_grid_step for f in finals),
        map_is_nearest_cell=count(f.map_is_nearest_cell for f in finals),
        confidently_wrong=count(f.confidently_wrong for f in finals),
        map_error_scaled_mean=float(np.mean([f.map_error_scaled for f in finals])),
        abs_error_scaled_mean={
            parameter: float(mean)
            for parameter, mean in zip(
                ETA_PARAMETERS,
                np.mean([f.abs_error_scaled for f in finals], axis=0),
            )
        },
        final_max_p_mean=float(
            np.mean([track.steps[name][-1].max_p for track in tracks])
        ),
        entropy_curve=curve,
    )


def build_report(
    tracks: Sequence[TrialTracks],
    grid: list[Eta],
    metadata: dict[str, float | int | str],
    dropped_empty: int,
) -> ReplayReport:
    """Aggregate replayed tracks into the report."""
    return ReplayReport(
        recorded_trials=len(tracks) + dropped_empty,
        scored_trials=len(tracks),
        dropped_empty_trials=dropped_empty,
        mean_pushes=float(
            np.mean([len(track.steps["student_t"]) - 1 for track in tracks])
        ),
        hypotheses=len(grid),
        metadata=metadata,
        likelihoods={
            name: likelihood_report(tracks, grid, name) for name in LIKELIHOODS
        },
    )


def format_report(report: ReplayReport) -> str:
    """Render the report as the command's human-readable summary."""
    summary = (
        f"replayed {report.scored_trials}/{report.recorded_trials} trials "
        f"({report.dropped_empty_trials} empty dropped), "
        f"mean {report.mean_pushes:.1f} pushes, "
        f"{report.hypotheses} hypotheses"
    )
    header = (
        f"{'likelihood':<16} {'within-1-step':>14} {'nearest-cell':>13} "
        f"{'confident-wrong':>16} {'MAP err':>8} {'mean max p':>10}"
    )
    lines = [summary, header]
    for name, stats in report.likelihoods.items():
        within, nearest, wrong = (
            stats.within_one_grid_step,
            stats.map_is_nearest_cell,
            stats.confidently_wrong,
        )
        lines.append(
            f"{name:<16} {within.count:>5}/{within.total:<3} "
            f"{within.rate:.2f} {nearest.count:>4}/{nearest.total:<3} "
            f"{nearest.rate:.2f} {wrong.count:>6}/{wrong.total:<3} "
            f"{wrong.rate:.2f} {stats.map_error_scaled_mean:>10.2f} "
            f"{stats.final_max_p_mean:>12.2f}"
        )
    return "\n".join(lines)


def run_replay(
    trials_path: Path,
    workers: int,
    report_path: Path,
    predictor: str = "drake",
    ensemble_path: Path | None = None,
) -> None:
    """Replay the recorded trials and write the report JSON next to a
    printed summary.  `predictor` picks the filter's prediction model: the
    Drake plant, or the trained surrogate."""
    trials, metadata = load_dataset(trials_path)
    scored = [trial for trial in trials if trial.observations]
    rod_mass = float(metadata["rod_mass"])
    grid = make_grid(
        GridSpec(), Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, rod_mass)).parts
    )
    print(
        f"replaying {len(scored)}/{len(trials)} trials over {len(grid)} "
        f"hypotheses with the {predictor} predictor"
    )
    r_matrix = CALIBRATED_R_SURROGATE if predictor == "surrogate" else CALIBRATED_R
    if predictor == "surrogate":
        assert ensemble_path is not None
        model: DynamicsModel = SurrogatePredictor(load_ensemble(ensemble_path))
        tracks = [replay_trial(trial, grid, model, r_matrix) for trial in scored]
    else:
        model_config = PushModelConfig(
            time_step=float(metadata["model_time_step"]), rod_mass=rod_mass
        )
        with ParallelPushModel(workers, model_config) as model:
            tracks = [replay_trial(trial, grid, model, r_matrix) for trial in scored]
    report = build_report(
        tracks,
        grid,
        {**metadata, "predictor": predictor},
        dropped_empty=len(trials) - len(scored),
    )
    print(format_report(report))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), indent=2) + "\n")
    print(f"wrote {report_path}")
