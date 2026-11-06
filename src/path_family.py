"""The executed-path deviation family: a low-dimensional model of how the
real arm's tracked tip paths deviate from the commanded ones.

The arm's PD tracking produces force- and gravity-dependent deviations of
~4 mm RMS from the commanded tip path — a distribution shift that
dominated the first transfer check (surrogate-trained policies degraded
on real executed paths).  This family is calibrated on the recorded
trials' measured paths: each recorded push's measured xy path minus its
reconstructed nominal path, stacked and decomposed by SVD.  The result is
a mean deviation plus principal components with calibrated amplitudes —
sampled per push, they generate executed paths that span the real arm's
deviation distribution (rank-8 residual 0.32 mm on the recorded pushes).

The deviation lives on PATH_SAMPLES uniform window-fraction points in the
world xy frame, and is interpolated onto whatever time grid a path uses.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydrake.all import RigidTransform

from .action import Action
from .geometry import SurfaceNormal
from .surrogate import (
    PATH_SAMPLES,
    nominal_tip_path,
    path_downsample,
)

KEEP_COMPONENTS = 8


@dataclass(frozen=True)
class PathFamily:
    """The calibrated executed-path deviation model (world xy)."""

    mean: np.ndarray  # (2, PATH_SAMPLES) average deviation
    components: np.ndarray  # (K, 2, PATH_SAMPLES) principal directions
    sigmas: np.ndarray  # (K,) per-component amplitudes

    @property
    def n_components(self) -> int:
        return len(self.sigmas)

    def coefficients(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one push's deviation coefficients."""
        return self.sigmas * rng.standard_normal(self.n_components)

    def deviation_curve(
        self, coeffs: np.ndarray, window_end: float, times: np.ndarray
    ) -> np.ndarray:
        """Evaluate the (2, n) deviation at absolute `times`, interpolating
        the (2, PATH_SAMPLES) model over the window's normalized time."""
        u_model = np.linspace(0.0, 1.0, PATH_SAMPLES)
        u = times / max(window_end, 1e-9)
        curve = self.mean + np.einsum("k,kij->ij", coeffs, self.components)
        return np.vstack([np.interp(u, u_model, curve[row]) for row in range(2)])


def calibrate_path_family(
    trials_path: Path, settle_time: float = 0.1, keep: int = KEEP_COMPONENTS
) -> tuple[PathFamily, float]:
    """Calibrate the family from a recorded-trials dataset: the deviations
    of each recorded push's measured xy path from its reconstructed nominal
    path, decomposed by SVD.  Returns the family and the residual RMS after
    the kept components (mm-scale diagnostics are the caller's job)."""
    with np.load(trials_path, allow_pickle=True) as data:
        tip_times = data["tip_times"]
        tip_positions = data["tip_positions"]
        start_poses = data["start_pose"]
        contacts = data["contact_position"]
        normals = data["contact_normal"]
        speeds = data["action_velocity"]
        times_cmd = data["action_time"]

    rows: list[np.ndarray] = []
    n_pushes = len(start_poses)
    for push in range(n_pushes):
        pose_matrix = start_poses[push]
        pose = _pose_from_matrix(pose_matrix)
        action = _action_from_arrays(
            contacts[push], normals[push], float(speeds[push]), float(times_cmd[push])
        )
        nominal = nominal_tip_path(pose, action, settle_time)
        measured = path_downsample(tip_times[push], tip_positions[push])[:2]
        nominal_xy = path_downsample(nominal.tip_times, nominal.tip_positions)[:2]
        rows.append((measured - nominal_xy).reshape(-1))

    deviations = np.array(rows)  # (P, 32)
    mean = deviations.mean(axis=0).reshape(2, PATH_SAMPLES)
    centered = deviations - deviations.mean(axis=0)
    u, sv, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:keep].reshape(keep, 2, PATH_SAMPLES)
    sigmas = sv[:keep] / np.sqrt(n_pushes)
    reconstructed = mean.reshape(-1) + (u[:, :keep] * sv[:keep]) @ vt[:keep]
    residual = float(np.sqrt(((deviations - reconstructed) ** 2).mean()))
    family = PathFamily(
        mean=mean,
        components=components,
        sigmas=sigmas,
    )
    return family, residual


def _pose_from_matrix(matrix: np.ndarray) -> RigidTransform:
    return RigidTransform(matrix)


def _action_from_arrays(
    contact_position: np.ndarray,
    contact_normal: np.ndarray,
    velocity: float,
    time: float,
) -> Action:
    return Action(
        contact=SurfaceNormal(position=contact_position, normal=contact_normal),
        velocity=velocity,
        time=time,
    )


def save_path_family(path: Path, family: PathFamily) -> None:
    """Write the family's mean, components and amplitudes to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "mean": family.mean,
        "sigmas": family.sigmas,
    }
    for k, component in enumerate(family.components):
        arrays[f"component{k}"] = component
    np.savez(path, allow_pickle=True, **arrays)


def load_path_family(path: Path) -> PathFamily:
    """Read a family written by save_path_family."""
    with np.load(path) as data:
        components = np.stack(
            [data[f"component{k}"] for k in range(len(data.files) - 2)]
        )
        return PathFamily(
            mean=data["mean"],
            components=components,
            sigmas=data["sigmas"],
        )


def perturb_tip_positions(
    tip_positions: np.ndarray,
    tip_times: np.ndarray,
    window_end: float,
    family: PathFamily,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return the (3, n) tip positions with the family's sampled xy
    deviation applied (the z row passes through)."""
    coeffs = family.coefficients(rng)
    deviation = family.deviation_curve(coeffs, window_end, tip_times)
    perturbed = tip_positions.copy()
    perturbed[:2] += deviation
    return perturbed
