"""The unknown physical parameters eta = (mu, b) and the estimator's
hypothesis grid over them.

mu is the object-table friction coefficient; b = (bx, by) is the welded
rod's body-frame position, the displaced point mass that shifts the COM.
The rod's mass is known and fixed.
"""

import itertools
from dataclasses import dataclass

import numpy as np

from .geometry import SurfacePart, contains_disk, footprint_center
from .object import Object, Rod

# Known rod mass (kg), matching main.py's --rod-mass default.
ROD_MASS = 0.2


@dataclass(frozen=True)
class Eta:
    """One parameter hypothesis: friction `mu` and rod position (bx, by) in
    the object's body frame."""

    mu: float
    bx: float
    by: float

    def as_array(self) -> np.ndarray:
        """Get (mu, bx, by) as a vector."""
        return np.array([self.mu, self.bx, self.by])

    def rod(self, mass: float = ROD_MASS) -> Rod:
        """Get the rod this hypothesis places."""
        return Rod(x=self.bx, y=self.by, mass=mass)

    def to_object(self, rng: np.random.Generator, rod_mass: float = ROD_MASS) -> Object:
        """Build the object this hypothesis describes."""
        return Object(rng, rod=self.rod(rod_mass), friction=self.mu)


@dataclass(frozen=True)
class GridSpec:
    """Hypothesis grid layout: `n_*` evenly spaced values per parameter over
    each range.  The b ranges default (None) to the footprint's bounding box
    shrunk by the rod radius, so edge values can still hold the rod."""

    mu_range: tuple[float, float] = (0.1, 0.9)
    n_mu: int = 9
    bx_range: tuple[float, float] | None = None
    n_bx: int = 9
    by_range: tuple[float, float] | None = None
    n_by: int = 9

    def __post_init__(self) -> None:
        if min(self.n_mu, self.n_bx, self.n_by) < 1:
            raise ValueError("every grid count must be at least 1")
        if self.mu_range[0] < 0.0:
            raise ValueError("friction must be nonnegative")


def b_ranges(
    spec: GridSpec, parts: list[SurfacePart], radius: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Resolve the spec's b ranges in the body frame, defaulting to the
    footprint's bounding box shrunk by `radius`."""
    fx, fy = footprint_center(parts)
    lo_x = min(cx - sx / 2 for sx, sy, cx, cy in parts) - fx + radius
    hi_x = max(cx + sx / 2 for sx, sy, cx, cy in parts) - fx - radius
    lo_y = min(cy - sy / 2 for sx, sy, cx, cy in parts) - fy + radius
    hi_y = max(cy + sy / 2 for sx, sy, cx, cy in parts) - fy - radius
    bx = spec.bx_range if spec.bx_range is not None else (lo_x, hi_x)
    by = spec.by_range if spec.by_range is not None else (lo_y, hi_y)
    return bx, by


def _values(lo: float, hi: float, n: int) -> np.ndarray:
    """Get n evenly spaced values over [lo, hi]; a single value sits at the
    midpoint."""
    return np.array([(lo + hi) / 2]) if n == 1 else np.linspace(lo, hi, n)


def make_grid(
    spec: GridSpec, parts: list[SurfacePart], radius: float = Rod.radius
) -> list[Eta]:
    """Lay out the hypothesis grid: the Cartesian product of the spec's
    values, keeping only rod positions that lie wholly inside the outline
    (the same rule BoxComposite enforces)."""
    (bx_lo, bx_hi), (by_lo, by_hi) = b_ranges(spec, parts, radius)
    mus = _values(*spec.mu_range, spec.n_mu)
    sites = [
        (float(bx), float(by))
        for bx, by in itertools.product(
            _values(bx_lo, bx_hi, spec.n_bx), _values(by_lo, by_hi, spec.n_by)
        )
        if contains_disk(parts, bx, by, radius)
    ]
    if not sites:
        raise ValueError("no grid rod position fits inside the outline")
    return [Eta(float(mu), bx, by) for mu in mus for bx, by in sites]


def sample_eta(
    rng: np.random.Generator,
    spec: GridSpec,
    parts: list[SurfacePart],
    radius: float = Rod.radius,
) -> Eta:
    """Draw a continuous eta uniformly over the spec's ranges, rejecting rod
    positions outside the outline: off-grid "true" parameters."""
    (bx_lo, bx_hi), (by_lo, by_hi) = b_ranges(spec, parts, radius)
    while True:
        bx, by = rng.uniform(bx_lo, bx_hi), rng.uniform(by_lo, by_hi)
        if contains_disk(parts, bx, by, radius):
            return Eta(float(rng.uniform(*spec.mu_range)), float(bx), float(by))


def nearest_hypothesis(grid: list[Eta], eta: Eta) -> Eta:
    """Get the grid hypothesis nearest `eta`, each parameter scaled by the
    grid's span so friction and position compare evenly."""
    points = np.array([g.as_array() for g in grid])
    span = np.ptp(points, axis=0)
    span[span == 0.0] = 1.0
    distances = (((points - eta.as_array()) / span) ** 2).sum(axis=1)
    return grid[int(np.argmin(distances))]
