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

    def rod(self, mass: float) -> Rod:
        """Get the rod this hypothesis places."""
        return Rod(x=self.bx, y=self.by, mass=mass)

    def to_object(self, rng: np.random.Generator, rod_mass: float) -> Object:
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


def grid_spans(grid: list[Eta]) -> np.ndarray:
    """Get each parameter's grid extent (1.0 where the grid is constant):
    the unit parameter distances are scaled in for comparison."""
    points = np.array([g.as_array() for g in grid])
    span = np.ptp(points, axis=0)
    span[span == 0.0] = 1.0
    return span


def grid_step(grid: list[Eta]) -> np.ndarray:
    """Get one grid step per parameter: the gap between adjacent values on
    each axis (uniform on the regular axes make_grid lays out)."""
    points = np.array([g.as_array() for g in grid])
    step = np.ones(points.shape[1])
    for axis in range(points.shape[1]):
        values = np.unique(points[:, axis])
        if len(values) > 1:
            step[axis] = np.diff(values).min()
    return step


@dataclass(frozen=True)
class Subcells:
    """Points spread evenly inside each grid cell (the box within half a grid
    step of a hypothesis), with interpolation weights over the grid: point
    k's value of any smooth per-hypothesis quantity (the estimator uses the
    wrapped measurement errors) is weights[k] @ (the grid's values)."""

    points: np.ndarray  # (M, 3) (mu, bx, by)
    owner: np.ndarray  # (M,) index of the grid hypothesis whose cell holds each point
    weights: np.ndarray  # (M, len(grid)) interpolation weights, rows summing to 1


def subcells(grid: list[Eta], n: int) -> Subcells:
    """Split every grid cell into n x n x n equal boxes and place a point at
    each box's centre; n = 1 gives the grid itself.

    A point's weights multilinearly interpolate the eight grid hypotheses
    around it.  Along an axis where one of those is missing (the outline's
    edge, the end of a parameter range), the value is held constant at the
    owning hypothesis's on that axis."""
    points = np.array([g.as_array() for g in grid])
    step = grid_step(grid)
    lattice = np.rint((points - points.min(axis=0)) / step).astype(int)
    index = {tuple(cell): i for i, cell in enumerate(lattice)}
    offsets = (np.arange(n) + 0.5) / n - 0.5
    owners, positions, rows = [], [], []
    for i, cell in enumerate(lattice):
        for offset in itertools.product(offsets, repeat=3):
            position = cell + np.array(offset)
            owners.append(i)
            positions.append(position)
            rows.append(_clamped_weights(position, cell, index))
    weights = np.zeros((len(rows), len(grid)))
    for k, row in enumerate(rows):
        for node, weight in row.items():
            weights[k, node] = weight
    return Subcells(
        points=points.min(axis=0) + np.array(positions) * step,
        owner=np.array(owners),
        weights=weights,
    )


def _corner_weights(
    position: np.ndarray, index: dict[tuple[int, ...], int]
) -> dict[int, float] | None:
    """Multilinear weights of the lattice nodes around `position` (lattice
    units), or None when a node with nonzero weight is missing."""
    base = np.floor(position + 1e-9).astype(int)
    frac = position - base
    weights: dict[int, float] = {}
    for corner in itertools.product((0, 1), repeat=len(position)):
        weight = float(np.prod([f if c else 1.0 - f for c, f in zip(corner, frac)]))
        if weight < 1e-12:
            continue
        node = index.get(tuple(base + np.array(corner)))
        if node is None:
            return None
        weights[node] = weight
    return weights


def _clamped_weights(
    position: np.ndarray, cell: np.ndarray, index: dict[tuple[int, ...], int]
) -> dict[int, float]:
    """Interpolation weights at `position`, snapping the fewest axes to the
    owning `cell` that leaves every needed node on the grid."""
    for count in range(len(position) + 1):
        for axes in itertools.combinations(range(len(position)), count):
            snapped = position.copy()
            snapped[list(axes)] = cell[list(axes)]
            weights = _corner_weights(snapped, index)
            if weights is not None:
                return weights
    raise AssertionError("the owning cell is always on the grid")


def nearest_subcell(cells: Subcells, grid: list[Eta], eta: Eta) -> int:
    """Index of the sub-cell point nearest `eta`, each parameter scaled by
    its grid step."""
    scaled = (cells.points - eta.as_array()) / grid_step(grid)
    return int(np.argmin((scaled**2).sum(axis=1)))


def nearest_hypothesis(grid: list[Eta], eta: Eta) -> Eta:
    """Get the grid hypothesis nearest `eta`, each parameter scaled by the
    grid's span so friction and position compare evenly."""
    points = np.array([g.as_array() for g in grid])
    distances = (((points - eta.as_array()) / grid_spans(grid)) ** 2).sum(axis=1)
    return grid[int(np.argmin(distances))]
