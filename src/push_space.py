"""The push space: where a push can contact the block and how it is commanded.

A push contacts one straight face of the block's outline at a point along the
face's usable span, at a commanded speed and contact time.  The span stops
INNER_CORNER_INSET short of an inner (concave) corner, so the lowered stick
clears the other part of the T, and OUTER_CORNER_INSET short of an outer
corner, so arm tracking error (a few mm) keeps the contact on one face.
Sampling, the MPC's candidate grid, reachability and the demo all use this one
definition.

Body frame: x and y in m, z = 0 at the footprint's bottom face; contacts sit at
mid-height.  Angles are radians and times seconds.
"""

from dataclasses import dataclass

import numpy as np

from .action import Action
from .geometry import (
    SurfaceEdge,
    SurfaceNormal,
    SurfacePart,
    boundary_edges,
    footprint_center,
    trace_outline,
)
from .iiwa import STANDOFF

SPEED_RANGE = (0.03, 0.12)  # m/s, commanded tip speed
DURATION_RANGE = (0.1, 0.5)  # s, contact time
# m; the inner inset exceeds the stick radius (0.01 m) plus 1 mm clearance.
INNER_CORNER_INSET = 0.012
OUTER_CORNER_INSET = 0.005
REACH_MIN, REACH_MAX = 0.39, 0.78  # m, tip radius band from the arm base


@dataclass(frozen=True)
class Face:
    """A straight side of the outline in the body frame: xy endpoints `start`
    and `end` (m), the inward unit `normal` (3,), the contact height `height`
    (m) above the footprint, and the insets (m) kept clear at each end."""

    start: np.ndarray  # (2,)
    end: np.ndarray  # (2,)
    normal: np.ndarray  # (3,)
    height: float
    start_inset: float
    end_inset: float

    @property
    def length(self) -> float:
        """Side length (m)."""
        return float(np.linalg.norm(self.end - self.start))

    @property
    def span(self) -> float:
        """Usable length (m): the side minus both end insets."""
        return self.length - self.start_inset - self.end_inset


@dataclass(frozen=True)
class PushSpace:
    """The faces a push may contact and the commanded speed (m/s) and contact
    time (s) ranges.  Faces are indexed in outline order."""

    faces: tuple[Face, ...]
    speed_range: tuple[float, float]  # m/s
    duration_range: tuple[float, float]  # s

    def usable_spans(self) -> np.ndarray:
        """Each face's usable length (m), (F,)."""
        return np.array([face.span for face in self.faces])

    def contact(self, face: int, s: float) -> SurfaceNormal:
        """The contact at normalized position `s` in [0, 1] along the usable
        span of face `face`: s = 0 at the inset from its start, s = 1 at the
        inset from its end."""
        f = self.faces[face]
        along = f.start_inset + s * f.span
        xy = f.start + along * (f.end - f.start) / f.length
        return SurfaceNormal(
            position=np.array([xy[0], xy[1], f.height]), normal=f.normal.copy()
        )

    def sample(self, rng: np.random.Generator) -> Action:
        """A uniform push: the face is chosen in proportion to its usable span,
        the position, speed and contact time uniformly over their ranges."""
        spans = self.usable_spans()
        face = int(rng.choice(len(self.faces), p=spans / spans.sum()))
        s = float(rng.uniform())
        velocity = float(rng.uniform(*self.speed_range))
        duration = float(rng.uniform(*self.duration_range))
        return Action(contact=self.contact(face, s), velocity=velocity, time=duration)

    def grid(self, spacing: float, n_speeds: int, n_durations: int) -> list[Action]:
        """Every push on a regular lattice: contact positions about `spacing` m
        apart along each usable span, both ends included (at least two per
        face), and speeds and durations evenly spaced over their ranges, both
        range endpoints included."""
        speeds = np.linspace(*self.speed_range, n_speeds)
        durations = np.linspace(*self.duration_range, n_durations)
        actions: list[Action] = []
        for face, span in enumerate(self.usable_spans()):
            positions = np.linspace(0.0, 1.0, max(2, round(span / spacing) + 1))
            for s in positions:
                contact = self.contact(face, float(s))
                for velocity in speeds:
                    for duration in durations:
                        actions.append(
                            Action(contact, float(velocity), float(duration))
                        )
        return actions


def make_push_space(
    parts: list[SurfacePart],
    height: float,
    inner_inset: float = INNER_CORNER_INSET,
    outer_inset: float = OUTER_CORNER_INSET,
) -> PushSpace:
    """The push space of the block with footprint `parts` and height `height`
    (m).  Faces are the outline's straight runs in counterclockwise order, with
    the inward normal to the left of each run; each end is inset by
    `inner_inset` at an inner corner and `outer_inset` at an outer one.  A face
    too short for a usable span is dropped."""
    center = footprint_center(parts)
    runs = _straight_runs(trace_outline(boundary_edges(parts)))
    faces: list[Face] = []
    for i, ((x0, y0), (x1, y1)) in enumerate(runs):
        start = np.array([x0, y0]) - center
        end = np.array([x1, y1]) - center
        length = float(np.linalg.norm(end - start))
        normal = np.array([-(end[1] - start[1]), end[0] - start[0], 0.0]) / length
        face = Face(
            start=start,
            end=end,
            normal=normal,
            height=height / 2,
            start_inset=_corner_inset(runs[i - 1], runs[i], inner_inset, outer_inset),
            end_inset=_corner_inset(
                runs[i], runs[(i + 1) % len(runs)], inner_inset, outer_inset
            ),
        )
        if face.span > 0.0:
            faces.append(face)
    return PushSpace(
        faces=tuple(faces), speed_range=SPEED_RANGE, duration_range=DURATION_RANGE
    )


def _corner_inset(
    incoming: SurfaceEdge, outgoing: SurfaceEdge, inner: float, outer: float
) -> float:
    """The inset at the corner between two counterclockwise runs: a left turn
    is an outer (convex) corner, a right turn an inner (concave) one."""
    (ax, ay), (bx, by) = _step(incoming), _step(outgoing)
    return outer if ax * by - ay * bx > 0 else inner


def reachable(pose: np.ndarray, action: Action) -> bool:
    """Whether the arm reaches both the approach point (STANDOFF before the
    contact along the normal) and the push's end (the contact advanced by
    velocity * time along the normal), for a block at `pose` (x m, y m, yaw
    rad) in the world.  The arm base is the world origin, so each tip's radius
    must lie in [REACH_MIN, REACH_MAX]."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    rot = np.array([[c, -s], [s, c]])
    tip = action.contact.position[:2]
    normal = action.contact.normal[:2]
    approach = tip - STANDOFF * normal
    push_end = tip + action.velocity * action.time * normal
    radii = [np.linalg.norm(pose[:2] + rot @ point) for point in (approach, push_end)]
    return all(REACH_MIN <= r <= REACH_MAX for r in radii)


def _step(edge: SurfaceEdge) -> tuple[int, int]:
    """The edge's unit direction along the axes (boundary edges are axis-aligned)."""
    (x0, y0), (x1, y1) = edge
    return int(np.sign(x1 - x0)), int(np.sign(y1 - y0))


def _straight_runs(loop: list[SurfaceEdge]) -> list[SurfaceEdge]:
    """Merge consecutive edges running the same way into one run.  The loop is
    rotated to start at a corner, so no run wraps around its start."""
    corner = next(i for i in range(len(loop)) if _step(loop[i - 1]) != _step(loop[i]))
    rotated = loop[corner:] + loop[:corner]
    runs: list[SurfaceEdge] = []
    for edge in rotated:
        if runs and _step(runs[-1]) == _step(edge):
            runs[-1] = (runs[-1][0], edge[1])
        else:
            runs.append(edge)
    return runs
