"""Planar geometry of a footprint built from axis-aligned boxes ("parts").

Everything here is math on the parts' coordinates:
- outline extraction
- containment tests
- push-contact sampling

Coordinates are in the body frame: x and y centered on the
footprint's bounding box, z = 0 at the footprint's bottom face.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceNormal:
    """A push contact on the footprint outline: position in the body frame
    (at mid-height) with the outline's inward unit normal there."""

    position: np.ndarray
    normal: np.ndarray


# One directed straight boundary edge (start point, end point), oriented so
# the footprint's interior lies on the edge's left.
SurfaceEdge = tuple[tuple[float, float], tuple[float, float]]
SurfacePart = tuple[float, float, float, float]


def footprint_center(parts: list[SurfacePart]) -> np.ndarray:
    """Get the xy center of the parts' bounding box."""
    lo_x = min(cx - sx / 2 for sx, sy, cx, cy in parts)
    hi_x = max(cx + sx / 2 for sx, sy, cx, cy in parts)
    lo_y = min(cy - sy / 2 for sx, sy, cx, cy in parts)
    hi_y = max(cy + sy / 2 for sx, sy, cx, cy in parts)
    return np.array([(lo_x + hi_x) / 2, (lo_y + hi_y) / 2])


def boundary_edges(parts: list[SurfacePart]) -> list[SurfaceEdge]:
    """Compute the boxes' union outline as directed edges running
    counterclockwise, keeping the union's interior on each edge's left."""
    xs = sorted({v for sx, _sy, cx, _cy in parts for v in (cx - sx / 2, cx + sx / 2)})
    ys = sorted({v for _sx, sy, _cx, cy in parts for v in (cy - sy / 2, cy + sy / 2)})

    def covered(x_lo: float, x_hi: float, y_lo: float, y_hi: float) -> bool:
        """Check whether the cell strictly between these coordinates lies in
        some box; box edges are coordinates, so no cell straddles one."""
        mx, my = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
        return any(
            px - sx / 2 <= mx <= px + sx / 2 and py - sy / 2 <= my <= py + sy / 2
            for sx, sy, px, py in parts
        )

    edges: list[SurfaceEdge] = []
    for i, x in enumerate(xs):
        for j in range(len(ys) - 1):
            west = covered(xs[i - 1], x, ys[j], ys[j + 1]) if i > 0 else False
            east = covered(x, xs[i + 1], ys[j], ys[j + 1]) if i + 1 < len(xs) else False
            if west != east:
                # Interior on the west side means the edge travels northward.
                a, b = (ys[j], ys[j + 1]) if west else (ys[j + 1], ys[j])
                edges.append(((x, a), (x, b)))
    for j, y in enumerate(ys):
        for i in range(len(xs) - 1):
            south = covered(xs[i], xs[i + 1], ys[j - 1], y) if j > 0 else False
            north = (
                covered(xs[i], xs[i + 1], y, ys[j + 1]) if j + 1 < len(ys) else False
            )
            if south != north:
                # Interior on the north side means the edge travels eastward.
                a, b = (xs[i], xs[i + 1]) if north else (xs[i + 1], xs[i])
                edges.append(((a, y), (b, y)))
    return edges


def trace_outline(edges: list[SurfaceEdge]) -> list[SurfaceEdge]:
    """Chain directed boundary edges into the one closed loop they form,
    started at the lexicographically smallest vertex.  Raises ValueError when
    they form none (parts touching only at corners) or several (disconnected
    parts)."""
    ordered = sorted(edges)
    if not ordered:
        raise ValueError("parts have an empty footprint")
    following: dict[tuple[float, float], list[SurfaceEdge]] = {}
    for edge in ordered:
        following.setdefault(edge[0], []).append(edge)
    loop = [ordered[0]]
    while True:
        starts = following[loop[-1][1]]
        if len(starts) != 1:
            raise ValueError("footprint outline is not a single simple loop")
        nxt = starts[0]
        if nxt == loop[0]:
            break
        loop.append(nxt)
    if len(loop) != len(ordered):
        raise ValueError("footprint outline is not a single loop")
    return loop


def sample_outline(
    parts: list[SurfacePart], height: float, density: float
) -> list[SurfaceNormal]:
    """Sample the footprint outline into a finite, deterministic set of push
    contacts: round(density * perimeter) sites evenly spaced along the
    counterclockwise outline, each at mid-height with its inward unit normal,
    in the body frame."""
    if density <= 0.0:
        raise ValueError("density must be positive")
    loop = trace_outline(boundary_edges(parts))
    lengths = [math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in loop]
    perimeter = sum(lengths)
    count = max(1, round(density * perimeter))
    fx, fy = footprint_center(parts)
    sites: list[SurfaceNormal] = []
    walked = 0.0
    i = 0
    for k in range(count):
        s = k * perimeter / count
        while s >= walked + lengths[i]:
            walked += lengths[i]
            i += 1
        (x0, y0), (x1, y1) = loop[i]
        t = (s - walked) / lengths[i]
        sites.append(
            SurfaceNormal(
                position=np.array(
                    [x0 + t * (x1 - x0) - fx, y0 + t * (y1 - y0) - fy, height / 2]
                ),
                normal=np.array([y0 - y1, x1 - x0, 0.0]) / lengths[i],
            )
        )
    return sites


def contains_disk(parts: list[SurfacePart], x: float, y: float, radius: float) -> bool:
    """Check whether a disk of `radius` centered at body-frame (x, y) lies
    wholly inside the parts' union outline."""
    fx, fy = footprint_center(parts)
    # body frame -> parts frame
    x, y = x + fx, y + fy
    lo, hi = x - radius, x + radius
    edges = {
        edge
        for sx, _sy, cx, _cy in parts
        for edge in (cx - sx / 2, cx + sx / 2)
        if lo < edge < hi
    }
    breaks = sorted({lo, hi, *edges})
    for u, v in itertools.pairwise(breaks):
        # The disk's y-interval is widest at the slab point nearest (x, y).
        w = np.sqrt(radius**2 - (min(max(x, u), v) - x) ** 2)
        spans = sorted(
            (cy - sy / 2, cy + sy / 2)
            for sx, sy, cx, cy in parts
            if cx - sx / 2 <= u and v <= cx + sx / 2
        )
        merged: list[list[float]] = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        if not any(a <= y - w and y + w <= b for a, b in merged):
            return False
    return True


def distance_to_footprint(parts: list[SurfacePart], x: float, y: float) -> float:
    """Get the distance from body-frame point (x, y) to the parts' union
    footprint, 0 inside it."""
    fx, fy = footprint_center(parts)
    # body frame -> parts frame
    x, y = x + fx, y + fy
    return min(
        math.hypot(max(abs(x - cx) - sx / 2, 0.0), max(abs(y - cy) - sy / 2, 0.0))
        for sx, sy, cx, cy in parts
    )
