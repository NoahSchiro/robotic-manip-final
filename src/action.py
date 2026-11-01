"""The push action

Selects a surface normal, a speed and time it can contact the object.
"""

from dataclasses import dataclass

import numpy as np

from .geometry import SurfaceNormal


@dataclass(frozen=True)
class Action:
    """A commanded push: tip speed `velocity` along `contact`'s inward normal,
    sustained for `time` seconds in contact.

    Raises ValueError when `velocity` or `time` is not positive.
    """

    contact: SurfaceNormal
    velocity: float  # m/s along the inward normal
    time: float  # s, duration in contact

    def __post_init__(self) -> None:
        if self.velocity <= 0.0:
            raise ValueError(f"velocity must be positive, got {self.velocity}")
        if self.time <= 0.0:
            raise ValueError(f"time must be positive, got {self.time}")


# The library's commanded tip speeds (m/s) and contact times (s).
SPEEDS = (0.05, 0.1)
TIMES = (0.2, 0.4)


def push_library(
    sites: list[SurfaceNormal],
    speeds: tuple[float, ...] = SPEEDS,
    times: tuple[float, ...] = TIMES,
) -> list[Action]:
    """List every library push: each contact site at each commanded speed
    and contact time."""
    return [
        Action(contact=site, velocity=speed, time=time)
        for site in sites
        for speed in speeds
        for time in times
    ]


def sample_action(
    rng: np.random.Generator,
    sites: list[SurfaceNormal],
    speeds: tuple[float, ...] = SPEEDS,
    times: tuple[float, ...] = TIMES,
) -> Action:
    """Draw a push uniformly from the library: a contact site, a commanded
    speed, and a contact time."""
    return Action(
        contact=sites[rng.integers(len(sites))],
        velocity=float(rng.choice(speeds)),
        time=float(rng.choice(times)),
    )
