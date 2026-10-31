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


def sample_action(
    rng: np.random.Generator,
    sites: list[SurfaceNormal],
    speeds: tuple[float, ...] = (0.05, 0.1),
    times: tuple[float, ...] = (0.2, 0.4),
) -> Action:
    """Draw a push uniformly from the library: a contact site, a commanded
    speed, and a contact time."""
    return Action(
        contact=sites[rng.integers(len(sites))],
        velocity=float(rng.choice(speeds)),
        time=float(rng.choice(times)),
    )
