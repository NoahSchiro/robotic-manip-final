"""The push action

Selects a surface normal, a speed and time it can contact the object.
"""

from dataclasses import dataclass

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
