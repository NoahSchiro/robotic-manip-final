import math

import pytest

from src.geometry import distance_to_footprint

# The T: a crossbar across the bounding box's -x end, a stem along +x.  Its
# bounding box is centered on the parts' origin, so body and parts frames agree.
T_PARTS = [(0.05, 0.14, -0.045, 0.0), (0.09, 0.05, 0.025, 0.0)]


@pytest.mark.parametrize(
    ("x", "y"),
    [(0.0, 0.0), (-0.045, 0.06), (0.06, 0.0), (0.07, 0.025)],
)
def test_points_on_or_inside_the_footprint_are_at_distance_zero(
    x: float, y: float
) -> None:
    assert distance_to_footprint(T_PARTS, x, y) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (0.08, 0.0, 0.01),  # past the stem's end
        (-0.08, 0.0, 0.01),  # behind the crossbar
        (0.03, 0.035, 0.01),  # beside the stem
        (-0.01, 0.06, 0.01),  # in the notch, nearest the crossbar's face
        (0.08, 0.035, math.hypot(0.01, 0.01)),  # off the stem's corner
    ],
)
def test_outside_points_measure_to_the_nearest_part(
    x: float, y: float, expected: float
) -> None:
    assert distance_to_footprint(T_PARTS, x, y) == pytest.approx(expected)


def test_body_frame_is_centered_on_the_bounding_box() -> None:
    # A big box and a thin one: the bounding box spans x in [-1, 2], so its
    # center is x = 0.5, where an area-weighted center would be near x = 0.07.
    parts = [(2.0, 2.0, 0.0, 0.0), (1.0, 0.2, 1.5, 0.0)]
    # Body x = 1.6 is parts x = 2.1, just past the thin box's end at x = 2.
    assert distance_to_footprint(parts, 1.6, 0.0) == pytest.approx(0.1)
