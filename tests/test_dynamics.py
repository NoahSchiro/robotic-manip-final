import numpy as np
import pytest

from src.estimation.dynamics import nominal_tip_path
from src.world.action import Action
from src.world.geometry import SurfaceNormal
from src.world.iiwa import (
    LEG_DWELL,
    LEG_LIFT,
    LEG_PUSH,
    LEG_RELEASE,
    RELEASE_SPEED,
    push_maneuver,
    transfer_height,
)
from src.world.object import Object, Rod


def make_push() -> tuple[Object, Action]:
    obj = Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, 0.2))
    obj.pose = (0.5, 0.0, 0.7)
    # The stem's +x end, pushed toward -x.
    site = SurfaceNormal(
        position=np.array([0.07, 0.0, 0.015]), normal=np.array([-1.0, 0.0, 0.0])
    )
    return obj, Action(contact=site, velocity=0.1, time=0.4)


def test_nominal_tip_path_runs_through_the_maneuver_from_the_standoff() -> None:
    obj, action = make_push()
    path = nominal_tip_path(obj, action)
    maneuver = push_maneuver(obj.x_wo(), action, transfer_height(obj))
    # From the standoff, where the dwell holds the tip: push, release, rise.
    legs = [LEG_DWELL, LEG_PUSH, LEG_RELEASE, LEG_LIFT]
    waypoints = [maneuver[leg][0].translation() for leg in legs]
    times = np.cumsum([0.0, *(maneuver[leg][1] for leg in legs[1:])])

    for t, waypoint in zip(times, waypoints):
        assert path.value(t).ravel() == pytest.approx(waypoint)
    assert path.end_time() == pytest.approx(times[-1])


def test_nominal_tip_path_pushes_at_the_commanded_speed_then_releases_slowly() -> None:
    obj, action = make_push()
    path = nominal_tip_path(obj, action)
    breaks = path.get_segment_times()
    expected_speeds = [action.velocity, RELEASE_SPEED]
    for i, speed in enumerate(expected_speeds):  # the push, then the release
        t0, t1 = breaks[i], breaks[i + 1]
        moved = np.linalg.norm(path.value(t1).ravel() - path.value(t0).ravel())
        assert moved / (t1 - t0) == pytest.approx(speed)
