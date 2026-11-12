import numpy as np
import pytest

from src.action import Action
from src.dynamics import nominal_tip_path
from src.geometry import SurfaceNormal
from src.iiwa import hover_tip_pose, push_maneuver
from src.object import Object, Rod


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
    maneuver = push_maneuver(obj, action, hover_tip_pose(obj))
    # From the standoff: push, back off, rise.
    waypoints = [pose.translation() for pose, _ in maneuver[1:5]]
    times = np.cumsum([0.0, *(duration for _pose, duration in maneuver[2:5])])

    for t, waypoint in zip(times, waypoints):
        assert path.value(t).ravel() == pytest.approx(waypoint)
    assert path.end_time() == pytest.approx(times[-1])


def test_nominal_tip_path_pushes_and_backs_off_at_the_commanded_speed() -> None:
    obj, action = make_push()
    path = nominal_tip_path(obj, action)
    breaks = path.get_segment_times()
    for i in (0, 1):  # the push, then the back-off
        t0, t1 = breaks[i], breaks[i + 1]
        moved = np.linalg.norm(path.value(t1).ravel() - path.value(t0).ravel())
        assert moved / (t1 - t0) == pytest.approx(action.velocity)
