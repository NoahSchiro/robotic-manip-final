import numpy as np
import pytest

from src.action import Action
from src.geometry import SurfaceNormal
from src.iiwa import MOVE_TIME, STANDOFF
from src.object import Object, Rod
from src.recording import push_window


def make_push() -> tuple[Object, Action]:
    obj = Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, 0.2))
    site = SurfaceNormal(
        position=np.array([0.07, 0.0, 0.015]), normal=np.array([-1.0, 0.0, 0.0])
    )
    return obj, Action(contact=site, velocity=0.1, time=0.4)


def test_push_window_opens_at_the_standoff_and_closes_after_settling() -> None:
    obj, action = make_push()
    window = push_window(obj, action, settle_time=0.1, trajectory_start=3.0)
    assert window.start == pytest.approx(3.0 + 2 * MOVE_TIME)
    assert window.push_end == pytest.approx(
        window.start + STANDOFF / action.velocity + action.time
    )
    assert window.end == pytest.approx(window.push_end + 0.1)
