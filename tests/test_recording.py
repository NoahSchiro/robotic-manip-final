import numpy as np
import pytest
from pydrake.all import PiecewisePolynomial

from src.world.iiwa import LEG_DWELL, LEG_PUSH, LEG_TRANSFER
from src.world.recording import push_window

# Breaks: start, then one end per push_maneuver leg, then the hold's end.
SEGMENT_ENDS = [0.0, 1.0, 1.5, 1.8, 2.8, 2.9, 3.3, 3.4]


def make_trajectory() -> PiecewisePolynomial:
    return PiecewisePolynomial.FirstOrderHold(
        SEGMENT_ENDS, np.zeros((1, len(SEGMENT_ENDS))).tolist()
    )


def test_push_window_reads_the_trajectory_breaks_on_the_sim_clock() -> None:
    breaks = np.asarray(SEGMENT_ENDS) + 10.0
    window = push_window(make_trajectory(), settle_time=0.3, trajectory_start=10.0)
    assert breaks[LEG_TRANSFER + 1] == pytest.approx(window.transfer_end)
    assert breaks[LEG_DWELL + 1] == pytest.approx(window.start)
    assert breaks[LEG_PUSH + 1] == pytest.approx(window.push_end)
    assert window.transfer_end == pytest.approx(11.0)
    assert window.start == pytest.approx(11.8)
    assert window.push_end == pytest.approx(12.8)
    assert window.end == pytest.approx(13.1)
    assert window.action_end == pytest.approx(13.4)
