"""Plan parsing, the plan's push sequencing, and the motion metrics on hand-made
inputs.  No physics."""

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from pydrake.all import RigidTransform, RotationMatrix

from scripts.diagnose_motion import (
    PlanSequence,
    PushSpec,
    commanded_push_line,
    joint_speed_ratio,
    load_plan,
    planar_drift,
    release_clearances,
    release_summary,
    spec_action,
    tracking_errors,
)
from src.control.probing import ProbingState
from src.world.action import Action
from src.world.geometry import SurfaceNormal
from src.world.iiwa import STANDOFF
from src.world.push_space import make_push_space

BLOCK = [(0.1, 0.05, 0.0, 0.0)]
BLOCK_HEIGHT = 0.05


def dummy_state() -> ProbingState:
    """The sequence ignores the state, so a placeholder stands in for it."""
    return cast(ProbingState, object())


def make_action(velocity: float) -> Action:
    contact = SurfaceNormal(
        position=np.array([0.05, 0.0, BLOCK_HEIGHT / 2]),
        normal=np.array([-1.0, 0.0, 0.0]),
    )
    return Action(contact=contact, velocity=velocity, time=0.2)


def test_load_plan_parses_fields_and_defaults_the_rod(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "friction": 0.4,
                "yaw": 0.3,
                "pushes": [
                    {"face": 2, "s": 0.25, "speed": 0.06, "duration": 0.3},
                    {"face": 7, "s": 1.0, "speed": 0.12, "duration": 0.1},
                ],
            }
        )
    )
    plan = load_plan(path)
    assert plan.friction == pytest.approx(0.4)
    assert plan.yaw == pytest.approx(0.3)
    assert plan.rod == (0.0, 0.0)
    assert plan.pushes == (
        PushSpec(face=2, s=0.25, speed=0.06, duration=0.3),
        PushSpec(face=7, s=1.0, speed=0.12, duration=0.1),
    )


def test_load_plan_reads_an_explicit_rod(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps({"friction": 0.5, "yaw": 0.0, "rod": [0.01, -0.02], "pushes": []})
    )
    assert load_plan(path).rod == (0.01, -0.02)


def test_spec_action_rejects_a_face_out_of_range() -> None:
    space = make_push_space(BLOCK, BLOCK_HEIGHT)
    with pytest.raises(ValueError):
        spec_action(
            space, PushSpec(face=len(space.faces), s=0.5, speed=0.05, duration=0.2)
        )


def test_plan_sequence_hands_out_pushes_in_order_and_skips_failed_ones() -> None:
    actions = [make_action(0.05), make_action(0.06), make_action(0.07)]
    sequence = PlanSequence(actions)
    state = dummy_state()

    assert sequence.select(state, []) is actions[0]
    assert sequence.executed() == 0
    assert sequence.select(state, []) is actions[1]
    assert sequence.select(state, [actions[1]]) is actions[2]
    assert sequence.ik_failed == [1]
    assert sequence.executed() == 2
    assert sequence.select(state, []) == "done"
    assert sequence.unrun() == []


def test_plan_sequence_reports_pushes_not_reached() -> None:
    actions = [make_action(0.05), make_action(0.06)]
    sequence = PlanSequence(actions)
    state = dummy_state()

    assert sequence.select(state, []) is actions[0]
    assert sequence.executed() == 0
    assert sequence.select(state, []) is actions[1]
    assert sequence.unrun() == [1]


def test_tracking_is_zero_on_the_commanded_line() -> None:
    times = np.linspace(0.0, 1.0, 11)
    approach = np.array([0.0, 0.0, 0.02])
    direction = np.array([1.0, 0.0, 0.0])
    tips = approach + 0.1 * times[:, None] * direction
    errors = tracking_errors(times, tips, approach, direction, 0.1, 1.0)
    assert errors["max_lateral_m"] == pytest.approx(0.0, abs=1e-12)
    assert errors["max_abs_z_m"] == pytest.approx(0.0, abs=1e-12)
    assert errors["z_at_push_end_m"] == pytest.approx(0.0, abs=1e-12)
    assert errors["z_rms_m"] == pytest.approx(0.0, abs=1e-12)
    assert errors["along_track_lag_m"] == pytest.approx(0.0, abs=1e-12)


def test_tracking_reports_lateral_height_and_lag_errors() -> None:
    times = np.linspace(0.0, 2.0, 21)
    approach = np.zeros(3)
    direction = np.array([1.0, 0.0, 0.0])
    # Half the commanded speed in x, a 3 mm offset in y, and 4 mm high in z.
    tips = np.column_stack(
        [0.05 * times, 0.003 * np.ones_like(times), 0.004 * np.ones_like(times)]
    )
    errors = tracking_errors(times, tips, approach, direction, 0.1, 2.0)
    assert errors["max_lateral_m"] == pytest.approx(0.003)
    assert errors["max_abs_z_m"] == pytest.approx(0.004)
    assert errors["z_at_push_end_m"] == pytest.approx(0.004)
    assert errors["z_rms_m"] == pytest.approx(0.004)
    # Commanded travel 0.2 m, actual 0.1 m.
    assert errors["along_track_lag_m"] == pytest.approx(0.1)


def test_tracking_separates_a_lowering_transient_from_an_in_push_dip() -> None:
    times = np.linspace(0.0, 2.0, 21)
    approach = np.array([0.0, 0.0, 0.015])
    direction = np.array([1.0, 0.0, 0.0])
    commanded = approach + 0.1 * times[:, None] * direction

    transient = commanded.copy()
    transient[0, 2] += 0.009  # tip high at window start, then on the line
    early = tracking_errors(times, transient, approach, direction, 0.1, 2.0)
    assert early["max_abs_z_m"] == pytest.approx(0.009)
    assert early["max_abs_z_time_s"] == pytest.approx(0.0)
    assert early["z_at_push_end_m"] == pytest.approx(0.0)

    dip = commanded.copy()
    dip[10, 2] -= 0.005  # 5 mm low at t = 1 s, back on the line by push end
    mid = tracking_errors(times, dip, approach, direction, 0.1, 2.0)
    assert mid["max_abs_z_m"] == pytest.approx(0.005)
    assert mid["max_abs_z_time_s"] == pytest.approx(1.0)
    assert mid["z_at_push_end_m"] == pytest.approx(0.0)
    assert mid["z_rms_m"] == pytest.approx(0.005 * np.sqrt(1 / 21))


def test_tracking_ignores_samples_after_the_push_segment() -> None:
    times = np.array([0.0, 1.0, 2.0])
    approach = np.zeros(3)
    direction = np.array([1.0, 0.0, 0.0])
    tips = np.column_stack([0.1 * times, np.zeros(3), np.zeros(3)])
    tips[2, 1] = 0.5  # after the push segment's end
    errors = tracking_errors(times, tips, approach, direction, 0.1, 1.0)
    assert errors["max_lateral_m"] == pytest.approx(0.0)


def test_commanded_line_starts_a_standoff_short_of_the_contact() -> None:
    block_start = RigidTransform(
        RotationMatrix.MakeZRotation(np.pi / 2), np.array([1.0, 0.0, 0.0])
    )
    contact = SurfaceNormal(
        position=np.array([-0.05, 0.0, 0.025]), normal=np.array([1.0, 0.0, 0.0])
    )
    action = Action(contact=contact, velocity=0.1, time=0.2)
    approach, direction = commanded_push_line(block_start, action)
    # Body +x maps to world +y after the block's 90 degree yaw.
    assert direction == pytest.approx([0.0, 1.0, 0.0])
    assert approach == pytest.approx([1.0, -0.05 - STANDOFF, 0.025])


def test_joint_speed_ratio_finds_the_worst_joint_and_sample() -> None:
    velocities = np.array(
        [
            [0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0],
        ]
    )
    limits = np.array([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 4.0])
    ratio, joint = joint_speed_ratio(velocities, limits)
    assert ratio == pytest.approx(0.75)
    assert joint == 6


def test_release_clearance_is_measured_from_the_rotated_block_edge() -> None:
    # Block at (1, 0) yawed 90 degrees: its body +x points along world +y.
    block = np.array([[1.0, 0.0, np.pi / 2]])
    tips = np.array([[1.0, 0.07, 0.01]])
    # The tip sits 2 cm outside the 10 cm-wide block's edge; minus the 1 cm stick.
    clearance = release_clearances(tips, block, BLOCK, 0.01)
    assert clearance == pytest.approx([0.01])


def test_release_summary_tracks_first_clear_and_recontact() -> None:
    times = np.array([0.0, 0.001, 0.002, 0.003, 0.004, 0.005])
    # Clearances by sample: -0.01 (inside), -0.005, +0.01 (clear), -0.002
    # (back in contact), +0.03 (tip above the top), +0.04 (clear, low again).
    tips = np.array(
        [
            [0.0, 0.0, 0.01],
            [0.055, 0.0, 0.01],
            [0.07, 0.0, 0.01],
            [0.058, 0.0, 0.01],
            [0.09, 0.0, 0.2],
            [0.1, 0.0, 0.01],
        ]
    )
    block = np.zeros((6, 3))
    summary = release_summary(times, tips, block, BLOCK, BLOCK_HEIGHT, 0.01)
    assert summary["min_clearance_m"] == pytest.approx(-0.01)
    assert summary["min_clearance_time_s"] == pytest.approx(0.0)
    assert summary["first_clear_time_s"] == pytest.approx(0.002)
    assert summary["recontact"] is True
    assert summary["min_clearance_after_clear_m"] == pytest.approx(-0.002)


def test_release_summary_counts_the_tip_rising_above_the_top_as_clear() -> None:
    times = np.array([0.0, 0.001, 0.002])
    tips = np.array([[0.0, 0.0, 0.01], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3]])
    block = np.zeros((3, 3))
    summary = release_summary(times, tips, block, BLOCK, BLOCK_HEIGHT, 0.01)
    assert summary["first_clear_time_s"] == pytest.approx(0.001)
    assert summary["recontact"] is False
    assert summary["min_clearance_after_clear_m"] is None


def test_release_summary_without_clearing_has_no_first_clear() -> None:
    times = np.array([0.0, 0.001])
    tips = np.array([[0.0, 0.0, 0.01], [0.01, 0.0, 0.01]])
    block = np.zeros((2, 3))
    summary = release_summary(times, tips, block, BLOCK, BLOCK_HEIGHT, 0.01)
    assert summary["min_clearance_m"] == pytest.approx(-0.01)
    assert summary["first_clear_time_s"] is None
    assert summary["recontact"] is False


def test_planar_drift_gives_translation_and_wrapped_yaw_change() -> None:
    before = np.zeros(6)
    after = np.zeros(6)
    after[:3] = [0.003, 0.004, 0.1]
    translation, yaw = planar_drift(before, after)
    assert translation == pytest.approx(0.005)
    assert yaw == pytest.approx(0.1)

    wrapped_before = np.array([0.0, 0.0, 3.1, 0.0, 0.0, 0.0])
    wrapped_after = np.array([0.0, 0.0, -3.1, 0.0, 0.0, 0.0])
    _, wrapped_yaw = planar_drift(wrapped_before, wrapped_after)
    assert wrapped_yaw == pytest.approx(2 * np.pi - 6.2)
