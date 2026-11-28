import numpy as np
import pytest
from pydrake.all import PiecewisePolynomial

from src.world.action import Action
from src.world.geometry import SurfaceNormal
from src.world.iiwa import (
    DWELL_TIME,
    FREE_SPACE_LEGS,
    HOLD_TIME,
    JOINT_SPEED_FRACTION,
    LEG_DWELL,
    LEG_LIFT,
    LEG_LOWER,
    LEG_PUSH,
    LEG_RELEASE,
    LEG_TRANSFER,
    MOVE_TIME,
    Q_HOME,
    RELEASE_DISTANCE,
    RELEASE_SPEED,
    STANDOFF,
    TOOL_DOWN,
    Pusher,
    build_iiwa_pusher_plant,
    clear_sites,
    free_space_duration,
    hold_trajectory,
    make_push_trajectory,
    push_maneuver,
    shift_trajectory,
    transfer_height,
)
from src.world.object import Object, Rod
from src.world.recording import push_window

LIMITS = np.array([1.0, 2.0])
SITES_PER_M = 200.0


def make_object() -> Object:
    return Object(rod=Rod(0.0, 0.0, 0.2))


def key(site: SurfaceNormal) -> tuple[float, float]:
    x, y = np.round(site.position[:2], 6)
    return float(x), float(y)


def test_clear_sites_removes_exactly_the_t_inner_corners() -> None:
    obj = make_object()
    sites = obj.surface_normals(SITES_PER_M)
    kept = clear_sites(obj.parts, sites, Pusher().radius, 1e-3)
    removed = {key(s) for s in sites} - {key(s) for s in kept}
    assert (len(sites), len(kept)) == (112, 102)
    assert removed == {
        (-0.02, -0.035),
        (-0.02, -0.03),
        (-0.02, -0.025),
        (-0.015, -0.025),
        (-0.01, -0.025),
        (-0.02, 0.025),
        (-0.02, 0.03),
        (-0.02, 0.035),
        (-0.015, 0.025),
        (-0.01, 0.025),
    }


def test_a_larger_gap_only_removes_more_sites() -> None:
    obj = make_object()
    sites = obj.surface_normals(SITES_PER_M)
    radius = Pusher().radius
    narrow = {key(s) for s in clear_sites(obj.parts, sites, radius, 1e-3)}
    wide = {key(s) for s in clear_sites(obj.parts, sites, radius, 2e-3)}
    assert wide <= narrow


def make_action(velocity: float = 0.05, contact_time: float = 0.2) -> Action:
    # The stem's +x end, pushed toward -x.
    site = SurfaceNormal(
        position=np.array([0.07, 0.0, 0.015]), normal=np.array([-1.0, 0.0, 0.0])
    )
    return Action(contact=site, velocity=velocity, time=contact_time)


def test_push_maneuver_starts_over_the_standoff_then_pushes_along_the_normal() -> None:
    obj = make_object()
    obj.pose = (0.5, 0.0, 0.0)
    action = make_action()
    maneuver = push_maneuver(obj.x_wo(), action, transfer_height(obj))
    over, lowered, dwell, push_end = [
        pose.translation() for pose, _ in maneuver[: LEG_PUSH + 1]
    ]

    contact = obj.x_wo() @ action.contact.position
    normal = obj.x_wo().rotation() @ action.contact.normal
    assert lowered == pytest.approx(contact - STANDOFF * normal)
    assert over[:2] == pytest.approx(lowered[:2])
    assert over[2] == pytest.approx(transfer_height(obj))
    assert dwell == pytest.approx(lowered)
    assert maneuver[LEG_DWELL][1] == pytest.approx(DWELL_TIME)
    assert push_end == pytest.approx(contact + action.velocity * action.time * normal)
    assert maneuver[LEG_PUSH][1] == pytest.approx(
        STANDOFF / action.velocity + action.time
    )


def test_release_is_one_centimeter_at_five_centimeters_per_second() -> None:
    obj = make_object()
    obj.pose = (0.5, 0.0, 0.7)
    action = make_action(velocity=0.1, contact_time=0.4)
    maneuver = push_maneuver(obj.x_wo(), action, transfer_height(obj))
    push_end, released, raised = [pose.translation() for pose, _ in maneuver[LEG_PUSH:]]

    normal = obj.x_wo().rotation() @ action.contact.normal
    assert RELEASE_DISTANCE == pytest.approx(0.01)
    assert RELEASE_SPEED == pytest.approx(0.05)
    assert released == pytest.approx(push_end - RELEASE_DISTANCE * normal)
    assert np.linalg.norm(released - push_end) == pytest.approx(0.01)
    assert maneuver[LEG_RELEASE][1] == pytest.approx(0.01 / 0.05)
    assert raised[:2] == pytest.approx(released[:2])
    assert raised[2] == pytest.approx(transfer_height(obj))
    assert maneuver[LEG_LIFT][1] == pytest.approx(MOVE_TIME)
    assert len(maneuver) == 6


def test_every_maneuver_waypoint_keeps_the_tool_pointing_down() -> None:
    obj = make_object()
    for pose, duration in push_maneuver(
        obj.x_wo(), make_action(), transfer_height(obj)
    ):
        assert pose.rotation().IsNearlyEqualTo(TOOL_DOWN, 1e-12)
        assert duration > 0.0


def test_free_space_leg_keeps_its_nominal_time_when_joints_are_slow_enough() -> None:
    q_from, q_to = np.zeros(2), np.array([0.1, -0.1])
    assert free_space_duration(q_from, q_to, MOVE_TIME, LIMITS) == MOVE_TIME


def test_free_space_leg_stretches_to_half_the_slowest_joint_limit() -> None:
    q_from, q_to = np.zeros(2), np.array([0.5, -2.0])
    # Joint 0 needs 0.5 / (0.5 * 1) = 1 s; joint 1 needs 2 / (0.5 * 2) = 2 s.
    assert free_space_duration(q_from, q_to, MOVE_TIME, LIMITS) == pytest.approx(2.0)


def test_hold_trajectory_stays_put() -> None:
    q = np.array([0.3, -0.4])
    held = hold_trajectory(q)
    assert held.value(0.0).ravel() == pytest.approx(q)
    assert held.value(held.end_time()).ravel() == pytest.approx(q)


def make_plan() -> PiecewisePolynomial:
    return PiecewisePolynomial.FirstOrderHold(
        [0.0, 1.0, 2.0], [[0.0, 1.0, 2.0], [0.0, -1.0, -2.0]]
    )


def test_shift_starts_the_plan_on_the_sim_clock_without_changing_it() -> None:
    plan = make_plan()
    shifted = shift_trajectory(plan, 10.0)
    assert shifted.start_time() == 10.0
    assert shifted.end_time() == pytest.approx(12.0)
    assert shifted.value(10.0).ravel() == pytest.approx([0.0, 0.0])
    assert shifted.value(11.0).ravel() == pytest.approx([1.0, -1.0])
    assert shifted.value(12.0).ravel() == pytest.approx([2.0, -2.0])
    assert (plan.start_time(), plan.end_time()) == (0.0, 2.0)


def test_planned_free_space_legs_stay_under_the_joint_speed_fraction() -> None:
    pusher = Pusher()
    plant = build_iiwa_pusher_plant(pusher)
    obj = make_object()
    obj.pose = (0.55, 0.0, 0.3)
    action = make_action(velocity=0.1, contact_time=0.3)
    plan = make_push_trajectory(plant, pusher, obj, action, Q_HOME)

    # Breaks: start, then one per leg in LEG_* order, then the hold's end.
    breaks = np.asarray(plan.get_segment_times())
    assert breaks.size == 8
    assert breaks[-1] == pytest.approx(plan.end_time())
    durations = np.diff(breaks)
    nominal = [
        d for _pose, d in push_maneuver(obj.x_wo(), action, transfer_height(obj))
    ]
    assert durations[LEG_DWELL] == pytest.approx(DWELL_TIME)
    assert durations[LEG_PUSH] == pytest.approx(nominal[LEG_PUSH])
    assert durations[LEG_RELEASE] == pytest.approx(nominal[LEG_RELEASE])
    assert durations[-1] == pytest.approx(HOLD_TIME)

    limits = plant.GetVelocityUpperLimits()
    for leg in FREE_SPACE_LEGS:
        assert durations[leg] >= nominal[leg] - 1e-12
        move = plan.value(breaks[leg + 1]).ravel() - plan.value(breaks[leg]).ravel()
        speed = np.abs(move) / durations[leg]
        assert np.all(speed <= JOINT_SPEED_FRACTION * limits + 1e-9)
    assert FREE_SPACE_LEGS == (LEG_TRANSFER, LEG_LOWER, LEG_LIFT)


def test_window_opens_after_the_dwell_with_the_tip_at_the_standoff() -> None:
    pusher = Pusher()
    plant = build_iiwa_pusher_plant(pusher)
    obj = make_object()
    obj.pose = (0.55, 0.0, 0.3)
    action = make_action(velocity=0.1, contact_time=0.3)
    plan = make_push_trajectory(plant, pusher, obj, action, Q_HOME)
    window = push_window(plan, settle_time=0.3, trajectory_start=0.0)

    dwell_start = window.start - DWELL_TIME
    assert plan.value(dwell_start).ravel() == pytest.approx(
        plan.value(window.start).ravel(), abs=1e-6
    )
    context = plant.CreateDefaultContext()
    plant.SetPositions(context, plan.value(window.start).ravel())
    stick = plant.GetFrameByName(pusher.body, plant.GetModelInstanceByName(pusher.name))
    tip = plant.CalcRelativeTransform(context, plant.world_frame(), stick) @ (
        pusher.tip_point
    )
    contact = obj.x_wo() @ action.contact.position
    normal = obj.x_wo().rotation() @ action.contact.normal
    assert tip == pytest.approx(contact - STANDOFF * normal, abs=1e-4)
    assert window.push_end - window.start == pytest.approx(
        STANDOFF / action.velocity + action.time
    )
