import numpy as np
import pytest
from pydrake.all import PiecewisePolynomial, RotationMatrix

from src.action import Action
from src.geometry import SurfaceNormal
from src.iiwa import (
    MOVE_TIME,
    STANDOFF,
    Pusher,
    chain_from_parked,
    clear_sites,
    hold_trajectory,
    hover_tip_pose,
    push_maneuver,
    transit_time,
)
from src.object import Object, Rod

LIMITS = np.array([1.0, 2.0])
SITES_PER_M = 200.0


def make_object() -> Object:
    return Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, 0.2))


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
    hover = hover_tip_pose(obj)
    maneuver = push_maneuver(obj, action, hover)
    over, lowered, push_end = [pose.translation() for pose, _ in maneuver[:3]]

    contact = obj.x_wo() @ action.contact.position
    normal = obj.x_wo().rotation() @ action.contact.normal
    assert lowered == pytest.approx(contact - STANDOFF * normal)
    assert over[:2] == pytest.approx(lowered[:2])
    assert over[2] == pytest.approx(hover.translation()[2])
    assert push_end == pytest.approx(contact + action.velocity * action.time * normal)
    assert maneuver[2][1] == pytest.approx(STANDOFF / action.velocity + action.time)


def test_retreat_backs_off_a_little_at_push_speed_then_lifts_straight_up() -> None:
    obj = make_object()
    obj.pose = (0.5, 0.0, 0.7)
    action = make_action(velocity=0.1, contact_time=0.4)
    hover = hover_tip_pose(obj)
    maneuver = push_maneuver(obj, action, hover)
    push_end, backed_off, raised = [pose.translation() for pose, _ in maneuver[2:]]

    normal = obj.x_wo().rotation() @ action.contact.normal
    assert backed_off == pytest.approx(push_end - STANDOFF * normal)
    assert np.linalg.norm(backed_off - push_end) / maneuver[3][1] == pytest.approx(
        action.velocity
    )
    assert raised[:2] == pytest.approx(backed_off[:2])
    assert raised[2] == pytest.approx(hover.translation()[2])
    assert len(maneuver) == 5


def test_every_maneuver_waypoint_keeps_the_tool_pointing_down() -> None:
    obj = make_object()
    hover = hover_tip_pose(obj)
    for pose, duration in push_maneuver(obj, make_action(), hover):
        assert pose.rotation().IsNearlyEqualTo(
            RotationMatrix.MakeXRotation(np.pi), 1e-12
        )
        assert duration > 0.0


def test_small_moves_take_the_normal_transit_time() -> None:
    q_from, q_to = np.zeros(2), np.array([0.1, -0.1])
    assert transit_time(q_from, q_to, LIMITS) == MOVE_TIME


def test_large_moves_take_as_long_as_the_slowest_joint_needs() -> None:
    q_from, q_to = np.zeros(2), np.array([0.5, -2.0])
    # Joint 0 needs 0.5 s at 1 rad/s; joint 1 needs 1 s at 2 rad/s.
    assert transit_time(q_from, q_to, LIMITS) == pytest.approx(1.0)


def test_hold_trajectory_stays_put() -> None:
    q = np.array([0.3, -0.4])
    held = hold_trajectory(q)
    assert held.value(0.0).ravel() == pytest.approx(q)
    assert held.value(held.end_time()).ravel() == pytest.approx(q)


def make_plan() -> PiecewisePolynomial:
    return PiecewisePolynomial.FirstOrderHold(
        [0.0, 1.0, 2.0], [[0.0, 1.0, 2.0], [0.0, -1.0, -2.0]]
    )


def test_chain_moves_to_the_plan_then_follows_it_on_the_sim_clock() -> None:
    q_parked = np.array([5.0, 5.0])
    plan = make_plan()
    start, transit = 10.0, 0.5
    chained = chain_from_parked(q_parked, plan, start, transit)

    assert chained.start_time() == start
    assert chained.end_time() == pytest.approx(start + transit + plan.end_time())
    assert chained.value(start).ravel() == pytest.approx(q_parked)
    assert chained.value(start + transit).ravel() == pytest.approx([0.0, 0.0])
    assert chained.value(start + transit + 1.0).ravel() == pytest.approx([1.0, -1.0])
    assert chained.value(chained.end_time()).ravel() == pytest.approx([2.0, -2.0])


def test_chaining_leaves_the_callers_plan_untouched() -> None:
    plan = make_plan()
    chain_from_parked(np.zeros(2), plan, 10.0, 0.5)
    assert (plan.start_time(), plan.end_time()) == (0.0, 2.0)
