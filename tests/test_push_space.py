"""Push-space geometry, sampling and reach, checked against the T's outline and
the env's reach mask (no physics)."""

import numpy as np
import pytest

from src.geometry import boundary_edges
from src.iiwa import Pusher, clear_sites
from src.object import Object, Rod
from src.push_space import (
    DURATION_RANGE,
    INNER_CORNER_INSET,
    MAX_PUSH_DISTANCE,
    OUTER_CORNER_INSET,
    SPEED_RANGE,
    PushSpace,
    make_push_space,
    reachable,
    reachable_mask,
)

SPACING = 0.012  # m, the MPC grid spacing
GRID_SIZE = 396  # candidates on the T at SPACING with 3 speeds and 3 distances


@pytest.fixture(scope="module")
def obj() -> Object:
    return Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, 0.2))


@pytest.fixture(scope="module")
def space(obj: Object) -> PushSpace:
    return make_push_space(obj.parts, obj.height)


def segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    t = np.clip(np.dot(point - start, direction) / np.dot(direction, direction), 0, 1)
    return float(np.linalg.norm(point - (start + t * direction)))


def nearest_face(space: PushSpace, xy: np.ndarray) -> tuple[int, float]:
    distances = [segment_distance(xy, f.start, f.end) for f in space.faces]
    face = int(np.argmin(distances))
    return face, distances[face]


def test_faces_are_the_eight_sides_of_the_t(obj: Object, space: PushSpace) -> None:
    assert len(space.faces) == 8
    perimeter = sum(
        float(np.hypot(x1 - x0, y1 - y0))
        for (x0, y0), (x1, y1) in boundary_edges(obj.parts)
    )
    assert sum(f.length for f in space.faces) == pytest.approx(perimeter)
    for f in space.faces:
        assert f.normal[2] == 0.0
        assert np.linalg.norm(f.normal) == pytest.approx(1.0)


def test_face_normals_agree_with_surface_normals(obj: Object, space: PushSpace) -> None:
    for site in obj.surface_normals(200.0):
        face, distance = nearest_face(space, site.position[:2])
        assert distance < 1e-9
        assert np.dot(space.faces[face].normal, site.normal) == pytest.approx(1.0)
        assert site.position[2] == pytest.approx(space.faces[face].height)


def test_spans_are_inset_more_at_inner_corners(space: PushSpace) -> None:
    assert space.usable_spans().sum() == pytest.approx(0.452)
    insets = sorted(
        inset for face in space.faces for inset in (face.start_inset, face.end_inset)
    )
    assert insets == [OUTER_CORNER_INSET] * 12 + [INNER_CORNER_INSET] * 4
    for face in range(len(space.faces)):
        f = space.faces[face]
        start = space.contact(face, 0.0).position[:2]
        end = space.contact(face, 1.0).position[:2]
        assert np.linalg.norm(start - f.start) == pytest.approx(f.start_inset)
        assert np.linalg.norm(end - f.end) == pytest.approx(f.end_inset)


def test_contacts_at_span_ends_clear_the_inner_corners(
    obj: Object, space: PushSpace
) -> None:
    for face in range(len(space.faces)):
        for s in (0.0, 1.0):
            kept = clear_sites(
                obj.parts, [space.contact(face, s)], Pusher().radius, 1e-3
            )
            assert len(kept) == 1, (face, s)


def test_every_grid_contact_clears_the_inner_corners(
    obj: Object, space: PushSpace
) -> None:
    contacts = [action.contact for action in space.grid(SPACING, 1, 1)]
    assert len(clear_sites(obj.parts, contacts, Pusher().radius, 1e-3)) == len(contacts)


def test_sample_stays_within_ranges_spans_and_the_distance_cap(
    space: PushSpace,
) -> None:
    rng = np.random.default_rng(0)
    draws = [space.sample(rng) for _ in range(4000)]
    velocities = np.array([a.velocity for a in draws])
    durations = np.array([a.time for a in draws])
    assert velocities.min() >= SPEED_RANGE[0] and velocities.max() <= SPEED_RANGE[1]
    assert durations.min() >= DURATION_RANGE[0] and durations.max() <= DURATION_RANGE[1]
    assert np.all(velocities * durations <= MAX_PUSH_DISTANCE)
    assert velocities.min() < SPEED_RANGE[0] + 0.01
    assert velocities.max() > SPEED_RANGE[1] - 0.01
    assert durations.min() < DURATION_RANGE[0] + 0.02
    assert durations.max() > DURATION_RANGE[1] - 0.02

    counts = np.zeros(len(space.faces))
    for action in draws:
        face, distance = nearest_face(space, action.contact.position[:2])
        assert distance < 1e-9
        counts[face] += 1
    expected = space.usable_spans() / space.usable_spans().sum()
    np.testing.assert_allclose(counts / len(draws), expected, atol=0.03)


def test_grid_has_expected_size_and_speed_endpoints(space: PushSpace) -> None:
    actions = space.grid(SPACING, 3, 3)
    assert len(actions) == GRID_SIZE
    speeds = sorted({round(a.velocity, 9) for a in actions})
    assert speeds == pytest.approx(
        [SPEED_RANGE[0], np.mean(SPEED_RANGE), SPEED_RANGE[1]]
    )


def test_grid_distances_run_from_min_duration_to_the_distance_cap(
    space: PushSpace,
) -> None:
    actions = space.grid(SPACING, 3, 3)
    durations = np.array([a.time for a in actions])
    distances = np.array([a.velocity * a.time for a in actions])
    assert durations.min() == pytest.approx(DURATION_RANGE[0])
    assert durations.max() == pytest.approx(DURATION_RANGE[1])
    assert distances.max() == pytest.approx(MAX_PUSH_DISTANCE)
    assert np.all(distances <= MAX_PUSH_DISTANCE + 1e-12)
    slowest = [a for a in actions if a.velocity == SPEED_RANGE[0]]
    assert min(a.time for a in slowest) == pytest.approx(DURATION_RANGE[0])
    assert max(a.time for a in slowest) == pytest.approx(DURATION_RANGE[1])


def test_grid_includes_both_span_ends_of_every_face(space: PushSpace) -> None:
    positions = {
        tuple(np.round(a.contact.position[:2], 9)) for a in space.grid(SPACING, 1, 1)
    }
    for face in range(len(space.faces)):
        for s in (0.0, 1.0):
            assert tuple(np.round(space.contact(face, s).position[:2], 9)) in positions


def test_vectorized_reach_agrees_with_single_pushes(space: PushSpace) -> None:
    rng = np.random.default_rng(1)
    verdicts = []
    for _ in range(100):
        pose = np.array(
            [rng.uniform(0.3, 0.8), rng.uniform(-0.3, 0.3), rng.uniform(-np.pi, np.pi)]
        )
        actions = [space.sample(rng) for _ in range(5)]
        mask = reachable_mask(
            pose,
            np.array([a.contact.position[:2] for a in actions]),
            np.array([a.contact.normal[:2] for a in actions]),
            np.array([a.velocity * a.time for a in actions]),
        )
        single = [reachable(pose, a) for a in actions]
        assert mask.tolist() == single
        verdicts += single
    assert any(verdicts) and not all(verdicts)
