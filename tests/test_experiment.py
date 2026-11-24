"""Placement tasks: where start and goal are drawn and where the block must stay."""

import numpy as np

from src.constants import (
    FAR_BAND_HALF_ANGLE,
    FAR_BAND_RADII,
    FAR_GOAL_DISTANCES,
    GOAL_RADIUS,
    START_RADIUS,
    WORKSPACE_CENTER,
)
from src.experiment import TASKS, episode_conditions, episode_goal, out_of_bounds

SEEDS = range(200)


def polar(xy: np.ndarray) -> tuple[float, float]:
    return float(np.hypot(xy[0], xy[1])), float(np.arctan2(xy[1], xy[0]))


def test_near_task_draws_within_the_workspace_disks() -> None:
    center = np.array(WORKSPACE_CENTER)
    for seed in SEEDS:
        _, start, _ = episode_conditions(seed)
        goal = np.array(episode_goal(seed, start))
        assert np.linalg.norm(start[:2] - center) <= START_RADIUS
        assert np.linalg.norm(goal[:2] - center) <= GOAL_RADIUS


def test_far_task_draws_on_the_band_the_set_distance_apart() -> None:
    far = TASKS["far"]
    for seed in SEEDS:
        _, start, _ = episode_conditions(seed, far)
        goal = np.array(episode_goal(seed, start, far))
        for xy in (start[:2], goal[:2]):
            radius, angle = polar(xy)
            assert FAR_BAND_RADII[0] <= radius <= FAR_BAND_RADII[1]
            assert abs(angle) <= FAR_BAND_HALF_ANGLE
            assert not out_of_bounds(xy, far)
        distance = np.linalg.norm(goal[:2] - start[:2])
        assert FAR_GOAL_DISTANCES[0] <= distance <= FAR_GOAL_DISTANCES[1]


def test_tasks_share_each_seeds_true_parameters() -> None:
    for seed in SEEDS:
        near_eta, near_start, _ = episode_conditions(seed)
        far_eta, far_start, _ = episode_conditions(seed, TASKS["far"])
        assert near_eta == far_eta
        assert not np.allclose(near_start, far_start)


def test_far_task_is_deterministic_per_seed() -> None:
    far = TASKS["far"]
    _, start_a, _ = episode_conditions(7, far)
    _, start_b, _ = episode_conditions(7, far)
    np.testing.assert_array_equal(start_a, start_b)
    assert episode_goal(7, start_a, far) == episode_goal(7, start_b, far)


def test_far_bounds_are_the_sector_around_the_band() -> None:
    far = TASKS["far"]
    assert not out_of_bounds(np.array([0.57, 0.0, 0.0]), far)
    assert out_of_bounds(np.array([0.35, 0.0, 0.0]), far)  # too close to the base
    assert out_of_bounds(np.array([0.80, 0.0, 0.0]), far)  # out of reach
    assert out_of_bounds(np.array([0.0, 0.57, 0.0]), far)  # 90 deg to the side
    # On the band but far outside the near task's workspace disk.
    side = np.array([0.30, 0.45, 0.0])
    assert not out_of_bounds(side, far)
    assert out_of_bounds(side)
