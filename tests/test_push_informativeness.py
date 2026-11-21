"""Push-informativeness helpers (no physics)."""

import numpy as np

from scripts.push_informativeness import (
    contacts_on_grid,
    feasible_combos,
    separation,
    summarize,
)


def test_separation_is_mahalanobis_with_wrapped_yaw() -> None:
    r = np.diag([1e-6, 1e-6, 1e-4, 1e-5, 1e-5, 1e-3])
    base = np.zeros(6)
    moved = np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert separation(base, moved, r) == np.float64(1.0)
    across_pi = base.copy()
    across_pi[2] = np.pi - 0.005
    moved_yaw = base.copy()
    moved_yaw[2] = -np.pi + 0.005
    assert np.isclose(separation(across_pi, moved_yaw, r), 0.01 / 1e-2)


def test_summary_reports_each_distance_and_speed() -> None:
    separations = np.array([[0.5, 2.0], [1.5, 4.0]])
    (slow, fast) = summarize(separations, [(0.015, 0.03), (0.015, 0.15)], 0.2, "mu")
    assert slow.median == 1.0 and slow.share_above_1 == 0.5
    assert fast.share_above_1 == 1.0 and fast.speed_m_per_s == 0.15


def test_contacts_are_the_distinct_grid_positions() -> None:
    contacts = contacts_on_grid(0.012)
    assert len(contacts) == 44
    assert len({c.position.tobytes() for c in contacts}) == 44


def test_combos_keep_contact_times_in_range() -> None:
    combos = feasible_combos((0.1, 1.0))
    assert (0.09, 0.06) not in combos  # 1.5 s of contact
    assert (0.015, 0.15) in combos  # exactly the 0.1 s minimum
    assert all(0.1 - 1e-9 <= d / v <= 1.0 + 1e-9 for d, v in combos)
