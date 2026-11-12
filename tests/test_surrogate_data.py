from pathlib import Path

import numpy as np
import pytest

from src.action import Action
from src.geometry import SurfaceNormal
from src.iiwa import HOLD_TIME, MOVE_TIME, STANDOFF, hover_tip_pose
from src.object import Object, Rod
from src.surrogate import nominal_action_tip_path
from src.surrogate_data import assign_split
from src.train_surrogate import dataset_arrays, world_frame_errors


def sample_arrays(seeds: np.ndarray) -> dict[str, np.ndarray]:
    n = len(seeds)
    return {
        "meta_schema_version": np.array(2),
        "meta_feature_version": np.array(1),
        "meta_target_version": np.array(1),
        "sample_seed": seeds,
        "split": assign_split(seeds),
        "eta": np.zeros((n, 3)),
        "start_pose": np.zeros((n, 3)),
        "start_velocity": np.zeros((n, 3)),
        "site_position": np.zeros((n, 3)),
        "site_normal": np.zeros((n, 3)),
        "speed": np.full(n, 0.05),
        "time": np.full(n, 0.2),
        "tip_path": np.zeros((n, 3, 16)),
        "displacement": np.zeros((n, 3)),
        "push_end_velocity": np.zeros((n, 3)),
    }


def test_seed_split_is_stable_and_has_separate_held_out_rows(
    tmp_path: Path,
) -> None:
    seeds = np.arange(10, dtype=np.uint64)
    arrays = sample_arrays(seeds)
    path = tmp_path / "samples.npz"
    np.savez(path, **arrays)
    x, y, split = dataset_arrays(path)
    assert x.shape == (10, 49)
    assert y.shape == (10, 6)
    assert np.bincount(split, minlength=3).tolist() == [8, 1, 1]
    assert assign_split(seeds).tolist() == split.tolist()


def test_one_seed_cannot_cross_splits(tmp_path: Path) -> None:
    arrays = sample_arrays(np.array([0, 1, 2, 2], dtype=np.uint64))
    arrays["split"][-1] = 1
    path = tmp_path / "leak.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="multiple splits"):
        dataset_arrays(path)


def test_action_path_events_and_local_lift() -> None:
    obj = Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, 0.2))
    obj.pose = (0.55, 0.0, 0.4)
    action = Action(
        SurfaceNormal(
            position=np.array([0.07, 0.0, 0.015]),
            normal=np.array([-1.0, 0.0, 0.0]),
        ),
        velocity=0.1,
        time=0.2,
    )
    lift_height = hover_tip_pose(obj).translation()[2]
    path = nominal_action_tip_path(obj.x_wo(), action, 0.1, lift_height)
    assert path.tip_times[path.window_samples - 1] == pytest.approx(path.window_end)
    assert path.action_end == pytest.approx(
        path.push_end_time + STANDOFF / action.velocity + MOVE_TIME + HOLD_TIME
    )
    assert np.diff(path.tip_times).min() > 0.0
    assert path.tip_positions[2, -1] == pytest.approx(lift_height)
    push_end = obj.x_wo() @ action.contact.position + action.velocity * action.time * (
        obj.x_wo().rotation() @ action.contact.normal
    )
    backed_off = push_end - STANDOFF * (obj.x_wo().rotation() @ action.contact.normal)
    assert path.tip_positions[:2, -1] == pytest.approx(backed_off[:2])


def test_training_residuals_rotate_from_start_frame_to_world() -> None:
    residual = np.array([[1.0, 0.0, 0.0, 2.0, 0.0, 0.0]])
    world = world_frame_errors(residual, np.zeros((1, 6)), np.array([np.pi / 2]))
    assert world[0] == pytest.approx([0.0, 1.0, 0.0, 0.0, 2.0, 0.0])
