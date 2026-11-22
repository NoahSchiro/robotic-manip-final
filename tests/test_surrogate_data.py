from pathlib import Path

import numpy as np
import pytest

from src.estimation.dynamics import DrakePushModel, PushModelConfig
from src.estimation.params import Eta
from src.surrogate.data import (
    assign_split,
    coverage_summary,
    make_surrogate_setup,
    stack_samples,
    surrogate_sample,
)
from src.surrogate.model import (
    N_INPUTS,
    N_OUTPUTS,
    PATH_SAMPLES,
    SurrogateEnsemble,
    SurrogatePredictor,
    nominal_action_tip_path,
    nominal_tip_path,
)
from src.surrogate.path_family import PathFamily
from src.surrogate.train import dataset_arrays, shard_arrays, world_frame_errors
from src.world.action import Action
from src.world.geometry import SurfaceNormal
from src.world.iiwa import (
    HOLD_TIME,
    LEG_PUSH,
    MOVE_TIME,
    RELEASE_DISTANCE,
    RELEASE_SPEED,
    STANDOFF,
    push_maneuver,
    transfer_height,
)
from src.world.object import Object, Rod


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


def make_push() -> tuple[Object, Action]:
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
    return obj, action


def test_action_path_events_and_local_lift() -> None:
    obj, action = make_push()
    X_WO = obj.x_wo()
    path = nominal_action_tip_path(X_WO, action, 0.3, transfer_height(obj))
    push_end_time = STANDOFF / action.velocity + action.time
    assert path.push_end_time == pytest.approx(push_end_time)
    assert path.tip_times[path.window_samples - 1] == pytest.approx(path.window_end)
    assert path.action_end == pytest.approx(
        push_end_time + RELEASE_DISTANCE / RELEASE_SPEED + MOVE_TIME + HOLD_TIME
    )
    assert np.diff(path.tip_times).min() > 0.0
    assert path.tip_positions[2, -1] == pytest.approx(transfer_height(obj))
    normal = X_WO.rotation() @ action.contact.normal
    push_end = X_WO @ action.contact.position + action.velocity * action.time * normal
    released = push_end - RELEASE_DISTANCE * normal
    assert path.tip_positions[:2, -1] == pytest.approx(released[:2])


def test_surrogate_paths_pass_through_the_maneuver_waypoints() -> None:
    obj, action = make_push()
    X_WO = obj.x_wo()
    maneuver = push_maneuver(X_WO, action, transfer_height(obj))
    # The window starts at the standoff, where the dwell ends: waypoints are
    # push end, release end, lift end.
    waypoints = [pose.translation() for pose, _ in maneuver[LEG_PUSH:]]
    breaks = np.cumsum([0.0, *(d for _pose, d in maneuver[LEG_PUSH:])])[1:]
    action_path = nominal_action_tip_path(X_WO, action, 0.3, transfer_height(obj))
    for t, waypoint in zip(breaks, waypoints):
        index = int(np.argmin(np.abs(action_path.tip_times - t)))
        assert action_path.tip_times[index] == pytest.approx(t, abs=1e-9)
        assert action_path.tip_positions[:, index] == pytest.approx(waypoint, abs=1e-9)
    assert action_path.tip_positions[:, -1] == pytest.approx(waypoints[-1])


def test_window_path_is_the_action_path_cut_at_the_window_end() -> None:
    obj, action = make_push()
    X_WO = obj.x_wo()
    height = transfer_height(obj)
    window = nominal_tip_path(X_WO, action, 0.3, height)
    full = nominal_action_tip_path(X_WO, action, 0.3, height)
    n = full.window_samples
    assert window.window_end == pytest.approx(full.window_end)
    assert window.tip_times == pytest.approx(full.tip_times[:n])
    assert window.tip_positions == pytest.approx(full.tip_positions[:, :n])
    assert window.tip_times[-1] == pytest.approx(window.window_end)


def test_training_residuals_rotate_from_start_frame_to_world() -> None:
    residual = np.array([[1.0, 0.0, 0.0, 2.0, 0.0, 0.0]])
    world = world_frame_errors(residual, np.zeros((1, 6)), np.array([np.pi / 2]))
    assert world[0] == pytest.approx([0.0, 1.0, 0.0, 0.0, 2.0, 0.0])


def zero_family() -> PathFamily:
    return PathFamily(
        mean=np.zeros((2, PATH_SAMPLES)),
        components=np.zeros((1, 2, PATH_SAMPLES)),
        sigmas=np.zeros(1),
    )


def test_samples_are_drawn_from_the_push_space() -> None:
    setup = make_surrogate_setup(
        rod_mass=0.2,
        settle_time=0.3,
        family=zero_family(),
        model=PushModelConfig(time_step=2e-3, rod_mass=0.2),
    )
    model = DrakePushModel(setup.model)
    space = setup.push_space
    samples = [surrogate_sample(seed, setup, model) for seed in (3, 4)]
    for sample in samples:
        speed, time = float(sample["speed"]), float(sample["time"])
        assert space.speed_range[0] <= speed <= space.speed_range[1]
        assert space.duration_range[0] <= time <= space.duration_range[1]
        assert speed * time <= space.max_distance + 1e-12
        assert np.isfinite(sample["displacement"]).all()
    dataset = stack_samples(samples)
    assert "samples per push speed" in coverage_summary(dataset, setup)


def test_shards_concatenate_and_drop_repeated_seeds(tmp_path: Path) -> None:
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(first, allow_pickle=False, **sample_arrays(np.arange(10, dtype=np.uint64)))
    np.savez(
        second, allow_pickle=False, **sample_arrays(np.arange(5, 20, dtype=np.uint64))
    )
    x, y, split = shard_arrays([first, second])
    assert len(x) == len(y) == len(split) == 20
    assert np.bincount(split, minlength=3).tolist() == [16, 2, 2]


def test_predictions_do_not_depend_on_the_yaw_convention() -> None:
    rng = np.random.default_rng(0)
    ensemble = SurrogateEnsemble(
        input_mean=np.zeros(N_INPUTS),
        input_std=np.ones(N_INPUTS),
        output_mean=np.zeros(N_OUTPUTS),
        output_std=np.ones(N_OUTPUTS),
        members=(((rng.normal(size=(N_OUTPUTS, N_INPUTS)), np.zeros(N_OUTPUTS)),),),
    )
    predictor = SurrogatePredictor(ensemble)
    parts = (np.zeros(3), np.array([1.0, 0.0, 0.0]), 0.1, 0.4, np.zeros(32))
    start = np.array([0.55, 0.0, -1.0, 0.0, 0.0, 0.0])
    wrapped = start.copy()
    wrapped[2] += 2 * np.pi
    eta = [Eta(0.5, 0.0, 0.0)]
    assert np.allclose(
        predictor.predict_from_parts(eta, start, *parts),
        predictor.predict_from_parts(eta, wrapped, *parts),
    )
