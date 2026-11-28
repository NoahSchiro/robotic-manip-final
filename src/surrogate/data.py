"""The surrogate's training dataset: Drake-predicted push outcomes for
sampled (eta, start pose, action) tuples.

Dataset schema (one .npz):
- eta: (N, 3) true (mu, bx, by) per sample
- start_pose: (N, 3) planar x, y, yaw (yaw in [0, 2 pi)); start_velocity:
  (N, 3) planar, +-1e-9 seeded dust (Drake's discrete solver rejects an
  exactly-zero start; see surrogate_sample)
- site_position, site_normal: (N, 3) contact point and inward normal in
  the object's body frame
- speed, time: (N,) commanded push speed (m/s) and contact time (s); each
  push is a uniform draw from the push space (push_space.PushSpace.sample)
- tip_path: (N, 3, PATH_SAMPLES) executed tip positions at uniform window
  times, world frame — the surrogate's path conditioning
- action_tip_path: (N, 3, PATH_SAMPLES) constructed tip positions through
  local lift and hold, world frame; action_end_time: (N,) s from window start
- displacement: (N, 3) window-end pose offset (x, y, yaw) in the start
  frame
- push_end_velocity: (N, 3) block planar velocity at the push segment's
  end, in the start frame
- window_end_velocity: (N, 3) block planar velocity at the window's end,
  in the start frame. A diagnostic for the at-rest assumption
- action_end_displacement, action_end_velocity: (N, 3) block state at the
  next decision boundary, in the start frame; not used by the current trainer
- sample_seed: (N,) seed of each independent sampled trajectory
- split: (N,) 0=train, 1=development, 2=held-out, assigned by sample seed
- meta_*: generation settings
"""

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from pydrake.all import SpatialVelocity

from src.constants import START_RADIUS, WORKSPACE_CENTER
from src.estimation.dynamics import DrakePushModel, PushModelConfig
from src.estimation.params import GridSpec, sample_eta
from src.experiment import reference_object
from src.surrogate.model import (
    SAMPLE_PERIOD,
    nominal_action_tip_path,
    path_downsample,
)
from src.surrogate.path_family import PathFamily, perturb_tip_positions
from src.world.geometry import SurfacePart
from src.world.iiwa import transfer_height
from src.world.observation import ObjectState, wrap_angle
from src.world.push_space import PushSpace, make_push_space

MU_DECADES = 10  # coverage bins along the friction range
COVERAGE_BINS = 3  # coverage bins along push speed and push distance
SPLIT_MODULUS = 10


@dataclass(frozen=True)
class SurrogateSetup:
    """Settings shared by all surrogate samples."""

    spec: GridSpec
    parts: list[SurfacePart]
    push_space: PushSpace
    rod_mass: float
    settle_time: float
    family: PathFamily
    model: PushModelConfig


def make_surrogate_setup(
    rod_mass: float,
    settle_time: float,
    family: PathFamily,
    model: PushModelConfig,
) -> SurrogateSetup:
    """Collect the settings shared by all samples. The footprint, and so the
    push space, is the same for every eta."""
    reference = reference_object(rod_mass)
    return SurrogateSetup(
        spec=GridSpec(),
        parts=reference.parts,
        push_space=make_push_space(reference.parts, reference.height),
        family=family,
        rod_mass=rod_mass,
        settle_time=settle_time,
        model=model,
    )


def surrogate_metadata(
    setup: SurrogateSetup, samples: int, seed: int
) -> dict[str, float | int]:
    """Generation settings and schema versions, stored in the dataset file."""
    space = setup.push_space
    return {
        "seed": seed,
        "samples": samples,
        "rod_mass": setup.rod_mass,
        "speed_min": space.speed_range[0],
        "speed_max": space.speed_range[1],
        "duration_min": space.duration_range[0],
        "duration_max": space.duration_range[1],
        "max_push_distance": space.max_distance,
        "model_time_step": setup.model.time_step,
        "settle_time": setup.settle_time,
        "sample_period": SAMPLE_PERIOD,
        "schema_version": 3,
        "feature_version": 1,
        "target_version": 1,
        "action_target_version": 1,
    }


def sample_start_pose(rng: np.random.Generator) -> tuple[float, float, float]:
    """A start pose uniform within START_RADIUS of the workspace center, yaw
    uniform in [0, 2 pi).  Three more draws are discarded so that existing
    datasets regenerate identically from their sample seeds."""
    rho = START_RADIUS * np.sqrt(rng.uniform())
    theta = rng.uniform(0.0, 2.0 * np.pi)
    pose = (
        WORKSPACE_CENTER[0] + rho * np.cos(theta),
        WORKSPACE_CENTER[1] + rho * np.sin(theta),
        rng.uniform(0.0, 2.0 * np.pi),
    )
    rng.uniform(size=3)
    return pose


def surrogate_sample(
    seed: int, setup: SurrogateSetup, model: DrakePushModel
) -> dict[str, np.ndarray]:
    """Draw one sample and predict its outcome. Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    eta = sample_eta(rng, setup.spec, setup.parts)
    obj = eta.to_object(setup.rod_mass)
    obj.pose = sample_start_pose(rng)
    action = setup.push_space.sample(rng)
    # The discrete solver rejects an exactly-zero start velocity; truth-sim
    # states carry ~1e-10 of numerical dust, so match that.
    start_velocity = SpatialVelocity(np.zeros(3), rng.uniform(-1e-9, 1e-9, size=3))

    path = nominal_action_tip_path(
        obj.x_wo(), action, setup.settle_time, transfer_height(obj)
    )
    path.tip_positions[:] = perturb_tip_positions(
        path.tip_positions, path.tip_times, path.window_end, setup.family, rng
    )
    model.set_eta(eta)
    push_end, window_end, action_end = model.predict_path(
        ObjectState(pose=obj.x_wo(), velocity=start_velocity),
        path.tip_times,
        path.tip_positions,
        path.tip_velocities,
        [path.push_end_time, path.window_end, path.action_end],
    )

    # Push effects in the start pose's frame: xy rotated by R(-yaw), yaw
    # offsets wrapped, wz unchanged.
    start = np.array(obj.pose, dtype=float)
    c, s = np.cos(start[2]), np.sin(start[2])
    rot = np.array([[c, s], [-s, c]])
    displacement = np.array(
        [*(rot @ (window_end[:2] - start[:2])), wrap_angle(window_end[2] - start[2])]
    )
    push_velocity = np.array([*(rot @ push_end[3:5]), push_end[5]])
    end_velocity = np.array([*(rot @ window_end[3:5]), window_end[5]])
    action_displacement = np.array(
        [*(rot @ (action_end[:2] - start[:2])), wrap_angle(action_end[2] - start[2])]
    )
    action_velocity = np.array([*(rot @ action_end[3:5]), action_end[5]])
    window_tip_times = path.tip_times[: path.window_samples]
    window_tip_positions = path.tip_positions[:, : path.window_samples]
    return {
        "sample_seed": np.array(seed, dtype=np.uint64),
        "eta": eta.as_array(),
        "start_pose": start,
        "start_velocity": np.array(
            [
                start_velocity.translational()[0],
                start_velocity.translational()[1],
                start_velocity.rotational()[2],
            ]
        ),
        "site_position": action.contact.position,
        "site_normal": action.contact.normal,
        "speed": np.array(action.velocity),
        "time": np.array(action.time),
        "tip_path": path_downsample(window_tip_times, window_tip_positions),
        "action_tip_path": path_downsample(path.tip_times, path.tip_positions),
        "push_end_time": np.array(path.push_end_time),
        "window_end_time": np.array(path.window_end),
        "action_end_time": np.array(path.action_end),
        "displacement": displacement,
        "push_end_velocity": push_velocity,
        "window_end_velocity": end_velocity,
        "action_end_displacement": action_displacement,
        "action_end_velocity": action_velocity,
    }


def surrogate_samples(
    seeds: Sequence[int], setup: SurrogateSetup
) -> list[dict[str, np.ndarray]]:
    """Predict the outcomes of `seeds`' samples in one worker process, sharing
    one prediction model."""
    model = DrakePushModel(setup.model)
    return [surrogate_sample(seed, setup, model) for seed in seeds]


def stack_samples(samples: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack per-sample arrays into the dataset schema."""
    return {key: np.array([sample[key] for sample in samples]) for key in samples[0]}


def generate_surrogate_data(
    setup: SurrogateSetup, samples: int, seed: int, workers: int
) -> dict[str, np.ndarray]:
    """Generate `samples` surrogate samples on `workers` processes,
    returning the stacked dataset.  Chunks print a line as they finish."""
    seeds = np.random.default_rng(seed).integers(2**32, size=samples)
    chunk_size = max(int(np.ceil(samples / (workers * 8))), 1)
    chunks = [
        seeds[start : start + chunk_size] for start in range(0, samples, chunk_size)
    ]
    results: dict[int, list[dict[str, np.ndarray]]] = {}
    done = 0
    with ProcessPoolExecutor(workers) as pool:
        futures = {
            pool.submit(surrogate_samples, [int(s) for s in chunk], setup): index
            for index, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            done += len(results[index])
            print(f"surrogate samples: {done}/{samples}")
    return stack_samples([s for i in range(len(chunks)) for s in results[i]])


def assign_split(sample_seeds: np.ndarray) -> np.ndarray:
    """Assign independent trajectories to train/development/held-out by seed."""
    buckets = np.asarray(sample_seeds, dtype=np.uint64) % SPLIT_MODULUS
    return np.where(buckets == 0, 1, np.where(buckets == 1, 2, 0)).astype(np.uint8)


def check_finite(dataset: dict[str, np.ndarray]) -> None:
    """Raise ValueError when any dataset column holds NaN or infinity."""
    for key, value in dataset.items():
        if not np.isfinite(value.astype(float)).all():
            raise ValueError(f"dataset column {key} holds non-finite values")


def coverage_summary(dataset: dict[str, np.ndarray], setup: SurrogateSetup) -> str:
    """Render sample counts over push speed x push distance bins and over
    friction decades."""
    space = setup.push_space
    distance = dataset["speed"] * dataset["time"]
    min_distance = space.speed_range[0] * space.duration_range[0]
    speed_edges = np.linspace(*space.speed_range, COVERAGE_BINS + 1)
    distance_edges = np.linspace(min_distance, space.max_distance, COVERAGE_BINS + 1)
    speed_bin = np.clip(
        np.digitize(dataset["speed"], speed_edges) - 1, 0, COVERAGE_BINS - 1
    )
    distance_bin = np.clip(
        np.digitize(distance, distance_edges) - 1, 0, COVERAGE_BINS - 1
    )
    lines = ["samples per push speed (rows, m/s) x push distance (columns, m):"]
    lines.append(
        " " * 14
        + "".join(
            f"{distance_edges[j]:.3f}-{distance_edges[j + 1]:.3f}  "
            for j in range(COVERAGE_BINS)
        )
    )
    for i in range(COVERAGE_BINS):
        counts = [
            int(((speed_bin == i) & (distance_bin == j)).sum())
            for j in range(COVERAGE_BINS)
        ]
        lines.append(
            f"{speed_edges[i]:.3f}-{speed_edges[i + 1]:.3f}  "
            + "".join(f"{c:>13d}  " for c in counts)
        )
    decade_width = (setup.spec.mu_range[1] - setup.spec.mu_range[0]) / MU_DECADES
    decades = np.minimum(
        np.floor((dataset["eta"][:, 0] - setup.spec.mu_range[0]) / decade_width).astype(
            int
        ),
        MU_DECADES - 1,
    )
    for decade in range(MU_DECADES):
        in_decade = decades == decade
        lo = setup.spec.mu_range[0] + decade * decade_width
        lines.append(
            f"mu [{lo:.2f}, {lo + decade_width:.2f}): {int(in_decade.sum())} samples"
        )
    return "\n".join(lines)
