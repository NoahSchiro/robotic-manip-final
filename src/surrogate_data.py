"""The surrogate's training dataset: Drake-predicted push outcomes for
sampled (eta, start pose, action) tuples.

Dataset schema (one .npz):
- eta: (N, 3) true (mu, bx, by) per sample
- start_pose: (N, 3) planar x, y, yaw; start_velocity: (N, 3) planar,
  ~1e-9 m/s seeded noise (see above)
- site_position, site_normal: (N, 3) contact point and inward normal in
  the object's body frame
- site_index: (N,) index into the setup's clear sites
- speed, time: (N,) commanded push speed (m/s) and contact time (s)
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

from .action import Action
from .dynamics import DrakePushModel, PushModelConfig
from .geometry import SurfaceNormal, SurfacePart
from .iiwa import Pusher, clear_sites, hover_tip_pose
from .object import Object, Rod
from .observation import ObjectState, wrap_angle
from .params import GridSpec, sample_eta
from .path_family import PathFamily, perturb_tip_positions
from .surrogate import nominal_action_tip_path, path_downsample

MU_DECADES = 10  # coverage bins along the friction range
SPLIT_MODULUS = 10


@dataclass(frozen=True)
class SurrogateSetup:
    """What every surrogate sample shares."""

    spec: GridSpec
    parts: list[SurfacePart]
    sites: list[SurfaceNormal]
    rod_mass: float
    speeds: tuple[float, ...]
    times: tuple[float, ...]
    settle_time: float
    family: PathFamily
    model: PushModelConfig


def make_surrogate_setup(
    rod_mass: float,
    push_density: float,
    site_clearance: float,
    settle_time: float,
    family: PathFamily,
    model: PushModelConfig,
    speeds: tuple[float, ...] = (0.05, 0.1),
    times: tuple[float, ...] = (0.2, 0.4),
) -> SurrogateSetup:
    """Collect what every surrogate sample shares.  The footprint, and with
    it the push library, is the same for every eta."""
    spec = GridSpec()
    reference = Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, rod_mass))
    return SurrogateSetup(
        spec=spec,
        parts=reference.parts,
        family=family,
        sites=clear_sites(
            reference.parts,
            reference.surface_normals(push_density),
            Pusher().radius,
            site_clearance,
        ),
        rod_mass=rod_mass,
        speeds=speeds,
        times=times,
        settle_time=settle_time,
        model=model,
    )


def surrogate_sample(
    seed: int, setup: SurrogateSetup, model: DrakePushModel
) -> dict[str, np.ndarray]:
    """Draw one sample and predict its outcome.  Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    eta = sample_eta(rng, setup.spec, setup.parts)
    obj = eta.to_object(rng, setup.rod_mass)
    site_index = int(rng.integers(len(setup.sites)))
    action = Action(
        contact=setup.sites[site_index],
        velocity=float(rng.choice(setup.speeds)),
        time=float(rng.choice(setup.times)),
    )
    # An exactly-zero start velocity is a degenerate contact state the
    # discrete solver rejects; the truth sim's states always carry ~1e-10
    # of numerical dust, so match that here.
    start_velocity = SpatialVelocity(np.zeros(3), rng.uniform(-1e-9, 1e-9, size=3))

    path = nominal_action_tip_path(
        obj.x_wo(), action, setup.settle_time, hover_tip_pose(obj).translation()[2]
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

    # Push effects in the start pose's frame: rotate world xy by R(-yaw);
    # yaw offsets wrap, wz passes through.
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
        "site_index": np.array(site_index, dtype=int),
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
    """Predict the outcomes of `seeds`' samples, running in a worker process
    with one shared prediction model."""
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
    """Render per-cell counts over speeds, contact times, friction decades
    and sites."""
    cell_count = len(setup.sites) * len(setup.speeds) * len(setup.times)
    header = (
        f"sites {len(setup.sites)} x speeds {len(setup.speeds)} "
        f"x times {len(setup.times)} = {cell_count} cells"
    )
    lines = [header]
    for speed in setup.speeds:
        for time in setup.times:
            in_cell = (dataset["speed"] == speed) & (dataset["time"] == time)
            per_site = np.bincount(
                dataset["site_index"][in_cell], minlength=len(setup.sites)
            )
            lines.append(
                f"speed {speed:.2f} time {time:.1f}: {int(in_cell.sum())} samples, "
                f"{int((per_site == 0).sum())} empty sites, min per site {int(per_site.min())}"
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
