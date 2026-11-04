"""Recorded random-push trials from the truth sim: the offline dataset the
estimator is evaluated on, and later the surrogate predictor is validated
against.

One trial draws off-grid true parameters, runs up to `pushes` random library
pushes in one continuous TruthSim and serializes every PushObservation to
plain arrays.  Serialization happens inside the worker process: Drake objects
need not cross process boundaries, and the stored dataset stays readable
without pydrake.

Dataset schema (one .npz):
- eta: (T, 3) true (mu, bx, by) per trial
- trial_id: (P,) ascending trial index per push
- push_end_time, action_velocity, action_time: (P,)
- contact_position, contact_normal: (P, 3)
- {start, push_end, end, final}_pose: (P, 4, 4) world transforms
- {start, push_end, end, final}_velocity: (P, 6) spatial velocities as
  [rotational; translational]
- tip_times: (P,) object of (n,), tip_positions/tip_velocities: (P,) object
  of (3, n) — the window's tip path, times from its start
- meta_<name>: the scalars a replay needs to rebuild the setup (seed, grid
  era settings, truth and model timesteps)
"""

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydrake.all import RigidTransform, SpatialVelocity

from .action import Action
from .geometry import SurfaceNormal, SurfacePart
from .iiwa import Pusher, clear_sites
from .object import Object, Rod
from .observation import ObjectState, PushObservation
from .params import Eta, GridSpec, sample_eta
from .recording import NoReachablePush, TruthSim, TruthSimConfig

STATE_NAMES = ("start", "push_end", "end", "final")
META_PREFIX = "meta_"


@dataclass(frozen=True)
class TrialData:
    """One recorded trial: its true parameters and its pushes in order."""

    eta: Eta
    observations: list[PushObservation]


@dataclass(frozen=True)
class RecordSetup:
    """What every recorded trial shares."""

    spec: GridSpec
    parts: list[SurfacePart]
    rod_mass: float
    push_density: float
    site_clearance: float
    pushes: int
    truth: TruthSimConfig


def make_record_setup(
    rod_mass: float,
    push_density: float,
    site_clearance: float,
    pushes: int,
    truth: TruthSimConfig,
) -> RecordSetup:
    """Collect what every recorded trial shares."""
    spec = GridSpec()
    # Only the footprint is needed, and it's the same for every eta.
    parts = Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, rod_mass)).parts
    return RecordSetup(
        spec=spec,
        parts=parts,
        rod_mass=rod_mass,
        push_density=push_density,
        site_clearance=site_clearance,
        pushes=pushes,
        truth=truth,
    )


def record_trial(seed: int, setup: RecordSetup) -> TrialData:
    """Record one trial: off-grid parameters, then up to `setup.pushes`
    random reachable pushes in one continuous truth sim, each from where the
    last left the block.  A trial ends early when no push is reachable."""
    rng = np.random.default_rng(seed)
    eta = sample_eta(rng, setup.spec, setup.parts)
    obj = eta.to_object(rng, setup.rod_mass)
    pusher = Pusher()
    sites = clear_sites(
        obj.parts,
        obj.surface_normals(setup.push_density),
        pusher.radius,
        setup.site_clearance,
    )
    sim = TruthSim(obj, pusher, setup.truth)
    observations = []
    for _ in range(setup.pushes):
        try:
            obs = sim.record_random_push(rng, obj, sites)
        except NoReachablePush:
            break
        observations.append(obs)
        obj.pose = obs.final_pose()
    return TrialData(eta=eta, observations=observations)


def pose_matrix(state: ObjectState) -> np.ndarray:
    """Get the (4, 4) world transform of an object state."""
    return state.pose.GetAsMatrix4()


def velocity_vector(state: ObjectState) -> np.ndarray:
    """Get the (6,) [rotational; translational] world velocity of a state."""
    return np.concatenate([state.velocity.rotational(), state.velocity.translational()])


def stacked(rows: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    """Stack per-push arrays into (n, *shape), surviving an empty list."""
    return np.array(rows, dtype=float).reshape((len(rows), *shape))


def trial_to_arrays(
    eta: Eta, observations: Sequence[PushObservation]
) -> dict[str, np.ndarray]:
    """Serialize one trial to plain arrays (see the module docstring)."""
    arrays: dict[str, np.ndarray] = {
        "eta": eta.as_array(),
        "push_end_time": np.array([o.push_end_time for o in observations]),
        "action_velocity": np.array([o.action.velocity for o in observations]),
        "action_time": np.array([o.action.time for o in observations]),
        "contact_position": stacked(
            [o.action.contact.position for o in observations], (3,)
        ),
        "contact_normal": stacked(
            [o.action.contact.normal for o in observations], (3,)
        ),
    }
    for name in STATE_NAMES:
        arrays[f"{name}_pose"] = stacked(
            [pose_matrix(getattr(o, name)) for o in observations], (4, 4)
        )
        arrays[f"{name}_velocity"] = stacked(
            [velocity_vector(getattr(o, name)) for o in observations], (6,)
        )
    for key in ("tip_times", "tip_positions", "tip_velocities"):
        # np.array(list, dtype=object) would fold equal-length arrays into a
        # rectangular array; the object array must hold one path per push.
        paths = np.empty(len(observations), dtype=object)
        for push, obs in enumerate(observations):
            paths[push] = getattr(obs, key)
        arrays[key] = paths
    return arrays


def record_trial_arrays(seed: int, setup: RecordSetup) -> dict[str, np.ndarray]:
    """Record one trial and serialize it, running in a worker process so
    only plain arrays cross back to the parent."""
    trial = record_trial(seed, setup)
    return trial_to_arrays(trial.eta, trial.observations)


def dataset_from_trial_arrays(
    trials: Sequence[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Stack per-trial arrays into the dataset schema: a (T, 3) eta table and
    push-axis arrays spanning all trials, ordered by trial."""
    push_keys = [key for key in trials[0] if key != "eta"]
    dataset: dict[str, np.ndarray] = {
        "eta": np.array([trial["eta"] for trial in trials], dtype=float),
        "trial_id": np.concatenate(
            [
                np.full(len(trial["push_end_time"]), index, dtype=int)
                for index, trial in enumerate(trials)
            ]
        ),
    }
    for key in push_keys:
        dataset[key] = np.concatenate([trial[key] for trial in trials], axis=0)
    return dataset


def record_trials(
    setup: RecordSetup, trials: int, seed: int, workers: int
) -> dict[str, np.ndarray]:
    """Record `trials` trials on `workers` processes, returning the stacked
    dataset arrays.  Trials print a line as they finish."""
    seeds = np.random.default_rng(seed).integers(2**32, size=trials)
    results: dict[int, dict[str, np.ndarray]] = {}
    with ProcessPoolExecutor(workers) as pool:
        futures = {
            pool.submit(record_trial_arrays, int(s), setup): index
            for index, s in enumerate(seeds)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            eta = results[index]["eta"]
            pushes = len(results[index]["push_end_time"])
            print(
                f"trial {index}: mu={eta[0]:.3f} b=({eta[1]:+.4f}, {eta[2]:+.4f}) m, "
                f"{pushes} pushes"
            )
    return dataset_from_trial_arrays([results[i] for i in range(trials)])


def save_dataset(
    path: Path, dataset: dict[str, np.ndarray], metadata: dict[str, float | int]
) -> None:
    """Write the dataset and its meta_ scalars to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        **dataset,
        **{f"{META_PREFIX}{key}": np.asarray(value) for key, value in metadata.items()},
    }
    np.savez(path, allow_pickle=True, **arrays)


def spatial_velocity(vector: np.ndarray) -> SpatialVelocity:
    """Rebuild a spatial velocity from its [rotational; translational] parts."""
    return SpatialVelocity(vector[:3], vector[3:])


def object_state(data: np.lib.npyio.NpzFile, name: str, push: int) -> ObjectState:
    """Rebuild one recorded object state."""
    return ObjectState(
        pose=RigidTransform(data[f"{name}_pose"][push]),
        velocity=spatial_velocity(data[f"{name}_velocity"][push]),
    )


def push_from_arrays(data: np.lib.npyio.NpzFile, push: int) -> PushObservation:
    """Rebuild one recorded push observation."""
    return PushObservation(
        action=Action(
            contact=SurfaceNormal(
                position=data["contact_position"][push],
                normal=data["contact_normal"][push],
            ),
            velocity=float(data["action_velocity"][push]),
            time=float(data["action_time"][push]),
        ),
        start=object_state(data, "start", push),
        push_end=object_state(data, "push_end", push),
        end=object_state(data, "end", push),
        final=object_state(data, "final", push),
        push_end_time=float(data["push_end_time"][push]),
        tip_times=data["tip_times"][push],
        tip_positions=data["tip_positions"][push],
        tip_velocities=data["tip_velocities"][push],
    )


def load_dataset(
    path: Path,
) -> tuple[list[TrialData], dict[str, float | int]]:
    """Read a recorded dataset, returning its trials in order and its
    metadata scalars."""
    with np.load(path, allow_pickle=True) as data:
        metadata = {
            key[len(META_PREFIX) :]: data[key].item()
            for key in data.files
            if key.startswith(META_PREFIX)
        }
        etas = data["eta"]
        trial_id = data["trial_id"]
        bounds = np.searchsorted(trial_id, np.arange(len(etas) + 1))
        trials = [
            TrialData(
                eta=Eta(*(float(v) for v in etas[trial])),
                observations=[
                    push_from_arrays(data, push)
                    for push in range(bounds[trial], bounds[trial + 1])
                ],
            )
            for trial in range(len(etas))
        ]
    return trials, metadata
