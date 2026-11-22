"""The trained surrogate: an ensemble of standardized MLPs predicting a
push's outcome from (eta, start state, action), and its adapter to the
estimator's DynamicsModel protocol.

The ensemble is trained on `gen-surrogate-data`'s dataset (Drake-predicted
outcomes of sampled pushes) and saved as one .npz of weights and
standardization statistics, so inference needs only numpy.

Input columns (49), in order:
    eta (mu, bx, by) — the hypothesis
    start pose (x, y, yaw) — world, yaw in [0, 2 pi) (feature_start)
    start velocity (vx, vy, wz) — world
    site position (x, y, z) — object body frame
    site normal (x, y, z) — object body frame
    commanded speed (m/s), contact time (s)
    tip path (x, y) at PATH_SAMPLES uniform window times; at
        training time the executed (lagged) constructed path, at inference
        the measured path downsampled exactly the same way.  The z row is
        dropped: the constructed paths hold it constant, so it carries no
        signal, while the arm's measured z wobbles by centimeters.

Output columns (6), all in the start pose's frame:
    displacement (dx, dy, dyaw). Window-end pose offset
    push-end velocity (vx, vy, wz)
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydrake.all import RigidTransform

from src.estimation.params import Eta
from src.world.action import Action
from src.world.iiwa import HOLD_TIME, LEG_DWELL, LEG_PUSH, push_maneuver
from src.world.observation import PushObservation, wrap_angle

N_OUTPUTS = 6
PATH_SAMPLES = 16  # tip path points the surrogate conditions on
N_INPUTS = 17 + 2 * PATH_SAMPLES

# Type aliases: a layer is (weight, bias); one member is a tuple of layers.
Layer = tuple[np.ndarray, np.ndarray]
MLPWeights = tuple[Layer, ...]


@dataclass(frozen=True)
class SurrogateEnsemble:
    """The trained surrogate: standardization statistics plus one weight set
    per member.  `predict` averages the members; `predict` with `member`
    returns that member alone (the env's randomization draws on it)."""

    input_mean: np.ndarray
    input_std: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray
    members: tuple[MLPWeights, ...]

    def predict(self, x: np.ndarray, member: int | None = None) -> np.ndarray:
        """Predict standardized-to-raw outputs for the (N, 49) inputs `x`:
        the members' mean, or one member's output."""
        standardized = ((x - self.input_mean) / self.input_std).astype(np.float32)
        if member is None:
            outputs = [mlp_forward(w, standardized) for w in self.members]
            raw = np.mean(np.stack(outputs), axis=0)
        else:
            raw = mlp_forward(self.members[member], standardized)
        return np.asarray(raw * self.output_std + self.output_mean)


SAMPLE_PERIOD = 0.01  # s, nominal tip path sampling period
TIME_DECIMALS = 9  # rounds maneuver event times so coincident events merge


@dataclass(frozen=True)
class TipPath:
    """A constructed push's executed tip path and its window times (s)."""

    tip_times: np.ndarray  # (n,) from 0 to window_end
    tip_positions: np.ndarray  # (3, n) world
    tip_velocities: np.ndarray  # (3, n) world
    push_end_time: float
    window_end: float


@dataclass(frozen=True)
class ActionTipPath:
    """Constructed tip path through local lift and hold, times from standoff."""

    tip_times: np.ndarray
    tip_positions: np.ndarray
    tip_velocities: np.ndarray
    push_end_time: float
    window_end: float
    action_end: float
    window_samples: int


def nominal_tip_path(
    X_WO: RigidTransform,
    action: Action,
    settle_time: float,
    transfer_height: float,
) -> TipPath:
    """The measurement window of `nominal_action_tip_path`: from the standoff
    point through the push, closing `settle_time` after the push segment.  The
    executed-path deviation (arm tracking error) is added by the caller from
    the calibrated path family."""
    path = nominal_action_tip_path(X_WO, action, settle_time, transfer_height)
    n = path.window_samples
    return TipPath(
        tip_times=path.tip_times[:n],
        tip_positions=path.tip_positions[:, :n],
        tip_velocities=path.tip_velocities[:, :n],
        push_end_time=path.push_end_time,
        window_end=path.window_end,
    )


def _path_times(events: np.ndarray, end: float) -> np.ndarray:
    """Sample times (s) from 0 to `end` on SAMPLE_PERIOD, with `events` (the
    maneuver's break times and the window end) included exactly.  Grid samples
    within half a period of an event are dropped, so no two times nearly
    coincide and the spline's breaks stay distinct."""
    events = np.unique(np.round(events, TIME_DECIMALS))
    grid = np.arange(0.0, end, SAMPLE_PERIOD)
    grid = grid[np.abs(grid[:, None] - events).min(axis=1) > SAMPLE_PERIOD / 2]
    return np.sort(np.concatenate([grid, events]))


def nominal_action_tip_path(
    X_WO: RigidTransform,
    action: Action,
    settle_time: float,
    transfer_height: float,
) -> ActionTipPath:
    """Sample push_maneuver's tip path from the standoff point through the
    push, the release, the lift, and the hold.  Each leg is straight at
    constant speed and the tip rests during the hold.  Times are seconds from
    the standoff; the measurement window closes `settle_time` after the push
    segment, wherever that falls along the path."""
    maneuver = push_maneuver(X_WO, action, transfer_height)
    # From the standoff, where the dwell holds the tip: push, release, lift,
    # then the hold at the lift's end.
    poses = [pose for pose, _duration in maneuver[LEG_DWELL:]]
    waypoints = np.column_stack(
        [*(pose.translation() for pose in poses), poses[-1].translation()]
    )
    durations = np.array(
        [duration for _pose, duration in maneuver[LEG_PUSH:]] + [HOLD_TIME]
    )
    breaks = np.concatenate([[0.0], np.cumsum(durations)])

    push_end_time = float(breaks[1])
    window_end = float(np.round(push_end_time + settle_time, TIME_DECIMALS))
    times = _path_times(breaks, float(breaks[-1]))
    positions = np.stack(
        [np.interp(times, breaks, waypoints[axis]) for axis in range(3)]
    )
    leg = np.clip(
        np.searchsorted(breaks, times, side="left") - 1, 0, len(durations) - 1
    )
    leg_velocities = np.diff(waypoints, axis=1) / durations
    return ActionTipPath(
        tip_times=times,
        tip_positions=positions,
        tip_velocities=leg_velocities[:, leg],
        push_end_time=push_end_time,
        window_end=window_end,
        action_end=float(breaks[-1]),
        window_samples=int(np.count_nonzero(times <= window_end)),
    )


def path_downsample(tip_times: np.ndarray, tip_positions: np.ndarray) -> np.ndarray:
    """Resample the (3, n) tip path at PATH_SAMPLES uniform window times —
    the conditioning block the surrogate sees (and the predictor rebuilds
    from a measured path the same way)."""
    samples = np.linspace(0.0, float(tip_times[-1]), PATH_SAMPLES)
    return np.vstack(
        [np.interp(samples, tip_times, tip_positions[row]) for row in range(3)]
    )


def mlp_forward(weights: MLPWeights, x: np.ndarray) -> np.ndarray:
    """Run the (N, 49) inputs `x` through one member's ReLU MLP."""
    activations = x
    last = len(weights) - 1
    for i, (weight, bias) in enumerate(weights):
        activations = activations @ weight.T + bias
        if i < last:
            activations = np.maximum(activations, 0.0)
    return activations


def save_ensemble(path: Path, ensemble: SurrogateEnsemble) -> None:
    """Write the ensemble's statistics and per-member weights to `path`."""
    arrays: dict[str, np.ndarray] = {
        "input_mean": ensemble.input_mean,
        "input_std": ensemble.input_std,
        "output_mean": ensemble.output_mean,
        "output_std": ensemble.output_std,
    }
    for i, member in enumerate(ensemble.members):
        for j, (weight, bias) in enumerate(member):
            arrays[f"member{i}_layer{j}_weight"] = weight
            arrays[f"member{i}_layer{j}_bias"] = bias
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, allow_pickle=True, **arrays)


def load_ensemble(path: Path) -> SurrogateEnsemble:
    """Read an ensemble written by save_ensemble."""
    with np.load(path) as data:
        names = sorted(key for key in data.files if key.startswith("member"))
        members: list[MLPWeights] = []
        for i in range(count_members(names)):
            layers: list[Layer] = []
            j = 0
            while f"member{i}_layer{j}_weight" in data:
                # Inference runs in float32: about twice as fast, and its
                # rounding sits far below the surrogate's own error.
                layers.append(
                    (
                        data[f"member{i}_layer{j}_weight"].astype(np.float32),
                        data[f"member{i}_layer{j}_bias"].astype(np.float32),
                    )
                )
                j += 1
            members.append(tuple(layers))
        return SurrogateEnsemble(
            input_mean=data["input_mean"],
            input_std=data["input_std"],
            output_mean=data["output_mean"],
            output_std=data["output_std"],
            members=tuple(members),
        )


def count_members(names: Sequence[str]) -> int:
    """Count distinct `memberN_layer0_weight` keys in a sorted name list."""
    return len({name.split("_")[0] for name in names})


def feature_start(start: np.ndarray) -> np.ndarray:
    """A start pose or planar state (..., 3 or 6; x, y, yaw first) as the
    surrogate's input sees it: yaw in [0, 2*pi), the convention the training
    data's start poses use.
    Observations and the environment report yaw in [-pi, pi), which the
    network never saw for negative yaws."""
    features = np.array(start, dtype=float)
    features[..., 2] = np.mod(features[..., 2], 2.0 * np.pi)
    return features


def surrogate_inputs(
    etas: Sequence[Eta],
    start: np.ndarray,
    contact_position: np.ndarray,
    contact_normal: np.ndarray,
    speed: float,
    time: float,
    path_block: np.ndarray,
) -> np.ndarray:
    """The ensemble's (len(etas), N_INPUTS) inputs: each eta with the shared
    start planar state (6; world x, y, yaw, velocities), the body-frame
    contact point and inward normal, the commanded speed (m/s) and contact
    time (s), and the downsampled (32,) world-frame tip path block."""
    shared = np.concatenate(
        [
            feature_start(start),
            contact_position,
            contact_normal,
            [speed, time],
            path_block,
        ]
    )
    eta_rows = np.array([eta.as_array() for eta in etas])
    return np.concatenate([eta_rows, np.tile(shared, (len(etas), 1))], axis=1)


def world_measurement(start: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """The estimator's world-frame measurements (N, 6) from the ensemble's
    start-frame outputs (N, 6): pose at the window's end, velocity at the
    push segment's end."""
    c, s = np.cos(start[2]), np.sin(start[2])
    to_world = np.array([[c, s], [-s, c]])
    end_xy = start[:2] + outputs[:, :2] @ to_world
    end_yaw = wrap_angle(start[2] + outputs[:, 2])
    velocity_xy = outputs[:, 3:5] @ to_world
    return np.column_stack([end_xy, end_yaw, velocity_xy, outputs[:, 5]])


def measured_path_block(obs: PushObservation) -> np.ndarray:
    """Downsample the observation's measured tip path to PATH_SAMPLES
    uniform window times — the surrogate's path-conditioning block, the
    same resampling the dataset's tip_path columns were built with."""
    samples = np.linspace(0.0, obs.duration, PATH_SAMPLES)
    return np.stack(
        [np.interp(samples, obs.tip_times, obs.tip_positions[row]) for row in range(2)]
    ).reshape(-1)


class SurrogatePredictor:
    """The estimator's DynamicsModel at surrogate speed: it predicts the
    measurement of `obs` under each eta with one batched ensemble call."""

    def __init__(self, ensemble: SurrogateEnsemble) -> None:
        self.ensemble = ensemble

    def predict(self, etas: Sequence[Eta], obs: PushObservation) -> np.ndarray:
        """Predict the push's measurement under each of `etas`."""
        start = obs.start.planar()
        action = obs.action
        return self.predict_from_parts(
            etas,
            start,
            action.contact.position,
            action.contact.normal,
            action.velocity,
            action.time,
            measured_path_block(obs),
        )

    def predict_from_parts(
        self,
        etas: Sequence[Eta],
        start: np.ndarray,
        contact_position: np.ndarray,
        contact_normal: np.ndarray,
        speed: float,
        time: float,
        path_block: np.ndarray,
    ) -> np.ndarray:
        """Predict the measurement under each of `etas` from plain arrays
        (see surrogate_inputs).  The environment calls this directly and
        never builds PushObservation objects."""
        inputs = surrogate_inputs(
            etas, start, contact_position, contact_normal, speed, time, path_block
        )
        return world_measurement(start, self.ensemble.predict(inputs))
