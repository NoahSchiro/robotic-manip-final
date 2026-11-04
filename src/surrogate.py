"""The trained surrogate: an ensemble of standardized MLPs predicting a
push's outcome from (eta, start state, action), and its adapter to the
estimator's DynamicsModel protocol.

The ensemble is trained on `gen-surrogate-data`'s dataset (Drake-predicted
outcomes of sampled pushes) and saved as one .npz of weights and
standardization statistics, so inference needs only numpy.

Input columns (65), in order:
    eta (mu, bx, by) — the hypothesis
    start pose (x, y, yaw) — world
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

from .observation import PushObservation, wrap_angle
from .params import Eta

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
        """Predict standardized-to-raw outputs for the (N, 17) inputs `x`:
        the members' mean, or one member's output."""
        standardized = (x - self.input_mean) / self.input_std
        if member is None:
            outputs = [mlp_forward(w, standardized) for w in self.members]
            raw = np.mean(np.stack(outputs), axis=0)
        else:
            raw = mlp_forward(self.members[member], standardized)
        return np.asarray(raw * self.output_std + self.output_mean)


def mlp_forward(weights: MLPWeights, x: np.ndarray) -> np.ndarray:
    """Run the (N, 17) inputs `x` through one member's ReLU MLP."""
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
                layers.append(
                    (
                        data[f"member{i}_layer{j}_weight"],
                        data[f"member{i}_layer{j}_bias"],
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
        shared = np.concatenate(
            [
                start,
                action.contact.position,
                action.contact.normal,
                [action.velocity, action.time],
                measured_path_block(obs),
            ]
        )
        eta_rows = np.array([eta.as_array() for eta in etas])
        inputs = np.concatenate([eta_rows, np.tile(shared, (len(etas), 1))], axis=1)
        outputs = self.ensemble.predict(inputs)

        # Start-frame outputs to the world-frame measurement: pose at the
        # window's end, velocity at the push segment's end.
        c, s = np.cos(start[2]), np.sin(start[2])
        to_world = np.array([[c, s], [-s, c]])
        end_xy = start[:2] + outputs[:, :2] @ to_world
        end_yaw = wrap_angle(start[2] + outputs[:, 2])
        velocity_xy = outputs[:, 3:5] @ to_world
        return np.column_stack([end_xy, end_yaw, velocity_xy, outputs[:, 5]])
