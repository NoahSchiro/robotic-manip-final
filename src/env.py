"""The PPO training environment: planar pushing with unknown parameters,
run entirely on the trained surrogate. No Drake simulation in the loop
(pydrake loads transitively for geometry constants, construction-time
only).

One episode: a true eta and a start pose are sampled, the filter starts
uniform over the grid, and the agent repeatedly picks a push from the
push-space lattice (the MPC's candidates) or `stop`.  Each push executes a sampled arm-lag path, whose
outcome a sampled ensemble member predicts with deployment-scale
measurement noise (`CALIBRATED_R_SURROGATE` minus the members' own
spread).  The filter updates from the noisy measurement, and the block
chains from the noisily measured pose.

Rewards:
- push: `entropy_weight * (H_before - H_after) - step_cost`
- stop, or a forced stop at `max_pushes`:
  `-||(eta_map - eta*) / spans||` — the MAP's normalized estimation
  error, what MPC consumes
- leaving the workspace: the stop reward minus `out_of_bounds_penalty`

Action space: Discrete(len(actions) + 1) over the push space's lattice
(`PushSpace.grid(PROBE_SPACING, PROBE_SPEEDS, PROBE_DISTANCES)`).  The last
action is `stop` and is never masked; pushes are masked when the arm cannot
reach their approach or end point (push_space.reachable).
"""

import typing
from dataclasses import dataclass
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium import spaces
from pydrake.all import RigidTransform, RotationMatrix
from sb3_contrib import MaskablePPO
from torch.distributions import Categorical

from .action import Action
from .estimator import CALIBRATED_R_SURROGATE, GridBayesFilter
from .geometry import SurfacePart
from .iiwa import transfer_height
from .object import Object, Rod
from .observation import wrap_angle
from .params import Eta, GridSpec, grid_spans, make_grid, sample_eta
from .path_family import PathFamily, load_path_family, perturb_tip_positions
from .push_space import make_push_space, reachable_mask
from .surrogate import (
    SurrogateEnsemble,
    SurrogatePredictor,
    load_ensemble,
    nominal_tip_path,
    path_downsample,
    surrogate_inputs,
    world_measurement,
)

# The probing lattice: the MPC's candidates (contacts about 1.2 cm apart,
# 3 speeds x 3 distances per contact).
PROBE_SPACING = 0.012  # m
PROBE_SPEEDS = 3
PROBE_DISTANCES = 3
SETTLE_TIME = 0.3  # s, matching the dataset's windows
ROD_MASS = 0.2  # kg, as everywhere

WORKSPACE_CENTER = (0.55, 0.0)
WORKSPACE_RADIUS = 0.25  # m, the block center's bound

MAX_PUSHES = 12
ENTROPY_WEIGHT = 0.1
STEP_COST = 0.05
OUT_OF_BOUNDS_PENALTY = 1.0


@dataclass(frozen=True)
class ProbingEnvConfig:
    """Everything one environment instance runs on.  Built once by
    make_probing_env_config; the env's steps are pure numpy.  Push action i
    is actions[i]; the arrays hold its body-frame contact point and inward
    normal (A, 3), speed (A,) m/s and contact time (A,) s."""

    ensemble: SurrogateEnsemble | None
    grid: list[Eta]
    actions: tuple[Action, ...]
    contact_positions: np.ndarray
    contact_normals: np.ndarray
    speeds: np.ndarray
    times: np.ndarray
    family: PathFamily | None
    transfer_height: float  # m, the tip's height for transfer and lift legs
    max_pushes: int = MAX_PUSHES
    entropy_weight: float = ENTROPY_WEIGHT
    step_cost: float = STEP_COST
    out_of_bounds_penalty: float = OUT_OF_BOUNDS_PENALTY

    @property
    def n_push_actions(self) -> int:
        return len(self.actions)

    @property
    def n_actions(self) -> int:
        return self.n_push_actions + 1  # the last action is stop

    def action_vector(self, index: int) -> np.ndarray:
        """Push `index` as the observation's (8,) last-action vector: contact
        point (3), inward normal (3), speed m/s, contact time s."""
        return np.concatenate(
            [
                self.contact_positions[index],
                self.contact_normals[index],
                [self.speeds[index], self.times[index]],
            ]
        )


def make_probing_env_config(
    ensemble_path: Path | None,
    rod_mass: float = ROD_MASS,
    family_path: Path | None = Path("data/path_family.npz"),
) -> ProbingEnvConfig:
    """Build the hypothesis grid and the probing lattice.  The ensemble and
    path family are needed only to run the environment."""
    reference = Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, rod_mass))
    space = make_push_space(reference.parts, reference.height)
    actions = tuple(space.grid(PROBE_SPACING, PROBE_SPEEDS, PROBE_DISTANCES))
    return ProbingEnvConfig(
        ensemble=load_ensemble(ensemble_path) if ensemble_path is not None else None,
        grid=make_grid(GridSpec(), reference.parts),
        actions=actions,
        contact_positions=np.array([a.contact.position for a in actions]),
        contact_normals=np.array([a.contact.normal for a in actions]),
        speeds=np.array([a.velocity for a in actions]),
        times=np.array([a.time for a in actions]),
        family=load_path_family(family_path) if family_path is not None else None,
        transfer_height=transfer_height(reference),
    )


def member_spread(
    ensemble: SurrogateEnsemble,
    grid: list[Eta],
    config: ProbingEnvConfig,
    rows: int = 64,
) -> np.ndarray:
    """Estimate the (6,) std of the members' disagreement over the
    measurement, on random lattice pushes from random workspace poses.  The
    env's injected noise is R minus this spread squared, since the
    truth-vs-predictor mismatch already contributes it."""
    rng = np.random.default_rng(0)
    inputs = []
    for _ in range(rows):
        index = int(rng.integers(config.n_push_actions))
        pose = np.array(
            [
                rng.uniform(0.45, 0.65),
                rng.uniform(-0.1, 0.1),
                rng.uniform(-np.pi, np.pi),
            ]
        )
        path = nominal_tip_path(
            planar_transform(pose),
            config.actions[index],
            SETTLE_TIME,
            config.transfer_height,
        )
        inputs.append(
            surrogate_inputs(
                grid,
                np.array([*pose, 0.0, 0.0, 0.0]),
                config.contact_positions[index],
                config.contact_normals[index],
                float(config.speeds[index]),
                float(config.times[index]),
                path_downsample(path.tip_times, path.tip_positions)[:2].reshape(-1),
            )
        )
    # One push's 405 rows at a time keeps the peak memory to one batch.
    spreads = [
        np.stack(
            [ensemble.predict(rows_in, member=m) for m in range(len(ensemble.members))]
        ).std(axis=0)
        for rows_in in inputs
    ]
    return np.concatenate(spreads).mean(axis=0)


def sample_episode_conditions(
    rng: np.random.Generator, parts: list[SurfacePart]
) -> tuple[Eta, np.ndarray]:
    """The true eta and start pose from `rng` — the exact sampling the
    env's reset uses, so paired evaluations (surrogate and Drake) see
    identical episode conditions."""

    eta = sample_eta(rng, GridSpec(), parts)
    rho = 0.1 * np.sqrt(rng.uniform())
    theta = rng.uniform(-np.pi, np.pi)
    pose = np.array(
        [
            WORKSPACE_CENTER[0] + rho * np.cos(theta),
            WORKSPACE_CENTER[1] + rho * np.sin(theta),
            rng.uniform(-np.pi, np.pi),
        ]
    )
    return eta, pose


def observation_vector(
    belief: GridBayesFilter,
    pose: np.ndarray,
    last_measurement: np.ndarray,
    last_action: np.ndarray,
    pushes: int,
) -> np.ndarray:
    """The policy's observation from a filter state (shared with the
    Drake-world transfer evaluation)."""
    probs = belief.probs
    log_probs = np.log(np.maximum(probs, 1e-8))
    return np.asarray(
        np.concatenate(
            [
                pose,
                log_probs,
                [belief.entropy(), probs.max()],
                belief.map_eta().as_array(),
                belief.mean_eta().as_array(),
                last_measurement,
                last_action,
                [float(pushes)],
            ]
        ),
        dtype=np.float32,
    )


def reachability_mask(pose: np.ndarray, config: ProbingEnvConfig) -> np.ndarray:
    """Mask over push actions: approach and push-end tip targets inside the
    reach band.  The stop action is handled by the callers."""
    return reachable_mask(
        pose,
        config.contact_positions[:, :2],
        config.contact_normals[:, :2],
        config.speeds * config.times,
    )


def select_push_action(
    model: MaskablePPO,
    belief: GridBayesFilter,
    pose: np.ndarray,
    last_measurement: np.ndarray,
    last_action: np.ndarray,
    pushes: int,
    config: ProbingEnvConfig,
    excluded: np.ndarray,
) -> int | None:
    """The deployment action selection: the trained policy picks the next
    push index from the lattice under the reachability mask, never one of
    the `excluded` (A,) pushes.  None when the policy stops."""
    obs = observation_vector(belief, pose, last_measurement, last_action, pushes)
    mask = reachability_mask(pose, config) & ~excluded
    chosen = policy_decision(model, obs, np.append(mask, True))
    return None if chosen == config.n_actions - 1 else chosen


def policy_decision(model: MaskablePPO, obs: np.ndarray, mask: np.ndarray) -> int:
    """The policy's deterministic choice under `mask` (stop last): stop when
    its probability exceeds that of all pushes together, else the most
    probable push.  With one stop action against hundreds of pushes, the
    single most probable action is stop long before the policy actually
    prefers stopping, so a plain argmax would stop too early."""
    tensor, _ = model.policy.obs_to_tensor(obs)
    distribution = model.policy.get_distribution(tensor, action_masks=mask)
    categorical = distribution.distribution
    assert isinstance(categorical, Categorical), "a Discrete action space"
    probs = categorical.probs.detach().cpu().numpy().reshape(-1)
    stop = len(probs) - 1
    if probs[stop] >= probs[:stop].sum():
        return stop
    return int(np.argmax(probs[:stop]))


def env_parts() -> list[tuple[float, float, float, float]]:
    """The block's surface parts, for sampling etas."""
    return Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, ROD_MASS)).parts


def planar_transform(pose: np.ndarray) -> RigidTransform:
    """A world transform from (x, y, yaw), the block resting on the table."""
    return RigidTransform(
        RotationMatrix.MakeZRotation(pose[2]),
        np.array([pose[0], pose[1], 0.0]),
    )


class ProbingEnv(gymnasium.Env):
    """Gymnasium environment wrapping the surrogate world.  Deterministic
    in `reset`'s seed; every step is pure numpy."""

    metadata: typing.ClassVar[dict[str, list[str]]] = {"render_modes": []}  # type: ignore[misc]

    def __init__(self, config: ProbingEnvConfig) -> None:
        super().__init__()
        if config.ensemble is None or config.family is None:
            raise ValueError(
                "the surrogate environment requires an ensemble and path family"
            )
        self.config = config
        self.grid_array = np.array([eta.as_array() for eta in config.grid])
        self.spans = grid_spans(config.grid)
        self.ensemble = config.ensemble
        self.family = config.family
        self.predictor = SurrogatePredictor(config.ensemble)

        # Measurement noise: the deployment error minus the members' own
        # spread, shrunk along R's correlation structure so the result
        # stays positive definite.
        spread = member_spread(self.ensemble, config.grid, config)
        r_diag = np.diag(CALIBRATED_R_SURROGATE)
        corr = CALIBRATED_R_SURROGATE / np.sqrt(np.outer(r_diag, r_diag))
        shrunk = np.maximum(r_diag - spread**2, 1e-12)
        self._noise_chol = np.linalg.cholesky(corr * np.sqrt(np.outer(shrunk, shrunk)))

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._observation_size(),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(config.n_actions)
        self._parts = env_parts()
        self.belief: GridBayesFilter | None = None
        self._true_eta: Eta | None = None
        self._pose = np.zeros(3)
        self._pushes = 0
        self._last_measurement = np.zeros(6)
        self._last_action = np.zeros(8)

    def _observation_size(self) -> int:
        return 3 + len(self.config.grid) + 2 + 3 + 3 + 6 + 8 + 1

    def _reachable(self) -> np.ndarray:
        """Mask over push actions: the approach and push-end tip targets
        must both sit inside the reach band."""
        return reachability_mask(self._pose, self.config)

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(self.config.n_actions, dtype=bool)
        mask[: self.config.n_push_actions] = self._reachable()
        mask[-1] = True  # stop is always available
        return mask

    def _observation(self) -> np.ndarray:
        assert self.belief is not None
        probs = self.belief.probs
        log_probs = np.log(np.maximum(probs, 1e-8))
        return np.asarray(
            np.concatenate(
                [
                    self._pose,
                    log_probs,
                    [self.belief.entropy(), probs.max()],
                    self.belief.map_eta().as_array(),
                    self.belief.mean_eta().as_array(),
                    self._last_measurement,
                    self._last_action,
                    [float(self._pushes)],
                ]
            ),
            dtype=np.float32,
        )

    def _info(self, reason: str) -> dict[str, object]:
        assert self.belief is not None and self._true_eta is not None
        return {
            "entropy": self.belief.entropy(),
            "map_eta": self.belief.map_eta().as_array(),
            "true_eta": self._true_eta.as_array(),
            "estimation_error": self.belief.estimation_error(self._true_eta),
            "pushes": self._pushes,
            "reason": reason,
        }

    def _stop_reward(self) -> float:
        assert self.belief is not None and self._true_eta is not None
        error = (
            self.belief.map_eta().as_array() - self._true_eta.as_array()
        ) / self.spans
        return -float(np.linalg.norm(error))

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        self._true_eta, self._pose = sample_episode_conditions(
            self.np_random, self._parts
        )
        self._pushes = 0
        self._last_measurement = np.zeros(6)
        self._last_action = np.zeros(8)
        self.belief = GridBayesFilter(
            self.config.grid,
            self.predictor,
            CALIBRATED_R_SURROGATE,
            nu=3.0,
        )
        return self._observation(), self._info("reset")

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        assert self.belief is not None and self._true_eta is not None
        if action == self.config.n_actions - 1:
            return (
                self._observation(),
                self._stop_reward(),
                True,
                False,
                self._info("stop"),
            )

        index = int(action)
        push = self.config.actions[index]
        entropy_before = self.belief.entropy()

        # The executed path: the commanded maneuver run with this push's
        # sampled arm lag.
        path = nominal_tip_path(
            planar_transform(self._pose),
            push,
            SETTLE_TIME,
            self.config.transfer_height,
        )
        path.tip_positions[:] = perturb_tip_positions(
            path.tip_positions,
            path.tip_times,
            path.window_end,
            self.family,
            self.np_random,
        )
        path_block = path_downsample(path.tip_times, path.tip_positions)[:2].reshape(-1)
        start = np.array([*self._pose, 0.0, 0.0, 0.0])
        parts = (
            self.config.contact_positions[index],
            self.config.contact_normals[index],
            float(self.config.speeds[index]),
            float(self.config.times[index]),
            path_block,
        )

        # Truth outcome: a sampled member's prediction of the executed path,
        # plus deployment-scale measurement noise (world frame).
        member = int(self.np_random.integers(len(self.ensemble.members)))
        truth = world_measurement(
            start,
            self.ensemble.predict(
                surrogate_inputs([self._true_eta], start, *parts), member=member
            ),
        )[0]
        noise = self._noise_chol @ self.np_random.standard_normal(6)
        measured = truth + noise
        measured[2] = wrap_angle(measured[2])

        # The filter predicts under every grid hypothesis from the same
        # measured path, then updates from the noisy measurement.
        predicted = self.predictor.predict_from_parts(self.config.grid, start, *parts)
        self.belief.update_from_predictions(measured, predicted)

        self._pose = np.array(measured[:3])
        self._pushes += 1
        self._last_measurement = measured
        self._last_action = self.config.action_vector(index)
        reward = (
            self.config.entropy_weight * (entropy_before - self.belief.entropy())
            - self.config.step_cost
        )

        out_of_bounds = (
            np.linalg.norm(self._pose[:2] - np.array(WORKSPACE_CENTER))
            > WORKSPACE_RADIUS
        )
        if out_of_bounds:
            reward += self._stop_reward() - self.config.out_of_bounds_penalty
            return self._observation(), reward, True, False, self._info("out_of_bounds")
        if self._pushes >= self.config.max_pushes:
            reward += self._stop_reward()
            return self._observation(), reward, True, False, self._info("max_pushes")
        return self._observation(), reward, False, False, self._info("push")

    def action_masks(self) -> np.ndarray:
        """MaskablePPO's hook: which actions are legal this step."""
        return self._action_mask()
