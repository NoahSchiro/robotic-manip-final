"""The PPO training environment: planar pushing with unknown parameters,
run entirely on the trained surrogate. No Drake simulation in the loop
(pydrake loads transitively for geometry constants, construction-time
only).

One episode: a true eta and a start pose are sampled, the filter starts
uniform over the grid, and the agent repeatedly picks a push from the
pruned library or `stop`.  Each push executes a sampled arm-lag path, whose
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

Action space: Discrete(n_sites * n_speeds * n_times + 1).  The last
action is `stop` and is never masked; sites are masked when their tip
targets fall outside the radial reach band the recorded reachable pushes
spanned.
"""

import typing
from dataclasses import dataclass
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium import spaces
from pydrake.all import RigidTransform, RotationMatrix
from sb3_contrib import MaskablePPO

from .action import Action
from .estimator import CALIBRATED_R_SURROGATE, GridBayesFilter
from .geometry import SurfaceNormal, SurfacePart
from .iiwa import STANDOFF, Pusher, clear_sites
from .object import Object, Rod
from .observation import wrap_angle
from .params import Eta, GridSpec, grid_spans, make_grid, sample_eta
from .path_family import PathFamily, load_path_family, perturb_tip_positions
from .surrogate import (
    SurrogateEnsemble,
    SurrogatePredictor,
    load_ensemble,
    nominal_tip_path,
    path_downsample,
)

N_PROBE_SITES = 32  # the pruned library's site count (MPC keeps all)
SPEEDS = (0.05, 0.1)
TIMES = (0.2, 0.4)
# Sites per meter of outline.  The dataset generator’s --push-density 2.0
# is per centimeter; converted (x100) it is 200/m.
PUSH_DENSITY = 200.0
SITE_CLEARANCE = 0.001  # m, as in the dataset
SETTLE_TIME = 0.1  # s, matching the dataset's windows
ROD_MASS = 0.2  # kg, as everywhere

WORKSPACE_CENTER = (0.55, 0.0)
WORKSPACE_RADIUS = 0.25  # m, the block center's bound
REACH_MIN, REACH_MAX = 0.39, 0.78  # m, tip radius band from the arm base

MAX_PUSHES = 12
ENTROPY_WEIGHT = 0.1
STEP_COST = 0.05
OUT_OF_BOUNDS_PENALTY = 1.0


@dataclass(frozen=True)
class ProbingEnvConfig:
    """Everything one environment instance runs on.  Built once by
    make_probing_env_config; the env's steps are pure numpy."""

    ensemble: SurrogateEnsemble | None
    grid: list[Eta]
    site_positions: np.ndarray  # (S, 3) body frame
    site_normals: np.ndarray  # (S, 3) body frame
    speeds: tuple[float, ...]
    times: tuple[float, ...]
    family: PathFamily | None
    tip_body_xy: np.ndarray  # (A, 2) approach-point body xy per push action
    normal_body_xy: np.ndarray  # (A, 2) inward normal xy per push action
    forward: np.ndarray  # (A,) approach + v*T distance per push action
    max_pushes: int = MAX_PUSHES
    entropy_weight: float = ENTROPY_WEIGHT
    step_cost: float = STEP_COST
    out_of_bounds_penalty: float = OUT_OF_BOUNDS_PENALTY

    @property
    def n_push_actions(self) -> int:
        return len(self.site_positions) * len(self.speeds) * len(self.times)

    @property
    def n_actions(self) -> int:
        return self.n_push_actions + 1  # the last action is stop


def prune_sites(positions: np.ndarray, keep: int = N_PROBE_SITES) -> np.ndarray:
    """Pick `keep` evenly spaced indices along the (outline-ordered) clear
    sites, so the pruned library stays spread over the footprint."""
    return np.linspace(0, len(positions) - 1, keep).astype(int)


def make_probing_env_config(
    ensemble_path: Path | None,
    keep: int = N_PROBE_SITES,
    rod_mass: float = ROD_MASS,
    push_density: float = PUSH_DENSITY,
    site_clearance: float = SITE_CLEARANCE,
    family_path: Path | None = Path("data/path_family.npz"),
) -> ProbingEnvConfig:
    """Build the hypothesis grid and pruned clear-site library. Optional
    ensemble and path-family assets are needed for surrogate execution."""
    spec = GridSpec()
    reference = Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, rod_mass))
    clear = clear_sites(
        reference.parts,
        reference.surface_normals(push_density),
        Pusher().radius,
        site_clearance,
    )
    picked = prune_sites(np.array([site.position for site in clear]), keep)
    picked_sites = [clear[i] for i in picked]
    n_speeds, n_times = len(SPEEDS), len(TIMES)
    return ProbingEnvConfig(
        ensemble=load_ensemble(ensemble_path) if ensemble_path is not None else None,
        grid=make_grid(spec, reference.parts),
        site_positions=np.array([site.position for site in picked_sites]),
        site_normals=np.array([site.normal for site in picked_sites]),
        speeds=SPEEDS,
        times=TIMES,
        family=load_path_family(family_path) if family_path is not None else None,
        tip_body_xy=np.repeat(
            np.array([site.position[:2] for site in picked_sites]),
            n_speeds * n_times,
            axis=0,
        ),
        normal_body_xy=np.repeat(
            np.array([site.normal[:2] for site in picked_sites]),
            n_speeds * n_times,
            axis=0,
        ),
        forward=np.tile(
            STANDOFF + np.repeat(SPEEDS, n_times) * np.tile(TIMES, n_speeds),
            len(picked_sites),
        ),
    )


def member_spread(
    ensemble: SurrogateEnsemble,
    grid: list[Eta],
    config: ProbingEnvConfig,
    rows: int = 64,
) -> np.ndarray:
    """Estimate the (6,) std of the members' disagreement over the
    measurement, on random in-distribution pushes.  The env's injected
    noise is R minus this spread squared, since the truth-vs-predictor
    mismatch already contributes it."""
    rng = np.random.default_rng(0)
    shared = []
    for _ in range(rows):
        site = int(rng.integers(len(config.site_positions)))
        speed = float(rng.choice(config.speeds))
        time = float(rng.choice(config.times))
        pose = planar_transform(
            np.array(
                [
                    rng.uniform(0.45, 0.65),
                    rng.uniform(-0.1, 0.1),
                    rng.uniform(-np.pi, np.pi),
                ]
            )
        )
        action = Action(
            contact=SurfaceNormal(
                position=config.site_positions[site],
                normal=config.site_normals[site],
            ),
            velocity=speed,
            time=time,
        )
        path = nominal_tip_path(pose, action, SETTLE_TIME)
        start = np.zeros(6)
        start[:2] = pose.translation()[:2]
        start[2] = np.arctan2(
            pose.rotation().matrix()[1, 0], pose.rotation().matrix()[0, 0]
        )
        shared.append(
            np.concatenate(
                [
                    [
                        rng.uniform(0.45, 0.65),
                        rng.uniform(-0.1, 0.1),
                        rng.uniform(-np.pi, np.pi),
                        0.0,
                        0.0,
                        0.0,
                    ],
                    config.site_positions[site],
                    config.site_normals[site],
                    [speed, time],
                    path_downsample(path.tip_times, path.tip_positions)[:2].reshape(-1),
                ]
            )
        )
    eta_rows = np.tile(np.array([e.as_array() for e in grid]), (rows, 1))
    inputs = np.concatenate(
        [eta_rows, np.repeat(np.array(shared), len(grid), axis=0)], axis=1
    )
    per_member = np.stack(
        [ensemble.predict(inputs, member=m) for m in range(len(ensemble.members))]
    )
    return per_member.std(axis=0).mean(axis=0)


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
    c, s = np.cos(pose[2]), np.sin(pose[2])
    rot = np.array([[c, -s], [s, c]])
    approach_body = config.tip_body_xy - STANDOFF * config.normal_body_xy
    push_end_body = approach_body + config.forward[:, None] * config.normal_body_xy
    approach = pose[:2] + approach_body @ rot.T
    push_end = pose[:2] + push_end_body @ rot.T
    r_approach = np.linalg.norm(approach, axis=1)
    r_push_end = np.linalg.norm(push_end, axis=1)
    inside = (r_approach >= REACH_MIN) & (r_approach <= REACH_MAX)
    inside &= (r_push_end >= REACH_MIN) & (r_push_end <= REACH_MAX)
    return np.asarray(inside, dtype=bool)


def select_push_action(
    model: MaskablePPO,
    belief: GridBayesFilter,
    pose: np.ndarray,
    last_measurement: np.ndarray,
    last_action: np.ndarray,
    pushes: int,
    config: ProbingEnvConfig,
    failed_sites: set[int],
) -> tuple[int, float, float] | None:
    """The deployment action selection: the trained policy picks the next
    push from the pruned library under the reachability mask, excluding
    sites whose IK failed.  None when the policy stops."""
    obs = observation_vector(belief, pose, last_measurement, last_action, pushes)
    mask = reachability_mask(pose, config).copy()
    per_site = len(config.speeds) * len(config.times)
    for site in failed_sites:
        mask[site * per_site : (site + 1) * per_site] = False
    full_mask = np.append(mask, True)
    action, _ = model.predict(obs, action_masks=full_mask, deterministic=True)
    chosen = int(action)
    if chosen == config.n_actions - 1:
        return None
    return split_action(chosen, config)


def split_action(action: int, config: ProbingEnvConfig) -> tuple[int, float, float]:
    """Split a push action index into (site index, speed, contact time)."""
    per_site = len(config.speeds) * len(config.times)
    site, rest = divmod(action, per_site)
    speed_idx, time_idx = divmod(rest, len(config.times))
    return site, config.speeds[speed_idx], config.times[time_idx]


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

        # Per (site, speed, time) body-frame tip targets: the approach and
        # push-end points, flattened in the action's index order.
        n_speeds, n_times = len(config.speeds), len(config.times)
        per_site_forward = STANDOFF + np.repeat(config.speeds, n_times) * np.tile(
            config.times, n_speeds
        )
        self._tip_body_xy = np.repeat(
            config.site_positions[:, :2], n_speeds * n_times, axis=0
        )
        self._normal_body_xy = np.repeat(
            config.site_normals[:, :2], n_speeds * n_times, axis=0
        )
        self._forward = np.tile(per_site_forward, len(config.site_positions))

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

        site, speed, time = split_action(int(action), self.config)
        entropy_before = self.belief.entropy()

        # The executed path: the commanded maneuver run with this push's
        # sampled arm lag.
        body_action = Action(
            contact=SurfaceNormal(
                position=self.config.site_positions[site],
                normal=self.config.site_normals[site],
            ),
            velocity=speed,
            time=time,
        )
        path = nominal_tip_path(planar_transform(self._pose), body_action, SETTLE_TIME)
        path.tip_positions[:] = perturb_tip_positions(
            path.tip_positions,
            path.tip_times,
            path.window_end,
            self.family,
            self.np_random,
        )
        path_block = path_downsample(path.tip_times, path.tip_positions)[:2].reshape(-1)

        # Truth outcome: the sampled member's prediction of the executed
        # path, plus deployment-scale measurement noise (world frame).
        start = np.array([*self._pose[:2], self._pose[2], 0.0, 0.0, 0.0])
        shared = np.concatenate(
            [
                start,
                self.config.site_positions[site],
                self.config.site_normals[site],
                [speed, time],
                path_block,
            ]
        )
        truth_inputs = np.concatenate([self._true_eta.as_array(), shared])[None, :]
        member = int(self.np_random.integers(len(self.ensemble.members)))
        truth = self.ensemble.predict(truth_inputs, member=member)[0]
        noise = self._noise_chol @ self.np_random.standard_normal(6)
        c, s = np.cos(self._pose[2]), np.sin(self._pose[2])
        to_world = np.array([[c, s], [-s, c]])
        measured = np.concatenate(
            [
                start[:2] + truth[:2] @ to_world + noise[:2],
                [wrap_angle(self._pose[2] + truth[2] + noise[2])],
                truth[3:5] @ to_world + noise[3:5],
                [truth[5] + noise[5]],
            ]
        )

        # The filter predicts under every grid hypothesis from the same
        # measured path, then updates from the noisy measurement.
        predicted = self.predictor.predict_from_parts(
            self.config.grid,
            start,
            self.config.site_positions[site],
            self.config.site_normals[site],
            speed,
            time,
            path_block,
        )
        self.belief.update_from_predictions(measured, predicted)

        self._pose = np.array(measured[:3])
        self._pushes += 1
        self._last_measurement = measured
        self._last_action = np.array(
            [
                *self.config.site_positions[site],
                *self.config.site_normals[site],
                speed,
                time,
            ]
        )
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


def _env_parts() -> list[tuple[float, float, float, float]]:
    """The block's surface parts, for sampling etas at reset."""
    return Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, ROD_MASS)).parts
