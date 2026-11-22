"""The surrogate probing environment: determinism, mask and reward
behavior, without running Drake."""

import typing
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from torch.distributions import Categorical

from src import paths
from src.env import (
    ProbingEnv,
    ProbingEnvConfig,
    make_probing_env_config,
    policy_decision,
    reachability_mask,
)
from src.push_space import reachable

ENSEMBLE = Path(__file__).resolve().parents[1] / paths.SURROGATE_ENSEMBLE


@pytest.fixture(scope="module")
def config() -> ProbingEnvConfig:
    return make_probing_env_config(ENSEMBLE)


def run(
    config: ProbingEnvConfig,
    seed: int,
    pushes: int,
    action_picker: Callable[[np.ndarray], int],
) -> tuple[float, dict[str, object]]:
    """Run one episode for `pushes` pushes then stop, returning the reward
    sum and the final info."""
    env = ProbingEnv(config)
    env.reset(seed=seed)
    total = 0.0
    info: dict[str, object] = {}
    for _ in range(pushes):
        mask = env.action_masks()
        legal = np.where(mask[:-1])[0]
        assert len(legal) > 0
        _, reward, terminated, _, info = env.step(int(legal[action_picker(legal)]))
        total += reward
        if terminated:
            return total, info
    _, reward, _, _, info = env.step(config.n_actions - 1)
    total += reward
    return total, info


def test_reset_is_deterministic(config: ProbingEnvConfig) -> None:
    a, b = ProbingEnv(config), ProbingEnv(config)
    obs_a, info_a = a.reset(seed=11)
    obs_b, info_b = b.reset(seed=11)
    assert np.array_equal(obs_a, obs_b)
    assert np.array_equal(
        typing.cast(np.ndarray, info_a["true_eta"]),
        typing.cast(np.ndarray, info_b["true_eta"]),
    )
    assert a._pose.tolist() == b._pose.tolist()


def test_entropy_collapses_across_pushes(config: ProbingEnvConfig) -> None:
    env = ProbingEnv(config)
    env.reset(seed=3)
    assert env.belief is not None
    initial = env.belief.entropy()
    info: dict[str, object] = {}
    for k in range(6):
        mask = env.action_masks()
        legal = np.where(mask[:-1])[0]
        _, _, terminated, _, info = env.step(int(legal[k % len(legal)]))
        if terminated:
            break
    assert typing.cast(float, info["entropy"]) < initial - 1.0


def test_stop_is_never_masked(config: ProbingEnvConfig) -> None:
    env = ProbingEnv(config)
    env.reset(seed=5)
    for _ in range(3):
        assert env.action_masks()[-1]
        env.step(int(np.where(env.action_masks()[:-1])[0][0]))
    assert env.belief is not None


def test_immediate_stop_is_worse_than_probing(config: ProbingEnvConfig) -> None:
    """The stop reward must not favor stopping before any information is
    gathered."""
    immediate = np.array(
        [run(config, seed, 0, lambda legal: 0)[0] for seed in range(8)]
    )
    probing = np.array([run(config, seed, 4, lambda legal: 0)[0] for seed in range(8)])
    assert probing.mean() > immediate.mean()


def test_episodes_terminate_within_the_push_bound(config: ProbingEnvConfig) -> None:
    """Pushing repeatedly must end the episode within max_pushes, with a
    termination reason."""
    env = ProbingEnv(config)
    env.reset(seed=1)
    terminated = False
    info: dict[str, object] = {}
    for _ in range(config.max_pushes + 1):
        mask = env.action_masks()
        legal = np.where(mask[:-1])[0]
        action = config.n_actions - 1 if len(legal) == 0 else int(legal[-1])
        _, _, terminated, _, info = env.step(action)
        if terminated:
            break
    assert terminated
    # "stop" happens when the block slides out of every site's reach.
    assert info["reason"] in ("max_pushes", "out_of_bounds", "stop")


def test_actions_are_the_push_space_lattice() -> None:
    lattice = make_probing_env_config(None, family_path=None)
    assert lattice.n_push_actions == 396
    assert lattice.n_actions == 397
    assert np.all((lattice.times >= 0.1 - 1e-9) & (lattice.times <= 1.0 + 1e-9))
    assert np.all(lattice.speeds * lattice.times <= 0.1 + 1e-9)
    assert lattice.action_vector(5).tolist() == [
        *lattice.contact_positions[5],
        *lattice.contact_normals[5],
        lattice.speeds[5],
        lattice.times[5],
    ]


def test_reachability_mask_matches_push_space_reachable() -> None:
    lattice = make_probing_env_config(None, family_path=None)
    pose = np.array([0.42, 0.18, 2.1])
    mask = reachability_mask(pose, lattice)
    expected = [reachable(pose, action) for action in lattice.actions]
    assert mask.tolist() == expected
    assert 0 < mask.sum() < len(mask)


class FixedPolicy:
    """Stands in for a MaskablePPO policy with fixed action probabilities."""

    def __init__(self, probs: list[float]) -> None:
        self.probs = torch.tensor([probs])

    def obs_to_tensor(self, obs: np.ndarray) -> tuple[torch.Tensor, bool]:
        return torch.as_tensor(obs), False

    def get_distribution(self, _obs: torch.Tensor, action_masks: np.ndarray) -> object:
        probs = self.probs * torch.as_tensor(action_masks)
        return SimpleNamespace(distribution=Categorical(probs=probs / probs.sum()))


def decide(probs: list[float], mask: list[bool] | None = None) -> int:
    model = typing.cast(MaskablePPO, SimpleNamespace(policy=FixedPolicy(probs)))
    full = np.ones(len(probs), dtype=bool) if mask is None else np.array(mask)
    return policy_decision(model, np.zeros(3, dtype=np.float32), full)


def test_decision_pushes_when_pushing_is_more_likely_than_stopping() -> None:
    # Stop is the single most probable action, but pushing totals 0.7.
    assert decide([0.2, 0.22, 0.28, 0.3]) == 2
    assert decide([0.1, 0.1, 0.2, 0.6]) == 3
    # A masked push never wins.
    assert decide([0.5, 0.1, 0.1, 0.3], mask=[False, True, True, True]) == 3


def test_masked_stop_forces_the_best_push() -> None:
    # The policy would stop (0.7), but stopping is masked.
    assert decide([0.1, 0.2, 0.0, 0.7], mask=[True, True, True, False]) == 1
