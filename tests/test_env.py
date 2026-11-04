"""The surrogate probing environment: determinism, mask and reward
behavior, without running Drake."""

import typing
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from src.env import ProbingEnv, ProbingEnvConfig, make_probing_env_config

ENSEMBLE = Path(__file__).resolve().parents[1] / "data" / "surrogate_ensemble.npz"


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
