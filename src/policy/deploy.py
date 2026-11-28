"""The trained probing policy in the Drake world: a Selector for the shared
probing loop (control.probing.run_probing)."""

import numpy as np
from sb3_contrib import MaskablePPO

from src.control.probing import ProbingState, Selector
from src.policy.env import ProbingEnvConfig, select_push_action
from src.world.action import Action


def policy_selector(
    model: MaskablePPO,
    config: ProbingEnvConfig,
    min_pushes: int = 0,
    observed_shift: np.ndarray | None = None,
) -> Selector:
    """The trained policy's masked choice from the probing lattice, or "stop".
    Stopping is masked until `min_pushes` pushes are done. Lattice pushes that
    share a contact point with a failed arm plan at this decision are masked.
    `observed_shift` (2,) m shifts the positions the policy sees (see
    select_push_action)."""

    def select(state: ProbingState, failed: list[Action]) -> Action | str:
        assert state.belief is not None, "the policy chooses from the belief"
        excluded = np.zeros(config.n_push_actions, dtype=bool)
        for action in failed:
            excluded |= np.all(
                np.isclose(config.contact_positions, action.contact.position), axis=1
            )
        chosen = select_push_action(
            model,
            state.belief,
            np.asarray(state.obj.pose, dtype=float),
            state.last_measurement,
            state.last_action,
            state.pushes,
            config,
            excluded,
            allow_stop=state.pushes >= min_pushes,
            observed_shift=observed_shift,
        )
        return "stop" if chosen is None else config.actions[chosen]

    return select
