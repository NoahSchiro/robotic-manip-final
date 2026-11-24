"""Where generated files live, relative to the repo root.

data/reference/  tracked: inputs that are expensive to regenerate (recorded
                 truth-sim trials, the fitted tip-path family)
data/results/    tracked: every result the report cites
data/models/     ignored: trained artefacts, published as GitHub release
                 assets (surrogate ensemble, PPO policy)
data/work/       ignored: regenerable intermediates (surrogate shards, PPO
                 checkpoints, prediction caches, pilots)
"""

from pathlib import Path

DATA = Path("data")
REFERENCE = DATA / "reference"
RESULTS = DATA / "results"
MODELS = DATA / "models"
WORK = DATA / "work"

REPLAY_TRIALS = REFERENCE / "replay_trials.npz"
PATH_FAMILY = REFERENCE / "path_family.npz"
SURROGATE_ENSEMBLE = MODELS / "surrogate_ensemble.npz"
PPO_POLICY = MODELS / "ppo_policy.zip"
SURROGATE_SHARDS = WORK / "surrogate"
PPO_RUNS = WORK / "ppo"
