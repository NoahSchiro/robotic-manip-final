# robotic-manip-final

A KUKA iiwa14 with a stick pushes a T-shaped block across a table. The block's
friction and the position of a hidden rod inside it are unknown. The robot
estimates them from probing pushes with a grid Bayes filter, then pushes the
block to a goal pose with MPC. The project compares probing strategies (none,
random, and a learned PPO policy) on estimation and placement.

Definitions (units, frames, push space, timing, measurement, models) are in
[the system reference](docs/system.md).

## Setup

On Windows, run everything in WSL. 

```sh
uv sync
uv run pytest
```

### On the MIT Engaging cluster

The cluster's system libraries are older than Drake's Linux wheels need, so the
project runs inside an Apptainer image with `uv` and Python 3.14:

```sh
module load apptainer
apptainer pull ~/uv-py314.sif docker://ghcr.io/astral-sh/uv:python3.14-bookworm
apptainer exec ~/uv-py314.sif uv sync
```

Build the image and the environment inside an allocation with enough memory
(`salloc -p mit_normal -c 4 --mem=16G`); creating the image needs several GB.
Prefix commands with `apptainer exec ~/uv-py314.sif` (add `--nv` on GPU
nodes). Copy `data/path_family.npz` and `data/surrogate_ensemble.npz` over,
since `data/` is not in git. Batch scripts are in `scripts/cluster/`; create
`logs/` once before submitting.

## Usage

Every command is a `main.py` subcommand; `uv run python main.py <command> --help`
lists its options. Commands that open Meshcat print its URL; `--headless`
skips visualization.

### Watch the simulation

```sh
# One demo push
uv run python main.py simulate
```

### Estimate the unknown parameters

```sh
# Random probing pushes with a Bayes update after each
uv run python main.py simulate --estimate --pushes 5
# Fix the truth instead of sampling it from the seed
uv run python main.py simulate --estimate --pushes 5 --seed 123 --mu 0.8 --rod-x 0.0 --rod-y -0.04
# Let the trained policy choose the pushes
uv run python main.py simulate --estimate --policy ppo
# Update the belief with the fast surrogate instead of the Drake model (the truth stays Drake)
uv run python main.py simulate --estimate --predictor surrogate
```

Without `--headless`, the belief is shown after every push. In Meshcat, yellow
discs on the block's top face show where the rod might be (area ∝
probability), beside a red marker at the true rod. A matplotlib window charts
the belief over rod position and friction against the truth, with the entropy,
the most likely hypothesis's probability, and its error in grid steps.

To save a push-by-push belief figure:

```sh
# Run one headless episode in Drake and save the belief after each push
uv run python -m scripts.plot_belief record --seed 123 --pushes 3 --out data/belief_seed123.npz
# Plot it (numpy and matplotlib only)
uv run python -m scripts.plot_belief plot data/belief_seed123.npz --out docs/figures/belief_seed123.png
```

### Place the block with MPC

```sh
# No probing: MPC from the first push, estimating eta as it goes
uv run python main.py simulate --mpc
# Random probing first: 3 random pushes, then MPC
uv run python main.py simulate --mpc --probe-pushes 3
# Plan with the true eta (the controller alone)
uv run python main.py simulate --mpc --plan-with-true-eta
# Set the start and goal poses (x m, y m, yaw deg)
uv run python main.py simulate --mpc --start 0.55 -0.15 0 --goal 0.55 0.15 130
```

Before each push, the MPC predicts where every candidate push would leave the
block under the current best estimate of eta and takes the cheapest one the
arm can reach. Probe and MPC pushes share one budget (`--max-pushes`). An
episode succeeds within `--position-tolerance` and `--yaw-tolerance` of the
goal. Meshcat shows the goal as a translucent block and the predicted outcome
of each MPC push as a yellow one.

### Evaluate probing strategies

```sh
# Random probing only (no trained assets needed)
uv run python main.py evaluate --random-only --push-budget 3 --episodes 4 --seed 70000 --out data/random_report.json
# Trained policy vs random probing on the same episodes
uv run python main.py evaluate
uv run python main.py evaluation-report
# Narrated episodes of the trained policy
uv run python main.py demo
```

To score a trained policy in the surrogate world and see which pushes it chooses:

```sh
uv run python -m scripts.evaluate_policy --model data/ppo/ppo_final.zip --out data/ppo/evaluation
```

Each `evaluate` episode runs, on identical conditions in the Drake truth sim,
the trained policy until it stops, then two baselines with the same number of
pushes: random pushes from the policy's 396-push lattice, and random pushes
from its long, fast subset (at least 6.8 cm and 11 cm/s). Reports give each
episode's parameter errors, reward, push count, and stop reason. On the
cluster, `scripts/cluster/evaluate_transfer.sbatch` runs one episode per array
task (`FIXED_PUSHES=3` makes every strategy push exactly three times, the
policy choosing which pushes; `--fixed-pushes` does the same for
`evaluate`); combine them with:

```sh
uv run python -m scripts.merge_evaluations data/transfer/episode_*.json --out data/transfer/summary.json
```

### Check the arm motion

```sh
# Random pushes: transfer clearance, joint speeds, tracking, release
uv run python -m scripts.diagnose_motion --seed 123 --pushes 2 --out data/motion.json
# The validation push sets, with a Meshcat replay to open in a browser
uv run python -m scripts.diagnose_motion --plan scripts/motion_plans/extremes.json --friction 0.1 --out data/extremes_mu01.json --html data/extremes_mu01.html
uv run python -m scripts.diagnose_motion --plan scripts/motion_plans/stress.json --out data/stress.json
```

A plan file lists pushes by side index (0–7 counterclockwise around the outline,
starting with the crossbar's bottom end), position `s` along the side's usable
span (0–1), speed (m/s), and contact time (s); `--friction` overrides the
plan's table friction. `scripts/motion_plans/extremes.json` covers the corners
of the push space (fastest and slowest, shortest and longest, every corner
inset, cross-T transfers), run at friction 0.1, 0.5, and 0.9;
`scripts/motion_plans/stress.json` goes beyond the speed and distance limits.

Each push reports its peak joint speed as a fraction of the joint's limit, its
sideways, along-track, and vertical tracking errors, when the stick clears the
block after the push and whether it touches it again, and how far the block
moves after the measurement ends.

### Compare how informative pushes are

```sh
uv run python -m scripts.push_informativeness --out data/push_informativeness.json
# Plot saved results without re-simulating
uv run python -m scripts.push_informativeness --from-json data/push_informativeness.json --plot docs/figures
```

For every contact on the MPC grid, at matched push distances (1.5, 4.5, and
9 cm) and several speeds, this predicts the estimator's measurement under a
hypothesis and under its neighbour one grid step away in friction, rod x, or
rod y. It reports how far apart the two predictions are relative to the
estimator's noise (`CALIBRATED_R`): about 1 or more means one push noticeably
shifts the belief between them. Planning-only, about 3 minutes on 12 workers.

### Check the prediction model against the truth sim

```sh
uv run python -m scripts.model_bias --out data/model_bias.npz
```

Re-predicts every recorded replay push at the true parameters (at the model's
timestep and at the truth sim's), at the nearest grid hypothesis, and over a
friction sweep, then prints the mismatch by friction band, the mean residual
per measurement component, and the friction each trial's pushes favour. About
12 minutes on 12 workers.

### Data, calibration, and training

| Command | What it does |
| --- | --- |
| `calibrate` | Measures the estimator's prediction-error covariance R from truth-simulation pushes and prints the constant for `estimator.CALIBRATED_R` |
| `record-trials` | Records multi-push trials in the truth simulation to `data/replay_trials.npz` |
| `replay` | Scores the estimator on recorded trials and writes `data/replay_report.json`; `--save-predictions` caches the grid predictions so `--load-predictions` can rescore them (for example with another `--subcells`) in seconds |
| `calibrate-path-family` | Fits how the arm's executed tip path deviates from the commanded one, from recorded trials |
| `gen-surrogate-data` | Generates one shard of surrogate training data from the Drake prediction model, with pushes drawn uniformly from the push space (about 10 samples/s on 12 workers); give each shard its own `--seed` and `--out` |
| `train-surrogate` | Trains the surrogate's MLP ensemble on one or more shards (`--dataset a.npz b.npz ...`); `--device cuda` trains on the GPU |
| `train-ppo` | Trains the probing policy in the surrogate world, choosing among the MPC's 396 lattice pushes or stop; `--resume` continues from the latest checkpoint in `--out` |

## Code layout

`main.py` holds the command-line interface and assembles each run.

| Module | Contents |
| --- | --- |
| `src/object.py`, `src/geometry.py` | The T-shaped block, the table, and outline geometry |
| `src/iiwa.py`, `src/pdcontroller.py`, `src/scene.py` | Arm, stick, IK and trajectories; arm controller; Drake scene and Meshcat drawing |
| `src/action.py`, `src/push_space.py` | A push action; the push space every module samples from |
| `src/recording.py`, `src/observation.py` | The continuous truth simulation; what a push records and the measurement |
| `src/params.py`, `src/estimator.py`, `src/calibration.py` | Unknown parameters and hypothesis grid; grid Bayes filter; R calibration |
| `src/dynamics.py` | Drake prediction model for the estimator and the MPC, with a process-parallel wrapper |
| `src/probing.py` | The shared push loop, selectors, and episode setup |
| `src/mpc.py`, `src/episode.py` | MPC cost and ranking; placement episodes |
| `src/belief_view.py` | Live belief views in Meshcat and matplotlib |
| `src/trials.py`, `src/replay.py` | Recorded trials and their offline replay |
| `src/surrogate.py`, `src/surrogate_data.py`, `src/train_surrogate.py`, `src/path_family.py`, `src/dataset_file.py` | Surrogate model, data generation, training, tip-path deviations, dataset files |
| `src/env.py`, `src/train_ppo.py` | The surrogate-world PPO environment and training |
| `src/evaluation.py`, `src/evaluation_report.py` | Drake-world evaluation and its report |
| `scripts/diagnose_motion.py` | Motion checks for random or listed pushes: joint speeds, tracking, release, post-measurement motion; optional Meshcat replay |
| `scripts/motion_plans/` | Push lists for the motion checks (extreme pushes, stress pushes) |
| `scripts/model_bias.py` | Prediction model against the truth sim on recorded pushes: mismatch, timestep effect, best-fit friction |
| `scripts/push_informativeness.py` | How well pushes of each speed and distance separate neighbouring hypotheses |
| `scripts/evaluate_policy.py` | Trained policy against random probing on paired surrogate episodes: errors, push counts, and the speeds, distances and directions it chooses |
| `scripts/plot_belief.py` | Push-by-push belief figure |
| `scripts/cluster/` | Slurm batch scripts for the Engaging cluster (PPO training, Drake transfer evaluation) |
| `scripts/merge_evaluations.py` | Paired statistics over per-episode evaluation reports |

## Implementation notes

- **Simulation time.** Pushes run one after another in one continuous
  simulation; nothing resets between them. The simulation clock and the views
  stop while the filter updates or the MPC plans.
- **Run times.** One truth-simulation push takes about 20–35 s of wall time. A Drake
  belief update over all 405 hypotheses takes roughly 1–2 min, and an MPC
  decision about 35–45 s, on 12 workers (`--workers`).
- **Workspace.** The seed draws start and goal poses within 10 cm of
  (0.55, 0) m. The arm reaches the push height for x between about 0.40 and
  0.75 m, so set wider custom layouts along y.
- **Recalibration.** After changing the prediction model, the measurement (including
  `--settle-time`), the hypothesis grid, or the push space, rerun `calibrate` and paste its constant
  into `src/estimator.py`; re-record trials after changing the arm motion or
  the push space.
- **Likelihood.** `--nu` sets the Student-t likelihood's degrees of freedom
  (default 3, which keeps one badly mispredicted push from eliminating the
  right hypothesis); `--nu 0` uses a Gaussian.
- **Tests.** Most tests take seconds; surrogate-environment tests need
  `data/surrogate_ensemble.npz` and `data/path_family.npz` (about two minutes).
  They do not replace short Drake runs for physics changes.
- **CPU-only PyTorch on Linux.** `uv sync` installs the GPU build from the
  lockfile. For a CPU-only local environment:

  ```sh
  uv venv --python 3.14 .venv
  uv pip install --python .venv/bin/python --torch-backend cpu 'torch>=2.15'
  uv pip install --python .venv/bin/python 'drake>=1.58.0' 'gymnasium>=1.4.0' 'matplotlib>=3.11.2' 'sb3-contrib>=2.9.2' 'stable-baselines3>=2.9.2' 'pytest>=9.2.0' 'mypy>=2.4.1' 'ruff>=0.17.3'
  ```

  Then use `uv run --no-sync`, since a plain `uv run` re-syncs to the GPU build.
