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

Reports give each episode's parameter errors, reward, push count, and stop
reason.

### Data, calibration, and training

| Command | What it does |
| --- | --- |
| `calibrate` | Measures the estimator's prediction-error covariance R from truth-simulation pushes and prints the constant for `estimator.CALIBRATED_R` |
| `record-trials` | Records multi-push trials in the truth simulation to `data/replay_trials.npz` |
| `replay` | Scores the estimator on recorded trials and writes `data/replay_report.json` |
| `calibrate-path-family` | Fits how the arm's executed tip path deviates from the commanded one, from recorded trials |
| `gen-surrogate-data` | Generates surrogate training data from the Drake prediction model |
| `train-surrogate` | Trains the surrogate's MLP ensemble |
| `train-ppo` | Trains the probing policy in the surrogate world |

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
| `scripts/diagnose_motion.py` | Tip clearance, joint speed, and residual-motion check for consecutive pushes |
| `scripts/plot_belief.py` | Push-by-push belief figure |

## Implementation notes

- **Simulation time.** Pushes run one after another in one continuous
  simulation; nothing resets between them. The simulation clock and the views
  stop while the filter updates or the MPC plans.
- **Run times.** One truth-simulation push takes about 20 s of wall time. A Drake
  belief update over all 405 hypotheses takes about 50–115 s, and an MPC
  decision about 35–45 s, on 12 workers (`--workers`).
- **Workspace.** The seed draws start and goal poses within 10 cm of
  (0.55, 0) m. The arm reaches the push height for x between about 0.40 and
  0.75 m, so set wider custom layouts along y.
- **Recalibration.** After changing the prediction model, the measurement, the
  hypothesis grid, or the push space, rerun `calibrate` and paste its constant
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
