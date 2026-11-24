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
nodes). Download the trained models into `data/models/` (see Data layout). Batch scripts are in `scripts/cluster/`; create
`logs/` once before submitting.

### Data layout

`src/paths.py` defines where generated files go:

| Folder | In git | Contents |
| --- | --- | --- |
| `data/reference/` | yes | inputs that are costly to regenerate: recorded truth-sim trials, the fitted tip-path family |
| `data/results/` | yes | results the report cites: replay reports, transfer and placement runs, PPO training curve and evaluation, surrogate logs |
| `data/models/` | no (GitHub release) | trained surrogate ensemble and PPO policy |
| `data/work/` | no | regenerable intermediates: surrogate shards, PPO checkpoints, caches |

Download the trained models from the
[`models-2026-11-22` release](https://github.com/NoahSchiro/robotic-manip-final/releases/tag/models-2026-11-22)
into `data/models/`:

```sh
mkdir -p data/models
for f in surrogate_ensemble.npz ppo_policy.zip; do curl -L -o data/models/$f https://github.com/NoahSchiro/robotic-manip-final/releases/download/models-2026-11-22/$f; done
sha256sum data/models/*
```

The checksums should match the release notes.

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
uv run python -m scripts.plot_belief record --seed 123 --pushes 3 --out data/work/belief_seed123.npz
# Plot it (numpy and matplotlib only)
uv run python -m scripts.plot_belief plot data/work/belief_seed123.npz --out docs/figures/belief_seed123.png
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
uv run python main.py evaluate --random-only --push-budget 3 --episodes 4 --seed 70000 --out data/results/random_report.json
# Trained policy vs random probing on the same episodes
uv run python main.py evaluate
uv run python main.py evaluation-report
# Narrated episodes of the trained policy
uv run python main.py demo
```

To score a trained policy in the surrogate world and see which pushes it chooses:

```sh
uv run python -m scripts.evaluate_policy --out data/results/ppo_evaluation
# Redraw behaviour.png and push_choices.png from the saved episodes
uv run python -m scripts.evaluate_policy --out data/results/ppo_evaluation --from-json
```

Each `evaluate` episode runs, on identical conditions in the Drake truth sim,
the trained policy until it stops, then two baselines with the same number of
pushes: random pushes from the policy's 396-push lattice, and random pushes
from its long, fast subset (at least 6.8 cm and 11 cm/s). Reports give each
episode's parameter errors, reward, push count, and stop reason. On the
cluster, `scripts/cluster/evaluate_transfer.sbatch` runs one episode per array
task (`FIXED_PUSHES=3` makes every strategy push exactly three times, the
policy choosing which pushes; `--fixed-pushes` does the same for
`evaluate`); combine them into paired statistics (mean difference, standard
error, paired t-test and Wilcoxon signed-rank p-values) with:

```sh
uv run python -m scripts.merge_evaluations data/results/transfer/episode_*.json --out data/results/transfer/summary.json
```

### Probing before placement

```sh
# One placement episode: 3 learned probes, then MPC (Drake truth and filter)
uv run python main.py placement-eval --seed 95000 --condition learned:3 --out data/results/placement/95000_learned_3.json
```

Conditions are `true_eta` (MPC with the true parameters), `none` (no
probing), or `<learned|random|long_fast>:<probes>`. Probes and placement
pushes share `--max-pushes`. The record holds the distance and angle to the
goal after every push, so success at any looser tolerance or smaller budget
can be scored afterwards; run at the tightest tolerance of interest.
`--task` picks where start and goal are drawn: `near` (default; both within
10 cm of (0.55, 0) m, the task the PPO policy was trained on) or `far` (both
0.50–0.65 m from the arm base within ±55°, where the arm reaches every lattice
push, and 0.30–0.45 m apart). A seed has the same true parameters in both.
`scripts/cluster/placement.sbatch` runs one (seed, condition) per array task;
its header lists the benchmark, headroom and full-comparison submissions.
Score the records (success by push budget at 0.5 cm/5°, 1 cm/10° and
2 cm/20°, and the paired difference between `none` and `true_eta`) with:

```sh
uv run python -m scripts.summarize_placement data/results/headroom/near data/results/headroom/far --out data/results/headroom
```

### Check the arm motion

```sh
# Random pushes: transfer clearance, joint speeds, tracking, release
uv run python -m scripts.diagnose_motion --seed 123 --pushes 2 --out data/results/motion/motion.json
# The validation push sets, with a Meshcat replay to open in a browser
uv run python -m scripts.diagnose_motion --plan scripts/motion_plans/extremes.json --friction 0.1 --out data/results/motion/extremes_mu01.json --html data/work/extremes_mu01.html
uv run python -m scripts.diagnose_motion --plan scripts/motion_plans/stress.json --out data/results/motion/stress.json
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
uv run python -m scripts.push_informativeness --out data/results/push_informativeness.json
# Plot saved results without re-simulating
uv run python -m scripts.push_informativeness --from-json data/results/push_informativeness.json --plot docs/figures
```

For every contact on the MPC grid, at matched push distances (1.5, 4.5, and
9 cm) and several speeds, this predicts the estimator's measurement under a
hypothesis and under its neighbour one grid step away in friction, rod x, or
rod y. It reports how far apart the two predictions are relative to the
estimator's noise (`CALIBRATED_R`): about 1 or more means one push noticeably
shifts the belief between them. The plots show the median over contacts by
speed and distance, and one push type's separation at every contact point.
Planning-only, about 3 minutes on 12 workers.

### Where the block can be pushed

```sh
uv run python -m scripts.plot_reach --out docs/figures/reach.png
```

For each block position on the table, the share of the 396 lattice pushes
whose approach and end points lie in the arm's reach band (0.42–0.78 m from
the base), averaged over block yaw, with the workspace and start/goal disks.
About 30 s.

### Check the prediction model against the truth sim

```sh
uv run python -m scripts.model_bias --out data/results/model_bias.npz
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
| `record-trials` | Records multi-push trials in the truth simulation to `data/reference/replay_trials.npz` |
| `replay` | Scores the estimator on recorded trials and writes `data/results/replay_report.json`; `--save-predictions` caches the grid predictions so `--load-predictions` can rescore them (for example with another `--subcells`) in seconds |
| `calibrate-path-family` | Fits how the arm's executed tip path deviates from the commanded one, from recorded trials |
| `gen-surrogate-data` | Generates one shard of surrogate training data from the Drake prediction model, with pushes drawn uniformly from the push space (about 10 samples/s on 12 workers); give each shard its own `--seed` and `--out` |
| `train-surrogate` | Trains the surrogate's MLP ensemble on one or more shards (`--dataset a.npz b.npz ...`); `--device cuda` trains on the GPU |
| `train-ppo` | Trains the probing policy in the surrogate world, choosing among the MPC's 396 lattice pushes or stop; `--resume` continues from the latest checkpoint in `--out` |

## Code layout

`main.py` holds the command-line interface and assembles each run.

| Module | Contents |
| --- | --- |
| `src/world/` | The block and table (`object`, `geometry`); arm, stick, IK and trajectories (`iiwa`, `pdcontroller`); Drake scene and Meshcat drawing (`scene`); a push and the push space every module samples from (`action`, `push_space`); the continuous truth simulation and what a push records (`recording`, `observation`) |
| `src/estimation/` | Unknown parameters and hypothesis grid (`params`); Drake prediction model (`dynamics`); grid Bayes filter (`estimator`); R calibration (`calibration`); recorded trials and their replay (`trials`, `replay`); live belief views (`belief_view`) |
| `src/surrogate/` | Surrogate network and predictor (`model`), data generation (`data`), training (`train`), tip-path deviation family (`path_family`) |
| `src/policy/` | The surrogate-world PPO environment (`env`), training (`train`), and the trained policy as a Drake-world push selector (`deploy`) |
| `src/control/` | The shared Drake push loop and selectors (`probing`); MPC (`mpc`); placement episodes (`episode`) |
| `src/evaluation/` | Drake transfer of the prober (`transfer`), placement under probing conditions (`placement`), training report (`report`) |
| `src/constants.py` | Values shared across modules, defined once: rod mass, simulation timing, workspace, push lattice, MPC cost |
| `src/experiment.py` | The setup built from them: simulation configs, the push lattice, the reference block, and how a seed becomes an episode |
| `src/paths.py`, `src/cli.py`, `src/dataset_file.py` | Data layout; shared command-line options; dataset files |
| `scripts/diagnose_motion.py` | Motion checks for random or listed pushes: joint speeds, tracking, release, post-measurement motion; optional Meshcat replay |
| `scripts/motion_plans/` | Push lists for the motion checks (extreme pushes, stress pushes) |
| `scripts/model_bias.py` | Prediction model against the truth sim on recorded pushes: mismatch, timestep effect, best-fit friction |
| `scripts/push_informativeness.py` | How well pushes of each speed, distance and contact point separate neighbouring hypotheses |
| `scripts/plot_reach.py` | Where the block can sit: share of lattice pushes the arm reaches at each block position |
| `scripts/evaluate_policy.py` | Trained policy against random probing on paired surrogate episodes: errors, push counts, and the speeds, distances and directions it chooses |
| `scripts/plot_belief.py` | Push-by-push belief figure |
| `scripts/cluster/` | Slurm batch scripts for the Engaging cluster (PPO training, Drake transfer evaluation) |
| `scripts/summarize_placement.py` | Placement records scored by tolerance and push budget, and the paired gap between conditions |
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
  into `src/estimation/estimator.py`; re-record trials after changing the arm motion or
  the push space.
- **Likelihood.** `--nu` sets the Student-t likelihood's degrees of freedom
  (default 3, which keeps one badly mispredicted push from eliminating the
  right hypothesis); `--nu 0` uses a Gaussian.
- **Tests.** Most tests take seconds; surrogate-environment tests need
  `data/models/surrogate_ensemble.npz` and `data/reference/path_family.npz` (about two minutes).
  They do not replace short Drake runs for physics changes.
- **CPU-only PyTorch on Linux.** `uv sync` installs the GPU build from the
  lockfile. For a CPU-only local environment:

  ```sh
  uv venv --python 3.14 .venv
  uv pip install --python .venv/bin/python --torch-backend cpu 'torch>=2.15'
  uv pip install --python .venv/bin/python 'drake>=1.58.0' 'gymnasium>=1.4.0' 'matplotlib>=3.11.2' 'sb3-contrib>=2.9.2' 'stable-baselines3>=2.9.2' 'pytest>=9.2.0' 'mypy>=2.4.1' 'ruff>=0.17.3' 'scipy>=1.18.1'
  ```

  Then use `uv run --no-sync`, since a plain `uv run` re-syncs to the GPU build.
