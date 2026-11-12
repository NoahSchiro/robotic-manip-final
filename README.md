# robotic-manip-final

Implementation details and timing contracts are in [the system description](docs/system.md).

## Running the simulation

```sh
uv run python main.py simulate
uv run python main.py simulate --headless
```

There is a number of options. Run `uv run main.py simulate --help` to see them.

`main.py` builds a KUKA iiwa14 welded to a table, with a T-shaped block
resting on the table in front of it and a stick pusher welded to the
arm's flange.

The subsystems live in `src/`
- `iiwa.py` (arm, pusher, IK, trajectories),
- `object.py` (the T-shaped block and table)
- `pdcontroller.py` (gains and controller)
- `params.py` (unknown parameters eta = (mu, rod x, rod y) and the hypothesis grid)
- `observation.py` (what a push produces: object start/end state, tip path, measurement math)
- `recording.py` (the truth sim, one continuous simulation: running pushes and recording them)
- `dynamics.py` (Drake prediction model: replays a recorded tip path for the estimator, or runs a push's nominal tip path for the MPC; plus a process-parallel wrapper)
- `estimator.py` (grid Bayes filter: MAP, mean, entropy)
- `calibration.py` (measuring the prediction-error covariance R)
- `mpc.py` (one-push MPC: pose cost, candidate pushes, ranking)
- `episode.py` (a placement episode: optional random probing, then MPC to the goal)
- `cli.py` (the command-line options `main.py`'s subcommands share)

`main.py` assembles the scene, sets the per-run block pose, and holds the
plant time step.

`uv run pytest` runs the tests. The mathematical and geometry checks take
seconds; surrogate-environment checks need `data/surrogate_ensemble.npz` and
`data/path_family.npz` and took about two minutes on the development laptop.
They do not replace short Drake motion checks for physics changes.

For a local CPU-only PyTorch environment on Linux:

```sh
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python --torch-backend cpu 'torch>=2.15'
uv pip install --python .venv/bin/python 'drake>=1.58.0' 'gymnasium>=1.4.0' 'matplotlib>=3.11.2' 'sb3-contrib>=2.9.2' 'stable-baselines3>=2.9.2' 'pytest>=9.2.0' 'mypy>=2.4.1' 'ruff>=0.17.3'
uv run --no-sync pytest -q
```

Use `uv run --no-sync` for later commands in this local environment. A regular
`uv run` syncs from the current lockfile, which selects the GPU PyTorch build.
This local choice does not change teammates' environments.

## Parameter estimation

Drake has no Windows wheels, so on Windows run everything under WSL.

```sh
# Random pushes with a Bayes update after each (true eta from --mu/--rod-x/--rod-y)
uv run python main.py simulate --headless --estimate --pushes 5
```

Without `--headless`, the belief is shown after every push.  In Meshcat,
yellow discs on the block's top face show where the rod might be (area ~
probability), beside the red marker at the true rod.  A matplotlib window
charts the belief over rod position and over friction against the truth,
the entropy and the MAP's probability, and the MAP's error in grid steps.
The windows freeze while the filter updates.  Press Enter in the terminal
to quit.

The pushes run one after another in one continuous simulation: nothing resets
between them, and the sim's clock stops while the filter updates.  The true
eta and start pose come from `probing.episode_conditions(--seed)`, the same
setup `evaluate` uses.  Each push is drawn uniformly from the probing library:
32 contact sites on the T's outline that clear the lowered stick (the T's inner
corners are excluded), times 2 speeds and 2 contact times, restricted to pushes
the arm can reach from the block's current pose.

The default grid is 9 values each of mu, rod x and rod y, keeping only rod
positions inside the T (405 hypotheses); change it via `params.GridSpec`.
The prediction-error covariance R is calibrated and stored in the code as
`estimator.CALIBRATED_R`, so no setup is needed.  After changing the
prediction model, the measurement, or the grid, rerun
`uv run python main.py calibrate` (~4 min) and paste the constant it
prints into `src/estimator.py`.
The likelihood is a Student-t (`--nu 3` by default), which keeps one badly
mispredicted push from eliminating the right hypothesis; `--nu 0` uses the
proposal's Gaussian instead.

### Belief visualization

```sh
# Drake: one headless episode; saves the belief before and after each push
uv run python -m scripts.plot_belief record --seed 123 --pushes 3 --out data/belief_seed123.npz
# numpy and matplotlib only: rod-position and friction marginals per step
uv run python -m scripts.plot_belief plot data/belief_seed123.npz --out docs/figures/belief_seed123.png
```

`record` takes `--workers`, and `--mu`, `--rod-x`, `--rod-y` to override the sampled truth.

## Offline replay (the estimator's reference)

The estimator's quality is tracked on a frozen dataset of recorded trials
(`data/replay_trials.npz`, committed).  Record it with
`uv run python main.py record-trials` (slow: runs the truth sim), score it
with `uv run python main.py replay` (minutes), which writes
`data/replay_report.json`.  Any change to the prediction model is judged by
re-running `replay` on the same dataset.  Trials use the same episode setup and
action set as `simulate --estimate` (see [the system description](docs/system.md)).
The committed dataset predates that action set, so re-record it before comparing
replay results across the change.

## Surrogate training data

`uv run python main.py gen-surrogate-data` generates the surrogate's dataset:
sampled (eta, start pose, action) tuples replayed through the Drake prediction
model with constructed nominal tip paths. Each independent sample is assigned
by seed to training, development, or held-out data. The current target is the
estimator's window-end pose and push-end velocity. Schema 3 also retains
modeled action-end pose and velocity labels for later MPC fitting; the current
trainer does not use those labels.

## Random Drake evaluation

```sh
uv run python main.py evaluate --random-only --push-budget 1 --episodes 1 --seed 70000 --out data/random_drake_report.json
```

This runs random probing and the Drake grid filter without PPO, ensemble, or
path-family files. The report gives signed per-episode parameter errors and a
range-normalized Euclidean MAP error, using each grid parameter's full span.
The default `evaluate` command still runs the paired policy comparison.

## Placement with MPC

```sh
# No-probing baseline: MPC from the first push, estimating eta as it goes
uv run python main.py simulate --headless --mpc
# Random probing: 3 random pushes first, then MPC (20 pushes in all)
uv run python main.py simulate --headless --mpc --probe-pushes 3
# Check the controller alone, planning with the true eta
uv run python main.py simulate --headless --mpc --plan-with-true-eta
# Set the start and goal poses (x m, y m, yaw deg) instead of the seed's
uv run python main.py simulate --headless --mpc --start 0.55 -0.15 0 --goal 0.55 0.15 130
```

The seed draws start and goal (from independent streams) within 10 cm of (0.55, 0), so at most
20 cm apart.  The arm reaches the push height for x between about 0.40
and 0.75 m, so set wider layouts along y.

Before each push, the MPC predicts where every candidate push would
leave the block under the MAP eta and picks the cheapest push the arm can
reach.  The cost is the proposal's pose cost plus a small time cost
(`--cost-length`, `--yaw-weight`, `--time-weight`).  The candidates are the full push library minus the inner-corner sites, where
the lowered stick would land on the block (`--site-clearance`); probe pushes
draw from the smaller probing library through the shared probing loop.  As in
estimation, the pushes run one after another in one continuous simulation,
whose clock stops while the MPC plans.  An episode succeeds within 1 cm
and 10 degrees of the goal (`--position-tolerance`, `--yaw-tolerance`).  It
stops after 20 pushes (`--max-pushes`, probes included) or when no push is
predicted to help.  Each MPC
decision takes about 35 to 50 s on 12 cores, plus the estimator update.

Without `--headless`, the belief views from parameter estimation update
after every push, probes included; Meshcat also shows a yellow ghost where
the MPC predicts its push leaves the block.  `--plan-with-true-eta` has no
belief, so it shows neither belief view.

