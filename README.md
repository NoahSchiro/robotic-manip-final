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
- `dynamics.py` (Drake prediction model replaying a recorded tip path, plus a process-parallel wrapper)
- `estimator.py` (grid Bayes filter: MAP, mean, entropy)
- `calibration.py` (measuring the prediction-error covariance R)
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

