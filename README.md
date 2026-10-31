# robotic-manip-final

## Running the simulation

```sh
uv run python main.py
HEADLESS=1 uv run python main.py
```

There is a number of options. Run `uv run main.py --help` to see them.

`main.py` builds a KUKA iiwa14 welded to a table, with a T-shaped block
resting on the table in front of it and a stick pusher welded to the
arm's flange.

The subsystems live in `src/`
- `iiwa.py` (arm, pusher, IK, trajectories),
- `object.py` (the T-shaped block and table)
- `pdcontroller.py` (gains and controller)
- `params.py` (unknown parameters eta = (mu, rod x, rod y) and the hypothesis grid)
- `observation.py` (recording a push from the truth sim: object start/end state, tip path)
- `dynamics.py` (Drake prediction model replaying a recorded tip path, plus a process-parallel wrapper)
- `estimator.py` (grid Bayes filter: MAP, mean, entropy)

`main.py` assembles the scene, sets the per-run block pose, and holds the
plant time step.

## Parameter estimation

Drake has no Windows wheels, so on Windows run everything under WSL.

```sh
# Calibrate the prediction-error covariance R (writes assets/R.npy, ~2 min)
uv run python -m scripts.calibrate_R --samples 48
# Random pushes with a Bayes update after each (true eta from --mu/--rod-x/--rod-y)
uv run python main.py --headless --estimate --pushes 5
```

The default grid is 9 values each of mu, rod x and rod y, keeping only rod
positions inside the T (405 hypotheses); change it via `params.GridSpec`.
Without `assets/R.npy` the estimator falls back to an uncalibrated diagonal R.
