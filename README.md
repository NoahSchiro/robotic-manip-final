# robotic-manip-final

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

`uv run pytest` runs the tests.  They cover the math and geometry without
running a simulation, so they take seconds; they can't catch physics problems.

## Parameter estimation

Drake has no Windows wheels, so on Windows run everything under WSL.

```sh
# Random pushes with a Bayes update after each (true eta from --mu/--rod-x/--rod-y)
uv run python main.py simulate --headless --estimate --pushes 5
```

The pushes run one after another in one continuous simulation: nothing resets
between them, and the sim's clock stops while the filter updates.  They draw
from the push library minus the sites where the lowered stick would land on
the block (`--site-clearance`), at the T's inner corners.

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

The seed draws start and goal within 10 cm of (0.55, 0), so at most
20 cm apart.  The arm reaches the push height for x between about 0.40
and 0.75 m, so set wider layouts along y.

Before each push, the MPC predicts where every candidate push would
leave the block under the MAP eta and picks the cheapest push the arm can
reach.  The cost is the proposal's pose cost plus a small time cost
(`--cost-length`, `--yaw-weight`, `--time-weight`).  The candidates, like
the probe pushes, are the push library minus the inner-corner sites, where
the lowered stick would land on the block (`--site-clearance`).  As in
estimation, the pushes run one after another in one continuous simulation,
whose clock stops while the MPC plans.  An episode succeeds within 1 cm
and 10 degrees of the goal (`--position-tolerance`, `--yaw-tolerance`).  It
stops after 20 pushes (`--max-pushes`, probes included) or when no push is
predicted to help.  Each MPC
decision takes about 35 to 50 s on 12 cores, plus the estimator update.
