# robotic-manip-final

## Running the simulation

```sh
uv run python main.py
HEADLESS=1 uv run python main.py
```

`main.py` builds a KUKA iiwa14 welded to a table, with a T-shaped block
resting on the table in front of it and a stick pusher welded to the
arm's flange.

The subsystems live in `src/`
- `iiwa.py` (arm, pusher, IK, trajectories),
- `object.py` (the T-shaped block and table)
- `pdcontroller.py` (gains and controller)

`main.py` assembles the scene, sets the per-run block pose, and holds the
plant time step.
