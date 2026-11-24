# System reference

Definitions used throughout the code. Commands and setup are in the [README](../README.md).

## Units and frames

Lengths in meters, time in seconds, angles in radians. World frame: the arm base at the origin, z up. Block body frame: the footprint's bottom center. A planar object state is `(x, y, yaw, vx, vy, wz)` in the world frame.

## World and unknowns

A KUKA iiwa14 with a vertical stick (radius 0.01 m) pushes a T-shaped block (a 5 × 14 cm crossbar and a 9 × 5 cm stem, 3 cm tall) on a table. The unknowns are `eta = (mu, bx, by)`: block–table friction and the body-frame position of a welded rod of known mass (0.2 kg) that shifts the center of mass. The truth simulation (`recording.TruthSim`, time step 1e-4 s) is one continuous Drake simulation: the block and arm keep their state between pushes, and the clock stops while the controller computes.

## Push space (`src/world/push_space.py`)

A push is a contact point on one of the T's 8 sides with the side's inward normal, a commanded tip speed, and a contact time. The push direction is always the inward normal.

| Parameter | Range |
| --- | --- |
| Contact position | Along each side, inset 1.2 cm from inner corners (the lowered stick clears the other part of the T) and 0.5 cm from outer corners; 45 of 56 cm of outline usable |
| Speed | 0.03–0.15 m/s |
| Contact time | 0.1–1.0 s |
| Push distance (speed × contact time) | at most 0.10 m |

Random pushes are uniform: a side in proportion to its usable length, then position uniformly, and speed and time uniformly over their ranges, redrawn together until their product is within 0.10 m. A push is reachable when the approach point and the push end lie 0.42–0.78 m from the arm base. The MPC's candidates are a grid: contacts about 1.2 cm apart, 3 speeds (0.03, 0.09, 0.15 m/s), and for each speed 3 push distances from speed × 0.1 s up to the smaller of speed × 1.0 s and 0.10 m; each distance sets its contact time (396 pushes).

## Arm maneuver and timing

From its current posture the arm moves the tip over the standoff point at block top + 4 cm, lowers to the standoff (0.015 m before the contact), dwells there at rest for 0.3 s so the lowering's vertical overshoot dies out, pushes at the commanded speed through the contact for the contact time, releases 1 cm back along the normal at 0.05 m/s, and lifts straight up to transfer height, then holds (0.1 s). The next push starts from the measured joint positions.

The transfer over the standoff, the lowering, and the lift are free-space legs. Each lasts at least its nominal duration (0.35 s), longer when its slowest joint would exceed half its velocity limit. The dwell, push, and release keep their nominal durations. The planner's time cost (MPC) uses the nominal durations, since the IK solutions that set the stretched legs are unknown before planning.

| Event | Meaning |
| --- | --- |
| Transfer end | End of the transfer over the standoff; lowering begins |
| Start | Tip at rest at the standoff (end of the dwell); start of the recorded tip path and of predictions |
| Push end | End of the commanded push segment |
| Measurement end | `settle_time` (0.3 s) after push end |
| Action end | Release, lift, and hold done; the next decision's state |

The surrogate's nominal tip paths sample this same maneuver from the standoff (push, release, lift, hold) at straight constant-speed legs with nominal durations, so the window's paths match the commanded geometry used by the Drake predictions.

## Measurement and estimator

The measurement is `(x, y, yaw)` at measurement end with `(vx, vy, wz)` at push end. The filter is a grid Bayes filter over 405 hypotheses: 9 values each of `mu`, `bx`, `by`, keeping rod positions inside the T. Each hypothesis's measurement is predicted by a Drake model that drives a lone stick tip along the measured tip path (time step 2e-3 s). The likelihood is a Student-t with `nu = 3` on the residual, scaled by the calibrated covariance `estimator.CALIBRATED_R`. Each hypothesis stands for its cell, the box within half a grid step of it. The belief is held on `n^3` points evenly spread inside every cell (`estimator.SUBCELLS = n`); a point's residual is the multilinear interpolation of the eight surrounding hypotheses' residuals (yaw wrapped before interpolating), held constant along any axis where one is missing (outline edge, end of a parameter range). A cell's probability is the sum over its points, so everything downstream sees 405 cell probabilities.

The MAP estimate is the most probable hypothesis. Its range-normalized error is `sqrt(sum_i ((eta_MAP_i − eta_true_i) / span_i)^2)`, with `span_i` the grid's range for parameter i.

## Episodes

`experiment.episode_conditions(seed, task)` gives the true `eta` and the start pose; random actions use an independent stream seeded by `SeedSequence([seed, 1])` and the placement goal one seeded by `SeedSequence([seed, 2])`. Every runner executes pushes through `probing.run_probing`, which selects, plans, executes, updates the belief, and stops on the push budget, the selector's stop reason, 40 consecutive arm-plan failures, or the block leaving the task's bounds.

Episodes use one of two tasks (`experiment.TASKS`). In `near`, the default and the task the PPO policy was trained on, start and goal are each drawn within 0.1 m of (0.55, 0) m, and the block must stay within 0.25 m of that point. In `far`, start and goal are drawn on the band 0.50–0.65 m from the arm base within ±55° of the +x axis, where the arm reaches every lattice push from any block yaw, and the goal is redrawn until it lies 0.30–0.45 m from the start; the block must stay 0.40–0.75 m from the base within ±75°. Start and goal yaw are uniform in both. `eta` is drawn before the start pose, so a seed has the same true `eta` in both tasks.

## MPC placement

A placement episode runs optional random probes, then MPC pushes, on one state and one push budget. Each decision predicts every candidate's outcome once the stick has risen clear, under the current MAP `eta`, and scores it

`V = |r − r_goal|² / l² + λ_yaw · wrap(yaw − yaw_goal)² + w_t · T_maneuver`

with `l = 0.07 m`, `λ_yaw = 1 rad⁻²`, `w_t = 0.005 s⁻¹`. Only candidates inside the arm's reach band (0.42–0.78 m from the base for the approach and push end) are predicted, and a candidate is ranked only if, from its predicted end pose, at least 80% of the candidates remain reachable (or, when the block is already below that, no fewer than now); otherwise the one-push lookahead drives the block toward the base, where it can no longer be pushed. It takes the cheapest ranked candidate whose arm plan succeeds and that improves on the current pose cost. An episode succeeds within 1 cm and 10° of the goal; otherwise it stops as "stuck" (no improving candidate), "unreachable" (none reachable), "push budget", "ik_exhausted" (40 consecutive arm-plan failures), or "out_of_bounds".

## Surrogate

The surrogate is an MLP ensemble that predicts a push's measurement, for fast belief updates and the PPO training world. Its 49 inputs are `eta`, the world start pose and velocity, the body-frame contact point and normal, speed and contact time, and the tip path's xy at 16 times. Its outputs are the measurement-end pose change and push-end velocity in the start pose's frame. Training data come from the Drake prediction model with sampled tip-path deviations. Samples are split by sample seed (modulo 10) into training (80%), development (10%), and held-out (10%) rows; each sample also stores its action-end pose and velocity for a future planning model.
