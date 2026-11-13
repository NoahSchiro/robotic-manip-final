# System reference

Definitions used throughout the code. Commands and setup are in the [README](../README.md).

## Units and frames

Lengths in meters, time in seconds, angles in radians. World frame: the arm base at the origin, z up. Block body frame: the footprint's bottom center. A planar object state is `(x, y, yaw, vx, vy, wz)` in the world frame.

## World and unknowns

A KUKA iiwa14 with a vertical stick (radius 0.01 m) pushes a T-shaped block (a 5 × 14 cm crossbar and a 9 × 5 cm stem, 3 cm tall) on a table. The unknowns are `eta = (mu, bx, by)`: block–table friction and the body-frame position of a welded rod of known mass (0.2 kg) that shifts the center of mass. The truth simulation (`recording.TruthSim`, time step 1e-4 s) is one continuous Drake simulation: the block and arm keep their state between pushes, and the clock stops while the controller computes.

## Push space (`src/push_space.py`)

A push is a contact point on one of the T's 8 sides with the side's inward normal, a commanded tip speed, and a contact time. The push direction is always the inward normal.

| Parameter | Range |
| --- | --- |
| Contact position | Along each side, inset 1.2 cm from inner corners (the lowered stick clears the other part of the T) and 0.5 cm from outer corners; 45 of 56 cm of outline usable |
| Speed | 0.03–0.12 m/s |
| Contact time | 0.1–0.5 s |

Random pushes are uniform: a side in proportion to its usable length, then position, speed, and time uniformly. A push is reachable when the approach point and the push end lie 0.39–0.78 m from the arm base. The MPC's candidates are a grid: contacts about 1.2 cm apart, 3 speeds, 3 contact times (396 pushes).

## Arm maneuver and timing

From its current posture the arm moves the tip over the standoff point at block top + 4 cm, lowers to the standoff (0.015 m before the contact), pushes at the commanded speed through the contact for the contact time, backs off 0.015 m at the same speed, lifts back to transfer height (0.35 s), and holds (0.1 s). The next push starts from the measured joint positions.

| Event | Meaning |
| --- | --- |
| Start | Tip at the standoff; start of the recorded tip path and of predictions |
| Push end | End of the commanded push segment |
| Measurement end | `settle_time` (0.1 s) after push end |
| Action end | Back-off, lift, and hold done; the next decision's state |

## Measurement and estimator

The measurement is `(x, y, yaw)` at measurement end with `(vx, vy, wz)` at push end. The filter is a grid Bayes filter over 405 hypotheses: 9 values each of `mu`, `bx`, `by`, keeping rod positions inside the T. Each hypothesis's measurement is predicted by a Drake model that drives a lone stick tip along the measured tip path (time step 2e-3 s). The likelihood is a Student-t with `nu = 3` on the residual, scaled by the calibrated covariance `estimator.CALIBRATED_R`.

The MAP estimate is the most probable hypothesis. Its range-normalized error is `sqrt(sum_i ((eta_MAP_i − eta_true_i) / span_i)^2)`, with `span_i` the grid's range for parameter i.

## Episodes

`probing.episode_conditions(seed)` gives the true `eta` and the start pose (within 0.1 m of (0.55, 0) m, any yaw); random actions use an independent stream seeded by `SeedSequence([seed, 1])` and the placement goal one seeded by `SeedSequence([seed, 2])`. Every runner executes pushes through `probing.run_probing`, which selects, plans, executes, updates the belief, and stops on the push budget, the selector's stop reason, 40 consecutive arm-plan failures, or the block leaving 0.25 m of (0.55, 0) m.

## MPC placement

A placement episode runs optional random probes, then MPC pushes, on one state and one push budget. Each decision predicts every candidate's outcome once the stick has risen clear, under the current MAP `eta`, and scores it

`V = |r − r_goal|² / l² + λ_yaw · wrap(yaw − yaw_goal)² + w_t · T_maneuver`

with `l = 0.07 m`, `λ_yaw = 1 rad⁻²`, `w_t = 0.005 s⁻¹`. It takes the cheapest candidate the arm can reach that improves on the current pose cost. An episode succeeds within 1 cm and 10° of the goal; otherwise it stops as "stuck" (no improving candidate), "unreachable" (none reachable), or "push budget".

## Surrogate

The surrogate is an MLP ensemble that predicts a push's measurement, for fast belief updates and the PPO training world. Its 49 inputs are `eta`, the world start pose and velocity, the body-frame contact point and normal, speed and contact time, and the tip path's xy at 16 times. Its outputs are the measurement-end pose change and push-end velocity in the start pose's frame. Training data come from the Drake prediction model with sampled tip-path deviations. Samples are split by sample seed (modulo 10) into training (80%), development (10%), and held-out (10%) rows; each sample also stores its action-end pose and velocity for a future planning model.

The surrogate data generator and the PPO environment still sample the earlier discrete library (32 sites, speeds 0.05 and 0.1 m/s, contact times 0.2 and 0.4 s).
