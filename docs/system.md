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

The measurement is `(x, y, yaw)` at measurement end with `(vx, vy, wz)` at push end. The pose includes the slide after release, which depends on friction. The velocity is read at push end because by measurement end the arm is reversing, and its tracking lag makes the stick tap the block, which would corrupt a velocity read.

The prediction model (`dynamics.DrakePushModel`, time step 2e-3 s) is a small Drake plant: the table, the block, and a lone stick tip on x/y/z prismatic joints that PD actuators (kp = 1e4, kd = 200) drive along a tip path, with no arm. The estimator replays each push's measured tip path under every hypothesis; the MPC drives candidate pushes along their nominal paths (straight, at the commanded speed), so its predictions omit the arm's tracking lag.

The filter is a grid Bayes filter over 405 hypotheses: 9 values each of `mu`, `bx`, `by`, keeping rod positions inside the T. After push t, with residual `e_i` = measured minus predicted measurement under hypothesis i (yaw wrapped) and `q_i = e_iᵀ R⁻¹ e_i`,

`p_(t+1, i) ∝ p_(t, i) · (1 + q_i / ν)^(−(ν + 6) / 2)`,

a Student-t likelihood with `ν = 3` and scale `R = estimator.CALIBRATED_R`, kept in log space. Under a Gaussian likelihood `exp(−q / 2)`, one badly mispredicted push costs the correct hypothesis hundreds of nats and a wrong one takes over with near certainty; the t's penalty grows only logarithmically in `q`, so no single push can eliminate a hypothesis.

Each hypothesis stands for its cell, the box within half a grid step of it. Scoring only the grid points let a wrong neighbour beat the nearest hypothesis when the truth lay between grid points (a rod offset traded against friction), and the filter could then rule out the right cell with near certainty. The belief is held on `n^3` points evenly spread inside every cell (`estimator.SUBCELLS = n`); a point's residual is the multilinear interpolation of the eight surrounding hypotheses' residuals (yaw wrapped before interpolating), held constant along any axis where one is missing (outline edge, end of a parameter range). A cell's probability is the sum over its points, so everything downstream sees 405 cell probabilities.

The MAP estimate is the most probable hypothesis. Its range-normalized error is `sqrt(sum_i ((eta_MAP_i − eta_true_i) / span_i)^2)`, with `span_i` the grid's range for parameter i (0.8 for `mu`, 0.132 m for each rod coordinate).

R is calibrated (`main.py calibrate`) from 192 truth pushes (seed 20000), each an episode's first push with off-grid true parameters and a random push. Each push is predicted at the belief point nearest the true parameters, so R covers model mismatch, sub-cell interpolation, and the remaining discretization. R is the residuals' second moment about zero, since the likelihood assumes zero-mean errors, plus a diagonal floor (standard deviations 1e-4 m, 1e-4 m, 1e-3 rad, 2e-4 m/s, 2e-4 m/s, 2e-3 rad/s). `CALIBRATED_R_SURROGATE` is calibrated the same way with surrogate predictions. Both hold only for the settings they were calibrated under (truth step 1e-4 s, model step 2e-3 s, 0.3 s settle time, 2³ sub-cells, this push space and maneuver), which is why the command-line defaults come from `src/constants.py` and no command sets its own.

## Episodes

`experiment.episode_conditions(seed, task)` gives the true `eta` and the start pose; random actions use an independent stream seeded by `SeedSequence([seed, 1])` and the placement goal one seeded by `SeedSequence([seed, 2])`. Every runner executes pushes through `probing.run_probing`, which selects, plans, executes, updates the belief, and stops on the push budget, the selector's stop reason, 40 consecutive arm-plan failures, or the block leaving the task's bounds.

Episodes use one of two tasks (`experiment.TASKS`). In `near`, the default and the task the PPO policy was trained on, start and goal are each drawn within 0.1 m of (0.55, 0) m, and the block must stay within 0.25 m of that point. In `far`, start and goal are drawn on the band 0.50–0.65 m from the arm base within ±55° of the +x axis, where the arm reaches every lattice push from any block yaw, and the goal is redrawn until it lies 0.30–0.45 m from the start, so the straight line between them stays at least 0.45 m from the base, where most pushes remain reachable; the block must stay 0.40–0.75 m from the base within ±75°. Start and goal yaw are uniform in both. `eta` is drawn before the start pose, so a seed has the same true `eta` in both tasks.

## MPC placement

A placement episode runs optional random probes, then MPC pushes, on one state and one push budget. Each decision predicts every candidate's outcome once the stick has risen clear, under the current MAP `eta`, and scores it

`V = |r − r_goal|² / l² + λ_yaw · wrap(yaw − yaw_goal)² + w_t · T_maneuver`

with `l = 0.07 m` (half the T's 14 cm extent, so 7 cm of position error costs as much as 1 rad of yaw error), `λ_yaw = 1 rad⁻²`, `w_t = 0.005 s⁻¹`. Only candidates inside the arm's reach band (0.42–0.78 m from the base for the approach and push end) are predicted, and a candidate is ranked only if, from its predicted end pose, at least 80% of the candidates remain reachable (or, when the block is already below that, no fewer than now); otherwise the one-push lookahead drives the block toward the base, where it can no longer be pushed. It takes the cheapest ranked candidate whose arm plan succeeds and that improves on the current pose cost. An episode succeeds within a position and yaw tolerance set per run (`simulate`: 1 cm and 10°; `placement-eval`: 0.5 cm and 5°); otherwise it stops as "stuck" (no improving candidate), "unreachable" (none reachable), "push budget", "ik_exhausted" (40 consecutive arm-plan failures), or "out_of_bounds".

## Surrogate

The surrogate is an MLP ensemble that predicts a push's measurement, for fast belief updates and the PPO training world. Its 49 inputs are `eta`, the world start pose and velocity, the body-frame contact point and normal, speed and contact time, and the tip path's xy at 16 times. Its outputs are the measurement-end pose change and push-end velocity in the start pose's frame. Training data (140k samples) come from the Drake prediction model, driving each push along its nominal tip path plus a sampled deviation. The arm's PD tracking makes executed tip paths deviate from the commanded ones by a few millimeters, depending on force and gravity, so a surrogate trained on commanded paths alone would meet different paths at deployment. The deviation family (`surrogate.path_family`) is calibrated on the recorded trials: each push's measured xy path minus its nominal path, sampled at 16 window fractions, stacked over P pushes and decomposed by SVD into a mean plus 8 components with amplitudes `σ_k = s_k / sqrt(P)`; each training sample draws one deviation from it. Samples are split by sample seed (modulo 10) into training (80%), development (10%), and held-out (10%) rows; each sample also stores its action-end pose and velocity, unused so far.

The deployed ensemble has 5 members, each a ReLU MLP with 4 hidden layers of 768 units, trained with AdamW (learning rate 1e-3, cosine decay, batch 1024) on standardized inputs and outputs, with early stopping on the development split. The loss weights each output by `(σ_out / sqrt(diag R))²`, so errors count in units of the Drake model's calibrated noise. The held-out split is never used for training or model selection.

## PPO probing environment (`src/policy/env.py`)

The learned prober trains entirely in the surrogate world. Each episode draws a true `eta` and start pose as the near task does, and the filter starts uniform. A push's outcome is predicted by a randomly chosen ensemble member plus Gaussian measurement noise whose covariance is `CALIBRATED_R_SURROGATE` reduced by the members' own spread (correlations kept), so the belief sharpens about as slowly as a deployed surrogate filter's. The block continues from the noisily measured pose.

The observation (431 values) is the block pose (3), the log cell probabilities (405), entropy and maximum probability (2), the MAP and mean `eta` (3 + 3), the last measurement (6), the last action (8), and the push count (1). The actions are the 396 lattice pushes, masked when unreachable, plus `stop`, never masked. Rewards: each push earns `0.1 · ΔH − 0.05`, with `H = −Σ p ln p` over the 405 cells (nats); stopping, or the forced stop after 12 pushes, earns minus the range-normalized MAP error; leaving the workspace adds −1 to that. The policy was trained with MaskablePPO for 3M steps on 90 parallel environments (`n_steps` 512, batch 2048, learning rate 3e-4, γ 0.99, GAE λ 0.95, clip 0.2, entropy coefficient 0.03). At deployment (`policy_decision`) it stops when the probability of `stop` is at least that of all pushes together, and otherwise takes the most probable push.

## Evaluation protocol

All comparisons are paired: every condition runs on the same seeds, so the same true `eta`, start pose, and goal.

- **Estimator replay** (`main.py replay`): 24 recorded truth trials of 5 random pushes each (seed 10000) are rescored offline. A final MAP is *within one grid step* when every parameter's error is at most its grid step, and *confidently wrong* when the final maximum probability is at least 0.9 but the MAP is not within one step.
- **Prober transfer** (`main.py evaluate`): in the truth simulation with the Drake filter, 40 seeds (90000–90039). The learned policy is compared with random pushes from the lattice and with random long, fast pushes (at least 6.8 cm and 11 cm/s), each making the same number of pushes as the policy in that episode, or exactly 3 pushes each. The score is the range-normalized MAP error; comparisons report the mean paired difference ± its standard error, a paired t-test, and a Wilcoxon signed-rank test (`scripts/merge_evaluations.py`).
- **Placement** (`main.py placement-eval`): 30 seeds (95000–95029) per task, run at 0.5 cm and 5° with a 20-push cap. The record holds the distance and yaw to the goal after every push, so success at looser tolerances (1 cm/10°, 2 cm/20°) and smaller budgets is scored afterwards (`scripts/summarize_placement.py`); the one-push MPC never looks at either. The conditions differ in the `eta` the MPC plans with: `true_eta` (the true value), `none` (the filter's running MAP, updated from the MPC's own placement pushes; no probing), `nominal_eta` (fixed at the grid point nearest the grid's mean), `farthest_eta` (fixed at the grid point farthest from the truth), and `<prober>:<k>` (k probe pushes first, then the filter's MAP). The gap between `true_eta` and `none` bounds what any probing strategy could gain; pushes saved by knowing `eta` are compared per seed with a Wilcoxon signed-rank test.
