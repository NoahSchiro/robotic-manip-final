# System description

## Physical and decision state

The truth world contains a KUKA iiwa14 with a vertical stick, a table, and a
T-shaped block. The unknown vector is `eta = (mu, bx, by)`: table friction and
the position of a known-mass welded rod in the block frame. Lengths are meters,
time is seconds, angles are radians, and planar object states are
`(x, y, yaw, vx, vy, wz)` in the world frame. Rod coordinates and contact sites
are in the object body frame. The truth simulator and its object and arm state
persist across pushes; computation between actions consumes no simulation time.

A push action selects a body-frame contact site and inward normal, commanded
tip speed in m/s, and contact duration in seconds. The arm transfers at block
top plus 4 cm, lowers to the standoff point, pushes, backs off, lifts locally,
and holds. The next action starts from the arm's measured joint positions.

| Event | Meaning |
| --- | --- |
| Start | Tip at standoff; prediction initial condition and start of recorded tip path |
| Push end | Commanded push segment ends; object velocity is read here |
| Measurement end | `settle_time` after push end; object pose is read here |
| Action end | Backoff, lift, and hold complete; object and arm states are the next decision state |

The estimator measurement is the multi-time vector `(x, y, yaw)` at
measurement end concatenated with `(vx, vy, wz)` at push end. Its prediction
uses the actual tip path measured within that window. The action-end state is
the prospective MPC target definition; the current surrogate does not yet
predict it. Planning may use only a nominal or modeled tip path available
before execution.

## Estimation and models

The reference filter is a grid Bayes update over 405 parameter hypotheses.
For each push it compares the measured vector with each hypothesis's Drake
prediction using the calibrated residual matrix `CALIBRATED_R` and a
Student-t likelihood with default 3 degrees of freedom. The MAP estimate is
used for scoring. The range-normalized error is
`sqrt(sum_i ((eta_MAP_i - eta_true_i) / span_i)^2)`, where each span is the
full hypothesis-grid range for that parameter. Reports also retain signed
friction and body-frame rod-position errors.

The existing surrogate input is `eta`, world-frame start pose and velocity,
body-frame contact point and normal, action speed and duration, and 16
world-frame tip-path samples. Its output is window-end pose displacement and
push-end velocity in the start pose's frame. The current generator samples
independent single pushes from the Drake prediction model. It has no
action-end target in the current fitted model. Schema 3 saves the modeled
action-end pose displacement, velocity, event time, and a downsampled full
tip path for later MPC fitting. It also saves each trajectory seed and assigns it to
training, development, or held-out by seed modulo 10 (80%, 10%, 10% in
expectation). Standardization fits training rows only. Existing
recorded full-arm trials group consecutive pushes by `trial_id` and retain
start, push-end, measurement-end, and action-end object states. New trials
also store the action-end event time; older trials lack that time and remain
readable.

## Evaluation protocol

Random-only Drake evaluation needs no learned assets. The paired evaluation
uses the same episode-condition seed for learned and random probing, with an
independent random-action stream, and matches the random baseline's push
budget to the learned run. A push consumes budget only after execution;
terminal reward is charged for immediate stop, exhausted budget, or
unreachable actions. The random-only report and the paired report include
range-normalized MAP error, per-component errors, reward, pushes, and reason.
The surrogate world remains a separate comparison and requires compatible
ensemble and path-family assets.
