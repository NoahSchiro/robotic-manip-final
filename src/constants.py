"""Values shared across modules, defined once.  Values used by a single module
live next to their use (the arm's maneuver timing in world.iiwa, the push
space's ranges in world.push_space, the PPO rewards in policy.env).

Units: m, s, kg, rad.  World frame: the arm's base is the origin.
"""

ROD_MASS = 0.2  # kg, the hidden rod, known to the estimator

TRUTH_TIME_STEP = 1e-4  # s, the truth sim's
MODEL_TIME_STEP = 2e-3  # s, the Drake prediction model's
SETTLE_TIME = 0.3  # s, measurement window after the push segment ends
LOG_PERIOD = 1e-3  # s, the truth sim's state logging

WORKSPACE_CENTER = (0.55, 0.0)  # m
WORKSPACE_RADIUS = 0.25  # m, the block center must stay within
START_RADIUS = 0.1  # m, start poses are drawn within this of the center
GOAL_RADIUS = 0.1  # m, goals likewise

# The far placement task.  Start and goal lie on the band where the arm
# reaches every lattice push from any block yaw (block center 0.50-0.65 m from
# the base), 0.30-0.45 m apart; the straight line between them then stays at
# least about 0.45 m from the base, where most pushes are still reachable.
# Angles are measured from the world +x axis about the base.
FAR_BAND_RADII = (0.50, 0.65)  # m
FAR_BAND_HALF_ANGLE = 0.96  # rad, 55 deg
FAR_GOAL_DISTANCES = (0.30, 0.45)  # m, start to goal
FAR_BOUNDS_RADII = (0.40, 0.75)  # m, the block center must stay within
FAR_BOUNDS_HALF_ANGLE = 1.31  # rad, 75 deg

# The push lattice the MPC ranks and the PPO policy chooses from: contacts
# about LATTICE_SPACING apart on each face, at LATTICE_SPEEDS speeds and
# LATTICE_DISTANCES push distances per speed (PushSpace.grid).
LATTICE_SPACING = 0.012  # m
LATTICE_SPEEDS = 3
LATTICE_DISTANCES = 3

# MPC pose cost |r - r_g|^2 / l^2 + lambda wrap(yaw - yaw_g)^2, plus
# MPC_TIME_WEIGHT per second of maneuver.  l = 0.07 m is half the T's 14 cm
# extent, so a 7 cm position error costs as much as a 1 rad yaw error.
MPC_LENGTH = 0.07  # m
MPC_YAW_WEIGHT = 1.0  # 1/rad^2
MPC_TIME_WEIGHT = 0.005  # 1/s
