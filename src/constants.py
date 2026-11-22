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
