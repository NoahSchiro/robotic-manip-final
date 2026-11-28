"""The KUKA iiwa14 plus the stick pusher welded to its flange, with the IK
and trajectory machinery that moves the stick around the block."""

from copy import copy
from pathlib import Path

import numpy as np
from pydrake.all import (
    InverseKinematics,
    MultibodyPlant,
    Parser,
    PiecewisePolynomial,
    RigidTransform,
    RotationMatrix,
    Solve,
)

from src import ASSET_DIR
from src.world.action import Action
from src.world.geometry import SurfaceNormal, SurfacePart, distance_to_footprint
from src.world.object import Object

# Joint angles (rad) of the arm's starting posture.
Q_HOME = np.array([0.0, 0.4, 0.0, -1.6, 0.0, 1.2, 0.0])

# Push maneuver timing and geometry.
STANDOFF = 0.015  # m, approach distance before the contact along the normal
MOVE_TIME = 0.35  # s, nominal duration of a free-space leg
HOLD_TIME = 0.1  # s
# s, the tip rests at the standoff after lowering: the lowering overshoots
# vertically by 1-1.5 cm about 0.17 s later, and the push must start after it.
DWELL_TIME = 0.3
HOVER_CLEARANCE = 0.04  # m, transfer height above the block top
RELEASE_DISTANCE = 0.01  # m, retreat along the normal after the push
RELEASE_SPEED = 0.05  # m/s
# Free-space legs may use this share of each joint's velocity limit.
JOINT_SPEED_FRACTION = 0.5
# push_maneuver's legs, in order.  In a make_push_trajectory plan, break k + 1
# ends leg k, and break 0 is the trajectory start.
LEG_TRANSFER, LEG_LOWER, LEG_DWELL, LEG_PUSH, LEG_RELEASE, LEG_LIFT = range(6)
# Legs that move through free space: the transfer, the lowering, and the lift.
# The dwell is a zero-motion leg and keeps its nominal duration.
FREE_SPACE_LEGS = (LEG_TRANSFER, LEG_LOWER, LEG_LIFT)
TOOL_DOWN = RotationMatrix.MakeXRotation(np.pi)  # tool axis -> straight down


class Pusher:
    """The stick pusher: a vertical cylinder welded to the flange
    (iiwa_link_7) along its +z tool axis; the tip is what pushes."""

    def __init__(self) -> None:
        self.asset_dir = ASSET_DIR
        self.name = "pusher"
        self.body = "stick"
        self.radius = 0.01
        self.length = 0.15
        self.mass = 0.2
        # The stick surface's friction; Drake combines it with the block's for
        # stick-block contact.
        self.friction = 0.6
        self.color = (0.90, 0.45, 0.10)
        # Pa, stiffer than the block's 1e6.
        self.hydroelastic_modulus = 1.0e7

    @property
    def tip_point(self) -> np.ndarray:
        """The stick tip in the stick's frame: the point that pushes."""
        return np.array([0.0, 0.0, self.length])

    def sdf(self) -> str:
        """Render the stick's model SDF text."""
        m = self.mass
        r, L = self.radius, self.length
        ixx = m * (3 * r**2 + L**2) / 12
        izz = m * r**2 / 2
        center = L / 2
        red, green, blue = self.color

        return f"""
          <?xml version="1.0"?>
            <sdf version="1.7">
              <model name="{self.name}">
                <link name="{self.body}">
                  <inertial>
                    <pose>0 0 {center} 0 0 0</pose>
                    <mass>{m}</mass>
                    <inertia>
                      <ixx>{ixx:.6g}</ixx><ixy>0</ixy><ixz>0</ixz>
                      <iyy>{ixx:.6g}</iyy><iyz>0</iyz>
                      <izz>{izz:.6g}</izz>
                    </inertia>
                  </inertial>
                  <collision name="collision">
                    <pose>0 0 {center} 0 0 0</pose>
                    <geometry><cylinder><radius>{r}</radius><length>{L}</length></cylinder></geometry>
                    <drake:proximity_properties>
                      <drake:compliant_hydroelastic/>
                      <drake:mesh_resolution_hint>0.005</drake:mesh_resolution_hint>
                      <drake:hydroelastic_modulus>{self.hydroelastic_modulus}</drake:hydroelastic_modulus>
                      <drake:mu_static>{self.friction}</drake:mu_static>
                      <drake:mu_dynamic>{self.friction}</drake:mu_dynamic>
                    </drake:proximity_properties>
                  </collision>
                  <visual name="visual">
                    <pose>0 0 {center} 0 0 0</pose>
                    <geometry><cylinder><radius>{r}</radius><length>{L}</length></cylinder></geometry>
                    <material><diffuse>{red} {green} {blue} 1</diffuse></material>
                  </visual>
                </link>
              </model>
            </sdf>"""

    def write_sdf(self) -> Path:
        """Write the stick's SDF into the assets directory, returning its path."""
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        path = self.asset_dir / f"{self.name}.sdf"
        path.write_text(self.sdf())
        return path


def add_iiwa_and_pusher(pusher: Pusher, plant: MultibodyPlant, parser: Parser) -> None:
    """Parse the iiwa14 into `plant` with `pusher`'s stick welded to the
    flange, bolted down at the origin."""
    parser.AddModelsFromUrl(
        "package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
    )
    parser.AddModelsFromString(pusher.sdf(), "sdf")
    plant.WeldFrames(
        plant.GetFrameByName("iiwa_link_7"),
        plant.GetFrameByName(pusher.body, plant.GetModelInstanceByName(pusher.name)),
        RigidTransform(),
    )
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))


def build_iiwa_pusher_plant(pusher: Pusher) -> MultibodyPlant:
    """Build the arm-plus-stick plant that the IK runs on."""
    # No table or block: the IK must not be free to move them.
    plant = MultibodyPlant(0.0)
    add_iiwa_and_pusher(pusher, plant, Parser(plant))
    plant.Finalize()
    return plant


def solve_tip_ik(
    plant: MultibodyPlant, pusher: Pusher, X_WT: RigidTransform, q_seed: np.ndarray
) -> np.ndarray:
    """Solve for the arm's joint angles that put `pusher`'s tip at `X_WT`."""
    world = plant.world_frame()
    stick = plant.GetFrameByName(pusher.body, plant.GetModelInstanceByName(pusher.name))
    ik = InverseKinematics(plant)
    ik.AddPositionConstraint(
        stick,
        pusher.tip_point,
        world,
        X_WT.translation(),
        X_WT.translation(),
    )
    ik.AddOrientationConstraint(stick, RotationMatrix(), world, X_WT.rotation(), 0.0)

    # Elbow-forward, tool-down fallback seeds spread over shoulder yaw: the IK
    # is nonconvex, so a diverse seed can rescue a locally infeasible solve.
    ik_fallback_shoulder_yaws = (0.0, 0.5, -0.5, 1.0)
    ik_fallback_seed = np.array([0.0, 1.0, 0.0, -1.8, 0.0, 1.4, 0.0])

    # Warm start with the caller's seed, then the fallback shapes.
    seeds = [
        q_seed,
        *(
            ik_fallback_seed + np.array([yaw, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            for yaw in ik_fallback_shoulder_yaws
        ),
    ]
    for seed in seeds:
        ik.get_mutable_prog().SetInitialGuess(ik.q(), seed)
        result = Solve(ik.get_mutable_prog())
        if result.is_success():
            return result.GetSolution(ik.q())
    raise RuntimeError(f"no IK solution for stick tip pose\n{X_WT}")


def transfer_height(obj: Object) -> float:
    """The tip's world height (m) for transfer and lift legs over `obj`."""
    return obj.height + HOVER_CLEARANCE


def hover_tip_pose(obj: Object) -> RigidTransform:
    """The tip pose parked a few centimeters above the block, tool pointing
    straight down."""
    # The body frame sits at the footprint's bottom center, so obj.pose's x, y
    # lies over the block, whose top face is at z = obj.height.
    x_block, y_block = obj.pose[:2]
    return RigidTransform(TOOL_DOWN, np.array([x_block, y_block, transfer_height(obj)]))


def clear_sites(
    parts: list[SurfacePart],
    sites: list[SurfaceNormal],
    stick_radius: float,
    clearance: float,
) -> list[SurfaceNormal]:
    """Keep the sites where a stick of `stick_radius`, lowered to its standoff
    point, clears the `parts`' footprint by `clearance`."""
    # At the T's inner corners the standoff sits within a stick radius of the
    # other part, so the stick would come down on top of the block.
    return [
        site
        for site in sites
        if distance_to_footprint(
            parts, *(site.position[:2] - STANDOFF * site.normal[:2])
        )
        > stick_radius + clearance
    ]


def push_maneuver(
    X_WO: RigidTransform, action: Action, transfer_height: float
) -> list[tuple[RigidTransform, float]]:
    """Compute the tip's maneuver for `action` on a block at `X_WO` (m, rad):
    waypoints paired with each leg's nominal duration (s), in the LEG_* order.
    The legs are transfer over the standoff point, lowering to it, a dwell
    there, the push, a release RELEASE_DISTANCE back along the normal, and a
    vertical lift to `transfer_height` (m). The tool points down throughout."""
    contact = X_WO @ action.contact.position
    normal = X_WO.rotation() @ action.contact.normal
    approach = contact - STANDOFF * normal
    push_end = contact + action.velocity * action.time * normal
    released = push_end - RELEASE_DISTANCE * normal
    push_time = (STANDOFF + action.velocity * action.time) / action.velocity
    over = np.array([approach[0], approach[1], transfer_height])
    lifted = np.array([released[0], released[1], transfer_height])
    return [
        (_tool_down(over), MOVE_TIME),
        (_tool_down(approach), MOVE_TIME),
        (_tool_down(approach), DWELL_TIME),
        (_tool_down(push_end), push_time),
        (_tool_down(released), RELEASE_DISTANCE / RELEASE_SPEED),
        (_tool_down(lifted), MOVE_TIME),
    ]


def _tool_down(position: np.ndarray) -> RigidTransform:
    return RigidTransform(TOOL_DOWN, position)


def maneuver_time(obj: Object, action: Action) -> float:
    """Get `action`'s nominal maneuver time (s): push_maneuver's durations plus
    the final hold. Ignores the joint-speed stretching of free-space legs in
    make_push_trajectory, which needs IK solutions planning does not compute."""
    maneuver = push_maneuver(obj.x_wo(), action, transfer_height(obj))
    return sum(duration for _pose, duration in maneuver) + HOLD_TIME


def free_space_duration(
    q_from: np.ndarray, q_to: np.ndarray, nominal: float, velocity_limits: np.ndarray
) -> float:
    """Get a free-space leg's duration (s): `nominal`, or longer when its
    slowest joint would otherwise run past JOINT_SPEED_FRACTION of its
    velocity limit (rad/s)."""
    # The IK can land on a far-off arm shape for a nearby tip pose.
    fastest_need = np.abs(q_to - q_from) / (JOINT_SPEED_FRACTION * velocity_limits)
    return max(nominal, float(fastest_need.max()))


def make_push_trajectory(
    ik_plant: MultibodyPlant,
    pusher: Pusher,
    obj: Object,
    action: Action,
    q_seed: np.ndarray,
) -> PiecewisePolynomial:
    """Plan from the arm posture `q_seed` at time 0 through the push maneuver,
    its dwell, release, lift, and hold. Free-space legs (FREE_SPACE_LEGS) last
    free_space_duration; the other legs keep their nominal durations."""
    maneuver = push_maneuver(obj.x_wo(), action, transfer_height(obj))

    q_samples = [np.asarray(q_seed, dtype=float)]
    for pose, _duration in maneuver:
        q_samples.append(solve_tip_ik(ik_plant, pusher, pose, q_samples[-1]))

    limits = ik_plant.GetVelocityUpperLimits()
    durations = [
        free_space_duration(q_samples[k], q_samples[k + 1], nominal, limits)
        if k in FREE_SPACE_LEGS
        else nominal
        for k, (_pose, nominal) in enumerate(maneuver)
    ]
    breaks = np.cumsum([0.0, *durations, HOLD_TIME])
    samples = np.column_stack([*q_samples, q_samples[-1]]).astype(float)
    return PiecewisePolynomial.FirstOrderHold(
        [float(t) for t in breaks], samples.tolist()
    )


def hold_trajectory(q: np.ndarray) -> PiecewisePolynomial:
    """Get the trajectory that holds the arm still at `q`."""
    return PiecewisePolynomial.FirstOrderHold(
        [0.0, 1.0], np.column_stack([q, q]).tolist()
    )


def shift_trajectory(
    q_traj: PiecewisePolynomial, start_time: float
) -> PiecewisePolynomial:
    """Copy `q_traj`, planned from time 0, to begin at `start_time` on a running
    sim's clock. Its first posture must be where the arm is parked."""
    shifted = copy(q_traj)
    shifted.shiftRight(start_time)
    return shifted
