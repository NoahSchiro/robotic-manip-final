"""The KUKA iiwa14 plus the stick pusher welded to its flange, with the IK
and trajectory machinery that moves the stick around the block."""

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

from . import ASSET_DIR
from .action import Action
from .object import Object

# Home position for the iiwa
Q_HOME = np.array([0.0, 0.4, 0.0, -1.6, 0.0, 1.2, 0.0])

# Maneuver knobs. This moves reasonably fast
STANDOFF = 0.015
MOVE_TIME = 0.35
HOLD_TIME = 0.1


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
        # grippier than table, can play with this later
        self.friction = 0.6
        self.color = (0.90, 0.45, 0.10)
        # stiffer than the block
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
    # Add stick
    plant.WeldFrames(
        plant.GetFrameByName("iiwa_link_7"),
        plant.GetFrameByName(pusher.body, plant.GetModelInstanceByName(pusher.name)),
        RigidTransform(),
    )
    # Weld the iiwa to the floor
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))


def build_iiwa_pusher_plant(pusher: Pusher) -> MultibodyPlant:
    """Build the arm-plus-stick plant that the IK runs on."""
    # No table or block: the IK must not be free to move them (ps2's trick).
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

    # Elbow-forward, tool-down IK fallback seeds at a spread of shoulder yaws:
    # the problem is nonconvex, so a diverse seed rescues the occasional
    # locally-infeasible solve.
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


def hover_tip_pose(obj: Object) -> RigidTransform:
    """The tip pose parked a few centimeters above the block, tool pointing
    straight down."""
    # The block's body frame sits at the footprint's bottom center, so
    # obj.pose's x, y is a point directly over the block and its top face
    # is at z = obj.height.
    x_block, y_block = obj.pose[:2]
    R_down = RotationMatrix.MakeXRotation(np.pi)  # tool axis -> straight down
    hover_clearance = 0.04
    return RigidTransform(
        R_down, np.array([x_block, y_block, obj.height + hover_clearance])
    )


def push_maneuver(
    obj: Object, action: Action, hover: RigidTransform
) -> list[tuple[RigidTransform, float]]:
    """Compute the tip's world-frame maneuver for `action`: waypoints paired
    with each straight segment's duration, starting and ending parked at
    `hover` with the tool pointing down."""
    X_WO = obj.x_wo()
    contact = X_WO @ action.contact.position
    normal = X_WO.rotation() @ action.contact.normal

    approach = contact - STANDOFF * normal
    tool_down = hover.rotation()
    over = RigidTransform(
        tool_down, np.array([approach[0], approach[1], hover.translation()[2]])
    )
    lowered = RigidTransform(tool_down, approach)
    push_end = RigidTransform(
        tool_down, contact + action.velocity * action.time * normal
    )
    push_time = (STANDOFF + action.velocity * action.time) / action.velocity

    return [
        (over, MOVE_TIME),  # move over the standoff point
        (lowered, MOVE_TIME),  # lower to the contact height
        (push_end, push_time),  # push at the commanded speed and contact time
        (lowered, push_time),  # back off the way it came
        (over, MOVE_TIME),  # raise
        (hover, MOVE_TIME),  # return to the hover
    ]


def make_push_trajectory(
    ik_plant: MultibodyPlant, pusher: Pusher, obj: Object, action: Action
) -> PiecewisePolynomial:
    """Plan the joint-space trajectory that starts parked at the block's hover
    pose, executes `push_maneuver` for `action`, and parks there for HOLD_TIME."""
    hover = hover_tip_pose(obj)
    maneuver = push_maneuver(obj, action, hover)

    q_samples = [solve_tip_ik(ik_plant, pusher, hover, Q_HOME)]
    for pose, _duration in maneuver:
        q_samples.append(solve_tip_ik(ik_plant, pusher, pose, q_samples[-1]))

    durations = [duration for _pose, duration in maneuver]
    breaks = np.cumsum([0.0, *durations, HOLD_TIME])
    samples = np.column_stack([*q_samples, q_samples[-1]]).astype(float)
    return PiecewisePolynomial.FirstOrderHold(
        [float(t) for t in breaks], samples.tolist()
    )
