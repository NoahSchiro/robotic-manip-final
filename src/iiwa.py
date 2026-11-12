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

from . import ASSET_DIR
from .action import Action
from .geometry import SurfaceNormal, SurfacePart, distance_to_footprint
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
    obj: Object, action: Action, hover: RigidTransform
) -> list[tuple[RigidTransform, float]]:
    """Compute the tip's world-frame maneuver for `action`: waypoints paired
    with each straight segment's duration. The final waypoint is lifted
    locally above the release point, with the tool pointing down."""
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
    # Back off only far enough to release the contact: the push has usually
    # turned the block, so retracing it all can drag the T's other part along.
    backed_off = RigidTransform(tool_down, push_end.translation() - STANDOFF * normal)
    raised = RigidTransform(
        tool_down,
        np.array(
            [
                backed_off.translation()[0],
                backed_off.translation()[1],
                hover.translation()[2],
            ]
        ),
    )

    return [
        (over, MOVE_TIME),  # move over the standoff point
        (lowered, MOVE_TIME),  # lower to the contact height
        (push_end, push_time),  # push at the commanded speed and contact time
        (backed_off, STANDOFF / action.velocity),  # back off at the same speed
        (raised, MOVE_TIME),  # raise
    ]


def make_push_trajectory(
    ik_plant: MultibodyPlant,
    pusher: Pusher,
    obj: Object,
    action: Action,
    q_seed: np.ndarray,
) -> PiecewisePolynomial:
    """Plan from the current arm posture through a push, local lift, and hold.
    The first segment transfers at the hover clearance height."""
    hover = hover_tip_pose(obj)
    maneuver = push_maneuver(obj, action, hover)

    q_samples = [np.asarray(q_seed, dtype=float)]
    for pose, _duration in maneuver:
        q_samples.append(solve_tip_ik(ik_plant, pusher, pose, q_samples[-1]))

    durations = [duration for _pose, duration in maneuver]
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


def transit_time(
    q_from: np.ndarray, q_to: np.ndarray, velocity_limits: np.ndarray
) -> float:
    """Get how long the arm takes between postures: MOVE_TIME, or longer when
    that would run a joint past `velocity_limits`."""
    # The IK can land on a far-off arm shape for a nearby tip pose, a move
    # MOVE_TIME can't cover at any speed the arm is rated for.
    return max(MOVE_TIME, float((np.abs(q_to - q_from) / velocity_limits).max()))


def chain_from_parked(
    q_parked: np.ndarray,
    q_traj: PiecewisePolynomial,
    start_time: float,
    transit: float,
) -> PiecewisePolynomial:
    """Put `q_traj`, planned from time 0, on a running sim's clock: from
    `start_time` the arm moves from `q_parked` to `q_traj`'s start over
    `transit` seconds, then follows `q_traj`."""
    q_start = q_traj.value(0.0).ravel()
    chained = PiecewisePolynomial.FirstOrderHold(
        [start_time, start_time + transit],
        np.column_stack([q_parked, q_start]).tolist(),
    )
    shifted = copy(q_traj)
    shifted.shiftRight(chained.end_time())
    chained.ConcatenateInTime(shifted)
    return chained
