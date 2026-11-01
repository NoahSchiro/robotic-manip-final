"""Scene construction and mechanics: meshcat bootstrap, Drake
diagram assembly, and the overlays drawn onto the running scene."""

import time

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    ConstantVectorSource,
    Context,
    Diagram,
    DiagramBuilder,
    LogVectorOutput,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    PiecewisePolynomial,
    RigidBody,
    RigidTransform,
    RotationMatrix,
    StartMeshcat,
    TrajectorySource,
)
from pydrake.geometry import Meshcat, Rgba

from src.geometry import SurfaceNormal
from src.iiwa import (
    Q_HOME,
    Pusher,
    add_iiwa_and_pusher,
    build_iiwa_pusher_plant,
    hover_tip_pose,
    solve_tip_ik,
)
from src.object import Object, Table, add_object, add_table
from src.pdcontroller import ArmPdController

# Name of the plant-state logger build_diagram adds when asked to log.
STATE_LOGGER = "state_logger"

# Color of the ghost showing where the MPC predicts its push leaves the block.
PREDICTION_COLOR = (0.95, 0.80, 0.15)


class Sim:
    meshcat: Meshcat | None
    time_step: float
    headless: bool

    def __init__(self, headless: bool, time_step: float) -> None:
        """Set up the run's pacing and, unless headless, its Meshcat server."""
        self.time_step = time_step
        self.headless = headless
        self.meshcat = None
        if headless:
            return
        # StartMeshcat lives in pydrake's unannotated _geometry_extra.py,
        # so strict mypy sees the call as untyped.
        self.meshcat = StartMeshcat()  # type: ignore[no-untyped-call]
        # Wait for connect, or timeout and run anyhow.
        deadline = time.time() + 10.0
        while time.time() < deadline and self.meshcat.GetNumActiveConnections() == 0:
            time.sleep(0.1)


def build_diagram(
    time_step: float,
    obj: Object,
    table: Table,
    pusher: Pusher,
    meshcat: Meshcat | None = None,
    q_ref: PiecewisePolynomial | None = None,
    log_period: float | None = None,
) -> tuple[Diagram, MultibodyPlant]:
    """Assemble the scene (arm with stick, table, block) plus the arm controller.

    With `log_period`, the plant state is also logged at that period, in a
    logger named STATE_LOGGER (see observation.py)."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    parser = Parser(plant)

    # The arm with its stick, bolted to the world at the origin.
    add_iiwa_and_pusher(pusher, plant, parser)
    # The table, welded down with its top surface at z = 0.
    add_table(table, plant, parser)
    # The block, a free body; its pose gets set per-run via place_object.
    add_object(obj, parser)

    plant.Finalize()

    iiwa = plant.GetBodyByName("iiwa_link_0").model_instance()
    q_start = Q_HOME if q_ref is None else q_ref.value(0.0).ravel()
    plant.SetDefaultPositions(iiwa, q_start)

    controller = builder.AddSystem(ArmPdController(plant))
    builder.Connect(plant.get_state_output_port(), controller.get_input_port(0))
    if q_ref is None:
        reference = builder.AddSystem(
            ConstantVectorSource(np.concatenate([Q_HOME, np.zeros(len(Q_HOME))]))
        )
    else:
        reference = builder.AddSystem(
            TrajectorySource(q_ref, output_derivative_order=1)
        )
    builder.Connect(reference.get_output_port(), controller.get_input_port(1))
    builder.Connect(controller.get_output_port(), plant.get_actuation_input_port())

    if log_period is not None:
        logger = LogVectorOutput(plant.get_state_output_port(), builder, log_period)
        logger.set_name(STATE_LOGGER)

    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    return builder.Build(), plant


def place_object(plant: MultibodyPlant, context: Context, obj: Object) -> RigidBody:
    """Rest the block on the table top at its start pose, returning the body
    for later pose queries."""
    X_WO = obj.x_wo()
    object_body = plant.GetBodyByName(
        obj.body_box, plant.GetModelInstanceByName(obj.name)
    )
    plant.SetFreeBodyPose(context, object_body, X_WO)
    return object_body


def show_scene(meshcat: Meshcat, obj: Object, pusher: Pusher, time_step: float) -> None:
    """Draw the scene as it stands between pushes: the block at its pose and
    the arm parked over it, as each push starts."""
    q_hover = solve_tip_ik(
        build_iiwa_pusher_plant(pusher), pusher, hover_tip_pose(obj), Q_HOME
    )
    diagram, plant = build_diagram(
        time_step, obj, Table(obj.friction), pusher, meshcat=meshcat
    )
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    place_object(plant, plant_context, obj)
    iiwa = plant.GetBodyByName("iiwa_link_0").model_instance()
    plant.SetPositions(plant_context, iiwa, q_hover)
    diagram.ForcedPublish(context)


def show_ghost(
    meshcat: Meshcat,
    obj: Object,
    X_WG: RigidTransform,
    prefix: str,
    color: tuple[float, float, float],
) -> None:
    """Draw a translucent ghost of the block at `X_WG` under `prefix`."""
    r, g, b = color
    for i, (part, center) in enumerate(zip(obj.parts, obj.body_part_centers())):
        sx, sy = part[:2]
        path = f"{prefix}/{i}"
        meshcat.SetObject(path, Box(sx, sy, obj.height), Rgba(r, g, b, 0.35))
        meshcat.SetTransform(path, X_WG @ RigidTransform(center))


def show_object_goal(meshcat: Meshcat, obj: Object) -> None:
    """Draw a translucent ghost of the block at its goal pose."""
    show_ghost(meshcat, obj, obj.x_wg(), "goal", obj.color)


def show_prediction(meshcat: Meshcat, obj: Object, pose: np.ndarray) -> None:
    """Draw a translucent ghost of the block at planar `pose` (x, y, yaw),
    where the MPC predicts its chosen push leaves it."""
    x, y, yaw = pose
    X_WP = RigidTransform(RotationMatrix.MakeZRotation(yaw), np.array([x, y, 0.0]))
    show_ghost(meshcat, obj, X_WP, "prediction", PREDICTION_COLOR)


def show_push_sites(
    meshcat: Meshcat,
    obj: Object,
    sites: list[SurfaceNormal],
    arrow_length: float = 0.02,
) -> None:
    """Draw the push library in world frame: one segment per site, approaching
    its contact point from outside along the inward normal."""
    X_WO = obj.x_wo()
    tips = [X_WO @ site.position for site in sites]
    # Tails reach outside the block: an inward arrow drawn from the contact
    # point onward would hide inside the block's opaque visual.
    tails = [
        tip - arrow_length * (X_WO.rotation() @ site.normal)
        for tip, site in zip(tips, sites)
    ]
    meshcat.SetLineSegments(
        "push_sites",
        np.stack(tails, axis=1),
        np.stack(tips, axis=1),
        line_width=4.0,
        rgba=Rgba(0.1, 0.65, 0.15, 1.0),
    )
