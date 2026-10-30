import argparse
import time

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    ConstantVectorSource,
    Diagram,
    DiagramBuilder,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    PiecewisePolynomial,
    RigidTransform,
    Simulator,
    StartMeshcat,
    TrajectorySource,
)
from pydrake.geometry import Meshcat, Rgba

from src.iiwa import (
    Q_HOME,
    Pusher,
    add_iiwa_and_pusher,
    build_iiwa_pusher_plant,
    demo_tip_waypoints,
    make_push_trajectory,
)
from src.object import Object, Rod, SurfaceNormal, Table, add_object, add_table
from src.pdcontroller import ArmPdController


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
    sim: Sim,
    obj: Object,
    table: Table,
    pusher: Pusher,
    q_ref: PiecewisePolynomial | None = None,
) -> tuple[Diagram, MultibodyPlant]:
    """Assemble the scene (arm with stick, table, block) plus the arm controller."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim.time_step)
    parser = Parser(plant)

    # The arm with its stick, bolted to the world at the origin.
    add_iiwa_and_pusher(pusher, plant, parser)
    # The table, welded down with its top surface at z = 0.
    add_table(table, plant, parser)
    # The block, a free body; its pose gets set per-run in main().
    add_object(obj, parser)

    plant.Finalize()

    iiwa = plant.GetBodyByName("iiwa_link_0").model_instance()
    plant.SetDefaultPositions(iiwa, Q_HOME)

    controller = builder.AddSystem(ArmPdController(plant))
    builder.Connect(plant.get_state_output_port(), controller.get_input_port(0))
    reference = builder.AddSystem(
        ConstantVectorSource(Q_HOME) if q_ref is None else TrajectorySource(q_ref)
    )
    builder.Connect(reference.get_output_port(), controller.get_input_port(1))
    builder.Connect(controller.get_output_port(), plant.get_actuation_input_port())

    if sim.meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, sim.meshcat)

    return builder.Build(), plant


def show_object_goal(meshcat: Meshcat, obj: Object) -> None:
    """Draw a translucent ghost of the block at its goal orientation."""
    X_WG = obj.x_wg()
    r, g, b = obj.color
    for i, (part, center) in enumerate(zip(obj.parts, obj.body_part_centers())):
        sx, sy = part[:2]
        path = f"goal/{i}"
        meshcat.SetObject(path, Box(sx, sy, obj.height), Rgba(r, g, b, 0.35))
        meshcat.SetTransform(path, X_WG @ RigidTransform(center))


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time-step",
        type=float,
        default=1e-4,
        help="simulation timestep (nonnegative)",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without visualization"
    )
    parser.add_argument("--seed", type=int, default=538, help="PRNG seed")
    parser.add_argument(
        "--mu",
        type=float,
        default=0.3,
        help="object-table friction coefficient (nonnegative)",
    )
    parser.add_argument(
        "--rod-x",
        type=float,
        default=0.05,
        help="rod x in the object body frame (m)",
    )
    parser.add_argument(
        "--rod-y",
        type=float,
        default=0,
        help="rod y in the object body frame (m)",
    )
    parser.add_argument(
        "--rod-mass",
        type=float,
        default=0.2,
        help="rod mass (kg), additional to the block's own mass",
    )
    parser.add_argument(
        "--push-density",
        type=float,
        default=2.0,
        help="push-contact density: surface normals per cm of object outline",
    )
    args = parser.parse_args()

    if args.time_step < 0.0:
        raise SystemExit("--time-step must be nonnegative")
    if args.mu < 0.0:
        raise SystemExit("--mu must be nonnegative")
    if args.rod_mass <= 0.0:
        raise SystemExit("--rod-mass must be positive")
    if args.push_density <= 0.0:
        raise SystemExit("--push-density must be positive")
    args.push_density *= 100  # count/meter -> count/cm

    return args


def main(args: argparse.Namespace) -> None:
    sim = Sim(time_step=args.time_step, headless=args.headless)
    rng = np.random.default_rng(seed=args.seed)

    obj = Object(
        rng,
        rod=Rod(x=args.rod_x, y=args.rod_y, mass=args.rod_mass),
        friction=args.mu,
    )
    pusher = Pusher()

    q_traj = make_push_trajectory(
        build_iiwa_pusher_plant(pusher), pusher, demo_tip_waypoints(obj)
    )
    diagram, plant = build_diagram(sim, obj, Table(args.mu), pusher, q_traj)

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # Rest the block on the table top in front of the arm.  Its body frame
    # sits at the footprint's bottom center.
    X_WO = obj.x_wo()
    object_body = plant.GetBodyByName(
        obj.body_box, plant.GetModelInstanceByName(obj.name)
    )
    plant.SetFreeBodyPose(plant_context, object_body, X_WO)
    print(f"object-table friction mu = {args.mu}")
    if obj.rod is not None:
        print(f"rod welded at ({obj.rod.x}, {obj.rod.y}) m, mass {obj.rod.mass} kg")
    print(
        f"object com (body frame) = "
        f"{np.array2string(obj.com_in_body_frame(), precision=3)}"
    )
    push_sites = obj.surface_normals(args.push_density)
    print(
        f"push library: {len(push_sites)} contact sites "
        f"({args.push_density} per cm of outline, all at mid-height)"
    )
    print(f"object starts at {np.array2string(X_WO.translation(), precision=3)}")
    print(f"object goal is {np.array2string(obj.x_wg().translation(), precision=3)}")
    if sim.meshcat is not None:
        show_object_goal(sim.meshcat, obj)
        show_push_sites(sim.meshcat, obj, push_sites)

    simulator = Simulator(diagram, context)
    if not sim.headless:
        simulator.set_target_realtime_rate(1.0)
    end_time = q_traj.end_time()
    simulator.AdvanceTo(0.05 if sim.headless else end_time)

    X_WO_final = plant.EvalBodyPoseInWorld(plant_context, object_body)
    print(f"object ends at   {np.array2string(X_WO_final.translation(), precision=3)}")

    # Keep meshcat open until user wants to quit
    if not sim.headless:
        input("Press Enter to quit.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
