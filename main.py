import os
import time

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ConstantVectorSource,
    DiagramBuilder,
    MeshcatVisualizer,
    Parser,
    PiecewisePolynomial,
    Simulator,
    StartMeshcat,
    TrajectorySource,
)
from pydrake.geometry import Meshcat

from src.iiwa import (
    Q_HOME,
    Pusher,
    add_iiwa_and_pusher,
    build_iiwa_pusher_plant,
    demo_tip_waypoints,
    make_push_trajectory,
)
from src.object import Object, Table, add_object, add_table
from src.pdcontroller import ArmPdController


class Sim:
    meshcat: Meshcat | None
    time_step: float
    headless: bool

    def __init__(self, headless, time_step):
        """Set up the run's pacing and, unless headless, its Meshcat server."""
        self.time_step = time_step
        self.headless = headless
        self.meshcat = None
        if headless:
            return
        self.meshcat = StartMeshcat()
        # Wait for connect, or timeout and run anyhow.
        deadline = time.time() + 10.0
        while time.time() < deadline and self.meshcat.GetNumActiveConnections() == 0:
            time.sleep(0.1)


def build_diagram(
    sim: Sim, obj: Object, pusher: Pusher, q_ref: PiecewisePolynomial | None = None
):
    """Assemble the scene (arm with stick, table, block) plus the arm controller."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim.time_step)
    parser = Parser(plant)

    # The arm with its stick, bolted to the world at the origin.
    add_iiwa_and_pusher(pusher, plant, parser)
    # The table, welded down with its top surface at z = 0.
    add_table(Table(), plant, parser)
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


def main():
    sim = Sim(
        time_step=1e-4,
        headless=bool(os.environ.get("HEADLESS"))
    )

    obj = Object()
    pusher = Pusher()
    q_traj = make_push_trajectory(
        build_iiwa_pusher_plant(pusher), pusher, demo_tip_waypoints(obj)
    )
    diagram, plant = build_diagram(sim, obj, pusher, q_traj)

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # Rest the block on the table top in front of the arm.  Its body frame
    # sits at the footprint's bottom center.
    X_WO = obj.x_wo()
    object_body = plant.GetBodyByName(
        obj.body_box, plant.GetModelInstanceByName(obj.name)
    )
    plant.SetFreeBodyPose(plant_context, object_body, X_WO)
    print(f"object starts at {np.array2string(X_WO.translation(), precision=3)}")

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
    main()
