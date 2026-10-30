"""Entry point."""

import argparse

import numpy as np
from pydrake.all import Simulator

from src.iiwa import (
    Pusher,
    build_iiwa_pusher_plant,
    demo_tip_waypoints,
    make_push_trajectory,
)
from src.object import Object, Rod, Table
from src.scene import (
    Sim,
    build_diagram,
    place_object,
    show_object_goal,
    show_push_sites,
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
    args.push_density *= 100  # count/cm -> count/m

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
    diagram, plant = build_diagram(
        args.time_step, obj, Table(args.mu), pusher, meshcat=sim.meshcat, q_ref=q_traj
    )

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # Rest the block on the table top in front of the arm.  Its body frame
    # sits at the footprint's bottom center.
    object_body = place_object(plant, plant_context, obj)
    X_WO = obj.x_wo()
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
        f"({args.push_density} per m of outline, all at mid-height)"
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
