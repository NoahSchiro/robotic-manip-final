"""Entry point."""

import argparse
import time

import numpy as np
from pydrake.all import Simulator

from src import ASSET_DIR
from src.action import Action
from src.dynamics import ParallelPushModel
from src.estimator import DEFAULT_NU, GridBayesFilter, load_R
from src.geometry import SurfaceNormal
from src.iiwa import Pusher, build_iiwa_pusher_plant, make_push_trajectory
from src.object import Object, Rod, Table
from src.observation import record_random_push
from src.params import Eta, GridSpec, make_grid
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
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="run random pushes and estimate (mu, rod x, rod y) after each",
    )
    parser.add_argument(
        "--pushes", type=int, default=5, help="pushes to run with --estimate"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="prediction processes for --estimate (default: all CPUs)",
    )
    parser.add_argument(
        "--model-time-step",
        type=float,
        default=2e-3,
        help="estimator prediction model timestep",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.1,
        help="measurement delay after each push segment ends (s)",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=DEFAULT_NU,
        help="estimator Student-t degrees of freedom (0: Gaussian likelihood)",
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
    if args.pushes < 1:
        raise SystemExit("--pushes must be at least 1")
    if args.nu < 0.0:
        raise SystemExit("--nu must be nonnegative")
    args.push_density *= 100  # count/cm -> count/m

    return args


def demo_action(obj: Object, push_sites: list[SurfaceNormal]) -> Action:
    """Pick the demo's push: the face nearest the arm, at a slow speed for a
    short contact time."""
    R_WO = obj.x_wo().rotation()
    site = max(push_sites, key=lambda s: (R_WO @ s.normal)[0])
    return Action(contact=site, velocity=0.05, time=0.4)


def run_estimation(
    args: argparse.Namespace,
    sim: Sim,
    rng: np.random.Generator,
    obj: Object,
    pusher: Pusher,
) -> None:
    """Run random pushes from the push library, chained from each push's
    final pose, and update the grid Bayes filter after each."""
    eta_true = Eta(args.mu, args.rod_x, args.rod_y)
    push_sites = obj.surface_normals(args.push_density)
    grid = make_grid(GridSpec(), obj.parts)
    r_path = ASSET_DIR / "R.npy"
    R = load_R(r_path)
    print(
        f"estimating eta = (mu, rod x, rod y), true {eta_true}, over "
        f"{len(grid)} hypotheses; R "
        f"{'from ' + str(r_path) if r_path.exists() else 'uncalibrated default'}"
    )

    with ParallelPushModel(
        args.workers, time_step=args.model_time_step, rod_mass=args.rod_mass
    ) as model:
        belief = GridBayesFilter(grid, model, R, nu=args.nu or None)
        print(f"push 0: entropy {belief.entropy():.3f}")
        for k in range(1, args.pushes + 1):
            if sim.meshcat is not None:
                show_object_goal(sim.meshcat, obj)
                show_push_sites(sim.meshcat, obj, push_sites)
            obs = record_random_push(
                rng,
                obj,
                pusher,
                push_sites,
                args.time_step,
                args.settle_time,
                meshcat=sim.meshcat,
                realtime_rate=0.0 if sim.headless else 1.0,
            )
            started = time.time()
            belief.update(obs)
            mu, bx, by = belief.map_eta().as_array()
            d_mu, d_bx, d_by = belief.estimation_error(eta_true)
            print(
                f"push {k}: v={obs.action.velocity:.2f} m/s for "
                f"{obs.action.time:.1f} s, update {time.time() - started:.1f} s, "
                f"entropy {belief.entropy():.3f}, max p {belief.probs.max():.3f}\n"
                f"  MAP mu={mu:.3f} b=({bx:+.4f}, {by:+.4f}) m; error "
                f"mu {d_mu:+.3f}, b ({d_bx * 100:+.2f}, {d_by * 100:+.2f}) cm"
            )
            # Chain the next push from where this one left the object.
            x, y, yaw = obs.end.planar()[:3]
            obj.pose = (float(x), float(y), float(yaw))


def main(args: argparse.Namespace) -> None:
    sim = Sim(time_step=args.time_step, headless=args.headless)
    rng = np.random.default_rng(seed=args.seed)

    obj = Object(
        rng,
        rod=Rod(x=args.rod_x, y=args.rod_y, mass=args.rod_mass),
        friction=args.mu,
    )
    pusher = Pusher()

    if args.estimate:
        run_estimation(args, sim, rng, obj, pusher)
        if not sim.headless:
            input("Press Enter to quit.")
        return

    push_sites = obj.surface_normals(args.push_density)
    action = demo_action(obj, push_sites)
    q_traj = make_push_trajectory(build_iiwa_pusher_plant(pusher), pusher, obj, action)
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
    print(
        f"push library: {len(push_sites)} contact sites "
        f"({args.push_density} per m of outline, all at mid-height)"
    )
    print(
        f"demo push: {action.velocity:.2f} m/s along the contact's inward "
        f"normal for {action.time:.2f} s in contact"
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
