"""Entry point."""

import argparse
import time

import numpy as np
from pydrake.all import Simulator
from pydrake.geometry import Meshcat

from src.action import Action, push_library
from src.calibration import calibrate, make_sample_setup
from src.cli import (
    add_shared_options,
    finish_shared_options,
    push_model_config,
    truth_sim_config,
)
from src.dynamics import ParallelPushModel, Pose
from src.episode import EpisodeResult, run_episode
from src.estimator import CALIBRATED_R, GridBayesFilter
from src.geometry import SurfaceNormal
from src.iiwa import (
    Q_HOME,
    Pusher,
    build_iiwa_pusher_plant,
    clear_sites,
    make_push_trajectory,
)
from src.mpc import GoalTolerance, MpcController, PoseCostWeights, goal_error
from src.object import Object, Rod, Table
from src.observation import PushObservation
from src.params import Eta, GridSpec, make_grid
from src.recording import NoReachablePush, TruthSim
from src.scene import (
    Sim,
    build_diagram,
    place_object,
    show_object_goal,
    show_push_sites,
)


def parse_args() -> argparse.Namespace:
    shared = argparse.ArgumentParser(add_help=False)
    add_shared_options(shared)

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    calibrate_parser = commands.add_parser(
        "calibrate",
        parents=[shared],
        help="calibrate the estimator's prediction-error covariance R",
    )
    calibrate_parser.add_argument(
        "--samples", type=int, default=48, help="truth pushes"
    )
    calibrate_parser.add_argument("--seed", type=int, default=0, help="PRNG seed")

    simulate_parser = commands.add_parser(
        "simulate", parents=[shared], help="run the arm and block simulation"
    )
    simulate_parser.add_argument(
        "--headless", action="store_true", help="Run without visualization"
    )
    simulate_parser.add_argument("--seed", type=int, default=538, help="PRNG seed")
    simulate_parser.add_argument(
        "--mu",
        type=float,
        default=0.3,
        help="object-table friction coefficient (nonnegative)",
    )
    simulate_parser.add_argument(
        "--rod-x",
        type=float,
        default=0.05,
        help="rod x in the object body frame (m)",
    )
    simulate_parser.add_argument(
        "--rod-y",
        type=float,
        default=0,
        help="rod y in the object body frame (m)",
    )
    simulate_parser.add_argument(
        "--start",
        type=float,
        nargs=3,
        metavar=("X", "Y", "YAW"),
        default=None,
        help="object start pose (m, m, deg), instead of the seed's random one",
    )
    simulate_parser.add_argument(
        "--goal",
        type=float,
        nargs=3,
        metavar=("X", "Y", "YAW"),
        default=None,
        help="object goal pose (m, m, deg), instead of the seed's random one",
    )
    simulate_parser.add_argument(
        "--estimate",
        action="store_true",
        help="run random pushes and estimate (mu, rod x, rod y) after each",
    )
    simulate_parser.add_argument(
        "--pushes", type=int, default=5, help="pushes to run with --estimate"
    )
    simulate_parser.add_argument(
        "--mpc",
        action="store_true",
        help="push the object to its goal with MPC, estimating eta along the way",
    )
    simulate_parser.add_argument(
        "--probe-pushes",
        type=int,
        default=0,
        help="random probe pushes before MPC with --mpc (0: no probing)",
    )
    simulate_parser.add_argument(
        "--max-pushes",
        type=int,
        default=20,
        help="push budget per --mpc episode, probes included",
    )
    simulate_parser.add_argument(
        "--plan-with-true-eta",
        action="store_true",
        help="with --mpc, plan with the true eta and skip estimation",
    )
    # 0.07 is half the T's 14 cm extent, so with --yaw-weight 1 a 7 cm position
    # error costs as much as a 1 rad yaw error.
    simulate_parser.add_argument(
        "--cost-length",
        type=float,
        default=0.07,
        help="MPC pose cost's length scale l (m)",
    )
    simulate_parser.add_argument(
        "--yaw-weight",
        type=float,
        default=1.0,
        help="MPC pose cost's yaw weight lambda_phi (1/rad^2)",
    )
    simulate_parser.add_argument(
        "--time-weight",
        type=float,
        default=0.005,
        help="MPC action cost per second of maneuver time",
    )
    simulate_parser.add_argument(
        "--position-tolerance",
        type=float,
        default=0.01,
        help="an episode succeeds within this of the goal position (m)",
    )
    simulate_parser.add_argument(
        "--yaw-tolerance",
        type=float,
        default=10.0,
        help="an episode succeeds within this of the goal yaw (deg)",
    )
    # 3 has heavy tails: chosen by an offline replay (see src/estimator.py).
    simulate_parser.add_argument(
        "--nu",
        type=float,
        default=3.0,
        help="estimator Student-t degrees of freedom (0: Gaussian likelihood)",
    )
    args = parser.parse_args()

    finish_shared_options(args)
    if args.command == "simulate":
        if args.mu < 0.0:
            raise SystemExit("--mu must be nonnegative")
        if args.pushes < 1:
            raise SystemExit("--pushes must be at least 1")
        if args.probe_pushes < 0:
            raise SystemExit("--probe-pushes must be nonnegative")
        if args.max_pushes < 1:
            raise SystemExit("--max-pushes must be at least 1")
        if args.cost_length <= 0.0:
            raise SystemExit("--cost-length must be positive")
        if args.yaw_weight < 0.0:
            raise SystemExit("--yaw-weight must be nonnegative")
        if args.time_weight < 0.0:
            raise SystemExit("--time-weight must be nonnegative")
        if args.position_tolerance <= 0.0:
            raise SystemExit("--position-tolerance must be positive")
        if args.yaw_tolerance <= 0.0:
            raise SystemExit("--yaw-tolerance must be positive")
        if args.nu < 0.0:
            raise SystemExit("--nu must be nonnegative")

    return args


def pose_from_degrees(x: float, y: float, yaw_degrees: float) -> Pose:
    """Get a planar pose (x, y, yaw) with the yaw converted to radians."""
    return x, y, float(np.deg2rad(yaw_degrees))


def demo_action(obj: Object, push_sites: list[SurfaceNormal]) -> Action:
    """Pick the demo's push: the face nearest the arm, at a slow speed for a
    short contact time."""
    R_WO = obj.x_wo().rotation()
    site = max(push_sites, key=lambda s: (R_WO @ s.normal)[0])
    return Action(contact=site, velocity=0.05, time=0.4)


def format_push_report(
    k: int,
    obs: PushObservation,
    update_seconds: float,
    belief: GridBayesFilter,
    eta_true: Eta,
) -> str:
    """Render push `k`'s outcome and what the filter now believes."""
    mu, bx, by = belief.map_eta().as_array()
    d_mu, d_bx, d_by = belief.estimation_error(eta_true)
    return (
        f"push {k}: v={obs.action.velocity:.2f} m/s for "
        f"{obs.action.time:.1f} s, update {update_seconds:.1f} s, "
        f"entropy {belief.entropy():.3f}, max p {belief.probs.max():.3f}\n"
        f"  MAP mu={mu:.3f} b=({bx:+.4f}, {by:+.4f}) m; error "
        f"mu {d_mu:+.3f}, b ({d_bx * 100:+.2f}, {d_by * 100:+.2f}) cm"
    )


def run_estimation(
    belief: GridBayesFilter,
    eta_true: Eta,
    pushes: int,
    truth_sim: TruthSim,
    meshcat: Meshcat | None,
    rng: np.random.Generator,
    obj: Object,
    push_sites: list[SurfaceNormal],
) -> None:
    """Update `belief` after each of `pushes` random library pushes, run one
    after another in `truth_sim`, moving `obj` to where each push leaves it."""
    print(
        f"estimating eta = (mu, rod x, rod y), true {eta_true}, over "
        f"{len(belief.grid)} hypotheses"
    )
    print(f"push 0: entropy {belief.entropy():.3f}")
    for k in range(1, pushes + 1):
        if meshcat is not None:
            show_object_goal(meshcat, obj)
            show_push_sites(meshcat, obj, push_sites)
        try:
            obs = truth_sim.record_random_push(rng, obj, push_sites)
        except NoReachablePush as unreachable:
            # Random pushes can walk the block out of reach; the belief so far
            # is still good.
            print(f"stopped after {k - 1} pushes: {unreachable}")
            return
        started = time.time()
        belief.update(obs)
        print(format_push_report(k, obs, time.time() - started, belief, eta_true))
        obj.pose = obs.final_pose()


def format_placement_start(eta_true: Eta, obj: Object, candidates: int) -> str:
    """Render what a placement episode starts from."""
    position_error, yaw_error = goal_error(obj.pose, obj.goal)
    return (
        f"placing the object, true eta {eta_true}: start "
        f"{np.array2string(np.array(obj.pose), precision=3)}, goal "
        f"{np.array2string(np.array(obj.goal), precision=3)} "
        f"({position_error * 100:.1f} cm, {np.rad2deg(yaw_error):.1f} deg away); "
        f"{candidates} candidate pushes"
    )


def format_episode_result(result: EpisodeResult) -> str:
    """Render how a placement episode ended."""
    return (
        f"{'success' if result.success else 'failure'} ({result.stop_reason}) "
        f"after {result.probe_pushes} probe + {result.mpc_pushes} MPC pushes: "
        f"goal error {result.position_error * 100:.1f} cm, "
        f"{np.rad2deg(result.yaw_error):.1f} deg; {result.sim_time:.1f} s "
        f"simulated, planning {result.planning_time:.0f} s, estimation "
        f"{result.estimation_time:.0f} s"
    )


def format_map_error(belief: GridBayesFilter, eta_true: Eta) -> str:
    """Render `belief`'s final MAP error against `eta_true`."""
    d_mu, d_bx, d_by = belief.estimation_error(eta_true)
    return (
        f"final MAP error mu {d_mu:+.3f}, b ({d_bx * 100:+.2f}, {d_by * 100:+.2f}) cm"
    )


def simulate(args: argparse.Namespace) -> None:
    sim = Sim(time_step=args.time_step, headless=args.headless)
    rng = np.random.default_rng(seed=args.seed)

    obj = Object(
        rng,
        rod=Rod(x=args.rod_x, y=args.rod_y, mass=args.rod_mass),
        friction=args.mu,
    )
    if args.start is not None:
        obj.pose = pose_from_degrees(*args.start)
    if args.goal is not None:
        obj.goal = pose_from_degrees(*args.goal)
    pusher = Pusher()

    if args.mpc:
        eta_true = Eta(args.mu, args.rod_x, args.rod_y)
        push_sites = clear_sites(
            obj.parts,
            obj.surface_normals(args.push_density),
            pusher.radius,
            args.site_clearance,
        )
        actions = push_library(push_sites)
        print(format_placement_start(eta_true, obj, len(actions)))
        with ParallelPushModel(args.workers, push_model_config(args)) as model:
            belief = (
                None
                if args.plan_with_true_eta
                else GridBayesFilter(
                    make_grid(GridSpec(), obj.parts),
                    model,
                    CALIBRATED_R,
                    args.nu or None,
                )
            )
            result = run_episode(
                rng=rng,
                obj=obj,
                truth_sim=TruthSim(
                    obj,
                    pusher,
                    truth_sim_config(args),
                    meshcat=sim.meshcat,
                    realtime_rate=0.0 if sim.headless else 1.0,
                ),
                meshcat=sim.meshcat,
                controller=MpcController(
                    model,
                    actions,
                    PoseCostWeights(args.cost_length, args.yaw_weight),
                    args.time_weight,
                ),
                belief=belief,
                plan_eta=belief.map_eta if belief is not None else lambda: eta_true,
                probe_sites=push_sites,
                probe_pushes=args.probe_pushes,
                max_pushes=args.max_pushes,
                tolerance=GoalTolerance(
                    position=args.position_tolerance,
                    yaw=float(np.deg2rad(args.yaw_tolerance)),
                ),
            )
        print(format_episode_result(result))
        if belief is not None:
            print(format_map_error(belief, eta_true))
        if not sim.headless:
            input("Press Enter to quit.")
        return

    if args.estimate:
        grid = make_grid(GridSpec(), obj.parts)
        with ParallelPushModel(args.workers, push_model_config(args)) as model:
            run_estimation(
                belief=GridBayesFilter(grid, model, CALIBRATED_R, args.nu or None),
                eta_true=Eta(args.mu, args.rod_x, args.rod_y),
                pushes=args.pushes,
                truth_sim=TruthSim(
                    obj,
                    pusher,
                    truth_sim_config(args),
                    meshcat=sim.meshcat,
                    realtime_rate=0.0 if sim.headless else 1.0,
                ),
                meshcat=sim.meshcat,
                rng=rng,
                obj=obj,
                push_sites=clear_sites(
                    obj.parts,
                    obj.surface_normals(args.push_density),
                    pusher.radius,
                    args.site_clearance,
                ),
            )
        if not sim.headless:
            input("Press Enter to quit.")
        return

    push_sites = obj.surface_normals(args.push_density)
    action = demo_action(obj, push_sites)
    q_traj = make_push_trajectory(
        build_iiwa_pusher_plant(pusher), pusher, obj, action, Q_HOME
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


def main(args: argparse.Namespace) -> None:
    if args.command == "calibrate":
        calibrate(
            samples=args.samples,
            seed=args.seed,
            workers=args.workers,
            setup=make_sample_setup(
                rod_mass=args.rod_mass,
                push_density=args.push_density,
                site_clearance=args.site_clearance,
                truth=truth_sim_config(args),
                model=push_model_config(args),
            ),
        )
    else:
        simulate(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
