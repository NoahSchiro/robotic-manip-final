"""Entry point."""

import argparse
import time

import numpy as np
from pydrake.all import Simulator

from src.action import Action, push_library
from src.cli import (
    add_shared_options,
    finish_shared_options,
    push_model_config,
    truth_sim_config,
)
from src.dynamics import ParallelPushModel
from src.episode import MAX_PUSHES, run_episode
from src.estimator import CALIBRATED_R, GridBayesFilter
from src.geometry import SurfaceNormal
from src.iiwa import Pusher, build_iiwa_pusher_plant, make_push_trajectory
from src.mpc import MpcController, clear_sites, goal_error
from src.object import Object, Rod, Table
from src.observation import PushObservation
from src.params import Eta, GridSpec, make_grid
from src.recording import TruthSimConfig, record_random_push
from src.scene import (
    Sim,
    build_diagram,
    place_object,
    show_object_goal,
    show_push_sites,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_shared_options(parser)
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
        "--start",
        type=float,
        nargs=3,
        metavar=("X", "Y", "YAW"),
        default=None,
        help="object start pose (m, m, deg), instead of the seed's random one",
    )
    parser.add_argument(
        "--goal",
        type=float,
        nargs=3,
        metavar=("X", "Y", "YAW"),
        default=None,
        help="object goal pose (m, m, deg), instead of the seed's random one",
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
        "--mpc",
        action="store_true",
        help="push the object to its goal with MPC, estimating eta along the way",
    )
    parser.add_argument(
        "--probe-pushes",
        type=int,
        default=0,
        help="random probe pushes before MPC with --mpc (0: no probing)",
    )
    parser.add_argument(
        "--max-pushes",
        type=int,
        default=MAX_PUSHES,
        help="push budget per --mpc episode, probes included",
    )
    parser.add_argument(
        "--plan-with-true-eta",
        action="store_true",
        help="with --mpc, plan with the true eta and skip estimation",
    )
    # 3 has heavy tails: chosen by an offline replay (see src/estimator.py).
    parser.add_argument(
        "--nu",
        type=float,
        default=3.0,
        help="estimator Student-t degrees of freedom (0: Gaussian likelihood)",
    )
    args = parser.parse_args()

    finish_shared_options(args)
    if args.mu < 0.0:
        raise SystemExit("--mu must be nonnegative")
    if args.pushes < 1:
        raise SystemExit("--pushes must be at least 1")
    if args.probe_pushes < 0:
        raise SystemExit("--probe-pushes must be nonnegative")
    if args.max_pushes < 1:
        raise SystemExit("--max-pushes must be at least 1")
    if args.nu < 0.0:
        raise SystemExit("--nu must be nonnegative")

    return args


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
    truth: TruthSimConfig,
    sim: Sim,
    rng: np.random.Generator,
    obj: Object,
    pusher: Pusher,
    push_sites: list[SurfaceNormal],
) -> None:
    """Update `belief` after each of `pushes` random library pushes, moving
    `obj` to where each push leaves it so the next starts from there."""
    print(
        f"estimating eta = (mu, rod x, rod y), true {eta_true}, over "
        f"{len(belief.grid)} hypotheses"
    )
    print(f"push 0: entropy {belief.entropy():.3f}")
    for k in range(1, pushes + 1):
        if sim.meshcat is not None:
            show_object_goal(sim.meshcat, obj)
            show_push_sites(sim.meshcat, obj, push_sites)
        obs = record_random_push(
            rng,
            obj,
            pusher,
            push_sites,
            truth,
            meshcat=sim.meshcat,
            realtime_rate=0.0 if sim.headless else 1.0,
        )
        started = time.time()
        belief.update(obs)
        print(format_push_report(k, obs, time.time() - started, belief, eta_true))
        obj.pose = obs.final_pose()


def run_mpc(
    args: argparse.Namespace,
    sim: Sim,
    rng: np.random.Generator,
    obj: Object,
    pusher: Pusher,
) -> None:
    """Run one placement episode: optional random probing, then MPC pushes
    toward the goal, with the grid Bayes filter updated after every push
    (unless planning with the true eta)."""
    eta_true = Eta(args.mu, args.rod_x, args.rod_y)
    push_sites = obj.surface_normals(args.push_density)
    actions = push_library(clear_sites(obj, push_sites, pusher))
    position_error, yaw_error = goal_error(obj)
    print(
        f"placing the object, true eta {eta_true}: start "
        f"{np.array2string(np.array(obj.pose), precision=3)}, goal "
        f"{np.array2string(np.array(obj.goal), precision=3)} "
        f"({position_error * 100:.1f} cm, {np.rad2deg(yaw_error):.1f} deg away); "
        f"{len(actions)} candidate pushes"
    )

    with ParallelPushModel(args.workers, push_model_config(args)) as model:
        controller = MpcController(model, actions)
        belief = None
        if not args.plan_with_true_eta:
            grid = make_grid(GridSpec(), obj.parts)
            belief = GridBayesFilter(grid, model, CALIBRATED_R, args.nu or None)
        result = run_episode(
            rng,
            obj,
            pusher,
            controller,
            belief,
            push_sites,
            truth_sim_config(args),
            probe_pushes=args.probe_pushes,
            max_pushes=args.max_pushes,
            known_eta=eta_true if args.plan_with_true_eta else None,
            meshcat=sim.meshcat,
            realtime_rate=0.0 if sim.headless else 1.0,
        )
    print(
        f"{'success' if result.success else 'failure'} ({result.stop_reason}) "
        f"after {result.probe_pushes} probe + {result.mpc_pushes} MPC pushes: "
        f"goal error {result.position_error * 100:.1f} cm, "
        f"{np.rad2deg(result.yaw_error):.1f} deg; {result.sim_time:.1f} s "
        f"simulated, planning {result.planning_time:.0f} s, estimation "
        f"{result.estimation_time:.0f} s"
    )
    if belief is not None:
        d_mu, d_bx, d_by = belief.estimation_error(eta_true)
        print(
            f"final MAP error mu {d_mu:+.3f}, b ({d_bx * 100:+.2f}, "
            f"{d_by * 100:+.2f}) cm"
        )


def main(args: argparse.Namespace) -> None:
    sim = Sim(time_step=args.time_step, headless=args.headless)
    rng = np.random.default_rng(seed=args.seed)

    obj = Object(
        rng,
        rod=Rod(x=args.rod_x, y=args.rod_y, mass=args.rod_mass),
        friction=args.mu,
    )
    if args.start is not None:
        x, y, yaw = args.start
        obj.pose = (x, y, float(np.deg2rad(yaw)))
    if args.goal is not None:
        x, y, yaw = args.goal
        obj.goal = (x, y, float(np.deg2rad(yaw)))
    pusher = Pusher()

    if args.mpc:
        run_mpc(args, sim, rng, obj, pusher)
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
                truth=truth_sim_config(args),
                sim=sim,
                rng=rng,
                obj=obj,
                pusher=pusher,
                push_sites=obj.surface_normals(args.push_density),
            )
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
