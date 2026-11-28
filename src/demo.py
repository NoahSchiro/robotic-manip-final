"""The interactive `simulate` command: one push in Meshcat, a run of
estimation pushes, or a placement episode, on the seed's episode conditions
unless flags pin them."""

import argparse

import numpy as np
from pydrake.all import Simulator
from pydrake.geometry import Meshcat
from sb3_contrib import MaskablePPO

from src.cli import (
    push_model_config,
    truth_sim_config,
)
from src.control.episode import EpisodeResult, run_episode
from src.control.mpc import GoalTolerance, MpcController, PoseCostWeights, goal_error
from src.control.probing import ProbingState, PushRecord, random_selector, run_probing
from src.estimation.belief_view import BeliefDashboard, show_belief
from src.estimation.dynamics import DynamicsModel, ParallelPushModel, Pose
from src.estimation.estimator import (
    CALIBRATED_R,
    CALIBRATED_R_SURROGATE,
    GridBayesFilter,
)
from src.estimation.params import Eta, GridSpec, make_grid
from src.experiment import (
    episode_conditions,
    episode_goal,
    episode_object,
    lattice,
)
from src.policy.deploy import policy_selector
from src.policy.env import ProbingEnvConfig, make_probing_env_config
from src.surrogate.model import SurrogatePredictor, load_ensemble
from src.world.action import Action
from src.world.iiwa import (
    Q_HOME,
    Pusher,
    build_iiwa_pusher_plant,
    make_push_trajectory,
)
from src.world.object import Object, Table
from src.world.observation import PushObservation
from src.world.push_space import PushSpace, make_push_space
from src.world.recording import TruthSim
from src.world.scene import (
    Sim,
    build_diagram,
    place_object,
    show_object_goal,
)


def pose_from_degrees(x: float, y: float, yaw_degrees: float) -> Pose:
    """Get a planar pose (x, y, yaw) with the yaw converted to radians."""
    return x, y, float(np.deg2rad(yaw_degrees))


def demo_action(obj: Object, space: PushSpace) -> Action:
    """Pick the demo's push: the face whose inward normal points most along +x in
    the world, at its midpoint, at a slow speed for a short contact time."""
    R_WO = obj.x_wo().rotation()
    face = max(
        range(len(space.faces)),
        key=lambda i: (R_WO @ space.faces[i].normal)[0],
    )
    return Action(contact=space.contact(face, 0.5), velocity=0.05, time=0.4)


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
    dashboard: BeliefDashboard | None,
    obj: Object,
    action_rng: np.random.Generator,
    policy: MaskablePPO | None = None,
    policy_config: ProbingEnvConfig | None = None,
) -> None:
    """Update `belief` after each of up to `pushes` probing pushes, run one after
    another in `truth_sim`, moving `obj` to where each push leaves it.  Without a
    `policy`, the pushes are random draws from the push space using `action_rng`;
    with one, the PPO policy chooses them from the current belief."""
    if policy is None:
        selector = random_selector(make_push_space(obj.parts, obj.height), action_rng)
    else:
        assert policy_config is not None
        selector = policy_selector(policy, policy_config)

    print(
        f"estimating eta = (mu, rod x, rod y), true {eta_true}, over "
        f"{len(belief.grid)} hypotheses"
    )
    print(f"push 0: entropy {belief.entropy():.3f}")
    show_belief(0, belief, obj, meshcat, dashboard)

    def report(state: ProbingState, record: PushRecord) -> None:
        print(
            format_push_report(
                state.pushes,
                record.observation,
                record.update_seconds,
                belief,
                eta_true,
            )
        )
        show_belief(state.pushes, belief, state.obj, meshcat, dashboard)

    state = ProbingState(truth_sim=truth_sim, obj=obj, belief=belief)
    outcome = run_probing(state, selector, pushes, on_push=report)
    print(f"stopped after {state.pushes} pushes: {outcome.reason}")


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


def make_truth_sim(
    args: argparse.Namespace, sim: Sim, obj: Object, pusher: Pusher
) -> TruthSim:
    """The episode's truth sim, paced in real time when visualized."""
    return TruthSim(
        obj,
        pusher,
        truth_sim_config(args),
        meshcat=sim.meshcat,
        realtime_rate=0.0 if sim.headless else 1.0,
    )


def estimation_filter(
    args: argparse.Namespace, grid: list[Eta], drake_model: DynamicsModel
) -> GridBayesFilter:
    """The estimator's filter on --predictor's model, with that model's
    calibrated R."""
    if args.predictor == "surrogate":
        surrogate = SurrogatePredictor(load_ensemble(args.ensemble))
        return GridBayesFilter(grid, surrogate, CALIBRATED_R_SURROGATE, args.nu or None)
    return GridBayesFilter(grid, drake_model, CALIBRATED_R, args.nu or None)


def simulate(args: argparse.Namespace) -> None:
    # Fresh random seed unless --seed is given; printed so the run can be reproduced.
    seed = args.seed
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**32))
    print(f"PRNG seed: {seed} (rerun with --seed {seed} to reproduce)")

    # eta is sampled from the seed unless --mu, --rod-x or --rod-y pins it.
    sampled, start_pose, action_rng = episode_conditions(seed)
    eta = Eta(
        mu=args.mu if args.mu is not None else sampled.mu,
        bx=args.rod_x if args.rod_x is not None else sampled.bx,
        by=args.rod_y if args.rod_y is not None else sampled.by,
    )
    print(
        f"true parameters: mu={eta.mu:.3f}, rod=({eta.bx:+.4f}, {eta.by:+.4f}) m "
        "(what the estimator must find)"
    )

    sim = Sim(time_step=args.time_step, headless=args.headless)

    obj = episode_object(eta, start_pose, args.rod_mass)
    if args.start is not None:
        obj.pose = pose_from_degrees(*args.start)
    obj.goal = (
        pose_from_degrees(*args.goal)
        if args.goal is not None
        else episode_goal(seed, start_pose)
    )
    pusher = Pusher()

    if args.mpc:
        eta_true = eta
        space = make_push_space(obj.parts, obj.height)
        actions = lattice(space)
        print(format_placement_start(eta_true, obj, len(actions)))
        grid = make_grid(GridSpec(), obj.parts)
        truth_sim = make_truth_sim(args, sim, obj, pusher)
        with ParallelPushModel(args.workers, push_model_config(args)) as model:
            belief = (
                None
                if args.plan_with_true_eta
                else estimation_filter(args, grid, model)
            )
            result = run_episode(
                obj=obj,
                truth_sim=truth_sim,
                meshcat=sim.meshcat,
                dashboard=(
                    None
                    if sim.headless or belief is None
                    else BeliefDashboard(grid, obj.parts, eta_true)
                ),
                controller=MpcController(
                    model,
                    actions,
                    PoseCostWeights(args.cost_length, args.yaw_weight),
                    args.time_weight,
                ),
                belief=belief,
                plan_eta=belief.map_eta if belief is not None else lambda: eta_true,
                probe_selector=random_selector(space, action_rng),
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
        policy = None
        policy_config = None
        if args.policy == "ppo":
            policy = MaskablePPO.load(args.model)
            policy_config = make_probing_env_config(
                args.ensemble, family_path=args.path_family
            )
        truth_sim = make_truth_sim(args, sim, obj, pusher)
        with ParallelPushModel(args.workers, push_model_config(args)) as model:
            run_estimation(
                belief=estimation_filter(args, grid, model),
                eta_true=eta,
                pushes=args.pushes,
                truth_sim=truth_sim,
                meshcat=sim.meshcat,
                dashboard=(
                    None if sim.headless else BeliefDashboard(grid, obj.parts, eta)
                ),
                obj=obj,
                action_rng=action_rng,
                policy=policy,
                policy_config=policy_config,
            )
        if not sim.headless:
            input("Press Enter to quit.")
        return

    space = make_push_space(obj.parts, obj.height)
    action = demo_action(obj, space)
    q_traj = make_push_trajectory(
        build_iiwa_pusher_plant(pusher), pusher, obj, action, Q_HOME
    )
    diagram, plant = build_diagram(
        args.time_step, obj, Table(eta.mu), pusher, meshcat=sim.meshcat, q_ref=q_traj
    )

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # Rest the block on the table in front of the arm.  Its body frame is the
    # footprint's bottom center.
    object_body = place_object(plant, plant_context, obj)
    X_WO = obj.x_wo()
    print(f"object-table friction mu = {eta.mu:.3f}")
    if obj.rod is not None:
        print(f"rod welded at ({obj.rod.x}, {obj.rod.y}) m, mass {obj.rod.mass} kg")
    print(
        f"object com (body frame) = "
        f"{np.array2string(obj.com_in_body_frame(), precision=3)}"
    )
    print(
        f"push space: {len(space.faces)} faces, "
        f"{space.usable_spans().sum() * 100:.1f} cm of usable outline"
    )
    print(
        f"demo push: {action.velocity:.2f} m/s along the contact's inward "
        f"normal for {action.time:.2f} s in contact"
    )
    print(f"object starts at {np.array2string(X_WO.translation(), precision=3)}")
    print(f"object goal is {np.array2string(obj.x_wg().translation(), precision=3)}")
    if sim.meshcat is not None:
        show_object_goal(sim.meshcat, obj)

    simulator = Simulator(diagram, context)
    if not sim.headless:
        simulator.set_target_realtime_rate(1.0)
    end_time = q_traj.end_time()
    simulator.AdvanceTo(0.05 if sim.headless else end_time)

    X_WO_final = plant.EvalBodyPoseInWorld(plant_context, object_body)
    print(f"object ends at   {np.array2string(X_WO_final.translation(), precision=3)}")

    if not sim.headless:
        input("Press Enter to quit.")
