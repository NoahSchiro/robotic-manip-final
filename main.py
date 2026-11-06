"""Entry point."""

import argparse
import time
from pathlib import Path

import numpy as np
from pydrake.all import Simulator
from pydrake.geometry import Meshcat
from sb3_contrib import MaskablePPO

from src.action import Action
from src.calibration import calibrate, make_sample_setup
from src.cli import (
    add_shared_options,
    finish_shared_options,
    push_model_config,
    truth_sim_config,
)
from src.dataset_file import save_dataset
from src.dynamics import ParallelPushModel
from src.env import ProbingEnvConfig, make_probing_env_config, select_push_action
from src.estimator import CALIBRATED_R, GridBayesFilter
from src.evaluation import run_evaluation
from src.geometry import SurfaceNormal
from src.iiwa import (
    Q_HOME,
    Pusher,
    build_iiwa_pusher_plant,
    clear_sites,
    make_push_trajectory,
)
from src.object import Object, Rod, Table
from src.observation import PushObservation
from src.params import Eta, GridSpec, make_grid, sample_eta
from src.path_family import (
    calibrate_path_family,
    load_path_family,
    save_path_family,
)
from src.recording import NoReachablePush, TruthSim
from src.replay import run_replay
from src.scene import (
    Sim,
    build_diagram,
    place_object,
    show_object_goal,
)
from src.surrogate import SAMPLE_PERIOD
from src.surrogate_data import (
    check_finite,
    coverage_summary,
    generate_surrogate_data,
    make_surrogate_setup,
    mark_validation_split,
)
from src.train_ppo import train_ppo
from src.train_surrogate import train_ensemble
from src.trials import make_record_setup, record_trials


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
    calibrate_parser.add_argument(
        "--predictor",
        choices=("drake", "surrogate"),
        default="drake",
        help="prediction model the calibration measures errors of",
    )
    calibrate_parser.add_argument(
        "--ensemble",
        type=Path,
        default=Path("data/surrogate_ensemble.npz"),
        help="trained surrogate ensemble (--predictor surrogate)",
    )

    record_parser = commands.add_parser(
        "record-trials",
        parents=[shared],
        help="record random-push trials for the estimator's offline replay",
    )
    record_parser.add_argument(
        "--trials", type=int, default=19, help="trials to record"
    )
    record_parser.add_argument("--pushes", type=int, default=5, help="pushes per trial")
    record_parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    record_parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/replay_trials.npz"),
        help="dataset output path",
    )

    replay_parser = commands.add_parser(
        "replay",
        parents=[shared],
        help="score the estimator against recorded trials",
    )
    replay_parser.add_argument(
        "--trials-file",
        type=Path,
        default=Path("data/replay_trials.npz"),
        help="recorded trial dataset",
    )
    replay_parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/replay_report.json"),
        help="report output path",
    )
    replay_parser.add_argument(
        "--predictor",
        choices=("drake", "surrogate"),
        default="drake",
        help="prediction model the filter uses",
    )
    replay_parser.add_argument(
        "--ensemble",
        type=Path,
        default=Path("data/surrogate_ensemble.npz"),
        help="trained surrogate ensemble (--predictor surrogate)",
    )

    surrogate_parser = commands.add_parser(
        "gen-surrogate-data",
        parents=[shared],
        help="generate the surrogate's training dataset (Drake-predicted pushes)",
    )
    surrogate_parser.add_argument(
        "--samples", type=int, default=300000, help="samples to generate"
    )
    surrogate_parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    surrogate_parser.add_argument(
        "--path-family",
        type=Path,
        default=Path("data/path_family.npz"),
        help="calibrated executed-path deviation family",
    )
    surrogate_parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/surrogate_dataset.npz"),
        help="dataset output path",
    )

    train_parser = commands.add_parser(
        "train-surrogate",
        parents=[shared],
        help="train the surrogate's MLP ensemble on the generated dataset",
    )
    train_parser.add_argument("--members", type=int, default=5, help="ensemble members")
    train_parser.add_argument(
        "--hidden", type=int, default=768, help="hidden layer width"
    )
    train_parser.add_argument(
        "--layers", type=int, default=4, help="hidden layer count"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=400, help="training epochs per member"
    )
    train_parser.add_argument(
        "--patience", type=int, default=40, help="early-stopping patience"
    )
    train_parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    train_parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/surrogate_dataset.npz"),
        help="training dataset from gen-surrogate-data",
    )
    train_parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/surrogate_ensemble.npz"),
        help="ensemble output path",
    )

    ppo_parser = commands.add_parser(
        "train-ppo",
        parents=[shared],
        help="train the probing policy with MaskablePPO in the surrogate world",
    )
    ppo_parser.add_argument(
        "--timesteps", type=int, default=3000000, help="total env steps"
    )
    ppo_parser.add_argument("--n-envs", type=int, default=12, help="vector envs")
    ppo_parser.add_argument(
        "--eval-every", type=int, default=100000, help="steps between paired evals"
    )
    ppo_parser.add_argument(
        "--eval-episodes", type=int, default=12, help="episodes per paired eval"
    )
    ppo_parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    ppo_parser.add_argument(
        "--ensemble",
        type=Path,
        default=Path("data/surrogate_ensemble.npz"),
        help="trained surrogate ensemble",
    )
    ppo_parser.add_argument(
        "--path-family",
        type=Path,
        default=Path("data/path_family.npz"),
        help="calibrated executed-path deviation family",
    )
    ppo_parser.add_argument(
        "--out", type=Path, default=Path("data/ppo"), help="output directory"
    )

    family_parser = commands.add_parser(
        "calibrate-path-family",
        parents=[shared],
        help="calibrate the executed-path deviation family from recorded trials",
    )
    family_parser.add_argument(
        "--trials",
        type=Path,
        default=Path("data/replay_trials.npz"),
        help="recorded-trials dataset to calibrate from",
    )
    family_parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/path_family.npz"),
        help="family output path",
    )

    demo_parser = commands.add_parser(
        "demo",
        parents=[shared],
        help="run the trained policy on fresh episodes with narration",
    )
    demo_parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/ppo_run5/ppo_600000.zip"),
        help="trained probing policy",
    )
    demo_parser.add_argument("--episodes", type=int, default=3, help="episodes to show")
    demo_parser.add_argument("--seed", type=int, default=12345, help="PRNG seed")
    demo_parser.add_argument(
        "--drake", action="store_true", help="run in the Drake world (real arm, slow)"
    )
    demo_parser.add_argument(
        "--ensemble",
        type=Path,
        default=Path("data/surrogate_ensemble.npz"),
        help="trained surrogate ensemble",
    )

    report_parser = commands.add_parser(
        "evaluation-report",
        parents=[shared],
        help="render the evaluation report: the paired table, plots and the behavioral summary",
    )
    report_parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/evaluation_report.json"),
        help="evaluation report json from the evaluate command",
    )
    report_parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/ppo_run5/ppo_600000.zip"),
        help="trained probing policy for the behavioral probe",
    )
    report_parser.add_argument(
        "--ensemble",
        type=Path,
        default=Path("data/surrogate_ensemble.npz"),
        help="trained surrogate ensemble",
    )
    report_parser.add_argument(
        "--path-family",
        type=Path,
        default=Path("data/path_family.npz"),
        help="calibrated executed-path deviation family",
    )
    report_parser.add_argument(
        "--out", type=Path, default=Path("data/evaluation"), help="output directory"
    )

    eval_parser = commands.add_parser(
        "evaluate",
        parents=[shared],
        help=(
            "evaluate the learned probing policy against the random-probing "
            "baseline — both run the real arm through the identical Drake "
            "pipeline"
        ),
    )
    eval_parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/ppo_run4/ppo_final.zip"),
        help="trained probing policy",
    )
    eval_parser.add_argument(
        "--ensemble",
        type=Path,
        default=Path("data/surrogate_ensemble.npz"),
        help="trained surrogate ensemble",
    )
    eval_parser.add_argument(
        "--episodes", type=int, default=24, help="evaluation episodes"
    )
    eval_parser.add_argument("--seed", type=int, default=70000, help="PRNG seed")
    eval_parser.add_argument(
        "--path-family",
        type=Path,
        default=Path("data/path_family.npz"),
        help="calibrated executed-path deviation family",
    )
    eval_parser.add_argument(
        "--predictor",
        choices=("drake", "surrogate"),
        default="drake",
        help="the filter's prediction model in the evaluation world",
    )
    eval_parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/transfer_report.json"),
        help="report output path",
    )

    simulate_parser = commands.add_parser(
        "simulate", parents=[shared], help="run the arm and block simulation"
    )
    simulate_parser.add_argument(
        "--headless", action="store_true", help="Run without visualization"
    )
    simulate_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="PRNG seed (a fresh random seed is drawn if omitted)",
    )
    simulate_parser.add_argument(
        "--policy",
        choices=("random", "ppo"),
        default="random",
        help="who selects the pushes during --estimate",
    )
    simulate_parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/ppo_run5/ppo_600000.zip"),
        help="trained probing policy (--policy ppo)",
    )
    simulate_parser.add_argument(
        "--ensemble",
        type=Path,
        default=Path("data/surrogate_ensemble.npz"),
        help="trained surrogate ensemble (--policy ppo)",
    )
    simulate_parser.add_argument(
        "--path-family",
        type=Path,
        default=Path("data/path_family.npz"),
        help="calibrated executed-path deviation family (--policy ppo)",
    )
    simulate_parser.add_argument(
        "--mu",
        type=float,
        default=None,
        help="object-table friction coefficient (sampled fresh if omitted)",
    )
    simulate_parser.add_argument(
        "--rod-x",
        type=float,
        default=None,
        help="rod x in the object body frame (sampled fresh if omitted)",
    )
    simulate_parser.add_argument(
        "--rod-y",
        type=float,
        default=None,
        help="rod y in the object body frame (sampled fresh if omitted)",
    )
    simulate_parser.add_argument(
        "--estimate",
        action="store_true",
        help="run random pushes and estimate (mu, rod x, rod y) after each",
    )
    simulate_parser.add_argument(
        "--pushes", type=int, default=5, help="pushes to run with --estimate"
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
        if args.mu is not None and args.mu < 0.0:
            raise SystemExit("--mu must be nonnegative")
        if args.pushes < 1:
            raise SystemExit("--pushes must be at least 1")
        if args.nu < 0.0:
            raise SystemExit("--nu must be nonnegative")
        if args.policy == "ppo" and not args.estimate:
            raise SystemExit("--policy ppo requires --estimate")
    if args.command == "record-trials" and (args.trials < 1 or args.pushes < 1):
        raise SystemExit("--trials and --pushes must be at least 1")
    if args.command == "gen-surrogate-data" and args.samples < 1:
        raise SystemExit("--samples must be at least 1")
    if args.command == "train-surrogate" and (
        args.members < 1 or args.epochs < 1 or args.patience < 1
    ):
        raise SystemExit("--members, --epochs and --patience must be at least 1")
    if args.command == "train-ppo" and (
        args.timesteps < 1 or args.n_envs < 1 or args.eval_episodes < 1
    ):
        raise SystemExit("--timesteps, --n-envs and --eval-episodes must be at least 1")
    if args.command == "evaluate" and args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")

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
    truth_sim: TruthSim,
    meshcat: Meshcat | None,
    rng: np.random.Generator,
    obj: Object,
    push_sites: list[SurfaceNormal],
    policy: MaskablePPO | None = None,
    policy_config: ProbingEnvConfig | None = None,
) -> None:
    """Update `belief` after each of `pushes` library pushes, run one after
    another in `truth_sim`, moving `obj` to where each push leaves it.  The
    pushes come from the random prober, or — with a trained `policy` — from
    the PPO policy acting on the current belief."""
    print(
        f"estimating eta = (mu, rod x, rod y), true {eta_true}, over "
        f"{len(belief.grid)} hypotheses"
    )
    print(f"push 0: entropy {belief.entropy():.3f}")
    last_measurement = np.zeros(6)
    last_action = np.zeros(8)
    failed_sites: set[int] = set()
    done = 0
    while done < pushes:
        k = done + 1
        if meshcat is not None:
            show_object_goal(meshcat, obj)
        if policy is None:
            try:
                obs = truth_sim.record_random_push(rng, obj, push_sites)
            except NoReachablePush as unreachable:
                # Random pushes can walk the block out of reach; the belief so far
                # is still good.
                print(f"stopped after {done} pushes: {unreachable}")
                return
        else:
            assert policy_config is not None
            block_pose = np.array([*obj.pose[:2], obj.pose[2]])
            chosen = select_push_action(
                policy,
                belief,
                block_pose,
                last_measurement,
                last_action,
                done,
                policy_config,
                failed_sites,
            )
            if chosen is None:
                print(
                    f"stopped after {done} pushes: the policy is confident "
                    f"in its estimate"
                )
                return
            site, speed, duration = chosen
            action = Action(
                contact=SurfaceNormal(
                    position=policy_config.site_positions[site],
                    normal=policy_config.site_normals[site],
                ),
                velocity=speed,
                time=duration,
            )
            try:
                q_traj = truth_sim.plan_push(obj, action)
            except RuntimeError:
                print(f"  push {k}: site {site} unreachable by the arm, re-planning")
                failed_sites.add(site)
                continue
            obs = truth_sim.record_push(obj, action, q_traj)
        started = time.time()
        belief.update(obs)
        done += 1
        print(format_push_report(k, obs, time.time() - started, belief, eta_true))
        obj.pose = obs.final_pose()
        last_measurement = obs.measured()
        last_action = np.array(
            [
                *obs.action.contact.position,
                *obs.action.contact.normal,
                obs.action.velocity,
                obs.action.time,
            ]
        )


def simulate(args: argparse.Namespace) -> None:
    # A fresh random seed unless the user pins one; the seed is printed so
    # any run can be reproduced with --seed.
    seed = args.seed
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**32))
    print(f"PRNG seed: {seed} (rerun with --seed {seed} to reproduce)")
    rng = np.random.default_rng(seed=seed)

    # The unknown parameters are the experiment: sampled fresh from the
    # estimation grid's ranges unless the user pins them by flag.
    reference = Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, args.rod_mass))
    sampled = sample_eta(rng, GridSpec(), reference.parts)
    mu = args.mu if args.mu is not None else sampled.mu
    rod_x = args.rod_x if args.rod_x is not None else sampled.bx
    rod_y = args.rod_y if args.rod_y is not None else sampled.by
    print(
        f"true parameters: mu={mu:.3f}, rod=({rod_x:+.4f}, {rod_y:+.4f}) m "
        "(what the estimator must find)"
    )

    sim = Sim(time_step=args.time_step, headless=args.headless)

    obj = Object(
        rng,
        rod=Rod(x=rod_x, y=rod_y, mass=args.rod_mass),
        friction=mu,
    )
    pusher = Pusher()

    if args.estimate:
        grid = make_grid(GridSpec(), obj.parts)
        policy = None
        policy_config = None
        if args.policy == "ppo":
            from sb3_contrib import MaskablePPO

            policy = MaskablePPO.load(args.model)
            policy_config = make_probing_env_config(
                args.ensemble, family_path=args.path_family
            )
        with ParallelPushModel(args.workers, push_model_config(args)) as model:
            run_estimation(
                belief=GridBayesFilter(grid, model, CALIBRATED_R, args.nu or None),
                eta_true=Eta(mu, rod_x, rod_y),
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
                policy=policy,
                policy_config=policy_config,
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


def record_trials_command(args: argparse.Namespace) -> None:
    """Record the offline replay's dataset, storing the settings it ran
    under so a replay can rebuild the same setup."""
    setup = make_record_setup(
        rod_mass=args.rod_mass,
        push_density=args.push_density,
        site_clearance=args.site_clearance,
        pushes=args.pushes,
        truth=truth_sim_config(args),
    )
    dataset = record_trials(
        setup, trials=args.trials, seed=args.seed, workers=args.workers
    )
    save_dataset(
        args.out,
        dataset,
        metadata={
            "seed": args.seed,
            "trials": args.trials,
            "pushes": args.pushes,
            "rod_mass": args.rod_mass,
            "push_density": args.push_density,
            "site_clearance": args.site_clearance,
            "truth_time_step": args.time_step,
            "settle_time": args.settle_time,
            "log_period": args.log_period,
            "model_time_step": push_model_config(args).time_step,
        },
    )
    print(f"wrote {args.trials} trials to {args.out}")


def generate_surrogate_data_command(args: argparse.Namespace) -> None:
    """Generate the surrogate's training dataset: sampled (eta, start pose,
    action) tuples replayed through the Drake prediction model."""
    setup = make_surrogate_setup(
        rod_mass=args.rod_mass,
        push_density=args.push_density,
        site_clearance=args.site_clearance,
        settle_time=args.settle_time,
        family=load_path_family(args.path_family),
        model=push_model_config(args),
    )
    dataset = generate_surrogate_data(
        setup, samples=args.samples, seed=args.seed, workers=args.workers
    )
    dataset["is_val"] = mark_validation_split(dataset, setup.spec)
    check_finite(dataset)
    print(coverage_summary(dataset, setup))
    save_dataset(
        args.out,
        dataset,
        metadata={
            "seed": args.seed,
            "samples": args.samples,
            "rod_mass": args.rod_mass,
            "push_density": args.push_density,
            "site_clearance": args.site_clearance,
            "model_time_step": push_model_config(args).time_step,
            "settle_time": args.settle_time,
            "sample_period": SAMPLE_PERIOD,
        },
    )
    print(f"wrote {args.samples} samples to {args.out}")


def train_surrogate_command(args: argparse.Namespace) -> None:
    """Train the surrogate's ensemble and report its validation errors
    against the estimator's tolerated prediction error."""
    print(
        train_ensemble(
            dataset_path=args.dataset,
            out_path=args.out,
            members=args.members,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            hidden=args.hidden,
            layers=args.layers,
        )
    )


def evaluation_report_command(args: argparse.Namespace) -> str:
    """Render the evaluation report: the paired table, the plots and the
    behavioral summary, from the evaluation report json."""
    from src.evaluation_report import make_evaluation_report

    return make_evaluation_report(
        report_path=args.report,
        model_path=args.model,
        ensemble_path=args.ensemble,
        family_path=args.path_family,
        out_dir=args.out,
    )


def calibrate_path_family_command(args: argparse.Namespace) -> None:
    """Calibrate the executed-path deviation family from the recorded
    trials and save it for data generation and the environment."""
    family, residual = calibrate_path_family(args.trials)
    save_path_family(args.out, family)
    print(
        f"path family: {family.n_components} components, "
        f"rank-kept residual {residual * 1000:.2f} mm, "
        f"deviation RMS {np.sqrt((family.sigmas**2).sum() + (family.mean**2).sum()) / np.sqrt(32) * 1000:.2f} mm; "
        f"wrote {args.out}"
    )


def evaluate_command(args: argparse.Namespace) -> None:
    """Evaluate the learned policy against the random-probing baseline:
    both run the real arm through the identical Drake pipeline."""
    print(
        run_evaluation(
            model_path=args.model,
            ensemble_path=args.ensemble,
            family_path=args.path_family,
            out_path=args.out,
            episodes=args.episodes,
            seed=args.seed,
            workers=args.workers,
            predictor=args.predictor,
        )
    )


def train_ppo_command(args: argparse.Namespace) -> None:
    """Train the probing policy with MaskablePPO and report the paired
    evaluation against random probing."""
    config = make_probing_env_config(args.ensemble, family_path=args.path_family)
    print(
        train_ppo(
            config=config,
            out_dir=args.out,
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
        )
    )


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
                predictor=args.predictor,
                ensemble_path=args.ensemble,
            ),
        )
    elif args.command == "record-trials":
        record_trials_command(args)
    elif args.command == "replay":
        run_replay(
            trials_path=args.trials_file,
            workers=args.workers,
            report_path=args.report,
            predictor=args.predictor,
            ensemble_path=args.ensemble,
        )
    elif args.command == "gen-surrogate-data":
        generate_surrogate_data_command(args)
    elif args.command == "train-surrogate":
        train_surrogate_command(args)
    elif args.command == "train-ppo":
        train_ppo_command(args)
    elif args.command == "evaluation-report":
        print(evaluation_report_command(args))
    elif args.command == "calibrate-path-family":
        calibrate_path_family_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    else:
        simulate(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
