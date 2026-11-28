"""Command-line entry point.  Run `uv run python main.py <command> --help` for
the options of each subcommand."""

import argparse
import json
from pathlib import Path

import numpy as np

from src import paths
from src.cli import (
    add_shared_options,
    finish_shared_options,
    push_model_config,
    truth_sim_config,
)
from src.constants import MPC_LENGTH, MPC_TIME_WEIGHT, MPC_YAW_WEIGHT
from src.control.mpc import GoalTolerance
from src.dataset_file import save_dataset
from src.demo import simulate
from src.estimation.calibration import calibrate, make_sample_setup
from src.estimation.estimator import (
    SUBCELLS,
)
from src.estimation.replay import run_replay
from src.estimation.trials import make_record_setup, record_trials, trials_metadata
from src.evaluation.placement import run_placement
from src.evaluation.transfer import run_evaluation
from src.experiment import (
    TASKS,
    reference_object,
)
from src.policy.env import MAX_PUSHES, make_probing_env_config
from src.policy.train import train_ppo
from src.surrogate.data import (
    assign_split,
    check_finite,
    coverage_summary,
    generate_surrogate_data,
    make_surrogate_setup,
    surrogate_metadata,
)
from src.surrogate.path_family import (
    calibrate_path_family,
    deviation_rms,
    load_path_family,
    save_path_family,
)
from src.surrogate.train import train_ensemble
from src.world.iiwa import (
    transfer_height,
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
    calibrate_parser.add_argument(
        "--predictor",
        choices=("drake", "surrogate"),
        default="drake",
        help="prediction model the calibration measures errors of",
    )
    calibrate_parser.add_argument(
        "--subcells",
        type=int,
        default=SUBCELLS,
        help="belief points per axis inside each grid cell the R is for",
    )
    calibrate_parser.add_argument(
        "--ensemble",
        type=Path,
        default=paths.SURROGATE_ENSEMBLE,
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
        default=paths.REPLAY_TRIALS,
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
        default=paths.REPLAY_TRIALS,
        help="recorded trial dataset",
    )
    replay_parser.add_argument(
        "--report",
        type=Path,
        default=paths.RESULTS / "replay_report.json",
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
        default=paths.SURROGATE_ENSEMBLE,
        help="trained surrogate ensemble (--predictor surrogate)",
    )
    replay_parser.add_argument(
        "--subcells",
        type=int,
        default=SUBCELLS,
        help="belief points per axis inside each grid cell (1: grid points only)",
    )
    replay_parser.add_argument(
        "--save-predictions",
        type=Path,
        help="save every push's grid predictions here for later replays",
    )
    replay_parser.add_argument(
        "--load-predictions",
        type=Path,
        help="replay predictions saved by --save-predictions instead of predicting",
    )

    surrogate_parser = commands.add_parser(
        "gen-surrogate-data",
        parents=[shared],
        help="generate the surrogate's training dataset (Drake-predicted pushes)",
    )
    surrogate_parser.add_argument(
        "--samples", type=int, default=10000, help="samples to generate"
    )
    surrogate_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed; give each shard its own so their samples are independent",
    )
    surrogate_parser.add_argument(
        "--path-family",
        type=Path,
        default=paths.PATH_FAMILY,
        help="calibrated executed-path deviation family",
    )
    surrogate_parser.add_argument(
        "--out",
        type=Path,
        default=paths.SURROGATE_SHARDS / "shard_0.npz",
        help="dataset (shard) output path",
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
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="cpu: members in parallel processes; cuda: members in turn on the GPU",
    )
    train_parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        default=[paths.SURROGATE_SHARDS / "shard_0.npz"],
        help="one or more dataset shards from gen-surrogate-data",
    )
    train_parser.add_argument(
        "--out",
        type=Path,
        default=paths.SURROGATE_ENSEMBLE,
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
        "--resume",
        action="store_true",
        help="continue from the latest checkpoint in --out (a fresh start if none)",
    )
    ppo_parser.add_argument(
        "--ensemble",
        type=Path,
        default=paths.SURROGATE_ENSEMBLE,
        help="trained surrogate ensemble",
    )
    ppo_parser.add_argument(
        "--path-family",
        type=Path,
        default=paths.PATH_FAMILY,
        help="calibrated executed-path deviation family",
    )
    ppo_parser.add_argument(
        "--out", type=Path, default=paths.PPO_RUNS, help="output directory"
    )

    family_parser = commands.add_parser(
        "calibrate-path-family",
        parents=[shared],
        help="calibrate the executed-path deviation family from recorded trials",
    )
    family_parser.add_argument(
        "--trials",
        type=Path,
        default=paths.REPLAY_TRIALS,
        help="recorded-trials dataset to calibrate from",
    )
    family_parser.add_argument(
        "--out",
        type=Path,
        default=paths.PATH_FAMILY,
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
        default=paths.PPO_POLICY,
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
        default=paths.SURROGATE_ENSEMBLE,
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
        default=paths.RESULTS / "evaluation_report.json",
        help="evaluation report json from the evaluate command",
    )
    report_parser.add_argument(
        "--model",
        type=Path,
        default=paths.PPO_POLICY,
        help="trained probing policy for the behavioral probe",
    )
    report_parser.add_argument(
        "--ensemble",
        type=Path,
        default=paths.SURROGATE_ENSEMBLE,
        help="trained surrogate ensemble",
    )
    report_parser.add_argument(
        "--path-family",
        type=Path,
        default=paths.PATH_FAMILY,
        help="calibrated executed-path deviation family",
    )
    report_parser.add_argument(
        "--out",
        type=Path,
        default=paths.RESULTS / "evaluation",
        help="output directory",
    )

    placement_parser = commands.add_parser(
        "placement-eval",
        parents=[shared],
        help="one placement episode under one probing condition",
    )
    placement_parser.add_argument(
        "--seed", type=int, required=True, help="episode seed"
    )
    placement_parser.add_argument(
        "--condition",
        required=True,
        help='"true_eta", "none", or "<learned|random|long_fast>:<probe pushes>"',
    )
    placement_parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        default="near",
        help="where start and goal are drawn: near (within 10 cm of the "
        "workspace center, the PPO training task) or far (0.30-0.45 m apart "
        "along the arm's full-reach band)",
    )
    placement_parser.add_argument(
        "--model",
        type=Path,
        default=paths.PPO_POLICY,
        help="trained probing policy (learned conditions)",
    )
    placement_parser.add_argument(
        "--max-pushes", type=int, default=20, help="push budget, probes included"
    )
    placement_parser.add_argument(
        "--position-tolerance",
        type=float,
        default=0.005,
        help="success radius (m); run at the tightest tolerance of interest",
    )
    placement_parser.add_argument(
        "--yaw-tolerance",
        type=float,
        default=5.0,
        help="success yaw tolerance (deg)",
    )
    placement_parser.add_argument(
        "--out", type=Path, required=True, help="episode record (JSON)"
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
        default=paths.PPO_POLICY,
        help="trained probing policy",
    )
    eval_parser.add_argument(
        "--ensemble",
        type=Path,
        default=paths.SURROGATE_ENSEMBLE,
        help="trained surrogate ensemble",
    )
    eval_parser.add_argument(
        "--episodes", type=int, default=24, help="evaluation episodes"
    )
    eval_parser.add_argument(
        "--random-only",
        action="store_true",
        help="run random probing with Drake without policy or surrogate assets",
    )
    eval_parser.add_argument(
        "--push-budget",
        type=int,
        default=None,
        help="executed push limit for --random-only (default: environment maximum)",
    )
    eval_parser.add_argument("--seed", type=int, default=70000, help="PRNG seed")
    eval_parser.add_argument(
        "--fixed-pushes",
        type=int,
        default=None,
        help="every strategy makes exactly this many pushes (the policy may not "
        "stop earlier); default: the policy decides when to stop",
    )
    eval_parser.add_argument(
        "--path-family",
        type=Path,
        default=paths.PATH_FAMILY,
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
        default=paths.RESULTS / "evaluation_report.json",
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
        default=paths.PPO_POLICY,
        help="trained probing policy (--policy ppo)",
    )
    simulate_parser.add_argument(
        "--ensemble",
        type=Path,
        default=paths.SURROGATE_ENSEMBLE,
        help="trained surrogate ensemble (--policy ppo, --predictor surrogate)",
    )
    simulate_parser.add_argument(
        "--predictor",
        choices=("drake", "surrogate"),
        default="drake",
        help="prediction model the estimator's filter uses; the truth sim and "
        "MPC planning stay Drake",
    )
    simulate_parser.add_argument(
        "--path-family",
        type=Path,
        default=paths.PATH_FAMILY,
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
    simulate_parser.add_argument(
        "--cost-length",
        type=float,
        default=MPC_LENGTH,
        help="MPC pose cost's length scale l (m)",
    )
    simulate_parser.add_argument(
        "--yaw-weight",
        type=float,
        default=MPC_YAW_WEIGHT,
        help="MPC pose cost's yaw weight lambda_phi (1/rad^2)",
    )
    simulate_parser.add_argument(
        "--time-weight",
        type=float,
        default=MPC_TIME_WEIGHT,
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
    # 3 has heavy tails; see src/estimation/estimator.py.
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
    if (
        args.command == "evaluate"
        and args.push_budget is not None
        and (not args.random_only or not 0 <= args.push_budget <= MAX_PUSHES)
    ):
        raise SystemExit(
            f"--push-budget requires --random-only and must be 0–{MAX_PUSHES}"
        )

    return args


def record_trials_command(args: argparse.Namespace) -> None:
    """Record the offline replay's dataset with the settings it ran under."""
    setup = make_record_setup(
        rod_mass=args.rod_mass,
        pushes=args.pushes,
        truth=truth_sim_config(args),
    )
    dataset = record_trials(
        setup, trials=args.trials, seed=args.seed, workers=args.workers
    )
    save_dataset(
        args.out,
        dataset,
        metadata=trials_metadata(
            setup, args.trials, args.seed, push_model_config(args).time_step
        ),
    )
    print(f"wrote {args.trials} trials to {args.out}")


def generate_surrogate_data_command(args: argparse.Namespace) -> None:
    """Generate the surrogate's training dataset: sampled (eta, start pose,
    action) tuples replayed through the Drake prediction model."""
    setup = make_surrogate_setup(
        rod_mass=args.rod_mass,
        settle_time=args.settle_time,
        family=load_path_family(args.path_family),
        model=push_model_config(args),
    )
    dataset = generate_surrogate_data(
        setup, samples=args.samples, seed=args.seed, workers=args.workers
    )
    dataset["split"] = assign_split(dataset["sample_seed"])
    check_finite(dataset)
    print(coverage_summary(dataset, setup))
    save_dataset(
        args.out, dataset, metadata=surrogate_metadata(setup, args.samples, args.seed)
    )
    print(f"wrote {args.samples} samples to {args.out}")


def train_surrogate_command(args: argparse.Namespace) -> None:
    """Train the surrogate's ensemble and report its validation errors
    against the estimator's tolerated prediction error."""
    print(
        train_ensemble(
            dataset_paths=args.dataset,
            out_path=args.out,
            members=args.members,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            hidden=args.hidden,
            layers=args.layers,
            device=args.device,
        )
    )


def evaluation_report_command(args: argparse.Namespace) -> str:
    """Render the evaluation report: the paired table, the plots and the
    behavioral summary, from the evaluation report json."""
    from src.evaluation.report import make_evaluation_report

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
    reference = reference_object()
    family, residual = calibrate_path_family(args.trials, transfer_height(reference))
    save_path_family(args.out, family)
    print(
        f"path family: {family.n_components} components, "
        f"rank-kept residual {residual * 1000:.2f} mm, "
        f"deviation RMS {deviation_rms(family) * 1000:.2f} mm; "
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
            random_only=args.random_only,
            random_pushes=args.push_budget,
            fixed_pushes=args.fixed_pushes,
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
            resume=args.resume,
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
                truth=truth_sim_config(args),
                model=push_model_config(args),
                predictor=args.predictor,
                ensemble_path=args.ensemble,
                n_subcells=args.subcells,
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
            n_subcells=args.subcells,
            predictions_out=args.save_predictions,
            predictions_in=args.load_predictions,
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
    elif args.command == "placement-eval":
        record, summary = run_placement(
            seed=args.seed,
            condition_name=args.condition,
            task_name=args.task,
            policy_path=args.model,
            workers=args.workers,
            max_pushes=args.max_pushes,
            tolerance=GoalTolerance(
                position=args.position_tolerance,
                yaw=float(np.deg2rad(args.yaw_tolerance)),
            ),
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=1) + "\n")
        print(summary)
    elif args.command == "evaluate":
        evaluate_command(args)
    else:
        simulate(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
