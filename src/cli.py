"""The command-line options main.py's subcommands share.

The defaults come from src.experiment: simulate must run with the settings
R was calibrated under, so no subcommand may define its own.
"""

import argparse
import os

from src.constants import (
    LOG_PERIOD,
    MODEL_TIME_STEP,
    ROD_MASS,
    SETTLE_TIME,
    TRUTH_TIME_STEP,
)
from src.estimation.dynamics import PushModelConfig
from src.world.recording import TruthSimConfig


def add_shared_options(parser: argparse.ArgumentParser) -> None:
    """Add the truth sim, prediction model and object options to `parser`."""
    parser.add_argument(
        "--time-step",
        type=float,
        default=TRUTH_TIME_STEP,
        help="simulation timestep (nonnegative)",
    )
    parser.add_argument(
        "--rod-mass",
        type=float,
        default=ROD_MASS,
        help="rod mass (kg), additional to the block's own mass",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="prediction processes (default: all CPUs)",
    )
    parser.add_argument(
        "--model-time-step",
        type=float,
        default=MODEL_TIME_STEP,
        help="estimator prediction model timestep",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=SETTLE_TIME,
        help="measurement delay after each push segment ends (s)",
    )
    parser.add_argument(
        "--log-period",
        type=float,
        default=LOG_PERIOD,
        help="plant-state logging period for recorded pushes (s)",
    )


def finish_shared_options(args: argparse.Namespace) -> None:
    """Validate the shared options in `args`.

    Raises SystemExit when an option is out of range.
    """
    if args.time_step < 0.0:
        raise SystemExit("--time-step must be nonnegative")
    if args.rod_mass <= 0.0:
        raise SystemExit("--rod-mass must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.log_period <= 0.0:
        raise SystemExit("--log-period must be positive")


def truth_sim_config(args: argparse.Namespace) -> TruthSimConfig:
    """Get the truth sim's settings from `args`."""
    return TruthSimConfig(
        time_step=args.time_step,
        settle_time=args.settle_time,
        log_period=args.log_period,
    )


def push_model_config(args: argparse.Namespace) -> PushModelConfig:
    """Get the prediction model's settings from `args`."""
    return PushModelConfig(time_step=args.model_time_step, rod_mass=args.rod_mass)
