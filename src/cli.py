"""The command-line options every estimation entry point shares.

The defaults live only here: main.py must run with the settings R was
calibrated under, so the two entry points must never disagree.
"""

import argparse
import os

from .dynamics import PushModelConfig
from .recording import TruthSimConfig


def add_shared_options(parser: argparse.ArgumentParser) -> None:
    """Add the truth sim, prediction model and object options to `parser`."""
    parser.add_argument(
        "--time-step",
        type=float,
        default=1e-4,
        help="simulation timestep (nonnegative)",
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
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="prediction processes (default: all CPUs)",
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
        "--log-period",
        type=float,
        default=1e-3,
        help="plant-state logging period for recorded pushes (s)",
    )


def finish_shared_options(args: argparse.Namespace) -> None:
    """Validate the shared options in `args`, converting --push-density from
    count/cm to count/m in place.

    Raises SystemExit when an option is out of range.
    """
    if args.time_step < 0.0:
        raise SystemExit("--time-step must be nonnegative")
    if args.rod_mass <= 0.0:
        raise SystemExit("--rod-mass must be positive")
    if args.push_density <= 0.0:
        raise SystemExit("--push-density must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.log_period <= 0.0:
        raise SystemExit("--log-period must be positive")
    args.push_density *= 100  # count/cm -> count/m


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
