"""Short truth-simulation check of consecutive push transfer and release."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.iiwa import MOVE_TIME, Pusher
from src.object import Object, Rod
from src.observation import PushObservation
from src.probing import ProbingState, PushRecord, random_selector, run_probing
from src.push_space import make_push_space
from src.recording import TruthSim, TruthSimConfig, read_tip


def diagnose_push(
    sim: TruthSim,
    pusher: Pusher,
    obj: Object,
    observation: PushObservation,
    previous_end: float,
    index: int,
) -> dict[str, object]:
    """Tip and joint behavior of one executed push, measured on the sim's log
    between `previous_end` and the push's action end."""
    action_end = sim.context.get_time()
    assert observation.action_end_time is not None
    window_start = action_end - observation.action_end_time
    transfer_end = window_start - MOVE_TIME
    log = sim.logger.FindLog(sim.context)
    times = log.sample_times()
    transfer = (times > previous_end + 1e-6) & (times < transfer_end - 1e-6)
    in_action = (times > previous_end + 1e-6) & (times < action_end - 1e-6)
    scratch = sim.plant.CreateDefaultContext()
    iiwa = sim.plant.GetBodyByName("iiwa_link_0").model_instance()
    tip_heights = []
    for state in log.data()[:, transfer].T:
        sim.plant.SetPositionsAndVelocities(scratch, state)
        tip_heights.append(float(read_tip(sim.plant, scratch, pusher)[0][2]))
    joint_speeds = []
    for state in log.data()[:, in_action].T:
        sim.plant.SetPositionsAndVelocities(scratch, state)
        joint_speeds.append(
            float(np.max(np.abs(sim.plant.GetVelocities(scratch, iiwa))))
        )
    return {
        "push": index + 1,
        "action": {
            "speed_m_per_s": observation.action.velocity,
            "contact_time_s": observation.action.time,
        },
        "physical_duration_s": action_end - previous_end,
        "max_abs_joint_speed_rad_per_s": max(joint_speeds),
        "transfer_min_tip_height_m": min(tip_heights) if tip_heights else None,
        "transfer_clearance_above_top_m": (
            min(tip_heights) - obj.height if tip_heights else None
        ),
        "action_end_object_velocity": observation.final.planar()[3:].tolist(),
        "action_end_tip_height_m": float(
            read_tip(sim.plant, sim.plant_context, pusher)[0][2]
        ),
    }


def run(seed: int, pushes: int, friction: float, yaw: float) -> dict[str, object]:
    obj = Object(np.random.default_rng(seed), rod=Rod(0.0, 0.0, 0.2), friction=friction)
    obj.pose = (0.55, 0.0, yaw)
    pusher = Pusher()
    sim = TruthSim(
        obj,
        pusher,
        TruthSimConfig(time_step=1e-4, settle_time=0.1, log_period=1e-3),
    )
    action_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    results: list[dict[str, object]] = []
    previous_end = [sim.context.get_time()]

    def on_push(state: ProbingState, record: PushRecord) -> None:
        results.append(
            diagnose_push(
                sim, pusher, obj, record.observation, previous_end[0], len(results)
            )
        )
        previous_end[0] = sim.context.get_time()

    state = ProbingState(truth_sim=sim, obj=obj, belief=None)
    started = time.perf_counter()
    run_probing(
        state,
        random_selector(make_push_space(obj.parts, obj.height), action_rng),
        budget=pushes,
        on_push=on_push,
    )
    return {
        "seed": seed,
        "friction": friction,
        "initial_yaw_rad": yaw,
        "pushes": results,
        "wall_seconds": time.perf_counter() - started,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--pushes", type=int, default=2)
    parser.add_argument("--friction", type=float, default=0.5)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("data/motion_diagnostic.json"))
    args = parser.parse_args()
    report = run(args.seed, args.pushes, args.friction, args.yaw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
