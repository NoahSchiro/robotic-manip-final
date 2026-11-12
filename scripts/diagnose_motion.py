"""Short truth-simulation check of consecutive push transfer and release."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.iiwa import MOVE_TIME, Pusher, clear_sites
from src.object import Object, Rod
from src.recording import TruthSim, TruthSimConfig, read_tip


def run(seed: int, pushes: int, friction: float, yaw: float) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    obj = Object(rng, rod=Rod(0.0, 0.0, 0.2), friction=friction)
    obj.pose = (0.55, 0.0, yaw)
    pusher = Pusher()
    sites = clear_sites(obj.parts, obj.surface_normals(200.0), pusher.radius, 0.001)
    sim = TruthSim(
        obj,
        pusher,
        TruthSimConfig(time_step=1e-4, settle_time=0.1, log_period=1e-3),
    )
    results = []
    started = time.perf_counter()
    for index in range(pushes):
        previous_end = sim.context.get_time()
        observation = sim.record_random_push(rng, obj, sites, max_tries=40)
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
        result = {
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
        results.append(result)
        obj.pose = observation.final_pose()
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
