"""Motion diagnostics for truth-simulation pushes.

Pushes come either from seeded random draws (--seed) or from a JSON plan of
listed pushes (--plan), run one after another on one continuous truth sim.
Each push is measured on the sim's state log and its PushObservation: transfer
clearance, joint speed, tracking of the commanded push line, release clearance
from the block, and block drift after the release.  World frame unless noted;
lengths in m, speeds in m/s, angles in rad, times in s.
"""

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
from pydrake.all import RigidTransform, StartMeshcat
from pydrake.geometry import Meshcat

from src import paths
from src.action import Action
from src.geometry import SurfacePart, distance_to_footprint
from src.iiwa import STANDOFF, Pusher
from src.object import Object, Rod
from src.observation import PushObservation, wrap_angle
from src.probing import ProbingState, PushRecord, random_selector, run_probing
from src.push_space import PushSpace, make_push_space
from src.recording import TruthSim, TruthSimConfig, read_object_state, read_tip
from src.scene import meshcat_visualizer

ROD_MASS = 0.2  # kg, the composite's welded-in rod
BLOCK_START_XY = (0.55, 0.0)  # m, block start in the world frame
TOP_MARGIN = 0.001  # m above the block top, where the tip counts as contact height
EPS = 1e-6  # s, excludes logged samples that sit on window edges


@dataclass(frozen=True)
class PushSpec:
    """One listed push: contact at normalized position `s` in [0, 1] along the
    usable span of outline face `face`, commanded speed (m/s), contact time (s)."""

    face: int
    s: float
    speed: float
    duration: float


@dataclass(frozen=True)
class Plan:
    """A listed push sequence on one block: friction coefficient `friction`,
    initial `yaw` (rad), the rod's body-frame (x, y) in m, and the pushes in
    order."""

    friction: float
    yaw: float
    rod: tuple[float, float]
    pushes: tuple[PushSpec, ...]


def load_plan(path: Path) -> Plan:
    """Read a plan JSON file: {"friction", "yaw", "rod" (optional, [x, y] m),
    "pushes": [{"face", "s", "speed", "duration"}, ...]}."""
    data = json.loads(path.read_text())
    rod = data.get("rod", [0.0, 0.0])
    return Plan(
        friction=float(data["friction"]),
        yaw=float(data["yaw"]),
        rod=(float(rod[0]), float(rod[1])),
        pushes=tuple(
            PushSpec(
                face=int(push["face"]),
                s=float(push["s"]),
                speed=float(push["speed"]),
                duration=float(push["duration"]),
            )
            for push in data["pushes"]
        ),
    )


def spec_action(space: PushSpace, spec: PushSpec) -> Action:
    """The commanded push for `spec` on `space`.

    Raises ValueError when the face index is out of range, `s` is outside
    [0, 1], or the speed or duration is not positive.
    """
    if not 0 <= spec.face < len(space.faces):
        raise ValueError(f"face {spec.face} out of range for {len(space.faces)} faces")
    if not 0.0 <= spec.s <= 1.0:
        raise ValueError(f"s must lie in [0, 1], got {spec.s}")
    return Action(
        contact=space.contact(spec.face, spec.s),
        velocity=spec.speed,
        time=spec.duration,
    )


class PlanSequence:
    """Hands out a plan's pushes in order, one per call of `select`.

    A push whose arm plan failed is recorded in `ik_failed` and skipped.  After
    the last push, `select` returns "done".
    """

    def __init__(self, actions: list[Action]) -> None:
        self.actions = actions
        self.next_index = 0
        self.pending: int | None = None  # index handed out, not yet resolved
        self.ik_failed: list[int] = []

    def select(self, _state: ProbingState, failed: list[Action]) -> Action | str:
        if failed:
            assert self.pending is not None
            self.ik_failed.append(self.pending)
            self.next_index = self.pending + 1
            self.pending = None
        if self.next_index >= len(self.actions):
            return "done"
        self.pending = self.next_index
        return self.actions[self.next_index]

    def executed(self) -> int:
        """Mark the pending push as executed, returning its index."""
        assert self.pending is not None
        index = self.pending
        self.next_index = index + 1
        self.pending = None
        return index

    def unrun(self) -> list[int]:
        """Indices of the pushes never executed nor failed in the arm plan."""
        start = self.next_index if self.pending is None else self.pending
        return list(range(start, len(self.actions)))


class LoggedSamples(NamedTuple):
    """Plant-state log samples inside a time window: times (n,) s, stick tip
    world positions (n, 3) m, block planar states (n, 6) as (x, y, yaw, vx, vy,
    wz), and iiwa joint velocities (n, 7) rad/s."""

    times: np.ndarray
    tips: np.ndarray
    blocks: np.ndarray
    joint_velocities: np.ndarray


def read_logged_samples(
    sim: TruthSim, obj: Object, pusher: Pusher, start: float, end: float
) -> LoggedSamples:
    """Read the state log at times strictly inside (`start`, `end`) s."""
    log = sim.logger.FindLog(sim.context)
    times = log.sample_times()
    inside = (times > start + EPS) & (times < end - EPS)
    scratch = sim.plant.CreateDefaultContext()
    iiwa = sim.plant.GetBodyByName("iiwa_link_0").model_instance()
    tips, blocks, velocities = [], [], []
    for state in log.data()[:, inside].T:
        sim.plant.SetPositionsAndVelocities(scratch, state)
        tips.append(read_tip(sim.plant, scratch, pusher)[0])
        blocks.append(read_object_state(sim.plant, scratch, obj).planar())
        velocities.append(sim.plant.GetVelocities(scratch, iiwa))
    return LoggedSamples(
        times[inside],
        np.array(tips).reshape(-1, 3),
        np.array(blocks).reshape(-1, 6),
        np.array(velocities).reshape(-1, 7),
    )


def commanded_push_line(
    block_start: RigidTransform, action: Action
) -> tuple[np.ndarray, np.ndarray]:
    """The commanded tip's approach point (3,) m and unit inward direction (3,)
    at the push window's start, in the world frame, for a block at
    `block_start`.  The tip starts STANDOFF short of the contact along the
    inward normal."""
    contact = block_start @ action.contact.position
    normal = block_start.rotation() @ action.contact.normal
    return contact - STANDOFF * normal, normal


def tracking_errors(
    times: np.ndarray,
    tips: np.ndarray,
    approach: np.ndarray,
    direction: np.ndarray,
    speed: float,
    push_time: float,
) -> dict[str, float]:
    """Deviation of the tip from the commanded line during the push segment.

    `times` (n,) s from the window start, `tips` (n, 3) m, `approach` (3,) m and
    unit horizontal `direction` (3,) of the commanded line, `speed` m/s, and
    `push_time` s to the push segment's end.  Returns:
    - max_lateral_m: max |error| perpendicular to the push, in xy (m)
    - max_abs_z_m, max_abs_z_time_s: max |z error| (m) and when it occurs (s
      from window start), so lowering transients can be told from in-push dips
    - z_at_push_end_m: signed z error at push end (m)
    - z_rms_m: RMS z error over the push segment (m, samples unweighted)
    - along_track_lag_m: commanded travel minus actual travel at push end (m),
      positive when the tip falls behind.
    """
    commanded = approach + speed * times[:, None] * direction
    error = tips - commanded
    lateral_axis = np.array([-direction[1], direction[0], 0.0])
    in_push = times <= push_time + EPS
    z_error = error[:, 2]
    z_in = z_error[in_push]
    times_in = times[in_push]
    worst = int(np.argmax(np.abs(z_in)))
    along_actual = (tips - approach) @ direction
    lag = speed * push_time - float(np.interp(push_time, times, along_actual))
    return {
        "max_lateral_m": float(np.max(np.abs((error @ lateral_axis)[in_push]))),
        "max_abs_z_m": float(np.abs(z_in[worst])),
        "max_abs_z_time_s": float(times_in[worst]),
        "z_at_push_end_m": float(np.interp(push_time, times, z_error)),
        "z_rms_m": float(np.sqrt(np.mean(z_in**2))),
        "along_track_lag_m": lag,
    }


def joint_speed_ratio(velocities: np.ndarray, limits: np.ndarray) -> tuple[float, int]:
    """The max of |joint velocity| / velocity limit over samples (n, 7) and
    joints, with the joint index of that max.  `limits` (7,) are rad/s."""
    ratios = np.abs(velocities) / limits
    row, joint = np.unravel_index(int(np.argmax(ratios)), ratios.shape)
    return float(ratios[row, joint]), int(joint)


def release_clearances(
    tips: np.ndarray,
    block_poses: np.ndarray,
    parts: list[SurfacePart],
    stick_radius: float,
) -> np.ndarray:
    """Stick-to-footprint clearance (m) per sample, (n,).

    `tips` (n, 3) are world tip positions and `block_poses` (n, 3) the block's
    planar (x m, y m, yaw rad) at the same samples.  The clearance is the
    footprint distance of the tip's xy, in the block's body frame, minus
    `stick_radius`; negative means overlap.
    """
    clearances = np.empty(len(tips))
    for i, (tip, (x, y, yaw)) in enumerate(zip(tips, block_poses)):
        dx, dy = tip[0] - x, tip[1] - y
        body_x = np.cos(yaw) * dx + np.sin(yaw) * dy
        body_y = -np.sin(yaw) * dx + np.cos(yaw) * dy
        clearances[i] = distance_to_footprint(parts, body_x, body_y) - stick_radius
    return clearances


def release_summary(
    times: np.ndarray,
    tips: np.ndarray,
    block_poses: np.ndarray,
    parts: list[SurfacePart],
    block_top: float,
    stick_radius: float,
) -> dict[str, float | bool | None]:
    """How the stick leaves the block, from push end on.

    `times` (n,) s from push end.  Only samples with the tip below
    `block_top` + TOP_MARGIN count as contact height.  Returns:
    - min_clearance_m, min_clearance_time_s: raw minimum over those samples
    - first_clear_time_s: first sample whose tip is above that height, or whose
      clearance is > 0 while below it, whichever comes first
    - recontact: a later below-height sample has clearance <= 0
    - min_clearance_after_clear_m: minimum over later below-height samples
    Values are None where no sample qualifies.
    """
    below = tips[:, 2] < block_top + TOP_MARGIN
    clearance = release_clearances(tips, block_poses, parts, stick_radius)
    summary: dict[str, float | bool | None] = {
        "min_clearance_m": None,
        "min_clearance_time_s": None,
        "first_clear_time_s": None,
        "recontact": False,
        "min_clearance_after_clear_m": None,
    }
    if np.any(below):
        best = int(np.argmin(np.where(below, clearance, np.inf)))
        summary["min_clearance_m"] = float(clearance[best])
        summary["min_clearance_time_s"] = float(times[best])
    cleared = ~below | (clearance > 0.0)
    if np.any(cleared):
        first_time = float(times[int(np.argmax(cleared))])
        summary["first_clear_time_s"] = first_time
        later_below = (times > first_time) & below
        summary["recontact"] = bool(np.any(later_below & (clearance <= 0.0)))
        if np.any(later_below):
            summary["min_clearance_after_clear_m"] = float(clearance[later_below].min())
    return summary


def planar_drift(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    """Block translation (m) and |yaw change| (rad) between two planar states
    (x, y, yaw, ...)."""
    translation = float(np.hypot(*(after[:2] - before[:2])))
    return translation, float(abs(wrap_angle(after[2] - before[2])))


def action_summary(action: Action, spec: PushSpec | None) -> dict[str, object]:
    """The commanded push: face and s when it came from a plan (else None),
    speed (m/s), contact time (s), push distance (m), and the contact point in
    the block's body frame (m)."""
    return {
        "face": None if spec is None else spec.face,
        "s": None if spec is None else spec.s,
        "speed_m_per_s": action.velocity,
        "contact_time_s": action.time,
        "push_distance_m": action.velocity * action.time,
        "contact_body_m": action.contact.position.tolist(),
    }


class MotionRun:
    """Diagnoses each push as it executes on one truth sim.  Transfer and joint
    windows open at the previous push's end, so the sim's clock must stop
    between pushes."""

    def __init__(self, sim: TruthSim, obj: Object, pusher: Pusher) -> None:
        self.sim = sim
        self.obj = obj
        self.pusher = pusher
        self.previous_end = sim.context.get_time()
        iiwa = sim.plant.GetBodyByName("iiwa_link_0").model_instance()
        self.joint_limits = np.asarray(
            sim.plant.GetVelocitiesFromArray(iiwa, sim.plant.GetVelocityUpperLimits())
        )
        self.joint_names = list(
            sim.plant.GetVelocityNames(
                iiwa, add_model_instance_prefix=False, always_add_suffix=False
            )
        )
        self.results: list[dict[str, object]] = []

    def diagnose(self, record: PushRecord, index: int, spec: PushSpec | None) -> None:
        """Measure the push in `record` (the index-th result) and advance the
        window start to the sim's current time."""
        self.results.append(self._measure(record.observation, index, spec))
        self.previous_end = self.sim.context.get_time()

    def _measure(
        self, observation: PushObservation, index: int, spec: PushSpec | None
    ) -> dict[str, object]:
        action_end = self.sim.context.get_time()
        assert observation.action_end_time is not None
        assert observation.transfer_end_time is not None
        window_start = action_end - observation.action_end_time
        push_end = window_start + observation.push_end_time
        transfer_end = window_start + observation.transfer_end_time
        samples = read_logged_samples(
            self.sim, self.obj, self.pusher, self.previous_end, action_end
        )
        ratio, joint = joint_speed_ratio(samples.joint_velocities, self.joint_limits)
        transfer_tips = samples.tips[samples.times < transfer_end - EPS]
        approach, direction = commanded_push_line(
            observation.start.pose, observation.action
        )
        return {
            "push": index + 1,
            "action": action_summary(observation.action, spec),
            "physical_duration_s": action_end - self.previous_end,
            "max_abs_joint_speed_rad_per_s": float(
                np.max(np.abs(samples.joint_velocities))
            ),
            "max_joint_speed_ratio": {
                "ratio": ratio,
                "joint": self.joint_names[joint],
            },
            "transfer_min_tip_height_m": _min_height(transfer_tips),
            "transfer_clearance_above_top_m": (
                None
                if transfer_tips.size == 0
                else float(transfer_tips[:, 2].min() - self.obj.height)
            ),
            "tracking": tracking_errors(
                observation.tip_times,
                observation.tip_positions.T,
                approach,
                direction,
                observation.action.velocity,
                observation.push_end_time,
            ),
            "release": self._release(samples, push_end),
            "post_measurement_motion": _drift_entry(observation),
            "action_end_object_velocity": observation.final.planar()[3:].tolist(),
            "action_end_tip_height_m": float(
                read_tip(self.sim.plant, self.sim.plant_context, self.pusher)[0][2]
            ),
        }

    def _release(
        self, samples: LoggedSamples, push_end: float
    ) -> dict[str, float | bool | None]:
        after_push = samples.times > push_end + EPS
        return release_summary(
            samples.times[after_push] - push_end,
            samples.tips[after_push],
            samples.blocks[after_push, :3],
            self.obj.parts,
            self.obj.height,
            self.pusher.radius,
        )


def _min_height(tips: np.ndarray) -> float | None:
    return None if tips.size == 0 else float(tips[:, 2].min())


def _drift_entry(observation: PushObservation) -> dict[str, float]:
    translation, yaw = planar_drift(
        observation.end.planar(), observation.final.planar()
    )
    return {"translation_m": translation, "abs_yaw_change_rad": yaw}


def start_sim(
    rod: tuple[float, float],
    friction: float,
    yaw: float,
    rng: np.random.Generator,
    meshcat: Meshcat | None,
) -> tuple[Object, Pusher, TruthSim]:
    """The block with its rod at `rod` (m), resting at BLOCK_START_XY with `yaw`
    (rad), and a truth sim holding it.  `rng` seeds the block's random start and
    goal, which this overrides."""
    obj = Object(rng, rod=Rod(rod[0], rod[1], ROD_MASS), friction=friction)
    obj.pose = (BLOCK_START_XY[0], BLOCK_START_XY[1], yaw)
    pusher = Pusher()
    sim = TruthSim(
        obj,
        pusher,
        TruthSimConfig(time_step=1e-4, settle_time=0.3, log_period=1e-3),
        meshcat=meshcat,
    )
    return obj, pusher, sim


def run_recorded[T](sim: TruthSim, meshcat: Meshcat | None, body: Callable[[], T]) -> T:
    """Run `body`, recording the sim's Meshcat frames when `meshcat` is given."""
    if meshcat is None:
        return body()
    visualizer = meshcat_visualizer(sim.diagram)
    visualizer.StartRecording()
    result = body()
    visualizer.PublishRecording()
    return result


def run_random(
    seed: int,
    pushes: int,
    friction: float,
    yaw: float,
    meshcat: Meshcat | None = None,
) -> dict[str, object]:
    """Diagnose `pushes` random reachable pushes drawn with `seed`."""
    obj, pusher, sim = start_sim(
        (0.0, 0.0), friction, yaw, np.random.default_rng(seed), meshcat
    )
    action_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    run = MotionRun(sim, obj, pusher)
    state = ProbingState(truth_sim=sim, obj=obj, belief=None)
    started = time.perf_counter()

    def body() -> str:
        outcome = run_probing(
            state,
            random_selector(make_push_space(obj.parts, obj.height), action_rng),
            budget=pushes,
            on_push=lambda _state, record: run.diagnose(record, len(run.results), None),
        )
        return outcome.reason

    reason = run_recorded(sim, meshcat, body)
    return {
        "seed": seed,
        "friction": friction,
        "initial_yaw_rad": yaw,
        "stop_reason": reason,
        "pushes": run.results,
        "wall_seconds": time.perf_counter() - started,
    }


def run_plan(plan: Plan, meshcat: Meshcat | None = None) -> dict[str, object]:
    """Execute the plan's pushes in order and diagnose each one.  Pushes whose
    arm plan fails are reported with `ik_failed`; pushes not reached are reported
    with `not_run` and the stop reason."""
    obj, pusher, sim = start_sim(
        plan.rod, plan.friction, plan.yaw, np.random.default_rng(0), meshcat
    )
    space = make_push_space(obj.parts, obj.height)
    actions = [spec_action(space, spec) for spec in plan.pushes]
    sequence = PlanSequence(actions)
    run = MotionRun(sim, obj, pusher)
    state = ProbingState(truth_sim=sim, obj=obj, belief=None)
    started = time.perf_counter()

    def on_push(_state: ProbingState, record: PushRecord) -> None:
        index = sequence.executed()
        run.diagnose(record, index, plan.pushes[index])

    def body() -> str:
        return run_probing(
            state, sequence.select, budget=len(actions), on_push=on_push
        ).reason

    reason = run_recorded(sim, meshcat, body)
    entries = run.results + _skipped_entries(plan, actions, sequence, reason)
    return {
        "friction": plan.friction,
        "initial_yaw_rad": plan.yaw,
        "rod_m": list(plan.rod),
        "stop_reason": reason,
        "pushes": sorted(entries, key=_push_number),
        "wall_seconds": time.perf_counter() - started,
    }


def _push_number(entry: dict[str, object]) -> int:
    number = entry["push"]
    assert isinstance(number, int)
    return number


def _skipped_entries(
    plan: Plan, actions: list[Action], sequence: PlanSequence, reason: str
) -> list[dict[str, object]]:
    """Entries for listed pushes that failed in the arm plan or never ran."""
    failed = [
        {
            "push": index + 1,
            "action": action_summary(actions[index], plan.pushes[index]),
            "ik_failed": True,
        }
        for index in sequence.ik_failed
    ]
    not_run = [
        {
            "push": index + 1,
            "action": action_summary(actions[index], plan.pushes[index]),
            "not_run": reason,
        }
        for index in sequence.unrun()
    ]
    return [*failed, *not_run]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, help="JSON plan of listed pushes")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--pushes", type=int)
    parser.add_argument(
        "--friction", type=float, help="table friction (overrides a plan's value)"
    )
    parser.add_argument("--yaw", type=float)
    parser.add_argument(
        "--out", type=Path, default=paths.RESULTS / "motion" / "motion_diagnostic.json"
    )
    parser.add_argument(
        "--html", type=Path, help="save a static Meshcat replay of the run here"
    )
    args = parser.parse_args()
    random_flags = [
        flag for flag in ("seed", "pushes", "yaw") if getattr(args, flag) is not None
    ]
    if args.plan is not None and random_flags:
        parser.error("--plan replaces " + " ".join(f"--{f}" for f in random_flags))

    meshcat = None
    if args.html is not None:
        # StartMeshcat lives in pydrake's unannotated _geometry_extra.py,
        # so strict mypy sees the call as untyped.
        meshcat = StartMeshcat()  # type: ignore[no-untyped-call]
    if args.plan is not None:
        plan = load_plan(args.plan)
        if args.friction is not None:
            plan = replace(plan, friction=args.friction)
        report = run_plan(plan, meshcat)
        report["plan"] = str(args.plan)
    else:
        report = run_random(
            args.seed if args.seed is not None else 123,
            args.pushes if args.pushes is not None else 2,
            args.friction if args.friction is not None else 0.5,
            args.yaw if args.yaw is not None else 0.0,
            meshcat,
        )
    if meshcat is not None:
        assert args.html is not None
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(meshcat.StaticHtml())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
