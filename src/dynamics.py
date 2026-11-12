"""Prediction models for the estimator: where the object ends a push window
under a hypothesized eta.

DrakePushModel replays the measured tip path in a small plant: the table,
the object, and a lone stick tip on x/y/z prismatic joints that Drake's
built-in actuator PD drives along the path (no arm, IK, or arm PD).  One
plant serves every hypothesis: each prediction swaps eta's inertia and
friction into the context.  ParallelPushModel fans the hypotheses for one
push out over worker processes, each chunk on its own DrakePushModel.
"""

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, Self

import numpy as np
from pydrake.all import (
    AddCompliantHydroelasticProperties,
    AddContactMaterial,
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
    Cylinder,
    DiagramBuilder,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
    PdControllerGains,
    PiecewisePolynomial,
    PrismaticJoint,
    ProximityProperties,
    RigidBody,
    RigidTransform,
    RoleAssign,
    RotationalInertia,
    Simulator,
    SpatialInertia,
    TrajectorySource,
)

from .iiwa import Pusher
from .object import Object, Table, add_object, add_table
from .observation import ObjectState, PushObservation, measurement
from .params import Eta


class DynamicsModel(Protocol):
    """Predicts a push's measurement under each hypothesis."""

    def predict(self, etas: Sequence[Eta], obs: PushObservation) -> np.ndarray:
        """Predict the push's measurement (x, y, yaw at the window's end;
        vx, vy, wz at the push segment's end; see observation.measurement)
        under each of `etas`, as an (N, 6) array."""
        ...


@dataclass(frozen=True)
class PushModelConfig:
    """How DrakePushModel simulates: plant timestep (s), the known rod mass
    (kg), and the PD gains driving the tip along its path."""

    time_step: float
    rod_mass: float
    tip_kp: float = 1e4
    tip_kd: float = 200.0


def _add_tip_body(
    plant: MultibodyPlant, pusher: Pusher, kp: float, kd: float
) -> tuple[ModelInstanceIndex, RigidBody]:
    """Add the stick on PD-actuated world x/y/z prismatic joints, its frame at
    the tip with the stick rising along +z, matching the truth stick's
    tool-down pose."""
    tip = plant.AddModelInstance("tip")
    # The tip's PD holds z directly, so gravity would only add sag.
    plant.set_gravity_enabled(tip, False)
    carriage_x = plant.AddRigidBody(
        "carriage_x", tip, SpatialInertia.SolidSphereWithMass(0.01, 0.005)
    )
    carriage_y = plant.AddRigidBody(
        "carriage_y", tip, SpatialInertia.SolidSphereWithMass(0.01, 0.005)
    )
    stick = plant.AddRigidBody(
        pusher.body,
        tip,
        SpatialInertia.SolidCylinderWithMassAboutEnd(
            pusher.mass, pusher.radius, pusher.length, np.array([0.0, 0.0, 1.0])
        ),
    )
    chain = [
        (plant.world_frame(), carriage_x.body_frame(), [1.0, 0.0, 0.0]),
        (carriage_x.body_frame(), carriage_y.body_frame(), [0.0, 1.0, 0.0]),
        (carriage_y.body_frame(), stick.body_frame(), [0.0, 0.0, 1.0]),
    ]
    for i, (parent, child, axis) in enumerate(chain):
        joint = plant.AddJoint(
            PrismaticJoint(f"tip_{'xyz'[i]}", parent, child, np.array(axis))
        )
        actuator = plant.AddJointActuator(joint.name(), joint)
        actuator.set_controller_gains(PdControllerGains(p=kp, d=kd))
    return tip, stick


def _add_tip_collision(plant: MultibodyPlant, pusher: Pusher, stick: RigidBody) -> None:
    """Register `stick`'s compliant hydroelastic collision cylinder."""
    props = ProximityProperties()
    AddContactMaterial(
        friction=CoulombFriction(pusher.friction, pusher.friction),
        properties=props,
    )
    AddCompliantHydroelasticProperties(
        resolution_hint=0.005,
        hydroelastic_modulus=pusher.hydroelastic_modulus,
        properties=props,
    )
    plant.RegisterCollisionGeometry(
        stick,
        RigidTransform(np.array([0.0, 0.0, pusher.length / 2])),
        Cylinder(pusher.radius, pusher.length),
        "collision",
        props,
    )


def _spatial_inertia(obj: Object) -> SpatialInertia:
    """Get `obj`'s spatial inertia about its body frame."""
    _, inertia = obj.compound_inertia()
    return SpatialInertia.MakeFromCentralInertia(
        obj.total_mass,
        obj.com_in_body_frame(),
        RotationalInertia(
            inertia[0, 0],
            inertia[1, 1],
            inertia[2, 2],
            inertia[0, 1],
            inertia[0, 2],
            inertia[1, 2],
        ),
    )


def _hypothesis_object(eta: Eta, rod_mass: float) -> Object:
    """Build `eta`'s object for its physical properties only."""
    # Object draws a random pose and goal on construction; predictions set the
    # state explicitly, so the draws never matter and any seed will do.
    return eta.to_object(np.random.default_rng(0), rod_mass)


class DrakePushModel(DynamicsModel):
    """Short-horizon Drake prediction: replay the measured tip path against
    the object under a hypothesized eta, from the measured start state."""

    def __init__(self, config: PushModelConfig) -> None:
        self.rod_mass = config.rod_mass
        obj = _hypothesis_object(Eta(mu=0.5, bx=0.0, by=0.0), config.rod_mass)
        table = Table()
        pusher = Pusher()

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=config.time_step
        )
        parser = Parser(plant)
        add_table(table, plant, parser)
        add_object(obj, parser)
        self.object_body = plant.GetBodyByName(
            obj.body_box, plant.GetModelInstanceByName(obj.name)
        )
        table_body = plant.GetBodyByName(
            table.body_box, plant.GetModelInstanceByName(table.name)
        )
        tip_instance, stick = _add_tip_body(plant, pusher, config.tip_kp, config.tip_kd)
        _add_tip_collision(plant, pusher, stick)
        plant.Finalize()

        # Placeholder path; predict() swaps in each observation's.
        self.source = TrajectorySource(
            PiecewisePolynomial.FirstOrderHold([0.0, 1.0], [[0.0, 0.0]] * 3),
            output_derivative_order=1,
        )
        builder.AddSystem(self.source)
        builder.Connect(
            self.source.get_output_port(),
            plant.get_desired_state_input_port(tip_instance),
        )
        self.diagram = builder.Build()

        self.plant = plant
        self.scene_graph = scene_graph
        self.tip_instance = tip_instance
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = plant.GetMyMutableContextFromRoot(self.context)
        self.scene_graph_context = scene_graph.GetMyMutableContextFromRoot(self.context)
        # Friction lives on both sides of the object-table contact.
        self.friction_geometries = [
            *plant.GetCollisionGeometriesForBody(self.object_body),
            *plant.GetCollisionGeometriesForBody(table_body),
        ]
        self.simulator = Simulator(self.diagram, self.context)

    def set_eta(self, eta: Eta) -> None:
        """Swap `eta`'s inertia and friction into the context."""
        self.object_body.SetSpatialInertiaInBodyFrame(
            self.plant_context,
            _spatial_inertia(_hypothesis_object(eta, self.rod_mass)),
        )
        inspector = self.scene_graph.model_inspector()
        for geometry in self.friction_geometries:
            props = ProximityProperties(inspector.GetProximityProperties(geometry))
            props.UpdateProperty(
                "material", "coulomb_friction", CoulombFriction(eta.mu, eta.mu)
            )
            self.scene_graph.AssignRole(
                self.scene_graph_context,
                self.plant.get_source_id(),
                geometry,
                props,
                RoleAssign.kReplace,
            )

    def predict(self, etas: Sequence[Eta], obs: PushObservation) -> np.ndarray:
        """Predict the push's measurement under each of `etas`."""
        self.source.UpdateTrajectory(
            PiecewisePolynomial.CubicHermite(
                [float(t) for t in obs.tip_times],
                obs.tip_positions.tolist(),
                obs.tip_velocities.tolist(),
            )
        )
        predictions = np.empty((len(etas), 6))
        for i, eta in enumerate(etas):
            self.set_eta(eta)
            push_end, end = self._simulate(
                obs.start,
                obs.tip_positions[:, 0],
                obs.tip_velocities[:, 0],
                [obs.push_end_time, obs.duration],
            )
            predictions[i] = measurement(push_end, end)
        return predictions

    def _simulate(
        self,
        start: ObjectState,
        tip_position: np.ndarray,
        tip_velocity: np.ndarray,
        times: Sequence[float],
    ) -> list[np.ndarray]:
        """Run the current tip path from time 0, with the object at `start`
        and the tip at `tip_position` moving at `tip_velocity`, reading the
        object's planar state at each of the increasing `times`."""
        self.context.SetTime(0.0)
        self.plant.SetPositions(self.plant_context, self.tip_instance, tip_position)
        self.plant.SetVelocities(self.plant_context, self.tip_instance, tip_velocity)
        self.plant.SetFreeBodyPose(self.plant_context, self.object_body, start.pose)
        self.plant.SetFreeBodySpatialVelocity(
            self.plant_context, self.object_body, start.velocity
        )
        self.simulator.Initialize()
        states = []
        for t in times:
            self.simulator.AdvanceTo(t)
            states.append(self._object_planar())
        return states

    def _object_planar(self) -> np.ndarray:
        """Read the object's current planar state."""
        return ObjectState(
            pose=self.plant.EvalBodyPoseInWorld(self.plant_context, self.object_body),
            velocity=self.plant.EvalBodySpatialVelocityInWorld(
                self.plant_context, self.object_body
            ),
        ).planar()


class ParallelPushModel(DynamicsModel):
    """DrakePushModel's predictions for one push split across `workers`
    processes: the hypotheses run in parallel, never the pushes themselves.
    Use as a context manager, or call close()."""

    def __init__(self, workers: int, config: PushModelConfig) -> None:
        self.workers = workers
        self.config = config
        self.pool = ProcessPoolExecutor(workers)

    @staticmethod
    def _predict_chunk(
        config: PushModelConfig, etas: list[Eta], obs: PushObservation
    ) -> np.ndarray:
        """Predict one chunk of hypotheses, running in a worker process."""
        # A model per chunk keeps workers stateless: building one takes tens of
        # milliseconds, against seconds simulating the chunk.
        return DrakePushModel(config).predict(etas, obs)

    def predict(self, etas: Sequence[Eta], obs: PushObservation) -> np.ndarray:
        """Predict the push's measurement under each of `etas`."""
        chunks = [
            [etas[int(i)] for i in chunk]
            for chunk in np.array_split(np.arange(len(etas)), self.workers)
            if chunk.size
        ]
        futures = [
            self.pool.submit(self._predict_chunk, self.config, chunk, obs)
            for chunk in chunks
        ]
        return np.concatenate([f.result() for f in futures])

    def close(self) -> None:
        """Shut the worker processes down."""
        self.pool.shutdown()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
