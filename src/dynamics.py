"""Prediction models for the estimator: where the object ends a push window
under a hypothesized eta.

DrakePushModel replays the measured tip path in a small plant: the table,
the object, and a lone stick tip on x/y/z prismatic joints that Drake's
built-in actuator PD drives along the path (no arm, IK, or arm PD).  One
plant serves every hypothesis: each prediction swaps eta's inertia and
friction into the context.  ParallelPushModel fans hypotheses out over
worker processes, each holding its own DrakePushModel.
"""

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
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
    RigidTransform,
    RoleAssign,
    RotationalInertia,
    Simulator,
    SpatialInertia,
    TrajectorySource,
)

from .iiwa import Pusher
from .object import Table, add_object, add_table
from .observation import ObjectState, PushObservation
from .params import ROD_MASS, Eta


class DynamicsModel(Protocol):
    """Predicts a push window's final planar state under each hypothesis."""

    def predict(self, etas: Sequence[Eta], obs: PushObservation) -> np.ndarray:
        """Predict the object's planar state (x, y, yaw, vx, vy, wz) at the
        window's end under each of `etas`, as an (N, 6) array."""
        ...


class DrakePushModel:
    """Short-horizon Drake prediction: replay the measured tip path against
    the object under a hypothesized eta, from the measured start state."""

    def __init__(
        self,
        time_step: float = 2e-3,
        tip_kp: float = 1e4,
        tip_kd: float = 200.0,
        rod_mass: float = ROD_MASS,
    ) -> None:
        self.rod_mass = rod_mass
        # The object's own pose is irrelevant: predict() sets its state.
        self.rng = np.random.default_rng(0)
        self.pusher = Pusher()
        obj = Eta(mu=0.5, bx=0.0, by=0.0).to_object(self.rng, rod_mass)
        table = Table()

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
        parser = Parser(plant)
        add_table(table, plant, parser)
        add_object(obj, parser)
        self.object_body = plant.GetBodyByName(
            obj.body_box, plant.GetModelInstanceByName(obj.name)
        )
        table_body = plant.GetBodyByName(
            table.body_box, plant.GetModelInstanceByName(table.name)
        )
        tip_instance = self._add_tip(plant, tip_kp, tip_kd)
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

    def _add_tip(
        self, plant: MultibodyPlant, kp: float, kd: float
    ) -> ModelInstanceIndex:
        """Add the stick on PD-actuated world x/y/z prismatic joints, its
        frame at the tip with the stick rising along +z, matching the truth
        stick's tool-down pose."""
        pusher = self.pusher
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
        return tip

    def set_eta(self, eta: Eta) -> None:
        """Swap `eta`'s inertia and friction into the context."""
        obj = eta.to_object(self.rng, self.rod_mass)
        _, inertia = obj.compound_inertia()
        self.object_body.SetSpatialInertiaInBodyFrame(
            self.plant_context,
            SpatialInertia.MakeFromCentralInertia(
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
            ),
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
        """Predict the window's final planar state under each of `etas`."""
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
            self.context.SetTime(0.0)
            self.plant.SetPositions(
                self.plant_context, self.tip_instance, obs.tip_positions[:, 0]
            )
            self.plant.SetVelocities(
                self.plant_context, self.tip_instance, obs.tip_velocities[:, 0]
            )
            self.plant.SetFreeBodyPose(
                self.plant_context, self.object_body, obs.start.pose
            )
            self.plant.SetFreeBodySpatialVelocity(
                self.plant_context, self.object_body, obs.start.velocity
            )
            self.simulator.Initialize()
            self.simulator.AdvanceTo(obs.duration)
            predictions[i] = ObjectState(
                pose=self.plant.EvalBodyPoseInWorld(
                    self.plant_context, self.object_body
                ),
                velocity=self.plant.EvalBodySpatialVelocityInWorld(
                    self.plant_context, self.object_body
                ),
            ).planar()
        return predictions


# Each worker process's own model, built once by _init_worker.
_worker_model: DrakePushModel | None = None


def _init_worker(
    time_step: float, tip_kp: float, tip_kd: float, rod_mass: float
) -> None:
    """Build this worker process's prediction model."""
    global _worker_model
    _worker_model = DrakePushModel(time_step, tip_kp, tip_kd, rod_mass)


def _predict_chunk(etas: list[Eta], obs: PushObservation) -> np.ndarray:
    """Predict one chunk of hypotheses on this worker's model."""
    assert _worker_model is not None, "worker was not initialized"
    return _worker_model.predict(etas, obs)


class ParallelPushModel:
    """DrakePushModel's predictions split across `workers` processes (all
    CPUs by default).  Use as a context manager, or call close()."""

    def __init__(
        self,
        workers: int | None = None,
        time_step: float = 2e-3,
        tip_kp: float = 1e4,
        tip_kd: float = 200.0,
        rod_mass: float = ROD_MASS,
    ) -> None:
        self.workers = workers if workers is not None else os.cpu_count() or 1
        self.pool = ProcessPoolExecutor(
            self.workers,
            initializer=_init_worker,
            initargs=(time_step, tip_kp, tip_kd, rod_mass),
        )

    def predict(self, etas: Sequence[Eta], obs: PushObservation) -> np.ndarray:
        """Predict the window's final planar state under each of `etas`."""
        chunks = [
            [etas[int(i)] for i in chunk]
            for chunk in np.array_split(np.arange(len(etas)), self.workers)
            if chunk.size
        ]
        futures = [self.pool.submit(_predict_chunk, chunk, obs) for chunk in chunks]
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
