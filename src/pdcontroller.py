"""Joint-space PD with gravity compensation for the iiwa's seven dofs.

Roughly: tau = tau_gravity(q) + Kp (q_d - q) - Kd v
"""

from typing import cast

import numpy as np
from pydrake.all import (
    BasicVector,
    Context,
    LeafSystem,
    MultibodyForces,
    MultibodyPlant,
)


class ArmPdController(LeafSystem):
    """Joint-space PD with gravity compensation on the iiwa's seven dofs."""

    def __init__(
        self, plant: MultibodyPlant, kp: float = 400.0, kd: float = 60.0
    ) -> None:
        LeafSystem.__init__(self)
        self.plant = plant
        self.context = plant.CreateDefaultContext()
        self.forces = MultibodyForces(plant)
        self.zero_vdot = np.zeros(plant.num_velocities())
        self.nq = plant.num_positions()
        self.nv = plant.num_velocities()
        self.arm_q = [
            plant.GetJointByName(f"iiwa_joint_{i}").position_start()
            for i in range(1, 8)
        ]
        self.arm_v = [
            plant.GetJointByName(f"iiwa_joint_{i}").velocity_start()
            for i in range(1, 8)
        ]
        self.kp = np.full(7, kp)
        self.kd = np.full(7, kd)

        self.DeclareVectorInputPort("plant_state", self.nq + self.nv)
        self.DeclareVectorInputPort("q_desired", 7)
        self.DeclareVectorOutputPort("tau", 7, self.calc_tau)

    def calc_tau(self, context: Context, output: BasicVector) -> None:
        """Blend the gravity feedforward with PD tracking toward `q_desired`."""
        # pydrake's InputPort.Eval is typed `-> object` because the port's
        # element type is only known at runtime; these ports hold vectors.
        x = cast(np.ndarray, self.get_input_port(0).Eval(context))
        q_desired = cast(np.ndarray, self.get_input_port(1).Eval(context))
        q, v = x[: self.nq], x[self.nq :]

        plant_context = self.context
        self.plant.SetPositions(plant_context, q)
        self.plant.SetVelocities(plant_context, v)
        self.forces.SetZero()
        self.plant.CalcForceElementsContribution(plant_context, self.forces)
        tau = self.plant.CalcInverseDynamics(plant_context, self.zero_vdot, self.forces)

        output.SetFromVector(
            tau[self.arm_v]
            + self.kp * (q_desired - q[self.arm_q])
            - self.kd * v[self.arm_v]
        )
