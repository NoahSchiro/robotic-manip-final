"""The pushed object: a T-shaped block whose inertial parameters are knobs.

Composite SDFs written here place the body frame at the center of the
footprint's bounding-box *bottom* face and require exactly touching
parts, so uniform-density volumes compose exactly.  Contact is compliant
hydroelastic on every part. BoxComposite.sdf doubles as the project's
shared box-prism SDF renderer.
"""

import itertools
from pathlib import Path
from typing import cast

import numpy as np
from pydrake.all import MultibodyPlant, Parser, RigidTransform, RotationMatrix

from . import ASSET_DIR


class BoxComposite:
    """A prism built from exactly-touching axis-aligned boxes ("parts").

    The shared writer behind the pushed object and the table: name,
    parts, height, mass, color, and friction are per-instance knobs that
    can be tuned.
    """

    body_box = "base_link"
    hydroelastic_modulus = 1.0e6

    def __init__(
        self,
        name: str,
        parts: list[tuple[float, float, float, float]],
        height: float,
        mass: float,
        color: tuple[float, float, float],
        friction: float,
    ) -> None:
        self.asset_dir = ASSET_DIR
        self.name = name
        self.parts = parts
        self.height = height
        self.mass = mass
        self.color = color
        self.friction = friction

    def compound_inertia(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the compound COM plus rotational inertia of exactly-touching
        boxes under uniform density."""
        for a, b in itertools.combinations(self.parts, 2):
            if (
                abs(a[2] - b[2]) < (a[0] + b[0]) / 2
                and abs(a[3] - b[3]) < (a[1] + b[1]) / 2
            ):
                raise ValueError("parts overlap; the mass model would double-count")

        volumes = [sx * sy * self.height for sx, sy, cx, cy in self.parts]
        density = self.mass / sum(volumes)
        masses = [density * v for v in volumes]
        part_coms = [
            np.array([cx, cy, self.height / 2]) for sx, sy, cx, cy in self.parts
        ]

        # Box inertia per part, then the parallel-axis theorem into the COM.
        # sum() starts at int 0, so mypy widens the element type to float;
        # parts is never empty, so the result is always the ndarray.
        com = cast(
            np.ndarray, sum(m * c for m, c in zip(masses, part_coms))
        ) / self.mass
        inertia = np.zeros((3, 3))
        for (sx, sy, cx, cy), m, c in zip(self.parts, masses, part_coms):
            inertia += m * np.diag(
                [
                    (sy**2 + self.height**2) / 12,
                    (sx**2 + self.height**2) / 12,
                    (sx**2 + sy**2) / 12,
                ]
            )
            d = c - com
            inertia += m * (d @ d * np.eye(3) - np.outer(d, d))
        return com, inertia

    def footprint_center(self) -> np.ndarray:
        """Get the xy center of the parts' bounding box."""
        lo_x = min(cx - sx / 2 for sx, sy, cx, cy in self.parts)
        hi_x = max(cx + sx / 2 for sx, sy, cx, cy in self.parts)
        lo_y = min(cy - sy / 2 for sx, sy, cx, cy in self.parts)
        hi_y = max(cy + sy / 2 for sx, sy, cx, cy in self.parts)
        return np.array([(lo_x + hi_x) / 2, (lo_y + hi_y) / 2])

    def body_part_centers(self) -> list[np.ndarray]:
        """Get each part's box center in the body frame, which sits at the
        footprint's bottom-face center."""
        center_xy = self.footprint_center()
        return [
            np.array([cx, cy, self.height / 2]) - np.r_[center_xy, 0.0]
            for sx, sy, cx, cy in self.parts
        ]

    def sdf(self) -> str:
        """Render the prism's model SDF text."""
        com, inertia = self.compound_inertia()
        com_body = com - np.r_[self.footprint_center(), 0.0]
        ixx, iyy, izz = inertia[0, 0], inertia[1, 1], inertia[2, 2]
        ixy, ixz, iyz = inertia[0, 1], inertia[0, 2], inertia[1, 2]

        r, g, b = self.color
        boxes = []
        for i, (part, center) in enumerate(zip(self.parts, self.body_part_centers())):
            sx, sy = part[:2]
            pose = " ".join(f"{v:.6g}" for v in [*center, 0.0, 0.0, 0.0])
            box = f"<box><size>{sx} {sy} {self.height}</size></box>"
            boxes.append(
                f"""<collision name="collision{i}">
                      <pose>{pose}</pose>
                      <geometry>{box}</geometry>
                      <drake:proximity_properties>
                        <drake:compliant_hydroelastic/>
                        <drake:hydroelastic_modulus>{self.hydroelastic_modulus}</drake:hydroelastic_modulus>
                        <drake:mu_static>{self.friction}</drake:mu_static>
                        <drake:mu_dynamic>{self.friction}</drake:mu_dynamic>
                      </drake:proximity_properties>
                    </collision>
                    <visual name="visual{i}">
                      <pose>{pose}</pose>
                      <geometry>{box}</geometry>
                      <material><diffuse>{r} {g} {b} 1</diffuse></material>
                    </visual>"""
            )
        bodies = "\n".join(boxes)

        return f"""
            <?xml version="1.0"?>
              <sdf version="1.7">
                <model name="{self.name}">
                  <link name="{self.body_box}">
                    <inertial>
                      <pose>{com_body[0]:.6g} {com_body[1]:.6g} {com_body[2]:.6g} 0 0 0</pose>
                      <mass>{self.mass}</mass>
                      <inertia>
                        <ixx>{ixx:.6g}</ixx><ixy>{ixy:.6g}</ixy><ixz>{ixz:.6g}</ixz>
                        <iyy>{iyy:.6g}</iyy><iyz>{iyz:.6g}</iyz>
                        <izz>{izz:.6g}</izz>
                      </inertia>
                    </inertial>
                    {bodies}
                  </link>
                </model>
              </sdf>"""

    def write_sdf(self) -> Path:
        """Write the prism's SDF into the assets directory, returning its path."""
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        path = self.asset_dir / f"{self.name}.sdf"
        path.write_text(self.sdf())
        return path


class Object(BoxComposite):
    """The pushed object: a T-shaped prism built from two exactly-touching
    boxes, a free body whose pose gets set per-run and which must be pushed
    to a random goal orientation."""

    pose_center = (0.55, 0.0)
    pose_radius = 0.10

    def __init__(self, rng: np.random.Generator, friction: float = 0.3):
        super().__init__(
            name="object",
            parts=[
                # (size_x, size_y, center_x, center_y)
                (0.05, 0.14, -0.045, 0.0),  # crossbar
                (0.09, 0.05, 0.025, 0.0),  # stem
            ],
            height=0.03,
            mass=0.4,
            color=(0.20, 0.45, 0.80),
            friction=friction,
        )
        rho_start = self.pose_radius * np.sqrt(rng.uniform())
        theta_start = rng.uniform(0.0, 2.0 * np.pi)
        # x, y, yaw (rad), resting on the table top
        self.pose = (
            self.pose_center[0] + rho_start * np.cos(theta_start),
            self.pose_center[1] + rho_start * np.sin(theta_start),
            rng.uniform(0.0, 2.0 * np.pi),
        )
        rho_goal = self.pose_radius * np.sqrt(rng.uniform())
        theta_goal = rng.uniform(0.0, 2.0 * np.pi)
        self.goal = (
            self.pose_center[0] + rho_goal * np.cos(theta_goal),
            self.pose_center[1] + rho_goal * np.sin(theta_goal),
            rng.uniform(0.0, 2.0 * np.pi),
        )

    def x_wo(self) -> RigidTransform:
        """Get the transformation from world origin to object."""
        return RigidTransform(
            RotationMatrix.MakeZRotation(self.pose[2]),
            np.array([self.pose[0], self.pose[1], 0.0]),
        )

    def x_wg(self) -> RigidTransform:
        """Get the goal transformation from world origin to object: its start
        position with the goal yaw."""
        return RigidTransform(
            RotationMatrix.MakeZRotation(self.goal[2]),
            np.array([self.goal[0], self.goal[1], 0.0]),
        )


def add_object(obj: Object, parser: Parser) -> None:
    """Add the block into `plant` as a free body."""
    parser.AddModelsFromUrl(obj.write_sdf().as_uri())


class Table(BoxComposite):
    """The scene's table: one big box whose top surface is the work plane."""

    def __init__(self, friction: float = 0.3):
        super().__init__(
            name="table",
            parts=[(2.0, 2.0, 0.0, 0.0)],
            height=0.1,
            mass=20.0,
            color=(0.55, 0.50, 0.45),
            friction=friction,
        )


def add_table(table: Table, plant: MultibodyPlant, parser: Parser) -> None:
    """Parse the table into `plant`, welded down with its top surface at z = 0."""
    parser.AddModelsFromUrl(table.write_sdf().as_uri())
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName(table.body_box, plant.GetModelInstanceByName(table.name)),
        RigidTransform(np.array([0.0, 0.0, -table.height])),
    )
