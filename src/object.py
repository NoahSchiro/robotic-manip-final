"""The pushed object: a T-shaped block whose inertial parameters are knobs.

Composite SDFs written here place the body frame at the center of the
footprint's bounding-box *bottom* face and require exactly touching
parts, so uniform-density volumes compose exactly.  Contact is compliant
hydroelastic on every part. Planar footprint geometry lives in geometry.py
"""

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from pydrake.all import MultibodyPlant, Parser, RigidTransform, RotationMatrix

from . import ASSET_DIR
from .geometry import SurfaceNormal, contains_disk, footprint_center, sample_outline


@dataclass(frozen=True)
class Rod:
    """A heavy thin cylinder welded into a composite at body-frame (x, y),
    z-centered in its height, to shift the composite's COM."""

    x: float
    y: float
    mass: float
    radius: float = 0.004


class BoxComposite:
    """A prism built from exactly-touching axis-aligned boxes ("parts").

    The shared writer behind the pushed object and the table: name,
    parts, height, mass, color, friction, and an optional welded-in rod
    are per-instance knobs that can be tuned.
    """

    body_box = "base_link"
    hydroelastic_modulus = 1.0e6
    rod_marker_color = (0.85, 0.15, 0.12)
    rod_marker_radius = 0.008
    rod_marker_thickness = 0.0005

    def __init__(
        self,
        name: str,
        parts: list[tuple[float, float, float, float]],
        height: float,
        mass: float,
        color: tuple[float, float, float],
        friction: float,
        rod: Rod | None = None,
    ) -> None:
        self.asset_dir = ASSET_DIR
        self.name = name
        self.parts = parts
        self.height = height
        self.mass = mass
        self.color = color
        self.friction = friction
        self.rod = rod
        if rod is not None and not contains_disk(self.parts, rod.x, rod.y, rod.radius):
            raise ValueError(
                f"rod at ({rod.x}, {rod.y}) pokes outside the parts' outline"
            )

    @property
    def total_mass(self) -> float:
        """Get the composite's mass, including any welded rod."""
        return self.mass + (self.rod.mass if self.rod is not None else 0.0)

    def compound_inertia(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the composite's COM plus rotational inertia: uniform-density
        exactly-touching boxes, plus the optional welded rod."""
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

        # Box inertia per part, accumulated about the body's own COM.
        # sum() starts at int 0, so mypy widens the element type to float;
        # parts is never empty, so the result is always the ndarray.
        com = (
            cast(np.ndarray, sum(m * c for m, c in zip(masses, part_coms))) / self.mass
        )
        inertia = np.zeros((3, 3))
        for (sx, sy, cx, cy), m, c in zip(self.parts, masses, part_coms):
            inertia += m * np.diag(
                [
                    (sy**2 + self.height**2) / 12,
                    (sx**2 + self.height**2) / 12,
                    (sx**2 + sy**2) / 12,
                ]
            )
            inertia += self._shift_inertia(m, c - com)

        if self.rod is None:
            return com, inertia

        # Adding the rod now. Its COM shares the boxes' z (half height), then
        # both bodies' inertias move onto the combined COM.
        fx, fy = footprint_center(self.parts)
        rod_com = np.array([self.rod.x + fx, self.rod.y + fy, self.height / 2])
        r, length = self.rod.radius, self.height
        rod_inertia = self.rod.mass * np.diag(
            [
                (3 * r**2 + length**2) / 12,
                (3 * r**2 + length**2) / 12,
                r**2 / 2,
            ]
        )
        total_mass = self.total_mass
        block_com = com
        com = (self.mass * block_com + self.rod.mass * rod_com) / total_mass
        inertia += self._shift_inertia(self.mass, block_com - com)
        inertia += rod_inertia + self._shift_inertia(self.rod.mass, rod_com - com)
        return com, inertia

    @staticmethod
    def _shift_inertia(mass: float, offset: np.ndarray) -> np.ndarray:
        """Shift a body's inertia to a point displaced by `offset` from its
        COM (parallel-axis theorem)."""
        return cast(
            np.ndarray, mass * (offset @ offset * np.eye(3) - np.outer(offset, offset))
        )

    def body_part_centers(self) -> list[np.ndarray]:
        """Get each part's box center in the body frame, which sits at the
        footprint's bottom-face center."""
        center_xy = footprint_center(self.parts)
        return [
            np.array([cx, cy, self.height / 2]) - np.r_[center_xy, 0.0]
            for sx, sy, cx, cy in self.parts
        ]

    def surface_normals(self, density: float) -> list[SurfaceNormal]:
        """Sample the footprint outline into a finite, deterministic set of
        push contacts."""
        return sample_outline(self.parts, self.height, density)

    def com_in_body_frame(self) -> np.ndarray:
        """Get the composite's COM in the body frame (footprint bottom
        center)."""
        com, _ = self.compound_inertia()
        return cast(np.ndarray, com - np.r_[footprint_center(self.parts), 0.0])

    def sdf(self) -> str:
        """Render the prism's model SDF text."""
        _, inertia = self.compound_inertia()
        com_body = self.com_in_body_frame()
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

        marker = ""
        if self.rod is not None:
            mr, mg, mb = self.rod_marker_color
            z = self.height + self.rod_marker_thickness / 2
            marker = f"""
                    <visual name="rod_marker">
                      <pose>{self.rod.x:.6g} {self.rod.y:.6g} {z:.6g} 0 0 0</pose>
                      <geometry><cylinder><radius>{self.rod_marker_radius}</radius><length>{self.rod_marker_thickness}</length></cylinder></geometry>
                      <material><diffuse>{mr} {mg} {mb} 1</diffuse></material>
                    </visual>"""

        return f"""
            <?xml version="1.0"?>
              <sdf version="1.7">
                <model name="{self.name}">
                  <link name="{self.body_box}">
                    <inertial>
                      <pose>{com_body[0]:.6g} {com_body[1]:.6g} {com_body[2]:.6g} 0 0 0</pose>
                      <mass>{self.total_mass}</mass>
                      <inertia>
                        <ixx>{ixx:.6g}</ixx><ixy>{ixy:.6g}</ixy><ixz>{ixz:.6g}</ixz>
                        <iyy>{iyy:.6g}</iyy><iyz>{iyz:.6g}</iyz>
                        <izz>{izz:.6g}</izz>
                      </inertia>
                    </inertial>
                    {bodies}{marker}
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

    def __init__(
        self,
        rng: np.random.Generator,
        rod: Rod,
        friction: float = 0.3,
    ):
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
            rod=rod,
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
    # Parsed from the string, not a file under assets/: parallel runs with
    # different objects would race on the one shared path.
    parser.AddModelsFromString(obj.sdf(), "sdf")


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
    parser.AddModelsFromString(table.sdf(), "sdf")
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName(table.body_box, plant.GetModelInstanceByName(table.name)),
        RigidTransform(np.array([0.0, 0.0, -table.height])),
    )
