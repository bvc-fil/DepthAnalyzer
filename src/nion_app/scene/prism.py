"""Rectangular prism data model and the geometry glue between it, the
rendered mesh, and the interactive vtkBoxWidget handles.

vtkBoxWidget's output PolyData always orders its first 8 corner points as a
standard hexahedron - verified against a live vtkBoxWidget instance:
    0=(xmin,ymin,zmin) 1=(xmax,ymin,zmin) 2=(xmax,ymax,zmin) 3=(xmin,ymax,zmin)
    4=(xmin,ymin,zmax) 5=(xmax,ymin,zmax) 6=(xmax,ymax,zmax) 7=(xmin,ymax,zmax)
That correspondence is fixed regardless of how the user has rotated/moved/
resized the widget, so the edges pts[1]-pts[0], pts[3]-pts[0], pts[4]-pts[0]
always give the box's current local X/Y/Z axes (direction and length) in
world space - which is exactly what's needed to recover center, size and
rotation after an interactive edit.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field

import numpy as np


def euler_xyz_deg_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Intrinsic X-then-Y-then-Z rotation, degrees, returned as R = Rz @ Ry @ Rx."""
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def matrix_to_euler_xyz_deg(rotation: np.ndarray) -> tuple[float, float, float]:
    """Inverse of euler_xyz_deg_to_matrix; degenerate (gimbal-lock) case falls
    back to zero roll, which is an acceptable simplification for a UI display."""
    sy = -rotation[2, 0]
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    if abs(sy) < 0.9999999:
        rx = math.atan2(rotation[2, 1], rotation[2, 2])
        rz = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        rx = math.atan2(-rotation[1, 2], rotation[1, 1])
        rz = 0.0
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


@dataclass
class Prism:
    center_mm: np.ndarray  # (3,) world-space center
    size_mm: np.ndarray  # (3,) full extents along the prism's own x/y/z axes
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))  # columns = local axes
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""

    def __post_init__(self) -> None:
        self.center_mm = np.asarray(self.center_mm, dtype=np.float64)
        self.size_mm = np.asarray(self.size_mm, dtype=np.float64)
        self.rotation = np.asarray(self.rotation, dtype=np.float64)
        if not self.name:
            self.name = f"Prism {self.id}"

    def transform_matrix(self) -> np.ndarray:
        """4x4 matrix placing a unit-size box (centered at the origin, local
        axes aligned) into world space - for use as an actor's user_matrix."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.center_mm
        return matrix

    def local_bounds(self) -> tuple[float, float, float, float, float, float]:
        hx, hy, hz = self.size_mm / 2
        return (-hx, hx, -hy, hy, -hz, hz)

    def corners(self) -> np.ndarray:
        """(8, 3) world-space corner points, in vtkBoxWidget's corner order."""
        hx, hy, hz = self.size_mm / 2
        local_corners = np.array(
            [
                [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
            ]
        )
        return local_corners @ self.rotation.T + self.center_mm

    @classmethod
    def from_box_widget_points(cls, points: np.ndarray, id: str, name: str = "") -> "Prism":
        """Decode a Prism from the first 8 corner points of a vtkBoxWidget's
        output PolyData (see module docstring for the ordering guarantee)."""
        origin = points[0]
        x_vec, y_vec, z_vec = points[1] - origin, points[3] - origin, points[4] - origin
        size = np.array([np.linalg.norm(x_vec), np.linalg.norm(y_vec), np.linalg.norm(z_vec)])
        rotation = np.column_stack(
            [
                x_vec / size[0] if size[0] > 0 else [1, 0, 0],
                y_vec / size[1] if size[1] > 0 else [0, 1, 0],
                z_vec / size[2] if size[2] > 0 else [0, 0, 1],
            ]
        )
        center = points[:8].mean(axis=0)
        return cls(center_mm=center, size_mm=size, rotation=rotation, id=id, name=name)
