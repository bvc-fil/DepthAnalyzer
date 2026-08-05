"""Depth-image -> XYZ point cloud reconstruction.

The Nion's GenICam node map only exposes Scan3dOutputMode = "UncalibratedC":
per-pixel radial/Z distance, with no on-device path to a calibrated XYZ
(Coord3D_ABC32f) output. Its 208-byte "LensCalibrationData" file (readable via
GenICam FileAccess) is a proprietary binary blob - ids_peak_icv's calibration
and serialization modules only support checkerboard-plate calibration and
JSON archives respectively, neither of which decodes it. Until IDS documents
that format (or a future firmware/SDK exposes a calibrated output mode), this
module reconstructs points with a standard pinhole back-projection using
nominal intrinsics derived from the Nion's published field of view
(71 deg horizontal x 57 deg vertical at 1280x960, see
https://en.ids-imaging.com/ueye-3d-camera-nion.html). This is an
approximation shared across all units of this model, not a per-unit factory
calibration - expect some geometric error, growing with distance from the
sensor. Swap in a real per-unit calibration here once one is available.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NION_HORIZONTAL_FOV_DEG = 71.0
NION_VERTICAL_FOV_DEG = 57.0


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov_degrees(
        cls, horizontal_fov_deg: float, vertical_fov_deg: float, width: int, height: int
    ) -> "CameraIntrinsics":
        fx = (width / 2) / math.tan(math.radians(horizontal_fov_deg) / 2)
        fy = (height / 2) / math.tan(math.radians(vertical_fov_deg) / 2)
        return cls(fx=fx, fy=fy, cx=width / 2, cy=height / 2, width=width, height=height)


def nion_nominal_intrinsics(width: int, height: int) -> CameraIntrinsics:
    return CameraIntrinsics.from_fov_degrees(
        NION_HORIZONTAL_FOV_DEG, NION_VERTICAL_FOV_DEG, width, height
    )


def depth_to_point_cloud(
    depth_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
    intensity: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    min_confidence: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Back-project a per-pixel depth image into an (N, 3) XYZ point cloud in
    millimeters, using the pinhole camera model. Invalid pixels (depth == 0,
    the Nion's Scan3dInvalidDataValue) and, if requested, low-confidence
    pixels are dropped.

    Returns (points_xyz_mm, point_intensity) where point_intensity is None if
    intensity was not provided.
    """
    height, width = depth_mm.shape
    valid = depth_mm > 0
    if min_confidence is not None and confidence is not None:
        valid &= confidence >= min_confidence

    rows, cols = np.nonzero(valid)
    z = depth_mm[rows, cols].astype(np.float64)
    x = (cols - intrinsics.cx) * z / intrinsics.fx
    y = (rows - intrinsics.cy) * z / intrinsics.fy
    points = np.column_stack([x, y, z]).astype(np.float32)

    point_intensity = intensity[rows, cols] if intensity is not None else None
    return points, point_intensity
