"""Locates the default stereo calibration file for singray_stereo_backend.py.

The calibration itself is produced entirely outside this project by
~/Desktop/singray/pythonTest/calibrationTooling/ (capture.py to gather
checkerboard image pairs, calibrate.py to compute the fisheye stereo
calibration) - see that folder's own comments for the calibration procedure.
This module only knows where to find the resulting .npz file.
"""
from __future__ import annotations

import os

_DEFAULT_CALIBRATION_PATHS = [
    os.path.expanduser("~/Desktop/singray/pythonTest/calibrationTooling/stereo_calibration.npz"),
]


def find_default_calibration_path() -> str | None:
    """Returns SINGRAY_STEREO_CALIBRATION_PATH if set, else the first default
    path that exists, else None if no calibration file can be found."""
    override = os.environ.get("SINGRAY_STEREO_CALIBRATION_PATH")
    candidates = [override] if override else _DEFAULT_CALIBRATION_PATHS
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None
