"""Shared helper for locating the vendor's `xvsdk` ctypes wrapper module -
used by both singray_backend.py (ToF) and singray_stereo_backend.py (fisheye
stereo), since neither the module's location nor its loading quirks differ
between the two backends."""
from __future__ import annotations

import os
import sys

from nion_app.camera.backend import CameraError

_DEFAULT_SDK_PYTHON_DIRS = [
    os.path.expanduser("~/Desktop/singray/sdk/StereoPRO/python"),
]


def import_xvsdk():
    """Locates and imports the vendor's `xvsdk` ctypes wrapper module."""
    if "xvsdk" in sys.modules:
        return sys.modules["xvsdk"]

    override = os.environ.get("SINGRAY_SDK_PYTHON_PATH")
    search_dirs = [override] if override else _DEFAULT_SDK_PYTHON_DIRS
    for directory in search_dirs:
        if directory and os.path.isfile(os.path.join(directory, "xvsdk.py")):
            if directory not in sys.path:
                sys.path.insert(0, directory)
            import xvsdk  # type: ignore

            return xvsdk

    raise CameraError(
        "Could not find the Singray Stereo PRO SDK's xvsdk.py. Set "
        "SINGRAY_SDK_PYTHON_PATH to the SDK's 'python' directory (containing "
        "xvsdk.py and its libxvisio-CInterface-wrapper.so dependency)."
    )
