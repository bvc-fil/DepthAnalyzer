"""Qt platform plugin auto-selection.

On this machine (and likely others with the same setup), VTK's OpenGL render
window creation fails under Qt's native Wayland backend (BadWindow/GLX
errors) but works fine through XWayland. Force the xcb platform when running
under Wayland and the user hasn't already chosen a platform explicitly, so
the app works without manually exporting QT_QPA_PLATFORM every time.

Must be called before any PySide6 import - Qt reads this environment
variable when its platform plugin loads.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def ensure_compatible_qt_platform() -> None:
    if os.environ.get("QT_QPA_PLATFORM"):
        return  # respect an explicit user choice
    if os.environ.get("WAYLAND_DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        logger.info("Wayland session detected; forcing QT_QPA_PLATFORM=xcb for VTK compatibility")
