#!/usr/bin/env python3
"""Launches the live depth viewer application against the real camera.

Which sensor to use is picked via --backend (or the NION_CAMERA_BACKEND env
var): "ids_peak" for the Nion ToF camera (default) or "realsense" for an
Intel RealSense D455.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nion_app.qt_bootstrap import ensure_compatible_qt_platform

ensure_compatible_qt_platform()  # must run before any PySide6 import

from PySide6.QtWidgets import QApplication

from nion_app.logging_setup import configure_logging
from nion_app.app.main_window import MainWindow

_BACKENDS = {"ids_peak", "realsense"}


def _make_backend(name: str):
    if name == "ids_peak":
        from nion_app.camera.ids_backend import IdsPeakBackend

        return IdsPeakBackend()
    if name == "realsense":
        from nion_app.camera.realsense_backend import RealSenseBackend

        return RealSenseBackend()
    raise ValueError(f"Unknown camera backend '{name}' (choose from {sorted(_BACKENDS)})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=sorted(_BACKENDS),
        default=os.environ.get("NION_CAMERA_BACKEND", "ids_peak"),
        help="Which camera backend to use (default: %(default)s)",
    )
    args = parser.parse_args()

    configure_logging()
    app = QApplication(sys.argv)

    backend = _make_backend(args.backend)
    window = MainWindow(backend)
    # start()'s return value is deliberately ignored here: even without a
    # live camera, the window still comes up so a previously saved dataset
    # can be loaded and viewed. start() has already told the user why via a
    # dialog and the noise panel's status label if the connection failed.
    window.start()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
