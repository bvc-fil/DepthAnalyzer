#!/usr/bin/env python3
"""Launches the live depth viewer application against the real camera."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nion_app.qt_bootstrap import ensure_compatible_qt_platform

ensure_compatible_qt_platform()  # must run before any PySide6 import

from PySide6.QtWidgets import QApplication

from nion_app.logging_setup import configure_logging
from nion_app.camera.ids_backend import IdsPeakBackend
from nion_app.app.main_window import MainWindow


def main() -> int:
    configure_logging()
    app = QApplication(sys.argv)

    backend = IdsPeakBackend()
    window = MainWindow(backend)
    if not window.start():
        return 1
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
