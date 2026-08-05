#!/usr/bin/env python3
"""Runs the live app for a few seconds to prove continuous frame streaming
works (not just a single capture), then exits automatically. Uses manual
event-loop pumping instead of app.exec() so this script is guaranteed to
terminate on its own for automated verification."""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PySide6.QtWidgets import QApplication

from nion_app.logging_setup import configure_logging
from nion_app.camera.ids_backend import IdsPeakBackend
from nion_app.app.main_window import MainWindow

logger = logging.getLogger(__name__)


def main() -> int:
    configure_logging()
    app = QApplication.instance() or QApplication(sys.argv)

    backend = IdsPeakBackend()
    window = MainWindow(backend)

    frame_count = 0
    original_update = MainWindow._update_frame

    def counting_update(self):
        nonlocal frame_count
        original_update(self)
        frame_count += 1

    MainWindow._update_frame = counting_update

    if not window.start():
        return 1
    window.show()

    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)

    logger.info("Processed %d live frames over the 4s test window", frame_count)

    from nion_app.scene.prism import Prism
    import numpy as np

    window._view.add_prism(Prism(center_mm=np.array([0.0, 0.0, 900.0]), size_mm=np.array([150.0, 150.0, 150.0])))
    app.processEvents()
    logger.info("Added a test prism on top of the live point cloud")

    screenshot_path = Path(__file__).resolve().parent.parent / "logs" / "live_app_test.png"
    window._view.plotter.screenshot(str(screenshot_path))
    logger.info("Saved final screenshot to %s", screenshot_path)

    window.close()
    return 0 if frame_count > 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
