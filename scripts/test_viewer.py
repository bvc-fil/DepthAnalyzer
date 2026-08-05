#!/usr/bin/env python3
"""Captures one real frame from the camera, builds a point cloud, and renders
it in the PointCloudView widget - screenshotted for visual verification."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PySide6.QtWidgets import QApplication

from nion_app.logging_setup import configure_logging
from nion_app.camera.connection_flow import run_guided_connection
from nion_app.camera.ids_backend import IdsPeakBackend
from nion_app.camera.point_cloud import depth_to_point_cloud, nion_nominal_intrinsics
from nion_app.viewer.point_cloud_view import PointCloudView

logger = logging.getLogger(__name__)


def main() -> int:
    configure_logging()
    app = QApplication.instance() or QApplication([])

    backend = IdsPeakBackend()
    run_guided_connection(backend)
    backend.start_acquisition()
    frame = backend.get_frame()
    backend.stop_acquisition()
    backend.disconnect()

    intrinsics = nion_nominal_intrinsics(frame.depth_mm.shape[1], frame.depth_mm.shape[0])
    points, point_intensity = depth_to_point_cloud(
        frame.depth_mm, intrinsics, intensity=frame.intensity, confidence=frame.confidence,
        min_confidence=1,
    )
    logger.info("Reconstructed %d points from %d total pixels", len(points), frame.depth_mm.size)

    view = PointCloudView()
    view.resize(1024, 768)
    view.set_points(points, scalars=point_intensity)
    view.show()

    for _ in range(20):
        app.processEvents()

    screenshot_path = Path(__file__).resolve().parent.parent / "logs" / "viewer_test.png"
    view.plotter.screenshot(str(screenshot_path))
    logger.info("Saved screenshot to %s", screenshot_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
