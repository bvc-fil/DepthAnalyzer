#!/usr/bin/env python3
"""Runs the full app against the real camera, simulates dragging a ROI on the
live IR image, runs a short recording, and opens the results view - proving
the noise-measurement workflow end-to-end without manual interaction."""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nion_app.qt_bootstrap import ensure_compatible_qt_platform

ensure_compatible_qt_platform()

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from nion_app.logging_setup import configure_logging
from nion_app.camera.ids_backend import IdsPeakBackend
from nion_app.app.main_window import MainWindow

logger = logging.getLogger(__name__)


def _synthesize_mouse_event(event_type, pos: QPoint) -> QMouseEvent:
    return QMouseEvent(
        event_type, QPointF(pos), QPointF(pos), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier
    )


def main() -> int:
    configure_logging()
    app = QApplication.instance() or QApplication(sys.argv)

    backend = IdsPeakBackend()
    window = MainWindow(backend)
    if not window.start():
        return 1
    window.show()

    # Let a few frames flow so the IR view has an image to drag over.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)

    from nion_app.app.noise_panel import NoisePanel

    noise_panel = window.findChild(NoisePanel)
    assert noise_panel is not None, "NoisePanel not found in window"
    ir_view = noise_panel._ir_view
    assert ir_view._image_size is not None, "IR view never received an image"
    logger.info("IR view has an image: %s", ir_view._image_size)

    # Simulate a drag from (50,50) to (250,200) in the IR view's own widget coords.
    start, end = QPoint(50, 50), QPoint(250, 200)
    ir_view.mousePressEvent(_synthesize_mouse_event(QEvent.MouseButtonPress, start))
    ir_view.mouseMoveEvent(_synthesize_mouse_event(QEvent.MouseMove, end))
    ir_view.mouseReleaseEvent(_synthesize_mouse_event(QEvent.MouseButtonRelease, end))
    app.processEvents()

    roi = ir_view.current_roi()
    assert roi is not None, "ROI was not set by the simulated drag"
    logger.info("Simulated ROI selection: %s", roi)
    assert noise_panel._record_button.isEnabled()

    # Short recording.
    noise_panel._duration_box.setValue(1.5)
    noise_panel._start_recording()

    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline and noise_panel._latest_recording is None:
        app.processEvents()
        time.sleep(0.02)

    recording = noise_panel._latest_recording
    assert recording is not None, "Recording never completed"
    logger.info(
        "Recording complete: frames=%d samples=%d roi=%s",
        recording.frame_count, len(recording.depth_mm), recording.roi,
    )
    assert recording.frame_count > 1
    assert len(recording.depth_mm) == recording.frame_count * roi.width * roi.height

    import numpy as np

    mean_grid, std_grid = recording.per_pixel_mean_std()
    valid_counts = recording.valid_sample_count_grid()
    logger.info(
        "Depth stats over ROI: mean range=[%.1f, %.1f]mm std range=[%.3f, %.3f]mm "
        "(%d/%d pixels had >=1 valid sample)",
        np.nanmin(mean_grid), np.nanmax(mean_grid), np.nanmin(std_grid), np.nanmax(std_grid),
        int((valid_counts > 0).sum()), valid_counts.size,
    )

    noise_panel._show_results()
    for _ in range(10):
        app.processEvents()
        time.sleep(0.05)
    results_view = noise_panel._results_windows[-1]
    screenshot_path = Path(__file__).resolve().parent.parent / "logs" / "noise_results.png"
    results_view.grab().save(str(screenshot_path))
    logger.info("Saved results screenshot to %s", screenshot_path)

    window.close()
    print("ALL NOISE PANEL INTEGRATION TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
