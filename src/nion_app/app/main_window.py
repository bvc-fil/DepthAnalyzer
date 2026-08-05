"""Main application window: connects to the camera on startup and continuously
streams live depth frames into the 3D point cloud viewer."""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QApplication, QDockWidget, QMainWindow, QMessageBox

from nion_app.camera.backend import CameraBackend
from nion_app.camera.connection_flow import ConnectionStepFailed, run_guided_connection
from nion_app.camera.point_cloud import depth_to_point_cloud, nion_nominal_intrinsics
from nion_app.app.noise_panel import NoisePanel
from nion_app.app.prism_panel import PrismPanel
from nion_app.viewer.prism_scene_view import PrismSceneView

logger = logging.getLogger(__name__)

_MIN_CONFIDENCE = 1
_FRAME_INTERVAL_MS = 100  # ~10 Hz refresh; the sensor itself runs up to 30 fps


class MainWindow(QMainWindow):
    frame_captured = Signal(object)  # Frame, emitted once per acquired frame

    def __init__(self, backend: CameraBackend) -> None:
        super().__init__()
        self.setWindowTitle("Nion Depth Viewer")

        self._backend = backend
        self._intrinsics = None
        self._frame_timer = QTimer(self)
        self._frame_timer.timeout.connect(self._update_frame)

        self._view = PrismSceneView(self)
        self.setCentralWidget(self._view)

        prism_dock = QDockWidget("Prisms", self)
        prism_dock.setWidget(PrismPanel(self._view, prism_dock))
        self.addDockWidget(Qt.RightDockWidgetArea, prism_dock)

        self._noise_panel = NoisePanel(self)
        self.frame_captured.connect(self._noise_panel.on_frame)
        noise_dock = QDockWidget("Noise Measurement", self)
        noise_dock.setWidget(self._noise_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, noise_dock)

        self.resize(1280, 800)

    def start(self, configuration_name: str = "Default") -> bool:
        try:
            run_guided_connection(self._backend, configuration_name=configuration_name)
        except ConnectionStepFailed as exc:
            logger.error("Connection failed at step '%s': %s", exc.step, exc.reason)
            QMessageBox.critical(self, "Camera connection failed", f"{exc.step}: {exc.reason}")
            return False

        self._backend.start_acquisition()
        self._frame_timer.start(_FRAME_INTERVAL_MS)
        logger.info("Live view started (refresh every %d ms)", _FRAME_INTERVAL_MS)
        return True

    def _update_frame(self) -> None:
        try:
            frame = self._backend.get_frame(timeout_ms=500)
        except Exception:
            logger.exception("Failed to acquire frame; skipping this refresh")
            return

        if self._intrinsics is None:
            height, width = frame.depth_mm.shape
            self._intrinsics = nion_nominal_intrinsics(width, height)

        points, point_intensity = depth_to_point_cloud(
            frame.depth_mm,
            self._intrinsics,
            intensity=frame.intensity,
            confidence=frame.confidence,
            min_confidence=_MIN_CONFIDENCE,
        )
        self._view.set_points(points, scalars=point_intensity)
        self.frame_captured.emit(frame)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._frame_timer.stop()
        # Each step runs in its own try/except: if stop_acquisition() raised
        # and disconnect() were skipped as a result, the native IDS peak
        # library (and whatever background resources it holds) would never
        # be released, leaving the process running after the window closes.
        try:
            self._backend.stop_acquisition()
        except Exception:
            logger.exception("Error stopping acquisition")
        try:
            self._backend.disconnect()
        except Exception:
            logger.exception("Error disconnecting camera backend")

        # Close any detached "Show Results" windows: they're separate
        # top-level windows, so leaving one open would stop Qt's
        # quitOnLastWindowClosed from ever firing.
        try:
            self._noise_panel.close_results_windows()
        except Exception:
            logger.exception("Error closing noise-results windows")

        # The pyvista/VTK view holds native render-window and OpenGL
        # resources (plus its own render_timer) that outlive this widget
        # unless explicitly closed - left alone, they keep the process
        # running in the background after the window disappears.
        try:
            self._view.shutdown()
        except Exception:
            logger.exception("Error shutting down the 3D view")

        # Each cleanup step above is independently guarded: one raising must
        # never prevent the steps after it - especially the app.quit() below,
        # since skipping it is exactly what left the process running in the
        # background instead of exiting when the window closed.
        super().closeEvent(event)

        # Belt-and-braces: guarantee the event loop actually exits (and the
        # process with it) regardless of any other window state.
        app = QApplication.instance()
        if app is not None:
            app.quit()
