"""Main application window: connects to the camera on startup and continuously
streams live depth frames into the noise-measurement panel."""
from __future__ import annotations

import logging
import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QApplication, QDockWidget, QMainWindow, QMessageBox

from nion_app.camera.backend import CameraBackend
from nion_app.camera.connection_flow import ConnectionStepFailed, run_guided_connection
from nion_app.app.camera_settings_panel import CameraSettingsPanel
from nion_app.app.noise_panel import NoisePanel
from nion_app.viewer.depth_view import DepthView

logger = logging.getLogger(__name__)

_FALLBACK_FRAME_INTERVAL_MS = 100  # used only until the camera's real frame rate is known

# Same rationale as NoisePanel's IR-preview throttle: while a recording is in
# progress, the depth preview's per-frame colorization competes with the
# recorder for main-thread time at exactly the frame rate that matters most.
_DEPTH_PREVIEW_THROTTLE_INTERVAL_S = 0.2


class MainWindow(QMainWindow):
    frame_captured = Signal(object)  # Frame, emitted once per acquired frame

    def __init__(self, backend: CameraBackend) -> None:
        super().__init__()
        self.setWindowTitle("Nion Noise Measurement")

        self._backend = backend
        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(_FALLBACK_FRAME_INTERVAL_MS)
        self._frame_timer.timeout.connect(self._update_frame)
        self._last_depth_preview_time = 0.0

        self._noise_panel = NoisePanel(backend, self)
        self.frame_captured.connect(self._noise_panel.on_frame)
        self.setCentralWidget(self._noise_panel)

        self._depth_view = DepthView(self)
        self.frame_captured.connect(self._on_frame_for_depth_preview)
        depth_dock = QDockWidget("Depth Preview", self)
        depth_dock.setWidget(self._depth_view)
        self.addDockWidget(Qt.RightDockWidgetArea, depth_dock)

        self._camera_settings_panel = CameraSettingsPanel(backend, self)
        self._camera_settings_panel.frame_rate_changed.connect(self._on_camera_frame_rate_changed)
        settings_dock = QDockWidget("Camera Settings", self)
        settings_dock.setWidget(self._camera_settings_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, settings_dock)

        self.resize(1300, 750)

    def start(self, configuration_name: str = "Default") -> bool:
        """Connects to the camera and starts live acquisition. Returns False
        if the camera couldn't be connected - the window is still fully
        usable in that case (recording just stays unavailable), so callers
        should show() it regardless rather than treating this as fatal."""
        try:
            run_guided_connection(self._backend, configuration_name=configuration_name)
        except ConnectionStepFailed as exc:
            logger.error("Connection failed at step '%s': %s", exc.step, exc.reason)
            QMessageBox.warning(
                self,
                "Camera connection failed",
                f"{exc.step}: {exc.reason}\n\n"
                "Continuing without a live camera - you can still load and "
                "view a previously saved dataset.",
            )
            self._noise_panel.set_camera_unavailable(f"{exc.step}: {exc.reason}")
            return False

        # Emits frame_rate_changed with the camera's actual current rate,
        # which _on_camera_frame_rate_changed uses to set the timer's real
        # interval before it's started below - so recording starts already
        # sampling at the camera's configured rate, not the fallback one.
        self._camera_settings_panel.refresh_from_backend()
        self._backend.start_acquisition()
        self._frame_timer.start()
        logger.info(
            "Live view started (refresh every %d ms)", self._frame_timer.interval()
        )
        return True

    def _on_camera_frame_rate_changed(self, frame_rate_fps: float) -> None:
        """Keeps the acquisition-polling timer in step with the camera's
        actual frame rate, so recorded samples land at that rate instead of
        a fixed, possibly mismatched, polling interval."""
        if frame_rate_fps <= 0:
            logger.warning(
                "Ignoring non-positive camera frame rate %.2f fps; keeping %d ms interval",
                frame_rate_fps, self._frame_timer.interval(),
            )
            return
        interval_ms = max(1, round(1000.0 / frame_rate_fps))
        self._frame_timer.setInterval(interval_ms)
        logger.info(
            "Camera frame rate is %.2f fps; polling interval set to %d ms",
            frame_rate_fps, interval_ms,
        )

    def _on_frame_for_depth_preview(self, frame) -> None:
        if self._noise_panel.is_recording:
            now = time.monotonic()
            if now - self._last_depth_preview_time < _DEPTH_PREVIEW_THROTTLE_INTERVAL_S:
                return
            self._last_depth_preview_time = now
        self._depth_view.on_frame(frame)

    def _update_frame(self) -> None:
        try:
            frame = self._backend.get_frame(timeout_ms=500)
        except Exception:
            logger.exception("Failed to acquire frame; skipping this refresh")
            return

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
