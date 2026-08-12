"""Panel driving the outline's noise-measurement workflow: pick a duration,
drag a ROI on the live IR image, record, then inspect per-pixel histograms
and standard deviations."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nion_app.app.noise_results_view import NoiseResultsView
from nion_app.camera.backend import CameraBackend, CameraError, Frame
from nion_app.camera.noise_recording import (
    NoiseRecorder,
    NoiseRecording,
    compute_recording_stats,
)
from nion_app.viewer.ir_roi_view import IrRoiView

logger = logging.getLogger(__name__)

_PROGRESS_TIMER_MS = 200

# While recording, every acquired frame must still reach the recorder at the
# camera's full configured rate - but redrawing the IR preview at that same
# rate (a rate the ROI is already locked and nobody is watching for changes)
# competes with that for the same main thread. Throttling it to ~5Hz keeps
# the preview live without eating into the budget the recorder needs to keep
# up with a high frame rate.
_PREVIEW_THROTTLE_INTERVAL_S = 0.2


class _StatsWorker(QObject):
    """Runs compute_recording_stats() on a worker QThread - that call alone
    can take seconds on a large recording (see max_pixel_histogram_count),
    and running it on the Qt main thread blocks event processing for that
    whole time, which is what makes the desktop's window manager decide the
    app has hung and show its own "not responding" dialog."""

    succeeded = Signal(object, object)  # (NoiseRecording, RecordingStats)
    failed = Signal(str)

    def __init__(self, recording: NoiseRecording) -> None:
        super().__init__()
        self._recording = recording

    def run(self) -> None:
        try:
            stats = compute_recording_stats(self._recording)
        except Exception as exc:
            logger.exception("Failed to compute recording statistics")
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(self._recording, stats)


class NoisePanel(QWidget):
    def __init__(self, backend: CameraBackend, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._backend = backend
        self._recorder: NoiseRecorder | None = None
        self._latest_recording: NoiseRecording | None = None
        # The file self._latest_recording currently corresponds to, if any -
        # set on save/load, cleared when a fresh recording finishes (it isn't
        # any file's contents yet). Used to title the results window.
        self._current_dataset_path: str | None = None
        self._last_preview_update_time = 0.0
        self._results_windows: list[NoiseResultsView] = []  # keep references alive
        self._stats_thread: QThread | None = None
        self._stats_worker: _StatsWorker | None = None

        self._ir_view = IrRoiView()
        self._ir_view.roi_selected.connect(self._on_roi_selected)

        self._duration_box = QDoubleSpinBox()
        self._duration_box.setRange(0.5, 3600.0)
        self._duration_box.setValue(5.0)
        self._duration_box.setSuffix(" s")

        self._record_button = QPushButton("Start Recording")
        self._record_button.setEnabled(False)
        self._record_button.clicked.connect(self._start_recording)

        self._results_button = QPushButton("Show Results")
        self._results_button.setEnabled(False)
        self._results_button.clicked.connect(self._show_results)

        self._save_button = QPushButton("Save Dataset...")
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self._save_dataset)

        self._load_button = QPushButton("Load Dataset...")
        self._load_button.clicked.connect(self._load_dataset)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._status_label = QLabel(
            "Drag a rectangle on the IR image to select a region of interest."
        )

        controls = QFormLayout()
        controls.addRow("Duration", self._duration_box)

        buttons = QHBoxLayout()
        buttons.addWidget(self._record_button)
        buttons.addWidget(self._results_button)
        buttons.addWidget(self._save_button)
        buttons.addWidget(self._load_button)

        group = QGroupBox("Noise Measurement")
        group_layout = QVBoxLayout(group)
        group_layout.addWidget(self._ir_view)
        group_layout.addWidget(self._status_label)
        group_layout.addLayout(controls)
        group_layout.addLayout(buttons)
        group_layout.addWidget(self._progress)

        layout = QVBoxLayout(self)
        layout.addWidget(group)

        self._progress_timer = QTimer(self)
        self._progress_timer.timeout.connect(self._update_progress)

    def set_camera_unavailable(self, reason: str) -> None:
        """Called by MainWindow when the camera couldn't be connected, so the
        panel doesn't just sit showing its normal "drag a ROI" prompt over a
        live view that will never arrive - recording stays unavailable, but
        loading and viewing a previously saved dataset still works fine."""
        self._status_label.setText(
            f"No camera connected ({reason}). Recording is unavailable, but "
            "you can still load and view a saved dataset below."
        )

    def _on_roi_selected(self, roi) -> None:
        self._record_button.setEnabled(True)
        self._status_label.setText(
            f"ROI selected: x={roi.x} y={roi.y} w={roi.width} h={roi.height} "
            f"({roi.width * roi.height} pixels)"
        )

    @property
    def is_recording(self) -> bool:
        return self._recorder is not None

    def on_frame(self, frame: Frame) -> None:
        """Called once per acquired frame by the app's single shared
        acquisition loop (see MainWindow.frame_captured)."""
        if not self.is_recording or self._preview_due():
            self._ir_view.set_image(frame.intensity)
        if self._recorder is not None:
            self._recorder.add_frame(frame)
            if self._recorder.is_done:
                self._finish_recording()

    def _preview_due(self) -> bool:
        now = time.monotonic()
        if now - self._last_preview_update_time < _PREVIEW_THROTTLE_INTERVAL_S:
            return False
        self._last_preview_update_time = now
        return True

    def _start_recording(self) -> None:
        roi = self._ir_view.current_roi()
        if roi is None:
            return
        duration_s = self._duration_box.value()
        try:
            camera_parameters = self._backend.get_parameters()
        except CameraError:
            logger.exception(
                "Could not read camera parameters at recording start; "
                "dataset will save without them"
            )
            camera_parameters = None
        self._recorder = NoiseRecorder(
            roi=roi,
            duration_s=duration_s,
            camera_parameters=camera_parameters,
            device_info=self._backend.device_info,
        )
        self._record_button.setEnabled(False)
        self._results_button.setEnabled(False)
        self._progress.setValue(0)
        self._progress_timer.start(_PROGRESS_TIMER_MS)
        device_info = self._backend.device_info
        logger.info(
            "Started noise recording: roi=(%d,%d,%d,%d) duration=%.1fs "
            "exposure=%s gain=%s frame_rate=%s device=%s",
            roi.x, roi.y, roi.width, roi.height, duration_s,
            camera_parameters.exposure_time_us.current if camera_parameters else "unknown",
            camera_parameters.gain.current if camera_parameters else "unknown",
            camera_parameters.frame_rate_fps.current if camera_parameters else "unknown",
            device_info.display_name if device_info else "unknown",
        )

    def _update_progress(self) -> None:
        if self._recorder is None:
            return
        fraction = min(1.0, self._recorder.elapsed_s / self._recorder.duration_s)
        self._progress.setValue(round(fraction * 100))

    def _finish_recording(self) -> None:
        self._progress_timer.stop()
        self._progress.setValue(100)
        self._latest_recording = self._recorder.finalize()
        self._current_dataset_path = None
        self._recorder = None
        self._record_button.setEnabled(True)
        self._results_button.setEnabled(True)
        self._save_button.setEnabled(True)
        logger.info(
            "Finished noise recording: %d frames, %d samples",
            self._latest_recording.frame_count,
            len(self._latest_recording.depth_mm),
        )

    def _show_results(self) -> None:
        if self._latest_recording is None or self._stats_thread is not None:
            return
        recording = self._latest_recording

        self._results_button.setEnabled(False)
        self._status_label.setText("Computing results...")

        # The heavy numpy analysis (compute_recording_stats) runs on this
        # worker thread; only building/showing the widget below happens back
        # on the main thread, once the results are already computed.
        thread = QThread(self)
        worker = _StatsWorker(recording)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Connected as bound methods of self (a QObject), not lambdas: Qt can
        # only tell these need queuing onto the main thread's event loop (via
        # AutoConnection) by checking the *receiver* QObject's thread
        # affinity - a plain lambda has no such affinity and would otherwise
        # run directly on this worker thread, which is illegal for a slot
        # that goes on to construct/parent a QWidget.
        worker.succeeded.connect(self._on_stats_ready)
        worker.failed.connect(self._on_stats_failed)
        worker.succeeded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.succeeded.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._stats_thread = thread
        self._stats_worker = worker
        thread.start()

    def _on_stats_ready(self, recording: NoiseRecording, stats) -> None:
        self._teardown_stats_thread()
        self._status_label.setText("Results ready.")
        # Parented to this panel (so it can't outlive it) but kept as its own
        # top-level window via Qt.Window, and WA_DeleteOnClose so the widget -
        # and the matplotlib figure/canvas it owns - is actually freed when
        # the user closes it, rather than lingering as a hidden top-level
        # window that also blocks Qt's quitOnLastWindowClosed shutdown.
        results_view = NoiseResultsView(recording, stats, parent=self)
        results_view.setWindowFlag(Qt.Window)
        results_view.setAttribute(Qt.WA_DeleteOnClose)
        title = "Noise Measurement Results"
        if self._current_dataset_path is not None:
            title += f" - {Path(self._current_dataset_path).name}"
        results_view.setWindowTitle(title)
        results_view.resize(1000, 550)
        # WA_DeleteOnClose means the underlying C++ object is gone as soon as
        # the user closes this window by hand - prune it from the tracking
        # list immediately, or a later close_results_windows() call would
        # try to close() an already-deleted widget and raise.
        results_view.destroyed.connect(lambda: self._forget_results_window(results_view))
        results_view.show()
        self._results_windows.append(results_view)

    def _on_stats_failed(self, message: str) -> None:
        self._teardown_stats_thread()
        self._status_label.setText("Failed to compute results.")
        QMessageBox.critical(self, "Failed to compute results", message)

    def _teardown_stats_thread(self) -> None:
        self._stats_thread = None
        self._stats_worker = None
        self._results_button.setEnabled(True)

    def _save_dataset(self) -> None:
        if self._latest_recording is None:
            return
        default_name = time.strftime("%Y%m%d_%H%M%S") + ".npz"
        # DontUseNativeDialog: see the same option on the calibration-file
        # dialog in camera_settings_panel.py - the native/portal dialog comes
        # back as an unusable ~100x30px stub on this machine.
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Noise Dataset",
            default_name,
            "NumPy Archive (*.npz)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return
        if not path.endswith(".npz"):
            path += ".npz"
        try:
            self._latest_recording.save(path)
        except OSError as exc:
            logger.exception("Failed to save dataset to '%s'", path)
            QMessageBox.critical(self, "Save dataset failed", str(exc))
        else:
            logger.info("Saved noise dataset to %s", path)
            self._current_dataset_path = path
            QMessageBox.information(self, "Dataset saved", f"Saved dataset to:\n{path}")

    def _load_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Noise Dataset",
            "",
            "NumPy Archive (*.npz)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return
        try:
            recording = NoiseRecording.load(path)
        except (OSError, KeyError, ValueError) as exc:
            logger.exception("Failed to load dataset from '%s'", path)
            QMessageBox.critical(self, "Load dataset failed", str(exc))
            return
        logger.info("Loaded noise dataset from %s", path)
        self._latest_recording = recording
        self._current_dataset_path = path
        self._save_button.setEnabled(True)
        self._results_button.setEnabled(True)
        self._show_results()

    def _forget_results_window(self, results_view: NoiseResultsView) -> None:
        if results_view in self._results_windows:
            self._results_windows.remove(results_view)

    def close_results_windows(self) -> None:
        """Closes every still-open results window - called when the main
        window shuts down so no orphaned top-level window is left open to
        block application exit."""
        for window in list(self._results_windows):
            try:
                window.close()
            except RuntimeError:
                pass  # already deleted (e.g. the user closed it themselves)
        self._results_windows.clear()
