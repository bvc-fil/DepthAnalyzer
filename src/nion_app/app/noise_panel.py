"""Panel driving the outline's noise-measurement workflow: pick a duration,
drag a ROI on the live IR image, record, then inspect per-pixel histograms
and standard deviations."""
from __future__ import annotations

import logging

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nion_app.app.noise_results_view import NoiseResultsView
from nion_app.camera.backend import Frame
from nion_app.camera.noise_recording import NoiseRecorder, NoiseRecording
from nion_app.viewer.ir_roi_view import IrRoiView

logger = logging.getLogger(__name__)

_PROGRESS_TIMER_MS = 200


class NoisePanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._recorder: NoiseRecorder | None = None
        self._latest_recording: NoiseRecording | None = None
        self._results_windows: list[NoiseResultsView] = []  # keep references alive

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

    def _on_roi_selected(self, roi) -> None:
        self._record_button.setEnabled(True)
        self._status_label.setText(
            f"ROI selected: x={roi.x} y={roi.y} w={roi.width} h={roi.height} "
            f"({roi.width * roi.height} pixels)"
        )

    def on_frame(self, frame: Frame) -> None:
        """Called once per acquired frame by the app's single shared
        acquisition loop (see MainWindow.frame_captured)."""
        self._ir_view.set_image(frame.intensity)
        if self._recorder is not None:
            self._recorder.add_frame(frame)
            if self._recorder.is_done:
                self._finish_recording()

    def _start_recording(self) -> None:
        roi = self._ir_view.current_roi()
        if roi is None:
            return
        duration_s = self._duration_box.value()
        self._recorder = NoiseRecorder(roi=roi, duration_s=duration_s)
        self._record_button.setEnabled(False)
        self._results_button.setEnabled(False)
        self._progress.setValue(0)
        self._progress_timer.start(_PROGRESS_TIMER_MS)
        logger.info(
            "Started noise recording: roi=(%d,%d,%d,%d) duration=%.1fs",
            roi.x, roi.y, roi.width, roi.height, duration_s,
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
        self._recorder = None
        self._record_button.setEnabled(True)
        self._results_button.setEnabled(True)
        logger.info(
            "Finished noise recording: %d frames, %d samples",
            self._latest_recording.frame_count,
            len(self._latest_recording.depth_mm),
        )

    def _show_results(self) -> None:
        if self._latest_recording is None:
            return
        # Parented to this panel (so it can't outlive it) but kept as its own
        # top-level window via Qt.Window, and WA_DeleteOnClose so the widget -
        # and the matplotlib figure/canvas it owns - is actually freed when
        # the user closes it, rather than lingering as a hidden top-level
        # window that also blocks Qt's quitOnLastWindowClosed shutdown.
        results_view = NoiseResultsView(self._latest_recording, parent=self)
        results_view.setWindowFlag(Qt.Window)
        results_view.setAttribute(Qt.WA_DeleteOnClose)
        results_view.setWindowTitle("Noise Measurement Results")
        results_view.resize(1000, 550)
        # WA_DeleteOnClose means the underlying C++ object is gone as soon as
        # the user closes this window by hand - prune it from the tracking
        # list immediately, or a later close_results_windows() call would
        # try to close() an already-deleted widget and raise.
        results_view.destroyed.connect(lambda: self._forget_results_window(results_view))
        results_view.show()
        self._results_windows.append(results_view)

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
