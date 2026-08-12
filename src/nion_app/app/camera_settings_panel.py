"""Live exposure/gain/frame-rate controls for the connected camera - lets the
user dial in a usable image (e.g. fix an underexposed IR view) by eye, rather
than being stuck with whatever the loaded configuration set at connect time."""
from __future__ import annotations

import logging

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nion_app.camera.backend import CameraBackend, CameraError

logger = logging.getLogger(__name__)

_DEFAULT_CALIBRATION_FILE_NAME = "LensCalibrationData"


class CameraSettingsPanel(QWidget):
    # Emitted with the camera's actual current frame rate whenever it becomes
    # known (on refresh) or is changed by the user - so the frame-acquisition
    # loop driving recording can be kept in step with it.
    frame_rate_changed = Signal(float)

    def __init__(self, backend: CameraBackend, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._backend = backend
        self._updating = False  # guards against feedback loops while populating fields

        self._exposure_box = QDoubleSpinBox()
        self._exposure_box.setDecimals(1)
        self._exposure_box.setSuffix(" µs")
        self._exposure_box.valueChanged.connect(self._on_exposure_changed)

        self._gain_box = QDoubleSpinBox()
        self._gain_box.setDecimals(2)
        self._gain_box.valueChanged.connect(self._on_gain_changed)

        self._frame_rate_box = QDoubleSpinBox()
        self._frame_rate_box.setDecimals(1)
        self._frame_rate_box.setSuffix(" fps")
        self._frame_rate_box.valueChanged.connect(self._on_frame_rate_changed)

        self._calibration_file_name_box = QLineEdit(_DEFAULT_CALIBRATION_FILE_NAME)
        self._calibration_file_name_box.setToolTip(
            "Device-side file slot to write to. If the device rejects this "
            "name (or rejects it for writing), try a different one - e.g. "
            "'UserData1', a conventional user-writable slot on many devices."
        )

        self._load_calibration_button = QPushButton("Load Camera Calibration File...")
        self._load_calibration_button.clicked.connect(self._on_load_calibration_clicked)

        form = QFormLayout()
        form.addRow("Exposure", self._exposure_box)
        form.addRow("Gain", self._gain_box)
        form.addRow("Frame Rate", self._frame_rate_box)
        form.addRow("Device file name", self._calibration_file_name_box)
        form.addRow(self._load_calibration_button)

        group = QGroupBox("Camera Settings")
        group.setLayout(form)

        layout = QVBoxLayout(self)
        layout.addWidget(group)

        self.setEnabled(False)  # re-enabled once refresh_from_backend() has real ranges

    def refresh_from_backend(self) -> None:
        """Populates the controls with the camera's actual supported ranges
        and current values. Call once the backend is connected (get_parameters()
        requires that), typically right after a successful guided connection."""
        try:
            params = self._backend.get_parameters()
        except CameraError:
            logger.exception("Could not read camera parameters; settings panel stays disabled")
            return

        self._updating = True
        try:
            self._exposure_box.setRange(
                params.exposure_time_us.minimum, params.exposure_time_us.maximum
            )
            self._exposure_box.setValue(params.exposure_time_us.current)
            self._gain_box.setRange(params.gain.minimum, params.gain.maximum)
            self._gain_box.setValue(params.gain.current)
            self._frame_rate_box.setRange(
                params.frame_rate_fps.minimum, params.frame_rate_fps.maximum
            )
            self._frame_rate_box.setValue(params.frame_rate_fps.current)
        finally:
            self._updating = False
        self.setEnabled(True)
        self.frame_rate_changed.emit(params.frame_rate_fps.current)

    def _on_exposure_changed(self, value: float) -> None:
        if self._updating:
            return
        try:
            self._backend.set_exposure_time_us(value)
        except CameraError:
            logger.exception("Failed to set exposure time to %.1f us", value)

    def _on_gain_changed(self, value: float) -> None:
        if self._updating:
            return
        try:
            self._backend.set_gain(value)
        except CameraError:
            logger.exception("Failed to set gain to %.2f", value)

    def _on_frame_rate_changed(self, value: float) -> None:
        if self._updating:
            return
        try:
            self._backend.set_frame_rate(value)
            actual = self._backend.get_parameters().frame_rate_fps.current
        except CameraError:
            logger.exception("Failed to set frame rate to %.1f fps", value)
            return
        if actual != value:
            # Some backends (e.g. RealSense) only support a fixed, discrete
            # set of frame rates and snap to the nearest one - reflect what
            # was actually applied rather than leaving the box showing a
            # value that isn't really in effect.
            self._updating = True
            try:
                self._frame_rate_box.setValue(actual)
            finally:
                self._updating = False
        self.frame_rate_changed.emit(actual)

    def _on_load_calibration_clicked(self) -> None:
        # DontUseNativeDialog: on this machine the native/portal file dialog
        # comes back as an unusable ~100x30px stub (its async native/portal
        # backend never actually populates it) - Qt's own built-in dialog
        # doesn't depend on that and always renders correctly.
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Camera Calibration File", options=QFileDialog.Option.DontUseNativeDialog
        )
        if not path:
            return
        file_name = self._calibration_file_name_box.text().strip() or None
        try:
            self._backend.load_calibration_file(path, file_name=file_name)
        except Exception as exc:
            # Broad on purpose: a failure here (including one from resuming
            # acquisition afterward) must always surface to the user, never
            # vanish as an uncaught exception in a Qt slot.
            logger.exception("Failed to load calibration file '%s'", path)
            QMessageBox.critical(self, "Load calibration file failed", str(exc))
        else:
            logger.info("Loaded calibration file from %s", path)
            QMessageBox.information(
                self, "Calibration file loaded", f"Loaded calibration file:\n{path}"
            )
