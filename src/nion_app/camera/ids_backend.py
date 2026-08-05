"""IDS peak SDK backend for the Nion 3D ToF camera.

Grounded against the real device (model NION10.67.G1.AF0130.7X) rather than
docs alone: the camera exposes three simultaneously-enabled GenICam
components in one multi-part buffer per frame -
    Intensity  -> Mono8       (used as the IR image for ROI selection)
    Range      -> Coord3D_C16 (raw 12-bit Z distance, scaled by
                               Scan3dCoordinateScale/Offset into millimeters)
    Confidence -> Confidence16
Gain/ExposureTime/AcquisitionFrameRate are standard GenICam float nodes, and
persisted configurations are loaded via the standard UserSetSelector/
UserSetLoad command pair.
"""
from __future__ import annotations

import glob
import logging
import os

import numpy as np

from nion_app.camera.backend import (
    CameraBackend,
    CameraError,
    CameraNotConnectedError,
    CameraNotFoundError,
    CameraParameters,
    DeviceInfo,
    Frame,
    ParameterRange,
)

logger = logging.getLogger(__name__)

_INTENSITY_FORMAT = "MONO_8"
_RANGE_FORMAT = "COORD3D_C16"
_CONFIDENCE_FORMAT = "CONFIDENCE_16"


def _ensure_gentl_producer_path() -> None:
    """The IDS peak GenTL layer refuses to enumerate devices unless
    GENICAM_GENTL64_PATH points at the .cti producers. A full SDK install
    doesn't set this for you, so auto-detect it under the standard install
    location instead of making every developer export it by hand."""
    if os.environ.get("GENICAM_GENTL64_PATH"):
        return
    candidates = sorted(glob.glob("/opt/ids-peak*/lib/*/ids-peak/cti"))
    if not candidates:
        raise CameraError(
            "GENICAM_GENTL64_PATH is not set and no IDS peak CTI directory was "
            "found under /opt. Install the IDS peak SDK or set the variable manually."
        )
    os.environ["GENICAM_GENTL64_PATH"] = candidates[0]
    logger.info("GENICAM_GENTL64_PATH not set; using detected path %s", candidates[0])


class IdsPeakBackend(CameraBackend):
    def __init__(self) -> None:
        _ensure_gentl_producer_path()
        import ids_peak.ids_peak as ids_peak

        self._ids_peak = ids_peak
        self._ids_peak.Library.Initialize()
        self._device = None
        self._node_map = None
        self._data_stream = None
        self._acquiring = False
        logger.info("IDS peak library initialized")

    def discover(self) -> list[DeviceInfo]:
        dm = self._ids_peak.DeviceManager.Instance()
        dm.Update()
        infos = [
            DeviceInfo(
                serial_number=d.SerialNumber(),
                model_name=d.ModelName(),
                display_name=d.DisplayName(),
            )
            for d in dm.Devices()
        ]
        if infos:
            logger.info("Discovered %d device(s): %s", len(infos), infos)
        else:
            logger.warning(
                "No devices discovered - check camera power and the ethernet cable"
            )
        return infos

    def connect(self, serial_number: str | None = None) -> None:
        dm = self._ids_peak.DeviceManager.Instance()
        dm.Update()
        descriptors = dm.Devices()
        if not descriptors:
            raise CameraNotFoundError(
                "No camera found. Verify power and the physical ethernet connection."
            )

        chosen = descriptors[0]
        if serial_number is not None:
            matches = [d for d in descriptors if d.SerialNumber() == serial_number]
            if not matches:
                raise CameraNotFoundError(f"No device with serial number {serial_number}")
            chosen = matches[0]

        self._device = chosen.OpenDevice(self._ids_peak.DeviceAccessType_Control)
        self._node_map = self._device.RemoteDevice().NodeMaps()[0]
        logger.info(
            "Connected to %s (serial %s)", chosen.ModelName(), chosen.SerialNumber()
        )

    def disconnect(self) -> None:
        if self._acquiring:
            self.stop_acquisition()
        self._data_stream = None
        self._device = None
        self._node_map = None
        self._ids_peak.Library.Close()
        logger.info("Disconnected and closed IDS peak library")

    @property
    def is_connected(self) -> bool:
        return self._node_map is not None

    def _require_connected(self):
        if self._node_map is None:
            raise CameraNotConnectedError("Camera is not connected; call connect() first")
        return self._node_map

    def available_configurations(self) -> list[str]:
        nm = self._require_connected()
        return [e.SymbolicValue() for e in nm.FindNode("UserSetSelector").Entries()]

    def load_configuration(self, name: str = "Default") -> None:
        nm = self._require_connected()
        nm.FindNode("UserSetSelector").SetCurrentEntry(name)
        load_node = nm.FindNode("UserSetLoad")
        load_node.Execute()
        load_node.WaitUntilDone()
        logger.info("Loaded configuration '%s'", name)
        self._ensure_all_components_enabled()

    def _ensure_all_components_enabled(self) -> None:
        """A loaded UserSet can leave Confidence (or any component) disabled,
        which silently drops that part from every subsequent buffer. Force all
        three required components on, since the app always needs intensity
        (IR/ROI selection), range (depth), and confidence together."""
        nm = self._require_connected()
        multipart_node = nm.FindNode("MultipartEnabled")
        if not multipart_node.Value():
            multipart_node.SetValue(True)
            logger.info("Enabled MultipartEnabled")

        selector = nm.FindNode("ComponentSelector")
        for name in ("Intensity", "Range", "Confidence"):
            selector.SetCurrentEntry(name)
            enable_node = nm.FindNode("ComponentEnable")
            if not enable_node.Value():
                enable_node.SetValue(True)
                logger.info("Enabled component '%s'", name)

    def get_parameters(self) -> CameraParameters:
        nm = self._require_connected()

        def read(node_name: str) -> ParameterRange:
            node = nm.FindNode(node_name)
            return ParameterRange(
                minimum=node.Minimum(), maximum=node.Maximum(), current=node.Value()
            )

        return CameraParameters(
            exposure_time_us=read("ExposureTime"),
            gain=read("Gain"),
            frame_rate_fps=read("AcquisitionFrameRate"),
        )

    def _set_clamped(self, node_name: str, value: float) -> None:
        nm = self._require_connected()
        node = nm.FindNode(node_name)
        clamped = max(node.Minimum(), min(node.Maximum(), value))
        if clamped != value:
            logger.warning(
                "%s=%.4f out of range [%.4f, %.4f]; clamped to %.4f",
                node_name, value, node.Minimum(), node.Maximum(), clamped,
            )
        node.SetValue(clamped)
        logger.info("Set %s to %.4f", node_name, clamped)

    def set_exposure_time_us(self, value: float) -> None:
        self._set_clamped("ExposureTime", value)

    def set_gain(self, value: float) -> None:
        self._set_clamped("Gain", value)

    def set_frame_rate(self, value: float) -> None:
        self._set_clamped("AcquisitionFrameRate", value)

    def start_acquisition(self) -> None:
        nm = self._require_connected()
        if self._acquiring:
            return

        self._ensure_all_components_enabled()
        self._data_stream = self._device.DataStreams()[0].OpenDataStream()
        payload_size = nm.FindNode("PayloadSize").Value()
        buffer_count = max(self._data_stream.NumBuffersAnnouncedMinRequired(), 3)
        for _ in range(buffer_count):
            buffer = self._data_stream.AllocAndAnnounceBuffer(payload_size)
            self._data_stream.QueueBuffer(buffer)

        self._data_stream.StartAcquisition()
        nm.FindNode("TLParamsLocked").SetValue(1)
        start_node = nm.FindNode("AcquisitionStart")
        start_node.Execute()
        start_node.WaitUntilDone()
        self._acquiring = True
        logger.info("Acquisition started (%d buffers queued)", buffer_count)

    def stop_acquisition(self) -> None:
        if not self._acquiring:
            return
        nm = self._require_connected()
        stop_node = nm.FindNode("AcquisitionStop")
        stop_node.Execute()
        stop_node.WaitUntilDone()
        self._data_stream.StopAcquisition(self._ids_peak.AcquisitionStopMode_Default)
        nm.FindNode("TLParamsLocked").SetValue(0)
        self._data_stream.Flush(self._ids_peak.DataStreamFlushMode_DiscardAll)
        self._acquiring = False
        logger.info("Acquisition stopped")

    def get_frame(self, timeout_ms: int = 2000) -> Frame:
        if not self._acquiring:
            raise CameraNotConnectedError("Acquisition is not running; call start_acquisition() first")

        nm = self._node_map
        buffer = self._data_stream.WaitForFinishedBuffer(timeout_ms)
        try:
            parts_by_format = {}
            for part in buffer.Parts():
                image_view = part.ToImageView()
                parts_by_format[image_view.pixel_format.name] = image_view.to_numpy_array(copy=True)

            missing = {_INTENSITY_FORMAT, _RANGE_FORMAT, _CONFIDENCE_FORMAT} - parts_by_format.keys()
            if missing:
                raise CameraError(f"Buffer is missing expected component(s): {missing}")

            scale = nm.FindNode("Scan3dCoordinateScale").Value()
            offset = nm.FindNode("Scan3dCoordinateOffset").Value()
            depth_mm = parts_by_format[_RANGE_FORMAT].astype(np.float32) * scale + offset

            frame = Frame(
                intensity=parts_by_format[_INTENSITY_FORMAT],
                depth_mm=depth_mm,
                confidence=parts_by_format[_CONFIDENCE_FORMAT],
                device_timestamp_ns=buffer.Timestamp_ns(),
                host_timestamp_ns=buffer.SystemTimestamp_ns(),
            )
        finally:
            self._data_stream.QueueBuffer(buffer)

        return frame
