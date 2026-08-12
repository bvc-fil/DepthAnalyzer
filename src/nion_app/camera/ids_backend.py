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

The device also stores a "LensCalibrationData" file in its persistent memory
that must be uploaded (via ids_peak's FileAdapter, per IDS's own documented
usage: FileAdapter(node_map, file_name).Write(data)) before the depth output
is properly calibrated. FileAdapter.Write() re-initializes every GenTL
producer DeviceManager knows about, not just the one for this device - see
_find_gev_producer_path's docstring for why this backend explicitly registers
only the GigE producer instead of letting GENICAM_GENTL64_PATH's directory
get scanned wholesale.
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
_CALIBRATION_FILE_NAME = "LensCalibrationData"


_GEV_PRODUCER_FILENAME = "ids_gevgentl.cti"  # GigE Vision - the only producer the Nion needs


def _find_gev_producer_path() -> str:
    """Locates the GigE Vision GenTL producer (.cti) the Nion needs.

    Returns one specific file rather than a directory to scan on purpose.
    DeviceManager.Update()'s default policy scans every producer under
    GENICAM_GENTL64_PATH, and a real IDS peak install directory can also
    bundle a producer for legacy uEye USB cameras (fails to load if its
    runtime .so is missing) or a "proxy" producer that fails its own
    unrelated internal init - plain device discovery tolerates either
    failure, but FileAdapter.Write() (used by load_calibration_file) does
    not, and surfaces it as a hard CTI_LOADING_ERROR. The Nion never needs
    any producer but this one, so the backend explicitly registers just this
    file via DeviceManager.AddProducerLibrary() and always updates with
    UpdatePolicy_DontScanEnvironmentForProducerLibraries (see __init__ and
    discover()/connect()) - environment-variable directory scanning, and
    every other producer that would come with it, is never touched at all.

    GENICAM_GENTL64_PATH is still honored as an override if set, pointing at
    either the producer file directly or a directory to find it under.
    """
    override = os.environ.get("GENICAM_GENTL64_PATH")
    if override and os.path.isfile(override):
        return override
    search_dirs = [override] if override else sorted(
        glob.glob("/opt/ids-peak*/lib/*/ids-peak/cti"),
        key=lambda path: ("ueyetl" in path, path),
    )
    for directory in search_dirs:
        candidate = os.path.join(directory, _GEV_PRODUCER_FILENAME)
        if os.path.isfile(candidate):
            return candidate

    raise CameraError(
        f"Could not find {_GEV_PRODUCER_FILENAME} under GENICAM_GENTL64_PATH or "
        "the standard IDS peak install locations under /opt. Install the IDS "
        "peak SDK or set GENICAM_GENTL64_PATH to the producer's file or directory."
    )


class IdsPeakBackend(CameraBackend):
    def __init__(self) -> None:
        import ids_peak.ids_peak as ids_peak

        self._ids_peak = ids_peak
        self._ids_peak.Library.Initialize()
        gev_producer_path = _find_gev_producer_path()
        self._ids_peak.DeviceManager.Instance().AddProducerLibrary(gev_producer_path)
        self._device = None
        self._device_info: DeviceInfo | None = None
        self._node_map = None
        self._data_stream = None
        self._acquiring = False
        logger.info("IDS peak library initialized (producer: %s)", gev_producer_path)

    def _update_device_manager(self):
        dm = self._ids_peak.DeviceManager.Instance()
        # Only the GigE producer explicitly registered in __init__ is ever
        # considered - environment-variable directory scanning (and every
        # other producer that would come with it) is skipped entirely.
        dm.Update(self._ids_peak.DeviceManager.UpdatePolicy_DontScanEnvironmentForProducerLibraries)
        return dm

    def discover(self) -> list[DeviceInfo]:
        dm = self._update_device_manager()
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
        dm = self._update_device_manager()
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

        # Exclusive rather than Control: writing the File Access Control
        # registers (for load_calibration_file) has been observed to fail at
        # the GVCP wire level with ACCESS_DENIED under Control access, even
        # though every other feature (exposure/gain/UserSet/acquisition)
        # works fine under either - the device firmware itself appears to
        # gate file writes behind the higher privilege level. This app is the
        # sole intended controller of the camera for its session, so there's
        # no concurrent-access use case being given up here.
        self._device = chosen.OpenDevice(self._ids_peak.DeviceAccessType_Exclusive)
        self._node_map = self._device.RemoteDevice().NodeMaps()[0]
        self._device_info = DeviceInfo(
            serial_number=chosen.SerialNumber(),
            model_name=chosen.ModelName(),
            display_name=chosen.DisplayName(),
        )
        logger.info(
            "Connected to %s (serial %s)", chosen.ModelName(), chosen.SerialNumber()
        )

    def disconnect(self) -> None:
        if self._acquiring:
            self.stop_acquisition()
        self._data_stream = None
        self._device = None
        self._device_info = None
        self._node_map = None
        self._ids_peak.Library.Close()
        logger.info("Disconnected and closed IDS peak library")

    @property
    def is_connected(self) -> bool:
        return self._node_map is not None

    @property
    def device_info(self) -> DeviceInfo | None:
        return self._device_info

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
        # Buffers announced via AllocAndAnnounceBuffer() in start_acquisition()
        # must be explicitly revoked (only possible now that Flush(DiscardAll)
        # has moved them out of the queued state) and the stream reference
        # dropped - otherwise a later start_acquisition() call in the same
        # session (e.g. resuming after loading a calibration file) opens a
        # second data stream while these are still announced on the first.
        for buffer in self._data_stream.AnnouncedBuffers():
            self._data_stream.RevokeBuffer(buffer)
        self._data_stream = None
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

    def load_calibration_file(self, path: str, file_name: str | None = None) -> None:
        target_name = file_name or _CALIBRATION_FILE_NAME
        nm = self._require_connected()
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError as exc:
            raise CameraError(f"Could not read calibration file '{path}': {exc}") from exc

        # FileSelector (and file access generally) is only writable while the
        # device isn't streaming - the same TLParamsLocked restriction that
        # already gates parameter changes during acquisition. Drop out of
        # acquisition for the write and always resume afterward, success or not.
        was_acquiring = self._acquiring
        if was_acquiring:
            self.stop_acquisition()

        try:
            # Matches IDS's own documented usage exactly: construct the
            # adapter for this file and Write() straight into it. Deliberately
            # skip FileAdapter.AvailableFileNames() - that static method isn't
            # part of the documented upload flow and goes through a
            # System-level path that re-initializes every registered GenTL
            # producer, including ones unrelated to this device (a "proxy"
            # producer has been observed to fail its own internal init on
            # this setup, surfacing as a CTI_LOADING_ERROR here even though
            # it's harmless to plain device discovery).
            adapter = self._ids_peak.FileAdapter(nm, target_name)
            adapter.Write(data)
        except self._ids_peak.Exception as exc:
            raise CameraError(
                f"Failed to write '{target_name}' onto the device: {exc}"
            ) from exc
        finally:
            if was_acquiring:
                self.start_acquisition()

        logger.info(
            "Loaded calibration file '%s' (%d bytes) onto the device from %s",
            target_name, len(data), path,
        )
