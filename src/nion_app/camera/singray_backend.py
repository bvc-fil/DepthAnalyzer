"""Singray Stereo PRO backend, using the vendor's `xvsdk` ctypes wrapper (a
thin Python layer over `libxvisio-CInterface-wrapper.so`) so the rest of the
app (setup wizard, viewer, noise recorder) works with a Stereo PRO exactly as
with the Nion or RealSense.

Differences from the other backends that shaped this implementation:
    - The vendor SDK ships `xvsdk.py` as a bare file alongside its C library,
      not a pip package - _import_xvsdk() locates and imports it the same way
      ids_backend.py locates its GenTL producer: an env var
      (SINGRAY_SDK_PYTHON_PATH) overriding a known default install location.
      This checkout of the SDK (~/Desktop/singray/sdk/StereoPRO) ships only
      Windows binaries (bin/*.dll, lib/*.lib) - the Linux
      libxvisio-CInterface-wrapper.so that xvsdk.py loads via CDLL() is not
      included here and must be obtained/built separately and placed
      somewhere the dynamic linker can find it (next to xvsdk.py, or on
      LD_LIBRARY_PATH) before this backend can actually open a device.
    - The wrapped C interface has no simultaneous IR + depth channel for the
      ToF sensor (unlike the Nion's multipart Intensity/Range/Confidence
      buffer): xv_get_tof() returns one image whose `type` says whether the
      device is currently streaming Depth_16 (sony, uint16 mm), Depth_32
      (pmd, float32 m), or something else (IR/Cloud/Raw/Eeprom/IQ) - and this
      Python wrapper exposes no call to switch modes. So Frame.intensity here
      is a percentile-stretched rendering of the depth image itself,
      guaranteeing it stays pixel-for-pixel aligned with Frame.depth_mm
      (noise_recording.py applies the same ROI rectangle to both arrays).
      The alternative - the fisheye stereo cameras' own images via
      xv_get_stereo() - was rejected: they're wide-FOV SLAM cameras with no
      calibrated pixel correspondence to the ToF sensor exposed through this
      API, so a rectangle dragged on one would not select the matching
      region of the other.
    - No per-pixel confidence is exposed (the C++ DepthImage struct has only
      a single scalar per-frame confidence, and even that isn't returned by
      this Python wrapper) - Frame.confidence is always None, same as the
      RealSense backend.
    - Exposure/gain/frame-rate: the wrapped C interface has setters only for
      the fisheye cameras (xv_set_fe_*), none for the ToF sensor itself, and
      no getter for any of them. get_parameters() reports the backend's own
      fixed values (minimum == maximum == current) rather than a live
      hardware read, and set_exposure_time_us()/set_gain()/set_frame_rate()
      log a warning and no-op - there is nothing to actually change through
      this SDK surface for the depth stream.
    - "Configurations" don't exist for this device via this API (no UserSet
      or preset-load call is exposed) - available_configurations() returns
      just ["Default"] and load_configuration() is a no-op.
    - xv_device_init()/xv_start_tof()'s return-code convention (0 vs. bool
      "success") isn't documented anywhere in the shipped SDK, and the
      vendor's own PythonDemo.py never checks it either - this backend logs
      the raw value for diagnostics on every start_*() call, and treats it as
      authoritative only for xv_device_init() (see the xv_get_sn() note
      below for why), letting a later get_frame() timeout surface any other
      real failure.
    - xv_start_tof() must not be called on its own: on this hardware it left
      the ToF stream permanently emitting corrupt frames ("Incorrect frame
      recieved. All frame counter are not the same." on every read) and
      eventually crashed the vendor's native library entirely. This was
      confirmed empirically (not documented anywhere in the SDK) against a
      working reference script that additionally starts SLAM and the stereo
      (fisheye) stream first - see ~/Desktop/singray/pythonTest/tofViewer.py.
      start_acquisition() therefore starts slam_start() and stereo_start()
      before tof_start(), even though this backend has no use for SLAM pose
      or fisheye images itself.
    - xv_get_sn() is never called. Empirically (repeated 3/3 crashes both
      before and after starting the streams) calling it anywhere in a
      process that also runs slam/stereo/tof corrupts native state and
      crashes the process later - regardless of call order. The device's
      real serial is also not available through the standard USB serial
      descriptor (`/sys/bus/usb/devices/*/serial` reports a placeholder
      "0.0" for this unit, not the real value xvsdk itself logs
      internally during xv_device_init() via some other, unexposed path).
      DeviceInfo.serial_number is therefore always the fixed placeholder
      _PLACEHOLDER_SERIAL, and discover()/connect() use xv_device_init()'s
      own return value as the liveness signal instead (empirically 1 on
      every successful run so far - its documented polarity is otherwise
      unknown, see the note below).
"""
from __future__ import annotations

import logging
import os
import sys
import time

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

_DEFAULT_SDK_PYTHON_DIRS = [
    os.path.expanduser("~/Desktop/singray/sdk/StereoPRO/python"),
]

# xv::DepthImage::Type (xv-types.h): Depth_16 = sony ToF (uint16, already mm),
# Depth_32 = pmd ToF (float32, meters). The remaining values (IR, Cloud, Raw,
# Eeprom, IQ) aren't distance images this backend can use.
_TOF_TYPE_DEPTH_16 = 0
_TOF_TYPE_DEPTH_32 = 1

_FRAME_POLL_INTERVAL_S = 0.005

# xv_get_sn() is never called (see module docstring) - this device offers no
# other safe way to read its real serial number.
_PLACEHOLDER_SERIAL = "singray-stereo-pro"


def _import_xvsdk():
    """Locates and imports the vendor's `xvsdk` ctypes wrapper module."""
    if "xvsdk" in sys.modules:
        return sys.modules["xvsdk"]

    override = os.environ.get("SINGRAY_SDK_PYTHON_PATH")
    search_dirs = [override] if override else _DEFAULT_SDK_PYTHON_DIRS
    for directory in search_dirs:
        if directory and os.path.isfile(os.path.join(directory, "xvsdk.py")):
            if directory not in sys.path:
                sys.path.insert(0, directory)
            import xvsdk  # type: ignore

            return xvsdk

    raise CameraError(
        "Could not find the Singray Stereo PRO SDK's xvsdk.py. Set "
        "SINGRAY_SDK_PYTHON_PATH to the SDK's 'python' directory (containing "
        "xvsdk.py and its libxvisio-CInterface-wrapper.so dependency)."
    )


def _depth_to_intensity(depth_mm: np.ndarray) -> np.ndarray:
    """Renders depth as a grayscale image for ROI selection - see the module
    docstring for why this substitutes for a real IR channel here."""
    valid = depth_mm > 0
    if not np.any(valid):
        return np.zeros(depth_mm.shape, dtype=np.uint8)
    low, high = np.percentile(depth_mm[valid], [2, 98])
    span = max(high - low, 1e-6)
    stretched = np.clip((depth_mm - low) / span * 255, 0, 255)
    stretched[~valid] = 0
    return stretched.astype(np.uint8)


class SingrayBackend(CameraBackend):
    def __init__(self) -> None:
        self._xvsdk = _import_xvsdk()
        self._connected = False
        self._acquiring = False
        self._device_info: DeviceInfo | None = None
        # No get_* calls exist for these on the wrapped C interface (see
        # module docstring) - these are just the backend's own fixed report,
        # not a live hardware read.
        self._exposure_time_us = 65_000.0
        self._gain = 15.0
        self._frame_rate_fps = 30.0
        self._discovered: DeviceInfo | None = None
        logger.info("Singray xvsdk backend initialized")

    def discover(self) -> list[DeviceInfo]:
        if self._discovered is not None:
            return [self._discovered]

        # xv_device_init() is both the discovery and the connect step in
        # this C interface - see module docstring for why its return value
        # (rather than xv_get_sn()) is the liveness signal used here.
        init_result = self._xvsdk.init()
        logger.info("xv_device_init() returned %s", init_result)
        self._connected = True
        if not init_result:
            logger.warning("No Stereo PRO device discovered (xv_device_init() returned %s)", init_result)
            return []

        self._discovered = DeviceInfo(
            serial_number=_PLACEHOLDER_SERIAL,
            model_name="Singray Stereo PRO",
            display_name="Singray Stereo PRO",
        )
        logger.info("Discovered device: %s", self._discovered)
        return [self._discovered]

    def connect(self, serial_number: str | None = None) -> None:
        devices = self.discover()
        if not devices:
            raise CameraNotFoundError(
                "No Stereo PRO camera found. Verify the USB connection and that "
                "libxvisio-CInterface-wrapper.so is on the library search path."
            )

        chosen = devices[0]
        if serial_number is not None and chosen.serial_number != serial_number:
            raise CameraNotFoundError(
                f"No device with serial number {serial_number} (found {chosen.serial_number})"
            )
        self._device_info = chosen
        logger.info("Connected to %s", chosen)

    def disconnect(self) -> None:
        if self._acquiring:
            self.stop_acquisition()
        if self._connected:
            self._xvsdk.stop()
            self._connected = False
        self._device_info = None
        self._discovered = None
        logger.info("Disconnected Singray device")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def device_info(self) -> DeviceInfo | None:
        return self._device_info

    def _require_connected(self) -> None:
        if not self._connected:
            raise CameraNotConnectedError("Camera is not connected; call connect() first")

    def available_configurations(self) -> list[str]:
        self._require_connected()
        return ["Default"]

    def load_configuration(self, name: str = "Default") -> None:
        self._require_connected()
        if name != "Default":
            raise CameraError(
                f"Singray backend has no loadable configurations (requested '{name}')"
            )
        logger.info("No configuration to load for Singray backend (device is factory-calibrated)")

    def get_parameters(self) -> CameraParameters:
        self._require_connected()
        return CameraParameters(
            exposure_time_us=ParameterRange(
                minimum=self._exposure_time_us,
                maximum=self._exposure_time_us,
                current=self._exposure_time_us,
            ),
            gain=ParameterRange(minimum=self._gain, maximum=self._gain, current=self._gain),
            frame_rate_fps=ParameterRange(
                minimum=self._frame_rate_fps,
                maximum=self._frame_rate_fps,
                current=self._frame_rate_fps,
            ),
        )

    def set_exposure_time_us(self, value: float) -> None:
        self._require_connected()
        logger.warning(
            "Singray backend's ToF stream has no exposure control in this SDK; ignoring requested %.1f us",
            value,
        )

    def set_gain(self, value: float) -> None:
        self._require_connected()
        logger.warning(
            "Singray backend's ToF stream has no gain control in this SDK; ignoring requested %.2f",
            value,
        )

    def set_frame_rate(self, value: float) -> None:
        self._require_connected()
        logger.warning(
            "Singray backend's ToF stream has no frame-rate control in this SDK; ignoring requested %.1f fps",
            value,
        )

    def start_acquisition(self) -> None:
        self._require_connected()
        if self._acquiring:
            return
        # Order matters - see module docstring: xv_start_tof() alone corrupts
        # the ToF stream on this hardware. SLAM/stereo must be running first.
        slam_result = self._xvsdk.slam_start()
        logger.info("xv_start_slam() returned %s", slam_result)
        stereo_result = self._xvsdk.stereo_start()
        logger.info("xv_start_stereo() returned %s", stereo_result)
        tof_result = self._xvsdk.tof_start()
        logger.info("xv_start_tof() returned %s", tof_result)
        self._acquiring = True
        logger.info("Acquisition started")

    def stop_acquisition(self) -> None:
        # The wrapped C interface has no xv_stop_tof()/per-stream stop call -
        # only the whole-device xv_device_uninit() (see disconnect()). This
        # just stops this backend from polling get_frame(); the ToF stream
        # itself keeps running on-device until disconnect().
        if not self._acquiring:
            return
        self._acquiring = False
        logger.info("Acquisition stopped (stream keeps running on-device; no per-stream stop in this SDK)")

    def get_frame(self, timeout_ms: int = 2000) -> Frame:
        if not self._acquiring:
            raise CameraNotConnectedError("Acquisition is not running; call start_acquisition() first")

        xvsdk = self._xvsdk
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            width, height, edge_ts, _host_ts, data, data_size, tof_type = xvsdk.xv_get_tof()
            if data_size.value > 0:
                break
            if time.monotonic() >= deadline:
                raise CameraError("Timed out waiting for a ToF frame")
            time.sleep(_FRAME_POLL_INTERVAL_S)

        w, h = width.value, height.value
        raw = bytes(data)
        kind = tof_type.value
        if kind == _TOF_TYPE_DEPTH_16:
            depth_mm = np.frombuffer(raw, dtype="<u2", count=w * h).reshape(h, w).astype(np.float32)
        elif kind == _TOF_TYPE_DEPTH_32:
            depth_m = np.frombuffer(raw, dtype="<f4", count=w * h).reshape(h, w)
            depth_mm = depth_m.astype(np.float32) * 1000.0
        else:
            raise CameraError(
                f"ToF frame is not a depth image (type={kind}); this backend requires the "
                "device to be streaming Depth_16 or Depth_32."
            )

        return Frame(
            intensity=_depth_to_intensity(depth_mm),
            depth_mm=depth_mm,
            confidence=None,
            device_timestamp_ns=int(edge_ts.value * 1000),  # edge timestamp is in microseconds
            host_timestamp_ns=time.time_ns(),
        )
