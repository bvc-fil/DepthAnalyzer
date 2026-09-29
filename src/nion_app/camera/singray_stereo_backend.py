"""Singray Stereo PRO backend that computes depth from the fisheye stereo
pair via OpenCV, instead of the ToF sensor used by singray_backend.py.

Why this backend exists: this unit's onboard SGBM (stereo block matching)
feature is disabled at the firmware level (the device's own feature banner
prints "SGBM: OFF" at xv_device_init()), so the vendor SDK offers no built-in
stereo depth path. ~/Desktop/singray/pythonTest/calibrationTooling/ built one
from scratch - capturing checkerboard pairs, running OpenCV's fisheye
calibration (with the FOV-guess search and outlier filtering documented
there), and saving the result to stereo_calibration.npz - and
~/Desktop/singray/pythonTest/stereoDepthViewer.py proved the resulting
rectify/disparity/reprojectImageTo3D pipeline out live. This backend adapts
that exact pipeline (same maps, same StereoSGBM parameters) into the
CameraBackend interface.

Differences from singray_backend.py and the other backends:
    - Frame.intensity is the real rectified left fisheye image, not a
      synthesized stand-in - unlike the ToF backend (which had no aligned
      IR source and fell back to rendering the depth image itself), depth
      here is computed by matching the SAME rectified left/right pair, so
      intensity and depth_mm are naturally pixel-for-pixel aligned with no
      workaround needed.
    - Depth requires a calibration file to be loaded before it means
      anything (fisheye intrinsics/distortion per camera, stereo
      rotation/translation, and the disparity-to-depth Q matrix - see
      calibrationTooling/calibrate.py for how one is produced).
      load_configuration("Default") - the guided connection's Step 4/5 -
      auto-loads one from SINGRAY_STEREO_CALIBRATION_PATH or the default
      path this dev machine's calibration lives at; load_calibration_file()
      (the existing "Load Camera Calibration File..." button in the Camera
      Settings dock) loads a different one afterward. Both load host-side,
      into this backend's own memory - there's no device-side file slot
      involved, unlike the Nion's LensCalibrationData upload.
    - Exposure/gain are real controls here (xv_set_fe_gain_and_exposureTimeMs,
      xv_set_fe_autoExposure) - unlike the ToF backend, this backend's image
      source *is* the fisheye cameras those calls affect. Gain's [0,255]
      range is documented in xv-sdk.h; exposure's upper bound is not
      documented anywhere, so _FE_EXPOSURE_MS_MAX is a generous, explicitly
      unverified guess. There's still no getter for either (see
      singray_backend.py's note on this same C interface limitation) -
      get_parameters() reports this backend's own last-commanded values.
      Frame rate still has no setter exposed for the stereo stream.
    - start_acquisition() starts slam/stereo/imu/rgb together, matching
      every one of the pythonTest reference scripts exactly (capture.py,
      verify.py, stereoDepthViewer.py, PythonDemo.py all start this same
      combination) - given singray_backend.py already found one hardware
      case (xv_start_tof()) where deviating from a proven-working start
      sequence corrupted the stream, this backend doesn't risk deviating
      from the fisheye stream's own proven sequence either, even though
      IMU/RGB data themselves are unused here.
    - xv_get_sn() is never called, for the same empirically-confirmed reason
      as singray_backend.py (see its docstring) - this is the same native
      library and device, not a per-backend quirk.
"""
from __future__ import annotations

import logging
import time
import zipfile
from ctypes import c_bool
from dataclasses import dataclass

import cv2
import numpy as np

from nion_app.camera._singray_sdk import import_xvsdk
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
from nion_app.camera.singray_calibration import find_default_calibration_path

logger = logging.getLogger(__name__)

_FRAME_POLL_INTERVAL_S = 0.005

# xv_get_sn() is never called (see module docstring) - this device offers no
# other safe way to read its real serial number.
_PLACEHOLDER_SERIAL = "singray-stereo-pro"

# Matches stereoDepthViewer.py exactly - these were tuned there against real
# captures, not re-derived here.
_NUM_DISPARITIES = 96  # must be a multiple of 16
_BLOCK_SIZE = 7

_FE_GAIN_MIN, _FE_GAIN_MAX = 0, 255  # documented in xv-sdk.h's setExposure()
_DEFAULT_FE_GAIN = 15
_DEFAULT_FE_EXPOSURE_MS = 65
# Not documented anywhere in the SDK - a generous, explicitly unverified
# upper bound for the exposure spin box, not a hardware-confirmed limit.
_FE_EXPOSURE_MS_MAX = 1000


@dataclass(frozen=True)
class _StereoCalibration:
    map1_left: np.ndarray
    map2_left: np.ndarray
    map1_right: np.ndarray
    map2_right: np.ndarray
    Q: np.ndarray
    image_size: tuple[int, int]  # (width, height)


def _load_calibration_maps(path: str) -> _StereoCalibration:
    try:
        calib = np.load(path)
        K_left, D_left = calib["K_left"], calib["D_left"]
        K_right, D_right = calib["K_right"], calib["D_right"]
        R1, R2, P1, P2, Q = calib["R1"], calib["R2"], calib["P1"], calib["P2"], calib["Q"]
        image_size = tuple(int(v) for v in calib["image_size"])
    except (OSError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise CameraError(f"Could not load stereo calibration from '{path}': {exc}") from exc

    map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
        K_left, D_left, R1, P1, image_size, cv2.CV_16SC2
    )
    map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
        K_right, D_right, R2, P2, image_size, cv2.CV_16SC2
    )
    return _StereoCalibration(map1_left, map2_left, map1_right, map2_right, Q, image_size)


class SingrayStereoBackend(CameraBackend):
    def __init__(self) -> None:
        self._xvsdk = import_xvsdk()
        self._connected = False
        self._acquiring = False
        self._device_info: DeviceInfo | None = None
        self._discovered: DeviceInfo | None = None
        self._calibration: _StereoCalibration | None = None
        self._exposure_time_ms = _DEFAULT_FE_EXPOSURE_MS
        self._gain = _DEFAULT_FE_GAIN
        self._frame_rate_fps = 30.0
        self._stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=_NUM_DISPARITIES,
            blockSize=_BLOCK_SIZE,
            P1=8 * _BLOCK_SIZE * _BLOCK_SIZE,
            P2=32 * _BLOCK_SIZE * _BLOCK_SIZE,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
        )
        logger.info("Singray stereo-depth backend initialized")

    def discover(self) -> list[DeviceInfo]:
        if self._discovered is not None:
            return [self._discovered]

        # xv_device_init() is both the discovery and the connect step in
        # this C interface - see singray_backend.py's docstring for why its
        # return value (rather than xv_get_sn()) is the liveness signal.
        init_result = self._xvsdk.init()
        logger.info("xv_device_init() returned %s", init_result)
        self._connected = True
        if not init_result:
            logger.warning("No Stereo PRO device discovered (xv_device_init() returned %s)", init_result)
            return []

        self._discovered = DeviceInfo(
            serial_number=_PLACEHOLDER_SERIAL,
            model_name="Singray Stereo PRO",
            display_name="Singray Stereo PRO (stereo depth)",
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

    def _require_calibrated(self) -> _StereoCalibration:
        if self._calibration is None:
            raise CameraError(
                "No stereo calibration loaded - use 'Load Camera Calibration File...' "
                "or set SINGRAY_STEREO_CALIBRATION_PATH and reconnect."
            )
        return self._calibration

    def available_configurations(self) -> list[str]:
        self._require_connected()
        return ["Default"]

    def load_configuration(self, name: str = "Default") -> None:
        """Auto-loads the default stereo calibration - see module docstring
        for why "configuration" means "calibration" for this backend."""
        self._require_connected()
        if name != "Default":
            raise CameraError(
                f"Singray stereo backend has no configuration named '{name}' (only 'Default')"
            )
        if self._calibration is not None:
            return

        path = find_default_calibration_path()
        if path is None:
            raise CameraError(
                "No stereo calibration found. Capture and compute one with "
                "calibrationTooling/capture.py + calibrate.py "
                "(~/Desktop/singray/pythonTest/calibrationTooling/), set "
                "SINGRAY_STEREO_CALIBRATION_PATH to the resulting .npz file, or use "
                "'Load Camera Calibration File...' after connecting."
            )
        self._calibration = _load_calibration_maps(path)
        logger.info("Loaded default stereo calibration from %s", path)

    def load_calibration_file(self, path: str, file_name: str | None = None) -> None:
        # file_name is unused - unlike the Nion's GenICam file upload, this
        # calibration is loaded host-side into this backend, not written to
        # a device-side file slot.
        self._require_connected()
        self._calibration = _load_calibration_maps(path)
        logger.info("Loaded stereo calibration from %s", path)

    def get_parameters(self) -> CameraParameters:
        self._require_connected()
        return CameraParameters(
            exposure_time_us=ParameterRange(
                minimum=0.0,
                maximum=_FE_EXPOSURE_MS_MAX * 1000.0,
                current=self._exposure_time_ms * 1000.0,
            ),
            gain=ParameterRange(
                minimum=float(_FE_GAIN_MIN), maximum=float(_FE_GAIN_MAX), current=float(self._gain)
            ),
            frame_rate_fps=ParameterRange(
                minimum=self._frame_rate_fps,
                maximum=self._frame_rate_fps,
                current=self._frame_rate_fps,
            ),
        )

    def _apply_fe_gain_and_exposure(self) -> None:
        result = self._xvsdk.xv_set_fe_gain_and_exposureTimeMs(self._gain, self._exposure_time_ms)
        logger.info(
            "xv_set_fe_gain_and_exposureTimeMs(%d, %d) returned %s",
            self._gain, self._exposure_time_ms, result,
        )

    def set_exposure_time_us(self, value: float) -> None:
        self._require_connected()
        exposure_ms = round(value / 1000.0)
        clamped = max(0, min(_FE_EXPOSURE_MS_MAX, exposure_ms))
        if clamped != exposure_ms:
            logger.warning(
                "Exposure %d ms out of range [0, %d]; clamped to %d",
                exposure_ms, _FE_EXPOSURE_MS_MAX, clamped,
            )
        self._exposure_time_ms = clamped
        self._apply_fe_gain_and_exposure()

    def set_gain(self, value: float) -> None:
        self._require_connected()
        gain = round(value)
        clamped = max(_FE_GAIN_MIN, min(_FE_GAIN_MAX, gain))
        if clamped != gain:
            logger.warning(
                "Gain %d out of range [%d, %d]; clamped to %d", gain, _FE_GAIN_MIN, _FE_GAIN_MAX, clamped
            )
        self._gain = clamped
        self._apply_fe_gain_and_exposure()

    def set_frame_rate(self, value: float) -> None:
        self._require_connected()
        logger.warning(
            "Singray stereo backend has no frame-rate control in this SDK; ignoring requested %.1f fps",
            value,
        )

    def start_acquisition(self) -> None:
        self._require_connected()
        if self._acquiring:
            return
        # Matches every pythonTest reference script exactly - see module
        # docstring for why IMU/RGB are started even though unused here.
        logger.info("xv_start_slam() returned %s", self._xvsdk.slam_start())
        logger.info("xv_start_stereo() returned %s", self._xvsdk.stereo_start())
        logger.info("xv_start_imu() returned %s", self._xvsdk.imu_start())
        logger.info("xv_start_rgb() returned %s", self._xvsdk.rgb_start())

        # Manual exposure/gain setters are only honored once auto-exposure is
        # off (xv-sdk.h: "Only valid in manual exposure mode").
        self._xvsdk.xv_set_fe_autoExposure(c_bool(False))
        self._apply_fe_gain_and_exposure()

        self._acquiring = True
        logger.info("Acquisition started")

    def stop_acquisition(self) -> None:
        # No per-stream stop call in this SDK - see singray_backend.py's note
        # on the same limitation. This just stops polling get_frame().
        if not self._acquiring:
            return
        self._acquiring = False
        logger.info("Acquisition stopped (streams keep running on-device; no per-stream stop in this SDK)")

    def get_frame(self, timeout_ms: int = 2000) -> Frame:
        if not self._acquiring:
            raise CameraNotConnectedError("Acquisition is not running; call start_acquisition() first")
        calibration = self._require_calibrated()

        xvsdk = self._xvsdk
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            width, height, edge_ts, _host_ts, left_data, right_data, data_size = xvsdk.xv_get_stereo()
            if data_size.value > 0:
                break
            if time.monotonic() >= deadline:
                raise CameraError("Timed out waiting for a stereo frame")
            time.sleep(_FRAME_POLL_INTERVAL_S)

        w, h = width.value, height.value
        left = np.frombuffer(bytes(left_data), dtype=np.uint8, count=w * h).reshape(h, w)
        right = np.frombuffer(bytes(right_data), dtype=np.uint8, count=w * h).reshape(h, w)

        calib_size = calibration.image_size
        if (w, h) != calib_size:
            left = cv2.resize(left, calib_size)
            right = cv2.resize(right, calib_size)

        rect_left = cv2.remap(left, calibration.map1_left, calibration.map2_left, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right, calibration.map1_right, calibration.map2_right, cv2.INTER_LINEAR)

        disparity = self._stereo_matcher.compute(rect_left, rect_right).astype(np.float32) / 16.0
        valid = disparity > 0

        points_3d = cv2.reprojectImageTo3D(disparity, calibration.Q)
        depth_mm = points_3d[:, :, 2].astype(np.float32)
        depth_mm[~valid] = 0.0
        depth_mm[~np.isfinite(depth_mm)] = 0.0
        depth_mm[depth_mm < 0] = 0.0

        return Frame(
            intensity=rect_left,
            depth_mm=depth_mm,
            confidence=None,
            device_timestamp_ns=int(edge_ts.value * 1000),  # edge timestamp is in microseconds
            host_timestamp_ns=time.time_ns(),
        )
