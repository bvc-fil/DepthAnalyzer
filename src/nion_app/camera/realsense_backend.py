"""Intel RealSense D455 backend, implementing the same CameraBackend interface
as the Nion/IDS peak backend so the rest of the app (setup wizard, viewer,
noise recorder) is unaffected by which sensor is attached.

Differences from the Nion that shaped this implementation:
    - No hardware confidence channel. The D455 is active-stereo, not ToF, and
      exposes no per-pixel confidence component - get_frame() always returns
      confidence=None (Frame.confidence is Optional for exactly this reason).
    - "Configurations" are visual presets (rs.option.visual_preset: Default,
      HighAccuracy, HighDensity, ...), not GenICam UserSets, but the
      available_configurations()/load_configuration() names still resolve to
      them.
    - Frame rate is chosen when a stream is requested, not a continuously
      adjustable sensor node - set_frame_rate() restarts the pipeline if
      acquisition is already running.
"""
from __future__ import annotations

import logging
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

_DEFAULT_WIDTH = 848
_DEFAULT_HEIGHT = 480
_DEFAULT_FPS = 30
_INFRARED_STREAM_INDEX = 1  # left IR camera; used as the intensity/ROI image


class RealSenseBackend(CameraBackend):
    def __init__(self) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        self._device = None
        self._device_info: DeviceInfo | None = None
        self._depth_sensor = None
        self._pipeline = None
        self._acquiring = False
        self._width = _DEFAULT_WIDTH
        self._height = _DEFAULT_HEIGHT
        self._fps = _DEFAULT_FPS
        logger.info("pyrealsense2 backend initialized")

    def discover(self) -> list[DeviceInfo]:
        ctx = self._rs.context()
        devices = ctx.query_devices()
        infos = [
            DeviceInfo(
                serial_number=d.get_info(self._rs.camera_info.serial_number),
                model_name=d.get_info(self._rs.camera_info.name),
                display_name=d.get_info(self._rs.camera_info.name),
            )
            for d in devices
        ]
        if infos:
            logger.info("Discovered %d device(s): %s", len(infos), infos)
        else:
            logger.warning("No RealSense devices discovered - check the USB connection")
        return infos

    def connect(self, serial_number: str | None = None) -> None:
        ctx = self._rs.context()
        devices = ctx.query_devices()
        if not devices:
            raise CameraNotFoundError("No RealSense camera found. Verify the USB connection.")

        chosen = devices[0]
        if serial_number is not None:
            matches = [
                d for d in devices if d.get_info(self._rs.camera_info.serial_number) == serial_number
            ]
            if not matches:
                raise CameraNotFoundError(f"No device with serial number {serial_number}")
            chosen = matches[0]

        self._device = chosen
        self._depth_sensor = chosen.first_depth_sensor()
        name = chosen.get_info(self._rs.camera_info.name)
        serial = chosen.get_info(self._rs.camera_info.serial_number)
        self._device_info = DeviceInfo(
            serial_number=serial, model_name=name, display_name=name
        )
        logger.info("Connected to %s (serial %s)", name, serial)

    def disconnect(self) -> None:
        if self._acquiring:
            self.stop_acquisition()
        self._depth_sensor = None
        self._device = None
        self._device_info = None
        logger.info("Disconnected RealSense device")

    @property
    def is_connected(self) -> bool:
        return self._device is not None

    @property
    def device_info(self) -> DeviceInfo | None:
        return self._device_info

    def _require_connected(self):
        if self._device is None:
            raise CameraNotConnectedError("Camera is not connected; call connect() first")
        return self._depth_sensor

    def available_configurations(self) -> list[str]:
        sensor = self._require_connected()
        preset_range = sensor.get_option_range(self._rs.option.visual_preset)
        return [
            sensor.get_option_value_description(self._rs.option.visual_preset, i)
            for i in range(int(preset_range.min), int(preset_range.max) + 1)
        ]

    def load_configuration(self, name: str = "Default") -> None:
        sensor = self._require_connected()
        preset_range = sensor.get_option_range(self._rs.option.visual_preset)
        for i in range(int(preset_range.min), int(preset_range.max) + 1):
            if sensor.get_option_value_description(self._rs.option.visual_preset, i) == name:
                sensor.set_option(self._rs.option.visual_preset, float(i))
                logger.info("Loaded configuration '%s'", name)
                return
        raise CameraError(
            f"No configuration named '{name}' (available: {self.available_configurations()})"
        )

    def _supported_fps_values(self) -> list[int]:
        """Frame rates the depth+infrared stream pair this backend enables
        can actually run at, at the currently configured resolution.
        RealSense stream profiles are a fixed, discrete set per
        resolution/format - not a continuously adjustable sensor option like
        exposure/gain - so this is the intersection of what both streams
        support rather than a min/max range."""
        sensor = self._require_connected()
        depth_fps: set[int] = set()
        ir_fps: set[int] = set()
        for profile in sensor.get_stream_profiles():
            video = profile.as_video_stream_profile()
            if video is None or (video.width(), video.height()) != (self._width, self._height):
                continue
            if profile.stream_type() == self._rs.stream.depth and profile.format() == self._rs.format.z16:
                depth_fps.add(video.fps())
            elif (
                profile.stream_type() == self._rs.stream.infrared
                and profile.stream_index() == _INFRARED_STREAM_INDEX
                and profile.format() == self._rs.format.y8
            ):
                ir_fps.add(video.fps())
        common = sorted(depth_fps & ir_fps)
        return common or [int(self._fps)]

    def get_parameters(self) -> CameraParameters:
        sensor = self._require_connected()

        def read(option) -> ParameterRange:
            option_range = sensor.get_option_range(option)
            return ParameterRange(
                minimum=option_range.min,
                maximum=option_range.max,
                current=sensor.get_option(option),
            )

        supported_fps = self._supported_fps_values()
        return CameraParameters(
            exposure_time_us=read(self._rs.option.exposure),
            gain=read(self._rs.option.gain),
            frame_rate_fps=ParameterRange(
                minimum=min(supported_fps), maximum=max(supported_fps), current=self._fps
            ),
        )

    def _set_clamped(self, option, value: float) -> None:
        sensor = self._require_connected()
        option_range = sensor.get_option_range(option)
        clamped = max(option_range.min, min(option_range.max, value))
        if clamped != value:
            logger.warning(
                "value=%.4f out of range [%.4f, %.4f]; clamped to %.4f",
                value, option_range.min, option_range.max, clamped,
            )
        sensor.set_option(option, clamped)
        logger.info("Set option to %.4f", clamped)

    def set_exposure_time_us(self, value: float) -> None:
        self._set_clamped(self._rs.option.exposure, value)

    def set_gain(self, value: float) -> None:
        self._set_clamped(self._rs.option.gain, value)

    def set_frame_rate(self, value: float) -> None:
        self._require_connected()
        supported = self._supported_fps_values()
        snapped = min(supported, key=lambda fps: abs(fps - value))
        if snapped != value:
            logger.warning(
                "Frame rate %.2f not supported at %dx%d (supported: %s); using nearest value %d",
                value, self._width, self._height, supported, snapped,
            )
        self._fps = snapped
        logger.info("Frame rate set to %d fps (applied on next stream start)", snapped)
        if self._acquiring:
            self.stop_acquisition()
            self.start_acquisition()

    def start_acquisition(self) -> None:
        self._require_connected()
        if self._acquiring:
            return

        serial = self._device.get_info(self._rs.camera_info.serial_number)
        config = self._rs.config()
        config.enable_device(serial)
        config.enable_stream(
            self._rs.stream.depth, self._width, self._height, self._rs.format.z16, int(self._fps)
        )
        config.enable_stream(
            self._rs.stream.infrared,
            _INFRARED_STREAM_INDEX,
            self._width,
            self._height,
            self._rs.format.y8,
            int(self._fps),
        )

        self._pipeline = self._rs.pipeline()
        self._pipeline.start(config)
        self._acquiring = True
        logger.info(
            "Acquisition started (%dx%d @ %sfps)", self._width, self._height, self._fps
        )

    def stop_acquisition(self) -> None:
        if not self._acquiring:
            return
        self._pipeline.stop()
        self._pipeline = None
        self._acquiring = False
        logger.info("Acquisition stopped")

    def get_frame(self, timeout_ms: int = 2000) -> Frame:
        if not self._acquiring:
            raise CameraNotConnectedError("Acquisition is not running; call start_acquisition() first")

        frames = self._pipeline.wait_for_frames(timeout_ms)
        depth_frame = frames.get_depth_frame()
        ir_frame = frames.get_infrared_frame(_INFRARED_STREAM_INDEX)
        if not depth_frame or not ir_frame:
            raise CameraError("Frame is missing depth or infrared data")

        depth_scale_m = self._depth_sensor.get_depth_scale()
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_mm = depth_raw.astype(np.float32) * depth_scale_m * 1000.0

        return Frame(
            intensity=np.asanyarray(ir_frame.get_data()),
            depth_mm=depth_mm,
            confidence=None,
            device_timestamp_ns=int(depth_frame.get_timestamp() * 1e6),
            host_timestamp_ns=time.time_ns(),
        )
