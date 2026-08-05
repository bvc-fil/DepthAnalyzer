"""Camera backend abstraction.

Keeping the SDK-specific implementation behind this interface means the rest of
the application (setup wizard, viewer, noise recorder) never touches ids_peak
directly, and a mock backend can stand in when no camera is attached.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class CameraError(RuntimeError):
    """Base class for camera-related failures."""


class CameraNotFoundError(CameraError):
    """No reachable device was found (unpowered or not physically connected)."""


class CameraNotConnectedError(CameraError):
    """An operation requiring an open connection was attempted before connect()."""


@dataclass(frozen=True)
class ParameterRange:
    minimum: float
    maximum: float
    current: float


@dataclass(frozen=True)
class CameraParameters:
    exposure_time_us: ParameterRange
    gain: ParameterRange
    frame_rate_fps: ParameterRange


@dataclass(frozen=True)
class Frame:
    """One synchronized capture from the depth sensor.

    depth_mm holds the raw per-pixel Z distance already scaled to millimeters;
    it is not yet a triangulated XYZ point cloud (that requires the lens
    calibration step covered separately in the viewer module).
    """

    intensity: np.ndarray  # (H, W) uint8 - IR/intensity image, used for ROI selection
    depth_mm: np.ndarray  # (H, W) float32 - Z distance in millimeters
    confidence: np.ndarray  # (H, W) uint16 - per-pixel measurement confidence
    device_timestamp_ns: int
    host_timestamp_ns: int


@dataclass(frozen=True)
class DeviceInfo:
    serial_number: str
    model_name: str
    display_name: str


class CameraBackend(ABC):
    @abstractmethod
    def discover(self) -> list[DeviceInfo]:
        """List reachable devices. An empty list means the camera is not
        powered, not physically connected, or not on the expected network."""

    @abstractmethod
    def connect(self, serial_number: str | None = None) -> None:
        """Open a control connection to a device (the first discovered one if
        serial_number is None)."""

    @abstractmethod
    def disconnect(self) -> None: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def available_configurations(self) -> list[str]:
        """Names of configurations (GenICam UserSets) stored on the device."""

    @abstractmethod
    def load_configuration(self, name: str = "Default") -> None: ...

    @abstractmethod
    def get_parameters(self) -> CameraParameters: ...

    @abstractmethod
    def set_exposure_time_us(self, value: float) -> None: ...

    @abstractmethod
    def set_gain(self, value: float) -> None: ...

    @abstractmethod
    def set_frame_rate(self, value: float) -> None: ...

    @abstractmethod
    def start_acquisition(self) -> None: ...

    @abstractmethod
    def stop_acquisition(self) -> None: ...

    @abstractmethod
    def get_frame(self, timeout_ms: int = 2000) -> Frame: ...
