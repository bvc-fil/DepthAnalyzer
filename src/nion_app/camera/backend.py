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
    """One synchronized capture from the depth sensor."""

    intensity: np.ndarray  # (H, W) uint8 - IR/intensity image, used for ROI selection
    depth_mm: np.ndarray  # (H, W) float32 - Z distance in millimeters
    # (H, W) uint16 per-pixel measurement confidence, or None on sensors (e.g.
    # stereo depth cameras) that don't expose a hardware confidence channel.
    confidence: np.ndarray | None
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

    @property
    @abstractmethod
    def device_info(self) -> DeviceInfo | None:
        """Identity of the currently connected device, or None if not
        connected - so callers (e.g. a saved recording) can record which
        physical sensor produced their data."""

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

    def load_calibration_file(self, path: str, file_name: str | None = None) -> None:
        """Loads a user-supplied calibration file onto the device, for
        sensors that need one uploaded before their depth output is usable.
        file_name selects which device-side file slot to target; None means
        the backend's own default. Most backends don't need this - the
        default here raises, and only the backends that do should override it."""
        raise CameraError(f"{type(self).__name__} does not support loading a calibration file")
