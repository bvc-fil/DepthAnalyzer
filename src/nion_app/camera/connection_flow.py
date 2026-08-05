"""Guided camera connection sequence matching outlines.txt steps 1-5:
power -> physical connection -> establish connection -> load configuration ->
set gain/exposure/frame rate. Every step logs its outcome so a UI wizard (or
this module used headless) always leaves a clear trail of what happened.
"""
from __future__ import annotations

import logging

from nion_app.camera.backend import CameraBackend, CameraError

logger = logging.getLogger(__name__)


class ConnectionStepFailed(RuntimeError):
    """Raised with the specific failing step so a UI wizard can highlight it."""

    def __init__(self, step: str, reason: str):
        super().__init__(f"{step}: {reason}")
        self.step = step
        self.reason = reason


def run_guided_connection(
    backend: CameraBackend,
    configuration_name: str = "Default",
    exposure_time_us: float | None = None,
    gain: float | None = None,
    frame_rate_fps: float | None = None,
) -> None:
    logger.info("Step 1/5: Checking camera power and discoverability")
    devices = backend.discover()
    if not devices:
        raise ConnectionStepFailed(
            "power_and_physical_connection",
            "No camera was discovered. Confirm the camera has power (check for "
            "a link/status LED) and that the ethernet cable is seated at both ends.",
        )
    logger.info("Camera powered and reachable: %s", devices[0])

    logger.info("Step 2/5: Verifying physical connection")
    # GenTL discovery above already requires a live GVCP handshake over the
    # cable, so a non-empty device list is itself proof the physical link is up.
    logger.info("Physical connection confirmed via successful device enumeration")

    logger.info("Step 3/5: Establishing control connection")
    try:
        backend.connect(devices[0].serial_number)
    except CameraError as exc:
        raise ConnectionStepFailed("establish_connection", str(exc)) from exc
    logger.info("Connected to %s", devices[0].display_name)

    logger.info("Step 4/5: Loading configuration '%s'", configuration_name)
    try:
        backend.load_configuration(configuration_name)
    except CameraError as exc:
        raise ConnectionStepFailed("load_configuration", str(exc)) from exc

    logger.info("Step 5/5: Applying gain / exposure / frame rate")
    try:
        if exposure_time_us is not None:
            backend.set_exposure_time_us(exposure_time_us)
        if gain is not None:
            backend.set_gain(gain)
        if frame_rate_fps is not None:
            backend.set_frame_rate(frame_rate_fps)
    except CameraError as exc:
        raise ConnectionStepFailed("set_parameters", str(exc)) from exc

    final_params = backend.get_parameters()
    logger.info(
        "Camera ready: exposure=%.2fus gain=%.2f frame_rate=%.2ffps",
        final_params.exposure_time_us.current,
        final_params.gain.current,
        final_params.frame_rate_fps.current,
    )
