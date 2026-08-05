#!/usr/bin/env python3
"""Runs the guided connection flow against the real camera and captures one
frame to prove the whole pipeline (connect, configure, acquire) works."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nion_app.logging_setup import configure_logging
from nion_app.camera.backend import CameraError
from nion_app.camera.connection_flow import ConnectionStepFailed, run_guided_connection
from nion_app.camera.ids_backend import IdsPeakBackend

logger = logging.getLogger(__name__)


def main() -> int:
    configure_logging()
    backend = IdsPeakBackend()
    try:
        run_guided_connection(backend, configuration_name="Default")
    except ConnectionStepFailed as exc:
        logger.error("Connection failed at step '%s': %s", exc.step, exc.reason)
        return 1

    backend.start_acquisition()
    try:
        frame = backend.get_frame()
        logger.info(
            "Captured frame: intensity=%s depth_mm(min/max)=%.1f/%.1f confidence=%s "
            "device_ts_ns=%d",
            frame.intensity.shape,
            float(frame.depth_mm.min()),
            float(frame.depth_mm.max()),
            frame.confidence.shape,
            frame.device_timestamp_ns,
        )
    finally:
        backend.stop_acquisition()
        backend.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
