"""Per-pixel depth noise recording over a rectangular ROI.

Per the spec, every recorded depth sample is identified by its pixel id
(x, y) and the unix timestamp of that recording, rather than just a bare
per-frame depth stack - so recordings stay self-describing and are simple to
re-slice later (e.g. a subset of pixels or a time window). Pixel ids and
timestamps are stored at their natural granularity rather than repeated once
per (frame, pixel) sample: pixel_x/pixel_y are the same ROI grid for every
frame, and every pixel in a frame shares that frame's one timestamp, so both
are kept at (height, width) / (frame_count,) and broadcast back out to
per-sample shape on read. Only depth_mm genuinely varies per (frame, pixel),
so it's the only field stored at full sample granularity - this keeps
long/large recordings from ballooning in memory for no informational gain.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from nion_app.camera.backend import CameraParameters, DeviceInfo, Frame

_UNKNOWN_PARAMETER = float("nan")  # camera parameter wasn't available when recorded
_UNKNOWN_DEVICE = "unknown"  # no live backend (or no connected device) at recording time


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    width: int
    height: int


@dataclass
class NoiseRecording:
    roi: ROI
    duration_s: float
    pixel_x: np.ndarray  # (height, width) uint16 - the ROI's pixel-x grid, same for every frame
    pixel_y: np.ndarray  # (height, width) uint16
    timestamp_unix: np.ndarray  # (frame_count,) float64, seconds since epoch - one per frame
    depth_mm: np.ndarray  # (frame_count * height * width,) float32 - one entry per (frame, pixel)
    ir_snapshot: np.ndarray  # (height, width) - IR intensity crop over the ROI,
    # captured once at recording start, so results can show what was measured

    # The camera's current exposure/gain/frame-rate at recording start, so a
    # saved dataset records what settings produced it, not just the depth
    # data. NaN when the caller didn't have a live backend to read from (e.g.
    # a NoiseRecorder built without camera_parameters).
    exposure_time_us: float = _UNKNOWN_PARAMETER
    gain: float = _UNKNOWN_PARAMETER
    frame_rate_fps: float = _UNKNOWN_PARAMETER

    # Which physical sensor recorded this - so a saved dataset is still
    # identifiable after the fact, e.g. if recordings from more than one
    # camera model end up being compared later.
    device_model_name: str = _UNKNOWN_DEVICE
    device_serial_number: str = _UNKNOWN_DEVICE

    @property
    def frame_count(self) -> int:
        pixels_per_frame = self.roi.width * self.roi.height
        return len(self.depth_mm) // pixels_per_frame if pixels_per_frame else 0

    def depth_grid_by_frame(self) -> np.ndarray:
        """(frames, height, width) - valid only while every recorded frame
        contributed a full, identically-ordered ROI grid, which is how
        NoiseRecorder always appends samples."""
        return self.depth_mm.reshape(self.frame_count, self.roi.height, self.roi.width)

    def timestamp_grid_by_frame(self) -> np.ndarray:
        """(frames, height, width) - every pixel in a frame shares that
        frame's single recorded timestamp; broadcast here on read rather than
        stored per-pixel."""
        return np.broadcast_to(
            self.timestamp_unix[:, np.newaxis, np.newaxis],
            (self.frame_count, self.roi.height, self.roi.width),
        )

    def per_pixel_mean_std(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean_grid, std_grid), each (height, width) in millimeters.

        depth_mm == 0 marks an invalid reading (Scan3dInvalidDataValue on the
        Nion), not a real zero-distance measurement - a pixel that drops out
        intermittently would otherwise register a wildly inflated "noise"
        std dominated by the dropout, not the sensor. Those samples are
        excluded here; a pixel with zero valid samples gets NaN.
        """
        depth_grid = self.depth_grid_by_frame()
        masked = np.where(depth_grid > 0, depth_grid, np.nan)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            # A pixel with zero valid samples (all dropouts) legitimately
            # produces NaN here - expected and handled by callers, not a bug.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(masked, axis=0), np.nanstd(masked, axis=0)

    def valid_sample_count_grid(self) -> np.ndarray:
        """(height, width) count of non-invalid (depth_mm > 0) samples per pixel."""
        return (self.depth_grid_by_frame() > 0).sum(axis=0)

    def all_valid_depths(self) -> np.ndarray:
        """Every valid (depth_mm > 0) sample across every pixel, flattened -
        the population behind the recording-wide depth histogram."""
        return self.depth_mm[self.depth_mm > 0]

    def valid_std_values(self) -> np.ndarray:
        """Every pixel's std. dev., excluding pixels with zero valid samples
        (NaN) - the population behind the recording-wide std. dev. histogram."""
        _mean_grid, std_grid = self.per_pixel_mean_std()
        return std_grid[~np.isnan(std_grid)]

    def depth_histogram_bin_edges(self, bins: int = 30) -> np.ndarray:
        """Shared bin edges spanning every valid depth in the recording, so
        every per-pixel histogram can be drawn on the same x-axis."""
        depths = self.all_valid_depths()
        if depths.size == 0:
            return np.linspace(0.0, 1.0, bins + 1)
        low, high = depths.min(), depths.max()
        if low == high:
            low, high = low - 0.5, high + 0.5
        return np.linspace(low, high, bins + 1)

    def max_pixel_histogram_count(self, bin_edges: np.ndarray) -> int:
        """Highest single-bin count across every individual pixel's own
        histogram (using bin_edges), so every per-pixel histogram in the
        results view can share one y-axis scale for fair comparison.

        Built as one pass over the data - digitize once, then a single
        bincount over a combined bin*pixel index - rather than looping over
        each bin and rescanning the full (frame, pixel) array per bin: that
        loop reprocesses every sample once per bin (O(bins * frames *
        pixels)); this touches every sample a constant number of times
        (O(frames * pixels)), which is what keeps a large recording's
        results view from stalling the UI thread for seconds.
        """
        depth_grid = self.depth_grid_by_frame()
        frame_count, height, width = depth_grid.shape
        n_pixels = height * width
        n_bins = len(bin_edges) - 1
        if n_pixels == 0 or n_bins <= 0:
            return 0

        valid = depth_grid > 0
        # digitize returns 1..n_bins for values inside [edges[0], edges[-1]);
        # shift to 0..n_bins-1 so it can be used directly as an index.
        bin_idx = np.digitize(depth_grid, bin_edges) - 1
        in_range = valid & (bin_idx >= 0) & (bin_idx < n_bins)
        if not np.any(in_range):
            return 0

        pixel_idx = np.broadcast_to(
            np.arange(n_pixels, dtype=np.int64).reshape(1, height, width), depth_grid.shape
        )
        combined = bin_idx[in_range].astype(np.int64) * n_pixels + pixel_idx[in_range]
        counts = np.bincount(combined, minlength=n_bins * n_pixels)
        return int(counts.max())

    def pixel_series(
        self, x: int, y: int, include_invalid: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (timestamps_unix, depth_mm) for one pixel, in recording
        order. Invalid (depth_mm == 0) samples are dropped unless
        include_invalid is set."""
        row, col = y - self.roi.y, x - self.roi.x
        timestamps = self.timestamp_grid_by_frame()[:, row, col]
        depths = self.depth_grid_by_frame()[:, row, col]
        if include_invalid:
            return timestamps, depths
        valid = depths > 0
        return timestamps[valid], depths[valid]

    def save(self, path: str) -> None:
        """Writes every field to a single compressed .npz archive - every
        field is already a plain numpy array or scalar, so no other format
        is needed to round-trip a recording exactly."""
        np.savez_compressed(
            path,
            roi_x=self.roi.x,
            roi_y=self.roi.y,
            roi_width=self.roi.width,
            roi_height=self.roi.height,
            duration_s=self.duration_s,
            pixel_x=self.pixel_x,
            pixel_y=self.pixel_y,
            timestamp_unix=self.timestamp_unix,
            depth_mm=self.depth_mm,
            ir_snapshot=self.ir_snapshot,
            exposure_time_us=self.exposure_time_us,
            gain=self.gain,
            frame_rate_fps=self.frame_rate_fps,
            device_model_name=self.device_model_name,
            device_serial_number=self.device_serial_number,
        )

    @classmethod
    def load(cls, path: str) -> "NoiseRecording":
        data = np.load(path)
        roi = ROI(
            x=int(data["roi_x"]),
            y=int(data["roi_y"]),
            width=int(data["roi_width"]),
            height=int(data["roi_height"]),
        )
        # exposure_time_us/gain/frame_rate_fps/device_* were added after this
        # format shipped, so older files may not have them - fall back to
        # unknown rather than failing to load.
        return cls(
            roi=roi,
            duration_s=float(data["duration_s"]),
            pixel_x=data["pixel_x"],
            pixel_y=data["pixel_y"],
            timestamp_unix=data["timestamp_unix"],
            depth_mm=data["depth_mm"],
            ir_snapshot=data["ir_snapshot"],
            exposure_time_us=float(data["exposure_time_us"])
            if "exposure_time_us" in data.files
            else _UNKNOWN_PARAMETER,
            gain=float(data["gain"]) if "gain" in data.files else _UNKNOWN_PARAMETER,
            frame_rate_fps=float(data["frame_rate_fps"])
            if "frame_rate_fps" in data.files
            else _UNKNOWN_PARAMETER,
            device_model_name=str(data["device_model_name"])
            if "device_model_name" in data.files
            else _UNKNOWN_DEVICE,
            device_serial_number=str(data["device_serial_number"])
            if "device_serial_number" in data.files
            else _UNKNOWN_DEVICE,
        )


HIST_BINS = 300  # 10x narrower bars than a plain 30-bin histogram, to make the spread easier to see


@dataclass(frozen=True)
class RecordingStats:
    """Every derived numpy result NoiseResultsView needs before it can draw
    anything. Computing these is the expensive part of showing results (see
    max_pixel_histogram_count) - pulling them out into one function lets the
    caller run that work off the Qt main thread and hand the view only
    already-computed arrays."""

    mean_grid: np.ndarray
    std_grid: np.ndarray
    hist_bin_edges: np.ndarray
    hist_ymax: int


def compute_recording_stats(recording: NoiseRecording, bins: int = HIST_BINS) -> RecordingStats:
    mean_grid, std_grid = recording.per_pixel_mean_std()
    hist_bin_edges = recording.depth_histogram_bin_edges(bins=bins)
    hist_ymax = recording.max_pixel_histogram_count(hist_bin_edges)
    return RecordingStats(mean_grid, std_grid, hist_bin_edges, hist_ymax)


class NoiseRecorder:
    """Accumulates samples for one ROI/duration, fed one frame at a time by
    the app's single shared acquisition loop (never acquires frames itself,
    so it can't race the 3D viewer for the same camera buffers)."""

    def __init__(
        self,
        roi: ROI,
        duration_s: float,
        camera_parameters: CameraParameters | None = None,
        device_info: DeviceInfo | None = None,
    ) -> None:
        self.roi = roi
        self.duration_s = duration_s
        self._start_time: float | None = None
        self._timestamps: list[float] = []
        self._depth_chunks: list[np.ndarray] = []
        self._ir_snapshot: np.ndarray | None = None
        # Snapshotted once at construction rather than re-read per frame -
        # exposure/gain/frame rate are assumed stable for the recording's
        # duration, and this is just for the saved dataset's audit trail.
        if camera_parameters is None:
            self._exposure_time_us = _UNKNOWN_PARAMETER
            self._gain = _UNKNOWN_PARAMETER
            self._frame_rate_fps = _UNKNOWN_PARAMETER
        else:
            self._exposure_time_us = camera_parameters.exposure_time_us.current
            self._gain = camera_parameters.gain.current
            self._frame_rate_fps = camera_parameters.frame_rate_fps.current

        if device_info is None:
            self._device_model_name = _UNKNOWN_DEVICE
            self._device_serial_number = _UNKNOWN_DEVICE
        else:
            self._device_model_name = device_info.model_name
            self._device_serial_number = device_info.serial_number

        # The ROI's pixel grid is identical for every frame, so it's computed
        # once here rather than re-appended per frame.
        x_coords, y_coords = np.meshgrid(
            np.arange(roi.x, roi.x + roi.width), np.arange(roi.y, roi.y + roi.height)
        )
        self._pixel_x = x_coords.astype(np.uint16)
        self._pixel_y = y_coords.astype(np.uint16)

    @property
    def is_done(self) -> bool:
        return self._start_time is not None and self.elapsed_s >= self.duration_s

    @property
    def elapsed_s(self) -> float:
        return 0.0 if self._start_time is None else time.time() - self._start_time

    def add_frame(self, frame: Frame) -> None:
        roi = self.roi
        if self._start_time is None:
            self._start_time = time.time()
            # One representative IR snapshot is enough for a visual reference
            # alongside the mean/std heatmaps - no need to average frames.
            self._ir_snapshot = frame.intensity[
                roi.y : roi.y + roi.height, roi.x : roi.x + roi.width
            ].copy()

        depth_roi = frame.depth_mm[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]

        self._timestamps.append(frame.host_timestamp_ns / 1e9)
        self._depth_chunks.append(depth_roi.ravel().astype(np.float32))

    def finalize(self) -> NoiseRecording:
        assert self._ir_snapshot is not None, "finalize() called before any frame was added"
        return NoiseRecording(
            roi=self.roi,
            duration_s=self.duration_s,
            pixel_x=self._pixel_x,
            pixel_y=self._pixel_y,
            timestamp_unix=np.array(self._timestamps, dtype=np.float64),
            depth_mm=np.concatenate(self._depth_chunks),
            ir_snapshot=self._ir_snapshot,
            exposure_time_us=self._exposure_time_us,
            gain=self._gain,
            frame_rate_fps=self._frame_rate_fps,
            device_model_name=self._device_model_name,
            device_serial_number=self._device_serial_number,
        )
