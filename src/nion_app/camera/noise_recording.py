"""Per-pixel depth noise recording over a rectangular ROI.

Per the spec, each sample keeps the pixel id (x, y) and the unix timestamp of
that recording alongside the depth reading, rather than just a per-frame
depth stack - so recordings stay self-describing and are simple to persist
(save/load) or re-slice later (e.g. a subset of pixels or a time window).
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from nion_app.camera.backend import Frame


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
    pixel_x: np.ndarray  # (N,) uint16 - one entry per (frame, pixel) sample
    pixel_y: np.ndarray  # (N,) uint16
    timestamp_unix: np.ndarray  # (N,) float64, seconds since epoch
    depth_mm: np.ndarray  # (N,) float32
    ir_snapshot: np.ndarray  # (height, width) - IR intensity crop over the ROI,
    # captured once at recording start, so results can show what was measured

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
        return self.timestamp_unix.reshape(self.frame_count, self.roi.height, self.roi.width)

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
        results view can share one y-axis scale for fair comparison."""
        depth_grid = self.depth_grid_by_frame()
        valid = depth_grid > 0
        bin_idx = np.digitize(depth_grid, bin_edges)
        n_bins = len(bin_edges) - 1
        max_count = 0
        for b in range(1, n_bins + 1):
            count = np.count_nonzero(valid & (bin_idx == b), axis=0)
            max_count = max(max_count, int(count.max()))
        return max_count

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


class NoiseRecorder:
    """Accumulates samples for one ROI/duration, fed one frame at a time by
    the app's single shared acquisition loop (never acquires frames itself,
    so it can't race the 3D viewer for the same camera buffers)."""

    def __init__(self, roi: ROI, duration_s: float) -> None:
        self.roi = roi
        self.duration_s = duration_s
        self._start_time: float | None = None
        self._x_chunks: list[np.ndarray] = []
        self._y_chunks: list[np.ndarray] = []
        self._ts_chunks: list[np.ndarray] = []
        self._depth_chunks: list[np.ndarray] = []
        self._ir_snapshot: np.ndarray | None = None

        x_coords, y_coords = np.meshgrid(
            np.arange(roi.x, roi.x + roi.width), np.arange(roi.y, roi.y + roi.height)
        )
        self._x_flat = x_coords.ravel().astype(np.uint16)
        self._y_flat = y_coords.ravel().astype(np.uint16)

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
        unix_timestamp_s = frame.host_timestamp_ns / 1e9

        self._x_chunks.append(self._x_flat)
        self._y_chunks.append(self._y_flat)
        self._ts_chunks.append(np.full(self._x_flat.shape, unix_timestamp_s, dtype=np.float64))
        self._depth_chunks.append(depth_roi.ravel().astype(np.float32))

    def finalize(self) -> NoiseRecording:
        assert self._ir_snapshot is not None, "finalize() called before any frame was added"
        return NoiseRecording(
            roi=self.roi,
            duration_s=self.duration_s,
            pixel_x=np.concatenate(self._x_chunks),
            pixel_y=np.concatenate(self._y_chunks),
            timestamp_unix=np.concatenate(self._ts_chunks),
            depth_mm=np.concatenate(self._depth_chunks),
            ir_snapshot=self._ir_snapshot,
        )
