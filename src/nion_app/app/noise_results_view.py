"""Post-recording visualization: per-pixel mean-depth, standard-deviation and
IR-reference heatmaps over the ROI (click any of them to drill in) plus that
pixel's depth histogram and time-ordered sequence, and a second tab of
recording-wide statistics - the outline's "histograms... and standard
deviations for each individual pixel" requirement, extended per follow-up
requests for a time sequence view, adjustable std. dev. gradient, an ROI IR
reference image, consistent histogram axes, and recording-wide summaries."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from nion_app.camera.noise_recording import NoiseRecording, RecordingStats

# A full rainbow gradient for the mean-depth image, distinct from the
# std. dev. image's colormap, with extreme readings pinned to the extreme
# ends of the gradient (vmin/vmax set to the data's actual min/max below).
_MEAN_CMAP = "rainbow"
_IR_CMAP = "gray"
# Perceptually-uniform sequential colormaps only (no rainbow/jet) so the
# std. dev. gradient stays readable as a magnitude encoding while still
# giving the user a choice of how contrast is distributed across it.
_STD_CMAP_OPTIONS = ["inferno", "viridis", "plasma", "magma", "cividis", "Blues", "YlOrRd"]
_STD_CMAP_DEFAULT = "inferno"

# One accent color used consistently across every single-series chart added
# here (histograms, sequence line, scatter) so they read as one family.
_ACCENT = "#2a78d6"
_MARKER_COLOR = "#e34948"


class NoiseResultsView(QWidget):
    def __init__(
        self, recording: NoiseRecording, stats: RecordingStats, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._recording = recording
        self._mean_grid, self._std_grid = stats.mean_grid, stats.std_grid
        self._hist_bin_edges = stats.hist_bin_edges
        self._hist_ymax = stats.hist_ymax

        roi = recording.roi
        self._selected_x, self._selected_y = roi.x, roi.y
        self._std_clip_low, self._std_clip_high = 0.0, 100.0

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        tabs.addTab(self._build_per_pixel_tab(), "Per-Pixel Heatmaps")
        tabs.addTab(self._build_overall_tab(), "Overall Statistics")

        self._select_pixel(roi.x, roi.y)  # default: the ROI's top-left pixel
        self._draw_overall_stats()

    # -- Tab 1: per-pixel heatmaps + drill-in ---------------------------------

    def _build_per_pixel_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        # A 3-column header mirroring the mean/std/IR heatmap columns below,
        # so the gradient controls sit as a small sub-region directly above
        # the std. dev. column instead of a full-width toolbar.
        header = QGridLayout()
        header.setColumnStretch(0, 1)
        header.setColumnStretch(1, 1)
        header.setColumnStretch(2, 1)

        header.addWidget(QLabel("Click a heatmap below to inspect a pixel."), 0, 0)

        std_controls_box = QGroupBox("Std. dev. gradient")
        std_controls = QHBoxLayout(std_controls_box)
        std_controls.setContentsMargins(6, 2, 6, 2)
        std_controls.setSpacing(4)

        self._std_cmap_box = QComboBox()
        self._std_cmap_box.addItems(_STD_CMAP_OPTIONS)
        self._std_cmap_box.setCurrentText(_STD_CMAP_DEFAULT)
        self._std_cmap_box.setMaximumWidth(90)
        self._std_cmap_box.currentTextChanged.connect(self._on_std_gradient_changed)
        std_controls.addWidget(self._std_cmap_box)

        std_controls.addWidget(QLabel("clip"))
        self._std_low_box = QDoubleSpinBox()
        self._std_low_box.setRange(0.0, 99.0)
        self._std_low_box.setSuffix("%")
        self._std_low_box.setMaximumWidth(60)
        self._std_low_box.valueChanged.connect(self._on_std_gradient_changed)
        std_controls.addWidget(self._std_low_box)

        std_controls.addWidget(QLabel("-"))
        self._std_high_box = QDoubleSpinBox()
        self._std_high_box.setRange(1.0, 100.0)
        self._std_high_box.setValue(100.0)
        self._std_high_box.setSuffix("%")
        self._std_high_box.setMaximumWidth(60)
        self._std_high_box.valueChanged.connect(self._on_std_gradient_changed)
        std_controls.addWidget(self._std_high_box)

        header.addWidget(std_controls_box, 0, 1)

        self._selected_label = QLabel()
        header.addWidget(self._selected_label, 0, 2)
        tab_layout.addLayout(header)

        self._figure = Figure(figsize=(13, 10), constrained_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        tab_layout.addWidget(self._canvas)

        # 3 rows now: heatmaps, then the sequence with its FFT stacked
        # underneath it (sharing the histogram's row so there's no dead
        # space - the histogram spans both of those rows instead).
        grid = self._figure.add_gridspec(3, 3)
        self._mean_ax = self._figure.add_subplot(grid[0, 0])
        self._std_ax = self._figure.add_subplot(grid[0, 1])
        self._ir_ax = self._figure.add_subplot(grid[0, 2])
        self._hist_ax = self._figure.add_subplot(grid[1:, 0])
        self._seq_ax = self._figure.add_subplot(grid[1, 1:])
        self._fft_ax = self._figure.add_subplot(grid[2, 1:])

        self._canvas.mpl_connect("button_press_event", self._on_heatmap_click)
        self._init_heatmaps()
        return tab

    def _heatmap_extent(self) -> tuple[float, float, float, float]:
        roi = self._recording.roi
        return (roi.x, roi.x + roi.width, roi.y + roi.height, roi.y)

    def _add_selected_pixel_marker(self, ax):
        (marker,) = ax.plot(
            [], [], marker="+", markersize=12, markeredgewidth=2, color=_MARKER_COLOR
        )
        return marker

    def _init_heatmaps(self) -> None:
        """Creates each heatmap's image, colorbar, and selected-pixel marker
        exactly once. Later updates (a new pixel click, a gradient change)
        mutate these in place rather than clearing the axes and re-adding a
        colorbar - figure.colorbar() puts the colorbar in its own axes, so
        recreating it on every redraw left a stack of stale color keys behind
        instead of replacing the previous one."""
        extent = self._heatmap_extent()

        ax = self._mean_ax
        self._mean_image = ax.imshow(
            self._mean_grid,
            extent=extent,
            cmap=_MEAN_CMAP,
            vmin=np.nanmin(self._mean_grid),
            vmax=np.nanmax(self._mean_grid),
        )
        ax.set_title("Mean depth (mm)", fontsize=10)
        ax.set_xlabel("pixel x")
        ax.set_ylabel("pixel y")
        self._mean_marker = self._add_selected_pixel_marker(ax)
        self._mean_cbar = self._figure.colorbar(
            self._mean_image, ax=ax, label="mm", shrink=0.85, pad=0.02
        )

        ax = self._std_ax
        vmin = np.nanpercentile(self._std_grid, self._std_clip_low)
        vmax = np.nanpercentile(self._std_grid, self._std_clip_high)
        self._std_image = ax.imshow(
            self._std_grid,
            extent=extent,
            cmap=self._std_cmap_box.currentText(),
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title("Std. dev. (mm)", fontsize=10)
        ax.set_xlabel("pixel x")
        ax.set_ylabel("pixel y")
        self._std_marker = self._add_selected_pixel_marker(ax)
        self._std_cbar = self._figure.colorbar(
            self._std_image, ax=ax, label="mm", shrink=0.85, pad=0.02
        )

        ax = self._ir_ax
        ir = self._recording.ir_snapshot.astype(np.float32)
        # Same 2nd/98th-percentile contrast stretch used for the live ROI
        # selector view, for a comparably legible reference image.
        low, high = np.percentile(ir, [2, 98])
        span = max(high - low, 1.0)
        self._ir_image = ax.imshow(ir, extent=extent, cmap=_IR_CMAP, vmin=low, vmax=low + span)
        ax.set_title("IR reference", fontsize=10)
        ax.set_xlabel("pixel x")
        ax.set_ylabel("pixel y")
        self._ir_marker = self._add_selected_pixel_marker(ax)
        self._ir_cbar = self._figure.colorbar(
            self._ir_image, ax=ax, label="intensity", shrink=0.85, pad=0.02
        )

    def _update_selected_pixel_markers(self) -> None:
        x, y = self._selected_x + 0.5, self._selected_y + 0.5
        for marker in (self._mean_marker, self._std_marker, self._ir_marker):
            marker.set_data([x], [y])

    def _update_std_gradient(self) -> None:
        vmin = np.nanpercentile(self._std_grid, self._std_clip_low)
        vmax = np.nanpercentile(self._std_grid, self._std_clip_high)
        self._std_image.set_cmap(self._std_cmap_box.currentText())
        self._std_image.set_clim(vmin, vmax)
        self._std_cbar.update_normal(self._std_image)

    def _draw_histogram(self, x: int, y: int) -> None:
        roi = self._recording.roi
        row, col = y - roi.y, x - roi.x
        _timestamps, depths = self._recording.pixel_series(x, y)
        ax = self._hist_ax
        ax.clear()
        mean, std = self._mean_grid[row, col], self._std_grid[row, col]
        if len(depths) == 0:
            ax.set_title(f"Pixel ({x},{y})\nno valid readings (all dropouts)", fontsize=10)
        else:
            ax.hist(depths, bins=self._hist_bin_edges, color=_ACCENT, edgecolor="none")
            ax.set_title(
                f"Pixel ({x},{y})\nmean={mean:.1f}mm  std={std:.2f}mm  n={len(depths)}",
                fontsize=10,
            )
        # Fixed axes (shared across every pixel) so histograms are directly
        # comparable to one another.
        ax.set_xlim(self._hist_bin_edges[0], self._hist_bin_edges[-1])
        ax.set_ylim(0, max(self._hist_ymax, 1) * 1.05)
        ax.set_xlabel("depth (mm)")
        ax.set_ylabel("count")

    def _draw_sequence(self, x: int, y: int) -> None:
        timestamps, depths = self._recording.pixel_series(x, y)
        ax = self._seq_ax
        ax.clear()
        if len(depths) == 0:
            ax.set_title(f"Pixel ({x},{y}) depth over time\nno valid readings", fontsize=10)
        else:
            start_time = self._recording.timestamp_unix.min()
            elapsed_s = timestamps - start_time
            ax.plot(elapsed_s, depths, marker=".", markersize=3, linewidth=0.8, color=_ACCENT)
            ax.set_title(f"Pixel ({x},{y}) depth readings in sequence", fontsize=10)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("depth (mm)")

    def _draw_fft(self, x: int, y: int) -> None:
        timestamps, depths = self._recording.pixel_series(x, y)
        ax = self._fft_ax
        ax.clear()
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("amplitude (mm)")

        if len(depths) < 2:
            ax.set_title(f"Pixel ({x},{y}) FFT\nnot enough valid readings", fontsize=10)
            return

        # Dropouts are already excluded by pixel_series(), so the remaining
        # samples aren't perfectly evenly spaced - same approximation the
        # sequence plot above already makes. The median sample spacing is a
        # robust enough stand-in for the sampling rate to get a usable
        # frequency axis out of a plain FFT.
        dt = float(np.median(np.diff(timestamps)))
        if not np.isfinite(dt) or dt <= 0:
            ax.set_title(f"Pixel ({x},{y}) FFT\nirregular sampling, can't compute", fontsize=10)
            return

        detrended = depths - depths.mean()
        spectrum = np.abs(np.fft.rfft(detrended)) / len(detrended)
        freqs = np.fft.rfftfreq(len(detrended), d=dt)
        # Skip the DC bin: it's ~0 after removing the mean, and leaving it in
        # would compress the axis for every frequency that actually matters.
        ax.plot(freqs[1:], spectrum[1:], linewidth=0.8, color=_ACCENT)
        ax.set_title(f"Pixel ({x},{y}) FFT of depth sequence (fs~{1 / dt:.1f}Hz)", fontsize=10)

    def _select_pixel(self, x: int, y: int) -> None:
        roi = self._recording.roi
        row, col = y - roi.y, x - roi.x
        if not (0 <= row < roi.height and 0 <= col < roi.width):
            return
        self._selected_x, self._selected_y = x, y
        self._selected_label.setText(f"Selected pixel: ({x}, {y})")

        self._update_selected_pixel_markers()
        self._draw_histogram(x, y)
        self._draw_sequence(x, y)
        self._draw_fft(x, y)
        self._canvas.draw_idle()

    def _on_heatmap_click(self, event) -> None:
        if event.inaxes not in (self._mean_ax, self._std_ax, self._ir_ax) or event.xdata is None:
            return
        self._select_pixel(int(event.xdata), int(event.ydata))

    def _on_std_gradient_changed(self, _value=None) -> None:
        low, high = self._std_low_box.value(), self._std_high_box.value()
        if low >= high:
            return  # ignore transient invalid states while the user is typing
        self._std_clip_low, self._std_clip_high = low, high
        self._update_std_gradient()
        self._canvas.draw_idle()

    # -- Tab 2: recording-wide statistics -------------------------------------

    def _build_overall_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        self._overall_figure = Figure(figsize=(17, 5), constrained_layout=True)
        self._overall_canvas = FigureCanvasQTAgg(self._overall_figure)
        tab_layout.addWidget(self._overall_canvas)

        grid = self._overall_figure.add_gridspec(1, 4)
        self._scatter_ax = self._overall_figure.add_subplot(grid[0, 0])
        self._all_depth_hist_ax = self._overall_figure.add_subplot(grid[0, 1])
        self._all_std_hist_ax = self._overall_figure.add_subplot(grid[0, 2])
        self._valid_count_hist_ax = self._overall_figure.add_subplot(grid[0, 3])
        return tab

    def _draw_overall_stats(self) -> None:
        mean_flat = self._mean_grid.ravel()
        std_flat = self._std_grid.ravel()
        valid = ~np.isnan(mean_flat) & ~np.isnan(std_flat)

        ax = self._scatter_ax
        ax.scatter(mean_flat[valid], std_flat[valid], s=10, color=_ACCENT, alpha=0.5, edgecolors="none")
        ax.set_title("Mean depth vs. std. dev. (one point per pixel)", fontsize=10)
        ax.set_xlabel("mean depth (mm)")
        ax.set_ylabel("std. dev. (mm)")

        all_depths = self._recording.all_valid_depths()
        ax = self._all_depth_hist_ax
        if all_depths.size:
            ax.hist(all_depths, bins=40, color=_ACCENT, edgecolor="none")
        ax.set_title("Depth histogram (all pixels, all readings)", fontsize=10)
        ax.set_xlabel("depth (mm)")
        ax.set_ylabel("count")

        all_stds = self._recording.valid_std_values()
        ax = self._all_std_hist_ax
        if all_stds.size:
            # 8x the previous bin count, and a log y-axis, so lower bars
            # (rarer std. dev. values) aren't flattened out by taller ones.
            # No edgecolor here: at this bin count the borders would dominate
            # over the (now very thin) bars themselves.
            ax.hist(all_stds, bins=240, color=_ACCENT)
            ax.set_yscale("log")
        ax.set_title("Std. dev. histogram (all pixels with recordings)", fontsize=10)
        ax.set_xlabel("std. dev. (mm)")
        ax.set_ylabel("count")

        recording = self._recording
        roi = recording.roi
        total_pixels = roi.width * roi.height
        avg_fps = recording.frame_count / recording.duration_s if recording.duration_s else 0.0

        valid_counts = recording.valid_sample_count_grid().ravel()
        zero_valid_pixels = int(np.count_nonzero(valid_counts == 0))
        incomplete_pixels = int(np.count_nonzero(valid_counts < recording.frame_count))
        caption = (
            f"duration: {recording.duration_s:.1f}s   "
            f"frames: {recording.frame_count}   "
            f"avg fps: {avg_fps:.1f}   \n"
            f"pixels: {total_pixels}   "
            f"pixels w/ 0 valid readings: {zero_valid_pixels}   \n"
            f"pixels w/ < {recording.frame_count} valid readings: {incomplete_pixels}"
        )

        ax = self._valid_count_hist_ax
        if valid_counts.size:
            max_count = max(int(valid_counts.max()), 1)
            ax.hist(
                valid_counts,
                bins=np.arange(max_count + 2) - 0.5,
                color=_ACCENT,
                edgecolor="none",
            )
            ax.set_yscale("log")
        ax.set_title("Valid readings per pixel", fontsize=10)
        ax.set_xlabel(f"valid readings (count)\n{caption}")
        ax.set_ylabel("number of pixels")

        self._overall_canvas.draw_idle()
