"""Live depth-frame preview for quick before-capture measurements: renders
the current depth image and, while the mouse hovers over it, reports the
sensor pixel coordinates and the depth reading directly under the cursor."""
from __future__ import annotations

import matplotlib as mpl
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from nion_app.camera.backend import Frame

_DISPLAY_WIDTH = 640
_DEPTH_CMAP = mpl.colormaps["rainbow"]  # same gradient used for mean depth in the results view
_INVALID_COLOR = np.array([40, 40, 40], dtype=np.uint8)  # dark gray for depth_mm == 0 (invalid)
_NO_READING_TEXT = "Hover over the image to read a pixel's depth."


def _colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 RGB. Invalid pixels (depth_mm == 0) are excluded from
    the color scale and rendered as a flat neutral color, so a few dropouts
    don't compress the whole gradient into a sliver of real depth values."""
    valid = depth_mm > 0
    rgb = np.empty((*depth_mm.shape, 3), dtype=np.uint8)
    rgb[...] = _INVALID_COLOR
    if valid.any():
        low, high = np.percentile(depth_mm[valid], [2, 98])
        span = max(high - low, 1.0)
        normalized = np.clip((depth_mm - low) / span, 0.0, 1.0)
        colored = (_DEPTH_CMAP(normalized)[..., :3] * 255).astype(np.uint8)
        rgb[valid] = colored[valid]
    return rgb


class DepthView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._depth_mm: np.ndarray | None = None
        self._image_size: tuple[int, int] | None = None  # (width, height) of the source image
        self._scale = 1.0
        self._hover_pos = None  # last QPoint the mouse was seen at, or None while not hovering

        self._image_label = QLabel(self)
        self._image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._image_label.setMouseTracking(True)

        self._readout_label = QLabel(_NO_READING_TEXT)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._image_label)
        layout.addWidget(self._readout_label)

        self.setMouseTracking(True)

    def on_frame(self, frame: Frame) -> None:
        self.set_depth(frame.depth_mm)

    def set_depth(self, depth_mm: np.ndarray) -> None:
        height, width = depth_mm.shape
        self._depth_mm = depth_mm
        self._image_size = (width, height)
        self._scale = _DISPLAY_WIDTH / width

        rgb = np.ascontiguousarray(_colorize_depth(depth_mm))
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888)
        display_height = round(height * self._scale)
        pixmap = QPixmap.fromImage(image).scaled(
            _DISPLAY_WIDTH, display_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._image_label.setPixmap(pixmap)
        self._image_label.setFixedSize(pixmap.size())

        # Re-read the pixel under the cursor against this new frame, so the
        # reading keeps live-updating even while the mouse stays still.
        self._update_readout()

    def _pixel_at(self, widget_pos) -> tuple[int, int] | None:
        if self._depth_mm is None or self._image_size is None:
            return None
        img_w, img_h = self._image_size
        x = int(widget_pos.x() / self._scale)
        y = int(widget_pos.y() / self._scale)
        if not (0 <= x < img_w and 0 <= y < img_h):
            return None
        return x, y

    def _update_readout(self) -> None:
        pixel = self._pixel_at(self._hover_pos) if self._hover_pos is not None else None
        if pixel is None:
            self._readout_label.setText(_NO_READING_TEXT)
            return
        x, y = pixel
        depth = float(self._depth_mm[y, x])
        if depth <= 0:
            self._readout_label.setText(f"({x}, {y}): no valid reading")
        else:
            self._readout_label.setText(f"({x}, {y}): {depth:.1f} mm")

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        self._hover_pos = event.position().toPoint()
        self._update_readout()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._hover_pos = None
        self._update_readout()
        super().leaveEvent(event)
