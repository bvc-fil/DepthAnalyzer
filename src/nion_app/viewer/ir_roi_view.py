"""Displays the live IR/intensity image and lets the user drag out a
rectangular region of interest on it, per the noise-measurement spec."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QRubberBand, QVBoxLayout, QWidget

from nion_app.camera.noise_recording import ROI

_DISPLAY_WIDTH = 640
_ROI_OUTLINE_COLOR = QColor(0, 255, 0)  # bright green
_ROI_OUTLINE_WIDTH = 3


class _RoiOutline(QWidget):
    """A persistent bright-green outline marking the currently selected ROI.

    Kept as a real (transparent) child widget rather than baked into the
    displayed pixmap, so it survives every live set_image() frame update
    without needing to be redrawn each time, and disappears only when
    replaced by a new drag selection.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.hide()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        pen = QPen(_ROI_OUTLINE_COLOR, _ROI_OUTLINE_WIDTH)
        painter.setPen(pen)
        half = _ROI_OUTLINE_WIDTH // 2 + 1
        painter.drawRect(self.rect().adjusted(half, half, -half, -half))


class IrRoiView(QWidget):
    roi_selected = Signal(ROI)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image_size: tuple[int, int] | None = None  # (width, height) of the source image
        self._scale = 1.0
        self._roi: ROI | None = None

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self._rubber_band = QRubberBand(QRubberBand.Rectangle, self._label)
        self._roi_outline = _RoiOutline(self._label)
        self._drag_origin = QPoint()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    def set_image(self, intensity: np.ndarray) -> None:
        height, width = intensity.shape
        self._image_size = (width, height)
        self._scale = _DISPLAY_WIDTH / width

        # Same rationale as the point cloud viewer: raw IR intensity is
        # low-signal, so stretch contrast for visibility.
        low, high = np.percentile(intensity, [2, 98])
        span = max(high - low, 1.0)
        stretched = np.clip((intensity.astype(np.float32) - low) / span * 255, 0, 255)
        stretched = np.ascontiguousarray(stretched.astype(np.uint8))

        image = QImage(
            stretched.data, width, height, stretched.strides[0], QImage.Format_Grayscale8
        )
        display_height = round(height * self._scale)
        pixmap = QPixmap.fromImage(image).scaled(
            _DISPLAY_WIDTH, display_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._label.setPixmap(pixmap)
        self._label.resize(pixmap.size())
        self.setFixedSize(pixmap.size())
        self._roi_outline.raise_()

    def current_roi(self) -> ROI | None:
        return self._roi

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._image_size is None:
            return
        self._drag_origin = event.position().toPoint()
        self._rubber_band.setGeometry(QRect(self._drag_origin, QSize()))
        self._rubber_band.show()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._rubber_band.isVisible():
            rect = QRect(self._drag_origin, event.position().toPoint()).normalized()
            self._rubber_band.setGeometry(rect)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._rubber_band.isVisible() or self._image_size is None:
            return
        rect = self._rubber_band.geometry()
        self._rubber_band.hide()

        img_w, img_h = self._image_size
        x0 = max(0, min(img_w - 1, round(rect.left() / self._scale)))
        y0 = max(0, min(img_h - 1, round(rect.top() / self._scale)))
        x1 = max(0, min(img_w, round(rect.right() / self._scale)))
        y1 = max(0, min(img_h, round(rect.bottom() / self._scale)))

        if x1 - x0 < 2 or y1 - y0 < 2:
            return  # ignore accidental clicks/tiny drags - keep any existing ROI outline as-is

        self._roi = ROI(x=x0, y=y0, width=x1 - x0, height=y1 - y0)
        self._roi_outline.setGeometry(rect)
        self._roi_outline.show()
        self._roi_outline.raise_()
        self.roi_selected.emit(self._roi)
