#!/usr/bin/env python3
"""Verifies the persistent green ROI outline: appears after a drag, survives
subsequent set_image() calls (as the live feed would produce), a too-small
drag is ignored (outline unchanged), and a second valid drag replaces it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nion_app.qt_bootstrap import ensure_compatible_qt_platform

ensure_compatible_qt_platform()

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from nion_app.viewer.ir_roi_view import IrRoiView


def _mouse_event(event_type, pos: QPoint) -> QMouseEvent:
    return QMouseEvent(
        event_type, QPointF(pos), QPointF(pos), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier
    )


def _drag(view, start: QPoint, end: QPoint) -> None:
    view.mousePressEvent(_mouse_event(QEvent.MouseButtonPress, start))
    view.mouseMoveEvent(_mouse_event(QEvent.MouseMove, end))
    view.mouseReleaseEvent(_mouse_event(QEvent.MouseButtonRelease, end))


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    view = IrRoiView()
    view.show()

    frame = (np.random.default_rng(0).integers(0, 50, size=(960, 1280))).astype(np.uint8)
    view.set_image(frame)
    app.processEvents()

    assert not view._roi_outline.isVisible(), "outline should be hidden before any selection"

    # 1) A valid drag shows the outline at the right geometry.
    _drag(view, QPoint(50, 50), QPoint(250, 200))
    app.processEvents()
    assert view._roi_outline.isVisible(), "outline should show after a valid drag"
    first_geometry = view._roi_outline.geometry()
    assert first_geometry.width() > 0 and first_geometry.height() > 0
    print("PASS: outline appears after a valid drag, geometry =", first_geometry)

    # 2) Simulate the live feed continuing to push frames - outline must persist.
    for _ in range(5):
        view.set_image(frame)
        app.processEvents()
    assert view._roi_outline.isVisible(), "outline must survive live set_image() updates"
    assert view._roi_outline.geometry() == first_geometry
    print("PASS: outline survives repeated set_image() calls (live frame updates)")

    # 3) A too-small drag (accidental click) must NOT change or hide the outline.
    _drag(view, QPoint(10, 10), QPoint(11, 10))
    app.processEvents()
    assert view._roi_outline.isVisible()
    assert view._roi_outline.geometry() == first_geometry
    print("PASS: accidental tiny drag leaves the existing outline untouched")

    # 4) A second valid drag replaces the outline at the new location.
    _drag(view, QPoint(300, 300), QPoint(450, 400))
    app.processEvents()
    second_geometry = view._roi_outline.geometry()
    assert second_geometry != first_geometry
    print("PASS: a new valid drag moves the outline to the new region:", second_geometry)

    print("ALL ROI OUTLINE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
