#!/usr/bin/env python3
"""Exercises the full prism lifecycle against a synthetic point cloud (no
camera needed): add, box-widget drag (simulated), numeric-panel edit,
click-to-select, and delete. Asserts the data model stays correct at each
step and saves a screenshot for visual confirmation.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import vtk
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from nion_app.logging_setup import configure_logging
from nion_app.scene.prism import Prism, euler_xyz_deg_to_matrix
from nion_app.app.prism_panel import PrismPanel
from nion_app.viewer.prism_scene_view import PrismSceneView

logger = logging.getLogger(__name__)


def main() -> int:
    configure_logging()
    app = QApplication.instance() or QApplication(sys.argv)

    view = PrismSceneView()
    view.resize(1024, 768)
    panel = PrismPanel(view)

    # Synthetic floor-ish point cloud so the scene isn't empty.
    grid_x, grid_y = np.meshgrid(np.linspace(-500, 500, 60), np.linspace(-500, 500, 60))
    points = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, 800.0)])
    view.set_points(points.astype(np.float32))
    view.show()
    app.processEvents()

    # 1) Add a prism via the panel and check it landed with the default size.
    panel._on_add_clicked()
    prism_id = view.selected_prism().id
    assert view.selected_prism().size_mm.tolist() == [100.0, 100.0, 100.0]
    assert prism_id in view._actors
    logger.info("PASS: add_prism created and selected %s", prism_id)

    # 2) Simulate a user drag by feeding the box widget's callback a moved +
    #    rotated + resized synthetic corner set, exactly like vtkBoxWidget would.
    new_center = np.array([120.0, -80.0, 950.0])
    new_size = np.array([200.0, 60.0, 40.0])
    rotation = euler_xyz_deg_to_matrix(0, 0, 25)
    dragged = Prism(center_mm=new_center, size_mm=new_size, rotation=rotation, id=prism_id)
    corners = dragged.corners()

    fake_points = vtk.vtkPoints()
    for corner in corners:
        fake_points.InsertNextPoint(*corner)
    fake_polydata = vtk.vtkPolyData()
    fake_polydata.SetPoints(fake_points)

    class _FakeWidget:
        def GetPolyData(self, pd):
            pd.SetPoints(fake_polydata.GetPoints())

    view._make_box_widget_callback(prism_id)(_FakeWidget(), "InteractionEvent")
    updated = view._prisms[prism_id]
    assert np.allclose(updated.center_mm, new_center, atol=1e-6), updated.center_mm
    assert np.allclose(updated.size_mm, new_size, atol=1e-6), updated.size_mm
    assert np.allclose(updated.rotation, rotation, atol=1e-6), updated.rotation
    logger.info("PASS: box-widget drag decoded center/size/rotation correctly")

    # 3) Confirm the numeric panel picked up the same values (prism_changed signal).
    app.processEvents()
    assert abs(panel.position_boxes[0].value() - 120.0) < 0.5
    assert abs(panel.rotation_boxes[2].value() - 25.0) < 0.5
    logger.info("PASS: numeric panel fields reflect the dragged prism")

    # 4) Edit via the numeric panel (simulating direct value entry) and check
    #    it reaches the data model and the box widget is rebuilt to match.
    panel.size_boxes[0].setValue(300.0)
    app.processEvents()
    assert view._prisms[prism_id].size_mm[0] == 300.0
    logger.info("PASS: numeric panel edit applied to prism")

    # 5) Click-to-select: project the prism's current world-space center to
    #    screen coordinates and simulate a left click there.
    view.select_prism(None)
    assert view.selected_prism() is None
    renderer = view.plotter.renderer
    world = view._prisms[prism_id].center_mm
    renderer.SetWorldPoint(*world, 1.0)
    renderer.WorldToDisplay()
    sx, sy, _ = renderer.GetDisplayPoint()
    view._handle_left_click(int(sx), int(sy))
    assert view.selected_prism() is not None and view.selected_prism().id == prism_id
    logger.info("PASS: click-to-select picked the prism at its projected screen position")

    # 6) Delete via keyboard (Delete key), matching the outline's requirement.
    key_event = QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier)
    view.keyPressEvent(key_event)
    assert prism_id not in view._prisms
    assert prism_id not in view._actors
    logger.info("PASS: Delete key removed the selected prism")

    screenshot_path = Path(__file__).resolve().parent.parent / "logs" / "prism_test.png"
    # Re-add one for the screenshot so there's something to look at.
    panel._on_add_clicked()
    view.plotter.reset_camera()
    view.plotter.screenshot(str(screenshot_path))
    logger.info("Saved screenshot to %s", screenshot_path)

    print("ALL PRISM TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
