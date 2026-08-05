"""User-controlled 3D point cloud viewer widget.

Mouse controls (per spec): wheel zooms, right-button-drag orbits, and
middle-button-drag pans - the latter two are the vtkInteractorStyleTrackballCamera
defaults, so only the left and right button bindings need remapping (left is
freed up for prism selection/dragging in a later phase; right takes over orbit
instead of its default dolly-zoom).
"""
from __future__ import annotations

import logging

import numpy as np
import pyvista as pv
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

logger = logging.getLogger(__name__)


class _OrbitPanZoomStyle(vtkInteractorStyleTrackballCamera):
    """Left button intentionally does not rotate the camera - it's reserved
    for prism selection, dispatched through on_left_click if a consumer
    (e.g. PrismSceneView) has set one."""

    on_left_click = None  # Optional[Callable[[int, int], None]], set by a consumer

    def OnLeftButtonDown(self):  # noqa: N802 - VTK virtual method naming convention
        if self.on_left_click is not None:
            x, y = self.GetInteractor().GetEventPosition()
            self.on_left_click(x, y)

    def OnLeftButtonUp(self):  # noqa: N802
        pass

    def OnRightButtonDown(self):  # noqa: N802
        self.StartRotate()

    def OnRightButtonUp(self):  # noqa: N802
        self.EndRotate()


class PointCloudView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.plotter = QtInteractor(self)
        self.interactor_style = _OrbitPanZoomStyle()
        self.plotter.iren.interactor.SetInteractorStyle(self.interactor_style)
        self.plotter.set_background("black")

        self._point_actor = None
        self._default_camera_position = None

        reset_button = QPushButton("Reset View")
        reset_button.clicked.connect(self.reset_view)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plotter.interactor)
        layout.addWidget(reset_button)
        self.setLayout(layout)

    def set_points(self, points_xyz_mm: np.ndarray, scalars: np.ndarray | None = None) -> None:
        """Replace the displayed point cloud. scalars, if given, colors each
        point (e.g. by intensity) and must have one value per point."""
        if self._point_actor is not None:
            self.plotter.remove_actor(self._point_actor)
            self._point_actor = None

        if points_xyz_mm.size == 0:
            self.plotter.render()
            return

        cloud = pv.PolyData(points_xyz_mm)
        # Raw sensor scalars (e.g. active-IR intensity) are typically low-signal
        # and clustered near zero, so a straight min/max color range renders as
        # near-black; stretch to the 2nd-98th percentile instead for a usable image.
        clim = None
        if scalars is not None:
            clim = tuple(np.percentile(scalars, [2, 98]))
        self._point_actor = self.plotter.add_points(
            cloud,
            scalars=scalars,
            cmap="gray" if scalars is not None else None,
            clim=clim,
            point_size=2,
            render_points_as_spheres=False,
            show_scalar_bar=False,
        )

        if self._default_camera_position is None:
            self.plotter.reset_camera()
            self._default_camera_position = self.plotter.camera_position

        self.plotter.render()

    def reset_view(self) -> None:
        if self._default_camera_position is not None:
            self.plotter.camera_position = self._default_camera_position
        else:
            self.plotter.reset_camera()
        self.plotter.render()
        logger.info("Viewer camera reset")

    def shutdown(self) -> None:
        """Releases the VTK render window and OpenGL context, and stops the
        QtInteractor's internal render_timer. pyvistaqt does not do this on
        its own when the widget is merely closed/hidden - without an
        explicit close() the native render window and its timer keep running
        after the main window closes, leaving the process alive in the
        background."""
        self.plotter.close()
