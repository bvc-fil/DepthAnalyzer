"""Layers interactive rectangular prisms on top of the point cloud view:
click to select (left button, freed up from camera control), an in-scene
vtkBoxWidget for dragging position/size/rotation while selected, and
Delete/Backspace to remove the selection. Numeric entry panels drive the
same state through set_prism_transform() so both input paths stay in sync.
"""
from __future__ import annotations

import logging

import numpy as np
import pyvista as pv
import vtk
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent

from nion_app.scene.prism import Prism
from nion_app.viewer.point_cloud_view import PointCloudView

logger = logging.getLogger(__name__)

_DEFAULT_COLOR = "orange"
_SELECTED_COLOR = "yellow"


def _to_vtk_matrix3x3(matrix: np.ndarray) -> "vtk.vtkMatrix4x4":
    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            vtk_matrix.SetElement(i, j, matrix[i, j])
    return vtk_matrix


class PrismSceneView(PointCloudView):
    prism_selected = Signal(object)  # Prism | None
    prism_changed = Signal(object)  # Prism

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._prisms: dict[str, Prism] = {}
        self._actors: dict[str, object] = {}
        self._selected_id: str | None = None
        self._box_widget: vtk.vtkBoxWidget | None = None

        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.005)

        self.interactor_style.on_left_click = self._handle_left_click
        self.setFocusPolicy(Qt.StrongFocus)

    # -- prism lifecycle -----------------------------------------------------
    def add_prism(self, prism: Prism) -> None:
        self._prisms[prism.id] = prism
        self._render_prism(prism)
        self.select_prism(prism.id)
        logger.info(
            "Added prism %s center=%s size_mm=%s", prism.id, prism.center_mm.round(1).tolist(),
            prism.size_mm.round(1).tolist(),
        )

    def remove_prism(self, prism_id: str) -> None:
        if prism_id not in self._prisms:
            return
        if self._selected_id == prism_id:
            self.select_prism(None)
        actor = self._actors.pop(prism_id)
        self.plotter.remove_actor(actor)
        del self._prisms[prism_id]
        self.plotter.render()
        logger.info("Removed prism %s", prism_id)

    def delete_selected(self) -> None:
        if self._selected_id is not None:
            self.remove_prism(self._selected_id)

    def selected_prism(self) -> Prism | None:
        return self._prisms.get(self._selected_id)

    def set_prism_transform(
        self, prism_id: str, center_mm=None, size_mm=None, rotation=None
    ) -> None:
        """Applies an edit coming from the numeric entry panel."""
        prism = self._prisms[prism_id]
        if center_mm is not None:
            prism.center_mm = np.asarray(center_mm, dtype=np.float64)
        if size_mm is not None:
            prism.size_mm = np.asarray(size_mm, dtype=np.float64)
        if rotation is not None:
            prism.rotation = np.asarray(rotation, dtype=np.float64)

        self._render_prism(prism)
        if prism_id == self._selected_id:
            self._teardown_box_widget()
            self._setup_box_widget(prism)
        self.plotter.render()
        self.prism_changed.emit(prism)

    # -- rendering -----------------------------------------------------
    def _render_prism(self, prism: Prism) -> None:
        existing = self._actors.pop(prism.id, None)
        if existing is not None:
            self.plotter.remove_actor(existing)

        box = pv.Box(bounds=prism.local_bounds())
        color = _SELECTED_COLOR if prism.id == self._selected_id else _DEFAULT_COLOR
        actor = self.plotter.add_mesh(
            box, color=color, style="wireframe", line_width=3, name=f"prism-{prism.id}"
        )
        actor.user_matrix = prism.transform_matrix()
        self._actors[prism.id] = actor

    # -- selection -----------------------------------------------------
    def select_prism(self, prism_id: str | None) -> None:
        if prism_id == self._selected_id:
            return
        previous_id = self._selected_id
        self._selected_id = prism_id
        self._teardown_box_widget()

        if previous_id is not None and previous_id in self._prisms:
            self._render_prism(self._prisms[previous_id])
        if prism_id is not None:
            self._render_prism(self._prisms[prism_id])
            self._setup_box_widget(self._prisms[prism_id])

        self.plotter.render()
        self.prism_selected.emit(self._prisms.get(prism_id))

    def _handle_left_click(self, x: int, y: int) -> None:
        self._picker.Pick(x, y, 0, self.plotter.renderer)
        actor = self._picker.GetActor()
        hit_id = next((pid for pid, candidate in self._actors.items() if candidate is actor), None)
        self.select_prism(hit_id)

    # -- box widget -----------------------------------------------------
    def _setup_box_widget(self, prism: Prism) -> None:
        widget = vtk.vtkBoxWidget()
        widget.SetInteractor(self.plotter.iren.interactor)
        widget.SetPlaceFactor(1.0)
        widget.PlaceWidget(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)

        transform = vtk.vtkTransform()
        transform.Translate(*prism.center_mm)
        transform.Concatenate(_to_vtk_matrix3x3(prism.rotation))
        transform.Scale(*prism.size_mm)
        widget.SetTransform(transform)

        widget.AddObserver("InteractionEvent", self._make_box_widget_callback(prism.id))
        widget.On()
        self._box_widget = widget

    def _make_box_widget_callback(self, prism_id: str):
        def callback(widget_obj, _event) -> None:
            polydata = vtk.vtkPolyData()
            widget_obj.GetPolyData(polydata)
            points = np.array([polydata.GetPoints().GetPoint(i) for i in range(8)])
            existing = self._prisms[prism_id]
            updated = Prism.from_box_widget_points(points, id=prism_id, name=existing.name)
            self._prisms[prism_id] = updated
            self._render_prism(updated)
            self.plotter.render()
            self.prism_changed.emit(updated)

        return callback

    def _teardown_box_widget(self) -> None:
        if self._box_widget is not None:
            self._box_widget.Off()
            self._box_widget = None

    def shutdown(self) -> None:
        self._teardown_box_widget()
        super().shutdown()

    # -- keyboard deletion -----------------------------------------------------
    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt override
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
        else:
            super().keyPressEvent(event)
