"""Numeric entry panel for the selected prism's position, size, and rotation,
plus Add/Delete buttons - the second of the two required editing paths
(the other being the in-scene box widget handles)."""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nion_app.scene.prism import Prism, euler_xyz_deg_to_matrix, matrix_to_euler_xyz_deg
from nion_app.viewer.prism_scene_view import PrismSceneView

_DEFAULT_SIZE_MM = (100.0, 100.0, 100.0)
_SPIN_RANGE_MM = 100_000.0
_SPIN_RANGE_DEG = 360.0


def _make_spinbox(range_limit: float, decimals: int = 1) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(-range_limit, range_limit)
    box.setDecimals(decimals)
    box.setSingleStep(1.0)
    return box


class PrismPanel(QWidget):
    def __init__(self, scene_view: PrismSceneView, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene_view = scene_view
        self._updating = False  # guards against feedback loops with the scene view

        self.position_boxes = [_make_spinbox(_SPIN_RANGE_MM) for _ in range(3)]
        self.size_boxes = [_make_spinbox(_SPIN_RANGE_MM) for _ in range(3)]
        self.rotation_boxes = [_make_spinbox(_SPIN_RANGE_DEG) for _ in range(3)]

        for box in (*self.position_boxes, *self.size_boxes, *self.rotation_boxes):
            box.valueChanged.connect(self._on_fields_edited)

        form = QFormLayout()
        form.addRow("Position X/Y/Z (mm)", self._row(self.position_boxes))
        form.addRow("Size X/Y/Z (mm)", self._row(self.size_boxes))
        form.addRow("Rotation X/Y/Z (deg)", self._row(self.rotation_boxes))

        group = QGroupBox("Selected Prism")
        group.setLayout(form)

        add_button = QPushButton("Add Prism")
        add_button.clicked.connect(self._on_add_clicked)
        delete_button = QPushButton("Delete Selected")
        delete_button.clicked.connect(scene_view.delete_selected)

        buttons = QHBoxLayout()
        buttons.addWidget(add_button)
        buttons.addWidget(delete_button)

        layout = QVBoxLayout(self)
        layout.addWidget(group)
        layout.addLayout(buttons)

        scene_view.prism_selected.connect(self._on_prism_selected)
        scene_view.prism_changed.connect(self._on_prism_changed)
        self._set_enabled(False)

    @staticmethod
    def _row(boxes: list[QDoubleSpinBox]) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for box in boxes:
            layout.addWidget(box)
        return row

    def _set_enabled(self, enabled: bool) -> None:
        for box in (*self.position_boxes, *self.size_boxes, *self.rotation_boxes):
            box.setEnabled(enabled)

    def _on_add_clicked(self) -> None:
        prism = Prism(center_mm=np.zeros(3), size_mm=np.array(_DEFAULT_SIZE_MM))
        self._scene_view.add_prism(prism)

    def _on_prism_selected(self, prism: Prism | None) -> None:
        self._set_enabled(prism is not None)
        if prism is not None:
            self._populate_fields(prism)

    def _on_prism_changed(self, prism: Prism) -> None:
        if prism.id == self._current_prism_id():
            self._populate_fields(prism)

    def _current_prism_id(self) -> str | None:
        selected = self._scene_view.selected_prism()
        return selected.id if selected is not None else None

    def _populate_fields(self, prism: Prism) -> None:
        self._updating = True
        try:
            for box, value in zip(self.position_boxes, prism.center_mm):
                box.setValue(float(value))
            for box, value in zip(self.size_boxes, prism.size_mm):
                box.setValue(float(value))
            for box, value in zip(self.rotation_boxes, matrix_to_euler_xyz_deg(prism.rotation)):
                box.setValue(float(value))
        finally:
            self._updating = False

    def _on_fields_edited(self, _value: float) -> None:
        if self._updating:
            return
        prism_id = self._current_prism_id()
        if prism_id is None:
            return

        center = np.array([box.value() for box in self.position_boxes])
        size = np.array([box.value() for box in self.size_boxes])
        rotation = euler_xyz_deg_to_matrix(*(box.value() for box in self.rotation_boxes))
        self._scene_view.set_prism_transform(
            prism_id, center_mm=center, size_mm=size, rotation=rotation
        )
