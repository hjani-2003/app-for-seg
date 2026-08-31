"""The RANO Measurements table, in its own window.

pyqtgraph equivalent of the napari-based design in the Phase 3 spec (this
app has no napari — see NOTES.md): instead of two Shapes layers, each
lesion's major/minor lines are `pg.LineSegmentROI` items added directly to
the mask panel's existing ViewBox; instead of napari's native shape
selection, removal is table-row-selection + Delete (or pg's own built-in
right-click "Remove" on a line, wired to the same code path).

Every row here is measured, never drawn: the four-click "Add Line Pair"
mode this widget used to carry has been taken out, so the table reports
what `rano_measure` computed from the mask and nothing a hand put there.

Coordinate mapping: rano_measure.geometry/lesion operate in raw voxel
(row, col) space — the same space `np.take(volume, idx, axis=axis)` uses
elsewhere in this app — NOT the flipped/rotated space
`MRIViewer._render()` builds only for on-screen display. Empirically,
_render()'s transpose->flipud->rot90(k=1) chain collapses to a single
180-degree point reflection: display[R-1-r, C-1-c] = base[r, c] (verified
against the actual transform in Phase 3 development, not assumed). And
pyqtgraph's default 'col-major' ImageItem convention maps array index
(axis0, axis1) directly to view-space (x, y) with no additional swap
(also empirically verified). So a voxel point (r, c) in an (R, C) slice
maps to view-space (R-1-r, C-1-c), and that mapping is its own inverse.
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from rano_measure.burden import select_target_lesions
from rano_measure.geometry import DEFAULT_MIN_DIAMETER_MM
from rano_measure.lesion import find_lesions
from rano_measure.regions import LABEL_IDS, REGION_DEFS, build_region_mask
from ui.style import RANO_LINE_COLORS
from ui.tool_window import ToolWindow

AXIAL_AXIS = 2  # matches core.constants.PLANE_AXES["Axial"]

COLUMNS = ["ID", "Region", "Slice", "Diameter 1 (mm)", "Diameter 2 (mm)", "Product (mm^2)", "Measurable", "Target"]


class RanoWindow(ToolWindow):
    # Eight columns of numbers, but rarely many rows: wide and short.
    DEFAULT_SIZE = (940, 520)

    def __init__(self, image_view):
        super().__init__("RANO Measurements")

        self._view_box = image_view.getView()
        self._active_rois = {}  # (lesion_id, "major"/"minor") -> pg.LineSegmentROI

        self.lesions = []
        self._label_volume = None
        self._spacing = None
        self._slice_shape = None
        self._current_slice_index = None
        self._current_plane = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        # Sized to the headings, not to a dock's width: "Diameter 1 (mm)" was
        # truncated to "Diameter 1 (mm" in the column it used to sit in.
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        # The first column is the lesion id, which is what a row is referred to
        # by; a second, different number down the side is only confusing.
        self.table.verticalHeader().setVisible(False)

        self.ce_sum_label = QLabel("CE target sum: -")
        self.nonce_sum_label = QLabel("nonCE target sum: -")

        # The callipers are drawn on axial slices only, so say so in the other
        # planes rather than leaving an empty overlay looking like a fault.
        self.axial_only_hint = QLabel(
            "Callipers are drawn on the Axial plane only — RANO is defined there."
        )
        self.axial_only_hint.setWordWrap(True)
        self.axial_only_hint.setVisible(False)

        sums_row = QHBoxLayout()
        sums_row.setSpacing(24)
        sums_row.addWidget(self.ce_sum_label)
        sums_row.addWidget(self.nonce_sum_label)
        sums_row.addStretch()

        layout.addWidget(self.table)
        layout.addLayout(sums_row)
        layout.addWidget(self.axial_only_hint)
        self.setLayout(layout)

        self.delete_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self.table)
        self.delete_shortcut.activated.connect(self._remove_selected_lesions)

    # ---- Auto-populate on load (Phase 3 §2) ----------------------------

    def populate_from_segmentation(self, label_volume, spacing):
        self.clear()
        if label_volume is None:
            return

        self._label_volume = label_volume
        self._spacing = spacing
        self._slice_shape = (label_volume.shape[0], label_volume.shape[1])

        lesions = []
        for region_name in ("CE", "nonCE"):
            region_mask = build_region_mask(label_volume, LABEL_IDS, REGION_DEFS, region_name)
            lesions.extend(find_lesions(region_mask, spacing, AXIAL_AXIS, region_name))

        # find_lesions ids restart at 1 per region call -> renumber for a
        # single globally-unique id space across the combined CE+nonCE list.
        for new_id, lesion in enumerate(lesions, start=1):
            lesion.id = new_id
        self.lesions = lesions

        self._refresh_table()
        self._render_rois_for_current_slice()

    def clear(self):
        self._remove_all_rois()
        self.lesions = []
        self._label_volume = None
        self._spacing = None
        self._slice_shape = None
        self._refresh_table()

    # ---- Slice/plane sync (called from MRIViewer.update_view) ----------

    def show_slice(self, slice_index, plane):
        self._current_slice_index = slice_index
        self._current_plane = plane
        self._render_rois_for_current_slice()
        self.axial_only_hint.setVisible(plane != "Axial")

    # ---- Coordinate mapping (voxel <-> display view space) -------------

    def _voxel_to_view(self, r, c):
        R, C = self._slice_shape
        return (R - 1 - r, C - 1 - c)

    def _view_to_voxel(self, x, y):
        R, C = self._slice_shape
        return (int(round(R - 1 - x)), int(round(C - 1 - y)))

    def _length_mm(self, p1_xy, p2_xy):
        row_mm, col_mm = self._spacing[0], self._spacing[1]
        dx = (p2_xy[0] - p1_xy[0]) * row_mm
        dy = (p2_xy[1] - p1_xy[1]) * col_mm
        return float(np.hypot(dx, dy))

    # ---- ROI rendering ---------------------------------------------------

    def _render_rois_for_current_slice(self):
        self._remove_all_rois()
        if self._label_volume is None or self._current_plane != "Axial":
            return
        for lesion in self.lesions:
            if lesion.slice_index == self._current_slice_index:
                self._add_line_roi(lesion, "major")
                self._add_line_roi(lesion, "minor")

    def _add_line_roi(self, lesion, kind):
        line = lesion.major_line if kind == "major" else lesion.minor_line
        if line is None:
            return
        p1, p2 = line
        v1 = self._voxel_to_view(*p1)
        v2 = self._voxel_to_view(*p2)

        roi = pg.LineSegmentROI(positions=[v1, v2], pen=pg.mkPen(RANO_LINE_COLORS[lesion.region_type], width=2))
        roi.removable = True
        roi.lesion_id = lesion.id
        roi.line_kind = kind
        roi.sigRegionChanged.connect(self._on_roi_changed)
        roi.sigRemoveRequested.connect(self._on_roi_remove_requested)

        self._view_box.addItem(roi)
        self._active_rois[(lesion.id, kind)] = roi

    def _remove_all_rois(self):
        for roi in list(self._active_rois.values()):
            self._view_box.removeItem(roi)
        self._active_rois = {}

    # ---- Live editing (Phase 3 §3) --------------------------------------

    def _on_roi_changed(self, roi):
        lesion = self._find_lesion(roi.lesion_id)
        if lesion is None:
            return

        p1, p2 = roi.listPoints()
        p1_xy, p2_xy = (p1.x(), p1.y()), (p2.x(), p2.y())
        length_mm = self._length_mm(p1_xy, p2_xy)
        voxel_line = (self._view_to_voxel(*p1_xy), self._view_to_voxel(*p2_xy))

        if roi.line_kind == "major":
            lesion.major_line = voxel_line
            lesion.major_mm = length_mm
        else:
            lesion.minor_line = voxel_line
            lesion.minor_mm = length_mm

        lesion.product_mm2 = lesion.major_mm * lesion.minor_mm
        lesion.measurable = lesion.major_mm >= DEFAULT_MIN_DIAMETER_MM and lesion.minor_mm >= DEFAULT_MIN_DIAMETER_MM

        self._update_table_row(lesion)
        self._refresh_sums()

    # ---- Removal (Phase 3 §4; adding by hand was taken out) -------------

    def _on_roi_remove_requested(self, roi):
        self._remove_lesion(roi.lesion_id)

    def _remove_selected_lesions(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        lesion_ids = [int(self.table.item(row, 0).text()) for row in rows]
        for lesion_id in lesion_ids:
            self._remove_lesion(lesion_id)

    def _remove_lesion(self, lesion_id):
        self.lesions = [lesion for lesion in self.lesions if lesion.id != lesion_id]
        for kind in ("major", "minor"):
            roi = self._active_rois.pop((lesion_id, kind), None)
            if roi is not None:
                self._view_box.removeItem(roi)
        self._refresh_table()

    # ---- Table + sums -----------------------------------------------------

    def _find_lesion(self, lesion_id):
        for lesion in self.lesions:
            if lesion.id == lesion_id:
                return lesion
        return None

    def _refresh_table(self):
        self.table.setRowCount(0)
        for lesion in self.lesions:
            self._append_table_row(lesion)
        self._refresh_sums()

    def _append_table_row(self, lesion):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._set_row_cells(row, lesion)

    def _set_row_cells(self, row, lesion):
        values = [
            str(lesion.id),
            lesion.region_type,
            str(lesion.slice_index),
            f"{lesion.major_mm:.1f}",
            f"{lesion.minor_mm:.1f}",
            f"{lesion.product_mm2:.1f}",
            "yes" if lesion.measurable else "no",
            "",  # target column, filled in by _refresh_sums
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, col, item)

    def _find_table_row(self, lesion_id):
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == str(lesion_id):
                return row
        return None

    def _update_table_row(self, lesion):
        row = self._find_table_row(lesion.id)
        if row is not None:
            self._set_row_cells(row, lesion)
        self._refresh_sums()

    def _refresh_sums(self):
        summary = select_target_lesions(self.lesions)
        self.ce_sum_label.setText(
            f"CE target sum: {summary.ce_product_sum_mm2:.1f} mm^2 ({len(summary.ce_target_lesions)} lesions)"
        )
        self.nonce_sum_label.setText(
            f"nonCE target sum: {summary.nonce_product_sum_mm2:.1f} mm^2 ({len(summary.nonce_target_lesions)} lesions)"
        )
        target_ids = {lesion.id for lesion in summary.target_lesions}
        for row in range(self.table.rowCount()):
            lesion_id = int(self.table.item(row, 0).text())
            self.table.item(row, 7).setText("*" if lesion_id in target_ids else "")
