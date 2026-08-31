"""The radiomic feature table, in its own window.

Features go down and regions go across, not the other way round. The natural
shape of the data is one row per (modality, region) — that is how it is saved,
and how any downstream model wants it — but that table is 110 columns wide on
the Standard preset and 1500 on Extended, which no window can show. Transposed,
the widest it ever gets is four modalities by five regions, and the long axis
is the one a scrollbar handles well.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
)

from core.constants import RADIOMICS_REGION_DESCRIPTIONS
from ui.style import TEXT_MUTED
from ui.tool_window import ToolWindow

ALL_CLASSES = "All classes"

# PyRadiomics names a feature "<imageType>_<class>_<name>", e.g.
# "original_firstorder_Mean" or "wavelet-LHL_glcm_Idm".


def feature_class(name):
    """The feature class a PyRadiomics feature name belongs to, or "" if unclear."""
    parts = name.split("_", 2)
    return parts[1] if len(parts) >= 2 else ""


def format_value(value):
    """Feature values span many orders of magnitude, so significant figures beat
    a fixed number of decimal places — Elongation is 0.7 and Volume is 40000."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e5 or magnitude < 1e-3:
        return f"{value:.4g}"
    return f"{value:.6g}"


class RadiomicsWindow(ToolWindow):
    # Wider than the default: the table is one column per (modality, region),
    # up to twenty of them, and a feature name is long.
    DEFAULT_SIZE = (1100, 700)

    def __init__(self):
        super().__init__("Radiomic Features")

        self._result = None

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.summary = QLabel("No features extracted")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")

        self.class_box = QComboBox()
        self.class_box.addItem(ALL_CLASSES)
        self.class_box.currentTextChanged.connect(self._apply_filter)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter features…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_filter)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(6)
        filter_row.addWidget(self.class_box)
        filter_row.addWidget(self.search)

        self.table = QTableWidget()
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )

        self.skipped = QLabel()
        self.skipped.setWordWrap(True)
        self.skipped.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.skipped.setVisible(False)

        layout.addWidget(self.summary)
        layout.addLayout(filter_row)
        layout.addWidget(self.table)
        layout.addWidget(self.skipped)
        self.setLayout(layout)

    def clear(self):
        self._result = None
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.summary.setText("No features extracted")
        self.skipped.setVisible(False)
        self.search.clear()
        self._reset_classes([])

    def set_content(self, result):
        """Rebuild the table for a finished run.

        Split from _apply_filter the way LegendDock splits its rebuild from its
        per-slice restyle: 1500 rows are expensive to build and cheap to hide.
        """
        self._result = result
        self.summary.setText(result.info)

        self._reset_classes(result.feature_names)
        self._rebuild()
        self._apply_filter()

        if result.skipped:
            lines = [
                f"{modality} {region}: {reason}"
                for modality, region, reason in result.skipped
            ]
            # Every modality skips an empty region for the same reason, so the
            # list is deduplicated on the reason rather than repeated four times.
            unique = []
            for line in lines:
                if line not in unique:
                    unique.append(line)
            self.skipped.setText("Skipped — " + "; ".join(unique))
            self.skipped.setVisible(True)
        else:
            self.skipped.setVisible(False)

    def _reset_classes(self, feature_names):
        classes = []
        for name in feature_names:
            cls = feature_class(name)
            if cls and cls not in classes:
                classes.append(cls)

        self.class_box.blockSignals(True)
        self.class_box.clear()
        self.class_box.addItem(ALL_CLASSES)
        self.class_box.addItems(classes)
        self.class_box.blockSignals(False)

    def _rebuild(self):
        result = self._result
        columns = [(row["Modality"], row["Region"]) for row in result.rows]

        self.table.clear()
        self.table.setColumnCount(1 + len(columns))
        self.table.setRowCount(len(result.feature_names))
        self.table.setHorizontalHeaderLabels(
            ["Feature"] + [f"{modality}\n{region}" for modality, region in columns]
        )
        for column, (_, region) in enumerate(columns, start=1):
            header = self.table.horizontalHeaderItem(column)
            header.setToolTip(RADIOMICS_REGION_DESCRIPTIONS.get(region, region))

        for row_index, name in enumerate(result.feature_names):
            item = QTableWidgetItem(name)
            item.setToolTip(name)
            self.table.setItem(row_index, 0, item)
            for column, source in enumerate(result.rows, start=1):
                value = source.get(name)
                cell = QTableWidgetItem(
                    "" if value is None else format_value(value)
                )
                cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row_index, column, cell)

    def _apply_filter(self):
        if self._result is None:
            return

        needle = self.search.text().strip().lower()
        wanted = self.class_box.currentText()
        for row_index, name in enumerate(self._result.feature_names):
            visible = (needle in name.lower()) and (
                wanted == ALL_CLASSES or feature_class(name) == wanted
            )
            self.table.setRowHidden(row_index, not visible)
