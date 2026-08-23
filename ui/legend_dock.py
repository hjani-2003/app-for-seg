"""Colour legend for the tumour and SynthSeg overlays, as a side dock.

It used to be a short scrolling strip above the images, which was the worst of
both worlds: it took height away from the slices and still hid most of its own
contents. A case carries 35 legend rows normally and about 100 with cortical
parcellation, so the legend needs a tall narrow column, not a wide short one —
and the window has horizontal space to spare where it has no vertical space.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QFrame,
)

from ui.style import TEXT, TEXT_MUTED, BORDER, SEGMENTATION_COLORS, CLASS_LABELS
from ui.synthseg_lut import TABLE, label_color_hex, label_name

SWATCH = 12
ROW_SPACING = 3

# Structure names run to "right inferior lateral ventricle", so the dock is
# sized from the text it actually holds rather than a guessed width — a
# truncated legend is no better than a hidden one. Capped so one long name
# cannot take the window over.
_WIDTH_PADDING = SWATCH + 7 + 20 + 16   # swatch, spacing, margins, scrollbar
_MAX_WIDTH = 340


class _Row(QWidget):
    """One swatch-and-name row, dimmed when absent from the current slice."""

    def __init__(self, color_hex, text):
        super().__init__()
        self.color_hex = color_hex

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)

        self.swatch = QLabel()
        self.swatch.setFixedSize(SWATCH, SWATCH)
        self.swatch.setStyleSheet(
            f"background-color: {color_hex}; border-radius: 3px;"
        )

        self.name = QLabel(text)
        layout.addWidget(self.swatch)
        layout.addWidget(self.name)
        layout.addStretch()
        self.setLayout(layout)
        self.set_present(False)

    def set_present(self, present):
        """Emphasise structures visible in the slice on screen.

        Only colour and weight change, never size, so rows do not reflow as the
        slider moves.
        """
        self.name.setStyleSheet(
            f"color: {TEXT}; font-weight: 600;" if present
            else f"color: {TEXT_MUTED}; font-weight: 400;"
        )
        # The swatch keeps its colour either way — it is the key to reading the
        # image, and dimming it would misrepresent the colour on screen.


class LegendDock(QDockWidget):
    def __init__(self):
        super().__init__("Legend")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )

        self._body = QVBoxLayout()
        self._body.setContentsMargins(10, 8, 10, 8)
        self._body.setSpacing(ROW_SPACING)
        self._body.addStretch()

        inner = QWidget()
        inner.setLayout(self._body)

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setWidget(scroll)

        self._tumor_rows = {}
        self._synthseg_rows = {}
        self._headers = {}
        self._content = None      # the label sets the rows were built from
        self._present = set()
        self._fit_width()

    # ---------- building ----------

    def _header(self, text):
        label = QLabel(text)
        label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-weight: 600; padding-top: 6px;"
            f" border-bottom: 1px solid {BORDER};"
        )
        return label

    def clear(self):
        while self._body.count():
            item = self._body.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Unparent before deleting: deleteLater only frees the widget
                # on the next event-loop pass, and until then it keeps painting
                # over the rows that replace it. Re-running SynthSeg on the
                # same case made old and new labels overlap.
                widget.setParent(None)
                widget.deleteLater()
        self._body.addStretch()
        self._tumor_rows, self._synthseg_rows, self._headers = {}, {}, {}
        self._content = None
        self._present = set()

    def set_content(self, tumor_labels, synthseg_labels):
        """Rebuild the rows for the labels this case contains.

        Rebuilding is the expensive part, so it happens only when the masks
        change — not on every slice.
        """
        content = (tuple(sorted(tumor_labels)), tuple(sorted(synthseg_labels)))
        if content == self._content:
            return
        self.clear()
        self._content = content

        self._body.takeAt(self._body.count() - 1)  # drop the trailing stretch

        if tumor_labels:
            self._headers["tumor"] = self._header("Tumour")
            self._body.addWidget(self._headers["tumor"])
            for value in sorted(tumor_labels):
                row = _Row(SEGMENTATION_COLORS[value],
                           CLASS_LABELS.get(value, f"class {value}"))
                self._tumor_rows[value] = row
                self._body.addWidget(row)

        if synthseg_labels:
            self._headers["synthseg"] = self._header("Anatomy")
            self._body.addWidget(self._headers["synthseg"])
            for value in sorted(synthseg_labels):
                row = _Row(label_color_hex(value), label_name(value))
                self._synthseg_rows[value] = row
                self._body.addWidget(row)

        self._body.addStretch()

    def _fit_width(self):
        """Fix the dock width once, from the longest label that can ever appear.

        Sizing to the current contents instead would make the dock jump —
        narrow with only the three tumour classes, then wide the moment a
        SynthSeg run lands.
        """
        names = [name for name, _ in TABLE.values()] + list(CLASS_LABELS.values())
        font = self.font()
        font.setBold(True)          # emphasised rows are the widest drawn
        metrics = QFontMetrics(font)
        widest = max(metrics.horizontalAdvance(name) for name in names)
        self.widget().setMinimumWidth(min(widest + _WIDTH_PADDING, _MAX_WIDTH))

    # ---------- per-slice state ----------

    def set_sections_visible(self, show_tumor, show_synthseg):
        for key, rows, visible in (
            ("tumor", self._tumor_rows, show_tumor),
            ("synthseg", self._synthseg_rows, show_synthseg),
        ):
            if key in self._headers:
                self._headers[key].setVisible(visible)
            for row in rows.values():
                row.setVisible(visible)

    def set_present(self, tumor_present, synthseg_present):
        """Emphasise the labels in the slice on screen.

        Only rows whose state actually changed are restyled — this runs on
        every slider move, and there can be a hundred of them.
        """
        present = {("t", v) for v in tumor_present} | {
            ("s", v) for v in synthseg_present
        }
        for key in present ^ self._present:
            kind, value = key
            rows = self._tumor_rows if kind == "t" else self._synthseg_rows
            row = rows.get(value)
            if row is not None:
                row.set_present(key in present)
        self._present = present
