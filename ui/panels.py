from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QFormLayout,
    QPushButton, QComboBox, QCheckBox
)

from core.constants import (
    MODEL_ARCHITECTURES, MODALITIES, PLANES, OVERLAY_MODES,
    OVERLAY_TUMOR, OVERLAY_SYNTHSEG, OVERLAY_BOTH, SYNTHSEG_DEFAULT_MODALITY,
)


class InputPanel(QGroupBox):
    load_requested = Signal()
    save_requested = Signal()
    save_synthseg_requested = Signal()

    def __init__(self):
        super().__init__("Input")

        layout = QHBoxLayout()
        self.load_btn = QPushButton("Load BraTS Folder")
        self.load_btn.clicked.connect(self.load_requested)

        self.save_btn = QPushButton("Save Segmentation")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_requested)

        self.save_synthseg_btn = QPushButton("Save SynthSeg")
        self.save_synthseg_btn.setEnabled(False)
        self.save_synthseg_btn.clicked.connect(self.save_synthseg_requested)

        layout.addWidget(self.load_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.save_synthseg_btn)
        layout.addStretch()
        self.setLayout(layout)

    def set_save_enabled(self, enabled):
        self.save_btn.setEnabled(enabled)

    def set_save_synthseg_enabled(self, enabled):
        self.save_synthseg_btn.setEnabled(enabled)


class ModelPanel(QGroupBox):
    run_requested = Signal()
    architecture_changed = Signal(str)

    def __init__(self):
        super().__init__("Model")

        layout = QFormLayout()
        layout.setLabelAlignment(Qt.AlignRight)

        self.model_box = QComboBox()
        self.model_box.addItems(MODEL_ARCHITECTURES)
        self.model_box.currentTextChanged.connect(self.architecture_changed)

        self.run_btn = QPushButton("Run Inference")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_requested)

        layout.addRow("Architecture:", self.model_box)
        layout.addRow("", self.run_btn)
        self.setLayout(layout)

    def current_model(self):
        return self.model_box.currentText()

    def set_run_enabled(self, enabled):
        self.run_btn.setEnabled(enabled)


class ViewPanel(QGroupBox):
    changed = Signal()
    plane_changed = Signal(str)

    def __init__(self):
        super().__init__("View")

        layout = QFormLayout()
        layout.setLabelAlignment(Qt.AlignRight)

        self.plane_box = QComboBox()
        self.plane_box.addItems(PLANES)
        self.plane_box.currentTextChanged.connect(self.plane_changed)

        self.modality_box = QComboBox()
        self.modality_box.addItems(MODALITIES)
        self.modality_box.currentTextChanged.connect(self.changed)

        self.overlay_box = QComboBox()
        self.overlay_box.addItems(OVERLAY_MODES)
        self.overlay_box.currentTextChanged.connect(self.changed)

        layout.addRow("Plane:", self.plane_box)
        layout.addRow("Modality:", self.modality_box)
        layout.addRow("Overlay:", self.overlay_box)
        self.setLayout(layout)

    def current_modality(self):
        return self.modality_box.currentText()

    def current_plane(self):
        return self.plane_box.currentText()

    def current_overlay(self):
        return self.overlay_box.currentText()

    def set_overlay(self, mode):
        """Select an overlay mode without re-triggering a redraw."""
        self.overlay_box.blockSignals(True)
        self.overlay_box.setCurrentText(mode)
        self.overlay_box.blockSignals(False)

    def update_overlay_availability(self, has_tumor, has_synthseg):
        """Grey out overlay modes whose mask has not been produced yet."""
        available = {
            OVERLAY_TUMOR: has_tumor,
            OVERLAY_SYNTHSEG: has_synthseg,
            OVERLAY_BOTH: has_tumor and has_synthseg,
        }
        model = self.overlay_box.model()
        for row in range(self.overlay_box.count()):
            enabled = available.get(self.overlay_box.itemText(row), True)
            model.item(row).setEnabled(enabled)

        # Never leave a disabled mode selected — fall back to whichever single
        # overlay does exist, else to Tumor so the combo always has a value.
        if not available.get(self.current_overlay(), True):
            fallback = OVERLAY_TUMOR if has_tumor else (
                OVERLAY_SYNTHSEG if has_synthseg else OVERLAY_TUMOR
            )
            self.set_overlay(fallback)


class SynthSegPanel(QGroupBox):
    run_requested = Signal()

    def __init__(self):
        super().__init__("SynthSeg")

        layout = QFormLayout()
        layout.setLabelAlignment(Qt.AlignRight)

        self.modality_box = QComboBox()
        self.modality_box.addItems(MODALITIES)
        self.modality_box.setCurrentText(SYNTHSEG_DEFAULT_MODALITY)

        self.fast_check = QCheckBox("Fast")
        self.fast_check.setToolTip(
            "Skip topology postprocessing — roughly twice as fast, marginally "
            "less accurate."
        )
        self.robust_check = QCheckBox("Robust")
        self.robust_check.setToolTip(
            "Use the robust model, which handles low-quality clinical scans "
            "better but is slower. Implies Fast."
        )
        self.parc_check = QCheckBox("Parcellation")
        self.parc_check.setToolTip(
            "Also parcellate the cortex into 68 Desikan-Killiany regions, "
            "replacing the single cortex label."
        )

        # Robust forces fast=True inside SynthSeg, so show that rather than
        # letting the checkbox claim otherwise.
        self.robust_check.toggled.connect(self._on_robust_toggled)

        options_row = QHBoxLayout()
        options_row.addWidget(self.fast_check)
        options_row.addWidget(self.robust_check)
        options_row.addWidget(self.parc_check)
        options_row.addStretch()

        self.run_btn = QPushButton("Run SynthSeg")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_requested)

        layout.addRow("Modality:", self.modality_box)
        layout.addRow("", options_row)
        layout.addRow("", self.run_btn)
        self.setLayout(layout)

    def _on_robust_toggled(self, checked):
        if checked:
            self.fast_check.setChecked(True)
        self.fast_check.setEnabled(not checked)

    def current_modality(self):
        return self.modality_box.currentText()

    def options(self):
        return {
            "fast": self.fast_check.isChecked(),
            "robust": self.robust_check.isChecked(),
            "parc": self.parc_check.isChecked(),
        }

    def set_run_enabled(self, enabled, reason=None):
        """Enable the button, or disable it and explain why on hover.

        The reason previously went to the status bar at start-up only, where
        the next message overwrote it — leaving a greyed-out button with no
        recoverable explanation of what was missing.
        """
        self.run_btn.setEnabled(enabled)
        self.run_btn.setToolTip("" if enabled else (reason or ""))
