from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QFormLayout, QGridLayout,
    QPushButton, QComboBox, QCheckBox, QLabel, QFileDialog
)

from core.constants import (
    MODEL_ARCHITECTURES, MODALITIES, PLANES, OVERLAY_MODES,
    OVERLAY_TUMOR, OVERLAY_SYNTHSEG, OVERLAY_BOTH, SYNTHSEG_DEFAULT_MODALITY,
    RADIOMICS_PRESETS, RADIOMICS_CUSTOM_PRESET,
)
from ui.style import TEXT_MUTED


class InputPanel(QGroupBox):
    load_requested = Signal()
    save_requested = Signal()
    save_synthseg_requested = Signal()
    save_radiomics_requested = Signal()

    def __init__(self):
        super().__init__("Input")

        # Two by two rather than a single row of four: in one row this panel
        # alone claimed ~590px of minimum width, a third of a 1920px screen,
        # and it is the widest thing in the control strip.
        layout = QGridLayout()
        self.load_btn = QPushButton("Load BraTS Folder")
        self.load_btn.clicked.connect(self.load_requested)

        self.save_btn = QPushButton("Save Segmentation")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_requested)

        self.save_synthseg_btn = QPushButton("Save SynthSeg")
        self.save_synthseg_btn.setEnabled(False)
        self.save_synthseg_btn.clicked.connect(self.save_synthseg_requested)

        self.save_radiomics_btn = QPushButton("Save Features")
        self.save_radiomics_btn.setEnabled(False)
        self.save_radiomics_btn.clicked.connect(self.save_radiomics_requested)

        layout.addWidget(self.load_btn, 0, 0)
        layout.addWidget(self.save_btn, 0, 1)
        layout.addWidget(self.save_synthseg_btn, 1, 0)
        layout.addWidget(self.save_radiomics_btn, 1, 1)
        self.setLayout(layout)

    def set_save_enabled(self, enabled):
        self.save_btn.setEnabled(enabled)

    def set_save_synthseg_enabled(self, enabled):
        self.save_synthseg_btn.setEnabled(enabled)

    def set_save_radiomics_enabled(self, enabled):
        self.save_radiomics_btn.setEnabled(enabled)


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

        # A disabled button with no visible explanation sends people digging
        # through logs; the reason belongs on screen, with the full text (which
        # can be a long path) kept in the tooltip.
        self.reason_label = QLabel()
        self.reason_label.setWordWrap(True)
        self.reason_label.setMaximumWidth(240)
        self.reason_label.setMinimumWidth(0)
        self.reason_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.reason_label.setVisible(False)

        layout.addRow("Modality:", self.modality_box)
        layout.addRow("", options_row)
        layout.addRow("", self.run_btn)
        layout.addRow("", self.reason_label)
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

    def set_run_enabled(self, enabled, reason=None, summary=None):
        """Enable the button, or disable it and say why — on screen and on hover.

        The reason previously went to the status bar at start-up only, where
        the next message overwrote it, leaving a greyed-out button with no
        recoverable explanation of what was missing.
        """
        self.run_btn.setEnabled(enabled)
        self.run_btn.setToolTip("" if enabled else (reason or ""))
        self.reason_label.setToolTip(reason or "")
        self.reason_label.setText("" if enabled else (summary or reason or ""))
        self.reason_label.setVisible(not enabled and bool(summary or reason))


class RadiomicsPanel(QGroupBox):
    run_requested = Signal()

    # Kept in the panel rather than alongside the YAML files: this is the
    # one-line "what will this cost me" a user needs at the moment of choosing,
    # not a description of the file's contents.
    _PRESET_TOOLTIPS = {
        "Fast": "Shape and first-order statistics only — 32 features per region.",
        "Standard": "All seven feature classes on the unfiltered image — "
                    "107 features per region.",
        "Extended": "Standard, plus Laplacian-of-Gaussian and wavelet filters — "
                    "1130 features per region, and about four times the wait.",
    }

    def __init__(self):
        super().__init__("Radiomics")

        layout = QFormLayout()
        layout.setLabelAlignment(Qt.AlignRight)

        # All four ticked: a radiomics table is normally reported across every
        # sequence, and unticking is cheaper than hunting for the ones you want.
        self.modality_checks = {}
        modality_row = QHBoxLayout()
        for modality in MODALITIES:
            check = QCheckBox(modality)
            check.setChecked(True)
            check.toggled.connect(self._on_modality_toggled)
            self.modality_checks[modality] = check
            modality_row.addWidget(check)
        modality_row.addStretch()

        self.preset_box = QComboBox()
        self.preset_box.addItems(RADIOMICS_PRESETS)
        self.preset_box.addItem(RADIOMICS_CUSTOM_PRESET)
        self.preset_box.setCurrentText("Standard")
        self.preset_box.currentTextChanged.connect(self._on_preset_changed)
        self._update_preset_tooltip(self.preset_box.currentText())

        # Set only by the Custom entry. Remembered so the file dialog can
        # reopen where it last was, and so cancelling out of it does not lose
        # the file already in use.
        self._custom_params = None

        self.run_btn = QPushButton("Extract Features")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_requested)

        # Same reasoning as SynthSegPanel: a greyed-out button with its
        # explanation only on hover sends people digging through logs.
        self.reason_label = QLabel()
        self.reason_label.setWordWrap(True)
        self.reason_label.setMaximumWidth(240)
        self.reason_label.setMinimumWidth(0)
        self.reason_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.reason_label.setVisible(False)

        layout.addRow("Modalities:", modality_row)
        layout.addRow("Features:", self.preset_box)
        layout.addRow("", self.run_btn)
        layout.addRow("", self.reason_label)
        self.setLayout(layout)

    def _on_modality_toggled(self, _checked):
        # Unticking the last box would run an extraction over nothing, so the
        # sole remaining one is locked until another is ticked.
        selected = self.modalities()
        for modality, check in self.modality_checks.items():
            check.setEnabled(len(selected) > 1 or modality not in selected)

    def _on_preset_changed(self, preset):
        self._update_preset_tooltip(preset)
        if preset != RADIOMICS_CUSTOM_PRESET:
            return

        # Asked every time Custom is selected, not just the first: the combo is
        # the only route to the file, so remembering it silently would leave no
        # way to switch to a different one.
        path, _ = QFileDialog.getOpenFileName(
            self, "Select PyRadiomics parameter file",
            self._custom_params or "", "YAML (*.yaml *.yml)",
        )
        if path:
            self._custom_params = path
        elif not self._custom_params:
            # Cancelled with nothing to fall back on: don't leave a Custom
            # selection with no file behind it.
            self.preset_box.setCurrentText("Standard")
        self._update_preset_tooltip(self.preset_box.currentText())

    def _update_preset_tooltip(self, preset):
        if preset == RADIOMICS_CUSTOM_PRESET:
            self.preset_box.setToolTip(
                self._custom_params
                or "Choose your own PyRadiomics parameter YAML."
            )
        else:
            self.preset_box.setToolTip(self._PRESET_TOOLTIPS.get(preset, ""))

    def modalities(self):
        return [m for m, c in self.modality_checks.items() if c.isChecked()]

    def current_preset(self):
        return self.preset_box.currentText()

    def params_path(self):
        """The parameter file to use, or None to take the preset's bundled one."""
        if self.current_preset() == RADIOMICS_CUSTOM_PRESET:
            return self._custom_params
        return None

    def set_run_enabled(self, enabled, reason=None, summary=None):
        self.run_btn.setEnabled(enabled)
        self.run_btn.setToolTip("" if enabled else (reason or ""))
        self.reason_label.setToolTip(reason or "")
        self.reason_label.setText("" if enabled else (summary or reason or ""))
        self.reason_label.setVisible(not enabled and bool(summary or reason))
