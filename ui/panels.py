from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QFormLayout,
    QPushButton, QComboBox
)

from core.constants import MODEL_ARCHITECTURES, MODALITIES, VIEW_MODES, PLANES


class InputPanel(QGroupBox):
    load_requested = Signal()

    def __init__(self):
        super().__init__("Input")

        layout = QHBoxLayout()
        self.load_btn = QPushButton("Load BraTS Folder")
        self.load_btn.clicked.connect(self.load_requested)

        layout.addWidget(self.load_btn)
        layout.addStretch()
        self.setLayout(layout)


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

    def __init__(self):
        super().__init__("View")

        layout = QFormLayout()
        layout.setLabelAlignment(Qt.AlignRight)

        self.modality_box = QComboBox()
        self.modality_box.addItems(MODALITIES)
        self.modality_box.currentTextChanged.connect(self.changed)

        self.plane_box = QComboBox()
        self.plane_box.addItems(PLANES)
        self.plane_box.currentTextChanged.connect(self.changed)

        self.view_box = QComboBox()
        self.view_box.addItems(VIEW_MODES)
        self.view_box.currentTextChanged.connect(self.changed)

        layout.addRow("Modality:", self.modality_box)
        layout.addRow("Plane:", self.plane_box)
        layout.addRow("Display:", self.view_box)
        self.setLayout(layout)

    def current_modality(self):
        return self.modality_box.currentText()

    def current_plane(self):
        return self.plane_box.currentText()

    def current_view(self):
        return self.view_box.currentText()
