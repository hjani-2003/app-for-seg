import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFileDialog, QSlider, QProgressBar, QStatusBar, QLabel
)

from core.constants import PLANES, PLANE_AXES
from core.data_loader import load_brats_folder, normalize_for_display
from core.inference import InferenceWorker
from ui.mask_render import overlay_image_with_mask
from ui.panels import InputPanel, ModelPanel, ViewPanel
from ui.style import DARK_STYLESHEET, apply_pg_theme, CLASS_LABELS, SEGMENTATION_COLORS


class MRIViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRI Tumor Segmentation - Viewer")
        self.resize(1100, 850)

        self.raw_volumes = {}
        self.volumes = {}
        self.spacing = None
        self.paths = {}
        self.segmentation = None
        self.worker = None
        self.plane_panels = {}

        apply_pg_theme()
        self._build_ui()
        self.setStyleSheet(DARK_STYLESHEET)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        self.input_panel = InputPanel()
        self.input_panel.load_requested.connect(self.load_folder)

        self.model_panel = ModelPanel()
        self.model_panel.run_requested.connect(self.run_inference)
        self.model_panel.architecture_changed.connect(self._refresh_run_enabled)

        self.view_panel = ViewPanel()
        self.view_panel.changed.connect(self.update_all)

        top_controls = QHBoxLayout()
        top_controls.setSpacing(10)
        top_controls.addWidget(self.input_panel, alignment=Qt.AlignTop)
        top_controls.addWidget(self.model_panel, alignment=Qt.AlignTop)
        top_controls.addWidget(self.view_panel, alignment=Qt.AlignTop)
        top_controls.addStretch()

        self.legend_widget = self._build_legend()
        self.legend_widget.setVisible(False)

        grid = QGridLayout()
        grid.setSpacing(10)
        for plane in PLANES:
            self.plane_panels[plane] = self._build_plane_panel(plane)
        grid.addWidget(self.plane_panels["Axial"]["container"], 0, 0)
        grid.addWidget(self.plane_panels["Coronal"]["container"], 0, 1)
        grid.addWidget(self.plane_panels["Sagittal"]["container"], 1, 0)

        self.label_view, label_container = self._build_label_panel()
        grid.addWidget(label_container, 1, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # indefinite progress

        main_layout.addLayout(top_controls)
        main_layout.addWidget(self.legend_widget)
        main_layout.addLayout(grid)
        main_layout.addWidget(self.progress_bar)
        central.setLayout(main_layout)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

    def _build_plane_panel(self, plane):
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QLabel(plane)
        title.setAlignment(Qt.AlignCenter)

        image_view = pg.ImageView()
        image_view.ui.roiBtn.hide()
        image_view.ui.menuBtn.hide()
        image_view.ui.histogram.hide()

        slider = QSlider(Qt.Horizontal)
        slider.setEnabled(False)
        slider.valueChanged.connect(lambda _v, p=plane: self.update_plane(p))

        slice_label = QLabel("No slice")
        slice_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(title)
        layout.addWidget(image_view)
        layout.addWidget(slider)
        layout.addWidget(slice_label)
        container.setLayout(layout)

        return {
            "container": container,
            "image_view": image_view,
            "slider": slider,
            "slice_label": slice_label,
        }

    def _build_label_panel(self):
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QLabel("Axial + Mask")
        title.setAlignment(Qt.AlignCenter)

        image_view = pg.ImageView()
        image_view.ui.roiBtn.hide()
        image_view.ui.menuBtn.hide()
        image_view.ui.histogram.hide()

        layout.addWidget(title)
        layout.addWidget(image_view)
        container.setLayout(layout)

        return image_view, container

    def _build_legend(self):
        legend = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 0, 0, 4)
        layout.setSpacing(18)

        for class_id, label in CLASS_LABELS.items():
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"background-color: {SEGMENTATION_COLORS[class_id]}; border-radius: 3px;"
            )

            entry = QHBoxLayout()
            entry.setSpacing(6)
            entry.addWidget(swatch)
            entry.addWidget(QLabel(label))
            layout.addLayout(entry)

        layout.addStretch()
        legend.setLayout(layout)
        return legend

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select BraTS Case Folder"
        )
        if not folder:
            return

        try:
            self.raw_volumes, self.spacing, self.paths = load_brats_folder(folder)
        except ValueError as exc:
            self.raw_volumes = {}
            self.volumes = {}
            self.spacing = None
            self.paths = {}
            self.segmentation = None
            for pv in self.plane_panels.values():
                pv["slider"].setEnabled(False)
            self.status.showMessage(str(exc))
            self._refresh_run_enabled()
            return

        self.volumes = {m: normalize_for_display(v) for m, v in self.raw_volumes.items()}
        self.segmentation = None

        modality = self.view_panel.current_modality()
        ref = self.volumes[modality]
        for plane, pv in self.plane_panels.items():
            max_idx = ref.shape[PLANE_AXES[plane]] - 1
            pv["slider"].blockSignals(True)
            pv["slider"].setEnabled(True)
            pv["slider"].setMaximum(max_idx)
            pv["slider"].setValue(max_idx // 2)
            pv["slider"].blockSignals(False)

        self.status.showMessage(f"Loaded case from {folder}")
        self._refresh_run_enabled()
        self.update_all()

    def _refresh_run_enabled(self):
        has_case = len(self.raw_volumes) == 4
        self.model_panel.set_run_enabled(has_case)

    def run_inference(self):
        if len(self.raw_volumes) != 4:
            self.status.showMessage("Please load a valid BraTS case")
            return

        model = self.model_panel.current_model()

        self.model_panel.set_run_enabled(False)
        self.progress_bar.setVisible(True)
        self.status.showMessage(f"Running inference using {model}...")

        self.worker = InferenceWorker(self.raw_volumes, self.paths, model)
        self.worker.finished.connect(self.on_inference_done)
        self.worker.failed.connect(self.on_inference_failed)
        self.worker.start()

    def on_inference_done(self, mask, info):
        self.segmentation = mask
        self.progress_bar.setVisible(False)
        self.status.showMessage(f"Inference completed ({info})")
        self._refresh_run_enabled()
        self.update_all()

    def on_inference_failed(self, message):
        self.progress_bar.setVisible(False)
        self.status.showMessage(message)
        self._refresh_run_enabled()

    def update_all(self):
        for plane in PLANES:
            self.update_plane(plane)

    def update_plane(self, plane):
        if not self.volumes:
            return

        modality = self.view_panel.current_modality()
        axis = PLANE_AXES[plane]
        ref = self.volumes[modality]
        axis_len = ref.shape[axis]

        pv = self.plane_panels[plane]
        idx = pv["slider"].value()

        display = np.take(ref, idx, axis=axis)
        self._render(pv["image_view"], display, axis)
        pv["slice_label"].setText(f"Slice {idx + 1} / {axis_len}  ·  {plane}")

        if plane == "Axial":
            self._update_label(idx)

    def _update_label(self, idx):
        modality = self.view_panel.current_modality()
        axis = PLANE_AXES["Axial"]
        ref = self.volumes[modality]
        has_mask = self.segmentation is not None

        base = np.take(ref, idx, axis=axis)
        if has_mask:
            display = overlay_image_with_mask(base, np.take(self.segmentation, idx, axis=axis))
        else:
            display = base

        self._render(self.label_view, display, axis)
        self.legend_widget.setVisible(has_mask)

    def _render(self, image_view, display, axis):
        # Volumes are RAS+ canonical: transpose to put the slice's rows/cols in image-plane order, flip vertically so superior/anterior is up, then rotate 90 degrees clockwise to match the desired on-screen layout.
        if display.ndim == 3:
            oriented = np.ascontiguousarray(np.rot90(np.flipud(display.transpose(1, 0, 2)), k=1))
        else:
            oriented = np.ascontiguousarray(np.rot90(np.flipud(display.T), k=1))

        # The pixel aspect ratio must be corrected by the physical voxel-spacing
        # ratio (spacing_y / spacing_x) of the two in-plane axes, or non-cubic
        # voxels make the slice look stretched/squished.
        ratio = 1
        if self.spacing:
            axis_x, axis_y = (a for a in (0, 1, 2) if a != axis)
            ratio = self.spacing[axis_y] / self.spacing[axis_x]
        image_view.getView().setAspectLocked(True, ratio)

        if oriented.ndim == 3:
            image_view.setImage(oriented, autoLevels=False, levels=(0, 1))
        else:
            image_view.setImage(oriented, autoLevels=False)
