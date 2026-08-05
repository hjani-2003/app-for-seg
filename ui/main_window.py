import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSlider, QProgressBar, QStatusBar, QLabel
)

from core.constants import PLANE_AXES
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
        self.view_panel.changed.connect(self.update_view)
        self.view_panel.plane_changed.connect(self._on_plane_changed)

        top_controls = QHBoxLayout()
        top_controls.setSpacing(10)
        top_controls.addWidget(self.input_panel, alignment=Qt.AlignTop)
        top_controls.addWidget(self.model_panel, alignment=Qt.AlignTop)
        top_controls.addWidget(self.view_panel, alignment=Qt.AlignTop)
        top_controls.addStretch()

        self.legend_widget = self._build_legend()
        self.legend_widget.setVisible(False)

        self.mri_panel = self._build_panel("Axial")
        self.mask_panel = self._build_panel("Axial + Mask")

        panels_row = QHBoxLayout()
        panels_row.setSpacing(10)
        panels_row.addWidget(self.mri_panel["container"])
        panels_row.addWidget(self.mask_panel["container"])

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self.update_view)

        self.slice_label = QLabel("No slice")
        self.slice_label.setAlignment(Qt.AlignCenter)

        slider_row = QHBoxLayout()
        slider_row.setSpacing(10)
        slider_row.addWidget(self.slice_slider)
        slider_row.addWidget(self.slice_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # indefinite progress

        main_layout.addLayout(top_controls)
        main_layout.addWidget(self.legend_widget)
        main_layout.addLayout(panels_row)
        main_layout.addLayout(slider_row)
        main_layout.addWidget(self.progress_bar)
        central.setLayout(main_layout)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

    def _build_panel(self, title):
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)

        image_view = pg.ImageView()
        image_view.ui.roiBtn.hide()
        image_view.ui.menuBtn.hide()
        image_view.ui.histogram.hide()

        layout.addWidget(title_label)
        layout.addWidget(image_view)
        container.setLayout(layout)

        return {
            "container": container,
            "image_view": image_view,
            "title_label": title_label,
        }

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
            self.slice_slider.setEnabled(False)
            self.status.showMessage(str(exc))
            self._refresh_run_enabled()
            return

        self.volumes = {m: normalize_for_display(v) for m, v in self.raw_volumes.items()}
        self.segmentation = None

        modality = self.view_panel.current_modality()
        axis = PLANE_AXES[self.view_panel.current_plane()]
        ref = self.volumes[modality]
        max_idx = ref.shape[axis] - 1
        self.slice_slider.blockSignals(True)
        self.slice_slider.setEnabled(True)
        self.slice_slider.setMaximum(max_idx)
        self.slice_slider.setValue(max_idx // 2)
        self.slice_slider.blockSignals(False)

        self.status.showMessage(f"Loaded case from {folder}")
        self._refresh_run_enabled()
        self.update_view()

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
        self.update_view()

    def on_inference_failed(self, message):
        self.progress_bar.setVisible(False)
        self.status.showMessage(message)
        self._refresh_run_enabled()

    def _on_plane_changed(self, plane):
        if not self.volumes:
            return

        modality = self.view_panel.current_modality()
        axis = PLANE_AXES[plane]
        ref = self.volumes[modality]
        max_idx = ref.shape[axis] - 1

        self.slice_slider.blockSignals(True)
        self.slice_slider.setMaximum(max_idx)
        self.slice_slider.setValue(max_idx // 2)
        self.slice_slider.blockSignals(False)

        self.update_view()

    def update_view(self):
        if not self.volumes:
            return

        plane = self.view_panel.current_plane()
        axis = PLANE_AXES[plane]
        modality = self.view_panel.current_modality()
        ref = self.volumes[modality]
        axis_len = ref.shape[axis]
        idx = self.slice_slider.value()

        base = np.take(ref, idx, axis=axis)
        self._render(self.mri_panel["image_view"], base, axis)

        has_mask = self.segmentation is not None
        if has_mask:
            overlay = overlay_image_with_mask(base, np.take(self.segmentation, idx, axis=axis))
        else:
            overlay = base
        self._render(self.mask_panel["image_view"], overlay, axis)

        self.mri_panel["title_label"].setText(plane)
        self.mask_panel["title_label"].setText(f"{plane} + Mask")
        self.slice_label.setText(f"Slice {idx + 1} / {axis_len}  ·  {plane}")
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
