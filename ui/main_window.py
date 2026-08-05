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
from ui.mask_render import colorize_mask, overlay_image_with_mask
from ui.panels import InputPanel, ModelPanel, ViewPanel
from ui.style import DARK_STYLESHEET, apply_pg_theme, CLASS_LABELS, SEGMENTATION_COLORS


class MRIViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRI Tumor Segmentation - Viewer")
        self.resize(1100, 750)

        self.raw_volumes = {}
        self.volumes = {}
        self.spacing = None
        self.paths = {}
        self.segmentation = None
        self.worker = None
        self._current_plane = None

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
        self.view_panel.changed.connect(self.update_slice)

        top_controls = QHBoxLayout()
        top_controls.setSpacing(10)
        top_controls.addWidget(self.input_panel, alignment=Qt.AlignTop)
        top_controls.addWidget(self.model_panel, alignment=Qt.AlignTop)
        top_controls.addWidget(self.view_panel, alignment=Qt.AlignTop)
        top_controls.addStretch()

        self.legend_widget = self._build_legend()
        self.legend_widget.setVisible(False)

        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.histogram.hide()

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self.update_slice)

        self.slice_label = QLabel("No slice")
        self.slice_label.setAlignment(Qt.AlignCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # indefinite progress

        main_layout.addLayout(top_controls)
        main_layout.addWidget(self.legend_widget)
        main_layout.addWidget(self.image_view)
        main_layout.addWidget(self.slice_slider)
        main_layout.addWidget(self.slice_label)
        main_layout.addWidget(self.progress_bar)
        central.setLayout(main_layout)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

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
        self._current_plane = None  # force slider range to resync for the new case
        self.slice_slider.setEnabled(True)

        self.status.showMessage(f"Loaded case from {folder}")
        self._refresh_run_enabled()
        self.update_slice()

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
        self.update_slice()

    def on_inference_failed(self, message):
        self.progress_bar.setVisible(False)
        self.status.showMessage(message)
        self._refresh_run_enabled()

    def update_slice(self):
        if not self.volumes:
            return

        modality = self.view_panel.current_modality()
        plane = self.view_panel.current_plane()
        axis = PLANE_AXES[plane]

        ref = self.volumes[modality]
        axis_len = ref.shape[axis]
        self._sync_slider_range(plane, axis_len)
        self._sync_view(axis, ref)

        idx = self.slice_slider.value()
        view = self.view_panel.current_view()
        has_mask = self.segmentation is not None

        base = np.take(ref, idx, axis=axis)

        if view == "Mask" and has_mask:
            display = colorize_mask(np.take(self.segmentation, idx, axis=axis))
        elif view == "Image + Mask" and has_mask:
            display = overlay_image_with_mask(base, np.take(self.segmentation, idx, axis=axis))
        else:
            display = base

        self.legend_widget.setVisible(view in ("Mask", "Image + Mask") and has_mask)

        # Volumes are RAS+ canonical: transpose to put the slice's rows/cols in image-plane order, flip vertically so superior/anterior is up, then rotate 90 degrees clockwise to match the desired on-screen layout.
        if display.ndim == 3:
            oriented = np.ascontiguousarray(np.rot90(np.flipud(display.transpose(1, 0, 2)), k=1))
            self.image_view.setImage(oriented, autoRange=False, autoLevels=False, levels=(0, 1))
        else:
            oriented = np.ascontiguousarray(np.rot90(np.flipud(display.T), k=1))
            self.image_view.setImage(oriented, autoRange=False, autoLevels=False)

        self.slice_label.setText(f"Slice {idx + 1} / {axis_len}  ·  {plane}")

    def _sync_view(self, axis, ref):
        # After the transpose/flip/rotate in update_slice, the displayed image's
        # x axis is the lower-numbered of the two remaining volume axes and its
        # y axis is the higher-numbered one, so the pixel aspect ratio must be
        # corrected by their physical voxel-spacing ratio (spacing_y / spacing_x)
        # or non-cubic voxels make the slice look stretched/squished.
        ratio = 1
        if self.spacing:
            axis_x, axis_y = (a for a in (0, 1, 2) if a != axis)
            ratio = self.spacing[axis_y] / self.spacing[axis_x]

        view = self.image_view.getView()
        view.setAspectLocked(True, ratio)
        # Keep the same on-screen frame as Axial (ref's sagittal/coronal extents)
        # for every plane, instead of auto-fitting/zooming to each plane's own
        # (smaller) slice size, which made switching planes jump in apparent zoom.
        view.setRange(xRange=(0, ref.shape[0]), yRange=(0, ref.shape[1]), padding=0)

    def _sync_slider_range(self, plane, axis_len):
        max_idx = axis_len - 1
        if plane != self._current_plane:
            self._current_plane = plane
            self.slice_slider.blockSignals(True)
            self.slice_slider.setMaximum(max_idx)
            self.slice_slider.setValue(max_idx // 2)
            self.slice_slider.blockSignals(False)
        elif self.slice_slider.maximum() != max_idx:
            self.slice_slider.blockSignals(True)
            self.slice_slider.setMaximum(max_idx)
            self.slice_slider.blockSignals(False)
