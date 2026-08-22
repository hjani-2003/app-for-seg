import os

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFileDialog, QSlider, QProgressBar, QStatusBar, QLabel, QScrollArea
)

from core.backends import synthseg_backend
from core.constants import (
    MODALITIES, PLANE_AXES, MODALITY_TO_FILE_KEY,
    OVERLAY_TUMOR, OVERLAY_SYNTHSEG, OVERLAY_BOTH,
)
from core.data_loader import (
    load_brats_folder, normalize_for_display, save_segmentation, save_label_map,
)
from core.inference import InferenceWorker
from core.synthseg_inference import SynthSegWorker
from ui.mask_render import overlay_image_with_masks
from ui.panels import InputPanel, ModelPanel, ViewPanel, SynthSegPanel
from ui.style import DARK_STYLESHEET, apply_pg_theme, CLASS_LABELS, SEGMENTATION_COLORS
from ui.synthseg_lut import label_color_hex, label_name


# A parcellated SynthSeg case lists ~100 structures, so the legend scrolls
# rather than growing the window. Wide enough for the longest structure names
# at the app's default width.
_LEGEND_COLUMNS = 6
_LEGEND_ROW_HEIGHT = 22
_LEGEND_MAX_HEIGHT = 72


class MRIViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRI Tumor Segmentation - Viewer")
        self.resize(1100, 850)

        self.raw_volumes = {}
        self.volumes = {}
        self.spacing = None
        self.paths = {}
        self.case_folder = None
        self.segmentation = None
        self.worker = None
        self.synthseg_mask = None
        self.synthseg_result = None
        self.synthseg_worker = None
        self.synthseg_running = False

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
        self.input_panel.save_requested.connect(self.save_folder)
        self.input_panel.save_synthseg_requested.connect(self.save_synthseg)

        self.model_panel = ModelPanel()
        self.model_panel.run_requested.connect(self.run_inference)
        self.model_panel.architecture_changed.connect(self._refresh_run_enabled)

        self.synthseg_panel = SynthSegPanel()
        self.synthseg_panel.run_requested.connect(self.run_synthseg)

        self.view_panel = ViewPanel()
        self.view_panel.changed.connect(self.update_view)
        self.view_panel.plane_changed.connect(self._on_plane_changed)
        self.view_panel.update_overlay_availability(False, False)

        top_controls = QHBoxLayout()
        top_controls.setSpacing(10)
        top_controls.addWidget(self.input_panel, alignment=Qt.AlignTop)
        top_controls.addWidget(self.model_panel, alignment=Qt.AlignTop)
        top_controls.addWidget(self.synthseg_panel, alignment=Qt.AlignTop)
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

        # Surface a broken SynthSeg install now rather than after a user has
        # loaded a case and waited on a run that could never start.
        self.synthseg_unavailable = synthseg_backend.check_available()
        self.status.showMessage(
            self.synthseg_unavailable or f"Ready · {synthseg_backend.runtime_summary()}"
        )

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
        """A scrollable wrapping grid of swatches.

        A single row was enough for three tumour classes, but a parcellated
        SynthSeg volume carries around a hundred labels, so entries wrap into a
        grid inside a height-capped scroll area instead of stretching the
        window.
        """
        self.legend_grid = QGridLayout()
        self.legend_grid.setContentsMargins(4, 2, 4, 2)
        self.legend_grid.setHorizontalSpacing(14)
        self.legend_grid.setVerticalSpacing(2)

        self.legend_inner = QWidget()
        self.legend_inner.setLayout(self.legend_grid)

        legend = QScrollArea()
        legend.setWidget(self.legend_inner)
        legend.setWidgetResizable(True)
        legend.setMaximumHeight(_LEGEND_MAX_HEIGHT)
        legend.setFrameShape(QScrollArea.NoFrame)
        legend.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return legend

    def _legend_entry(self, color_hex, text):
        swatch = QLabel()
        swatch.setFixedSize(12, 12)
        swatch.setStyleSheet(f"background-color: {color_hex}; border-radius: 3px;")

        entry = QWidget()
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(swatch)
        row.addWidget(QLabel(text))
        entry.setLayout(row)
        return entry

    def _update_legend(self, show_tumor, show_synthseg):
        while self.legend_grid.count():
            item = self.legend_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        entries = []
        if show_tumor:
            entries += [
                (SEGMENTATION_COLORS[class_id], label)
                for class_id, label in CLASS_LABELS.items()
            ]
        if show_synthseg and self.synthseg_mask is not None:
            # Only the structures actually present, so the legend describes
            # this case rather than the whole label space.
            present = np.unique(self.synthseg_mask)
            entries += [
                (label_color_hex(value), label_name(value))
                for value in present
                if value > 0
            ]

        for index, (color_hex, text) in enumerate(entries):
            self.legend_grid.addWidget(
                self._legend_entry(color_hex, text),
                index // _LEGEND_COLUMNS,
                index % _LEGEND_COLUMNS,
            )

        # widgetResizable would otherwise squash the rows to the viewport
        # height and overlap their text. The height is computed from the row
        # count rather than the layout's sizeHint, which is not yet valid for
        # widgets added moments ago.
        rows = -(-len(entries) // _LEGEND_COLUMNS)
        self.legend_inner.setMinimumHeight(rows * _LEGEND_ROW_HEIGHT)
        self.legend_widget.setVisible(bool(entries))

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
            self.case_folder = None
            self.segmentation = None
            self._reset_synthseg()
            self.input_panel.set_save_enabled(False)
            self.slice_slider.setEnabled(False)
            self.status.showMessage(str(exc))
            self._refresh_run_enabled()
            return

        self.volumes = {m: normalize_for_display(v) for m, v in self.raw_volumes.items()}
        self.case_folder = folder
        self.segmentation = None
        self._reset_synthseg()
        self.input_panel.set_save_enabled(False)

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

    def save_folder(self):
        if self.segmentation is None:
            self.status.showMessage("Run inference before saving")
            return

        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if not directory:
            return

        case_name = os.path.basename(os.path.normpath(self.case_folder))
        out_path = os.path.join(directory, f"{case_name}_seg.nii.gz")

        try:
            save_segmentation(self.segmentation, self.paths[MODALITIES[0]], out_path)
        except Exception as exc:
            self.status.showMessage(f"Save failed: {exc}")
            return

        self.status.showMessage(f"Saved segmentation to {out_path}")

    def save_synthseg(self):
        if self.synthseg_mask is None:
            self.status.showMessage("Run SynthSeg before saving")
            return

        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not directory:
            return

        result = self.synthseg_result
        case_name = os.path.basename(os.path.normpath(self.case_folder))
        # The modality is part of the name because SynthSeg can be run on any
        # loaded modality, and parcellated runs use a different label space —
        # neither should silently overwrite the other.
        stem = f"{case_name}_synthseg_{MODALITY_TO_FILE_KEY[result.modality]}"
        if result.parc:
            stem += "_parc"

        outputs = [
            (f"{stem}.nii.gz", None),
            (f"{stem}_volumes.csv", result.volumes_csv),
            (f"{stem}_qc.csv", result.qc_csv),
        ]

        written = []
        try:
            for filename, text in outputs:
                out_path = os.path.join(directory, filename)
                if text is None:
                    save_label_map(
                        self.synthseg_mask, self.paths[result.modality], out_path
                    )
                elif not text:
                    # SynthSeg writes nothing when a CSV stage is skipped;
                    # don't leave an empty file behind pretending otherwise.
                    continue
                else:
                    with open(out_path, "w", newline="") as handle:
                        handle.write(text)
                written.append(filename)
        except Exception as exc:
            self.status.showMessage(f"Save failed: {exc}")
            return

        self.status.showMessage(
            f"Saved {', '.join(written)} to {directory}"
        )

    def _reset_synthseg(self):
        self.synthseg_mask = None
        self.synthseg_result = None
        self.input_panel.set_save_synthseg_enabled(False)
        self.view_panel.update_overlay_availability(
            self.segmentation is not None, False
        )

    def _refresh_run_enabled(self):
        has_case = len(self.raw_volumes) == 4
        self.model_panel.set_run_enabled(has_case)

        # Re-checked rather than cached from start-up, so installing the env or
        # the weights takes effect without restarting the app.
        self.synthseg_unavailable = synthseg_backend.check_available()

        if self.synthseg_unavailable is not None:
            reason = self.synthseg_unavailable
        elif not has_case:
            reason = "Load a BraTS case first"
        elif self.synthseg_running:
            # Loading a case calls this, so without the in-flight check a user
            # could start a second run on top of one already going.
            reason = "A SynthSeg run is already in progress"
        else:
            reason = None

        self.synthseg_panel.set_run_enabled(reason is None, reason)

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
        self.input_panel.set_save_enabled(True)
        self.progress_bar.setVisible(False)
        self.view_panel.update_overlay_availability(
            True, self.synthseg_mask is not None
        )
        self.status.showMessage(f"Inference completed ({info})")
        self._refresh_run_enabled()
        self.update_view()

    def on_inference_failed(self, message):
        self.progress_bar.setVisible(False)
        self.status.showMessage(message)
        self._refresh_run_enabled()

    def run_synthseg(self):
        if len(self.raw_volumes) != 4:
            self.status.showMessage("Please load a valid BraTS case")
            return

        modality = self.synthseg_panel.current_modality()
        options = self.synthseg_panel.options()

        self.synthseg_running = True
        self.synthseg_panel.set_run_enabled(False)
        self.progress_bar.setVisible(True)
        self.status.showMessage(
            f"Running SynthSeg on {modality}... "
            f"({synthseg_backend.runtime_summary()})"
        )

        # SynthSeg reads and writes in the file's own space, so it gets the
        # original path rather than the reoriented array the viewer holds.
        self.synthseg_worker = SynthSegWorker(
            self.paths[modality],
            self.volumes[modality].shape,
            modality,
            options,
            self.case_folder,
        )
        self.synthseg_worker.finished.connect(self.on_synthseg_done)
        self.synthseg_worker.failed.connect(self.on_synthseg_failed)
        self.synthseg_worker.progress.connect(self.status.showMessage)
        self.synthseg_worker.start()

    def on_synthseg_done(self, mask, result):
        self.synthseg_running = False
        self.progress_bar.setVisible(False)

        # A run takes minutes; if the case was swapped meanwhile this mask
        # belongs to the old one and must not be shown or saved against the
        # new case's images and affine.
        if self.sender() is not None and self.sender().case_folder != self.case_folder:
            self.status.showMessage(
                "Discarded SynthSeg result — the case changed while it was running"
            )
            self._refresh_run_enabled()
            return

        self.synthseg_mask = mask
        self.synthseg_result = result
        self.input_panel.set_save_synthseg_enabled(True)

        self.view_panel.update_overlay_availability(self.segmentation is not None, True)
        self.view_panel.set_overlay(
            OVERLAY_BOTH if self.segmentation is not None else OVERLAY_SYNTHSEG
        )

        self.status.showMessage(f"SynthSeg completed ({result.info})")
        self._refresh_run_enabled()
        self.update_view()

    def on_synthseg_failed(self, message):
        self.synthseg_running = False
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

        mode = self.view_panel.current_overlay()
        show_tumor = (
            mode in (OVERLAY_TUMOR, OVERLAY_BOTH) and self.segmentation is not None
        )
        show_synthseg = (
            mode in (OVERLAY_SYNTHSEG, OVERLAY_BOTH) and self.synthseg_mask is not None
        )

        if show_tumor or show_synthseg:
            overlay = overlay_image_with_masks(
                base,
                tumor_slice=(
                    np.take(self.segmentation, idx, axis=axis) if show_tumor else None
                ),
                synthseg_slice=(
                    np.take(self.synthseg_mask, idx, axis=axis)
                    if show_synthseg
                    else None
                ),
            )
        else:
            overlay = base
        self._render(self.mask_panel["image_view"], overlay, axis)

        if show_tumor and show_synthseg:
            overlay_title = "Tumor + SynthSeg"
        elif show_synthseg:
            overlay_title = "SynthSeg"
        else:
            overlay_title = "Mask"

        self.mri_panel["title_label"].setText(plane)
        self.mask_panel["title_label"].setText(f"{plane} + {overlay_title}")
        self.slice_label.setText(f"Slice {idx + 1} / {axis_len}  ·  {plane}")
        self._update_legend(show_tumor, show_synthseg)

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
