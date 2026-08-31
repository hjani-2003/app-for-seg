import os

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSlider, QProgressBar, QStatusBar, QLabel, QSizePolicy
)

from core.backends import radiomics_backend, synthseg_backend
from core.constants import (
    MODALITIES, PLANE_AXES, MODALITY_TO_FILE_KEY,
    OVERLAY_TUMOR, OVERLAY_SYNTHSEG, OVERLAY_BOTH,
)
from core.data_loader import (
    load_brats_folder, normalize_for_display, save_segmentation, save_label_map,
)
from core.inference import InferenceWorker
from core.radiomics_extraction import RadiomicsWorker
from core.synthseg_inference import SynthSegWorker
from ui.mask_render import overlay_image_with_masks
from ui.panels import (
    InputPanel, ModelPanel, ViewPanel, SynthSegPanel, RadiomicsPanel, RanoPanel,
)
from ui.flow_layout import FlowStrip
from ui.legend_dock import LegendDock
from ui.radiomics_window import RadiomicsWindow
from ui.rano_window import RanoWindow
from ui.screen_fit import fit_to_screen, size_within
from ui.style import DARK_STYLESHEET, apply_pg_theme


class MRIViewer(QMainWindow):
    # The size the layout is designed around. Not a demand: it is scaled down
    # to whatever screen the app actually opens on. The width is where the
    # control strip stops needing a third wrapped row next to the legend, so a
    # machine with the room for it shows every control without scrolling.
    DESIGN_SIZE = (1400, 900)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRI Tumor Segmentation - Viewer")

        self.raw_volumes = {}
        self.volumes = {}
        self.spacing = None
        self.paths = {}
        self.case_folder = None
        self.output_dir = None
        self.segmentation = None
        self.worker = None
        self.synthseg_mask = None
        self.synthseg_result = None
        self.synthseg_worker = None
        self.synthseg_running = False
        self.radiomics_result = None
        self.radiomics_worker = None
        self.radiomics_running = False
        self._screen_hooked = False

        apply_pg_theme()
        self._build_ui()
        self.setStyleSheet(DARK_STYLESHEET)
        fit_to_screen(self, *self.DESIGN_SIZE)

    def showEvent(self, event):
        super().showEvent(event)
        # Connected here rather than in __init__ because the native window —
        # and so the signal that reports which screen it is on — does not exist
        # until the window is shown.
        handle = self.windowHandle()
        if handle is not None and not self._screen_hooked:
            handle.screenChanged.connect(self._on_screen_changed)
            self._screen_hooked = True

    def _on_screen_changed(self, screen):
        """Re-fit after a move to another monitor, or a resolution change.

        Undocking a laptop or plugging in a projector can leave a window that
        fitted the old screen larger than the new one. Only ever shrinks: a
        size the user chose is theirs to keep if it still fits.
        """
        if self.isMaximized() or self.isFullScreen() or screen is None:
            return

        available = screen.availableGeometry()
        width, height = size_within(self, available, self.width(), self.height())
        if (width, height) != (self.width(), self.height()):
            self.resize(width, height)

        # A shrunk window keeps its old top-left, which can sit beyond the new
        # screen's edge, so it is pulled back inside.
        x = min(max(self.x(), available.x()), available.right() - width + 1)
        y = min(max(self.y(), available.y()), available.bottom() - height + 1)
        if (x, y) != (self.x(), self.y()):
            self.move(x, y)

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
        self.input_panel.save_radiomics_requested.connect(self.save_radiomics)

        self.model_panel = ModelPanel()
        self.model_panel.run_requested.connect(self.run_inference)
        self.model_panel.architecture_changed.connect(self._refresh_run_enabled)

        self.synthseg_panel = SynthSegPanel()
        self.synthseg_panel.run_requested.connect(self.run_synthseg)

        self.radiomics_panel = RadiomicsPanel()
        self.radiomics_panel.run_requested.connect(self.run_radiomics)
        self.radiomics_panel.show_table_requested.connect(self.show_radiomics_window)

        self.rano_panel = RanoPanel()
        self.rano_panel.open_requested.connect(self.show_rano_window)

        self.view_panel = ViewPanel()
        self.view_panel.changed.connect(self.update_view)
        self.view_panel.plane_changed.connect(self._on_plane_changed)
        self.view_panel.update_overlay_availability(False, False)

        # Wrapping and scrolling, not a fixed row: side by side these five
        # panels demand more width than the screen has, which Qt honours by
        # pushing the window off screen when it is maximised.
        top_controls = FlowStrip(spacing=10)
        top_controls.addWidget(self.input_panel)
        top_controls.addWidget(self.model_panel)
        top_controls.addWidget(self.synthseg_panel)
        top_controls.addWidget(self.radiomics_panel)
        top_controls.addWidget(self.rano_panel)
        top_controls.addWidget(self.view_panel)

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

        main_layout.addWidget(top_controls)
        main_layout.addLayout(panels_row, 1)
        main_layout.addLayout(slider_row)
        main_layout.addWidget(self.progress_bar)
        central.setLayout(main_layout)

        # The legend stays docked: it is a key to what is on screen, read at a
        # glance beside the slices, and narrow enough to cost them nothing.
        self.legend_dock = LegendDock()
        self.addDockWidget(Qt.RightDockWidgetArea, self.legend_dock)

        # The two tables do not. Both were docks tabbed behind the legend,
        # which meant one visible at a time in a column too narrow for either.
        # They open on their own buttons instead, and are built up front so
        # they keep their contents across being closed and reopened.
        self.radiomics_window = RadiomicsWindow()

        # RANO takes the overlay panel's view because it draws its calliper
        # ROIs onto the displayed slice, not into a table alone — so the lines
        # stay on the slice whether or not its window is open.
        self.rano_window = RanoWindow(self.mask_panel["image_view"])

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # Surface a broken SynthSeg install now rather than after a user has
        # loaded a case and waited on a run that could never start.
        self.synthseg_unavailable = synthseg_backend.check_available()
        self.status.showMessage(
            self.synthseg_unavailable or f"Ready · {synthseg_backend.runtime_summary()}"
        )

    def closeEvent(self, event):
        # The tool windows are top-level and parentless, so one left open would
        # keep the application running after the viewer itself is gone.
        self.radiomics_window.close()
        self.rano_window.close()
        super().closeEvent(event)

    def show_radiomics_window(self):
        self.radiomics_window.show_window()

    def show_rano_window(self):
        self.rano_window.show_window()

    def _build_panel(self, title):
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)

        image_view = pg.ImageView()
        # The slices take the slack when the window grows, and are allowed to
        # give width back when it shrinks — they are the one part of the
        # layout with nothing that has to stay legible at a fixed size.
        image_view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        image_view.setMinimumSize(160, 160)
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
            self._reset_radiomics()
            self.rano_window.clear()
            self.input_panel.set_save_enabled(False)
            self.slice_slider.setEnabled(False)
            self.status.showMessage(str(exc))
            self._refresh_run_enabled()
            return

        self.volumes = {m: normalize_for_display(v) for m, v in self.raw_volumes.items()}
        self.case_folder = folder
        self.output_dir = None
        self.segmentation = None
        self._reset_synthseg()
        self._reset_radiomics()
        self.rano_window.clear()
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

    def _default_output_dir(self):
        """Where results go unless the user picks somewhere else.

        A sibling of the case folder rather than the case folder itself: the
        loader scans a case folder for modality files, so results written next
        to the inputs are at best clutter and at worst mistaken for inputs.
        """
        case_path = os.path.normpath(self.case_folder)
        return os.path.join(
            os.path.dirname(case_path), f"{os.path.basename(case_path)}_output"
        )

    def _choose_output_dir(self, title):
        """Ask for a save directory, defaulting to (and creating) the sibling
        results folder. Returns None if the user cancels or picks the case
        folder, which is refused. The choice is remembered for this case so the
        tumour mask and the SynthSeg outputs land together.
        """
        start = self.output_dir or self._default_output_dir()
        try:
            os.makedirs(start, exist_ok=True)
        except OSError:
            start = os.path.dirname(os.path.normpath(self.case_folder))

        directory = QFileDialog.getExistingDirectory(self, title, start)
        if not directory:
            return None

        if os.path.normpath(directory) == os.path.normpath(self.case_folder):
            self.status.showMessage(
                "Pick a directory outside the case folder — saving results "
                "next to the input scans stops the case loading cleanly."
            )
            return None

        self.output_dir = directory
        return directory

    def save_folder(self):
        if self.segmentation is None:
            self.status.showMessage("Run inference before saving")
            return

        directory = self._choose_output_dir("Select Output Directory")
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

        directory = self._choose_output_dir("Select SynthSeg Output Directory")
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
        self.legend_dock.clear()
        self.input_panel.set_save_synthseg_enabled(False)
        self.view_panel.update_overlay_availability(
            self.segmentation is not None, False
        )

    def _reset_radiomics(self):
        self.radiomics_result = None
        self.radiomics_window.clear()
        self.radiomics_panel.set_show_table_enabled(False)
        self.input_panel.set_save_radiomics_enabled(False)

    def _refresh_run_enabled(self):
        has_case = len(self.raw_volumes) == 4
        self.model_panel.set_run_enabled(has_case)

        # Re-checked rather than cached from start-up, so installing the env or
        # the weights takes effect without restarting the app.
        self.synthseg_unavailable = synthseg_backend.check_available()

        if self.synthseg_unavailable is not None:
            # The full message names an absolute path, too long to sit under
            # the button, so the panel shows a short form and keeps the rest
            # in the tooltip.
            reason = self.synthseg_unavailable
            summary = "SynthSeg is not set up — hover for details"
        elif not has_case:
            reason = summary = "Load a BraTS case first"
        elif self.synthseg_running:
            # Loading a case calls this, so without the in-flight check a user
            # could start a second run on top of one already going.
            reason = summary = "A SynthSeg run is already in progress"
        else:
            reason = summary = None

        self.synthseg_panel.set_run_enabled(reason is None, reason, summary)

        # Re-checked here too, so creating the radiomics env takes effect
        # without restarting the app.
        radiomics_unavailable = radiomics_backend.check_available()
        if radiomics_unavailable is not None:
            reason = radiomics_unavailable
            summary = "PyRadiomics is not set up — hover for details"
        elif not has_case:
            reason = summary = "Load a BraTS case first"
        elif self.segmentation is None:
            # Features are extracted over the tumour mask, so there is nothing
            # to extract from until a model has produced one.
            reason = summary = "Run inference first — features need a tumour mask"
        elif self.radiomics_running:
            reason = summary = "A feature extraction is already in progress"
        else:
            reason = summary = None

        self.radiomics_panel.set_run_enabled(reason is None, reason, summary)

        # RANO needs no environment and no options — only a mask to measure.
        # Enabled even when nothing was found automatically, because the window
        # is also where a lesion is measured by hand.
        self.rano_panel.set_open_enabled(
            self.segmentation is not None,
            "Run inference first — RANO measures the tumour mask",
        )

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
        # Features describe the mask they were extracted from, so a new mask
        # retires them rather than sitting alongside as if still current.
        self._reset_radiomics()
        self.rano_window.populate_from_segmentation(self.segmentation, self.spacing)
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
        self.synthseg_panel.set_run_enabled(
            False, "A SynthSeg run is already in progress",
            "Running… this takes a minute or two on CPU",
        )
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

    def run_radiomics(self):
        if self.segmentation is None:
            self.status.showMessage("Run inference before extracting features")
            return

        modalities = self.radiomics_panel.modalities()
        preset = self.radiomics_panel.current_preset()

        self.radiomics_running = True
        self.radiomics_panel.set_run_enabled(
            False, "A feature extraction is already in progress",
            "Extracting… the Extended preset takes about a minute",
        )
        self.progress_bar.setVisible(True)
        self.status.showMessage(
            f"Extracting {preset} features on {', '.join(modalities)}... "
            f"({radiomics_backend.runtime_summary()})"
        )

        # The raw volumes, not the display copies: those are min-max normalized
        # per volume, which would make first-order features a property of the
        # scaling. MODALITIES[0] supplies the geometry, as the save paths do.
        self.radiomics_worker = RadiomicsWorker(
            self.raw_volumes,
            self.segmentation,
            self.paths[MODALITIES[0]],
            modalities,
            preset,
            self.radiomics_panel.params_path(),
            self.case_folder,
        )
        self.radiomics_worker.finished.connect(self.on_radiomics_done)
        self.radiomics_worker.failed.connect(self.on_radiomics_failed)
        self.radiomics_worker.progress.connect(self.status.showMessage)
        self.radiomics_worker.start()

    def on_radiomics_done(self, result):
        self.radiomics_running = False
        self.progress_bar.setVisible(False)

        # A run takes long enough to swap the case meanwhile; if that
        # happened these features belong to the old one.
        if self.sender() is not None and self.sender().case_folder != self.case_folder:
            self.status.showMessage(
                "Discarded radiomic features — the case changed while they were "
                "being extracted"
            )
            self._refresh_run_enabled()
            return

        self.radiomics_result = result
        self.radiomics_window.set_content(result)
        self.radiomics_panel.set_show_table_enabled(True)
        # Opened rather than merely enabled: the table is what the run was for,
        # and the window is where it now lives.
        self.radiomics_window.show_window()
        self.input_panel.set_save_radiomics_enabled(True)

        self.status.showMessage(f"Feature extraction completed ({result.info})")
        self._refresh_run_enabled()

    def on_radiomics_failed(self, message):
        self.radiomics_running = False
        self.progress_bar.setVisible(False)
        self.status.showMessage(message)
        self._refresh_run_enabled()

    def save_radiomics(self):
        if self.radiomics_result is None:
            self.status.showMessage("Extract features before saving")
            return

        directory = self._choose_output_dir("Select Feature Output Directory")
        if not directory:
            return

        result = self.radiomics_result
        case_name = os.path.basename(os.path.normpath(self.case_folder))
        # The preset is part of the name because a Standard and an Extended run
        # of the same case are different tables, not versions of one.
        out_path = os.path.join(
            directory, f"{case_name}_radiomics_{result.preset.lower()}.csv"
        )

        try:
            with open(out_path, "w", newline="") as handle:
                handle.write(result.to_csv())
        except Exception as exc:
            self.status.showMessage(f"Save failed: {exc}")
            return

        self.status.showMessage(f"Saved features to {out_path}")

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
        self._update_legend(show_tumor, show_synthseg, idx, axis)
        self.rano_window.show_slice(idx, plane)

    def _update_legend(self, show_tumor, show_synthseg, idx, axis):
        tumor_labels = (
            self._labels_in(self.segmentation) if self.segmentation is not None else []
        )
        synthseg_labels = (
            self._labels_in(self.synthseg_mask) if self.synthseg_mask is not None else []
        )
        self.legend_dock.set_content(tumor_labels, synthseg_labels)
        self.legend_dock.set_sections_visible(show_tumor, show_synthseg)
        self.legend_dock.set_present(
            self._labels_in(np.take(self.segmentation, idx, axis=axis))
            if show_tumor else [],
            self._labels_in(np.take(self.synthseg_mask, idx, axis=axis))
            if show_synthseg else [],
        )

    @staticmethod
    def _labels_in(mask):
        return [int(v) for v in np.unique(mask) if v > 0]

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
