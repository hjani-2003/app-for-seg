# Phase 0 — Repo orientation notes (bidimensional-measurement feature)

Findings below are keyed to the current viewer implementation on `HEAD`
(`ui/main_window.py`, `ui/panels.py`, `core/data_loader.py`). Later phases
should treat this as the source of truth over the phase-0 task description,
since one assumption in that description (napari, synced 3-plane panes) does
not match this codebase — see "Correction" below.

## Correction: this is not a napari app

There is no napari anywhere in this repo (`grep -ril napari .` → no hits).
The viewer is a plain **PySide6 (Qt) desktop app**:
- `MRIViewer(QMainWindow)` in `ui/main_window.py`
- Image rendering via `pyqtgraph.ImageView` (`pg.ImageView`), not napari layers
- No "image layer" / "label layer" objects — each panel is a `pg.ImageView`
  instance that gets a plain `numpy` array pushed into it via
  `image_view.setImage(...)`.

So "napari layers and their naming convention" doesn't apply. The closest
analogue is the panel dict structure described below.

## 1. Segmentation label volume

- **Storage**: `MRIViewer.segmentation` — a plain `numpy.ndarray`, set to
  `None` until inference runs (or ground truth is injected for testing).
- **dtype**: `np.uint8` from every backend (`core/backends/swin_unetr_backend.py`
  / `mavin_backend.py` via `core/inference.py:60`, and
  `core/backends/nnunet_backend.py:58`). All three converge on `uint8`.
- **Label values**: `0=BG, 1=NCR, 2=ED, 3=ET` (`ui/style.py:20`,
  `CLASS_LABELS`). Matches the raw BraTS scheme referenced in your phase
  description — TC/WT still need deriving, nothing in the codebase computes
  them yet.
- **Axis order**: RAS+ canonical, same as the MRI volumes — **axis 0 =
  sagittal, axis 1 = coronal, axis 2 = axial** (`core/constants.py:13`,
  `PLANE_AXES = {"Axial": 2, "Coronal": 1, "Sagittal": 0}`). Every backend
  explicitly reorients its raw prediction with `nib.as_closest_canonical(...)`
  before returning the mask (see `nnunet_backend.py:56-58`), specifically so
  it lines up axis-for-axis with `self.volumes[modality]`. Shape is identical
  to any one modality volume's shape.

## 2. Voxel spacing / affine

- **Where it's loaded**: `core/data_loader.py::load_brats_folder()`, line 32:
  `spacing = tuple(nib.affines.voxel_sizes(img.affine))`, computed once from
  the first modality file's affine **after** `nib.as_closest_canonical()`
  reorientation.
- **Where it lives at runtime**: `MRIViewer.spacing` — a plain 3-tuple of
  floats in mm, e.g. `(sag_mm, cor_mm, ax_mm)`, index-aligned with the RAS+
  axis order above (spacing[0]↔axis 0/sagittal, spacing[1]↔axis 1/coronal,
  spacing[2]↔axis 2/axial).
- **This is the exact attribute to use for mm conversion.** It's already
  proven correct in `MRIViewer._render()` (`ui/main_window.py:263-270`),
  which uses it to fix the display aspect ratio:
  ```python
  axis_x, axis_y = (a for a in (0, 1, 2) if a != axis)
  ratio = self.spacing[axis_y] / self.spacing[axis_x]
  ```
- **No full affine is retained** — only the derived spacing tuple. The
  original `img.affine` is discarded after `voxel_sizes()` is computed. Fine
  for axis-aligned in-plane distance (all we need for 2D bidimensional
  measurement on an axial/coronal/sagittal slice), but there is no shear/
  rotation info available if a future phase ever needed oblique measurement.
- **Open risk for Phase 1**: `_render()` reorients the raw 2D slice before
  display — `transpose(1,0,2)` (or `.T`) → `flipud` → `rot90(k=1)`
  (`ui/main_window.py:258-261`) — before pyqtgraph ever sees it. Any
  measurement tool that draws lines in *displayed* pixel space and needs to
  convert back to mm must account for this same transform chain (the
  transpose swaps which physical spacing axis maps to image rows vs.
  columns; the flip/rotate don't affect distances but do affect which raw
  axis is "x" vs "y" on screen). This needs to be nailed down with a concrete
  worked example in Phase 1 before writing geometry code — not solved here.

## 3. How planes are synced (⚠ differs from what Phase 0 assumed)

The 3-simultaneous-synced-panes layout described in your phase-0 prompt
**existed in an earlier commit** (`c847695`, "Add 4-panel synced view...")
but was **replaced** by commit `f668c45` ("Replace 4-panel grid with 2-panel
synced MRI/mask view") — that's the tip of the parent branch
(`two-panel-view`) this branch was cut from.

Current behavior (`ui/main_window.py`):
- Exactly **one plane is visible at a time**, chosen via a `QComboBox`
  (`ViewPanel.plane_box`, `ui/panels.py`), not three simultaneous panes.
- There are two `pg.ImageView` panels — `self.mri_panel` (plain slice) and
  `self.mask_panel` (slice + mask overlay) — that always show the **same
  plane and same slice index**, because both are driven from one shared
  state: one `QComboBox` (plane) + one `QSlider` (`self.slice_slider`,
  slice index), both feeding a single `update_view()` method
  (`ui/main_window.py:230-254`).
- Switching the plane dropdown fires `_on_plane_changed()`
  (`ui/main_window.py:214-228`), which recomputes the slider's max from the
  new axis's length on the current modality volume, recenters the slider,
  then calls `update_view()`. There is no cross-plane index correspondence
  to maintain since only one plane's slider exists at a time.
- Panel identity: `self._build_panel(title)` (`ui/main_window.py:98-120`)
  returns `{"container": QWidget, "image_view": pg.ImageView, "title_label":
  QLabel}`. No other panels exist besides `mri_panel` / `mask_panel`.

**Implication for the measurement feature**: since only one plane/slice is
ever on screen, the RANO 2D measurement tool only ever needs to reason about
*the currently displayed slice* — there's no need to handle "which of 3
panes did the user click in" the way a synced-triple-view would.

## 4. Where the "RANO Measurements" dock will attach

`MRIViewer` is a `QMainWindow`, and its current UI is entirely built inside
one central widget (`_build_ui()`, `ui/main_window.py:34-96`: top controls →
legend → panels row → slider row → progress bar, all stacked in a single
`QVBoxLayout`). `QMainWindow` reserves separate dock areas outside the
central widget, so a new `QDockWidget` can be added with:

```python
self.measurements_dock = QDockWidget("RANO Measurements", self)
self.addDockWidget(Qt.RightDockWidgetArea, self.measurements_dock)
```

added at the end of `_build_ui()` (or its own `_build_measurements_dock()`
method called from `__init__`). This attaches without touching
`main_layout`/`central` at all — zero risk of disrupting the existing
panel/slider layout. The dock can be made floatable/closable via standard
`QDockWidget` features if that's ever wanted, without extra code.

## Confirmed before Phase 1

- ✅ Label volume access: `self.segmentation` (`np.uint8`, RAS+ axis order
  0/1/2 = sagittal/coronal/axial, values 0-3, `None` until inference runs).
- ✅ Spacing access: `self.spacing` (mm 3-tuple, same axis order, set
  alongside `self.volumes` in `load_folder()`).
- ⚠ Flagged, not yet solved: the pixel→mm mapping must go through the same
  transpose/flip/rotate pipeline as `_render()` — worked out with a concrete
  example in Phase 1, before any geometry/line-fitting code is written.
