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

## Phase 3 addendum: napari → pyqtgraph translation

The Phase 3 spec was written in napari terms (Shapes layers,
`layer.events.data`, native shape selection). Confirmed with the user this
app has no napari (see the Phase 0 correction above) and got explicit
sign-off to reinterpret every napari-specific instruction as its pyqtgraph
equivalent rather than adding napari as a second embedded GUI framework.
Mapping used, in `ui/rano_window.py`:

| Spec (napari) | This app (pyqtgraph) |
|---|---|
| Two Shapes layers, one per region type | One `pg.LineSegmentROI` pair (major+minor) per lesion, added directly to `mask_panel`'s existing `ViewBox`; colored by `region_type` via `ui/style.py:RANO_LINE_COLORS` |
| `layer.events.data` | Each ROI's `sigRegionChanged` (fires live during drag, not just on release) |
| Native shape selection + Delete | `QTableWidget` row selection + a `QShortcut(Qt.Key_Delete)` scoped to the table. Bonus: `pg.ROI`'s own built-in right-click "Remove" (`sigRemoveRequested`) is also wired to the same removal path, since it turned out to exist natively. |
| Add-line-pair mode | A checkable "Add Line Pair" button + region-type combo; hooks the mask panel's `scene().sigMouseClicked` and collects exactly 4 clicks (major p1, major p2, minor p1, minor p2) per lesion, then stays in add mode for further pairs until toggled off. |

**Coordinate mapping (the load-bearing part).** `rano_measure.geometry`/
`lesion` compute lines in raw voxel (row, col) space — the same space
`np.take(volume, idx, axis=axis)` uses — not the flipped/rotated space
`_render()` builds only for display. This was flagged as an open risk back
in Phase 1 and had to be resolved here. Empirically verified (not assumed)
in development:
- `_render()`'s `transpose → flipud → rot90(k=1)` chain collapses to a
  single 180° point reflection: `display[R-1-r, C-1-c] = base[r, c]` for a
  slice of shape `(R, C)`. Verified by probing every cell of a small test
  array through the actual `_render()` code.
- pyqtgraph's default `imageAxisOrder='col-major'` maps an `ImageItem`'s
  array index `(axis0, axis1)` directly to view-space `(x, y)` with no
  swap (verified via `ImageItem.mapToView()` and `boundingRect()` on a
  known asymmetric test array).
- Composing these: voxel `(r, c)` in an `(R, C)` slice → view-space
  `(R-1-r, C-1-c)`, and since a point reflection is its own inverse, the
  same formula converts a clicked/dragged view-space point back to voxel
  space. Implemented as `RanoWindow._voxel_to_view` / `_view_to_voxel`.
- End-to-end correctness (not just the isolated formula) was confirmed by
  driving the real app against a real BraTS case: the computed CE lesion
  line rendered exactly on top of the ET (orange) region in the mask
  overlay panel, and a synthetic drag/add-line-pair test round-tripped an
  exact 30mm/20mm move back through the pipeline correctly.

**Scope decisions made explicit** (per the spec's own instruction not to
leave these implicit):
- Manual line drawing/editing is Axial-only, exactly as the spec
  requested — enforced by disabling the add controls and hiding all ROIs
  whenever the plane selector isn't "Axial" (`RanoWindow.show_slice`).
- RANO lesion detection itself (`populate_from_segmentation`) always runs
  against the axial axis regardless of which plane is currently displayed
  — it's computed once per segmentation, not per plane.
- The "Add Line Pair" toggle stays active across multiple lesions (doesn't
  auto-untoggle after one pair) so the user isn't forced to re-click it
  for every new lesion; the spec didn't say either way, this was the more
  usable reading.
- Manual-pair validation warnings surface via a persistent non-modal label
  in the dock, not a popup dialog — keeps "warn, don't block" from being
  disruptive.

## Phase 4: edge cases + module layout

**Real bug found and fixed**: a lesion touching the image boundary
(row 0 / col 0 / last row / last col of the 2D slice) came back completely
unmeasurable. `skimage.measure.find_contours` can't detect a 0.5-level
crossing at the array edge (there's no "outside" pixel to compare
against), so the returned contour for an edge-touching shape was an
open/incomplete loop rather than its true closed boundary — and for at
least one concrete case that degenerate contour then hit a
round-half-to-even rounding-tie pathology in `_segment_valid` that killed
every candidate segment outright. Fixed in `rano_measure/geometry.py`'s
`_contour_points` by padding the mask with a 1px border of `False` before
calling `find_contours`, then shifting the returned coordinates back by
that padding. Covered by
`test_geometry.py::test_lesion_touching_image_boundary_is_measured_not_dropped`
and the equivalent full-pipeline test in `test_lesion.py`. Re-ran the
Phase 2 real-BraTS-case validation afterward — numbers unchanged (real
tumors in that sample aren't edge-touching), confirming no regression.

**Other three edge cases were already correctly handled by the Phase 2/3
design**, verified (not just assumed) with new tests + a UI driver script:
- Multiple disjoint same-raw-label components (e.g. two separate ET
  blobs) already become separate `Lesion` objects with distinct ids
  (`ndimage.label` + per-component id assignment) — confirmed neither the
  table nor the ROI overlay dedupe/merge/hide them, including the case
  where two lesions share the exact same best-candidate slice_index (both
  lesions' ROI pairs render simultaneously).
- A component under the 10mm threshold was already returned as a `Lesion`
  with `measurable=False` rather than dropped (`lesion.py`'s
  `_measure_best_slice` never filters on measurability) — the table always
  shows it, just flagged.
- Zero lesions for a region already produces an empty table + `0.0 mm^2 (0
  lesions)` sum labels via `select_target_lesions([])`, not an exception.

**Layering bug found and fixed**: `rano_measure/regions.py` imported
`ui.style.CLASS_LABELS` for its `LABEL_IDS` mapping (a Phase 1 decision,
made before this "zero Qt imports in rano_measure/" constraint was
stated) — and `ui/style.py` imports `pyqtgraph`, so `rano_measure/` was
transitively pulling in a GUI framework. Fixed by moving the canonical
label id↔name mapping to `core/constants.py:LABEL_NAMES` (a plain-Python
module with no GUI deps) and having both `ui/style.py:CLASS_LABELS` and
`rano_measure/regions.py:LABEL_IDS` derive from it. Verified with
`grep -rn "^import\|^from" rano_measure/*.py` — zero Qt/napari/ui imports
across all four files.

**Suggested module layout — one deliberate deviation, not applied**: the
spec's suggested layout puts the Phase 3 GUI widget at
`rano_measure/napari_widget.py`, i.e. *inside* the pure-logic package, to
enforce "zero napari/Qt imports outside that one file." This app's
widget lives at `ui/rano_window.py` instead — *outside* `rano_measure/`
entirely, matching this codebase's existing convention of keeping all Qt
code under `ui/` (`ui/main_window.py`, `ui/panels.py`, `ui/style.py`,
etc.) and business logic under `core/`. This satisfies the underlying
goal more strictly than the suggested layout would: `rano_measure/` has
*zero* GUI imports, not just GUI imports isolated to one file within it.
Not renaming/moving `rano_window.py` into `rano_measure/` to match the
suggested path, since doing so would break the codebase's own existing
separation of concerns for no functional benefit.


---

# Later change — the table left the dock area

Phase 3 attached the measurements to a `QDockWidget` on the right, tabbed
behind the legend (see section 4 above), and radiomics arrived the same way.
Two things went wrong with that in use:

- Tabbed, only one of the three was visible at a time, and the two that
  mattered most were the two hidden behind the legend.
- Docked, they took width from the slices — and neither table fits the
  column they were given. RANO is eight columns of numbers; radiomics is up
  to twenty columns and 1500 rows.

Both now open as their own top-level windows (`ui/tool_window.py`), on
buttons in the control strip: **Radiomics → Feature Table** and
**RANO → Open Measurements**. What this changed, and what it deliberately
did not:

- The measurement state and the calliper ROIs live on in `RanoWindow`
  whether or not the window is open, so closing it is not "clear" — the
  lines stay on the slice, and reopening shows the same table. `show_slice`
  is still driven from `MRIViewer.update_view` for the same reason.
- Add-line-pair mode reads clicks on the *viewer's* slice, not on its own
  window, so `RanoWindow.closeEvent` disarms it. Otherwise closing the
  window would leave clicks on the image being collected with the prompt
  that explains them out of sight.
- The windows are deliberately parentless, so they can be pushed behind the
  viewer or dragged onto a second monitor rather than being pinned above it.
  The cost is that `MRIViewer.closeEvent` has to close them itself: a
  top-level window left open keeps the application alive after the viewer
  is gone.
- The legend stayed a dock. It is a key to what is on screen, read at a
  glance beside the slices, and narrow enough to cost them nothing.
