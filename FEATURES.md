# Features

What the viewer does, and the reasoning behind the parts that aren't obvious.
For getting it running see [README.md](README.md); for plugging in your own
model see [CUSTOM_MODELS.md](CUSTOM_MODELS.md).

---

## Viewing

**Two synced panels.** The raw slice on the left, the same slice with overlays
on the right. One plane selector and one slider drive both, so they can never
drift out of step.

**Three planes.** Axial, coronal, sagittal. Volumes are reoriented to RAS+
canonical on load, so array axes always mean the same thing regardless of how
the file was stored.

**Four modalities.** T1, T1Gd, T2, FLAIR, matched from filenames by substring.
Switching modality changes the underlying image without disturbing overlays.

**Correct aspect ratio.** In-plane pixel aspect is corrected by the physical
voxel spacing read from the affine, so non-cubic voxels don't render stretched.

---

## Tumour segmentation

Three architectures, selected from the Model panel:

| Architecture | Backing repo | Notes |
|---|---|---|
| **SwinUNETR** | `models/swin_unetr` | MONAI SwinUNETR, built from `configs/model.yaml`. CPU or GPU. |
| **MaViN** | `models/mavin-hpc` | MambaVision-UNet. **GPU only** — the `mamba_ssm` CUDA kernels have no CPU fallback. |
| **nnUnet** | `models/nnunet` | nnU-Netv2, reading a full trained results folder. |

Output is a 4-class mask: background, NCR (necrotic core), ED (edema), ET
(enhancing tumour).

**Checkpoints are discovered, not configured.** Drop a `.pth` into
`models/<name>/checkpoints/` and it is picked up — both raw `state_dict` and
`{"state_dict": ...}` formats. With no checkpoint the architecture still runs
with random weights and says so, which is useful for testing the plumbing but
worth noticing, because the output looks like a segmentation.

**Inference runs off the UI thread** in a `QThread`, so the window stays
responsive. Failures that a user could fix — a missing results folder, a
GPU-only kernel on a CPU machine — surface as status-bar messages rather than
tracebacks.

---

## SynthSeg anatomical segmentation

[SynthSeg](https://github.com/BBillot/SynthSeg) segments whole-brain anatomy
from a **single** scan: 32 FreeSurfer `aseg` structures, optionally 68
Desikan-Killiany cortical parcels. It answers a different question from the
tumour models — which structures a lesion abuts — so it has its own panel,
overlay layer, and save button rather than joining the architecture dropdown.

### It runs in its own environment

SynthSeg needs Python 3.8 + TensorFlow 2.2 + Keras 2.3; the app needs Python
3.11 + torch + numpy 2.x. These cannot share an interpreter, so SynthSeg runs
as a **subprocess** in its own conda env and communicates over files. Nothing
is imported in-process.

### Options

| Option | Effect |
|---|---|
| **Modality** | Which loaded volume to segment. Defaults to `T1`. SynthSeg is contrast-agnostic by design, so others work but are less validated. |
| **Fast** | Skips topology postprocessing. Roughly twice as fast, marginally less accurate. |
| **Robust** | Better on low-quality clinical scans, slower. Implies Fast, so that box locks on. |
| **Parcellation** | Replaces cortex labels 3/42 with 68 Desikan-Killiany regions, and adds 68 columns to the volumes CSV. |

### It adapts to the machine

- **GPU is detected and vetted.** TensorFlow 2.2 ships CUDA 10.1 kernels for
  compute capability ≤ 7.5. Anything newer — A100, RTX 30xx/40xx/50xx, H100 —
  cannot be used at all, so those fall back to CPU *with an explanation*
  rather than failing. The status bar always says which mode was chosen.
- **Threads scale with RAM.** TF 2.2's multithreaded `Conv3D` allocates a
  workspace per thread and aborts with `std::bad_alloc` at full-brain patch
  sizes on a 16 GB machine. The default is 1 thread under 32 GB, otherwise
  `min(8, cores/2)`.
- **The analysed patch is sized to the brain.** Rather than SynthSeg's fixed
  192³, each axis is sized from the brain's furthest voxel from the image
  centre — about a third less volume for BraTS, at no cost in coverage.

All three are overridable with `SYNTHSEG_GPU`, `SYNTHSEG_THREADS` and
`SYNTHSEG_CROP`.

---

## RANO bidimensional measurement

[RANO](https://pubmed.ncbi.nlm.nih.gov/37307495/) sizes a tumour by two
perpendicular diameters on the slice where it is largest, and tracks the sum of
their products across visits. It is the criterion trials are read against, so
the viewer measures it directly rather than leaving it to be re-derived by hand
from a volume.

Measurements appear in the **RANO Measurements** dock as soon as inference
finishes, one row per lesion, with the diameters drawn on the overlay panel.

### Two regions, and one honest unknown

| Region | Labels | |
|---|---|---|
| **CE** | ET | contrast-enhancing, matching RANO's CE-lesion definition |
| **nonCE** | NCR + ED | non-enhancing |

The CE region is uncontroversial — RANO's definition already excludes the
necrotic core. **nonCE is a modelling choice the app does not pretend to have
settled.** RANO's non-enhancing tumour excludes vasogenic edema, and edema is
not separable from tumour by segmentation label alone. The default includes
both NCR and ED; `rano_measure/regions.py:REGION_DEFS_ED_ONLY` is the
alternative, exposed as a toggle rather than decided silently.

### How a lesion is measured

Lesions are 3D connected components. For each, the top few candidate slices by
area are measured and the best kept — the largest cross-section is not always
the one with the largest bidimensional product.

On a slice, the contour is reduced to at most 80 points, every chord between
them that stays inside the mask is a candidate, and the pair maximising the
product is chosen subject to being perpendicular within 5°. A lesion is
**measurable** only if both diameters reach 10mm, which is RANO's threshold.

Diameters are computed in millimetres from the voxel spacing, not in pixels, so
non-isotropic data measures correctly.

### Target lesion selection

RANO 2.0 caps a mixed tumour at **3 CE targets and 4 total**. Measurable
lesions are ranked by product and selected under those caps
(`rano_measure/burden.py`), and the dock sums the products per region — the
number that actually gets compared between visits.

### Measurements can be corrected by hand

Automatic diameters are a starting point, not a verdict. Each pair is a
draggable ROI on the slice, and a pair can be drawn from scratch by clicking
two points; a manual pair is validated the same way an automatic one is
(inside the mask, perpendicular within tolerance). This is axial-only — the
RANO criterion is defined on axial slices, and the controls disable themselves
in the other planes rather than silently measuring something else.

### The measurement core has no Qt in it

`rano_measure/` is pure numpy and scikit-image: region composition, connected
components, contour geometry, burden selection. The widget lives in
`ui/rano_dock.py`, outside the package. This was enforced after
`rano_measure/regions.py` reached into `ui/style.py` for the label names and
transitively pulled in pyqtgraph — the canonical mapping now lives in
`core/constants.py:LABEL_NAMES`, which is also where the radiomics region
definitions read it from. `NOTES.md` records that and the rest of the design
log.

---

## Radiomic features

[PyRadiomics](https://pyradiomics.readthedocs.io/) computes standardised,
IBSI-aligned descriptors of an ROI — shape, intensity distribution, and five
families of texture matrix. Where the tumour model says *where* the lesion is,
this says *what it looks like*, as numbers a downstream model can consume.

It runs over the tumour mask, so **Extract Features** stays disabled until a
segmentation exists.

### Five regions, not three

Features are extracted over each model class on its own and over the two
composites the BraTS literature reports against:

| Region | Labels | |
|---|---|---|
| **NCR** | 1 | necrotic core |
| **ED** | 2 | peritumoral edema |
| **ET** | 3 | enhancing tumour |
| **TC** | 1 + 3 | tumour core |
| **WT** | 1 + 2 + 3 | whole tumour |

The composites are extracted, not derived. A texture feature of the whole
lesion is not recoverable from the same feature computed over its parts.

A region with no voxels in a given case — a non-enhancing tumour has no ET — is
reported as skipped rather than failing the run, because that is a fact about
the case, not an error.

### Three presets

Selected per run; each is a PyRadiomics parameter YAML in
`core/radiomics_params/`, and **Custom params.yaml…** takes one of your own
(`RADIOMICS_PARAMS` overrides the presets from the environment).

| Preset | Contents | Features per region | A full run |
|---|---|---|---|
| **Fast** | shape + first-order, unfiltered | 32 | ~14s |
| **Standard** | all seven feature classes, unfiltered | 107 | ~17s |
| **Extended** | adds Laplacian-of-Gaussian (σ 1, 3, 5) and one wavelet level | 1130 | ~60s |

Timings are for all five regions across all four modalities on a 182×218×182
BraTS case, single-threaded on a desktop CPU.

**MR is normalised before binning.** PyRadiomics' default `binWidth` of 25 is a
CT setting: a Hounsfield unit means the same thing in every scan, so a fixed
bin width discretises comparably. MR intensities have no such scale, so the
presets normalise to a fixed mean and standard deviation first
(`normalize: true`, `normalizeScale: 100`) and then bin finely (`binWidth: 5`).
Without that, the same tissue in two scans lands in different grey levels and
every texture feature is partly a measurement of the scanner.

**Resampling is off.** BraTS is already 1mm isotropic and co-registered. Point
the panel at a custom YAML with a `resampledPixelSpacing` for data that is not.

### It runs in its own environment

Same reason as SynthSeg, different specifics: PyRadiomics' last release ships
wheels for CPython 3.7-3.9 with C extensions built against the numpy 1.x ABI,
and this app is Python 3.11 with numpy 2. In a Python 3.9 env it is a prebuilt
wheel with nothing to compile, so it runs as a subprocess there — see the
README for the setup, and `python check_radiomics.py` to diagnose one.

Unlike SynthSeg there is no CLI worth driving. The `pyradiomics` command can do
batches, but a batch there is all-or-nothing: one region too small for a
texture matrix aborts the lot. `core/backends/radiomics_runner.py` runs in the
child env instead and isolates each pair, so a degenerate region costs you that
row and nothing else.

**Image and mask are written from the same affine.** Both go through
`core/data_loader.py` with the canonical affine of the reference modality, so
PyRadiomics sees two volumes on byte-identical geometry — which is what keeps
it from rejecting the pair as misaligned when the source file was not stored
RAS+ to begin with. The images handed over are the **raw** volumes, never the
display copies: those are min-max scaled per volume, which would make every
first-order feature a property of the scaling rather than of the tissue.

### The table is transposed

Features go down and regions across. The natural shape is one row per
(modality, region) — that is how it is saved and how a model wants it — but
that table is 107 columns wide on Standard and 1130 on Extended. Transposed,
the widest it gets is four modalities by five regions, and the long axis is the
one a scrollbar handles well. A class filter and a name search sit above it,
which is the only way Extended is readable at all.

---

## Overlays

The right-hand panel draws **Tumor**, **SynthSeg**, or **Both**. Modes whose
mask hasn't been produced yet are greyed out.

In `Both`, SynthSeg is drawn underneath at a lower alpha (0.30) with the
tumour mask on top (0.45), so tumour stays readable against anatomy.

### Colours are chosen, not picked

**Tumour classes** — red (NCR), cyan (ED), yellow (ET). Selected by maximising
CIEDE2000 separation *as displayed* (alpha-blended over brain-intensity greys)
between the three classes, with distinctness under simulated deuteranopia and
protanopia as a hard constraint, and distance from the anatomy palette as the
tie-break.

This matters more than it sounds. The previous blue/green/orange set sat at
ΔE 2.3 from the accumbens colour — below the just-noticeable-difference
threshold. With both overlays on, tumour classes were literally the same
colour as the structures beneath them.

| | among the three | colour-blind | nearest anatomy |
|---|---|---|---|
| before | ΔE 24.9 | ΔE 13.3 | ΔE 2.3 |
| after | ΔE 43.3 | ΔE 26.4 | ΔE 9.5 |

**Anatomy** uses the canonical FreeSurfer `aseg` colours, so the overlay
matches what freeview and ITK-SNAP show. SynthSeg ships no colour table — its
own `labels table.txt` leaves it to the viewer — so they are vendored in
`ui/synthseg_lut.py`. Cortical parcels get a generated palette; point
`SYNTHSEG_LUT` at a real `FreeSurferColorLUT.txt` to make those canonical too.

---

## Legend

A dock on the right listing whatever the overlay is currently showing, in two
sections: Tumour and Anatomy.

Structures **present in the slice on screen** are bold; the rest are dimmed.
The list itself never changes as you scrub — only weight and colour do — so
nothing reflows under the cursor while you move the slider.

It lives beside the images rather than above them because a case carries 35
legend rows normally and around 100 with parcellation. That needs a tall
narrow column, not a wide short one, and the window has horizontal space to
spare where it has none vertically. Drag it to the left edge, float it, or
close it from its title bar.

---

## Saving

Three independent buttons. The two that write a `.nii.gz` use the affine of
their reference modality, taken from the canonical form so the file matches
what was displayed.

**Save Segmentation** — the tumour mask:

```
<case>_seg.nii.gz                                  uint8, labels 0-3
```

**Save Features** — the radiomics table, long-form:

```
<case>_radiomics_<preset>.csv     one row per (modality, region), features across
```

The preset is in the name because a Standard and an Extended run of the same
case are different tables, not versions of one. Like the SynthSeg CSVs, the
result is held in memory from the run, so saving never re-extracts.

**Save SynthSeg** — the anatomy mask plus both CSVs:

```
<case>_synthseg_<modality>[_parc].nii.gz           int16, FreeSurfer labels
<case>_synthseg_<modality>[_parc]_volumes.csv      per-structure volumes, mm³
<case>_synthseg_<modality>[_parc]_qc.csv           per-region QC scores, 0-1
```

The modality is in the name because SynthSeg can run on any loaded modality,
and `_parc` because parcellated runs use a different label space — neither
should silently overwrite the other. CSVs are held in memory from the run, so
saving never re-runs the model.

---

## Failure handling

The parts that exist because they were needed:

**Results are discarded when the case changed.** A SynthSeg run, or an Extended
feature extraction, takes minutes — long enough to load a different case
meanwhile. Each worker records which case it was started for and stale results
are dropped, rather than overlaying one patient's anatomy on another's images.

**Features are retired when the mask changes.** Re-running inference clears the
feature table, rather than leaving numbers on screen that describe a mask no
longer displayed.

**Only one run at a time.** Loading a case refreshes the UI, which previously
re-enabled the Run button mid-run and allowed a second run on top of the
first.

**A disabled button says why**, in the panel and in its tooltip:
`SynthSeg is not set up`, `Load a BraTS case first`, `Running…`. Availability
is re-checked whenever a case loads, so installing the environment takes
effect without restarting.

**`check_synthseg.py` and `check_radiomics.py`** print every path their backend
looks for and whether it exists. It tolerates a half-updated checkout and reports that as the
diagnosis, because a diagnostic that crashes is worse than none.

**Errors reach the user.** A subprocess that dies takes its last 25 lines of
output with it into the error message — for an out-of-memory abort that tail
is the only diagnosis available.
