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

Two independent buttons; each writes a `.nii.gz` with the affine of its
reference modality, taken from the canonical form so the file matches what was
displayed.

**Save Segmentation** — the tumour mask:

```
<case>_seg.nii.gz                                  uint8, labels 0-3
```

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

**Results are discarded when the case changed.** A SynthSeg run takes minutes,
long enough to load a different case meanwhile. The worker records which case
it was started for and stale results are dropped, rather than overlaying one
patient's anatomy on another's images.

**Only one run at a time.** Loading a case refreshes the UI, which previously
re-enabled the Run button mid-run and allowed a second run on top of the
first.

**A disabled button says why**, in the panel and in its tooltip:
`SynthSeg is not set up`, `Load a BraTS case first`, `Running…`. Availability
is re-checked whenever a case loads, so installing the environment takes
effect without restarting.

**`check_synthseg.py`** prints every path the backend looks for and whether it
exists. It tolerates a half-updated checkout and reports that as the
diagnosis, because a diagnostic that crashes is worse than none.

**Errors reach the user.** A subprocess that dies takes its last 25 lines of
output with it into the error message — for an out-of-memory abort that tail
is the only diagnosis available.
