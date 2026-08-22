# MRI Tumor Segmentation Viewer

A PySide6 desktop app for loading a BraTS-style brain MRI case, running one of
three segmentation architectures on it, and viewing the result slice-by-slice
with a color-coded tumor sub-region overlay.

---

## Contents

- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running the app](#running-the-app)
- [Using the viewer](#using-the-viewer)
- [Model backends](#model-backends)
- [SynthSeg](#synthseg)
- [Checkpoints](#checkpoints)
- [Troubleshooting](#troubleshooting)

---

## Project structure

```
app-for-seg/
├── main.py                        # entry point — python main.py
├── requirements.txt
├── core/
│   ├── constants.py                # modality/architecture/view-mode lists, channel order
│   ├── data_loader.py              # loads a BraTS case folder into raw numpy volumes
│   ├── preprocessing.py            # z-score-over-nonzero-voxels normalization (matches training)
│   ├── inference.py                # InferenceWorker (QThread): dispatches to a backend, runs
│   │                                #   sliding-window inference, reports mask or error
│   ├── synthseg_inference.py       # SynthSegWorker (QThread): runs SynthSeg, streams progress
│   └── backends/
│       ├── swin_unetr_backend.py    # builds MONAI SwinUNETR from models/swin_unetr/configs
│       ├── mavin_backend.py         # builds MambaVision-UNet from models/mavin-hpc/src (GPU only)
│       ├── nnunet_backend.py        # nnUNetv2 predictor over the raw modality files
│       ├── synthseg_backend.py      # SynthSeg via subprocess — see "SynthSeg" below
│       ├── checkpoints.py           # find_checkpoint(): looks in models/<name>/checkpoints/*.pth
│       └── errors.py                # ModelUnavailableError
├── ui/
│   ├── main_window.py               # MRIViewer — wires panels together, load/run/view logic
│   ├── panels.py                    # InputPanel, ModelPanel, SynthSegPanel, ViewPanel
│   ├── style.py                     # dark theme QSS + segmentation class colors
│   ├── synthseg_lut.py              # FreeSurfer label names/colors for the SynthSeg overlay
│   └── mask_render.py               # colorize_mask() / overlay_image_with_masks()
└── models/                          # training repos for each architecture (see each GUIDE.md)
    ├── swin_unetr/
    ├── mavin-hpc/                   # MambaVision
    ├── nnUnet/
    └── synthseg/                    # vendored SynthSeg + weights (not in git)
```

The `models/*` directories are separate training pipelines (each has its own
`GUIDE.md`) — the viewer only *reads* their `configs/*.yaml` and, if present,
a trained checkpoint from `models/<name>/checkpoints/`. It does not need you
to run any of their training scripts to work; without a checkpoint it just
runs the architecture with random weights (see [Checkpoints](#checkpoints)).

---

## Requirements

- Python 3.11 (developed against 3.11.15)
- A conda environment named `unet_venv` is used across this whole project
  (viewer + all three training repos) — reuse the same one rather than
  creating a separate env for the viewer.
- GPU is **optional** for the viewer itself and for SwinUNETR, but
  **required** for MambaVision (see [Model backends](#model-backends)).

Versions this was developed/tested against:

| Package | Version |
|---|---|
| PySide6 | 6.11.1 |
| pyqtgraph | 0.14.0 |
| nibabel | 5.4.2 |
| numpy | 2.4.4 |
| torch | 2.12.0 |
| monai | 1.5.2 |
| pyyaml | 6.0.3 |
| einops | 0.8.2 |

## Setup

```bash
# 1. Create/activate the shared environment (skip if you already have it
#    from setting up one of the models/ training repos)
conda create -n unet_venv python=3.11
conda activate unet_venv

# 2. Install the viewer's dependencies
cd app-for-seg
pip install -r requirements.txt

# 3. Install torch matching your machine (CPU-only or a specific CUDA build)
#    — see https://pytorch.org/get-started/locally/ for the right command.
#    A plain `pip install torch` works but pulls a CUDA build by default.
```

### Optional: enabling MambaVision

MambaVision needs the Mamba selective-scan CUDA kernels, which require a CUDA
GPU to build (**no CPU fallback exists**). Skip this if you don't have one —
the viewer will still run, it'll just report MambaVision as unavailable.

```bash
conda activate unet_venv
pip install causal-conv1d>=1.1.0
pip install mamba-ssm

# verify:
python -c "import mamba_ssm, causal_conv1d; print('mamba kernels OK')"
```

---

## Running the app

```bash
cd app-for-seg
python main.py
```

This opens the viewer window. No arguments, no config files to edit first.

---

## Using the viewer

1. **Load BraTS Folder** (Input panel) — pick a folder containing exactly
   four NIfTI files for one case, matched by filename substring:

   | Substring | Modality |
   |---|---|
   | `t1n` | T1 |
   | `t1c` | T1Gd (contrast-enhanced) |
   | `t2w` | T2 |
   | `t2f` | FLAIR |

   If any of the four is missing, the status bar names exactly which ones
   (e.g. `Missing modalities in folder: T2, FLAIR`) and nothing else changes.
   On success the slice slider activates, centered on the middle slice.

2. **Architecture** (Model panel) — pick `SwinUNETR`, `MaViN`, or `nnUnet`
   (see [Model backends](#model-backends)).

3. **Run Inference** — enabled once a valid case is loaded and the selected
   architecture is available. Runs on a background thread (the window stays
   responsive; the button disables and an indeterminate progress bar shows
   while it runs). When it finishes, the status bar reports either the
   checkpoint file used or that it ran with random weights.

4. **Run SynthSeg** (SynthSeg panel) — whole-brain anatomical segmentation of
   a single scan, independent of the tumour model. See
   [SynthSeg](#synthseg) below.

5. **Plane** / **Modality** / **Overlay** (View panel) — `Plane` picks
   axial/coronal/sagittal for both panels at once; `Modality` picks which of
   the 4 loaded volumes to show; `Overlay` picks what the right-hand panel
   draws on top — `Tumor`, `SynthSeg`, or `Both`. Modes whose mask has not
   been produced yet are greyed out.

   The window shows two synced panels: the raw slice on the left, the same
   slice with the selected overlay on the right. In `Both`, SynthSeg is drawn
   underneath at a lower alpha so the tumour mask stays readable on top.

6. **Slice slider** — scrubs through the volume along the selected plane's
   axis.

7. **Legend** — appears above the images and lists whatever the overlay is
   currently showing. Tumour colors are fixed per class:

   | Class | Meaning | Color |
   |---|---|---|
   | NCR | Necrotic core | blue |
   | ED | Edema | aqua/green |
   | ET | Enhancing tumor | yellow/orange |

   SynthSeg entries list only the structures present in that case, using the
   canonical FreeSurfer colors.

---

## Model backends

The three dropdown options map to the three `models/` training repos:

| Architecture | Backing repo | Status |
|---|---|---|
| **SwinUNETR** | `models/swin_unetr` | Fully wired. Runs on CPU or GPU. |
| **MaViN** | `models/mavin-hpc` | Fully wired, **GPU only** — the mamba_ssm CUDA kernels have no CPU fallback. On a machine without them, selecting it and running produces a clear status-bar error instead of a crash. |
| **nnUnet** | `models/nnUnet` | Fully wired. Unlike the other two it needs a trained *results folder* (`plans.json` + `dataset.json` + `fold_0/checkpoint_best.pth` from `nnUNetv2_train`) at `models/nnunet/results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres`, and it reads the original NIfTI files directly rather than the preloaded arrays, because it does its own preprocessing from `plans.json`. |

SynthSeg is deliberately *not* in this dropdown: it segments healthy anatomy
from one scan rather than tumour from four, so it is a separate control with
its own overlay layer.

For SwinUNETR and MambaVision, the viewer:
1. Builds the architecture from `models/<name>/configs/model.yaml`.
2. Loads a checkpoint if one is found (see below), else runs with random
   weights (dummy inference — useful for testing the UI/plumbing before real
   checkpoints exist).
3. Normalizes the 4 input channels (z-score over nonzero voxels, per
   channel) and stacks them in the order `[T1Gd, T1, FLAIR, T2]` — this is
   the channel order those models were actually trained on (from their
   dataset JSONs), *not* the order the modalities are listed in the UI.
4. Runs MONAI's `sliding_window_inference` using the ROI/overlap from
   `models/<name>/configs/train.yaml`.
5. Argmaxes the 4-class softmax output into an integer mask
   (0=background, 1=NCR, 2=ED, 3=ET).

## SynthSeg

[SynthSeg](https://github.com/BBillot/SynthSeg) segments whole-brain anatomy
(33 FreeSurfer `aseg` structures, optionally 68 Desikan-Killiany cortical
parcels) from a **single** scan. It is orthogonal to the tumour models, so it
gets its own panel, its own overlay layer, and its own save button.

### Why it runs in a separate process

SynthSeg needs Python 3.8 + `tensorflow==2.2.0` + `Keras==2.3.1` +
`numpy==1.23.5`. The viewer needs Python ≥3.10 + torch + nnunetv2 + numpy 2.x.
These cannot share an interpreter — TF 2.2 wheels stop at Python 3.8, and
PySide6/nnunetv2 require ≥3.9. So `core/backends/synthseg_backend.py` invokes
SynthSeg's own CLI in its own conda env and communicates over files, which is
exactly what that CLI is designed for. Nothing is imported in-process.

### Setup

```bash
# 1. The code and weights, under models/synthseg/ (gitignored — copy manually)
#    From a SynthSeg checkout, you need:
#      SynthSeg/  ext/  scripts/  models/*.h5  data/labels_classes_priors/
#    The five .h5 files are ~400 MB total and are gitignored upstream too,
#    so they will not arrive via `git clone` — copy models/ explicitly.

# 2. Its interpreter
conda create -n synthseg_38 python=3.8 -y
conda activate synthseg_38
pip install -r models/synthseg/requirements_python3.8.txt
```

**`models/` is gitignored, so neither the code nor the weights travel with
`git clone` or `git pull`.** Moving this branch to another machine means
copying `models/synthseg/` across by hand (rsync/scp) and creating the
`synthseg_38` env there — the most common reason Run SynthSeg is greyed out on
a fresh machine.

If the app cannot find either piece, the **Run SynthSeg** button stays
disabled and hovering it names the missing piece. Availability is re-checked
whenever a case is loaded, so installing the env or the weights takes effect
without restarting the app.

To diagnose from the shell, run `python check_synthseg.py` from the repo root;
it prints every path it looks for and whether it was found.

### Environment overrides

| Variable | Default | Purpose |
|---|---|---|
| `SYNTHSEG_PYTHON` | auto-discovered | Interpreter to run SynthSeg with. Searched for as a sibling of the active conda env, then under `CONDA_EXE`'s base, then the usual conda install locations. Set this only if it lives somewhere unusual. |
| `SYNTHSEG_ENV_NAME` | `synthseg_38` | Name of the conda env to look for. |
| `SYNTHSEG_HOME` | `models/synthseg` | Where the SynthSeg code and weights live. |
| `SYNTHSEG_THREADS` | 1 under 32 GB RAM, else `min(8, cores/2)` | TensorFlow intra/inter-op threads. **See the memory note below before raising this.** |
| `SYNTHSEG_CROP` | auto | Space-separated per-axis patch size, e.g. `160 192 160`. Overrides the automatic crop. |
| `SYNTHSEG_GPU` | auto-detected | `1` forces the GPU on, `0` forces it off. See the GPU note below. |
| `SYNTHSEG_LUT` | unset | Path to a real `FreeSurferColorLUT.txt`, used for overlay colors if present. |

### GPU support: compute capability 7.5 or lower

TensorFlow 2.2 ships CUDA 10.1 kernels. Its `.so` files carry code for
`sm_35 ... sm_75` and nothing above, so it can only use cards at **compute
capability 7.5 or lower**:

| Works | Does not work |
|---|---|
| P100 (6.0), V100 (7.0), T4 (7.5), RTX 20xx (7.5) | A100 (8.0), RTX 30xx (8.6), RTX 40xx (8.9), H100 (9.0) |

Ampere and newer need CUDA 11+, which stock TF 2.2 has no kernels for. The
backend detects this with `nvidia-smi` and falls back to CPU rather than
failing, and the status bar says which mode it picked and why — that fallback
would otherwise be silent, turning a seconds-long run into a minutes-long one
with no explanation.

Check the target machine with:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

If the card is 8.0 or newer, the options are: run on CPU (fine — with enough
RAM the thread default scales up and a BraTS volume takes a couple of
minutes), or port SynthSeg to a newer TensorFlow. The latter is real work, not
a config change: SynthSeg 2.0 imports standalone `Keras 2.3.1`
(`import keras.layers as KL`), which does not work against TF 2.4+ where Keras
moved inside `tf.keras`.

### Memory: why one thread

TensorFlow 2.2's multi-threaded `Conv3D` allocates a workspace buffer per
intra-op thread. At a full-brain patch those buffers overflow available memory
and the process aborts with `std::bad_alloc` — reproduced at both 2 and 4
threads on a 16 GB machine. At 1 thread the run peaks around 3.9 GB and takes
about a minute for a BraTS volume.

The default therefore scales with the machine rather than being pinned to the
smallest box it ran on: 1 thread under 32 GB of RAM, otherwise
`min(8, cores/2)`. `SYNTHSEG_THREADS` overrides it either way. If a larger
machine still aborts with `std::bad_alloc`, lower it.

The backend also shrinks the analysed patch per-axis to whatever actually
contains the brain (`auto_crop`), instead of SynthSeg's fixed 192³. SynthSeg
crops around the *image* centre after reorienting to RAS+, so the patch is
sized from the brain's furthest voxel from that centre, then rounded up to a
multiple of 32 by SynthSeg itself. For BraTS this cuts the analysed volume by
roughly a third with no clipping.

### Options

| Option | Effect |
|---|---|
| **Modality** | Which loaded volume to segment. Defaults to `T1`, which is what SynthSeg is trained and validated on; it is contrast-agnostic by design, so other modalities work but are less validated. |
| **Fast** | Skips topology postprocessing. Roughly twice as fast, marginally less accurate. |
| **Robust** | Uses the robust model — better on low-quality clinical scans, slower. Implies Fast (SynthSeg forces it), so the Fast box locks on. |
| **Parcellation** | Also parcellates the cortex into 68 Desikan-Killiany regions. Cortex labels 3/42 are *replaced* by labels 1001–1035 / 2001–2035, and the volumes CSV gains 68 columns. |

### Saved outputs

**Save SynthSeg** asks for a directory and writes three files. The modality is
in the name because SynthSeg can be run on any loaded modality, and a `_parc`
infix is added for parcellated runs, because that is a different label space —
neither should silently overwrite the other:

```
<case>_synthseg_<modality>[_parc].nii.gz          # int16, FreeSurfer labels, RAS+ canonical
<case>_synthseg_<modality>[_parc]_volumes.csv     # header + 1 row, per-structure volumes in mm3
<case>_synthseg_<modality>[_parc]_qc.csv          # header + 1 row, per-region QC scores in [0,1]
```

e.g. `BraTS-GoAT-00000_synthseg_t1n.nii.gz`. The `.nii.gz` is written with the
affine of the modality SynthSeg ran on, taken from its canonical form so it
matches the mask the viewer displays. The existing **Save Segmentation**
button is untouched and still writes `<case>_seg.nii.gz` for the tumour model.

### Colors

SynthSeg ships no color table — its `data/labels table.txt` says the color of
each structure is left to the viewer. `ui/synthseg_lut.py` therefore vendors
the canonical FreeSurfer `aseg` colors, so the overlay matches what freeview
and ITK-SNAP show. Cortical parcels (only present with **Parcellation**) get a
generated palette instead of the canonical one. Point `SYNTHSEG_LUT` at a real
`FreeSurferColorLUT.txt` and every label, parcels included, uses its canonical
color.

## Checkpoints

Drop a trained checkpoint (`.pth`, either a raw `state_dict` or a dict with a
`state_dict` key — both save formats used by the training scripts are
supported) into:

```
models/swin_unetr/checkpoints/*.pth
models/mavin-hpc/checkpoints/*.pth
```

The first file found (alphabetical) is loaded automatically — no code or
config changes needed. If the directory doesn't exist or is empty, the
viewer just runs with random weights and says so in the status bar.

---

## Troubleshooting

- **"Missing modalities in folder: ..."** — the selected folder doesn't have
  all 4 expected files, or their names don't contain the `t1n`/`t1c`/`t2w`/`t2f`
  substrings. Rename or pick a different folder.
- **MambaVision error about mamba_ssm / causal_conv1d** — you're on a machine
  without a CUDA GPU, or the kernels aren't installed. See
  [Optional: enabling MambaVision](#optional-enabling-mambavision). There is
  no CPU fallback; this isn't a bug in the viewer.
- **"nnUNet results folder not found at ..."** — nnUNet needs a trained
  results folder, not a single checkpoint. See
  [Model backends](#model-backends).
- **"SynthSeg interpreter not found" / "SynthSeg not found" / "SynthSeg
  weights missing"** — one of the two setup pieces is missing. See
  [SynthSeg → Setup](#setup-1), or set `SYNTHSEG_PYTHON` / `SYNTHSEG_HOME`.
- **"SynthSeg produced no segmentation"** followed by `std::bad_alloc` — it
  ran out of memory. Make sure `SYNTHSEG_THREADS` is not set above 1, and see
  [Memory: why one thread](#memory-why-one-thread).
- **Inference is slow** — on CPU, a single sliding-window pass at the
  configured ROI (128³ by default) can take well over a minute depending on
  the machine; this is expected without a GPU.
