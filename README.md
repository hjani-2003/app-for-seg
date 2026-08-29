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
│   └── backends/
│       ├── swin_unetr_backend.py    # builds MONAI SwinUNETR from models/swin_unetr/configs
│       ├── mambavision_backend.py   # builds MambaVision-UNet from models/mavin-hpc/src (GPU only)
│       ├── nnunet_backend.py        # stub — see "Model backends" below
│       ├── checkpoints.py           # find_checkpoint(): looks in models/<name>/checkpoints/*.pth
│       └── errors.py                # ModelUnavailableError
├── ui/
│   ├── main_window.py               # MRIViewer — wires panels together, load/run/view logic
│   ├── panels.py                    # InputPanel, ModelPanel, ViewPanel (QGroupBox widgets)
│   ├── style.py                     # dark theme QSS + segmentation class colors
│   └── mask_render.py               # colorize_mask() / overlay_image_with_mask()
└── models/                          # training repos for each architecture (see each GUIDE.md)
    ├── swin_unetr/
    ├── mavin-hpc/                   # MambaVision
    └── nnUnet/
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

2. **Architecture** (Model panel) — pick `SwinUNETR`, `MambaVision`, or
   `nnU-Net`. Selecting `nnU-Net` disables **Run Inference** immediately and
   explains why in the status bar (see [Model backends](#model-backends)).

3. **Run Inference** — enabled once a valid case is loaded and the selected
   architecture is available. Runs on a background thread (the window stays
   responsive; the button disables and an indeterminate progress bar shows
   while it runs). When it finishes, the status bar reports either the
   checkpoint file used or that it ran with random weights.

4. **Modality** / **Display** (View panel) — `Modality` picks which of the 4
   loaded volumes to show; `Display` picks `Image`, `Mask`, or `Image + Mask`.
   The two mask modes only do anything once inference has produced a result.

5. **Slice slider** — scrubs through the volume along its first axis.

6. **Legend** — appears above the image whenever a mask view is showing a
   real segmentation. Colors are fixed per class:

   | Class | Meaning | Color |
   |---|---|---|
   | NCR | Necrotic core | blue |
   | ED | Edema | aqua/green |
   | ET | Enhancing tumor | yellow/orange |

---

## Model backends

The three dropdown options map to the three `models/` training repos:

| Architecture | Backing repo | Status |
|---|---|---|
| **SwinUNETR** | `models/swin_unetr` | Fully wired. Runs on CPU or GPU. |
| **MambaVision** | `models/mavin-hpc` | Fully wired, **GPU only** — the mamba_ssm CUDA kernels have no CPU fallback. On a machine without them, selecting it and running produces a clear status-bar error instead of a crash. |
| **nnU-Net** | `models/nnUnet` | Stubbed. nnU-Net needs a full trained *results folder* (`plans.json` + `dataset.json` + fold checkpoints from `nnUNetv2_train`), not just a weights file, so it can't be pointed at a single checkpoint the way the other two can. Wiring this up is future work — see `models/nnUnet/GUIDE.md` for its own training/predict pipeline in the meantime. |

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
- **nnU-Net's Run Inference is greyed out** — expected, see
  [Model backends](#model-backends). Use `models/nnUnet`'s own scripts to
  train/predict until it's wired into the viewer.
- **Inference is slow** — on CPU, a single sliding-window pass at the
  configured ROI (128³ by default) can take well over a minute depending on
  the machine; this is expected without a GPU.

---

Made with love by me and Claude, for pretty hoomans ❤️
