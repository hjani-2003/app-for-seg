# MRI Tumour Segmentation Viewer

A desktop viewer for brain MRI segmentation. Load a four-modality BraTS case,
run a tumour segmentation model over it, run whole-brain anatomical
segmentation alongside, and inspect both as overlays on the same slices.

Built with PySide6 and pyqtgraph.

---

## Why this exists

Training a segmentation model produces a checkpoint. Finding out whether it is
any *good* means looking at its output on real cases, slice by slice, against
the image it came from — and the usual options are awkward. General-purpose
viewers such as ITK-SNAP or freeview don't run your model; running inference
in a notebook and dumping PNGs is slow to iterate on and easy to get subtly
wrong.

This app closes that loop. It runs your model and shows the result in the same
window, so the question "does this checkpoint work?" takes seconds instead of
a round trip through the filesystem.

Three things follow from that purpose:

- **Checkpoints are discovered, not configured.** Drop a `.pth` into the right
  folder and it is picked up. Comparing two checkpoints shouldn't require
  editing code.
- **Backends are pluggable.** Adding an architecture is a module plus two
  lines in a registry. See [CUSTOM_MODELS.md](CUSTOM_MODELS.md).
- **Tumour and anatomy are separate layers.** A tumour mask says where the
  lesion is; SynthSeg says which structures it abuts. They compose.

---

## Documentation

| | |
|---|---|
| **[FEATURES.md](FEATURES.md)** | What the viewer does, and why the non-obvious parts are the way they are |
| **[CUSTOM_MODELS.md](CUSTOM_MODELS.md)** | Using it with your own models — the data contract, and four ways to plug one in |

---

## Quick start

```bash
conda create -n unet_venv python=3.11
conda activate unet_venv

cd app-for-seg
pip install -r requirements.txt

# torch is per-machine (CPU vs a specific CUDA build):
# https://pytorch.org/get-started/locally/

python main.py
```

That opens the viewer. No arguments, no config to edit first.

Then: **Load BraTS Folder** → pick a case folder → **Run Inference**.

### Optional extras

**MambaVision (MaViN)** needs the Mamba selective-scan CUDA kernels, which
require a GPU to build — there is no CPU fallback. Skip it and the viewer
still runs; MaViN just reports itself unavailable.

```bash
pip install causal-conv1d>=1.1.0 mamba-ssm
python -c "import mamba_ssm, causal_conv1d; print('mamba kernels OK')"
```

**SynthSeg** needs a second conda environment, because TensorFlow 2.2 and this
app's Python 3.11 cannot coexist:

```bash
conda create -n synthseg_38 python=3.8 -y
conda activate synthseg_38
pip install -r models/synthseg/requirements_python3.8.txt
conda activate unet_venv
```

It is found automatically as a sibling of the active env. Check it with
`python check_synthseg.py`, which prints every path it looks for.

---

## Input format

One folder per case, four NIfTI files, matched by substring:

| Filename contains | Modality |
|---|---|
| `t1n` | T1 |
| `t1c` | T1Gd |
| `t2w` | T2 |
| `t2f` | FLAIR |

Missing files are named individually in the status bar. To use a different
naming scheme, edit `MODALITY_FILE_KEYS` in `core/constants.py`.

---

## Repository layout

```
app-for-seg/
├── main.py                     entry point
├── check_synthseg.py           diagnoses a SynthSeg setup
├── core/
│   ├── constants.py            modalities, architectures, channel order
│   ├── data_loader.py          case loading, RAS+ canonicalisation, saving
│   ├── preprocessing.py        per-channel z-score over nonzero voxels
│   ├── inference.py            InferenceWorker — dispatches to a backend
│   ├── synthseg_inference.py   SynthSegWorker — subprocess, streams progress
│   └── backends/               one module per model
├── ui/
│   ├── main_window.py          MRIViewer — wires the panels together
│   ├── panels.py               Input, Model, SynthSeg, View panels
│   ├── legend_dock.py          the colour legend
│   ├── mask_render.py          overlay compositing
│   ├── style.py                dark theme, tumour class colours
│   └── synthseg_lut.py         FreeSurfer label names and colours
└── models/                     weights and training repos (gitignored)
```

**`models/` is gitignored.** Weights do not travel with `git clone` — copy
them across explicitly. This is the most common reason a fresh checkout has a
greyed-out button.

---

## Requirements

Python 3.11 (developed against 3.11.15). A GPU is optional for the viewer and
for SwinUNETR, required for MaViN, and unusable by SynthSeg on cards newer
than compute capability 7.5 (see [FEATURES.md](FEATURES.md)).

Developed against: PySide6 6.11.1, pyqtgraph 0.14.0, nibabel 5.4.2,
numpy 2.4.4, torch 2.12.0, monai 1.5.2, nnunetv2 2.7.0, scipy 1.17.1.

---

## Troubleshooting

- **"Missing modalities in folder: …"** — the folder doesn't have all four
  files, or their names lack the `t1n`/`t1c`/`t2w`/`t2f` substrings.
- **MambaVision error about `mamba_ssm` / `causal_conv1d`** — no CUDA GPU, or
  the kernels aren't installed. There is no CPU fallback; not a bug.
- **"nnUNet results folder not found at …"** — nnU-Net needs a full trained
  results folder, not a single checkpoint. See
  [CUSTOM_MODELS.md](CUSTOM_MODELS.md).
- **Run SynthSeg is greyed out** — hover it; the reason is under the button
  too. Then run `python check_synthseg.py`. Usually either `models/synthseg/`
  wasn't copied across or the `synthseg_38` env doesn't exist on that machine.
- **"SynthSeg produced no segmentation"** with `std::bad_alloc` — out of
  memory. Lower `SYNTHSEG_THREADS`.
- **Inference is slow on CPU** — a sliding-window pass at 128³ takes well over
  a minute without a GPU. Expected.
- **The overlay looks mirrored** — a backend returned a mask that wasn't
  reoriented to RAS+ canonical. See
  [CUSTOM_MODELS.md → Orientation](CUSTOM_MODELS.md#orientation).

---

Made with love by me and Claude, for pretty hoomans ❤️
