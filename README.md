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
- **A mask is the start, not the answer.** RANO sizes the lesion the way trials
  read it; PyRadiomics describes it as features a model can consume. Both run
  on the mask the viewer just produced, in the same window.

---

## Documentation

| | |
|---|---|
| **[FEATURES.md](FEATURES.md)** | What the viewer does, and why the non-obvious parts are the way they are |
| **[CUSTOM_MODELS.md](CUSTOM_MODELS.md)** | Using it with your own models — the data contract, and four ways to plug one in |
| **[NOTES.md](NOTES.md)** | The design log for the RANO measurement feature, phase by phase |

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

**PyRadiomics** needs a third environment, for a different reason: its last
release ships wheels for CPython 3.7-3.9 only, built against the numpy 1.x C
ABI. On Python 3.11 with numpy 2 there is nothing to install but a source build
that does not survive the numpy major version; on Python 3.9 it is a prebuilt
wheel and nothing is compiled at all.

```bash
conda create -n radiomics_39 python=3.9 -y
conda run -n radiomics_39 pip install "numpy<2" pyradiomics
```

**The `numpy<2` pin is not optional.** numpy 2 supports Python 3.9 too, so a
bare `pip install pyradiomics` produces an env that looks right and fails at
import. Found automatically as a sibling of the active env, like SynthSeg;
check it with `python check_radiomics.py`.

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
├── check_radiomics.py          diagnoses a PyRadiomics setup
├── rano_measure/               RANO bidimensional measurement — pure logic,
│                               no Qt: regions, lesions, geometry, burden
├── tests/                      pytest suite over the pure-logic packages
├── core/
│   ├── constants.py            modalities, architectures, channel order, ROIs
│   ├── data_loader.py          case loading, RAS+ canonicalisation, saving
│   ├── preprocessing.py        per-channel z-score over nonzero voxels
│   ├── inference.py            InferenceWorker — dispatches to a backend
│   ├── synthseg_inference.py   SynthSegWorker — subprocess, streams progress
│   ├── radiomics_extraction.py RadiomicsWorker — subprocess, streams progress
│   ├── radiomics_params/       the Fast / Standard / Extended preset YAMLs
│   └── backends/               one module per model
├── ui/
│   ├── main_window.py          MRIViewer — wires the panels together
│   ├── panels.py               Input, Model, SynthSeg, Radiomics, View panels
│   ├── legend_dock.py          the colour legend
│   ├── radiomics_dock.py       the feature table
│   ├── rano_dock.py            the RANO table and its on-slice callipers
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
numpy 2.4.4, torch 2.12.0, monai 1.5.2, nnunetv2 2.7.0, scipy 1.17.1,
scikit-image 0.25.2.

Run the tests with `pytest`. They cover the pure-logic packages only —
`rano_measure/` and the radiomics planning — never the Qt layer.

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
- **Extract Features is greyed out** — hover it. Either no tumour mask has been
  produced yet (features are extracted over one), or the `radiomics_39` env is
  missing; `python check_radiomics.py` says which.
- **"PyRadiomics produced no features"** mentioning numpy — the child env has
  numpy 2. Recreate it with the `"numpy<2"` pin above.
- **A region is listed as skipped** — that region has no voxels in this case's
  mask, or fewer than `minimumROISize`. Not a failure; the rest of the table is
  still valid — a non-enhancing tumour genuinely has no ET region.
- **The RANO dock is empty after inference** — no lesion reached RANO's 10mm
  minimum on both diameters. Rows appear as unmeasurable rather than vanishing,
  so an empty table means no connected component survived at all.
- **RANO's add/edit controls are greyed out** — they are axial-only. RANO is
  defined on axial slices, so the dock disables them in the other planes rather
  than measuring something the criterion doesn't describe.
- **Inference is slow on CPU** — a sliding-window pass at 128³ takes well over
  a minute without a GPU. Expected.
- **The overlay looks mirrored** — a backend returned a mask that wasn't
  reoriented to RAS+ canonical. See
  [CUSTOM_MODELS.md → Orientation](CUSTOM_MODELS.md#orientation).

---

## Contributing

Contributions are very welcome — issues, fixes, new backends, whole new
features. This started as a way to close one loop, and it gets better the more
people bend it toward what they actually need.

A few things worth knowing before you open a pull request, because they are
conventions rather than rules you could guess:

- **Qt stays in `ui/`, logic stays in `core/` and the pure packages.**
  `rano_measure/` imports no Qt at all, and that is enforced deliberately —
  see the layering note in [NOTES.md](NOTES.md) for the bug that prompted it.
  Anything you can test without a running application belongs on that side.
- **Adding a model is a module and two lines in a registry.**
  [CUSTOM_MODELS.md](CUSTOM_MODELS.md) covers the data contract and four ways
  to plug one in, including the out-of-process pattern for dependencies that
  cannot share an interpreter with the app.
- **Tests cover pure logic, never the Qt layer.** `pytest` should stay green;
  small synthetic volumes with hand-placed voxels are the house style.
- **A failure a user could fix should say so on screen.** Greyed-out buttons
  explain themselves, and the `check_*.py` scripts exist so a broken setup
  produces a diagnosis rather than a traceback. New setup requirements deserve
  the same treatment.

Comments here explain *why*, not *what* — if a decision was non-obvious or you
picked one option over another for a reason, that reason is worth a line.

---

Made with love by me and Claude, for pretty hoomans ❤️
