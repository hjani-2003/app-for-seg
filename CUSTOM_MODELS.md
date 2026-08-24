# Using this viewer with your own models

This app is a thin viewer wrapped around a small backend contract. Nothing
about it is specific to the three models that ship with it — if your model
takes a stack of MRI volumes and returns an integer label map, you can plug it
in without touching the UI.

This guide covers the four ways to do that, cheapest first:

1. [Drop in a checkpoint](#1-drop-in-a-checkpoint) for an architecture already wired up
2. [Point at an nnU-Net results folder](#2-point-at-an-nnu-net-results-folder)
3. [Add a new in-process backend](#3-add-a-new-in-process-backend)
4. [Add an out-of-process backend](#4-add-an-out-of-process-backend) when the dependencies conflict

Before any of them, read [the data contract](#the-data-contract) — most
integration bugs are a mismatch there, not a code error, and they fail
silently by producing a plausible-looking but wrong overlay.

---

## The data contract

### Input folder layout

`Load BraTS Folder` expects **one folder per case, containing exactly four
NIfTI files**, matched by substring in the filename
(`core/data_loader.py:load_brats_folder`):

| Filename contains | Becomes modality |
|---|---|
| `t1n` | `T1` |
| `t1c` | `T1Gd` |
| `t2w` | `T2` |
| `t2f` | `FLAIR` |

```
BraTS-GoAT-00000/
├── BraTS-GoAT-00000-t1n.nii.gz
├── BraTS-GoAT-00000-t1c.nii.gz
├── BraTS-GoAT-00000-t2w.nii.gz
└── BraTS-GoAT-00000-t2f.nii.gz
```

If any of the four is missing the status bar names which, and nothing else
changes. To use different names, edit `MODALITY_FILE_KEYS` in
`core/constants.py`.

### Orientation

Every volume is reoriented to **RAS+ canonical** on load
(`nib.as_closest_canonical`), so array axes 0/1/2 always mean
sagittal/coronal/axial regardless of how the file was stored. Anything your
backend returns must be in that same space.

Two ways to satisfy this:

- Take the preloaded arrays the viewer hands you — already canonical, nothing
  to do.
- Read the original files yourself (as nnU-Net and SynthSeg do, because they
  do their own preprocessing) and canonicalise the **output** before returning
  it. Both backends do exactly this:
  ```python
  seg = nib.as_closest_canonical(nib.load(output_path))
  mask = np.asarray(seg.dataobj).astype(np.uint8)
  ```

Get this wrong and the overlay is mirrored left-right — which looks entirely
plausible on a brain. Check it on an axial slice through the ventricles.

### Channel order

The bundled models were trained on `[T1Gd, T1, FLAIR, T2]` — **not** the order
the modalities are listed in the UI. That is `MODEL_INPUT_CHANNELS` in
`core/constants.py`, and it is the single easiest thing to get wrong: a model
fed a permuted stack still returns a confident, wrong segmentation.

Set it to whatever *your* model was trained on.

### Normalisation

`core/preprocessing.py:normalize_for_model` applies a per-channel z-score over
nonzero voxels, matching MONAI's
`NormalizeIntensityd(nonzero=True, channel_wise=True)`. If your model was
trained with different preprocessing, do it inside your backend and don't call
this.

### Output labels

The viewer expects an integer array, same shape as the input volumes:

| Value | Meaning |
|---|---|
| 0 | background |
| 1 | NCR — necrotic core |
| 2 | ED — edema |
| 3 | ET — enhancing tumour |

For a different number of classes or different meanings, update
`CLASS_LABELS` and `SEGMENTATION_COLORS` in `ui/style.py` — they are keyed by
label value and drive both the overlay and the legend. Any label you do not
give a colour renders as black.

`dtype` matters when saving: `save_segmentation` writes `uint8`. If your
labels exceed 255, use `save_label_map` instead, which writes `int16`.

---

## 1. Drop in a checkpoint

For **SwinUNETR** and **MaViN**, no code changes at all:

```
models/swin_unetr/checkpoints/your_weights.pth
models/mavin-hpc/checkpoints/your_weights.pth
```

`core/backends/checkpoints.py:find_checkpoint` takes the **first `.pth` in
alphabetical order**. Both save formats work — a raw `state_dict`, or a dict
with a `state_dict` key.

The architecture is built from `models/<name>/configs/model.yaml`, so the
config must match the checkpoint or `load_state_dict` will fail:

```yaml
model:
  in_channels: 4
  out_channels: 4      # background + 3 tumour classes
  feature_size: 48
  use_v2: True
```

Sliding-window parameters come from `configs/train.yaml`:

```yaml
data:
  roi: [128, 128, 128]
  sw_batch_size: 4
  infer_overlap: 0.7
```

If no checkpoint is found the viewer still runs the architecture with **random
weights** and says so in the status bar — useful for testing the plumbing, and
worth watching for, because the output looks like a segmentation.

---

## 2. Point at an nnU-Net results folder

nnU-Net needs a full trained results folder, not a single weights file,
because it reads its preprocessing from `plans.json`. Put it at the path in
`core/backends/nnunet_backend.py:MODEL_DIR`:

```
models/nnunet/results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/
├── dataset.json
├── plans.json
└── fold_0/
    └── checkpoint_best.pth
```

Adjust `USE_FOLDS`, `CHECKPOINT_NAME` and `STEP_SIZE` in that module for a
different trainer, fold or tile step. `dataset.json` must list channels in the
order your model expects; the bundled one uses
`{"0": "T1c", "1": "T1n", "2": "T2f", "3": "T2w"}`, matching
`MODEL_INPUT_CHANNELS`.

Note the bundled `dataset.json` calls class 2 `SNFH` where the UI calls it
`ED`. Same class, different vocabulary.

---

## 3. Add a new in-process backend

Use this when your model is importable in the app's own environment.

### Step 1 — write the backend module

Create `core/backends/mymodel_backend.py` with two functions:

```python
from pathlib import Path

import torch

from core.backends.checkpoints import find_checkpoint
from core.backends.errors import ModelUnavailableError

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "mymodel"


def get_roi_and_overlap():
    """(roi, overlap, sw_batch_size) for MONAI's sliding_window_inference."""
    return [128, 128, 128], 0.5, 1


def build_model(device):
    """Return (model_in_eval_mode, checkpoint_path_or_None)."""
    model = MyNet(in_channels=4, out_channels=4).to(device)

    checkpoint = find_checkpoint(MODEL_DIR)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state.get("state_dict", state))

    model.eval()
    return model, checkpoint
```

Raise `ModelUnavailableError` for anything the user could fix — a missing
dependency, absent weights, a GPU-only kernel on a CPU box. The UI shows its
message in the status bar instead of a traceback. `mavin_backend.py` does this
for the Mamba CUDA kernels.

### Step 2 — register it

```python
# core/inference.py
_BACKENDS = {
    "SwinUNETR": swin_unetr_backend,
    "MaViN": mavin_backend,
    "MyModel": mymodel_backend,        # add
}
```

```python
# core/constants.py
MODEL_ARCHITECTURES = ["SwinUNETR", "MaViN", "nnUnet", "MyModel"]
```

That is all. `InferenceWorker` runs it on a background thread, feeds it a
normalised `[T1Gd, T1, FLAIR, T2]` stack through MONAI's
`sliding_window_inference`, argmaxes the result, and the UI picks it up.

### If your model needs more than `(device)`

`build_model(device)` is the contract. `MaViN` needs the ROI at construction
time, and `core/inference.py` special-cases it by name to pass
`build_model(device, roi)`. It is a wart: a third signature means another
branch there. Prefer reading what you need from your own config inside
`build_model`.

### If your model doesn't fit sliding-window inference

Take the nnU-Net route — `core/inference.py` special-cases it before the
`_BACKENDS` lookup:

```python
if self.model_name == "nnUnet":
    mask, checkpoint = nnunet_backend.predict(self.paths, device)
```

Your `predict()` receives `self.paths` (`{modality: original file path}`) and
returns `(mask, checkpoint)`. Use this when your model does its own
preprocessing from the files on disk.

---

## 4. Add an out-of-process backend

Use this when your model's dependencies **cannot** coexist with the app's.
SynthSeg is the worked example, and the reason the pattern exists.

### When you need it

SynthSeg requires Python 3.8 + `tensorflow==2.2.0` + `Keras==2.3.1` +
`numpy==1.23.5`. The app requires Python 3.11 + torch + nnunetv2 + numpy 2.x.
TensorFlow 2.2 wheels stop at Python 3.8; PySide6 and nnunetv2 need ≥ 3.9.
There is no interpreter that satisfies both, and no amount of pinning fixes
it.

The answer is not to compromise either environment. Run the model in **its
own environment, as a subprocess**, and communicate over files.

### The shape of it

`core/backends/synthseg_backend.py` is about 300 lines and worth reading in
full before writing your own. The parts that matter:

**Locate the interpreter — don't hardcode it.**

```python
SYNTHSEG_PYTHON = Path(os.environ.get("SYNTHSEG_PYTHON") or _discover())
```

An absolute path baked into the source breaks the moment the repo moves to
another machine or another user's home. Search for the env as a sibling of the
active conda env (`CONDA_PREFIX`), then under `CONDA_EXE`'s base, then the
usual install locations — and let an environment variable override.

**Check availability before running, not during.**

```python
def check_available():
    """Return None if it can run, else a human-readable reason."""
```

The UI calls this to enable or disable the Run button and to explain why, so a
missing piece is reported in a second rather than after a multi-minute run.

**Decide success by the output file, not the exit code.**

SynthSeg's CLI exits 1 if *any* image in a batch failed, even when the one you
asked for succeeded. So:

```python
if not seg_path.is_file():
    raise RuntimeError(f"produced no segmentation:\n{log_tail}")
```

Keep the last ~25 lines of stdout for that message. When the subprocess dies
of `std::bad_alloc`, that tail is the only diagnosis you get.

**Stream stdout for progress.** A subprocess can't emit Qt signals, so scrape
its output and forward the useful lines to a `progress` signal.

**Bring the result into the viewer's space.** Canonicalise, and guard the
shape — SynthSeg segments on a 1 mm isotropic grid, which is a no-op for BraTS
but not for arbitrary data:

```python
if mask.shape != tuple(reference_shape):
    mask = resample_from_to(seg_img, ref_img, order=0)   # nearest neighbour
```

`order=0` is not optional. Any interpolation invents label values that mean
nothing.

### Its worker

`core/synthseg_inference.py` is a separate `QThread` from `InferenceWorker`
because it takes a single file rather than a four-channel stack, needs no
torch device, and runs long enough that progress is worth surfacing:

```python
class SynthSegWorker(QThread):
    finished = Signal(np.ndarray, object)
    failed = Signal(str)
    progress = Signal(str)
```

One thing to copy: the worker records **which case it was started for**, and
the UI discards results whose case no longer matches. A run takes minutes —
long enough for the user to load a different case meanwhile, and applying one
patient's mask to another's images is worse than no result at all.

### Setting up the second environment

Document it, because it will not come from `requirements.txt`:

```bash
conda create -n synthseg_38 python=3.8 -y
conda activate synthseg_38
pip install -r models/synthseg/requirements_python3.8.txt
```

Note also that `models/` is **gitignored** — weights do not travel with
`git clone`. Copy them explicitly and say so in your docs; it is the most
common reason a fresh checkout has a greyed-out button.

---

## Testing your backend

Test the backend on its own before touching the UI. A GUI is a bad place to
discover that your channel order is wrong.

```bash
python -c "
from core.data_loader import load_brats_folder
from core.backends import mymodel_backend as b
import torch, numpy as np

volumes, spacing, paths = load_brats_folder('/path/to/BraTS-GoAT-00000')
model, ckpt = b.build_model(torch.device('cpu'))
print('checkpoint:', ckpt)
print('roi/overlap:', b.get_roi_and_overlap())
"
```

Then check the output against what the viewer expects:

| Check | Why |
|---|---|
| `mask.shape == volumes['T1'].shape` | a mismatch means orientation or resampling is wrong |
| `np.unique(mask)` is a subset of `{0,1,2,3}` | stray labels render black |
| ventricles are on the correct side | catches a left-right flip, which otherwise looks plausible |
| `mask.dtype` fits the save path | `uint8` for `save_segmentation`, `int16` for `save_label_map` |

For the round trip, save from the app and confirm the geometry survived:

```bash
python -c "
import nibabel as nib, numpy as np
a = nib.load('out/CASE_seg.nii.gz')
b = nib.as_closest_canonical(nib.load('CASE/CASE-t1n.nii.gz'))
print(a.shape, a.get_data_dtype(), np.allclose(a.affine, b.affine))
"
```

Finally, run it in the app on a case whose answer you know, and scrub through
all three planes. Overlay bugs that are invisible on one slice are obvious
across a volume.
