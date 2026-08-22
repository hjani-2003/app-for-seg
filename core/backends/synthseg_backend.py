"""SynthSeg whole-brain anatomical segmentation, run as a subprocess.

SynthSeg needs Python 3.8 + TensorFlow 2.2 + Keras 2.3, which cannot coexist
with this app's Python 3.11 + torch + nnUNet environment (TF 2.2 wheels stop at
3.8; PySide6 and nnunetv2 need >= 3.9). So unlike the other backends it is not
imported — it is invoked through its own interpreter and communicates over
files, which is exactly what its CLI is built for.
"""
import os
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from core.backends.errors import ModelUnavailableError

# `or` rather than a get() default throughout: an env var set but left empty
# would otherwise resolve to the current directory.
SYNTHSEG_HOME = Path(
    os.environ.get("SYNTHSEG_HOME")
    or Path(__file__).resolve().parents[2] / "models" / "synthseg"
)
SYNTHSEG_SCRIPT = SYNTHSEG_HOME / "scripts" / "commands" / "SynthSeg_predict.py"
SYNTHSEG_PYTHON = Path(
    os.environ.get("SYNTHSEG_PYTHON")
    or "/home/harman/miniconda3/envs/synthseg_38/bin/python"
)

# Weights the CLI picks per flag combination. Checked up front so a missing
# download is reported before a multi-minute run rather than during it — but
# only the ones a given run actually needs, so a partial download still allows
# the runs it can support. The base and QC models are always used.
_BASE_WEIGHTS = ("synthseg_2.0.h5", "synthseg_qc_2.0.h5")
_OPTION_WEIGHTS = {
    "robust": "synthseg_robust_2.0.h5",
    "parc": "synthseg_parc_2.0.h5",
}

# TensorFlow 2.2 ships CUDA 10.1 kernels compiled for compute capability 7.5
# and below (checked with `strings` over its .so files: sm_35 ... sm_75). An
# Ampere or newer card — A100 sm_80, RTX 30xx sm_86, RTX 40xx sm_89, H100
# sm_90 — needs CUDA 11+, so TF 2.2 cannot use it at all and must fall back to
# CPU. SYNTHSEG_GPU=1 forces the GPU on, =0 forces it off.
MAX_COMPUTE_CAPABILITY = 7.5

# Multi-threaded Conv3D in TF 2.2 allocates a workspace buffer per intra-op
# thread. On a 16 GB machine those overflow memory at a full-brain crop and
# abort the process with std::bad_alloc (reproduced at 2 and 4 threads; 1
# thread peaks at ~3.9 GB and takes about a minute). Machines with real memory
# headroom can afford more, so the default scales rather than being pinned to
# the smallest box this ran on. SYNTHSEG_THREADS overrides.
_THREAD_HEADROOM_GB = 32
_MAX_AUTO_THREADS = 8

# SynthSeg analyses a centred patch, 192 on every axis by default. Patch size
# drives peak memory, so it is instead shrunk per-axis to whatever actually
# contains the brain — about a third less volume for BraTS, at no cost in
# coverage. SYNTHSEG_CROP overrides it; this stays the upper bound.
MAX_CROP = 192
# Voxels of slack kept around the brain so the crop never shaves its boundary.
_CROP_MARGIN = 8

_LOG_TAIL = 25


def _total_ram_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return None


def detect_gpu():
    """Return (usable, description) for the local GPU, without importing TF.

    TF 2.2 would take tens of seconds to import just to answer this, and it is
    needed before every run, so nvidia-smi is asked instead.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False, "no NVIDIA GPU detected"

    if out.returncode != 0 or not out.stdout.strip():
        return False, "no NVIDIA GPU detected"

    name, _, cap = out.stdout.strip().splitlines()[0].partition(",")
    name, cap = name.strip(), cap.strip()

    # compute_cap needs a reasonably recent driver; without it the card cannot
    # be vetted, so the safe answer is CPU.
    try:
        capability = float(cap)
    except ValueError:
        return False, f"{name} (driver did not report compute capability)"

    if capability > MAX_COMPUTE_CAPABILITY:
        return False, (
            f"{name} is compute capability {capability}, but TensorFlow 2.2 only "
            f"has kernels up to {MAX_COMPUTE_CAPABILITY} — running on CPU"
        )
    return True, f"{name} (compute capability {capability})"


def default_threads():
    env = os.environ.get("SYNTHSEG_THREADS")
    if env:
        return int(env)
    ram_gb = _total_ram_gb()
    if ram_gb is None or ram_gb < _THREAD_HEADROOM_GB:
        return 1
    return max(1, min(_MAX_AUTO_THREADS, (os.cpu_count() or 2) // 2))


def use_gpu():
    env = os.environ.get("SYNTHSEG_GPU")
    if env:
        return env == "1"
    return detect_gpu()[0]


@dataclass
class SynthSegResult:
    """One SynthSeg run, with its CSVs held as text so saving never re-runs it."""

    mask: np.ndarray
    volumes_csv: str
    qc_csv: str
    modality: str
    parc: bool
    info: str


def check_available(robust=False, parc=False):
    """Return None if SynthSeg can run, else a human-readable reason why not.

    Only the weights the requested options need are required, so a partial
    download still permits the runs it can support.
    """
    if not SYNTHSEG_PYTHON.is_file():
        return (
            f"SynthSeg interpreter not found at {SYNTHSEG_PYTHON}. Create the "
            "synthseg_38 env (see README) or set SYNTHSEG_PYTHON."
        )
    if not SYNTHSEG_SCRIPT.is_file():
        return (
            f"SynthSeg not found at {SYNTHSEG_HOME}. Copy the SynthSeg repo "
            "there (see README) or set SYNTHSEG_HOME."
        )
    needed = list(_BASE_WEIGHTS)
    if robust:
        needed.append(_OPTION_WEIGHTS["robust"])
    if parc:
        needed.append(_OPTION_WEIGHTS["parc"])

    missing = [
        name for name in needed if not (SYNTHSEG_HOME / "models" / name).is_file()
    ]
    if missing:
        return (
            f"SynthSeg weights missing from {SYNTHSEG_HOME / 'models'}: "
            f"{', '.join(missing)}"
        )
    return None


def auto_crop(image_path):
    """Smallest centred patch that still contains the whole brain, per axis.

    SynthSeg crops around the *image* centre (edit_volumes.crop_volume with
    mode='center'), after reorienting to RAS+, so the extent each axis needs is
    twice its furthest brain voxel from the centre. SynthSeg rounds each value
    up to a multiple of 32 itself, so raw sizes can be passed through.

    Returns None if the image cannot be read, letting SynthSeg use its default.
    """
    try:
        img = nib.as_closest_canonical(nib.load(image_path))
        data = np.asarray(img.dataobj)
    except Exception:
        return None

    nonzero = np.argwhere(data > 0)
    if nonzero.size == 0:
        return None

    low = nonzero.min(axis=0)
    high = nonzero.max(axis=0)
    centre = np.array(data.shape[:3]) // 2
    half = np.maximum(np.abs(high - centre), np.abs(centre - low))
    needed = 2 * (half + _CROP_MARGIN) + 1

    return [int(min(n, MAX_CROP)) for n in needed]


def runtime_summary():
    """One line describing how a run will execute, for the status bar.

    Worth showing because the GPU fallback is silent otherwise: a card newer
    than compute capability 7.5 simply gets skipped, and a run that the user
    expected to take seconds takes minutes instead.
    """
    if use_gpu():
        return f"SynthSeg: GPU — {detect_gpu()[1]}"
    return f"SynthSeg: CPU, {default_threads()} thread(s) — {detect_gpu()[1]}"


def _build_command(image_path, out_dir, fast, robust, parc, threads, crop):
    cmd = [
        str(SYNTHSEG_PYTHON),
        # The CLI derives its model/label directories from sys.argv[0], so it
        # has to be handed the real absolute path of the script.
        str(SYNTHSEG_SCRIPT),
        "--i", str(image_path),
        "--o", str(out_dir / "seg.nii.gz"),
        "--vol", str(out_dir / "volumes.csv"),
        "--qc", str(out_dir / "qc.csv"),
        "--threads", str(threads),
    ]
    if not use_gpu():
        cmd.append("--cpu")
    if fast:
        cmd.append("--fast")
    if robust:
        cmd.append("--robust")  # implies fast=True upstream
    if parc:
        cmd.append("--parc")
    if crop:
        cmd += ["--crop"] + [str(c) for c in crop]
    return cmd


def _run(cmd, on_progress):
    """Run the CLI, streaming stdout, and return its last few lines."""
    tail = deque(maxlen=_LOG_TAIL)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(SYNTHSEG_HOME),
    )
    for line in process.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail.append(line)
        if on_progress is not None:
            on_progress(line)
    process.wait()
    return "\n".join(tail)


def _load_mask(seg_path, reference_path, reference_shape):
    # Reorient to RAS+ canonical to match the app's other volumes (see
    # core/data_loader.py). int16, not uint8: aseg labels reach 60, and
    # parcellation labels reach 2035.
    seg_img = nib.as_closest_canonical(nib.load(seg_path))
    mask = np.asarray(seg_img.dataobj).astype(np.int16)

    if mask.shape == tuple(reference_shape):
        return mask

    # SynthSeg segments on a 1mm isotropic grid. BraTS is already 1mm iso so
    # this is a no-op there, but other data needs resampling back onto the
    # displayed grid before it can be overlaid.
    try:
        from nibabel.processing import resample_from_to

        ref_img = nib.as_closest_canonical(nib.load(reference_path))
        resampled = resample_from_to(seg_img, ref_img, order=0)
        mask = np.asarray(resampled.dataobj).astype(np.int16)
    except Exception as exc:
        raise RuntimeError(
            f"SynthSeg returned a {mask.shape} segmentation but the displayed "
            f"volume is {tuple(reference_shape)}, and resampling onto it "
            f"failed: {exc}"
        ) from exc

    if mask.shape != tuple(reference_shape):
        raise RuntimeError(
            f"SynthSeg segmentation is {mask.shape} after resampling but the "
            f"displayed volume is {tuple(reference_shape)}."
        )
    return mask


def predict(
    image_path,
    reference_shape,
    modality="T1",
    fast=False,
    robust=False,
    parc=False,
    threads=None,
    crop=None,
    on_progress=None,
):
    """Segment one scan with SynthSeg and return a SynthSegResult.

    image_path is the original on-disk NIfTI, not a preloaded array: SynthSeg
    reads and writes in the file's own space, and the result is brought into
    the app's RAS+ canonical space afterwards.
    """
    reason = check_available(robust=robust, parc=parc)
    if reason is not None:
        raise ModelUnavailableError(reason)

    if threads is None:
        threads = default_threads()

    if crop is None:
        env_crop = os.environ.get("SYNTHSEG_CROP")
        crop = (
            [int(c) for c in env_crop.split()]
            if env_crop
            else auto_crop(image_path)
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        cmd = _build_command(image_path, out_dir, fast, robust, parc, threads, crop)
        log_tail = _run(cmd, on_progress)

        seg_path = out_dir / "seg.nii.gz"
        # The CLI exits 1 when any image in a batch fails even if others
        # succeeded, so the output file — not the return code — decides.
        if not seg_path.is_file():
            raise RuntimeError(f"SynthSeg produced no segmentation:\n{log_tail}")

        mask = _load_mask(seg_path, image_path, reference_shape)
        volumes_csv = _read_text(out_dir / "volumes.csv")
        qc_csv = _read_text(out_dir / "qc.csv")

    variant = "SynthSeg-robust" if robust else "SynthSeg 2.0"
    details = [modality]
    if fast and not robust:
        details.append("fast")
    if parc:
        details.append("parcellation")
    info = f"{variant} ({', '.join(details)})"

    return SynthSegResult(
        mask=mask,
        volumes_csv=volumes_csv,
        qc_csv=qc_csv,
        modality=modality,
        parc=parc,
        info=info,
    )


def _read_text(path):
    return path.read_text() if path.is_file() else ""
