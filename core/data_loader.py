import os

import numpy as np
import nibabel as nib

from core.constants import MODALITY_FILE_KEYS, MODALITIES

NIFTI_SUFFIXES = (".nii", ".nii.gz")

# Names produced by this app's own save actions. A SynthSeg output is named
# after the modality it was run on, so without this it would be picked up as
# that modality on the next load.
DERIVED_MARKERS = (
    "synthseg", "_seg.nii", "_volumes.csv", "_qc.csv", "_radiomics",
)


def _is_case_input(fname):
    lower = fname.lower()
    if not lower.endswith(NIFTI_SUFFIXES):
        return False
    return not any(marker in lower for marker in DERIVED_MARKERS)


def load_brats_folder(folder):
    """Load the raw (un-normalized) BraTS modality volumes from a case folder.

    Returns (volumes, spacing, paths) where spacing is the (sagittal, coronal,
    axial) voxel size in mm read from the RAS+-reoriented affine, and paths is
    {modality: source file path} for backends (e.g. nnUNet) that need to read
    the original, un-reoriented files themselves.

    Non-NIfTI files and this app's own saved outputs are ignored, so a folder
    that has been used as a save target still loads.
    """
    volumes = {}
    paths = {}
    spacing = None

    # Sorted so a folder with more than one file matching a modality key
    # resolves the same way every time.
    for fname in sorted(os.listdir(folder)):
        if not _is_case_input(fname):
            continue
        lower = fname.lower()
        for key, modality in MODALITY_FILE_KEYS.items():
            if key in lower and modality not in volumes:
                path = os.path.join(folder, fname)
                # Reorient to RAS+ canonical so axis 0/1/2 reliably mean sagittal/coronal/axial regardless of how the file was stored.
                img = nib.as_closest_canonical(nib.load(path))
                vol = img.get_fdata().astype(np.float32)
                volumes[modality] = vol
                paths[modality] = path
                if spacing is None:
                    spacing = tuple(nib.affines.voxel_sizes(img.affine))

    missing = [m for m in MODALITIES if m not in volumes]
    if missing:
        raise ValueError(f"Missing modalities in folder: {', '.join(missing)}")

    return volumes, spacing, paths


def normalize_for_display(volume):
    vmin, vmax = volume.min(), volume.max()
    return (volume - vmin) / (vmax - vmin + 1e-6)


def save_segmentation(mask, reference_path, out_path):
    """Write a segmentation mask to out_path as a .nii.gz.

    The mask is in RAS+ canonical space (see load_brats_folder), so the
    geometry is taken from the canonical form of reference_path — one of
    the case's modality files — rather than its on-disk affine.
    """
    ref = nib.as_closest_canonical(nib.load(reference_path))
    img = nib.Nifti1Image(mask.astype(np.uint8), ref.affine)
    img.header.set_xyzt_units(*ref.header.get_xyzt_units())
    nib.save(img, out_path)


def save_label_map(mask, reference_path, out_path):
    """Write a multi-label segmentation to out_path as a .nii.gz.

    Like save_segmentation, but keeps the label values intact: FreeSurfer aseg
    labels reach 60 and SynthSeg's cortical parcels reach 2035, so uint8 would
    silently wrap them.
    """
    ref = nib.as_closest_canonical(nib.load(reference_path))
    img = nib.Nifti1Image(mask.astype(np.int16), ref.affine)
    img.header.set_xyzt_units(*ref.header.get_xyzt_units())
    nib.save(img, out_path)


def save_volume(volume, reference_path, out_path):
    """Write an intensity volume to out_path as a .nii.gz.

    The float-valued twin of save_segmentation, for handing a loaded modality
    to a tool that reads files rather than arrays. Both write with the *canonical*
    affine of reference_path, so an image and a mask saved this way share byte
    identical geometry — which is what stops PyRadiomics rejecting the pair as
    misaligned when the source file was not stored RAS+ to begin with.
    """
    ref = nib.as_closest_canonical(nib.load(reference_path))
    img = nib.Nifti1Image(volume.astype(np.float32), ref.affine)
    img.header.set_xyzt_units(*ref.header.get_xyzt_units())
    nib.save(img, out_path)
