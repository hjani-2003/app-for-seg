import os

import numpy as np
import nibabel as nib

from core.constants import MODALITY_FILE_KEYS, MODALITIES


def load_brats_folder(folder):
    """Load the raw (un-normalized) BraTS modality volumes from a case folder."""
    volumes = {}

    for fname in os.listdir(folder):
        lower = fname.lower()
        for key, modality in MODALITY_FILE_KEYS.items():
            if key in lower:
                path = os.path.join(folder, fname)
                # Reorient to RAS+ canonical so axis 0/1/2 reliably mean
                # sagittal/coronal/axial regardless of how the file was stored.
                img = nib.as_closest_canonical(nib.load(path))
                vol = img.get_fdata().astype(np.float32)
                volumes[modality] = vol

    missing = [m for m in MODALITIES if m not in volumes]
    if missing:
        raise ValueError(f"Missing modalities in folder: {', '.join(missing)}")

    return volumes


def normalize_for_display(volume):
    vmin, vmax = volume.min(), volume.max()
    return (volume - vmin) / (vmax - vmin + 1e-6)
